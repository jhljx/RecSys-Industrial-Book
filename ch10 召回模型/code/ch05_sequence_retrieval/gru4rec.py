r"""
GRU4Rec Session-based 序列召回（第10章 §10.4.2「RNN序列召回 - GRU4Rec模型」）

论文：Hidasi et al., "Session-based Recommendations with Recurrent Neural
      Networks", ICLR 2016

核心思想：
- 用 GRU 按时间顺序逐步编码用户当前 Session 的点击序列，
  用最后一个隐藏状态作为用户兴趣表示 u = h_T，再与物品 Embedding 打分
  score(u, v) = u^T v，线上用 u 在 ANN 索引里检索 Top-K。
- 与 Item2Vec 的本质区别：Item2Vec 丢弃时序只学共现，GRU4Rec 显式建模时序依赖。

本实现覆盖 GRU4Rec 论文的三个要点：
1. GRU 单元公式（由 nn.GRU 实现，batch_first=True）：
     z_t = σ(W_z e_t + U_z h_{t-1} + b_z)                 更新门
     r_t = σ(W_r e_t + U_r h_{t-1} + b_r)                 重置门
     h~_t = tanh(W_h e_t + U_h (r_t ⊙ h_{t-1}) + b_h)     候选隐状态
     h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t               隐状态更新
2. Session-parallel mini-batch：每个 batch 槽位（slot）跑一个独立 Session，
   逐时间步并行推进；某个 Session 结束后该槽位换入新 Session 并重置隐状态。
   这样既避免了 padding，也让同 batch 内其它 Session 的目标物品天然成为负样本。
3. Ranking 损失（而非纯 Softmax），N_S 为负样本数，r_i 为正样本得分：
     BPR : L = -(1/N_S) Σ_j log σ(r_i - r_j)
     TOP1: L =  (1/N_S) Σ_j [ σ(r_j - r_i) + σ(r_j^2) ]   第二项是正则，压低负样本得分

反向传播：使用 PyTorch autograd，完整 BPTT 通过 nn.GRU 自动完成。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── GRU4Rec 模型 ──────────────────────────────────────────────────────────────

class GRU4Rec(nn.Module):
    """
    单层 GRU4Rec。

    参数：
      item_emb: nn.Embedding [n_items × dim]   输入侧物品 Embedding e_t
      item_out: nn.Embedding [n_items × dim]   输出侧物品 Embedding v（打分/ANN 索引用）
      gru: nn.GRU (input_size=dim, hidden_size=dim, batch_first=True)
    """

    def __init__(self, n_items: int, dim: int = 16, lr: float = 0.3,
                 loss: str = "top1", reg: float = 1e-5):
        super().__init__()
        assert loss in ("top1", "bpr"), "loss 只支持 'top1' 或 'bpr'"
        self.n_items = n_items
        self.dim = dim
        self.lr = lr
        self.loss_type = loss
        self.reg = reg

        scale = math.sqrt(1.0 / dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.item_out = nn.Embedding(n_items, dim)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_out.weight, -scale, scale)

        self.gru = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)

    def _rank_loss(self, h: torch.Tensor, pos: torch.Tensor,
                   negs: torch.Tensor) -> torch.Tensor:
        """
        计算 TOP1 / BPR 损失。
        h:    [B, dim]
        pos:  [B]        正样本 item id
        negs: [B, N_neg]  负样本 item id
        """
        v_pos = self.item_out(pos)          # [B, dim]
        v_neg = self.item_out(negs)         # [B, N_neg, dim]

        r_i = (h * v_pos).sum(-1, keepdim=True)              # [B, 1]
        r_j = torch.bmm(v_neg, h.unsqueeze(-1)).squeeze(-1)  # [B, N_neg]

        if self.loss_type == "bpr":
            loss = -F.logsigmoid(r_i - r_j).mean()
        else:  # top1
            loss = (torch.sigmoid(r_j - r_i) + torch.sigmoid(r_j ** 2)).mean()
        return loss

    # ── Session-parallel mini-batch 训练 ──────────────────────────────────────
    def train_model(self, sessions: list[list[int]], epochs: int = 6,
                    batch_size: int = 3, n_extra_neg: int = 2) -> None:
        """
        sessions: 每个元素是一个 Session 的点击序列。
        Session-parallel：batch 中每个槽位独立推进一个 Session，同 batch 内其它
        槽位的目标物品直接作为负样本（GRU4Rec 论文的 popularity-based sampling 近似）。
        """
        usable = [s for s in sessions if len(s) >= 2]
        optimizer = optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.reg)

        for epoch in range(epochs):
            order = list(range(len(usable)))
            random.shuffle(order)

            next_idx = min(batch_size, len(usable))
            slot_sess: list[list[int]] = [usable[i] for i in order[:next_idx]]
            slot_pos: list[int] = [0] * next_idx
            # Per-slot hidden states stored as a list of [dim] tensors (detached).
            # Stacked into [1, B, dim] each step, then unpacked back.
            slot_h: list[torch.Tensor] = [torch.zeros(self.dim) for _ in range(next_idx)]

            total_loss, n_steps = 0.0, 0
            while slot_sess:
                B = len(slot_sess)
                # 1) 本时间步各槽位的当前输入 item
                cur_items = torch.tensor(
                    [slot_sess[s][slot_pos[s]] for s in range(B)], dtype=torch.long)

                # 2) Stack per-slot hidden states → [1, B, dim], run GRU one step
                h_in = torch.stack(slot_h, dim=0).unsqueeze(0)  # [1, B, dim]
                e_t = self.item_emb(cur_items).unsqueeze(1)      # [B, 1, dim]
                _, h_out = self.gru(e_t, h_in)                   # h_out: [1, B, dim]

                # 3) 正样本 = 每槽位的下一个 item；负样本 = 同 batch 其它槽位目标 + 随机
                pos_items = [slot_sess[s][slot_pos[s] + 1] for s in range(B)]
                neg_lists: list[list[int]] = []
                for idx in range(B):
                    negs = [p for j, p in enumerate(pos_items) if j != idx and p != pos_items[idx]]
                    while len(negs) < B - 1 + n_extra_neg:
                        cand = random.randrange(self.n_items)
                        if cand != pos_items[idx]:
                            negs.append(cand)
                    neg_lists.append(negs[:B - 1 + n_extra_neg])

                pos_t = torch.tensor(pos_items, dtype=torch.long)
                max_neg = max(len(n) for n in neg_lists)
                neg_t = torch.tensor(
                    [n + [n[0]] * (max_neg - len(n)) for n in neg_lists],
                    dtype=torch.long)

                h_flat = h_out.squeeze(0)  # [B, dim]
                optimizer.zero_grad()
                loss = self._rank_loss(h_flat, pos_t, neg_t)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_steps += 1

                # 4) Unpack updated hidden states back into the per-slot list (detached)
                h_detached = h_out.squeeze(0).detach()  # [B, dim]
                slot_h = [h_detached[s] for s in range(B)]

                # 5) 推进指针；Session 用完则换入新 Session 并重置隐状态
                for s in reversed(range(B)):
                    slot_pos[s] += 1
                    if slot_pos[s] + 1 >= len(slot_sess[s]):
                        if next_idx < len(order):
                            slot_sess[s] = usable[order[next_idx]]
                            slot_pos[s] = 0
                            slot_h[s] = torch.zeros(self.dim)   # 新 Session → 隐状态清零
                            next_idx += 1
                        else:
                            slot_sess.pop(s)
                            slot_pos.pop(s)
                            slot_h.pop(s)

            print(f"  epoch {epoch:2d}  {self.loss_type}_loss={total_loss / max(n_steps, 1):.4f}  steps={n_steps}")

    # ── 召回接口 ──────────────────────────────────────────────────────────────
    def encode(self, seq: list[int]) -> torch.Tensor:
        """对整条序列做 GRU 编码，返回用户兴趣表示 u = h_T，shape [dim]。"""
        with torch.no_grad():
            ids = torch.tensor(seq, dtype=torch.long)
            e = self.item_emb(ids).unsqueeze(0)      # [1, T, dim]
            _, h = self.gru(e)                        # h: [1, 1, dim]
            return h.squeeze()                        # [dim]

    def recommend(self, seq: list[int], top_k: int = 5,
                  exclude_seen: bool = True) -> list[tuple[int, float]]:
        """给定 Session 序列，预测下一个物品 Top-K（线上即 u 在 ANN 中检索）。"""
        with torch.no_grad():
            u = self.encode(seq)
            scores = self.item_out.weight @ u         # [n_items]
            if exclude_seen:
                for iid in seq:
                    scores[iid] = -1e9
            top_vals, top_idx = scores.topk(top_k)
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items, seq_len = 8, 20, 8
    # 合成「有时序规律」的 Session：在某个兴趣簇内按 +1 递进地点击（i -> i+1 -> i+2 ...）
    sessions: list[list[int]] = []
    for u in range(n_users):
        start = (u % 4) * 5 + random.randint(0, 2)        # 兴趣簇起点
        sessions.append([(start + k) % n_items for k in range(seq_len)])

    print("合成 Session 序列（具有明显时序递进规律）：")
    for u, s in enumerate(sessions):
        print(f"  session {u}: {s}")

    for loss_name in ("top1", "bpr"):
        print(f"\nGRU4Rec 训练（loss={loss_name}，session-parallel batch=3）：")
        model = GRU4Rec(n_items, dim=16, lr=0.3, loss=loss_name)
        model.train_model(sessions, epochs=6, batch_size=3)

        probe, truth = sessions[0][:5], sessions[0][5]
        print(f"  输入前缀 {probe}（真实下一个物品 = {truth}）Top-5 召回：")
        for iid, s in model.recommend(probe, top_k=5, exclude_seen=False):
            flag = "  <-- 命中" if iid == truth else ""
            print(f"    item {iid}: score={s:.4f}{flag}")
