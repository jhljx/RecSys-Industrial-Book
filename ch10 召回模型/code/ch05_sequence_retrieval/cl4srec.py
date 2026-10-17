"""
CL4SRec 对比学习序列召回（第10章 §10.4.5「对比学习序列召回 - CL4SRec召回」）

论文：Xie et al., "Contrastive Learning for Sequential Recommendation", ICDE 2022
核心思想：
- 以 SASRec（因果 Self-Attention）作为序列编码器；
- 对同一条用户行为序列做两次随机数据增强，得到两个视图，互为正样本对；
  同一个 Mini-Batch 内其它用户的增强序列作为负样本；
- 三种序列数据增强算子：
    * Item Crop   ：随机截取一段连续子序列（学习局部兴趣，适配不同序列长度）
    * Item Mask   ：随机把部分物品替换为 [MASK] 标记（抗行为缺失 / 噪声）
    * Item Reorder：随机打乱一段连续子序列的顺序（降低对严格顺序的依赖）
- 对比损失采用 InfoNCE：
      L_cl = -log( exp(sim(h_i, h_j)/τ) / Σ_k exp(sim(h_i, h_k)/τ) )
      实现为 F.cross_entropy(sim_matrix / tau, torch.arange(B))
- 多任务联合训练：L_total = L_rec + λ·L_cl （L_rec 为下一物品预测损失）

本实现：PyTorch autograd，增强操作用 Python 列表操作，编码器用
nn.TransformerEncoderLayer + causal mask，完整反向传播由 autograd 自动完成。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 三种序列数据增强算子 ───────────────────────────────────────────────────────

class SeqAugment:
    """CL4SRec 的三种数据增强算子：Crop / Mask / Reorder。"""

    def __init__(self, mask_token: int, eta: float = 0.6, gamma: float = 0.3, beta: float = 0.5):
        self.mask_token = mask_token   # [MASK] 占位 id（= n_items）
        self.eta = eta                 # crop 保留比例
        self.gamma = gamma             # mask 比例
        self.beta = beta               # reorder 子序列比例

    def item_crop(self, seq: list[int]) -> list[int]:
        """随机截取长度为 ⌊η·T⌋ 的连续子序列。"""
        T = len(seq)
        sub_len = max(1, int(self.eta * T))
        start = random.randint(0, T - sub_len)
        return seq[start:start + sub_len]

    def item_mask(self, seq: list[int]) -> list[int]:
        """随机将 ⌊γ·T⌋ 个位置替换为 [MASK]。"""
        out = list(seq)
        n_mask = max(1, int(self.gamma * len(seq)))
        for idx in random.sample(range(len(seq)), n_mask):
            out[idx] = self.mask_token
        return out

    def item_reorder(self, seq: list[int]) -> list[int]:
        """随机选取长度为 ⌊β·T⌋ 的连续片段并打乱其内部顺序。"""
        out = list(seq)
        T = len(seq)
        sub_len = max(2, int(self.beta * T)) if T >= 2 else T
        sub_len = min(sub_len, T)
        start = random.randint(0, T - sub_len)
        chunk = out[start:start + sub_len]
        random.shuffle(chunk)
        out[start:start + sub_len] = chunk
        return out

    def random_augment(self, seq: list[int]) -> tuple[str, list[int]]:
        """等概率随机挑选一种增强算子。"""
        name = random.choice(["crop", "mask", "reorder"])
        return name, {"crop": self.item_crop,
                      "mask": self.item_mask,
                      "reorder": self.item_reorder}[name](seq)


# ── CL4SRec 模型 ──────────────────────────────────────────────────────────────

class CL4SRec(nn.Module):
    """
    CL4SRec = SASRec 编码器 + 自监督对比学习辅助任务。

    组件：
      item_emb : nn.Embedding [(n_items + 1) × dim]，最后一行为 [MASK] 的 Embedding
      pos_emb  : nn.Embedding [max_len × dim]
      transformer_layer: nn.TransformerEncoderLayer（因果 mask，单头）
    """

    def __init__(self, n_items: int, dim: int = 16, max_len: int = 10,
                 lr: float = 0.05, reg: float = 1e-4,
                 tau: float = 0.5, lam: float = 0.3, n_heads: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.max_len = max_len
        self.lr = lr
        self.reg = reg
        self.tau = tau                 # 温度系数 τ
        self.lam = lam                 # 对比损失权重 λ
        self.mask_token = n_items

        scale = math.sqrt(1.0 / dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)   # +1 for [MASK]
        self.pos_emb = nn.Embedding(max_len, dim)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)
        nn.init.uniform_(self.pos_emb.weight, -scale, scale)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.aug = SeqAugment(self.mask_token)

    def _encode_batch(self, seqs: list[list[int]], pool: str = "last") -> torch.Tensor:
        """
        编码一批序列，自动左填充到同长度。
        pool="last"：取最后位置（下一物品预测任务）；
        pool="mean"：均值池化（对比学习视图，对增强后长度不一的序列更稳健）。
        返回 [B, dim]。
        """
        T = max(len(s) for s in seqs)
        T = min(T, self.max_len)
        B = len(seqs)

        padded = torch.zeros(B, T, dtype=torch.long)
        for i, seq in enumerate(seqs):
            s = seq[-T:]
            padded[i, T - len(s):] = torch.tensor(s, dtype=torch.long)

        positions = torch.arange(T).unsqueeze(0)                   # [1, T]
        x = self.item_emb(padded) + self.pos_emb(positions)        # [B, T, dim]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        out = self.transformer_layer(x, src_mask=causal_mask, is_causal=True)  # [B, T, dim]

        if pool == "mean":
            # Mean over non-padding positions only
            lengths = torch.tensor([min(len(s), T) for s in seqs], dtype=torch.float)
            mask = torch.zeros(B, T)
            for i, seq in enumerate(seqs):
                l = min(len(seq), T)
                mask[i, T - l:] = 1.0
            return (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        else:
            return out[:, -1, :]

    # ── 主任务：下一物品预测 ─────────────────────────────────────────────
    def _rec_loss(self, contexts: list[list[int]], pos_ids: list[int],
                  neg_ids: list[int]) -> torch.Tensor:
        """BCE 形式的下一物品预测损失。"""
        h = self._encode_batch(contexts, pool="last")               # [B, dim]
        v_pos = self.item_emb(torch.tensor(pos_ids, dtype=torch.long))
        v_neg = self.item_emb(torch.tensor(neg_ids, dtype=torch.long))
        s_pos = (h * v_pos).sum(-1)
        s_neg = (h * v_neg).sum(-1)
        return (F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))
                + F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg)))

    # ── 辅助任务：InfoNCE 对比学习 ──────────────────────────────────────
    def _cl_loss(self, view_seqs_a: list[list[int]],
                 view_seqs_b: list[list[int]]) -> torch.Tensor:
        """
        Batch 内 InfoNCE：view_seqs_a[i] 与 view_seqs_b[i] 来自同一序列，互为正样本。
        L_cl = F.cross_entropy(sim_matrix / τ, torch.arange(B))
        """
        ha = F.normalize(self._encode_batch(view_seqs_a, pool="mean"), dim=-1)  # [B, dim]
        hb = F.normalize(self._encode_batch(view_seqs_b, pool="mean"), dim=-1)  # [B, dim]

        # sim_matrix[i, j] = cosine(ha[i], hb[j]) / τ
        sim = (ha @ hb.T) / self.tau                                # [B, B]
        labels = torch.arange(sim.size(0), dtype=torch.long)
        # Both directions: ha predicts hb and hb predicts ha
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0

    def train_model(self, user_seqs: dict[int, list[int]], epochs: int = 6,
                    batch_size: int = 4) -> None:
        """多任务联合训练：L_total = L_rec + λ·L_cl。"""
        uids = list(user_seqs.keys())
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        for epoch in range(epochs):
            random.shuffle(uids)
            rec_sum, cl_sum, n_batch, n_rec = 0.0, 0.0, 0, 0

            for b in range(0, len(uids), batch_size):
                batch_uids = uids[b:b + batch_size]
                contexts_rec, pos_ids, neg_ids = [], [], []
                view_seqs_a, view_seqs_b = [], []

                for uid in batch_uids:
                    seq = user_seqs[uid]
                    # 主任务：序列前缀预测下一物品（每个位置都作为监督信号）
                    for t in range(1, len(seq)):
                        contexts_rec.append(seq[:t])
                        pos_ids.append(seq[t])
                        neg = random.randrange(self.n_items)
                        while neg == seq[t]:
                            neg = random.randrange(self.n_items)
                        neg_ids.append(neg)
                        n_rec += 1
                    # 辅助任务：同一序列两次独立增强 → 正样本对
                    _, va = self.aug.random_augment(seq[:-1])
                    _, vb = self.aug.random_augment(seq[:-1])
                    view_seqs_a.append(va)
                    view_seqs_b.append(vb)

                optimizer.zero_grad()
                rec_loss = self._rec_loss(contexts_rec, pos_ids, neg_ids)
                cl_loss = self._cl_loss(view_seqs_a, view_seqs_b)
                loss = rec_loss + self.lam * cl_loss
                loss.backward()
                optimizer.step()

                rec_sum += rec_loss.item()
                cl_sum += cl_loss.item()
                n_batch += 1

            print(f"  epoch {epoch:2d}  L_rec={rec_sum / max(n_batch, 1):.4f}  "
                  f"L_cl={cl_sum / max(n_batch, 1):.4f}  "
                  f"L_total={rec_sum / max(n_batch, 1) + self.lam * cl_sum / max(n_batch, 1):.4f}")

    def recommend(self, seq: list[int], top_k: int = 5,
                  exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """用序列表示 h 对全量物品打分（线上等价于 ANN 检索）。"""
        exclude = exclude if exclude is not None else set(seq)
        with torch.no_grad():
            h = self._encode_batch([seq], pool="last")[0]           # [dim]
            scores = self.item_emb.weight[:self.n_items] @ h        # [n_items]
            for iid in exclude:
                scores[iid] = -1e9
            top_vals, top_idx = scores.topk(top_k)
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_items, n_users = 20, 8
    user_seqs = {u: random.sample(range(n_items), k=8) for u in range(n_users)}

    demo_seq = user_seqs[0]
    print("三种数据增强算子作用在同一条序列上的示例：")
    print(f"  原始序列      : {demo_seq}")
    aug = SeqAugment(mask_token=n_items)
    print(f"  Item Crop     : {aug.item_crop(demo_seq)}")
    print(f"  Item Mask     : {aug.item_mask(demo_seq)}   （{n_items} 表示 [MASK]）")
    print(f"  Item Reorder  : {aug.item_reorder(demo_seq)}")

    model = CL4SRec(n_items, dim=16, max_len=10, lr=0.05, tau=0.5, lam=0.3)
    print(f"\nCL4SRec 训练（{n_users} 用户，λ={model.lam}，τ={model.tau}）：")
    model.train_model(user_seqs, epochs=6, batch_size=4)

    recs = model.recommend(demo_seq, top_k=5)
    print(f"\n用户 0 序列 {demo_seq} 的 Top-5 召回：")
    for iid, sc in recs:
        print(f"  item {iid:2d}  score={sc:.4f}")

    # 对比学习效果观察：同一序列两视图相似度 vs 不同用户序列相似度
    pos_sims, neg_sims = [], []
    with torch.no_grad():
        reps = {u: F.normalize(model._encode_batch([s], pool="mean")[0], dim=-1)
                for u, s in user_seqs.items()}
        for u, s in user_seqs.items():
            _, va = aug.random_augment(s)
            _, vb = aug.random_augment(s)
            ha = F.normalize(model._encode_batch([va], pool="mean")[0], dim=-1)
            hb = F.normalize(model._encode_batch([vb], pool="mean")[0], dim=-1)
            pos_sims.append((ha @ hb).item())
            for v in user_seqs:
                if v != u:
                    neg_sims.append((ha @ reps[v]).item())

    print(f"\n同序列两视图平均余弦相似度={sum(pos_sims) / len(pos_sims):.4f}  "
          f"不同用户序列平均余弦相似度={sum(neg_sims) / len(neg_sims):.4f}")
