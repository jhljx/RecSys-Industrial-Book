r"""
BERT4Rec 双向 Transformer 序列召回（第10章 §10.4.3「Transformer序列召回 - BERT4Rec模型」）

论文：Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder
      Representations from Transformer", CIKM 2019

核心思想：
- 用双向 Transformer Encoder 编码用户行为序列，训练目标为 Cloze（完形填空 / MLM）：
  随机把序列中的若干物品替换成 [MASK]，再用「前后双向上下文」还原被遮蔽的物品。
      [i1,i2,i3,i4,i5] => [i1,[MASK],i3,[MASK],i5]
      L = - Σ_{t in M} log P(i_t | S_\t)
- 推理时在序列末尾追加一个 [MASK] token，用该位置的隐状态预测下一个物品，
  再用它作为 User Embedding 去 ANN 索引中检索 Top-K。

与 SASRec 的关键区别（见同目录 sasrec.py）：
- 注意力掩码：SASRec 用因果 Mask（位置 t 只能看 ≤t）；BERT4Rec 不加因果 Mask，双向看全序列。
- 训练目标：SASRec 自回归预测下一个物品；BERT4Rec 随机 Mask 若干位置后用双向上下文还原。
- 监督信号：SASRec 每个位置 1 个；BERT4Rec 一条序列可采样多组 Mask，数据利用率更高。
- 推理方式：SASRec 直接取最后位置隐状态；BERT4Rec 需在末尾追加 [MASK] 再取该位置隐状态。

本实现：PyTorch autograd，使用 nn.TransformerEncoder（无因果 Mask）+ Tied Embedding 输出层。
完整反向传播由 autograd 自动完成。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── BERT4Rec 模型 ─────────────────────────────────────────────────────────────

class BERT4Rec(nn.Module):
    """
    精简版 BERT4Rec：双向多头 Transformer Encoder + Cloze 训练目标。

    参数：
      item_emb: nn.Embedding [(n_items+1) × dim]  物品 Embedding，下标 n_items 为 [MASK] token
      pos_emb:  nn.Embedding [max_len × dim]       位置 Embedding
      transformer: nn.TransformerEncoder（无因果 Mask，双向注意力）
      out_bias: [n_items]  输出层 bias（输出矩阵与 item_emb 共享，Tied Embedding）
    """

    def __init__(self, n_items: int, dim: int = 16, max_len: int = 10, n_layers: int = 2,
                 n_heads: int = 2, d_ff: int = 32, lr: float = 0.05, mask_prob: float = 0.3,
                 dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim 必须能被 n_heads 整除"
        self.n_items = n_items
        self.mask_id = n_items      # 下标 n_items 即 [MASK] token
        self.dim = dim
        self.max_len = max_len
        self.lr = lr
        self.mask_prob = mask_prob

        scale = math.sqrt(1.0 / dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)   # +1 for [MASK]
        self.pos_emb = nn.Embedding(max_len, dim)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)
        nn.init.uniform_(self.pos_emb.weight, -scale, scale)

        self.out_bias = nn.Parameter(torch.zeros(n_items))

        # 双向 Transformer Encoder（关键点 1：无因果 Mask，batch_first=True）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        完整前向：Embedding → 双向 Transformer Encoder（无因果 Mask）
        tokens: [B, T]  包含 [MASK] id 的 token 序列
        返回: [B, T, dim]  每个位置的隐状态
        """
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # [1, T]
        x = self.item_emb(tokens) + self.pos_emb(positions)             # [B, T, dim]
        # 注意：这里不传 src_mask，双向看全序列（与 SASRec 的因果 mask 的关键区别）
        return self.transformer(x)  # [B, T, dim]

    # ── Cloze 训练（关键点 2：随机 Mask + 双向上下文还原） ───────────────────
    def _make_cloze_sample(self, seq: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
        """以 mask_prob 随机遮蔽序列中的物品；至少遮蔽一个位置。"""
        tokens, targets = list(seq), []
        for t, iid in enumerate(seq):
            if random.random() < self.mask_prob:
                tokens[t] = self.mask_id
                targets.append((t, iid))
        if not targets:                           # 保证至少有一个监督信号
            t = random.randrange(len(seq))
            tokens[t] = self.mask_id
            targets.append((t, seq[t]))
        return tokens, targets

    def _cloze_loss(self, tokens: torch.Tensor,
                    mask_positions: list[list[int]],
                    mask_labels: list[list[int]]) -> torch.Tensor:
        """
        批量 Cloze 交叉熵损失。
        tokens:         [B, T]
        mask_positions: list of per-sample masked position indices
        mask_labels:    list of per-sample true item ids at masked positions
        Tied Embedding: logits = h @ item_emb.T + out_bias
        """
        hs = self._forward(tokens)   # [B, T, dim]
        total_loss = torch.zeros((), device=tokens.device)
        n = 0
        for b, (positions, labels) in enumerate(zip(mask_positions, mask_labels)):
            if not positions:
                continue
            pos_t = torch.tensor(positions, dtype=torch.long, device=tokens.device)
            lbl_t = torch.tensor(labels, dtype=torch.long, device=tokens.device)
            h = hs[b, pos_t, :]                          # [M, dim]
            # Tied Embedding 输出层（共享 item_emb，不含 [MASK] 行）
            logits = h @ self.item_emb.weight[:self.n_items].T + self.out_bias  # [M, n_items]
            total_loss = total_loss + F.cross_entropy(logits, lbl_t)
            n += 1
        return total_loss / max(n, 1)

    def train_model(self, user_seqs: dict[int, list[int]], epochs: int = 6,
                    batch_size: int = 32) -> None:
        """
        Cloze 训练主循环。每条序列产出两类样本：
          (1) 随机 Mask 样本：学习双向上下文还原能力；
          (2) 仅 Mask 末尾位置的样本：论文中用于缓解「训练目标与线上 next-item 预测」
              之间的不一致（与推理时末尾追加 [MASK] 的做法对齐）。
        """
        seqs = [s[-self.max_len:] for s in user_seqs.values() if len(s) >= 2]
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            random.shuffle(seqs)
            total, n_batches = 0.0, 0

            # Build all samples: (tokens, positions, labels) per sequence × 2 variants
            all_samples: list[tuple[list[int], list[int], list[int]]] = []
            for seq in seqs:
                toks, tgts = self._make_cloze_sample(seq)
                all_samples.append((toks, [t for t, _ in tgts], [v for _, v in tgts]))
                # Last-position variant
                last_toks = list(seq[:-1]) + [self.mask_id]
                all_samples.append((last_toks, [len(seq) - 1], [seq[-1]]))

            for start in range(0, len(all_samples), batch_size):
                chunk = all_samples[start:start + batch_size]
                T = max(len(s[0]) for s in chunk)
                # Pad tokens to uniform length (left-pad with 0)
                padded = torch.zeros(len(chunk), T, dtype=torch.long)
                for i, (toks, _, _) in enumerate(chunk):
                    offset = T - len(toks)
                    padded[i, offset:] = torch.tensor(toks, dtype=torch.long)
                    # Adjust positions to account for left-padding
                mask_positions = []
                mask_labels = []
                for i, (toks, positions, labels) in enumerate(chunk):
                    offset = T - len(toks)
                    mask_positions.append([p + offset for p in positions])
                    mask_labels.append(labels)

                optimizer.zero_grad()
                loss = self._cloze_loss(padded, mask_positions, mask_labels)
                loss.backward()
                optimizer.step()

                total += loss.item()
                n_batches += 1

            print(f"  epoch {epoch:2d}  cloze_loss={total / max(n_batches, 1):.4f}")

    # ── 推理（关键点 3：末尾追加 [MASK]） ───────────────────────────────────
    def user_embedding(self, seq: list[int]) -> torch.Tensor:
        """在序列末尾追加 [MASK]，取该位置隐状态作为 User Embedding，shape [dim]。"""
        with torch.no_grad():
            tokens = list(seq[-(self.max_len - 1):]) + [self.mask_id]
            tok_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, T]
            hs = self._forward(tok_t)                                     # [1, T, dim]
            return hs[0, -1, :]                                           # [dim]

    def recommend(self, seq: list[int], top_k: int = 5,
                  exclude_seen: bool = True) -> list[tuple[int, float]]:
        """给定历史序列，预测下一个物品 Top-K。"""
        with torch.no_grad():
            u = self.user_embedding(seq)
            # Tied Embedding 打分
            logits = self.item_emb.weight[:self.n_items] @ u + self.out_bias  # [n_items]
            if exclude_seen:
                for iid in seq:
                    logits[iid] = -1e9
            top_vals, top_idx = logits.topk(top_k)
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]

    def fill_mask(self, tokens: list[int], top_k: int = 3) -> dict[int, list[tuple[int, float]]]:
        """完形填空：对序列中所有 [MASK] 位置，用双向上下文给出 Top-K 候选。"""
        with torch.no_grad():
            tok_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, T]
            hs = self._forward(tok_t)                                     # [1, T, dim]
            result: dict[int, list[tuple[int, float]]] = {}
            for t, tok in enumerate(tokens):
                if tok != self.mask_id:
                    continue
                logits = hs[0, t, :] @ self.item_emb.weight[:self.n_items].T + self.out_bias
                probs = logits.softmax(dim=-1)
                top_vals, top_idx = probs.topk(top_k)
                result[t] = list(zip(top_idx.tolist(), top_vals.tolist()))
        return result


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, seq_len = 8, 20, 8
    # 合成序列：兴趣簇内按 +1 递进点击，前后文都能提供预测线索（适合双向建模）
    user_seqs: dict[int, list[int]] = {}
    for u in range(n_users):
        start = (u % 4) * 5 + random.randint(0, 2)
        user_seqs[u] = [(start + k) % n_items for k in range(seq_len)]

    print("合成用户行为序列：")
    for u in range(n_users):
        print(f"  user {u}: {user_seqs[u]}")

    model = BERT4Rec(n_items, dim=16, max_len=10, n_layers=2, n_heads=2, d_ff=32, lr=0.05)
    print("\nBERT4Rec Cloze 训练（双向注意力，无因果 Mask）：")
    model.train_model(user_seqs, epochs=6)

    seq, probe, truth = user_seqs[0], user_seqs[0][:-1], user_seqs[0][-1]
    print(f"\n历史序列 {probe}（真实下一个物品 = {truth}）\n末尾追加 [MASK] 后的 Top-5 召回：")
    for iid, s in model.recommend(probe, top_k=5, exclude_seen=False):
        flag = "  <-- 命中" if iid == truth else ""
        print(f"  item {iid}: score={s:.4f}{flag}")

    # 演示双向能力：遮蔽序列中间位置，用前后文同时推断
    mid, tokens = len(seq) // 2, list(seq)
    tokens[mid] = model.mask_id
    print(f"\n完形填空演示：把位置 {mid}（真实物品 {seq[mid]}）替换为 [MASK]")
    print(f"  输入 tokens（{model.mask_id} 表示 [MASK]）：{tokens}")
    for pos, cands in model.fill_mask(tokens, top_k=3).items():
        pretty = ", ".join(f"item {i}(p={p:.3f})" for i, p in cands)
        print(f"  位置 {pos} 的双向预测：{pretty}")
