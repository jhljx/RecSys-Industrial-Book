"""
SASRec Transformer 序列召回（第10章 §10.4.3）

论文：Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018

核心思想：
- 用单向（Causal）Self-Attention 对用户行为序列建模
- 每个位置的输出作为「下一个物品」的预测向量
- 训练用 BCE loss + 负采样；线上取最后一个位置的输出做 ANN 检索

本实现：PyTorch autograd，使用 nn.TransformerEncoderLayer 配合 causal mask，
完整 BPTT 由 autograd 自动完成。
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── SASRec 模型 ───────────────────────────────────────────────────────────────

class SASRec(nn.Module):
    """
    单层、单头 SASRec。

    组件：
      item_emb: nn.Embedding [n_items × dim]
      pos_emb:  nn.Embedding [max_len × dim]
      transformer_layer: nn.TransformerEncoderLayer (causal mask)
      输出 item Embedding 与 item_emb 共享（Tied Embedding 风格）
    """

    def __init__(
        self,
        n_items: int,
        dim: int = 32,
        max_len: int = 50,
        lr: float = 0.01,
        reg: float = 1e-4,
        n_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.max_len = max_len
        self.lr = lr
        self.reg = reg

        scale = math.sqrt(1.0 / dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)
        nn.init.uniform_(self.pos_emb.weight, -scale, scale)

        # 单层因果 Transformer Encoder（batch_first=True）
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

    def _encode(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        seq_ids: [B, T]  item id 序列（已截断到 max_len）
        返回: [B, T, dim]  每个位置的上下文表示
        """
        B, T = seq_ids.shape
        positions = torch.arange(T, device=seq_ids.device).unsqueeze(0)  # [1, T]
        x = self.item_emb(seq_ids) + self.pos_emb(positions)             # [B, T, dim]

        # 因果 mask：位置 t 只能关注 ≤ t 的位置（上三角设为 -inf）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=seq_ids.device)
        out = self.transformer_layer(x, src_mask=causal_mask, is_causal=True)  # [B, T, dim]
        return out

    def forward(self, seq_ids: torch.Tensor, pos_ids: torch.Tensor,
                neg_ids: torch.Tensor) -> torch.Tensor:
        """
        seq_ids: [B, T]   输入序列
        pos_ids: [B]      正样本（下一个物品）
        neg_ids: [B]      负样本
        返回 BCE loss
        """
        out = self._encode(seq_ids)         # [B, T, dim]
        user_repr = out[:, -1, :]           # [B, dim]  最后一个位置的表示

        v_pos = self.item_emb(pos_ids)      # [B, dim]
        v_neg = self.item_emb(neg_ids)      # [B, dim]

        s_pos = (user_repr * v_pos).sum(-1) # [B]
        s_neg = (user_repr * v_neg).sum(-1) # [B]

        loss = F.binary_cross_entropy_with_logits(
            s_pos, torch.ones_like(s_pos)
        ) + F.binary_cross_entropy_with_logits(
            s_neg, torch.zeros_like(s_neg)
        )
        return loss

    def train_model(
        self,
        user_seqs: dict[int, list[int]],
        epochs: int = 10,
        neg_k: int = 1,
        batch_size: int = 64,
    ) -> None:
        """Leave-one-out 训练：最后一个物品为正样本，前面为序列输入。"""
        samples: list[tuple[list[int], int]] = []
        for uid, seq in user_seqs.items():
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                samples.append((seq[:t], seq[t]))

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(samples), batch_size):
                chunk = samples[start:start + batch_size]
                # Truncate / pad context sequences to max_len
                contexts = []
                for ctx, _ in chunk:
                    ctx = ctx[-self.max_len:]
                    contexts.append(ctx)
                max_t = max(len(c) for c in contexts)
                # Left-pad with item 0 (padding token)
                padded = torch.zeros(len(chunk), max_t, dtype=torch.long)
                for i, ctx in enumerate(contexts):
                    padded[i, max_t - len(ctx):] = torch.tensor(ctx, dtype=torch.long)

                pos_ids = torch.tensor([p for _, p in chunk], dtype=torch.long)
                neg_ids = torch.tensor(
                    [self._sample_neg(p, self.n_items) for _, p in chunk],
                    dtype=torch.long,
                )

                optimizer.zero_grad()
                loss = self(padded, pos_ids, neg_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if epoch % 2 == 0:
                print(f"  epoch {epoch:3d}  avg_loss={total_loss / max(n_batches, 1):.4f}")

    @staticmethod
    def _sample_neg(pos: int, n_items: int) -> int:
        neg = random.randint(0, n_items - 1)
        while neg == pos:
            neg = random.randint(0, n_items - 1)
        return neg

    def recommend(self, seq: list[int], top_k: int = 10,
                  exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """给定序列，推荐下一个物品。"""
        exclude = exclude or set(seq)
        with torch.no_grad():
            seq_t = torch.tensor(seq[-self.max_len:], dtype=torch.long).unsqueeze(0)  # [1, T]
            out = self._encode(seq_t)               # [1, T, dim]
            user_repr = out[0, -1, :]               # [dim]
            scores = self.item_emb.weight @ user_repr  # [n_items]
            for iid in exclude:
                scores[iid] = -1e9
            top_vals, top_idx = scores.topk(top_k)
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_items = 15
    user_seqs = {
        u: random.sample(range(n_items), k=random.randint(4, 8))
        for u in range(8)
    }

    model = SASRec(n_items, dim=16, max_len=10, lr=0.05)
    print("SASRec 训练：")
    model.train_model(user_seqs, epochs=6)

    seq = user_seqs[0]
    recs = model.recommend(seq, top_k=5)
    print(f"\n用户序列 {seq} 的 Top-5 推荐：")
    for iid, s in recs:
        print(f"  item {iid}: {s:.4f}")
