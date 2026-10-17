"""
Item2Vec 物品 Embedding 学习（第10章 §10.4.1「List2Vec召回 - Item2Vec模型」）

论文：Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative
      Filtering", MLSP 2016

核心思想：
- 把一个用户的一次行为列表当作一个「句子」，把物品当作「词」，
  直接用 Word2Vec 的 Skip-Gram with Negative Sampling（SGNS）学习物品向量。
- 共同出现在同一用户行为集合中的物品，向量距离更近（协同过滤信号）。

与 word2vec 的关键区别（本实现的重点）：
- word2vec：中心词只与固定大小滑动窗口 [t-c, t+c] 内的词构成正样本，
  因此「空间/时间距离」会影响共现关系。
- Item2Vec：丢弃了窗口概念，把一次行为集合 S={i_1,...,i_K} 当作「动态窗口」，
  集合内任意两个物品（i_a != i_b）都构成正样本，与先后顺序无关。
  因此 A->B->C 与 C->B->A 产生完全相同的训练样本，Item2Vec 学到的是
  共现关系（Co-occurrence）而非时序依赖。

优化目标（对应书中公式）：
    L = -(1/K) * sum_{i=1..K} sum_{j != i} log p(i_j | i_i)
其中 p(i_j|i_i) 用 SGNS 近似：
    log p(j|i) ≈ log σ(u_i · v_j) + sum_{n in Neg} log σ(-u_i · v_n)

本实现：PyTorch autograd，SGNS 损失用 F.binary_cross_entropy_with_logits 计算。
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── Item2Vec 模型 ─────────────────────────────────────────────────────────────

class Item2Vec(nn.Module):
    """
    Item2Vec（Skip-Gram + 负采样）。

    参数矩阵：
      item_emb: [n_items × dim]  中心物品向量 u_i（最终用于召回的 Item Embedding）
      ctx_emb:  [n_items × dim]  上下文物品向量 v_j（训练辅助参数）
    """

    def __init__(
        self,
        n_items: int,
        dim: int = 16,
        lr: float = 0.05,
        n_neg: int = 5,
        reg: float = 1e-5,
    ):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.lr = lr
        self.n_neg = n_neg
        self.reg = reg

        self.item_emb = nn.Embedding(n_items, dim)
        self.ctx_emb = nn.Embedding(n_items, dim)

        nn.init.uniform_(self.item_emb.weight, -math.sqrt(1.0 / dim), math.sqrt(1.0 / dim))
        nn.init.zeros_(self.ctx_emb.weight)

        self._neg_table: list[int] = []

    # ── 样本构造：整个 session 两两共现（Item2Vec 的核心差异点） ──────────────
    def _build_pairs(self, user_seqs: dict[int, list[int]]) -> list[tuple[int, int]]:
        """
        把每个 session 看作无序集合，集合内任意有序对 (i_a, i_b), a != b 都是正样本。
        注意：这里没有任何窗口大小参数，这正是 Item2Vec 与 word2vec 的区别。
        """
        pairs: list[tuple[int, int]] = []
        for seq in user_seqs.values():
            items = list(dict.fromkeys(seq))  # 集合语义：同一 session 内去重
            for a in range(len(items)):
                for b in range(len(items)):
                    if a != b:
                        pairs.append((items[a], items[b]))
        return pairs

    def _build_neg_table(self, user_seqs: dict[int, list[int]], table_size: int = 2000) -> None:
        """按 freq^0.75 构造负采样表（与 word2vec 的 unigram 分布一致）。"""
        freq: dict[int, int] = defaultdict(int)
        for seq in user_seqs.values():
            for iid in seq:
                freq[iid] += 1
        weights = {iid: (c ** 0.75) for iid, c in freq.items()}
        total = sum(weights.values()) or 1.0
        table: list[int] = []
        for iid, w in weights.items():
            cnt = max(1, int(round(table_size * w / total)))
            table.extend([iid] * cnt)
        self._neg_table = table or list(range(self.n_items))

    def _sample_negs(self, exclude: int, k: int) -> list[int]:
        negs = []
        for _ in range(k * 3):
            cand = random.choice(self._neg_table)
            if cand != exclude:
                negs.append(cand)
            if len(negs) == k:
                break
        while len(negs) < k:
            negs.append((exclude + 1) % self.n_items)
        return negs

    def forward(self, center_ids: torch.Tensor, pos_ids: torch.Tensor,
                neg_ids: torch.Tensor) -> torch.Tensor:
        """
        center_ids: [B]
        pos_ids:    [B]
        neg_ids:    [B, n_neg]

        loss = BCE(u·v_pos, 1) + sum_neg BCE(u·v_neg, 0)
             = F.binary_cross_entropy_with_logits on stacked logits
        """
        u = self.item_emb(center_ids)           # [B, dim]
        v_pos = self.ctx_emb(pos_ids)           # [B, dim]
        v_neg = self.ctx_emb(neg_ids)           # [B, n_neg, dim]

        pos_logits = (u * v_pos).sum(-1)        # [B]
        neg_logits = torch.bmm(v_neg, u.unsqueeze(-1)).squeeze(-1)  # [B, n_neg]

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits))
        return pos_loss + neg_loss

    def train_model(self, user_seqs: dict[int, list[int]], epochs: int = 5,
                    batch_size: int = 256) -> None:
        """用所有用户行为集合训练 Item Embedding。"""
        self._build_neg_table(user_seqs)
        pairs = self._build_pairs(user_seqs)
        print(f"  共现物品对数量：{len(pairs)}（整 session 两两组合，无滑动窗口）")

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        for epoch in range(epochs):
            random.shuffle(pairs)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                center_ids = torch.tensor([p[0] for p in batch], dtype=torch.long)
                pos_ids = torch.tensor([p[1] for p in batch], dtype=torch.long)
                neg_ids = torch.tensor(
                    [self._sample_negs(p[1], self.n_neg) for p in batch],
                    dtype=torch.long)

                optimizer.zero_grad()
                loss = self(center_ids, pos_ids, neg_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            print(f"  epoch {epoch:2d}  avg_loss={total_loss / max(n_batches, 1):.4f}")

    # ── 召回接口 ──────────────────────────────────────────────────────────────
    def item_vector(self, iid: int) -> torch.Tensor:
        """返回物品的最终 Embedding（线上写入 ANN 索引的向量）。"""
        with torch.no_grad():
            return self.item_emb.weight[iid]

    def most_similar(self, iid: int, top_k: int = 5) -> list[tuple[int, float]]:
        """I2I 检索：给定 Trigger 物品，返回余弦最相似的 Top-K 物品。"""
        with torch.no_grad():
            W = F.normalize(self.item_emb.weight, dim=-1)  # [n_items, dim]
            base = W[iid]
            sims = (W @ base)  # [n_items]
            sims[iid] = -1e9
            top_vals, top_idx = sims.topk(top_k)
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]

    def recommend(self, seq: list[int], top_k: int = 5) -> list[tuple[int, float]]:
        """
        多 Trigger 融合召回：用行为列表中的每个物品做 I2I 检索，分数取最大值融合。
        """
        exclude = set(seq)
        fused: dict[int, float] = {}
        for trigger in seq:
            for cand, s in self.most_similar(trigger, top_k=self.n_items):
                if cand in exclude:
                    continue
                fused[cand] = max(fused.get(cand, -2.0), s)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items = 8, 20
    # 构造合成行为序列：把物品分成 4 个「兴趣簇」，用户主要在某个簇内行为
    clusters = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20))]
    user_seqs: dict[int, list[int]] = {}
    for u in range(n_users):
        main = clusters[u % len(clusters)]
        seq = random.sample(main, k=4)                       # 簇内 4 个物品
        seq += random.sample(range(n_items), k=4)            # 混入 4 个随机噪声物品
        random.shuffle(seq)
        user_seqs[u] = seq

    print("合成用户行为序列（session 视为无序集合）：")
    for u in range(n_users):
        print(f"  user {u}: {user_seqs[u]}")

    model = Item2Vec(n_items, dim=16, lr=0.05, n_neg=5)
    print("\nItem2Vec 训练：")
    model.train_model(user_seqs, epochs=6)

    print("\nI2I 相似物品（Top-5，簇内物品应彼此靠近）：")
    for trigger in (0, 6, 12):
        sims = model.most_similar(trigger, top_k=5)
        pretty = ", ".join(f"{i}({s:.3f})" for i, s in sims)
        print(f"  trigger item {trigger} -> {pretty}")

    seq = user_seqs[0]
    print(f"\n用户 0 的行为列表 {seq}")
    print("多 Trigger 融合召回 Top-5：")
    for iid, s in model.recommend(seq, top_k=5):
        print(f"  item {iid}: sim={s:.4f}")

    # 验证「无序性」：顺序颠倒后训练样本完全相同
    fwd = sorted(model._build_pairs({0: [1, 2, 3]}))
    bwd = sorted(model._build_pairs({0: [3, 2, 1]}))
    print(f"\n顺序颠倒后共现样本是否完全一致：{fwd == bwd}（Item2Vec 不建模时序）")
