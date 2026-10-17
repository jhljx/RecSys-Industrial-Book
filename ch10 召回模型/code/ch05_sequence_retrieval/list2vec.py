r"""
List2Vec 召回（第10章 §10.4.1「List2Vec召回算法原理」）

参考：Barkan & Koenigstein, "Item2Vec: Neural Item Embedding for Collaborative
      Filtering", MLSP 2016（List2Vec 是工业界在 Item2Vec 之上的召回范式）

核心思想：
- List2Vec 不是某个具体模型，而是一类「以用户行为列表（List）为触发源」的召回范式：
  离线用行为列表语料学 Item Embedding，在线用最近行为列表做多 Trigger ANN 检索、融合去重。
- 本实现在 Item2Vec（纯 item-item 共现）的基础上，进一步把「列表/Session 级表示」
  纳入建模，联合优化两类关系：
    (1) item-item 关系：同一列表内任意两个物品互为正样本（Item2Vec 的 SGNS 目标）
        L_ii = -log σ(u_a · v_b) - Σ_n log σ(-u_a · v_n)
    (2) item-list 关系：列表向量应与列表内物品靠近、与列表外物品远离
        l_L\{j} = (1/|L\{j}|) Σ_{k in L, k != j} u_k          （列表向量 = 列表内物品向量聚合）
        L_il = -log σ(l · v_j) - Σ_n log σ(-l · v_n)
  联合损失：L = L_ii + alpha * L_il
- 召回阶段两种用法都支持：
    * 用单个物品向量做 I2I 检索（多 Trigger 融合）→ recommend_by_item / recommend
    * 用整个列表聚合出的列表向量做 L2I 检索          → recommend_by_list

本实现：PyTorch autograd，两类损失都用 F.binary_cross_entropy_with_logits 计算，
列表向量（Mean Pooling）本身是张量运算，梯度自动回传到每个成员物品向量。

说明：训练入口命名为 train_model，避免与 nn.Module.train() 冲突。
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── List2Vec 模型 ─────────────────────────────────────────────────────────────

class List2Vec(nn.Module):
    """
    List2Vec：item-item + item-list 联合训练的列表表示学习模型。

    参数矩阵：
      item_emb: [n_items × dim]  物品向量 u（既用于 I2I 检索，也用于聚合列表向量）
      ctx_emb:  [n_items × dim]  上下文/输出向量 v（训练辅助参数）
    """

    def __init__(
        self,
        n_items: int,
        dim: int = 16,
        lr: float = 0.05,
        n_neg: int = 5,
        alpha: float = 0.5,
        reg: float = 1e-5,
    ):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.lr = lr
        self.n_neg = n_neg
        self.alpha = alpha            # item-list 损失权重
        self.reg = reg

        scale = math.sqrt(1.0 / dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.ctx_emb = nn.Embedding(n_items, dim)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)
        nn.init.zeros_(self.ctx_emb.weight)

        self._neg_table: list[int] = []

    # ── 负采样表（freq^0.75） ─────────────────────────────────────────────────
    def _build_neg_table(self, user_lists: dict[int, list[int]], table_size: int = 2000) -> None:
        freq: dict[int, int] = defaultdict(int)
        for lst in user_lists.values():
            for iid in lst:
                freq[iid] += 1
        weights = {iid: c ** 0.75 for iid, c in freq.items()}
        total = sum(weights.values()) or 1.0
        table: list[int] = []
        for iid, w in weights.items():
            table.extend([iid] * max(1, int(round(table_size * w / total))))
        self._neg_table = table or list(range(self.n_items))

    def _sample_negs(self, exclude: set[int], k: int) -> list[int]:
        negs: list[int] = []
        for _ in range(k * 4):
            cand = random.choice(self._neg_table)
            if cand not in exclude:
                negs.append(cand)
            if len(negs) == k:
                break
        while len(negs) < k:
            cand = random.randrange(self.n_items)
            if cand not in exclude:
                negs.append(cand)
        return negs

    # ── 列表向量：列表内物品 Embedding 聚合（Mean Pooling，纯张量运算） ────────
    def list_vector(self, items: torch.Tensor) -> torch.Tensor:
        """l_L = (1/|L|) Σ_{k in L} u_k，输入为物品 id 张量 [T]。"""
        return self.item_emb(items).mean(dim=0)

    # ── 关系一：item-item（Item2Vec SGNS） ───────────────────────────────────
    def _item_item_loss(self, center: torch.Tensor, pos: torch.Tensor,
                        neg: torch.Tensor) -> torch.Tensor:
        u = self.item_emb(center)                      # [B, dim]
        v_pos = self.ctx_emb(pos)                      # [B, dim]
        v_neg = self.ctx_emb(neg)                      # [B, n_neg, dim]
        pos_logits = (u * v_pos).sum(-1)
        neg_logits = torch.bmm(v_neg, u.unsqueeze(-1)).squeeze(-1)
        return (F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)))

    # ── 关系二：item-list ─────────────────────────────────────────────────────
    def _item_list_loss(self, members: list[torch.Tensor], target: torch.Tensor,
                        neg: torch.Tensor) -> torch.Tensor:
        """
        用「列表（剔除 target 后）」的聚合向量预测 target，负样本为列表外物品。
        列表向量 l = mean(u_k) 是纯张量运算，梯度由 autograd 自动按 1/|L| 回传到成员向量。
        """
        if not members:
            return torch.zeros((), dtype=torch.float32)
        l_vec = torch.stack(members, dim=0).mean(dim=0)          # [dim]
        v_pos = self.ctx_emb(target)                             # [dim]
        v_neg = self.ctx_emb(neg)                                # [n_neg, dim]

        pos_logits = (l_vec * v_pos).sum()
        neg_logits = (v_neg * l_vec).sum(-1)
        loss = (F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)))
        return self.alpha * loss

    # ── 训练 ──────────────────────────────────────────────────────────────────
    def train_model(self, user_lists: dict[int, list[int]], epochs: int = 6,
                    batch_size: int = 64) -> None:
        """联合训练 item-item 与 item-list 两类关系。"""
        self._build_neg_table(user_lists)
        lists = [list(dict.fromkeys(l)) for l in user_lists.values() if len(set(l)) >= 2]

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        for epoch in range(epochs):
            random.shuffle(lists)
            l_ii = l_il = 0.0
            n_ii = n_il = 0

            # (1) item-item：整列表两两共现（无滑动窗口，继承 Item2Vec）
            pairs = [(a, b, tuple(items)) for items in lists for a in items for b in items if a != b]
            random.shuffle(pairs)
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start:start + batch_size]
                center = torch.tensor([c for c, _, _ in chunk], dtype=torch.long)
                pos = torch.tensor([p for _, p, _ in chunk], dtype=torch.long)
                forbid = [set(s) for _, _, s in chunk]
                neg = torch.tensor([self._sample_negs(f, self.n_neg) for f in forbid],
                                   dtype=torch.long)
                optimizer.zero_grad()
                loss = self._item_item_loss(center, pos, neg)
                loss.backward()
                optimizer.step()
                l_ii += loss.item()
                n_ii += 1

            # (2) item-list：列表级表示预测列表内物品
            for items in lists:
                forbid = set(items)
                members_all = list(dict.fromkeys(items))
                for target in items:
                    members = [torch.tensor(m, dtype=torch.long) for m in members_all
                               if m != target]
                    if not members:
                        continue
                    member_vecs = [self.item_emb(m) for m in members]
                    neg = torch.tensor(self._sample_negs(forbid, self.n_neg), dtype=torch.long)
                    optimizer.zero_grad()
                    loss = self._item_list_loss(
                        member_vecs, torch.tensor(target, dtype=torch.long), neg)
                    loss.backward()
                    optimizer.step()
                    l_il += loss.item()
                    n_il += 1

            print(
                f"  epoch {epoch:2d}  item_item_loss={l_ii / max(n_ii, 1):.4f}  "
                f"item_list_loss={l_il / max(n_il, 1):.4f}"
            )

    # ── 召回接口 ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _normalized_item_table(self) -> torch.Tensor:
        return F.normalize(self.item_emb.weight, dim=-1)

    def recommend_by_item(self, iid: int, top_k: int = 5) -> list[tuple[int, float]]:
        """I2I：用单个 Trigger 物品向量检索。"""
        with torch.no_grad():
            W = self._normalized_item_table()
            sims = W @ W[iid]
            sims[iid] = -1e9
            top_vals, top_idx = sims.topk(min(top_k, self.n_items - 1))
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]

    def recommend_by_list(self, items: list[int], top_k: int = 5) -> list[tuple[int, float]]:
        """L2I：用整个行为列表聚合出的列表向量检索。"""
        with torch.no_grad():
            idx = torch.tensor(list(dict.fromkeys(items)), dtype=torch.long)
            l_vec = F.normalize(self.list_vector(idx), dim=-1)
            W = self._normalized_item_table()
            sims = W @ l_vec
            sims[idx] = -1e9
            top_vals, top_idx = sims.topk(top_k)
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]

    def recommend(self, items: list[int], top_k: int = 5, trigger_limit: int = 5) -> list[tuple[int, float]]:
        """
        工业界 List2Vec 在线流程：多 Trigger（取最近若干行为）分别 ANN 检索，
        再与列表向量召回结果融合去重（分数取最大值）。
        """
        exclude = set(items)
        triggers = list(dict.fromkeys(reversed(items)))[:trigger_limit]
        fused: dict[int, float] = {}
        for trig in triggers:
            for cand, s in self.recommend_by_item(trig, top_k=self.n_items - 1):
                if cand not in exclude:
                    fused[cand] = max(fused.get(cand, -2.0), s)
        for cand, s in self.recommend_by_list(items, top_k=self.n_items):
            if cand not in exclude:
                fused[cand] = max(fused.get(cand, -2.0), s)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 8, 20
    clusters = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20))]
    user_lists: dict[int, list[int]] = {}
    for u in range(n_users):
        main = clusters[u % len(clusters)]
        lst = random.sample(main, k=4) + random.sample(range(n_items), k=4)
        random.shuffle(lst)
        user_lists[u] = lst

    print("合成用户行为列表：")
    for u in range(n_users):
        print(f"  user {u}: {user_lists[u]}")

    model = List2Vec(n_items, dim=16, lr=0.05, n_neg=5, alpha=0.5)
    print("\nList2Vec 联合训练（item-item + item-list）：")
    model.train_model(user_lists, epochs=6)

    items = user_lists[0]
    print(f"\n用户 0 的行为列表：{items}")

    print("I2I 召回（单 Trigger = 列表最后一个物品，未过滤历史）Top-5：")
    for iid, s in model.recommend_by_item(items[-1], top_k=5):
        print(f"  item {iid}: sim={s:.4f}")

    print("L2I 召回（列表向量）Top-5：")
    for iid, s in model.recommend_by_list(items, top_k=5):
        print(f"  item {iid}: sim={s:.4f}")

    print("多 Trigger + 列表向量融合召回 Top-5：")
    for iid, s in model.recommend(items, top_k=5):
        print(f"  item {iid}: score={s:.4f}")
