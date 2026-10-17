"""
矩阵分解召回 —— BPR-MF（第10章 §10.1.5）

使用贝叶斯个性化排序（BPR）损失训练 Embedding，
适合隐式反馈场景（点击/观看，无显式评分）。

模型：
    score(u, i) = e_u · e_i
    BPR loss: -log σ(score(u, pos) - score(u, neg))
"""

import random
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, reg: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.reg = reg
        # Xavier 初始化
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward(self, uid: torch.Tensor, iid: torch.Tensor) -> torch.Tensor:
        """返回 user-item 内积得分。"""
        eu = self.user_emb(uid)   # (B, dim)
        ei = self.item_emb(iid)   # (B, dim)
        return (eu * ei).sum(dim=-1)  # (B,)

    def train_model(self, interactions: list[tuple[int, int]], n_items: int,
                    epochs: int = 10, lr: float = 0.01) -> None:
        """BPR 训练：每个正样本随机采一个负样本。使用 PyTorch autograd。"""
        user_pos: dict[int, set[int]] = defaultdict(set)
        for uid, iid in interactions:
            user_pos[uid].add(iid)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)

        for epoch in range(epochs):
            random.shuffle(interactions)
            total_loss = 0.0
            optimizer.zero_grad()

            uids, pos_ids, neg_ids = [], [], []
            for uid, pos_id in interactions:
                neg_id = random.randint(0, n_items - 1)
                while neg_id in user_pos[uid]:
                    neg_id = random.randint(0, n_items - 1)
                uids.append(uid)
                pos_ids.append(pos_id)
                neg_ids.append(neg_id)

            u_t = torch.tensor(uids, dtype=torch.long)
            p_t = torch.tensor(pos_ids, dtype=torch.long)
            n_t = torch.tensor(neg_ids, dtype=torch.long)

            s_pos = self.forward(u_t, p_t)
            s_neg = self.forward(u_t, n_t)
            # BPR loss: -log σ(s_pos - s_neg)
            loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-10).mean()

            loss.backward()
            optimizer.step()
            total_loss = loss.item()

            if epoch % 2 == 0:
                print(f"  epoch {epoch:3d}  avg_loss={total_loss:.4f}")

    def recommend(self, uid: int, top_k: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """对用户 uid 按内积分排序返回 Top-K 物品。"""
        exclude = exclude or set()
        n_items = self.item_emb.weight.shape[0]
        with torch.no_grad():
            eu = self.user_emb(torch.tensor([uid], dtype=torch.long))  # (1, dim)
            all_items = self.item_emb.weight  # (n_items, dim)
            scores = (eu * all_items).sum(dim=-1).tolist()
        result = [(iid, scores[iid]) for iid in range(n_items) if iid not in exclude]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 10, 20
    # 模拟交互数据
    interactions = [(random.randint(0, n_users - 1), random.randint(0, n_items - 1)) for _ in range(200)]
    interactions = list(set(interactions))  # 去重

    model = MatrixFactorization(n_users, n_items, dim=16, reg=1e-3)
    print("BPR-MF 训练过程：")
    model.train_model(interactions, n_items, epochs=10, lr=0.05)

    uid = 0
    seen = {iid for u, iid in interactions if u == uid}
    recs = model.recommend(uid, top_k=5, exclude=seen)
    print(f"\n用户 {uid} 的推荐结果：")
    for iid, score in recs:
        print(f"  item {iid:3d}: {score:.4f}")
