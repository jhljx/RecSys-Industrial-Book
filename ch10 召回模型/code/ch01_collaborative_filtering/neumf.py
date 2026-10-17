"""
NeuMF 神经协同过滤（第10章 §10.1.6）

NeuMF = GMF（广义矩阵分解）+ MLP 两路并联，最终拼接后经 Sigmoid 输出点击概率。

参考论文：He et al., "Neural Collaborative Filtering", WWW 2017
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


class NeuMF(nn.Module):
    """
    NeuMF（PyTorch 实现，使用 autograd 自动求导）。

    架构：
      GMF 分支：element-wise 乘积 → 线性层
      MLP 分支：拼接 → 多层 ReLU 全连接
      融合：concat(gmf_out, mlp_out) → sigmoid
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 16,
        mlp_dims: list[int] | None = None,
        reg: float = 1e-4,
    ):
        super().__init__()
        mlp_dims = mlp_dims or [32, 16, 8]
        self.reg = reg
        self.gmf_dim = gmf_dim
        mlp_embed_dim = mlp_dims[0] // 2

        # GMF Embedding
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)
        # MLP Embedding
        self.mlp_user = nn.Embedding(n_users, mlp_embed_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_embed_dim)
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.xavier_normal_(emb.weight)

        # MLP 各层：in_dim → out_dim
        mlp_layers: list[nn.Module] = []
        dims = mlp_dims
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            mlp_layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 2:  # 中间层加 ReLU，最后一层保持线性
                mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        # 最终融合层：gmf_dim + mlp_dims[-1] → 1
        final_in = gmf_dim + mlp_dims[-1]
        self.out = nn.Linear(final_in, 1)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, uid: torch.Tensor, iid: torch.Tensor) -> torch.Tensor:
        """前向传播，返回点击概率（已过 sigmoid）。"""
        # GMF 分支：element-wise 乘积
        eu_gmf = self.gmf_user(uid)   # (B, gmf_dim)
        ei_gmf = self.gmf_item(iid)   # (B, gmf_dim)
        gmf_out = eu_gmf * ei_gmf     # (B, gmf_dim)

        # MLP 分支：拼接 → 多层全连接
        eu_mlp = self.mlp_user(uid)   # (B, mlp_embed_dim)
        ei_mlp = self.mlp_item(iid)   # (B, mlp_embed_dim)
        mlp_in = torch.cat([eu_mlp, ei_mlp], dim=-1)  # (B, 2*mlp_embed_dim)
        mlp_out = self.mlp(mlp_in)    # (B, mlp_dims[-1])

        # 融合：concat → sigmoid
        concat = torch.cat([gmf_out, mlp_out], dim=-1)  # (B, gmf_dim + mlp_dims[-1])
        logit = self.out(concat).squeeze(-1)           # (B,)
        return torch.sigmoid(logit)

    def train_model(self, interactions: list[tuple[int, int]], n_items: int,
                    epochs: int = 5, neg_ratio: int = 4, lr: float = 0.01) -> None:
        """BCE 损失训练，使用 PyTorch autograd。"""
        user_pos: dict[int, set[int]] = defaultdict(set)
        for uid, iid in interactions:
            user_pos[uid].add(iid)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)

        for epoch in range(epochs):
            samples = [(uid, iid, 1.0) for uid, iid in interactions]
            for uid, iid in interactions:
                for _ in range(neg_ratio):
                    neg = random.randint(0, n_items - 1)
                    while neg in user_pos[uid]:
                        neg = random.randint(0, n_items - 1)
                    samples.append((uid, neg, 0.0))
            random.shuffle(samples)

            u_t = torch.tensor([s[0] for s in samples], dtype=torch.long)
            i_t = torch.tensor([s[1] for s in samples], dtype=torch.long)
            y_t = torch.tensor([s[2] for s in samples], dtype=torch.float)

            optimizer.zero_grad()
            pred = self.forward(u_t, i_t)  # (B,)
            eps = 1e-7
            loss = -(y_t * torch.log(pred + eps) + (1 - y_t) * torch.log(1 - pred + eps)).mean()
            loss.backward()
            optimizer.step()

            print(f"  epoch {epoch}  avg_loss={loss.item():.4f}")

    def predict(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            u_t = torch.tensor([uid], dtype=torch.long)
            i_t = torch.tensor([iid], dtype=torch.long)
            return self.forward(u_t, i_t).item()

    def recommend(self, uid: int, n_items: int, top_k: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        exclude = exclude or set()
        u_t = torch.tensor([uid] * n_items, dtype=torch.long)
        i_t = torch.tensor(list(range(n_items)), dtype=torch.long)
        with torch.no_grad():
            scores = self.forward(u_t, i_t).tolist()
        result = [(iid, scores[iid]) for iid in range(n_items) if iid not in exclude]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 8, 15
    interactions = list(set((random.randint(0, n_users - 1), random.randint(0, n_items - 1)) for _ in range(60)))

    model = NeuMF(n_users, n_items, gmf_dim=8, mlp_dims=[16, 8, 4])
    print("NeuMF 训练：")
    model.train_model(interactions, n_items, epochs=5, neg_ratio=2, lr=0.05)

    uid = 0
    seen = {iid for u, iid in interactions if u == uid}
    recs = model.recommend(uid, n_items, top_k=5, exclude=seen)
    print(f"\n用户 {uid} 推荐结果：")
    for iid, score in recs:
        print(f"  item {iid}: {score:.4f}")
