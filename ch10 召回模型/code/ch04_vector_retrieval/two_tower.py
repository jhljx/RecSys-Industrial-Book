"""
双塔召回模型（第10章 §10.3.2）

核心思想：
- 用户塔和物品塔各为一个 MLP，分别将用户特征和物品特征映射到同一语义空间
- 训练用 In-Batch 负采样（同 batch 内其他物品作为负样本）
- 线上：提前计算物品 Embedding 存入向量数据库，查询用最近邻检索

本实现简化特征为 ID Embedding + MLP，使用 PyTorch autograd。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _build_tower(emb_dim: int, hidden_dims: list, output_dim: int) -> nn.Sequential:
    """构建单塔 MLP：emb_dim → hidden_dims → output_dim（ReLU 激活，最后一层线性输出）。"""
    layer_dims = [emb_dim] + hidden_dims + [output_dim]
    layers = []
    for i, (in_d, out_d) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(layer_dims) - 2:   # 最后一层不加激活
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TwoTowerRetrieval(nn.Module):
    """
    双塔召回模型。

    特征：用户/物品均用 ID Embedding（可扩展为其他特征）。
    训练：In-Batch 负采样 Softmax loss（温度系数 τ）。
    相似度：余弦相似度。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        hidden_dims: list = None,
        output_dim: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64]
        self.n_items = n_items
        self.temperature = temperature

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.user_tower = _build_tower(emb_dim, hidden_dims, output_dim)
        self.item_tower = _build_tower(emb_dim, hidden_dims, output_dim)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """返回 L2 归一化的用户向量 (B, output_dim)。"""
        x = self.user_emb(user_ids)
        out = self.user_tower(x)
        return F.normalize(out, dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """返回 L2 归一化的物品向量 (B, output_dim)。"""
        x = self.item_emb(item_ids)
        out = self.item_tower(x)
        return F.normalize(out, dim=-1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        In-Batch Sampled Softmax：构造相似度矩阵，对角线为正样本。
        返回 cross_entropy loss。
        """
        u_emb = self.encode_users(user_ids)   # (B, d)
        i_emb = self.encode_items(item_ids)   # (B, d)
        # 相似度矩阵 (B, B)
        sim = u_emb @ i_emb.T / self.temperature
        labels = torch.arange(len(user_ids))
        return F.cross_entropy(sim, labels)

    def train_model(
        self,
        interactions: list,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 0.01,
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            random.shuffle(interactions)
            total_loss = 0.0
            n_batches = 0
            self.train()
            for start in range(0, len(interactions), batch_size):
                batch = interactions[start: start + batch_size]
                if len(batch) < 2:
                    continue
                uids = torch.tensor([u for u, _ in batch], dtype=torch.long)
                iids = torch.tensor([i for _, i in batch], dtype=torch.long)
                optimizer.zero_grad()
                loss = self.forward(uids, iids)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if epoch % 2 == 0:
                print(f"  epoch {epoch:3d}  avg_loss={total_loss / max(n_batches, 1):.4f}")

    def get_item_embeddings(self) -> torch.Tensor:
        """获取所有物品的 Embedding（用于离线构建向量索引）。"""
        self.eval()
        with torch.no_grad():
            all_ids = torch.arange(self.n_items)
            return self.encode_items(all_ids)

    def recommend(self, uid: int, top_k: int = 10, exclude: set = None) -> list:
        exclude = exclude or set()
        self.eval()
        with torch.no_grad():
            u_emb = self.encode_users(torch.tensor([uid]))   # (1, d)
            item_ids = [i for i in range(self.n_items) if i not in exclude]
            i_tensor = torch.tensor(item_ids, dtype=torch.long)
            i_embs = self.encode_items(i_tensor)             # (n, d)
            scores = (u_emb @ i_embs.T).squeeze(0).tolist()
            pairs = list(zip(item_ids, scores))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 10, 20
    interactions = list(set(
        (random.randint(0, n_users - 1), random.randint(0, n_items - 1))
        for _ in range(80)
    ))

    model = TwoTowerRetrieval(
        n_users, n_items, emb_dim=16, hidden_dims=[16], output_dim=16, temperature=0.1
    )
    print("双塔召回训练：")
    model.train_model(interactions, epochs=6, batch_size=8, lr=0.05)

    uid = 0
    seen = {iid for u, iid in interactions if u == uid}
    recs = model.recommend(uid, top_k=5, exclude=seen)
    print(f"\n用户 {uid} Top-5 推荐：")
    for iid, s in recs:
        print(f"  item {iid}: {s:.4f}")
