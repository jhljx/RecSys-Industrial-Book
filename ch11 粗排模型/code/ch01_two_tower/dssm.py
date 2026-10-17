"""
DSSM 双塔粗排模型（第11章 §11.1.1）

粗排场景下的 DSSM：
- 用户塔和物品塔各为一个 MLP
- 训练时从精排日志中采样正/负样本
- 离线预计算物品 Embedding，线上仅计算用户 Embedding + 内积排序

本实现：PyTorch autograd，演示粗排双塔的核心结构与训练流程。
"""

import math
import random
import collections
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TowerMLP(nn.Module):
    """双塔共用的 MLP 结构：多层线性 + ReLU，最后一层无激活。"""

    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class DSSM(nn.Module):
    """
    粗排 DSSM 双塔模型。

    训练策略：Sampled Softmax（In-Batch 负采样），温度系数 τ
    匹配函数：余弦相似度
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 32,
        temperature: float = 0.05,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        self.temperature = temperature

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        dims = [emb_dim] + hidden_dims + [output_dim]
        self.user_tower = TowerMLP(dims)
        self.item_tower = TowerMLP(dims)

        # 初始化 Embedding
        nn.init.uniform_(self.user_emb.weight, -math.sqrt(1.0 / emb_dim), math.sqrt(1.0 / emb_dim))
        nn.init.uniform_(self.item_emb.weight, -math.sqrt(1.0 / emb_dim), math.sqrt(1.0 / emb_dim))

    def encode_user(self, uid: torch.Tensor) -> torch.Tensor:
        return self.user_tower(self.user_emb(uid))

    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        return self.item_tower(self.item_emb(iid))

    def forward(self, uid: torch.Tensor, iid: torch.Tensor) -> torch.Tensor:
        """
        In-Batch Sampled Softmax：
          sim[i, j] = cosine(u_emb[i], i_emb[j]) / temperature
          loss = cross_entropy(sim, arange(B))  — 对角线为正样本
        """
        u_emb = self.encode_user(uid)   # [B, D]
        i_emb = self.encode_item(iid)   # [B, D]
        # sim[i, j] = cosine(u_emb[i], i_emb[j])
        sim = F.cosine_similarity(u_emb.unsqueeze(1), i_emb.unsqueeze(0), dim=2) / self.temperature  # [B, B]
        labels = torch.arange(len(uid))
        loss = F.cross_entropy(sim, labels)
        return loss

    def score(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            u = torch.tensor([uid])
            i = torch.tensor([iid])
            ue = self.encode_user(u)
            ie = self.encode_item(i)
            return F.cosine_similarity(ue, ie).item()

    def pre_rank(self, uid: int, candidates: List[int], top_k: int = 20) -> List[Tuple[int, float]]:
        """粗排：对候选集打分并返回 Top-K。"""
        with torch.no_grad():
            u = torch.tensor([uid])
            ue = self.encode_user(u)  # [1, D]
            cand_t = torch.tensor(candidates)
            ie = self.encode_item(cand_t)  # [N, D]
            sims = F.cosine_similarity(ue, ie, dim=1)  # [N]
            scores = [(candidates[j], sims[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def train(
    model: DSSM,
    interactions: List[Tuple[int, int]],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.05,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(interactions)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(interactions), batch_size):
            batch = interactions[start: start + batch_size]
            if len(batch) < 2:
                continue
            uids = torch.tensor([u for u, _ in batch])
            iids = torch.tensor([i for _, i in batch])
            optimizer.zero_grad()
            loss = model(uids, iids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 2 == 0:
            print(f"  epoch {epoch:3d}  avg_loss={total_loss / max(n_batches, 1):.4f}")


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items = 12, 30
    interactions = list(set(
        (random.randint(0, n_users - 1), random.randint(0, n_items - 1))
        for _ in range(100)
    ))

    model = DSSM(n_users, n_items, emb_dim=16, hidden_dims=[32, 16], output_dim=16, temperature=0.1)
    print("DSSM 粗排模型训练：")
    train(model, interactions, epochs=6, batch_size=16, lr=0.05)

    uid = 0
    seen = {iid for u, iid in interactions if u == uid}
    candidates = [i for i in range(n_items) if i not in seen]
    result = model.pre_rank(uid, candidates, top_k=5)
    print(f"\n用户 {uid} 粗排 Top-5：")
    for iid, s in result:
        print(f"  item {iid}: {s:.4f}")
