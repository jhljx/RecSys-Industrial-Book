"""
LightGCN 图神经网络召回（第10章 §10.2.4）

LightGCN 去掉了 GCN 中的特征变换和非线性激活，
仅保留图上的邻域聚合（消息传递），训练速度快、效果好。

参考论文：He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020

消息传播公式：
    e_u^{(k+1)} = Σ_{i ∈ N(u)} e_i^{(k)} / sqrt(|N(u)| * |N(i)|)
    e_i^{(k+1)} = Σ_{u ∈ N(i)} e_u^{(k)} / sqrt(|N(i)| * |N(u)|)

最终 Embedding = 各层平均

实现说明：模型为 PyTorch nn.Module，归一化邻接矩阵用 torch tensor 存储，
    消息传递即稀疏矩阵乘法，梯度由 autograd 自动求导。
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 32,
        n_layers: int = 3,
        lr: float = 0.01,
        reg: float = 1e-4,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.lr = lr
        self.reg = reg

        scale = 1.0 / math.sqrt(dim)
        self.user_emb0 = nn.Embedding(n_users, dim)
        self.item_emb0 = nn.Embedding(n_items, dim)
        nn.init.uniform_(self.user_emb0.weight, -scale, scale)
        nn.init.uniform_(self.item_emb0.weight, -scale, scale)

        self.n_nodes = n_users + n_items
        self.adj_norm = torch.eye(self.n_nodes)  # Â = D^(-1/2) A D^(-1/2)

    def build_graph(self, interactions: list[tuple[int, int]]) -> None:
        """构建对称归一化邻接矩阵（torch 稠密 tensor，本文件图规模小）。"""
        user_items: dict[int, list[int]] = defaultdict(list)
        item_users: dict[int, list[int]] = defaultdict(list)
        for u, i in interactions:
            user_items[u].append(i)
            item_users[i].append(u)

        adj = torch.zeros(self.n_nodes, self.n_nodes)
        for u, items in user_items.items():
            du = len(items)
            for i in items:
                di = len(item_users[i])
                coeff = 1.0 / math.sqrt(du * di)
                adj[u, self.n_users + i] = coeff
                adj[self.n_users + i, u] = coeff
        self.adj_norm = adj

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN 前向传播：返回各层平均后的最终用户/物品 Embedding。
        e^(k+1) = Â e^(k)，最终 Embedding = 各层平均。
        """
        # 拼接为 (N, dim) 的统一节点表示：用户在前，物品在后
        h = torch.cat([self.user_emb0.weight, self.item_emb0.weight], dim=0)
        layers = [h]
        for _ in range(self.n_layers):
            h = self.adj_norm @ h   # 邻域聚合（已含度归一化）
            layers.append(h)

        final = torch.stack(layers, dim=0).mean(dim=0)  # 各层平均
        final_u = final[: self.n_users]
        final_i = final[self.n_users:]
        return final_u, final_i

    def train_model(
        self,
        interactions: list[tuple[int, int]],
        epochs: int = 10,
        batch_size: int = 16,
    ) -> None:
        """BPR 损失 + LightGCN 传播，梯度由 PyTorch autograd 自动求导。"""
        self.build_graph(interactions)
        user_pos: dict[int, set[int]] = defaultdict(set)
        for u, i in interactions:
            user_pos[u].add(i)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        samples = list(interactions)

        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss, n_batch = 0.0, 0
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                u_ids, p_ids, n_ids = [], [], []
                for u, pos in batch:
                    neg = random.randint(0, self.n_items - 1)
                    while neg in user_pos[u]:
                        neg = random.randint(0, self.n_items - 1)
                    u_ids.append(u)
                    p_ids.append(pos)
                    n_ids.append(neg)

                eu, ei = self.propagate()
                eu_b = eu[torch.tensor(u_ids, dtype=torch.long)]
                ei_p = ei[torch.tensor(p_ids, dtype=torch.long)]
                ei_n = ei[torch.tensor(n_ids, dtype=torch.long)]
                diff = (eu_b * ei_p).sum(-1) - (eu_b * ei_n).sum(-1)
                loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1

            if epoch % 2 == 0:
                print(f"  epoch {epoch:3d}  avg_loss={total_loss / max(n_batch, 1):.4f}")

    def recommend(
        self,
        uid: int,
        interactions: list[tuple[int, int]],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        with torch.no_grad():
            eu, ei = self.propagate()
        user_items: dict[int, list[int]] = defaultdict(list)
        for u, i in interactions:
            user_items[u].append(i)
        seen = set(user_items.get(uid, []))
        scores = [
            (iid, (eu[uid] * ei[iid]).sum().item())
            for iid in range(self.n_items)
            if iid not in seen
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 8, 12
    interactions = list(set((random.randint(0, n_users - 1), random.randint(0, n_items - 1)) for _ in range(40)))

    model = LightGCN(n_users, n_items, dim=8, n_layers=2, lr=0.05)
    print("LightGCN 训练：")
    model.train_model(interactions, epochs=6)

    recs = model.recommend(0, interactions, top_k=5)
    print("\n用户 0 推荐结果：")
    for iid, s in recs:
        print(f"  item {iid}: {s:.4f}")
