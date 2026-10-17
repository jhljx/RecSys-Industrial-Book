"""
GraphSAGE 归纳式图采样聚合召回（第10章 §10.2.2 图网络召回 —— GraphSAGE召回）

核心思想：
    1) 邻居采样（Neighbor Sampling）：每层只采样固定数量 S 个邻居参与聚合，把每层
       复杂度从「全邻居」降到「常数 S」，从而能扩展到工业级大图；
    2) 归纳学习（Inductive Learning）：学习的是通用的聚合函数（Aggregator）+ 变换矩阵，
       而不是每个节点的 Embedding 表，新节点只要能拿到邻居就可直接推理，无需重训整图。

核心公式（对应正文公式）：
    S(v) ⊆ N(v), |S(v)| = S                               # 固定数量邻居采样
    h_N(v)^(l) = AGGREGATE^(l)({h_u^(l-1), u ∈ S(v)})       # 邻居聚合
    h_v^(l) = σ( W^(l) · CONCAT(h_v^(l-1), h_N(v)^(l)) )    # 自身与邻域分开建模
    h_v^(l) = h_v^(l) / ||h_v^(l)||_2                       # L2 归一化
    Mean Aggregator：(1/|S(v)|) Σ_u h_u
    Pooling Aggregator：m_u = σ(W_pool h_u + b)，再逐维 max

与 GCN / LightGCN / GAT 的关键区别：
    - GCN/GAT/LightGCN 是 Transductive（传导式）的：要对全图传播，新增节点必须重训；
      GraphSAGE 是 Inductive（归纳式）的，新节点只要有邻居即可直接推理（见 embed_new_node）；
    - 融合方式也不同：GCN 把自身和邻居「统一加权平均」，GraphSAGE 先聚合邻居，
      再与自身表示 CONCAT 后做线性变换，能分别建模自身特征与邻域特征。

参考论文：Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017

实现说明：模型为 PyTorch nn.Module，邻接表用 Python dict 存储，聚合与变换用 tensor 操作
    完成，梯度由 autograd 自动求导。
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ── 基础工具函数（本文件自包含，不跨文件 import） ────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class GraphSAGERecall(nn.Module):
    """二部图上的 GraphSAGE 召回模型，支持 mean / pooling 两种 Aggregator。
    PyTorch autograd 自动求导。"""

    def __init__(
        self, n_users: int, n_items: int, dim: int = 16, n_layers: int = 2,
        sample_sizes: tuple[int, ...] = (5, 3), aggregator: str = "mean",
        lr: float = 0.05, reg: float = 1e-4,
    ):
        super().__init__()
        assert aggregator in ("mean", "pool"), "aggregator 仅支持 mean / pool"
        self.n_users, self.n_items = n_users, n_items
        self.n_nodes = n_users + n_items  # 用户 u -> u，物品 i -> n_users + i
        self.dim, self.n_layers = dim, n_layers
        self.sample_sizes = list(sample_sizes) + [sample_sizes[-1]] * max(0, n_layers - len(sample_sizes))
        self.aggregator = aggregator
        self.lr, self.reg = lr, reg

        scale = 1.0 / math.sqrt(dim)
        self.emb = nn.Embedding(self.n_nodes, dim)
        nn.init.uniform_(self.emb.weight, -scale, scale)

        # W^(l)：形状 (dim, 2*dim)，作用在 CONCAT(自身, 邻域) 上
        self.W = nn.ModuleList([nn.Linear(2 * dim, dim, bias=False) for _ in range(n_layers)])
        # Pooling Aggregator 的 MLP 参数 (dim -> dim)
        self.W_pool = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        for lin in self.W:
            nn.init.xavier_normal_(lin.weight)
        for lin in self.W_pool:
            nn.init.xavier_normal_(lin.weight)

        self.nbrs: list[list[int]] = []
        self.layer_h: list[torch.Tensor] = []  # 各层节点表示，供归纳推理使用

    # ── 图构建与邻居采样 ──────────────────────────────────────────────────
    def build_graph(self, interactions: list[tuple[int, int]]) -> None:
        adj: dict[int, set[int]] = defaultdict(set)
        for u, i in interactions:
            adj[u].add(self.n_users + i)
            adj[self.n_users + i].add(u)
        self.nbrs = [sorted(adj[v]) for v in range(self.n_nodes)]

    def sample_neighbors(self, cand: list[int], size: int) -> list[int]:
        """固定数量邻居采样：邻居过多则无放回采样，过少则有放回重复采样。"""
        if not cand:
            return []
        if len(cand) >= size:
            return random.sample(cand, size)
        return [random.choice(cand) for _ in range(size)]

    # ── 聚合函数 ──────────────────────────────────────────────────────────
    def aggregate(self, layer: int, vecs: torch.Tensor) -> torch.Tensor:
        """AGGREGATE：mean 为逐维平均，pool 为「MLP + 逐维 Max Pooling」。
        vecs: (S, dim) — 已采样邻居的 Embedding；返回 (dim,)。"""
        if vecs.shape[0] == 0:
            return torch.zeros(self.dim)
        if self.aggregator == "mean":  # Mean Aggregator
            return vecs.mean(dim=0)
        # Pooling Aggregator：m_u = ReLU(W_pool h_u + b)，再逐维取最大
        m = torch.relu(self.W_pool[layer](vecs))   # (S, dim)
        return m.max(dim=0).values                 # (dim,)

    # ── 前向传播 ──────────────────────────────────────────────────────────
    def propagate(self) -> torch.Tensor:
        """逐层「采样 → 聚合 → CONCAT → 线性 → ReLU → L2 归一化」，返回 (N, dim)。"""
        h = self.emb.weight  # (N, dim)
        self.layer_h = [h]
        for layer in range(self.n_layers):
            new_h_list = []
            for v in range(self.n_nodes):
                sampled = self.sample_neighbors(self.nbrs[v], self.sample_sizes[layer])
                if sampled:
                    h_nbr = self.aggregate(layer, h[torch.tensor(sampled, dtype=torch.long)])
                else:
                    h_nbr = torch.zeros(self.dim)
                x = torch.cat([h[v], h_nbr], dim=0)   # CONCAT(自身, 邻域)，维度 2*dim
                pre = torch.relu(self.W[layer](x))     # ReLU
                new_h_list.append(F.normalize(pre, p=2, dim=0))  # L2 归一化
            h = torch.stack(new_h_list, dim=0)         # (N, dim)
            self.layer_h.append(h)
        return h

    def embed_new_node(self, neighbor_items: list[int], is_user: bool = True) -> list[float]:
        """
        归纳推理（Inductive）：为训练时未出现的新节点生成 Embedding。
        只需给出它的邻居，即可复用已学好的聚合函数与权重前向计算，无需重新训练。
        """
        assert self.layer_h, "请先调用 propagate / train_model"
        nbr_nodes = [self.n_users + i if is_user else i for i in neighbor_items]
        # 新节点没有自己的 ID Embedding，用邻居第 0 层表示的均值作为冷启动特征
        h_self = self.layer_h[0][torch.tensor(nbr_nodes, dtype=torch.long)].mean(dim=0) if nbr_nodes else torch.zeros(self.dim)
        with torch.no_grad():
            for layer in range(self.n_layers):
                sampled = self.sample_neighbors(nbr_nodes, self.sample_sizes[layer])
                if sampled:
                    h_nbr = self.aggregate(layer, self.layer_h[layer][torch.tensor(sampled, dtype=torch.long)])
                else:
                    h_nbr = torch.zeros(self.dim)
                x = torch.cat([h_self, h_nbr], dim=0)
                h_self = F.normalize(torch.relu(self.W[layer](x)), p=2, dim=0)
        return h_self.tolist()

    # ── 训练 ──────────────────────────────────────────────────────────────
    def train_model(self, interactions: list[tuple[int, int]], epochs: int = 6,
                    batch_size: int = 16) -> None:
        """BPR 损失训练，梯度由 PyTorch autograd 自动求导。"""
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
                    neg = random.randrange(self.n_items)
                    while neg in user_pos[u]:
                        neg = random.randrange(self.n_items)
                    u_ids.append(u)
                    p_ids.append(self.n_users + pos)
                    n_ids.append(self.n_users + neg)

                h = self.propagate()
                z_u = h[torch.tensor(u_ids, dtype=torch.long)]
                z_p = h[torch.tensor(p_ids, dtype=torch.long)]
                z_n = h[torch.tensor(n_ids, dtype=torch.long)]
                diff = (z_u * z_p).sum(-1) - (z_u * z_n).sum(-1)
                loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1

            print(f"  [{self.aggregator:4s}] epoch {epoch:2d}  BPR_loss={total_loss / max(n_batch, 1):.4f}")

    # ── 召回 ──────────────────────────────────────────────────────────────
    def recommend(self, uid: int, interactions: list[tuple[int, int]], top_k: int = 5):
        """余弦检索 Top-K 物品。"""
        with torch.no_grad():
            emb = self.propagate()
        seen = {i for u, i in interactions if u == uid}
        z_u = emb[uid].tolist()
        scores = [(i, _cosine(z_u, emb[self.n_users + i].tolist()))
                  for i in range(self.n_items) if i not in seen]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def recommend_for_vector(self, vec: list[float], top_k: int = 5):
        """给定任意 Embedding（例如新节点的归纳推理结果）做向量检索。"""
        assert self.layer_h, "请先调用 propagate / train_model"
        emb = self.layer_h[-1]
        scores = [(i, _cosine(vec, emb[self.n_users + i].tolist())) for i in range(self.n_items)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 8, 20

    # 构造带兴趣簇的合成二部图：前 4 个用户偏好物品 0~9，后 4 个偏好 10~19
    interactions = set()
    for u in range(n_users):
        lo, hi = (0, 9) if u < 4 else (10, 19)
        for _ in range(6):
            interactions.add((u, random.randint(lo, hi)))
        interactions.add((u, random.randrange(n_items)))  # 少量跨簇噪声
    interactions = sorted(interactions)
    print(f"合成二部图：{n_users} 个用户，{n_items} 个物品，{len(interactions)} 条交互边")

    models = {}
    for agg in ("mean", "pool"):
        random.seed(42)
        torch.manual_seed(42)
        print(f"\nGraphSAGE 训练（2 层，采样数 (5, 3)，aggregator={agg}）：")
        models[agg] = GraphSAGERecall(n_users, n_items, dim=16, n_layers=2, sample_sizes=(5, 3), aggregator=agg)
        models[agg].train_model(interactions, epochs=6)
        print(f"[{agg}] 用户 0（偏好物品 0~9）的 Top-5 召回：")
        for iid, s in models[agg].recommend(0, interactions, top_k=5):
            print(f"  item {iid:2d}  cosine={s:.4f}")

    # 归纳能力演示：训练时不存在的新用户，只给邻居即可直接推理出 Embedding
    new_vec = models["mean"].embed_new_node(neighbor_items=[1, 2, 3], is_user=True)
    print("\n归纳推理（新用户，历史行为 = item 1/2/3，未参与训练）的 Top-5 召回：")
    for iid, s in models["mean"].recommend_for_vector(new_vec, top_k=5):
        print(f"  item {iid:2d}  cosine={s:.4f}")
