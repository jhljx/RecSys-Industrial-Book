"""
PinSage 工业级图卷积召回（第10章 §10.2.2 图网络召回 —— PinSage召回）

核心思想：
    PinSage 没有发明新的图卷积结构，而是在 GraphSAGE 框架上做了面向超大规模推荐
    系统的工程化改造，三个最核心的创新点：
    1) Importance Sampling：用 Random Walk with Restart（RWR）游走，以访问频次
       （Visit Count）衡量邻居重要性，只取 Top-T 个重要邻居；
    2) Importance Pooling：按重要性权重做加权平均，而不是等权 mean/max；
    3) Margin Ranking Loss + Hard Negative：训练目标直接对齐检索排序。

核心公式（对应正文公式）：
    访问频次 → α_u = count(u) / Σ_{u'∈N(v)} count(u')      # Importance Weight
    m_u = ReLU(Q h_u + q)                                   # 共享 MLP
    n_v = Σ_{u∈N(v)} α_u m_u                                # Importance Pooling（加权平均）
    z_v = ReLU(W [z_v ‖ n_v] + w)，z_v ← z_v / ||z_v||_2     # CONCAT + 线性 + L2 归一化
    L = Σ max(0, z_q·z_j - z_q·z_i + Δ)                      # Margin Ranking Loss

与 GraphSAGE / LightGCN 的关键区别：
    - GraphSAGE 对邻居「随机均匀采样 + 等权聚合」，PinSage 用 RWR 访问频次做
      「重要性采样 + 重要性加权聚合」，更聚焦真实兴趣传播路径；
    - GraphSAGE 用 Skip-Gram 式无监督目标，PinSage 用 Margin Ranking Loss，
      并刻意构造 Hard Negative（有点像但最终没被点击的物品）；
    - LightGCN 要全图传播，PinSage 只在 Mini-batch 的 K 跳子图上卷积，
      访问频次权重还能由 CPU 离线预计算，天然适合十亿级图。

参考论文：Ying et al., "Graph Convolutional Neural Networks for Web-Scale
         Recommender Systems", KDD 2018

实现说明：模型为 PyTorch nn.Module，邻接表用 Python list 存储，消息传递与线性变换用
    tensor 操作完成，梯度由 autograd 自动求导。
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 基础工具函数（本文件自包含，不跨文件 import） ────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class PinSageRecall(nn.Module):
    """基于 RWR 重要性采样 + Importance Pooling 的 PinSage 召回模型。
    PyTorch autograd 自动求导。"""

    def __init__(
        self, n_users: int, n_items: int, dim: int = 16, n_layers: int = 2, top_t: int = 5,
        margin: float = 0.2, lr: float = 0.05, reg: float = 1e-4,
    ):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_nodes = n_users + n_items  # 用户 u -> u，物品 i -> n_users + i
        self.dim, self.n_layers, self.top_t = dim, n_layers, top_t
        self.margin, self.lr, self.reg = margin, lr, reg

        scale = 1.0 / math.sqrt(dim)
        # 初始特征取非负（模拟 PinSage 中经过 ReLU 的视觉/文本特征），避免逐层 ReLU 把信号截断为 0
        self.emb = nn.Embedding(self.n_nodes, dim)
        nn.init.uniform_(self.emb.weight, 0.0, 2 * scale)

        # Q, q：邻居共享的 MLP (dim -> dim)；W, w：CONCAT (2*dim -> dim)
        self.Q = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.W = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(n_layers)])
        for lin in self.Q + self.W:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.adj: list[list[int]] = []
        self.importance: list[list[tuple[int, float]]] = []  # 每个节点的 Top-T 重要邻居及权重
        self.visit_count: list[list[tuple[int, int]]] = []   # 原始访问频次（便于观察）
        self.node_emb: torch.Tensor | None = None

    # ── 图构建 + RWR 重要性采样 ───────────────────────────────────────────
    def build_graph(self, interactions: list[tuple[int, int]]) -> None:
        """构建二部图邻接表（PinSage 工程上即 Pin-Board 图）。"""
        adj: dict[int, set[int]] = defaultdict(set)
        for u, i in interactions:
            adj[u].add(self.n_users + i)
            adj[self.n_users + i].add(u)
        self.adj = [sorted(adj[v]) for v in range(self.n_nodes)]

    def sample_importance_neighbors(self, n_walks: int = 60, walk_len: int = 6, restart: float = 0.5) -> None:
        """RWR：每步以 restart 概率跳回起点；访问频次作为邻居重要性，取 Top-T 并归一化。"""
        self.importance, self.visit_count = [], []
        for v in range(self.n_nodes):
            counts: dict[int, int] = defaultdict(int)
            for _ in range(n_walks):
                cur = v
                for _ in range(walk_len):
                    if random.random() < restart:
                        cur = v
                        continue
                    if not self.adj[cur]:
                        break
                    cur = random.choice(self.adj[cur])
                    # 只统计物品节点：用户节点得到 U2I 邻居，物品节点得到二跳的 I2I 邻居
                    if cur != v and cur >= self.n_users:
                        counts[cur] += 1
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_t]
            total = sum(c for _, c in top) or 1  # α_u = count(u) / Σ count
            self.importance.append([(n, c / total) for n, c in top])
            self.visit_count.append(top)

    # ── 前向传播（PinSage Convolution 模块） ──────────────────────────────
    def propagate(self) -> torch.Tensor:
        """逐层执行 Importance Pooling + CONCAT + 线性 + L2 归一化，返回 (N, dim)。"""
        h = self.emb.weight   # (N, dim)
        for layer in range(self.n_layers):
            new_h_list = []
            for v in range(self.n_nodes):
                # (1) 共享 MLP：m_u = ReLU(Q h_u + q)；(2) Importance Pooling：n_v = Σ α_u m_u
                if self.importance[v]:
                    nbr_ids = [u for u, _ in self.importance[v]]
                    alphas_t = torch.tensor([a for _, a in self.importance[v]], dtype=torch.float)
                    h_nbrs = h[torch.tensor(nbr_ids, dtype=torch.long)]   # (T, dim)
                    m = torch.relu(self.Q[layer](h_nbrs))                 # (T, dim)
                    n_v = (alphas_t.unsqueeze(-1) * m).sum(dim=0)         # (dim,)
                else:
                    n_v = torch.zeros(self.dim)
                # (3) CONCAT(自身, 邻域) 后线性变换 + ReLU + L2 归一化
                x = torch.cat([h[v], n_v], dim=0)    # (2*dim,)
                pre = torch.relu(self.W[layer](x))    # (dim,)
                new_h_list.append(F.normalize(pre, p=2, dim=0))
            h = torch.stack(new_h_list, dim=0)
        self.node_emb = h
        return h

    # ── Hard Negative 采样 ────────────────────────────────────────────────
    def _hard_negative(self, q_node: int, pos_items: set[int], emb: torch.Tensor, pool: int = 5) -> int:
        """从随机候选池里挑与 query 最相似但用户没点过的物品，构造 Hard Negative。"""
        cands = [j for j in (random.randrange(self.n_items) for _ in range(pool)) if j not in pos_items]
        if not cands:
            return random.randrange(self.n_items)
        eu = emb[q_node]
        best = max(cands, key=lambda j: (eu * emb[self.n_users + j]).sum().item())
        return best

    # ── 训练 ──────────────────────────────────────────────────────────────
    def train_model(self, interactions: list[tuple[int, int]], epochs: int = 6) -> None:
        """Margin Ranking Loss 训练，梯度由 PyTorch autograd 自动求导。"""
        self.build_graph(interactions)
        self.sample_importance_neighbors()
        user_pos: dict[int, set[int]] = defaultdict(set)
        for u, i in interactions:
            user_pos[u].add(i)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        samples = list(interactions)

        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss, n_violate = 0.0, 0

            # Full forward once per epoch (recompute per sample is too slow)
            h = self.propagate()
            batch_u, batch_p, batch_n = [], [], []
            for u, pos in samples:
                if random.random() < 0.5:
                    neg = self._hard_negative(u, user_pos[u], h.detach())
                else:
                    neg = random.randrange(self.n_items)
                    while neg in user_pos[u]:
                        neg = random.randrange(self.n_items)
                batch_u.append(u)
                batch_p.append(self.n_users + pos)
                batch_n.append(self.n_users + neg)

            h = self.propagate()  # recompute with grad
            z_u = h[torch.tensor(batch_u, dtype=torch.long)]
            z_p = h[torch.tensor(batch_p, dtype=torch.long)]
            z_n = h[torch.tensor(batch_n, dtype=torch.long)]
            # L = max(0, z_q·z_j - z_q·z_i + Δ)
            raw_loss = (z_u * z_n).sum(-1) - (z_u * z_p).sum(-1) + self.margin
            loss = torch.clamp(raw_loss, min=0.0).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = loss.item()
            n_violate = (raw_loss > 0).sum().item()
            print(f"  epoch {epoch:2d}  margin_loss={total_loss:.4f}  violate={n_violate}/{len(samples)}")

    # ── 召回 ──────────────────────────────────────────────────────────────
    def recommend(self, uid: int, interactions: list[tuple[int, int]], top_k: int = 5):
        """U2I 召回：用户向量检索 Top-K 物品。"""
        with torch.no_grad():
            emb = self.propagate()
        seen = {i for u, i in interactions if u == uid}
        z_u = emb[uid].tolist()
        scores = [(i, _cosine(z_u, emb[self.n_users + i].tolist()))
                  for i in range(self.n_items) if i not in seen]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def similar_items(self, iid: int, top_k: int = 5):
        """I2I 召回：PinSage 线上最常用的形态（只保留物品 Embedding 建 ANN 索引）。"""
        with torch.no_grad():
            emb = self.propagate()
        base = emb[self.n_users + iid].tolist()
        scores = [(j, _cosine(base, emb[self.n_users + j].tolist()))
                  for j in range(self.n_items) if j != iid]
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

    model = PinSageRecall(n_users, n_items, dim=16, n_layers=2, top_t=5, margin=0.2)
    print("\nPinSage 训练（RWR 重要性采样 + Importance Pooling + Margin Loss）：")
    model.train_model(interactions, epochs=6)

    print("\n用户 0 的 Top-5 重要邻居（RWR 访问频次 → Importance Weight）：")
    for (node, cnt), (_, alpha) in zip(model.visit_count[0], model.importance[0]):
        print(f"  item {node - n_users:2d}  visit_count={cnt:3d}  α={alpha:.4f}")

    for uid, desc in ((0, "偏好物品 0~9"), (7, "偏好物品 10~19")):
        print(f"\n用户 {uid}（{desc}）的 Top-5 U2I 召回：")
        for iid, s in model.recommend(uid, interactions, top_k=5):
            print(f"  item {iid:2d}  cosine={s:.4f}")

    print("\n物品 3 的 Top-5 I2I 召回（线上以物品 Embedding 建 ANN 索引）：")
    for jid, s in model.similar_items(3, top_k=5):
        print(f"  item {jid:2d}  cosine={s:.4f}")
