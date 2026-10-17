"""
GCN 图卷积网络召回（第10章 §10.2.2 图网络召回 —— GCN召回）

核心思想：
    在用户-物品二部图上做图卷积，每一层让节点聚合「自身 + 一阶邻居」的信息，
    并经过可学习的线性变换与非线性激活；堆叠 L 层即可感知 L 跳邻域的高阶结构。

核心公式（对应正文公式）：
    Ã = A + I                                   # 加入自连接的邻接矩阵
    Â = D̃^(-1/2) Ã D̃^(-1/2)                     # 对称归一化（Renormalization Trick）
    H^(l+1) = σ( Â H^(l) W^(l) )                 # 逐层传播

与 LightGCN 的关键区别：
    LightGCN 把特征变换 W^(l) 和非线性激活 σ 全部去掉，只保留邻域聚合，
    并对各层输出做平均；而 GCN 每一层都保留 W^(l) 与 σ，表达能力更强但更容易
    过拟合，层数稍深还会出现 Over-smoothing（过平滑）问题，工业界一般只堆 2~3 层。

参考论文：Kipf & Welling, "Semi-Supervised Classification with Graph
         Convolutional Networks", ICLR 2017

实现说明：模型为 PyTorch nn.Module，图结构（归一化邻接矩阵）用 torch tensor 稠密存储，
    消息传递用张量乘法完成（本文件图规模很小，稠密矩阵足够；大图可换稀疏实现）。
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


# ── 基础工具函数（本文件自包含，不跨文件 import） ────────────────────────────
def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class GCNRecall(nn.Module):
    """二部图上的 GCN 召回模型（User-to-Item 召回），PyTorch autograd 自动求导。"""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 16,
        n_layers: int = 2,
        lr: float = 0.05,
        reg: float = 1e-4,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items  # 节点编号：用户 u -> u，物品 i -> n_users + i
        self.dim = dim
        self.n_layers = n_layers
        self.lr = lr
        self.reg = reg

        scale = 1.0 / math.sqrt(dim)
        # H^(0)：节点初始特征（这里直接用可学习的 ID Embedding 充当 X）
        self.emb = nn.Embedding(self.n_nodes, dim)
        nn.init.uniform_(self.emb.weight, -scale, scale)

        # W^(l)：每层的特征变换矩阵，初始化为「单位阵 + 小噪声」，保证前向数值稳定
        self.weights = nn.ModuleList()
        for _ in range(n_layers):
            lin = nn.Linear(dim, dim, bias=False)
            with torch.no_grad():
                lin.weight.copy_(torch.eye(dim) + (torch.rand(dim, dim) * 2 - 1) * 0.05)
            self.weights.append(lin)

        # Â：对称归一化邻接矩阵（稠密 tensor），build_graph 中填充
        self.adj_norm = torch.eye(self.n_nodes)

    # ── 图构建 ────────────────────────────────────────────────────────────
    def build_graph(self, interactions: list[tuple[int, int]]) -> None:
        """构建对称归一化邻接矩阵 Â = D̃^(-1/2)(A+I)D̃^(-1/2)（torch tensor 稠密存储）。"""
        nbrs: dict[int, set[int]] = defaultdict(set)
        for u, i in interactions:
            node_i = self.n_users + i
            nbrs[u].add(node_i)
            nbrs[node_i].add(u)

        # D̃ 的对角元 = 邻居数 + 1（自连接）
        deg = {v: len(nbrs[v]) + 1 for v in range(self.n_nodes)}

        adj = torch.zeros(self.n_nodes, self.n_nodes)
        for v in range(self.n_nodes):
            adj[v, v] = 1.0 / deg[v]  # 自连接权重 1/sqrt(d_v * d_v) = 1/d_v
            for n in nbrs[v]:
                adj[v, n] = 1.0 / math.sqrt(deg[v] * deg[n])
        self.adj_norm = adj

    # ── 前向传播 ──────────────────────────────────────────────────────────
    def propagate(self) -> torch.Tensor:
        """逐层传播 H^(l+1) = σ(Â H^(l) W^(l))，返回最终节点 Embedding (N, dim)。"""
        h = self.emb.weight  # (N, dim)
        for layer in range(self.n_layers):
            agg = self.adj_norm @ h               # Â H^(l)
            pre = self.weights[layer](agg)        # (Â H^(l)) W^(l)
            if layer == self.n_layers - 1:
                # 最后一层保持线性输出，便于直接用于向量检索（也避免 ReLU 截断梯度）
                h = pre
            else:
                h = torch.relu(pre)
        return h

    # ── 训练 ──────────────────────────────────────────────────────────────
    def train_model(self, interactions: list[tuple[int, int]], epochs: int = 6, batch_size: int = 16) -> None:
        """BPR 损失训练：更新每层 W^(l) 与初始 Embedding，梯度由 PyTorch autograd 自动求导。"""
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
                diff = (z_u * z_p).sum(-1) - (z_u * z_n).sum(-1)   # BPR 排序差
                loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1

            print(f"  epoch {epoch:2d}  BPR_loss={total_loss / max(n_batch, 1):.4f}")

    # ── 召回 ──────────────────────────────────────────────────────────────
    def recommend(self, uid: int, interactions: list[tuple[int, int]], top_k: int = 5):
        """余弦相似度检索：用户向量与全部物品向量算 cosine，取 Top-K。"""
        with torch.no_grad():
            emb = self.propagate()
        seen = {i for u, i in interactions if u == uid}
        z_u = emb[uid].tolist()
        scores = [(i, _cosine(z_u, emb[self.n_users + i].tolist()))
                  for i in range(self.n_items) if i not in seen]
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

    model = GCNRecall(n_users, n_items, dim=16, n_layers=2, lr=0.05)
    print("\nGCN 训练（2 层图卷积 + BPR 损失）：")
    model.train_model(interactions, epochs=6)

    print("\n用户 0（偏好物品 0~9）的 Top-5 召回：")
    for iid, s in model.recommend(0, interactions, top_k=5):
        print(f"  item {iid:2d}  cosine={s:.4f}")

    print("\n用户 7（偏好物品 10~19）的 Top-5 召回：")
    for iid, s in model.recommend(7, interactions, top_k=5):
        print(f"  item {iid:2d}  cosine={s:.4f}")
