"""
GAT 图注意力网络召回（第10章 §10.2.2 图网络召回 —— GAT召回）

核心思想：
    GCN 用固定的归一化系数 1/sqrt(d_i·d_j) 聚合邻居，GAT 则让模型自己学习
    「每个邻居有多重要」：先给邻居打注意力分数，softmax 归一化后再加权聚合，
    因此对度数分布不均、噪声边较多的推荐图更鲁棒，也更具可解释性。

核心公式（对应正文公式）：
    z_i = W h_i                                       # 共享线性变换
    e_ij = LeakyReLU( a^T [z_i ‖ z_j] )                # 未归一化注意力分数
    α_ij = exp(e_ij) / Σ_{k∈N(i)} exp(e_ik)            # 邻居维度 softmax
    h_i' = σ( Σ_{j∈N(i)} α_ij W h_j )                  # 注意力加权聚合
    多头：h_i' = ‖_{k=1..K} σ(Σ_j α_ij^(k) W^(k) h_j)   # 中间层拼接，最后一层取平均

与 GCN / LightGCN 的关键区别：
    - GCN：聚合权重由图结构（度数）事先算好，不可学习；LightGCN 更进一步把 W、σ
      也去掉，只做度归一化的邻域平均；
    - GAT：每条边的聚合权重 α_ij 都由 Embedding 学习得到，表达能力更强，
      代价是每条边都要算一次注意力，复杂度与显存占用明显高于 GCN。

参考论文：Veličković et al., "Graph Attention Networks", ICLR 2018

实现说明：模型为 PyTorch nn.Module，邻接表用普通 Python dict 存储；注意力分数与加权
    聚合用 tensor 掩码操作完成（masked softmax），梯度由 autograd 自动求导。
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


class GATRecall(nn.Module):
    """二部图上的多头图注意力召回模型（User-to-Item 召回），PyTorch autograd 自动求导。"""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 16,
        n_layers: int = 2,
        n_heads: int = 2,
        lr: float = 0.05,
        reg: float = 1e-4,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim 必须能被 n_heads 整除（中间层多头拼接）"
        self.n_users, self.n_items = n_users, n_items
        self.n_nodes = n_users + n_items  # 用户 u -> u，物品 i -> n_users + i
        self.dim, self.n_layers, self.n_heads = dim, n_layers, n_heads
        self.lr, self.reg = lr, reg

        scale = 1.0 / math.sqrt(dim)
        self.emb = nn.Embedding(self.n_nodes, dim)
        nn.init.uniform_(self.emb.weight, -scale, scale)

        # 每层多头参数：W^(k) 形状 (dim, d_out)，注意力向量 a^(k) 长度 2*d_out
        self.W = nn.ModuleList()
        self.A = nn.ParameterList()
        for layer in range(n_layers):
            # 中间层每头输出 dim/K（拼接回 dim）；最后一层每头输出 dim（多头取平均）
            d_out = dim if layer == n_layers - 1 else dim // n_heads
            heads = nn.ModuleList()
            for _ in range(n_heads):
                lin = nn.Linear(dim, d_out, bias=False)
                nn.init.xavier_normal_(lin.weight)
                heads.append(lin)
            self.W.append(heads)
            self.A.append(nn.Parameter((torch.rand(n_heads, 2 * d_out) * 2 - 1) * 0.1))

        self.nbrs: list[list[int]] = []                               # 邻接表（含自连接）
        self.attn: list[dict[int, list[tuple[int, float]]]] = []      # 各层注意力（头平均，便于观察）

    # ── 图构建 ────────────────────────────────────────────────────────────
    def build_graph(self, interactions: list[tuple[int, int]]) -> None:
        """构建邻接表，并按 GAT 惯例加入自连接（注意力也会分配给自己）。"""
        adj: dict[int, set[int]] = defaultdict(set)
        for u, i in interactions:
            adj[u].add(self.n_users + i)
            adj[self.n_users + i].add(u)
        self.nbrs = [sorted(adj[v]) + [v] for v in range(self.n_nodes)]

    # ── 前向传播 ──────────────────────────────────────────────────────────
    def _propagate_with_attn(self) -> torch.Tensor:
        """多头图注意力逐层传播，返回最终节点 Embedding (N, dim)；同时记录各层注意力。"""
        h = self.emb.weight  # (N, dim)
        n = self.n_nodes
        self.attn = []

        for layer in range(self.n_layers):
            last = layer == self.n_layers - 1
            d_out = self.dim if last else self.dim // self.n_heads
            head_outs = []
            attn_sum: dict[int, torch.Tensor] = {}

            for k in range(self.n_heads):
                z = self.W[layer][k](h)                 # (N, d_out): z_i = W h_i
                a_vec = self.A[layer][k]                # (2*d_out,)
                a_self, a_nbr = a_vec[:d_out], a_vec[d_out:]

                # e_ij = LeakyReLU(a^T [z_i ‖ z_j])
                e_self = (z @ a_self).unsqueeze(1)                       # (N, 1)
                e_nbr = (z @ a_nbr).unsqueeze(0)                         # (1, N)
                e = e_self + e_nbr                                       # (N, N)
                e = F.leaky_relu(e, negative_slope=0.2)

                # 掩码：仅保留邻接表（含自连接）中的边，其余置 -inf
                mask = torch.full((n, n), float("-inf"))
                for i in range(n):
                    for j in self.nbrs[i]:
                        mask[i, j] = 0.0
                alphas = torch.softmax(e + mask, dim=1)                  # (N, N) 邻居维度 softmax

                out_k = alphas @ z                                       # Σ_j α_ij W h_j
                head_outs.append(out_k)

                # 记录注意力（对多头求平均），供 attention_of 观察
                for i in range(n):
                    vals = alphas[i, self.nbrs[i]]
                    attn_sum[i] = vals if i not in attn_sum else attn_sum[i] + vals

            self.attn.append({
                v: sorted(zip(self.nbrs[v], (attn_sum[v] / self.n_heads).tolist()))
                for v in range(n)
            })

            if last:
                # 最后一层：多头取平均，且不加激活，直接作为检索向量
                h = torch.stack(head_outs, dim=0).mean(dim=0)
            else:
                # 中间层：多头拼接 + ELU 激活
                h = F.elu(torch.cat(head_outs, dim=-1))

        return h

    def propagate(self) -> torch.Tensor:
        """前向传播（不记录注意力观察值）。"""
        return self._propagate_with_attn()

    # ── 训练 ──────────────────────────────────────────────────────────────
    def train_model(self, interactions: list[tuple[int, int]], epochs: int = 6, batch_size: int = 16) -> None:
        """BPR 损失训练：更新注意力层参数与初始 Embedding，梯度由 autograd 自动求导。"""
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

                h = self._propagate_with_attn()
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
        """余弦检索 Top-K 物品。"""
        with torch.no_grad():
            emb = self._propagate_with_attn()
        seen = {i for u, i in interactions if u == uid}
        z_u = emb[uid].tolist()
        scores = [(i, _cosine(z_u, emb[self.n_users + i].tolist()))
                  for i in range(self.n_items) if i not in seen]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def attention_of(self, node: int, layer: int = 0) -> list[tuple[int, float]]:
        """返回某节点在指定层对各邻居的注意力权重（已对多头求平均）。"""
        return sorted(self.attn[layer][node], key=lambda x: x[1], reverse=True)


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

    model = GATRecall(n_users, n_items, dim=16, n_layers=2, n_heads=2)
    print("\nGAT 训练（2 层 × 2 头注意力 + BPR 损失）：")
    model.train_model(interactions, epochs=6)

    with torch.no_grad():
        model._propagate_with_attn()  # 重新前向一次，刷新注意力记录
    print("\n用户 0 第 1 层的邻居注意力权重 α_0j（GCN 中这些权重固定为 1/sqrt(d_i·d_j)）：")
    for j, al in model.attention_of(0, layer=0):
        print(f"  {'自身' if j == 0 else f'item {j - n_users}':>8s}  α={al:.4f}")

    for uid, desc in ((0, "偏好物品 0~9"), (7, "偏好物品 10~19")):
        print(f"\n用户 {uid}（{desc}）的 Top-5 召回：")
        for iid, s in model.recommend(uid, interactions, top_k=5):
            print(f"  item {iid:2d}  cosine={s:.4f}")
