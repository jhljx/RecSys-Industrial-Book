"""
HybridGNN 多重关系异构图召回（第10章 §10.2.2 图网络召回 —— HybridGNN召回）

核心思想（推荐系统中的用户反馈天然是多关系的：点击/点赞/分享/关注/作者创作…）：
    1) 跨关系随机探索：两阶段采样（先采样关系类型 r，再采样该关系下的邻居 v'），
       无需人工设计 Meta Path 也能发现隐藏的跨关系模式；
    2) 混合聚合：每种关系（以及随机探索流）各自用 relation-specific 变换聚合邻域，
       形成多条 Aggregation Flow；
    3) 层次化注意力：对多条聚合流做注意力融合，再与节点基础 Embedding 残差相加。

核心公式（对应正文公式）：
    p(v_{t+1}, r_{t+1}|v_t) = p(r_{t+1}|v_t) · p(v_{t+1}|r_{t+1}, v_t)   # 两阶段采样
    h_{v,r} = AGG({e_u, u ∈ N_r(v)}) · W_r                               # relation-specific 聚合
    β_{v,f} = softmax_f( LeakyReLU(a^T h_{v,f}) )                        # 跨关系注意力
    e*_v = e_v + ( Σ_f β_{v,f} h_{v,f} ) · W_out                         # 融合 + 残差

与 GCN / LightGCN / GraphSAGE / PinSage 的关键区别：
    - 上述模型都只在「单一关系的同构图或二部图」上传播（一条边就是一条边）；
    - HybridGNN 显式区分边的类型：每种关系有自己的变换矩阵与聚合流，再用注意力决定
      「这个用户的兴趣更多体现在点击还是分享上」，能联合建模多种行为语义；
    - 代价是参数量随关系数线性增长，图存储与采样也更复杂。

参考论文：Gu et al., "HybridGNN: Learning Hybrid Representation for Recommendation
         in Multiplex Heterogeneous Networks", 2022（基础组件沿用 GraphSAGE,
         Hamilton et al., NeurIPS 2017）

实现说明：模型为 PyTorch nn.Module，异构邻接表用 Python dict 存储，聚合与线性变换用
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
    return 0.0 if na < 1e-12 or nb < 1e-12 else sum(x * y for x, y in zip(a, b)) / (na * nb)


class HybridGNNRecall(nn.Module):
    """多重关系异构图上的 HybridGNN 召回模型（U2I 召回），PyTorch autograd 自动求导。"""

    RAND_FLOW = "rand"  # 跨关系随机探索对应的聚合流

    def __init__(
        self, n_users: int, n_items: int, n_authors: int, relations: tuple[str, ...],
        dim: int = 16, n_layers: int = 2, lr: float = 0.05, reg: float = 1e-4,
    ):
        super().__init__()
        self.n_users, self.n_items, self.n_authors = n_users, n_items, n_authors
        self.n_nodes = n_users + n_items + n_authors  # 节点编号：用户 | 物品 | 作者 三种类型
        self.relations = list(relations)
        self.flows = self.relations + [self.RAND_FLOW]  # 全部聚合流 = 各关系 + 随机探索
        self.dim, self.n_layers, self.lr, self.reg = dim, n_layers, lr, reg

        scale = 1.0 / math.sqrt(dim)
        self.emb = nn.Embedding(self.n_nodes, dim)
        nn.init.uniform_(self.emb.weight, -scale, scale)

        # 每层：每条聚合流一个 relation-specific 变换 Wf，外加融合矩阵 Wout 与注意力向量 attn_vec
        # 使用 ModuleList of ModuleDict 保证所有参数被 parameters() 收录
        self.Wf = nn.ModuleList([
            nn.ModuleDict({f: nn.Linear(dim, dim, bias=False) for f in self.flows})
            for _ in range(n_layers)
        ])
        self.Wout = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.attn_vec = nn.ParameterList([nn.Parameter(torch.randn(dim) * 0.1) for _ in range(n_layers)])

        for layer in range(n_layers):
            for f in self.flows:
                lin = self.Wf[layer][f]
                with torch.no_grad():
                    lin.weight.copy_(torch.eye(dim) + (torch.rand(dim, dim) * 2 - 1) * 0.1)
            with torch.no_grad():
                self.Wout[layer].weight.copy_(torch.eye(dim) + (torch.rand(dim, dim) * 2 - 1) * 0.05)

        self.rel_adj: dict[str, dict[int, list[int]]] = {r: defaultdict(list) for r in self.relations}
        self.rand_nbrs: list[list[int]] = []                # 跨关系随机探索得到的邻域
        self._beta: dict[int, list[tuple[str, float]]] = {}  # 最后一层跨关系注意力（便于观察）

    # ── 异构图构建 ────────────────────────────────────────────────────────
    def build_graph(self, edges: dict[str, list[tuple[int, int]]]) -> None:
        """edges[relation] = [(src, dst), ...]（全局节点编号），按关系类型分别存双向邻接表。"""
        self.rel_adj = {r: defaultdict(list) for r in self.relations}
        for r, pairs in edges.items():
            for src, dst in pairs:
                self.rel_adj[r][src].append(dst)
                self.rel_adj[r][dst].append(src)

    def explore(self, n_walks: int = 4, walk_len: int = 2) -> None:
        """跨关系随机探索：先均匀采样一种有邻居的关系 r，再在 N_r(v) 中均匀采样下一跳节点。"""
        self.rand_nbrs = [[] for _ in range(self.n_nodes)]
        for v in range(self.n_nodes):
            for _ in range(n_walks):
                cur = v
                for _ in range(walk_len):
                    cands = [r for r in self.relations if self.rel_adj[r][cur]]
                    if not cands:
                        break
                    r = random.choice(cands)                   # 第一阶段 p(r | v_t)
                    cur = random.choice(self.rel_adj[r][cur])  # 第二阶段 p(v_{t+1} | r, v_t)
                    if cur != v:
                        self.rand_nbrs[v].append(cur)

    # ── 前向传播：混合聚合 + 跨关系注意力融合 ─────────────────────────────
    def propagate(self) -> torch.Tensor:
        """逐层：按边类型分别聚合 → relation-specific 变换 → 跨关系注意力融合 → 残差相加。
        返回最终节点 Embedding (N, dim)。"""
        h = self.emb.weight   # (N, dim)，所有层共享初始 Embedding
        beta_rec: dict[int, list[tuple[str, float]]] = {}

        for layer in range(self.n_layers):
            last = layer == self.n_layers - 1
            new_h_list = []

            for v in range(self.n_nodes):
                flow_names, flow_vecs = [], []
                for f in self.flows:
                    nbrs = self.rand_nbrs[v] if f == self.RAND_FLOW else self.rel_adj[f][v]
                    if not nbrs:
                        continue
                    # h_{v,f} = AGG(邻居) · W_f（relation-specific 变换）
                    h_nbrs = h[torch.tensor(nbrs, dtype=torch.long)]   # (K, dim)
                    h_agg = h_nbrs.mean(dim=0)                         # Mean Aggregator: (dim,)
                    h_flow = self.Wf[layer][f](h_agg)                  # (dim,)
                    if not last:  # 中间层加非线性，最后一层保持线性便于检索
                        h_flow = F.leaky_relu(h_flow, negative_slope=0.2)
                    flow_names.append(f)
                    flow_vecs.append(h_flow)

                if flow_vecs:
                    # β_{v,f} = softmax(LeakyReLU(a^T h_{v,f}))
                    scores = torch.stack([
                        F.leaky_relu((self.attn_vec[layer] * hf).sum(), negative_slope=0.2)
                        for hf in flow_vecs
                    ])                                                  # (F,)
                    betas = torch.softmax(scores, dim=0)                # (F,)
                    fused = torch.stack(flow_vecs, dim=0)               # (F, dim)
                    fused_vec = (betas.unsqueeze(-1) * fused).sum(dim=0)  # (dim,)
                    if last:
                        beta_rec[v] = list(zip(flow_names, betas.tolist()))
                else:
                    fused_vec = torch.zeros(self.dim)
                    if last:
                        beta_rec[v] = []

                # e*_v = e_v + fused · W_out（残差融合）
                proj = self.Wout[layer](fused_vec)
                new_h_list.append(h[v] + proj)

            h = torch.stack(new_h_list, dim=0)   # (N, dim)

        self._beta = beta_rec
        return h

    # ── 训练 ──────────────────────────────────────────────────────────────
    def train_model(self, edges: dict[str, list[tuple[int, int]]], train_pairs: list[tuple[int, int]],
                    epochs: int = 6) -> None:
        """BPR 损失训练（以点击关系作为监督信号），梯度由 PyTorch autograd 自动求导。"""
        self.build_graph(edges)
        self.explore()   # 两阶段采样得到跨关系随机邻域
        user_pos: dict[int, set[int]] = defaultdict(set)
        for u, i in train_pairs:
            user_pos[u].add(i)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        samples = list(train_pairs)

        for epoch in range(epochs):
            random.shuffle(samples)
            u_ids, p_ids, n_ids = [], [], []
            for u, pos in samples:
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

            print(f"  epoch {epoch:2d}  BPR_loss={loss.item():.4f}")

    # ── 召回 ──────────────────────────────────────────────────────────────
    def recommend(self, uid: int, seen_items: set[int], top_k: int = 5):
        """余弦检索 Top-K 物品（用户/物品 Embedding 已融合全部关系的语义）。"""
        with torch.no_grad():
            emb = self.propagate()
        z_u = emb[uid].tolist()
        scores = [(i, _cosine(z_u, emb[self.n_users + i].tolist()))
                  for i in range(self.n_items) if i not in seen_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def relation_weights(self, node: int) -> list[tuple[str, float]]:
        """返回某节点最后一层各聚合流（关系 + 随机探索）的注意力权重 β。"""
        return sorted(self._beta.get(node, []), key=lambda x: x[1], reverse=True)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, n_authors = 8, 20, 4
    RELS = ("click", "like", "share", "follow", "create")
    item_base, author_base = n_users, n_users + n_items

    # 合成多重关系异构图：用户-物品（点击/点赞/分享）、用户-作者（关注）、作者-物品（创作）
    edges: dict[str, list[tuple[int, int]]] = {r: [] for r in RELS}
    click_pairs: list[tuple[int, int]] = []  # 以点击关系作为训练监督信号
    for u in range(n_users):
        lo, hi = (0, 9) if u < 4 else (10, 19)  # 前 4 个用户偏好物品 0~9，后 4 个偏好 10~19
        for _ in range(6):
            i = random.randint(lo, hi)
            edges["click"].append((u, item_base + i))
            click_pairs.append((u, i))
        edges["like"] += [(u, item_base + random.randint(lo, hi)) for _ in range(2)]
        edges["share"].append((u, item_base + random.randint(lo, hi)))
        edges["follow"].append((u, author_base + (0 if u < 4 else 2) + random.randint(0, 1)))
    for i in range(n_items):  # 作者创作物品：作者 0/1 创作前 10 个，作者 2/3 创作后 10 个
        edges["create"].append((author_base + (0 if i < 10 else 2) + random.randint(0, 1), item_base + i))
    click_pairs = sorted(set(click_pairs))
    print(f"合成异构图：{n_users} 用户 / {n_items} 物品 / {n_authors} 作者，关系 {len(RELS)} 种，"
          f"边数 {dict((r, len(v)) for r, v in edges.items())}")

    model = HybridGNNRecall(n_users, n_items, n_authors, RELS, dim=16, n_layers=2)
    print("\nHybridGNN 训练（按关系分别聚合 + 跨关系注意力融合 + BPR 损失）：")
    model.train_model(edges, click_pairs, epochs=6)

    with torch.no_grad():
        model.propagate()   # 刷新 _beta 记录
    print("\n用户 0 的跨关系注意力权重 β（哪些行为语义更重要）：")
    for name, b in model.relation_weights(0):
        print(f"  {name:>6s}  β={b:.4f}")

    for uid, desc in ((0, "偏好物品 0~9"), (7, "偏好物品 10~19")):
        seen = {i for u, i in click_pairs if u == uid}
        print(f"\n用户 {uid}（{desc}）的 Top-5 召回：")
        for iid, s in model.recommend(uid, seen, top_k=5):
            print(f"  item {iid:2d}  cosine={s:.4f}")
