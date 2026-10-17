"""
MVKE 粗排多兴趣模型（第11章 §11.5.1）

MVKE = Mixture of Virtual-Kernel Experts（虚拟核专家混合模型）
论文出处：Xu et al., "Mixture of Virtual-Kernel Experts for Multi-Objective User Profile
          Modeling", KDD 2022（腾讯）。

核心思想（对应书中 §11.5.1）：
    传统双塔把用户多样化兴趣压成一个固定 Embedding，表达能力不足。MVKE 保持双塔的高效
    架构，但把用户塔换成 K 个虚拟核专家（VKE），再用虚拟核门控（VKG）按目标物品动态聚合：
    1) 第 k 个 VKE 维护一个可学习虚拟核 W_VK^k（隐式的「兴趣查询」）。虚拟核经非线性映射
       得到 Query，用户字段 Embedding 映射为 Key / Value：
           Q = σ(W_Qᵀ·W_VK^k + b_Q)，K = σ(W_Kᵀ·E_uf + b_K)，V = σ(W_Vᵀ·E_uf + b_V)
       注意力聚合：C_VKE^k = softmax(Q·Kᵀ / √d_K)·V
       兴趣表示：    E_ui^k = f_u^k(C_VKE^k)
       所有 VKE 共享同一份用户字段 Embedding，但虚拟核不同 → 关注兴趣的不同侧面。
    2) VKG 以目标物品 Embedding E_Ti 为 Query，以虚拟核 {W_VK^k} 为 Key，以 VKE 输出
       {E_ui^k} 为 Value：
           E_ui = Σ_k softmax( Q(E_Ti)·K(W_VK^k)ᵀ / √d_K ) · V(E_ui^k)
       同一用户面对不同目标物品时，门控权重不同 → 动态多兴趣表征。
    3) 在线效率：K 组 VKE 兴趣表示可离线预计算并存储，线上只需算 K 个门控权重再加权求和，
       相比双塔只增加存储开销，不显著增加计算复杂度。

PyTorch 实现（autograd 自动微分），可直接 `python3 mvke.py` 运行。
"""
import math
import random
import collections
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _dot_py(a, b):
    return sum(x * y for x, y in zip(a, b))

def _sigmoid_py(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

def _auc(pairs: List[Tuple[float, int]]) -> float:
    pos, neg = [s for s, y in pairs if y == 1], [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return 0.5
    hit = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return hit / (len(pos) * len(neg))


class VKE(nn.Module):
    """虚拟核专家：以可学习虚拟核 W_VK^k 为 Query，对用户各字段 Embedding 做注意力聚合。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, n_fields: int, field_dim: int, d_vk: int = 8, d_k: int = 8,
                 d_v: int = 8, out_dim: int = 8, init_offset: float = 0.0):
        super().__init__()
        self.m = n_fields
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

        # 可学习虚拟核 W_VK^k（一维向量）；init_offset 使不同专家初始化错开，打破对称性
        w = torch.empty(d_vk).uniform_(-1/math.sqrt(d_vk), 1/math.sqrt(d_vk))
        w[0] = w[0].abs() * (1.0 if init_offset >= 0 else -1.0) + init_offset
        self.w_vk = nn.Parameter(w)

        # Q = ReLU(W_Q · W_VK + b_Q)
        self.q_proj = nn.Linear(d_vk, d_k)
        # K = ReLU(W_K · e + b_K)，V = ReLU(W_V · e + b_V)  （逐字段共享权重）
        self.k_proj = nn.Linear(field_dim, d_k)
        self.v_proj = nn.Linear(field_dim, d_v)
        # 兴趣建模网络 f_u^k：线性输出（保持正负极性）
        self.f_u = nn.Linear(d_v, out_dim)

    def forward(self, fields: torch.Tensor):
        """
        fields: (n_fields, field_dim)
        Returns (E_ui_k: (out_dim,), attn: (n_fields,))
        """
        Q = torch.tanh(self.q_proj(self.w_vk))                    # (d_k,)
        K = F.relu(self.k_proj(fields))                            # (n_fields, d_k)
        V = F.relu(self.v_proj(fields))                            # (n_fields, d_v)
        logits = (K @ Q) / self.scale                              # (n_fields,)
        a = torch.softmax(logits, dim=0)                           # (n_fields,)
        C = (a.unsqueeze(1) * V).sum(dim=0)                        # (d_v,)
        E = self.f_u(C)                                            # (out_dim,)
        return E, a


class VKG(nn.Module):
    """虚拟核门控：目标物品 Embedding 作 Query，虚拟核作 Key，VKE 兴趣表示作 Value。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, vkes: nn.ModuleList, item_dim: int, out_dim: int, d_k: int = 8):
        super().__init__()
        self.vkes = vkes
        self.K = len(vkes)
        self.d_k = d_k
        self.scale = math.sqrt(d_k)
        # Q(E_Ti): ReLU 激活，保持与 VKE 中 Q 网络一致的风格
        self.q_proj = nn.Linear(item_dim, d_k)
        # K(W_VK^k): 线性投影（无激活），使门控 logits 可正可负，不同虚拟核对比度更高
        d_vk = vkes[0].w_vk.shape[0]
        self.k_proj = nn.Linear(d_vk, d_k)
        # V(E_ui^k): 线性投影
        self.v_proj = nn.Linear(out_dim, out_dim)

    def forward(self, item_e: torch.Tensor, interests: List[torch.Tensor]):
        """
        item_e: (item_dim,)
        interests: list of K tensors, each (out_dim,)
        Returns (E_ui: (out_dim,), gate: (K,))
        """
        Q = torch.tanh(self.q_proj(item_e))                        # (d_k,)
        # Stack virtual kernel keys
        vk_stack = torch.stack([vke.w_vk for vke in self.vkes])    # (K, d_vk)
        Ks = self.k_proj(vk_stack)                                 # (K, d_k)
        logits = (Ks @ Q) / self.scale                             # (K,)
        gate = torch.softmax(logits, dim=0)                        # (K,)
        int_stack = torch.stack(interests)                         # (K, out_dim)
        Vs = self.v_proj(int_stack)                                # (K, out_dim)
        E_ui = (gate.unsqueeze(1) * Vs).sum(dim=0)                 # (out_dim,)
        return E_ui, gate


class MVKE(nn.Module):
    """MVKE 粗排模型：K 个 VKE 构成用户塔 + VKG 目标相关门控 + 轻量目标塔。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, n_users: int, n_items: int, n_u_fields: int = 4, n_i_fields: int = 2,
                 field_dim: int = 4, n_vke: int = 3, d_vk: int = 8, d_k: int = 8,
                 out_dim: int = 8, lr: float = 0.02, reg: float = 1e-5, emb_scale: float = 0.8):
        super().__init__()
        self.n_items = n_items
        self.mu = n_u_fields
        self.mi = n_i_fields
        self.fd = field_dim
        self.K = n_vke
        self.out_dim = out_dim

        self.user_emb = nn.Parameter(
            torch.empty(n_users, n_u_fields, field_dim).uniform_(-emb_scale, emb_scale))
        self.item_emb = nn.Parameter(
            torch.empty(n_items, n_i_fields, field_dim).uniform_(-emb_scale, emb_scale))

        # init_offset staggers virtual kernels so experts break symmetry from step 0
        offsets = [k * 1.0 - (n_vke - 1) / 2.0 for k in range(n_vke)]
        self.vkes = nn.ModuleList([
            VKE(n_u_fields, field_dim, d_vk, d_k, out_dim, out_dim, init_offset=offsets[k])
            for k in range(n_vke)
        ])
        # 目标塔：两层 MLP，输出层线性（使 E_Ti 可正可负，内积可区分负样本）
        self.item_l1 = nn.Linear(n_i_fields * field_dim, out_dim)
        self.item_l2 = nn.Linear(out_dim, out_dim)

        self.vkg = VKG(self.vkes, out_dim, out_dim, d_k)
        self.bias = nn.Parameter(torch.zeros(1))

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def forward(self, uid: int, iid: int):
        """Returns (logit scalar, gate (K,), attn list of (n_u_fields,))"""
        fields = self.user_emb[uid]                                # (n_u_fields, field_dim)
        interests, attns = [], []
        for vke in self.vkes:
            E, a = vke(fields)
            interests.append(E)
            attns.append(a)

        item_flat = self.item_emb[iid].reshape(-1)                 # (n_i_fields * field_dim,)
        h = F.relu(self.item_l1(item_flat))
        E_T = self.item_l2(h)                                      # (out_dim,) — linear output

        E_ui, gate = self.vkg(E_T, interests)
        logit = torch.dot(E_ui, E_T) / math.sqrt(self.out_dim) + self.bias.squeeze()
        return logit, gate, attns

    def score(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            logit, _, _ = self.forward(uid, iid)
            return float(torch.sigmoid(logit).item())

    def gate_of(self, uid: int, iid: int) -> List[float]:
        with torch.no_grad():
            _, gate, _ = self.forward(uid, iid)
            return gate.tolist()

    def field_attention(self, uid: int) -> List[List[float]]:
        """各 VKE 对用户字段的注意力分布（体现不同虚拟核关注兴趣的不同侧面）。"""
        with torch.no_grad():
            fields = self.user_emb[uid]
            return [vke(fields)[1].tolist() for vke in self.vkes]

    def train_model(self, pos_pairs, epochs: int = 6) -> None:
        data, pos_set = list(pos_pairs), set(pos_pairs)
        for epoch in range(epochs):
            random.shuffle(data)
            total = 0.0
            for uid, iid in data:
                self.optimizer.zero_grad()
                # Positive sample
                logit_pos, _, _ = self.forward(uid, iid)
                loss = F.binary_cross_entropy_with_logits(logit_pos, torch.tensor(1.0))
                # Negative sample
                nid = random.randrange(self.n_items)
                while (uid, nid) in pos_set:
                    nid = random.randrange(self.n_items)
                logit_neg, _, _ = self.forward(uid, nid)
                loss = loss + F.binary_cross_entropy_with_logits(logit_neg, torch.tensor(0.0))
                loss.backward()
                self.optimizer.step()
                total += loss.item()
            print(f"  epoch {epoch}  logloss={total / max(1, len(data) * 2):.4f}")

    def pre_rank(self, uid: int, candidates: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """粗排：K 组兴趣表示可预计算，线上逐候选只需算门控权重 + 加权求和 + 一次内积。"""
        scored = [(iid, self.score(uid, iid)) for iid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, n_vke = 10, 30, 3

    # 合成多兴趣数据：每个用户有 3 个兴趣分量（与 K=3 簇对应），用户偏好分量 k 的物品簇 k。
    # 这样不同簇的物品需要激活不同专家，驱动 VKG 产生簇相关的门控分布。
    n_clusters = 3
    field_dim = 4
    cluster = [i % n_clusters for i in range(n_items)]

    # 用户在每个簇方向上有独立的兴趣强度（4维）；物品完全由其簇方向决定
    interest_dirs = [[random.gauss(0, 1) for _ in range(field_dim)] for _ in range(n_clusters)]
    user_modes = [[[random.gauss(0, 1) for _ in range(field_dim)]
                   for _ in range(n_clusters)] for _ in range(n_users)]
    item_vec = [interest_dirs[cluster[i]] for i in range(n_items)]

    pos_pairs = []
    pos_set = set()
    for u in range(n_users):
        for i in range(n_items):
            c = cluster[i]
            s = _dot_py(user_modes[u][c], item_vec[i]) / field_dim
            if s > 0.2:
                pos_pairs.append((u, i))
                pos_set.add((u, i))

    model = MVKE(n_users, n_items, n_u_fields=4, n_i_fields=2, n_vke=n_vke)
    print(f"MVKE 粗排：n_users={n_users}  n_items={n_items}  VKE 数 K={n_vke}  "
          f"正样本对={len(pos_pairs)}")

    print("\n[1] 训练（用户塔 = K 个虚拟核专家，门控由目标物品决定）")
    model.train_model(pos_pairs, epochs=12)
    auc = _auc([(model.score(u, i), 1 if (u, i) in pos_set else 0)
                for u in range(n_users) for i in range(n_items)])
    print(f"  训练集 AUC = {auc:.4f}")

    print("\n[2] 各 VKE 内部的字段注意力 softmax(Q·Kᵀ/√d_K)：不同虚拟核关注不同字段")
    for u in [0, 3]:
        for k, a in enumerate(model.field_attention(u)):
            print(f"  用户{u} VKE{k}: " + "  ".join(f"字段{i}={v:.3f}" for i, v in enumerate(a)))

    print("\n[3] VKG 门控权重分布（Query=目标物品 Embedding，故门控由目标侧驱动）")
    sample_items = [i * (n_items // (n_clusters * 2)) for i in range(n_clusters * 2)][:6]
    for i in sample_items:
        g = model.gate_of(0, i)
        print(f"  目标 item{i:2d}（簇{cluster[i]}）：门控=[" + ", ".join(f"{v:.3f}" for v in g)
              + f"]  最偏好 VKE{g.index(max(g))}")
    spread = [max(model.gate_of(0, i)) - min(model.gate_of(0, i)) for i in range(n_items)]
    print(f"  全部 {n_items} 个目标物品上的门控极差均值={sum(spread) / len(spread):.3f}"
          f"（>0 说明门控确实在按目标区分专家）")

    uid = 0
    cands = [i for i in range(n_items) if (uid, i) not in pos_set]
    print(f"\n[4] 用户 {uid} 粗排 Top-5（候选 {len(cands)} 个）")
    for iid, s in model.pre_rank(uid, cands, top_k=5):
        print(f"  item {iid:2d}（簇{cluster[iid]}）: {s:.4f}  门控="
              f"{[round(v, 3) for v in model.gate_of(uid, iid)]}")
