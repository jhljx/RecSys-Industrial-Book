"""
IntTower 粗排模型（第11章 §11.4.4）

IntTower = Interaction enhanced Two-Tower
论文出处：Li et al., "IntTower: the Next Generation of Two-Tower Model for Pre-Ranking
          System", CIKM 2022（华为诺亚方舟实验室）。

精度-时延权衡（对应书中 §11.4.4）：朴素双塔是 Late Interaction——两侧各自编码到最后一层
才做一次内积，细粒度交互信息被压缩殆尽，精度受限；精排式交互 DNN 精度高却无法预计算物品
向量，时延扛不住。IntTower 在「物品侧表示仍可离线预计算」的效率前提下，把交互提前到塔内部
（Early Interaction），用几乎无参数的轻量算子换取接近交互式模型的精度。三个子模块：
    1) LightSE：轻量 SENET。squeeze 用 Mean Pooling 压成标量 z_i = (1/d)·Σ_t e_{i,t}；
       excitation 只用一层 FC + Softmax，k = Softmax(W·z + b)；再重加权 ẽ_i = k_i·e_i。
    2) FEBlock：用户塔第 i 层投影到 H 个子空间 m_u^{i,h} = W_u^{i,h}·h_u^i + b，物品塔只投影
       最后一层 m_v^{L,h}；S^i = Σ_{hu} max_{hv} ⟨m_u^{i,hu}, m_v^{L,hv}⟩/√d_h，ŷ = Σ_i S^i。
       交互算子本身无参数，物品侧只需缓存最后一层多头表示 → 可预计算。
    3) CIR：对比交互正则（Contrastive Interaction Regularization）。在最后一层多头表示
       M_u^L / M_v^L 上做 InfoNCE，同批次其他物品为负样本，温度 τ。
训练目标：L = L_ctr + λ1·L_cir + λ2·‖Θ‖²（λ2 以逐参数权重衰减实现）。

PyTorch 实现（autograd 自动微分），可直接 `python3 inttower.py` 运行。
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


class LightSE(nn.Module):
    """Light-SE 模块：Mean-Pooling squeeze → 单层 FC → Softmax excitation → 逐字段重加权。"""

    def __init__(self, m: int):
        super().__init__()
        self.m = m
        self.fc = nn.Linear(m, m)

    def forward(self, fields: List[torch.Tensor]):
        # fields: list of m tensors, each shape (field_dim,)
        z = torch.stack([f.mean() for f in fields])        # (m,)
        lg = self.fc(z)                                    # (m,)
        k = torch.softmax(lg, dim=0)                       # (m,)
        out = [k[i] * fields[i] for i in range(self.m)]
        return out, k


class FEBlock(nn.Module):
    """FE-Block：用户塔「每一层」表示 × 物品塔「最后一层」表示的多头 Max-Similarity 早期交互。"""

    def __init__(self, user_dims: List[int], item_dim: int, n_heads: int = 4, d_head: int = 8):
        super().__init__()
        self.L = len(user_dims)
        self.H = n_heads
        self.d = d_head
        self.scale = math.sqrt(d_head)
        # user_proj[i][h]: project user layer-i rep to head h
        self.user_proj = nn.ModuleList([
            nn.ModuleList([nn.Linear(du, d_head) for _ in range(n_heads)])
            for du in user_dims
        ])
        self.item_proj = nn.ModuleList([nn.Linear(item_dim, d_head) for _ in range(n_heads)])

    def forward(self, user_hs: List[torch.Tensor], item_h: torch.Tensor):
        """
        user_hs: list of L tensors, item_h: shape (item_dim,)
        Returns scalar score (sum of MaxSim over all layers and heads).
        Also returns last-layer multi-head reps for CIR.
        """
        # Item projections (only last layer needed, but computed once)
        m_v = [self.item_proj[h](item_h) for h in range(self.H)]  # list of H tensors (d_head,)

        total = torch.tensor(0.0)
        for i in range(self.L):
            for hu in range(self.H):
                m_u = self.user_proj[i][hu](user_hs[i])            # (d_head,)
                # MaxSim over item heads
                sims = torch.stack([torch.dot(m_u, m_v[hv]) for hv in range(self.H)])
                total = total + sims.max() / self.scale
        return total, m_v

    def last_layer_reps(self, user_hs: List[torch.Tensor], item_h: torch.Tensor):
        """M_u^L / M_v^L 拼接，作为 CIR 模块输入。"""
        m_u_L = torch.cat([self.user_proj[self.L - 1][h](user_hs[-1]) for h in range(self.H)])
        m_v_L = torch.cat([self.item_proj[h](item_h) for h in range(self.H)])
        return m_u_L, m_v_L


class IntTower(nn.Module):
    """IntTower 粗排模型：Light-SE + 双塔 DNN + FE-Block 早期交互 + CIR 对比正则。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, n_users: int, n_items: int, n_fields: int = 4, field_dim: int = 4,
                 hidden: int = 24, out_dim: int = 16, n_heads: int = 4, d_head: int = 8,
                 lr: float = 0.05, reg: float = 1e-5, lam_cir: float = 0.1, tau: float = 0.2,
                 emb_scale: float = 2.0):
        super().__init__()
        self.n_items = n_items
        self.m = n_fields
        self.fd = field_dim
        self.lam_cir = lam_cir
        self.tau = tau

        in_dim = n_fields * field_dim

        # Embeddings: each user/item has n_fields × field_dim parameters
        self.user_emb = nn.Parameter(
            torch.empty(n_users, n_fields, field_dim).uniform_(-emb_scale, emb_scale))
        self.item_emb = nn.Parameter(
            torch.empty(n_items, n_fields, field_dim).uniform_(-emb_scale, emb_scale))

        self.user_se = LightSE(n_fields)
        self.item_se = LightSE(n_fields)

        self.user_l1 = nn.Linear(in_dim, hidden)
        self.user_l2 = nn.Linear(hidden, out_dim)
        self.item_l1 = nn.Linear(in_dim, hidden)
        self.item_l2 = nn.Linear(hidden, out_dim)

        self.user_dims = [hidden, out_dim]
        self.fe = FEBlock(self.user_dims, out_dim, n_heads, d_head)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def _tower(self, fields_param: torch.Tensor, se: LightSE,
               l1: nn.Linear, l2: nn.Linear):
        """fields_param: (n_fields, field_dim); returns (list of layer outputs, LightSE k)"""
        fields = [fields_param[i] for i in range(self.m)]
        gated, k = se(fields)
        h = torch.cat(gated)                    # (n_fields * field_dim,)
        h1 = F.relu(l1(h))
        h2 = F.relu(l2(h1))
        return [h1, h2], k

    def forward(self, uid: int, iid: int):
        u_hs, u_k = self._tower(self.user_emb[uid], self.user_se, self.user_l1, self.user_l2)
        i_hs, i_k = self._tower(self.item_emb[iid], self.item_se, self.item_l1, self.item_l2)
        y, m_v = self.fe(u_hs, i_hs[-1])
        return y, u_hs, i_hs, m_v, u_k, i_k

    def score(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            y, *_ = self.forward(uid, iid)
            return float(torch.sigmoid(y).item())

    def train_model(self, pos_pairs, epochs: int = 6, batch_size: int = 8) -> None:
        data, pos_set = list(pos_pairs), set(pos_pairs)
        for epoch in range(epochs):
            random.shuffle(data)
            l_ctr = l_cir = 0.0
            nb = 0
            for st in range(0, len(data), batch_size):
                batch = data[st:st + batch_size]
                if len(batch) < 2:
                    continue
                self.optimizer.zero_grad()
                loss_ctr = torch.tensor(0.0)
                mu_list, mv_list = [], []
                for uid, iid in batch:
                    y, u_hs, i_hs, m_v, _, _ = self.forward(uid, iid)
                    loss_ctr = loss_ctr + F.binary_cross_entropy_with_logits(
                        y, torch.tensor(1.0))
                    # Negative sample
                    nid = random.randrange(self.n_items)
                    while (uid, nid) in pos_set:
                        nid = random.randrange(self.n_items)
                    yn, u_hs_n, i_hs_n, _, _, _ = self.forward(uid, nid)
                    loss_ctr = loss_ctr + F.binary_cross_entropy_with_logits(
                        yn, torch.tensor(0.0))
                    # Collect last-layer multi-head reps for CIR
                    mu, mv = self.fe.last_layer_reps(u_hs, i_hs[-1])
                    mu_list.append(mu)
                    mv_list.append(mv)

                # CIR loss: InfoNCE over batch
                B = len(mu_list)
                mu_stack = torch.stack(mu_list)   # (B, H*d_head)
                mv_stack = torch.stack(mv_list)   # (B, H*d_head)
                mu_norm = F.normalize(mu_stack, dim=1)
                mv_norm = F.normalize(mv_stack, dim=1)
                sim_matrix = torch.mm(mu_norm, mv_norm.t()) / self.tau  # (B, B)
                labels = torch.arange(B)
                loss_cir = F.cross_entropy(sim_matrix, labels)

                loss = loss_ctr / (len(batch) * 2) + self.lam_cir * loss_cir
                loss.backward()
                self.optimizer.step()
                l_ctr += (loss_ctr / (len(batch) * 2)).item()
                l_cir += loss_cir.item()
                nb += 1

            l_ctr /= max(1, nb)
            l_cir /= max(1, nb)
            print(f"  epoch {epoch}  L_ctr={l_ctr:.4f}  L_cir={l_cir:.4f}  "
                  f"L_total={l_ctr + self.lam_cir * l_cir:.4f}")

    def pre_rank(self, uid: int, candidates: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """粗排：用户侧实时前向各层表示，物品侧多头表示可预计算，FE-Block 聚合打分取 Top-K。"""
        scored = [(iid, self.score(uid, iid)) for iid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, n_fields, fd, epochs = 8, 24, 4, 4, 8
    lat_u = [[random.gauss(0, 1) for _ in range(4)] for _ in range(n_users)]
    lat_i = [[random.gauss(0, 1) for _ in range(4)] for _ in range(n_items)]
    pos_pairs = [(u, i) for u in range(n_users) for i in range(n_items)
                 if _dot_py(lat_u[u], lat_i[i]) / 2.0 > 0.5]
    pos_set = set(pos_pairs)
    model = IntTower(n_users, n_items, n_fields, fd, lam_cir=0.1, tau=0.2)
    print(f"IntTower 粗排：n_users={n_users}  n_items={n_items}  正样本对={len(pos_pairs)}")

    print("\n[1] 各模块输出维度流转（结构自洽性检查）")
    with torch.no_grad():
        y, u_hs, i_hs, m_v, u_k, i_k = model.forward(0, 0)
        mu_L, mv_L = model.fe.last_layer_reps(u_hs, i_hs[-1])
    print(f"  输入字段 Embedding ：{n_fields} 字段 × {fd} 维 = {n_fields * fd}")
    k_sum = float(u_k.sum().item())
    k_list = [round(float(v), 3) for v in u_k.tolist()]
    print(f"  Light-SE 重加权    ：{n_fields} × {fd}（门控 k 和={k_sum:.3f}，k={k_list}）")
    print(f"  用户塔 / 物品塔    ：{n_fields * fd} → {model.user_dims[0]} → {model.user_dims[1]}")
    print(f"  FE-Block 多头投影  ：用户 {model.fe.L} 层 × {model.fe.H} 头 × d_h={model.fe.d}，物品仅最后一层")
    print(f"  M_u^L / M_v^L      ：{len(mu_L)} / {len(mv_L)} 维（CIR 输入）")
    print(f"  输出 ŷ = Σ_i S^i   ：标量 = {float(y):.4f}")
    print(f"\n[2] 联合训练 L = L_ctr + λ1·L_cir + λ2·‖Θ‖²（λ1={model.lam_cir}，τ={model.tau}）")
    model.train_model(pos_pairs, epochs=epochs, batch_size=8)
    auc = _auc([(model.score(u, i), 1 if (u, i) in pos_set else 0)
                for u in range(n_users) for i in range(n_items)])
    print(f"  训练集 AUC = {auc:.4f}（早期交互使 ŷ 能区分正负样本，而物品侧仍可预计算）")
    uid = 0
    cands = [i for i in range(n_items) if (uid, i) not in pos_set]
    print(f"\n[3] 用户 {uid} 粗排 Top-5（候选 {len(cands)} 个，排除已交互物品）")
    for iid, s in model.pre_rank(uid, cands, top_k=5):
        print(f"  item {iid:2d}: {s:.4f}")
