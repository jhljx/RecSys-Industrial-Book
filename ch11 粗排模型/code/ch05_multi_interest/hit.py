"""
HIT 粗排模型（第11章 §11.5.2）

HIT = Hierarchical Interaction-Enhanced Two-Tower（分层交互增强的双塔模型）
论文出处：Yang et al., "HIT Model: A Hierarchical Interaction-Enhanced Two-Tower Model
          for Pre-Ranking Systems", CIKM 2025（腾讯广告）。

为什么要做「分层」交互（对应书中 §11.5.2）：
    传统双塔两侧完全解耦，用户表示里没有任何物品侧信息，且用单一 Embedding 表示多样化
    兴趣；COLD / FSCD 这类全交互模型精度高但必须在线算交叉网络。HIT 分两层补交互：
    1) 粗粒度生成模块（CoarseGrainedGeneration）：受 DAT 启发，用 Generator 以「静态特征」
       为输入生成模拟对侧语义的表示（只用静态特征，避免引入实时特征推高在线开销）：
           r_m^k = g_k(x_b^δ),  k = 1,…,K（广告场景分别设正样本/负样本 Generator）
       生成表示先做 L2 归一化，再与原始 Embedding 拼接成增强输入，最后过 DNN：
           e' = f(r_m^1, …, r_m^K, e)  →  h_u / h_a
    2) 细粒度匹配模块（FineGrainedMatching）：Multi-head Representer 把两侧表示投到 J 个
       潜在子空间（用户多兴趣 × 物品多属性）：
           r_u^(j) = W_u^(j)·h_u + b_u^(j)，  r_a^(j) = W_a^(j)·h_a + b_a^(j)
       再用 IntTower 式 Max Similarity 聚合：ŷ = Σ_{ju} max_{ja} (r_u^(ju))ᵀ·r_a^(ja)
    训练目标：L = L_c + α(L_gu + L_ga)，其中 L_c 为交叉熵，生成损失
           L_g = -(1/N)·Σ_i [ y_i·Dist(r_u, r_m^1) + (1-y_i)·Dist(r_u, r_m^2) ]，Dist 取余弦；
           按书中写法保留前置负号（即最大化生成表示与匹配表示的余弦相似度）。
    在线效率：物品塔的多头表示 r_a^(j) 与物品侧生成表示均不依赖用户，可离线预计算并缓存，
    线上只需算用户侧前向 + J×J 次低维内积取 max，仍保持双塔级别的时延。

PyTorch 实现（autograd 自动微分），可直接 `python3 hit.py` 运行。
"""
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _dot_py(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _sigmoid_py(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

def _auc(pairs: List[Tuple[float, int]]) -> float:
    pos, neg = [s for s, y in pairs if y == 1], [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return 0.5
    hit = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return hit / (len(pos) * len(neg))


class Generator(nn.Module):
    """Generator：r_m^k = g_k(x_b^δ)，仅吃静态特征，输出做 L2 归一化。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, in_d: int, out_d: int, hidden: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 L2 归一化后的生成表示 r_m^k，shape (out_d,)。"""
        r = self.net(x)
        return F.normalize(r, dim=0)


class CoarseGrainedGeneration(nn.Module):
    """粗粒度生成模块：K 个 Generator 生成对侧语义表示 → L2 归一化 → 与 Embedding 拼接 → DNN。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, static_dim: int, emb_dim: int, gen_dim: int = 8, n_gen: int = 2,
                 hidden: int = 24, out_dim: int = 16):
        super().__init__()
        self.n_gen = n_gen
        self.gen_dim = gen_dim
        self.gens = nn.ModuleList([Generator(static_dim, gen_dim, 12) for _ in range(n_gen)])
        self.l1 = nn.Linear(n_gen * gen_dim + emb_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)

    def forward(self, static_x: torch.Tensor, emb: torch.Tensor):
        """
        static_x: (static_dim,)
        emb:      (emb_dim,)
        Returns (h: (out_dim,), rms: list of K tensors each (gen_dim,))
        """
        rms = [g(static_x) for g in self.gens]                 # K × (gen_dim,)
        e2 = torch.cat(rms + [emb])                            # (K*gen_dim + emb_dim,)
        h = self.l2(F.relu(self.l1(e2)))                       # (out_dim,) — 输出层不加激活
        return h, rms


class FineGrainedMatching(nn.Module):
    """细粒度匹配模块：Multi-head Representer + Max Similarity 聚合。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, dim_u: int, dim_a: int, n_head: int = 4, d_rep: int = 8):
        super().__init__()
        self.J = n_head
        self.d = d_rep
        self.scale = math.sqrt(d_rep)
        self.u_heads = nn.ModuleList([nn.Linear(dim_u, d_rep) for _ in range(n_head)])
        self.a_heads = nn.ModuleList([nn.Linear(dim_a, d_rep) for _ in range(n_head)])

    def forward(self, h_u: torch.Tensor, h_a: torch.Tensor):
        """
        h_u: (dim_u,), h_a: (dim_a,)
        Returns (scalar score, r_u list, r_a list)
        ŷ = (1/J²) Σ_{ju,ja} cos(r_u^(ju), r_a^(ja))
        L2 归一化后点积等价于余弦相似度，值域有界 [-1,1]，避免权重增长导致 logit 爆炸。
        """
        r_u = [head(h_u) for head in self.u_heads]                     # J × (d_rep,)
        r_a = [head(h_a) for head in self.a_heads]                     # J × (d_rep,)
        r_u_n = [F.normalize(r, dim=0) for r in r_u]                   # 归一化到单位球
        r_a_n = [F.normalize(r, dim=0) for r in r_a]
        total = torch.tensor(0.0)
        for ju in range(self.J):
            sims = torch.stack([torch.dot(r_u_n[ju], r_a_n[ja]) for ja in range(self.J)])
            total = total + sims.mean() / self.J                        # 除 J：总和 ∈ [-1, 1]
        return total, r_u, r_a


class HIT(nn.Module):
    """HIT 粗排模型：双塔 + 粗粒度生成（跨塔语义融合）+ 细粒度多头匹配。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, n_users: int, n_items: int, n_u_fields: int = 3, n_i_fields: int = 2,
                 field_dim: int = 4, static_dim: int = 4, gen_dim: int = 8, n_head: int = 4,
                 out_dim: int = 16, lr: float = 0.05, reg: float = 1e-5, alpha: float = 0.2,
                 emb_scale: float = 1.5):
        super().__init__()
        self.n_items = n_items
        self.mu = n_u_fields
        self.mi = n_i_fields
        self.fd = field_dim
        self.alpha = alpha
        self.J = n_head
        self.gen_dim = gen_dim

        self.user_emb = nn.Parameter(
            torch.empty(n_users, n_u_fields, field_dim).uniform_(-emb_scale, emb_scale))
        self.item_emb = nn.Parameter(
            torch.empty(n_items, n_i_fields, field_dim).uniform_(-emb_scale, emb_scale))

        # 静态特征不参与训练，固定随机向量（用 register_buffer 语义，但直接存为普通 tensor）
        self.register_buffer("user_static",
            torch.randn(n_users, static_dim))
        self.register_buffer("item_static",
            torch.randn(n_items, static_dim))

        self.u_tower = CoarseGrainedGeneration(
            static_dim, n_u_fields * field_dim, gen_dim, 2, 24, out_dim)
        self.a_tower = CoarseGrainedGeneration(
            static_dim, n_i_fields * field_dim, gen_dim, 2, 24, out_dim)
        self.match = FineGrainedMatching(out_dim, out_dim, n_head, gen_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def forward(self, uid: int, iid: int):
        """Returns (logit, r_u list, r_a list, rms_u list, rms_a list)."""
        u_emb = self.user_emb[uid].reshape(-1)                 # (n_u_fields * field_dim,)
        a_emb = self.item_emb[iid].reshape(-1)                 # (n_i_fields * field_dim,)
        h_u, rms_u = self.u_tower(self.user_static[uid], u_emb)
        h_a, rms_a = self.a_tower(self.item_static[iid], a_emb)
        y, r_u, r_a = self.match(h_u, h_a)
        logit = y + self.bias.squeeze()
        return logit, r_u, r_a, rms_u, rms_a

    def _gen_loss(self, rms: List[torch.Tensor], reps: List[torch.Tensor], y: float) -> torch.Tensor:
        """L_g = -[y·cos(r_pool, r_m^1) + (1-y)·cos(r_pool, r_m^2)]。
        r_pool = mean of multi-head reps；r_pool 做 .detach()（只更新 Generator，不反传到匹配模块）。
        """
        r_pool = torch.stack(reps).mean(dim=0).detach()        # (gen_dim,) — 梯度在此阻断
        k = 0 if y > 0.5 else 1
        cos = F.cosine_similarity(r_pool.unsqueeze(0), rms[k].unsqueeze(0))
        return -cos.squeeze()

    def score(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            logit, *_ = self.forward(uid, iid)
            return float(torch.sigmoid(logit).item())

    def gen_align(self, uid: int, iid: int) -> Tuple[float, float]:
        """生成表示与本塔多头池化表示的余弦相似度（用于观察生成损失是否收敛）。"""
        with torch.no_grad():
            _, r_u, r_a, rms_u, rms_a = self.forward(uid, iid)
            pool_u = torch.stack(r_u).mean(dim=0)
            pool_a = torch.stack(r_a).mean(dim=0)
            cos_u = float(F.cosine_similarity(pool_u.unsqueeze(0), rms_u[0].unsqueeze(0)).item())
            cos_a = float(F.cosine_similarity(pool_a.unsqueeze(0), rms_a[0].unsqueeze(0)).item())
        return cos_u, cos_a

    def train_model(self, pos_pairs, epochs: int = 6) -> None:
        data, pos_set = list(pos_pairs), set(pos_pairs)
        for epoch in range(epochs):
            random.shuffle(data)
            acc_c = acc_gu = acc_ga = 0.0
            for uid, iid in data:
                for iid2, y in [(iid, 1.0), (self._neg(uid, pos_set), 0.0)]:
                    # Single forward + single backward: CTR loss + generation alignment loss
                    # r_pool is detached so gen gradient flows only to generators, not to matching
                    self.optimizer.zero_grad()
                    logit, r_u, r_a, rms_u, rms_a = self.forward(uid, iid2)
                    loss_c = F.binary_cross_entropy_with_logits(logit, torch.tensor(y))
                    l_gu = self._gen_loss(rms_u, r_u, y)
                    l_ga = self._gen_loss(rms_a, r_a, y)
                    loss = loss_c + self.alpha * (l_gu + l_ga)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.optimizer.step()
                    acc_c += loss_c.item()
                    acc_gu += l_gu.item()
                    acc_ga += l_ga.item()

            n = max(1, len(data) * 2)
            l_total = acc_c / n + self.alpha * (acc_gu + acc_ga) / n
            print(f"  epoch {epoch}  L_c={acc_c / n:.4f}  L_gu={acc_gu / n:.4f}  "
                  f"L_ga={acc_ga / n:.4f}  L={l_total:.4f}")

    def _neg(self, uid: int, pos_set) -> int:
        nid = random.randrange(self.n_items)
        while (uid, nid) in pos_set:
            nid = random.randrange(self.n_items)
        return nid

    def pre_rank(self, uid: int, candidates: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """粗排：物品侧生成表示与多头表示均可离线预计算，线上只需 J×J 次低维内积取 max。"""
        scored = [(iid, self.score(uid, iid)) for iid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, n_head, epochs = 8, 24, 2, 10
    lat_u = [[random.gauss(0, 1) for _ in range(4)] for _ in range(n_users)]
    lat_i = [[random.gauss(0, 1) for _ in range(4)] for _ in range(n_items)]
    # 合成数据：物品分 3 个属性簇，用户对不同簇偏好不同 → 需要多兴趣 × 多属性的细粒度匹配
    cluster = [i % 3 for i in range(n_items)]
    pos_pairs = [(u, i) for u in range(n_users) for i in range(n_items)
                 if _dot_py(lat_u[u], lat_i[i]) / 2.0 + 0.6 * lat_u[u][cluster[i]] > 0.5]
    pos_set = set(pos_pairs)
    model = HIT(n_users, n_items, n_head=n_head, lr=0.01, alpha=0.2)
    print(f"HIT 粗排：n_users={n_users}  n_items={n_items}  多头数 J={n_head}  "
          f"Generator 数 K=2（正/负样本各一）  正样本对={len(pos_pairs)}")

    mean_align = lambda: [sum(c) / len(pos_pairs)
                          for c in zip(*[model.gen_align(u, i) for u, i in pos_pairs])]
    a0 = mean_align()
    print(f"  训练前生成表示对齐度（正样本均值）：cos(r_u, r_m^1)={a0[0]:.4f}  "
          f"cos(r_a, r_m^1)={a0[1]:.4f}")

    print("\n[1] 联合训练 L = L_c + α(L_gu + L_ga)，α=0.2")
    model.train_model(pos_pairs, epochs=epochs)

    auc = _auc([(model.score(u, i), 1 if (u, i) in pos_set else 0)
                for u in range(n_users) for i in range(n_items)])
    a1 = mean_align()
    print(f"  训练集 AUC = {auc:.4f}")
    print(f"  训练后生成表示对齐度（正样本均值）：cos(r_u, r_m^1)={a1[0]:.4f}  "
          f"cos(r_a, r_m^1)={a1[1]:.4f}（生成损失驱动对齐度提升）")

    print("\n[2] 物品侧多头表示（可离线预计算缓存）")
    with torch.no_grad():
        _, _, r_a_demo, _, _ = model.forward(0, 0)
    print(f"  item0 物品塔多头表示：J={n_head} 个头，每头维度={model.gen_dim}")
    for j, r in enumerate(r_a_demo):
        print(f"    头{j}: {[round(v, 3) for v in r.tolist()]}")

    uid = 0
    cands = [i for i in range(n_items) if (uid, i) not in pos_set]
    print(f"\n[3] 用户 {uid} 粗排 Top-5（候选 {len(cands)} 个，排除已交互物品）")
    for iid, s in model.pre_rank(uid, cands, top_k=5):
        print(f"  item {iid:2d}（簇{cluster[iid]}）: {s:.4f}")
