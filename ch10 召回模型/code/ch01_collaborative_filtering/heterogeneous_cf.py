"""
Heterogeneous CF 召回 —— 异构行为协同过滤 / EHCF（第10章 §10.1.7）

核心思想：用户反馈是异构的（点击 -> 点赞 -> 收藏 -> 完播，漏斗逐级稀疏）。传统 CF 只用单一
    行为建模，既浪费监督信号又高度依赖负采样。Heterogeneous CF 将多种行为统一建模，并通过
    跨行为知识迁移（Knowledge Transfer）把高频行为学到的知识迁移给低频行为，从而在无需负
    采样的情况下学到更准确的用户/物品 Embedding。
参考论文：Chen et al., "Efficient Heterogeneous Collaborative Filtering without Negative
    Sampling for Recommendation", AAAI 2020（EHCF）
一、HeteroItemCF（统计派）：按行为类型构造加权共现矩阵，再做多行为融合；召回支持
    「先融合相似度再召回」与「分行为召回后加权融合」两种模式。
        C^(k)(i,j) = Σ_u (α_k·x^(k)_ui)(α_k·x^(k)_uj) / log(1+|N_k(u)|)
        S^(k)(i,j) = C^(k)(i,j) / sqrt( W_k(i)·W_k(j) )，  S(i,j) = Σ_k λ_k·S^(k)(i,j)
二、EHCFModel（模型派，PyTorch 实现，autograd 自动求导）：Embedding + Transfer-based
    Prediction Layer，无负采样的加权平方回归损失（借鉴 WMF），并按 λ_k 做多任务联合优化。
    正文中 EHCF 把 R̂^2 展开以解耦用户侧/物品侧二阶统计量来降低复杂度；本文件数据规模很小，
    直接遍历全物品集即可。
        z_uv = p_u ⊙ q_v（Hadamard 积），R̂^(k)_uv = h_k^T z_uv（第 k 个行为的线性预测层）
        h_k  = Σ_{t∈N(k)} ( h_t M_tk + r_tk )（跨行为知识迁移）
        L_k = Σ_{u∈B} Σ_{v∈V} c^(k)_uv ( R^(k)_uv - R̂^(k)_uv )^2，  L = Σ_k λ_k L_k
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

BEHAVIOR_ALPHA = {"click": 1.0, "like": 3.0, "collect": 5.0, "finish": 8.0}       # 置信度权重 α_k
BEHAVIOR_LAMBDA = {"click": 0.2, "like": 0.25, "collect": 0.25, "finish": 0.3}    # 融合权重 λ_k
BEHAVIOR_CHAIN = ["click", "like", "collect", "finish"]                           # 行为漏斗顺序


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(_dot(a, a)), math.sqrt(_dot(b, b))
    return _dot(a, b) / (na * nb) if na > 0 and nb > 0 else 0.0


# ══ 第一部分：加权共现矩阵的异构 ItemCF ══════════════════════════════════════
class HeteroItemCF:
    """按行为类型分别构造加权共现矩阵，再做多行为融合的异构协同过滤（纯统计，无待学习参数）。"""

    def __init__(self, alpha: dict[str, float] | None = None, lam: dict[str, float] | None = None):
        self.alpha = alpha or dict(BEHAVIOR_ALPHA)
        self.lam = lam or dict(BEHAVIOR_LAMBDA)
        self.sim_k: dict[str, dict[int, dict[int, float]]] = {}   # S^(k)
        self.sim_fused: dict[int, dict[int, float]] = defaultdict(dict)
        self.hist: dict[str, dict[int, dict[int, float]]] = {}    # 行为 -> 用户 -> {物品: α_k}

    def fit(self, logs: list[tuple[int, int, str]]) -> None:
        """从异构行为日志 [(user, item, behavior), ...] 构建加权共现相似度。"""
        for uid, iid, beh in logs:
            if beh in self.alpha:
                self.hist.setdefault(beh, defaultdict(dict))[uid][iid] = self.alpha[beh]

        for beh, user_map in self.hist.items():
            co: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
            w_sum: dict[int, float] = defaultdict(float)
            for item_w in user_map.values():
                # IUF 惩罚：行为过多的活跃用户对共现的贡献打折
                iuf = 1.0 / math.log(1.0 + len(item_w)) if len(item_w) > 1 else 1.0
                for i, wi in item_w.items():
                    w_sum[i] += wi * wi
                    for j, wj in item_w.items():
                        if i != j:
                            co[i][j] += wi * wj * iuf
            sim: dict[int, dict[int, float]] = defaultdict(dict)
            lam = self.lam.get(beh, 0.0)
            for i, row in co.items():
                for j, c in row.items():
                    denom = math.sqrt(w_sum[i] * w_sum[j])
                    sim[i][j] = c / denom if denom > 0 else 0.0
                    # 融合：S(i,j) = Σ_k λ_k S^(k)(i,j)
                    self.sim_fused[i][j] = self.sim_fused[i].get(j, 0.0) + lam * sim[i][j]
            self.sim_k[beh] = sim

    def similar_items(self, item: int, behavior: str | None = None, top_k: int = 5) -> list[tuple[int, float]]:
        """查询相似物品；behavior=None 表示使用多行为融合相似度 S(i,j)。"""
        table = self.sim_fused if behavior is None else self.sim_k.get(behavior, {})
        return sorted(table.get(item, {}).items(), key=lambda x: (-x[1], x[0]))[:top_k]

    def recall(self, uid: int, top_n: int = 5, mode: str = "fused") -> list[tuple[int, float]]:
        """异构召回两种模式：mode="fused" 先融合得到 S(i,j) 再用全部行为历史（取最高置信度
        作触发权重）召回；mode="per_behavior" 按行为分别召回再按 λ_k 加权融合（更常用）。"""
        seen = {i for users in self.hist.values() for i in users.get(uid, {})}
        scores: dict[int, float] = defaultdict(float)
        if mode == "fused":
            trigger: dict[int, float] = {}
            for users in self.hist.values():
                for i, w in users.get(uid, {}).items():
                    trigger[i] = max(trigger.get(i, 0.0), w)
            for i, w in trigger.items():
                for j, s in self.similar_items(i, None, 10):
                    if j not in seen:
                        scores[j] += w * s
        else:
            for beh, users in self.hist.items():
                for i in users.get(uid, {}):
                    for j, s in self.similar_items(i, beh, 10):
                        if j not in seen:
                            scores[j] += self.lam.get(beh, 0.0) * s
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]


# ══ 第二部分：EHCF（Embedding + Transfer-based Prediction Layer）═════════════
class EHCFModel(nn.Module):
    """多行为联合建模 + 跨行为知识迁移 + 无负采样的加权平方回归优化（PyTorch autograd）。"""

    def __init__(self, n_users: int, n_items: int, dim: int = 16,
                 reg: float = 1e-4, c_pos: float = 1.0, c_neg: float = 0.02):
        super().__init__()
        self.dim = dim
        self.reg = reg
        self.c_pos, self.c_neg = c_pos, c_neg          # 置信权重 c^+ / c^-
        self.behaviors, self.lam = list(BEHAVIOR_CHAIN), dict(BEHAVIOR_LAMBDA)

        self.p = nn.Embedding(n_users, dim)  # 用户 Embedding
        self.q = nn.Embedding(n_items, dim)  # 物品 Embedding
        nn.init.xavier_normal_(self.p.weight)
        nn.init.xavier_normal_(self.q.weight)

        # N(k) = {k 的上游行为, k 本身}：上游行为更稠密，知识沿漏斗向下迁移
        self.parents = {k: ([self.behaviors[i - 1]] if i > 0 else []) + [k]
                        for i, k in enumerate(self.behaviors)}
        self.h_base = nn.ParameterDict({
            k: nn.Parameter(torch.randn(dim) * 0.1) for k in self.behaviors
        })
        # 迁移矩阵 M_tk（近似单位矩阵初始化，加小噪声）与迁移偏置 r_tk
        self.m = nn.ParameterDict()
        self.r = nn.ParameterDict()
        for k in self.behaviors:
            for t in self.parents[k]:
                init = torch.eye(dim) + torch.randn(dim, dim) * 0.01
                self.m[f"{t}|{k}"] = nn.Parameter(init)
                self.r[f"{t}|{k}"] = nn.Parameter(torch.zeros(dim))

    def effective_h(self, k: str) -> torch.Tensor:
        """h_k = (1/|N(k)|) Σ_{t∈N(k)} ( M_tk^T h_t + r_tk )，实现跨行为知识迁移。"""
        parents = self.parents[k]
        out = torch.zeros(self.dim)
        for t in parents:
            ht = self.h_base[t]
            mat = self.m[f"{t}|{k}"]
            rr = self.r[f"{t}|{k}"]
            out = out + ht @ mat + rr
        return out / len(parents)

    def predict(self, uid, iid, behavior: str, h: torch.Tensor | None = None):
        """R̂^(k)_uv = h_k^T (p_u ⊙ q_v)。"""
        h = h if h is not None else self.effective_h(behavior)
        pu = self.p(uid)   # (B, dim)
        qv = self.q(iid)   # (B, dim)
        return (h * pu * qv).sum(dim=-1)

    def train_model(self, label: dict[str, dict[int, set[int]]], n_items: int, epochs: int = 6,
                    lr: float = 0.02) -> None:
        """无负采样训练：对每个用户遍历全部物品做加权平方回归（正样本 c^+，未观测 c^-）。
        使用 PyTorch autograd，梯度由反向传播自动完成。"""
        optimizer = optim.Adam(self.parameters(), lr=lr)

        all_items = torch.arange(n_items, dtype=torch.long)  # 全部物品 ID
        y_all = torch.zeros(n_items)
        for epoch in range(epochs):
            total, cnt = 0.0, 0
            order = list(range(self.p.weight.shape[0]))
            random.shuffle(order)

            optimizer.zero_grad()
            loss_sum = None
            for uid in order:
                u_t = torch.tensor([uid], dtype=torch.long)
                for k in self.behaviors:
                    pos = label.get(k, {}).get(uid, set())
                    h = self.effective_h(k)
                    lam = self.lam.get(k, 1.0)

                    # R：正样本为 1，未观测为 0；c：正样本 c^+，未观测 c^-
                    y = y_all.clone()
                    if pos:
                        y[list(pos)] = 1.0
                    c = torch.where(y > 0, torch.full_like(y, self.c_pos),
                                    torch.full_like(y, self.c_neg))

                    pred = self.predict(u_t, all_items, k, h)  # (n_items,)
                    err = pred - y
                    loss_k = (lam * c * err * err).sum()       # L_k = Σ_v λ_k c_v (R̂ - R)^2
                    loss_sum = loss_k if loss_sum is None else loss_sum + loss_k
                    total += loss_k.item()
                    cnt += n_items

            loss_sum.backward()   # 迁移层 / Embedding 梯度均由 autograd 自动求导
            optimizer.step()
            print(f"  epoch {epoch}  加权平方损失/样本 = {total / cnt:.6f}")

    def recommend(self, uid: int, n_items: int, exclude: set[int], top_k: int = 5,
                  behavior: str | None = None) -> list[tuple[int, float]]:
        """behavior=None 时按 λ_k 加权融合各行为的预测得分。"""
        hs = {k: self.effective_h(k) for k in self.behaviors}
        all_items = torch.arange(n_items, dtype=torch.long)
        u_t = torch.tensor([uid] * n_items, dtype=torch.long)
        with torch.no_grad():
            if behavior is None:
                s = sum(self.lam[k] * self.predict(u_t, all_items, k, hs[k]) for k in self.behaviors)
            else:
                s = self.predict(u_t, all_items, behavior, hs[behavior])
            scores = s.tolist()
        out = [(iid, scores[iid]) for iid in range(n_items) if iid not in exclude]
        return sorted(out, key=lambda x: -x[1])[:top_k]


def _fmt(pairs: list[tuple[int, float]]) -> str:
    return "  ".join(f"item{j:02d}({s:+.4f})" for j, s in pairs) or "(无结果)"


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, dim = 8, 20, 16
    # 合成异构行为日志：模拟「点击 -> 点赞 -> 收藏 -> 完播」的漏斗衰减
    funnel = {"like": 0.55, "collect": 0.35, "finish": 0.2}
    logs: list[tuple[int, int, str]] = []
    label: dict[str, dict[int, set[int]]] = {k: defaultdict(set) for k in BEHAVIOR_CHAIN}
    for uid in range(n_users):
        for iid in random.sample(range(n_items), 8):          # 每个用户的偏好物品子集
            if random.random() >= 0.85:
                continue
            logs.append((uid, iid, "click"))
            label["click"][uid].add(iid)
            for beh, rate in funnel.items():
                if random.random() < rate:
                    logs.append((uid, iid, beh))
                    label[beh][uid].add(iid)
    stat = {k: sum(1 for _, _, b in logs if b == k) for k in BEHAVIOR_CHAIN}
    print("异构行为日志统计（漏斗越深越稀疏）：\n  " + "  ".join(
        f"{k}={stat[k]}条(α={BEHAVIOR_ALPHA[k]:.0f},λ={BEHAVIOR_LAMBDA[k]:.2f})" for k in BEHAVIOR_CHAIN))

    print("\n【第一部分】加权共现矩阵 + 多行为融合相似度")
    hicf = HeteroItemCF()
    hicf.fit(logs)
    for beh in BEHAVIOR_CHAIN:
        print(f"  item00 在 {beh:8s} 行为下的 Top-3：{_fmt(hicf.similar_items(0, beh, 3))}")
    print(f"  item00 多行为融合后的 Top-3：{_fmt(hicf.similar_items(0, None, 3))}")
    print("\n  两种召回模式对比（用户 0）：")
    print(f"    先融合相似度再召回：{_fmt(hicf.recall(0, 5, 'fused'))}")
    print(f"    分行为召回后融合  ：{_fmt(hicf.recall(0, 5, 'per_behavior'))}")

    print("\n【第二部分】EHCF 多行为联合训练（无负采样 + Transfer-based Prediction Layer）")
    model = EHCFModel(n_users, n_items, dim=dim, reg=1e-4, c_pos=1.0, c_neg=0.02)
    model.train_model(label, n_items, epochs=6, lr=0.02)
    with torch.no_grad():
        hs = {k: model.effective_h(k) for k in BEHAVIOR_CHAIN}
    hs_np = {k: v.tolist() for k, v in hs.items()}
    print("\n  相邻行为预测头 h_k 的余弦相似度（迁移矩阵使其参数彼此靠近）：\n    " + "  ".join(
        f"cos(h_{a},h_{b})={_cosine(hs_np[a], hs_np[b]):+.4f}" for a, b in zip(BEHAVIOR_CHAIN, BEHAVIOR_CHAIN[1:])))

    seen = {iid for u, iid, _ in logs if u == 0}
    print("\n  用户 0 分行为 Top-3 召回：")
    for beh in BEHAVIOR_CHAIN:
        print(f"    {beh:8s}: {_fmt(model.recommend(0, n_items, seen, 3, beh))}")
    print(f"\n  用户 0 多行为加权融合 Top-5 召回：{_fmt(model.recommend(0, n_items, seen, 5, None))}")
