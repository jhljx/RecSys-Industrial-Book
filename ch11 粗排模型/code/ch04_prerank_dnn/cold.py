"""
COLD 粗排模型（第11章 §11.4.2）

COLD = Computing power cost-aware Online and Lightweight Deep pre-ranking system
论文出处：Wang et al., "COLD: Towards the Next Generation of Pre-Ranking System", 阿里巴巴 2020。

核心思想（对应书中 §11.4.2）：
    (1) 粗排不再被「必须用向量内积」的双塔结构绑死，而是允许直接使用表达能力更强的
        全连接 DNN（用户特征、物品特征、交叉特征统一 Embedding 后拼接 → MLP → CTR）；
    (2) 用「算力代价感知」的手段把复杂 DNN 的在线开销压下来，核心是基于 SE
        （Squeeze-and-Excitation）模块的特征组选择（Feature Group Selection）：
            squeeze    : z_i = mean(e_i)                        （每个特征组压成一个标量统计量）
            excitation : s   = σ(W2·ReLU(W1·z + b1) + b2)       （书中写作 s = σ(W[e_1..e_M] + b)）
            re-weight  : v_i = s_i · e_i                        （逐特征组缩放）
        随后按重要性 s_i 挑 Top-K 特征组，并结合 GAUC / QPS / RT 不断调整 K，
        在「模型效果」和「算力代价」之间找平衡点。

本文件在书中公式基础上补充了一张显式的「特征组算力代价表」，并实现了预算约束下的
贪心特征组选择（按 收益/代价 比贪心），用于直观演示 COLD 的算力-效果权衡。

PyTorch 实现（autograd 自动微分），可直接 `python3 cold.py` 运行。
"""
import math
import random
import collections
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _sigmoid_py(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

def _dot_py(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _auc(pairs: List[Tuple[float, int]]) -> float:
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return 0.5
    hit = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return hit / (len(pos) * len(neg))


# ── 特征组算力代价表 ─────────────────────────────────────────────────────────
# (名称, 所属侧, Embedding 维度, 单次在线计算代价)
# 代价的量纲可理解为「相对 RT 开销」：用户静态画像类最便宜（可缓存），
# 物品/用户实时行为序列类最贵（需要对每个候选物品实时拉取并计算）。
FEATURE_GROUPS: List[Tuple[str, str, int, float]] = [
    ("u_id",           "user", 4, 0.5),
    ("u_profile",      "user", 4, 0.6),
    ("u_stat_7d",      "user", 4, 1.8),
    ("u_realtime_seq", "user", 4, 4.5),
    ("i_id",           "item", 4, 0.6),
    ("i_cate",         "item", 4, 0.8),
    ("i_stat_7d",      "item", 4, 1.7),
    ("i_realtime_ctr", "item", 4, 4.0),
]


class COLD(nn.Module):
    """
    COLD 粗排模型：SE 特征组选择 + 全连接 DNN 打分。

    与双塔模型的差别：用户/物品/交叉特征在底层就拼接进同一个 MLP，
    因此能够学习任意形式的非线性交互；代价是无法像双塔那样预计算物品向量，
    必须靠特征组裁剪 + 工程优化（列式计算、Float16、多级并行）把 RT 压回毫秒级。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, groups=FEATURE_GROUPS, hidden: int = 24, lr: float = 0.05, reg: float = 1e-5):
        super().__init__()
        self.groups = list(groups)
        self.names  = [g[0] for g in self.groups]
        self.dims   = [g[2] for g in self.groups]
        self.costs  = {g[0]: g[3] for g in self.groups}
        self.M      = len(self.groups)
        self.total_dim = sum(self.dims)
        self.active = set(self.names)

        # SE excitation 网络：M → M//2 → M，输出过 Sigmoid 得到各组重要性 s
        self.se = nn.Sequential(
            nn.Linear(self.M, max(2, self.M // 2)),
            nn.ReLU(),
            nn.Linear(max(2, self.M // 2), self.M),
        )
        # 打分 DNN：拼接后的全部特征 → hidden → 1
        self.dnn = nn.Sequential(
            nn.Linear(self.total_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.gate_sum = [0.0] * self.M
        self.gate_cnt = 0

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def _squeeze(self, fields: List[torch.Tensor]) -> torch.Tensor:
        """Squeeze：对每个特征组做全局平均池化 z_i = (1/d)·Σ_t e_{i,t}。"""
        return torch.stack([f.mean() for f in fields])

    def se_forward(self, fields: List[torch.Tensor]):
        """SE 全流程：squeeze → excitation → re-weight（v_i = s_i·e_i）。"""
        z = self._squeeze(fields)                 # shape: (M,)
        s = torch.sigmoid(self.se(z))             # shape: (M,)
        v = [s[i] * fields[i] for i in range(self.M)]
        return v, s

    def forward(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        """返回 logit（未过 Sigmoid）。"""
        fields = []
        for name, _, dim, _ in self.groups:
            if name in self.active:
                fields.append(feat[name])
            else:
                fields.append(torch.zeros(dim))
        v, s = self.se_forward(fields)
        # 更新门控统计（detach 避免影响梯度图）
        with torch.no_grad():
            for i in range(self.M):
                self.gate_sum[i] += float(s[i].detach())
            self.gate_cnt += 1
        flat = torch.cat(v)
        return self.dnn(flat)

    def train_model(self, samples, epochs: int = 5) -> None:
        data = list(samples)
        for epoch in range(epochs):
            random.shuffle(data)
            self.gate_sum = [0.0] * self.M
            self.gate_cnt = 0
            total = 0.0
            for feat, y in data:
                self.optimizer.zero_grad()
                feat_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat.items()}
                logit = self.forward(feat_t).squeeze()
                loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(float(y)))
                loss.backward()
                self.optimizer.step()
                total += loss.item()
            print(f"  epoch {epoch}  logloss={total / max(1, len(data)):.4f}")

    # ── 算力代价感知的特征组选择 ──
    def importance(self) -> Dict[str, float]:
        """特征组重要性 = 训练过程中 SE 门控权重 s_i 的均值。"""
        c = max(1, self.gate_cnt)
        return {self.names[i]: self.gate_sum[i] / c for i in range(self.M)}

    def select_by_budget(self, budget: float) -> List[str]:
        """
        预算约束下的贪心特征组选择：
            按「收益密度 = 重要性 / 算力代价」从高到低贪心加入，直到耗尽代价预算。
        对应书中「按权重大小选 Top-K，并结合 GAUC/QPS/RT 调整 K」的工程做法。
        """
        imp = self.importance()
        order = sorted(self.names, key=lambda n: imp[n] / self.costs[n], reverse=True)
        chosen, used = [], 0.0
        for name in order:
            if used + self.costs[name] <= budget:
                chosen.append(name)
                used += self.costs[name]
        return chosen

    def apply_selection(self, names: List[str]) -> None:
        self.active = set(names)

    def cost_of(self, names=None) -> float:
        names = self.names if names is None else names
        return sum(self.costs[n] for n in names)

    def score(self, feat: Dict[str, List[float]]) -> float:
        with torch.no_grad():
            feat_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat.items()}
            return float(torch.sigmoid(self.forward(feat_t)).item())

    def pre_rank(self, uid: int, candidates: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """粗排：对候选集逐个跑 DNN 打分并返回 Top-K（仅计算激活的特征组）。"""
        scored = [(iid, self.score(FEATURE_STORE[(uid, iid)])) for iid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


FEATURE_STORE: Dict[Tuple[int, int], Dict[str, List[float]]] = {}


def make_data(n_users: int, n_items: int, dim: int = 4):
    """合成特征：u_id / i_id 承载真实兴趣信号，实时类特征信号更强但贵，画像/统计类掺大量噪声。"""
    lat_u = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n_users)]
    lat_i = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n_items)]
    samples = []

    def nz(v, sd):
        return [x + random.gauss(0, sd) for x in v]

    for u in range(n_users):
        for i in range(n_items):
            s = _dot_py(lat_u[u], lat_i[i]) / 2.0
            y = 1 if random.random() < _sigmoid_py(s) else 0
            feat = {
                "u_id":           list(lat_u[u]),
                "u_profile":      [random.gauss(0, 1) for _ in range(dim)],
                "u_stat_7d":      nz(lat_u[u], 0.9),
                "u_realtime_seq": nz(lat_u[u], 0.2),
                "i_id":           list(lat_i[i]),
                "i_cate":         [1.0 if x > 0 else -1.0 for x in lat_i[i]],
                "i_stat_7d":      [random.gauss(0, 1) for _ in range(dim)],
                "i_realtime_ctr": nz(lat_i[i], 0.2),
            }
            FEATURE_STORE[(u, i)] = feat
            samples.append((feat, y))
    return samples


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 8, 20
    samples = make_data(n_users, n_items)
    model = COLD(lr=0.05)
    print(f"COLD 粗排：特征组数 M={model.M}  拼接维度={model.total_dim}  "
          f"全量算力代价={model.cost_of():.1f}  样本数={len(samples)}")

    print("\n[1] 全特征组训练（SE 模块同时学习各特征组重要性）")
    model.train_model(samples, epochs=5)

    print("\n[2] SE 门控给出的特征组重要性 / 算力代价")
    imp = model.importance()
    print(f"  {'特征组':<16}{'重要性 s_i':>12}{'代价':>8}{'收益密度':>12}")
    for n in sorted(model.names, key=lambda n: imp[n] / model.costs[n], reverse=True):
        print(f"  {n:<16}{imp[n]:>12.4f}{model.costs[n]:>8.1f}{imp[n] / model.costs[n]:>12.4f}")

    print("\n[3] 不同算力预算下的特征组选择与粗排效果")
    uid = 0
    cands = list(range(n_items))
    for budget in [2.0, 5.0, 14.5]:
        chosen = model.select_by_budget(budget)
        model.apply_selection(chosen)
        auc = _auc([(model.score(f), y) for f, y in samples])
        top = model.pre_rank(uid, cands, top_k=5)
        print(f"\n  预算={budget:>5.1f}  实际代价={model.cost_of(chosen):>4.1f}  AUC={auc:.4f}")
        print(f"    选中特征组：{chosen}")
        print(f"    用户 {uid} 粗排 Top-5：" + "  ".join(f"item{i}:{s:.3f}" for i, s in top))
