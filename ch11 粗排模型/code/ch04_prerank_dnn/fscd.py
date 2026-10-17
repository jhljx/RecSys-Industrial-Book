"""
FSCD 粗排模型（第11章 §11.4.3）

FSCD = Feature Selection method based on feature Complexity and variational Dropout
论文出处：Ma et al., "Towards a Better Tradeoff between Effectiveness and Efficiency in
          Pre-Ranking: A Learnable Feature Selection based Approach", SIGIR 2021（阿里巴巴）。

核心思想（对应书中 §11.4.3）：
    FSCD 不重新设计粗排网络，而是直接继承精排的交互式 DNN 结构，只在「输入特征」这一层做裁剪，
    把特征选择变成一个可学习、且把在线算力成本写进目标函数的过程。
    1) 为每个特征域 j 引入伯努利门控   z_j ~ Bern(θ_j)
    2) 特征复杂度                      c_j = G(o_j, e_j, n_j) = γ1·o_j + γ2·e_j + γ3·n_j
       （o_j 在线计算复杂度、e_j Embedding 维度、n_j 特征取值数量）
    3) 复杂度决定先验保留概率           θ_j = 1 - σ(c_j)      → 越贵越不容易被保留
    4) 统一训练目标                     L = -1/N·logP(D|v,w,z) + λ/N·(‖v‖² + ‖w‖²) + 1/N·Σ_j α_j·z_j
       其中特征选择正则权重             α_j = log(1-θ_j) - log(θ_j)
    5) 伯努利不可导 → Gumbel-Sigmoid 连续松弛
       z_j = σ( ( logδ_j - log(1-δ_j) + log u_j - log(1-u_j) ) / t ),  u_j ~ U(0,1)
    6) 两阶段：第一阶段联合学 δ_j 与模型参数；收敛后按 δ_j 取 Top-K 特征，
       仅保留这些特征重新微调（目标退化为纯 CTR 损失）。

PyTorch 实现（autograd 自动微分），可直接 `python3 fscd.py` 运行。
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


# ── 特征域定义：(名称, 所属侧, o_j 在线计算复杂度, e_j Embedding 维度, n_j 取值数量) ──
FEATURE_FIELDS: List[Tuple[str, str, float, int, int]] = [
    ("u_profile", "user",  0.4, 4,     500),
    ("u_city",    "user",  0.4, 2,    3000),
    ("u_id",      "user",  1.0, 4,   50000),
    ("u_stat_7d", "user",  2.0, 4,  100000),
    ("i_cate",    "item",  0.5, 4,     800),
    ("i_price",   "item",  0.5, 2,     100),
    ("i_id",      "item",  1.0, 4,   20000),
    ("x_ui_seq",  "cross", 4.0, 6, 1000000),
]


class FSCD(nn.Module):
    """
    FSCD：复杂度感知的可学习特征选择 + 交互式 DNN 粗排模型。
    网络结构与精排一致（所有特征域 Embedding 拼接 → MLP → CTR），
    唯一的区别是在 Embedding 与 DNN 之间插入一层可学习门控 z_j。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, fields=FEATURE_FIELDS, hidden: int = 24, lr: float = 0.05,
                 reg: float = 1e-5, gamma=(0.5, 0.3, 0.3), temperature: float = 0.5,
                 sel_weight: float = 0.05):
        super().__init__()
        self.fields    = list(fields)
        self.names     = [f[0] for f in self.fields]
        self.dims      = [f[3] for f in self.fields]
        self.M         = len(self.fields)
        self.total_dim = sum(self.dims)
        self.t         = temperature
        self.sel_weight = sel_weight

        # 特征复杂度 c_j、先验保留概率 θ_j、选择正则权重 α_j（Python 常量，不参与训练）
        g1, g2, g3 = gamma
        self.c     = [g1 * o + g2 * (e / 4.0) + g3 * (math.log10(n) / 2.0)
                      for _, _, o, e, n in self.fields]
        self.theta = [1.0 - _sigmoid_py(cj) for cj in self.c]
        self.alpha = [math.log(1 - th) - math.log(th) for th in self.theta]

        # 可学习参数 δ_j（logit 形式）：nn.Parameter → autograd 自动追踪
        self.logit_delta = nn.Parameter(torch.zeros(self.M))

        # 打分 DNN
        self.dnn = nn.Sequential(
            nn.Linear(self.total_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # 第二阶段使用的硬门控（不是参数，只是 buffer）
        self.register_buffer("mask", torch.ones(self.M))

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def gumbel_sigmoid(self) -> torch.Tensor:
        """Gumbel-Sigmoid 连续松弛（训练阶段可微）：
           z_j = σ((logit δ_j + gumbel_noise) / t)
           gumbel = -log(-log(u))，u ~ U(0,1)。
        """
        u = torch.rand(self.M).clamp(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(u))            # Gumbel(0,1) 样本
        logit_u = torch.log(u) - torch.log(1 - u)    # logit(u)
        return torch.sigmoid((self.logit_delta + logit_u) / self.t)

    def delta(self) -> torch.Tensor:
        """返回当前特征保留强度 δ_j = σ(logit_delta_j)。"""
        return torch.sigmoid(self.logit_delta)

    def _apply_gate(self, feat: Dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """逐特征域缩放后拼接：x = [z_1·v_1, ..., z_M·v_M]。"""
        parts = []
        for j, name in enumerate(self.names):
            parts.append(z[j] * feat[name])
        return torch.cat(parts)

    def forward(self, feat: Dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """返回 logit（未过 Sigmoid）。"""
        flat = self._apply_gate(feat, z)
        return self.dnn(flat)

    # ── 第一阶段：联合优化模型参数与特征选择 ──
    def train_stage1(self, samples, epochs: int = 5) -> None:
        data = list(samples)
        alpha_t = torch.tensor(self.alpha, dtype=torch.float32)
        for epoch in range(epochs):
            random.shuffle(data)
            c_sum = s_sum = 0.0
            for feat, y in data:
                self.optimizer.zero_grad()
                feat_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat.items()}
                z = self.gumbel_sigmoid()
                logit = self.forward(feat_t, z).squeeze()
                loss_ctr = F.binary_cross_entropy_with_logits(logit, torch.tensor(float(y)))
                # 特征选择正则：Σ_j α_j · z_j（α_j 为正则权重，复杂度越高越大）
                loss_sel = self.sel_weight * (alpha_t * z).sum()
                loss = loss_ctr + loss_sel
                loss.backward()
                self.optimizer.step()
                c_sum += loss_ctr.item()
                s_sum += loss_sel.item()
            n = max(1, len(data))
            print(f"  [stage1] epoch {epoch}  CTR_loss={c_sum / n:.4f}  "
                  f"选择正则={s_sum / n:.4f}  总损失={(c_sum + s_sum) / n:.4f}")

    # ── 第二阶段：Top-K 特征微调 ──
    def select_top_k(self, k: int) -> List[str]:
        """按 δ_j 降序选 Top-K，更新硬门控 mask。"""
        with torch.no_grad():
            d = self.delta()
            order = torch.argsort(d, descending=True)[:k].tolist()
        self.mask = torch.zeros(self.M)
        for j in order:
            self.mask[j] = 1.0
        return [self.names[j] for j in sorted(order)]

    def set_mask(self, names: List[str]) -> None:
        keep = set(names)
        self.mask = torch.tensor(
            [1.0 if n in keep else 0.0 for n in self.names])

    def train_stage2(self, samples, epochs: int = 3) -> None:
        """仅保留选中特征，目标退化为纯 CTR 损失（以第一阶段参数作为初始化）。"""
        data = list(samples)
        for epoch in range(epochs):
            random.shuffle(data)
            total = 0.0
            for feat, y in data:
                self.optimizer.zero_grad()
                feat_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat.items()}
                logit = self.forward(feat_t, self.mask).squeeze()
                loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(float(y)))
                loss.backward()
                self.optimizer.step()
                total += loss.item()
            print(f"  [stage2] epoch {epoch}  CTR_loss={total / max(1, len(data)):.4f}")

    def online_cost(self) -> float:
        """被保留特征的在线计算复杂度之和。"""
        return sum(self.fields[j][2] for j in range(self.M) if self.mask[j].item() > 0)

    # ── 推理接口 ──
    def score(self, feat: Dict[str, List[float]]) -> float:
        with torch.no_grad():
            feat_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat.items()}
            return float(torch.sigmoid(self.forward(feat_t, self.mask)).item())

    def pre_rank(self, uid: int, candidates: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """粗排：只用筛选后的特征对候选集打分并返回 Top-K。"""
        scored = [(iid, self.score(FEATURE_STORE[(uid, iid)])) for iid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


FEATURE_STORE: Dict[Tuple[int, int], Dict[str, List[float]]] = {}


def make_data(n_users: int, n_items: int):
    """
    合成特征：u_profile / i_cate 廉价且携带完整兴趣信号；u_id / i_id / u_stat_7d 信号被噪声稀释
    且词表大、复杂度高；x_ui_seq 信号最强但在线复杂度最高（FSCD 重点裁剪对象）；
    u_city / i_price 为纯噪声但复杂度极低。
    """
    lat_u = [[random.gauss(0, 1) for _ in range(4)] for _ in range(n_users)]
    lat_i = [[random.gauss(0, 1) for _ in range(4)] for _ in range(n_items)]
    samples = []
    for u in range(n_users):
        for i in range(n_items):
            s = _dot_py(lat_u[u], lat_i[i]) / 2.0
            y = 1 if random.random() < _sigmoid_py(s) else 0
            feat = {
                "u_profile": list(lat_u[u]),
                "u_city":    [random.gauss(0, 1) for _ in range(2)],
                "u_id":      [x + random.gauss(0, 0.6) for x in lat_u[u]],
                "u_stat_7d": [x + random.gauss(0, 1.2) for x in lat_u[u]],
                "i_cate":    list(lat_i[i]),
                "i_price":   [random.gauss(0, 1) for _ in range(2)],
                "i_id":      [x + random.gauss(0, 0.6) for x in lat_i[i]],
                "x_ui_seq":  [s + random.gauss(0, 0.2)] * 3 + [s, s, s],
            }
            FEATURE_STORE[(u, i)] = feat
            samples.append((feat, y))
    return samples


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 8, 20
    samples = make_data(n_users, n_items)
    model = FSCD(lr=0.05, temperature=0.5)
    print(f"FSCD 粗排：特征域数 M={model.M}  拼接维度={model.total_dim}  样本数={len(samples)}")

    print("\n[1] 特征复杂度与选择正则（复杂度越高 → θ 越小 → α 越大 → 越容易被裁掉）")
    print(f"  {'特征域':<12}{'o_j':>6}{'e_j':>5}{'n_j':>10}{'c_j':>8}{'θ_j':>8}{'α_j':>8}")
    for j, (name, _, o, e, n) in enumerate(model.fields):
        print(f"  {name:<12}{o:>6.1f}{e:>5d}{n:>10d}{model.c[j]:>8.3f}"
              f"{model.theta[j]:>8.3f}{model.alpha[j]:>8.3f}")

    print("\n[2] 第一阶段：Gumbel-Sigmoid 变分 Dropout，联合学习 δ_j 与 DNN")
    model.train_stage1(samples, epochs=5)
    with torch.no_grad():
        deltas = model.delta().tolist()
    print("  学到的特征保留强度 δ_j（降序）：")
    for j in sorted(range(model.M), key=lambda j: deltas[j], reverse=True):
        print(f"    {model.names[j]:<12}δ={deltas[j]:.4f}")

    full_auc = _auc([(model.score({k: v for k, v in f.items()}) if False
                      else float(torch.sigmoid(model.dnn(
                          torch.cat([torch.tensor(f[n], dtype=torch.float32)
                                     for n in model.names]))).item()), y)
                     for f, y in samples])
    # 全特征（mask=1）的 AUC
    with torch.no_grad():
        orig_mask = model.mask.clone()
        model.mask = torch.ones(model.M)
        full_auc = _auc([(model.score(f), y) for f, y in samples])
        model.mask = orig_mask

    chosen = model.select_top_k(4)
    fscd_auc  = _auc([(model.score(f), y) for f, y in samples])
    fscd_cost = model.online_cost()

    # 对照组：人工按「复杂度最高」挑 4 个特征
    heavy = sorted(model.names, key=lambda n: model.c[model.names.index(n)], reverse=True)[:4]
    model.set_mask(heavy)
    heavy_auc  = _auc([(model.score(f), y) for f, y in samples])
    heavy_cost = model.online_cost()
    model.set_mask(chosen)

    total_cost = sum(f[2] for f in model.fields)
    print(f"\n[3] 效果-成本权衡（全量在线复杂度={total_cost:.1f}）")
    print(f"  全特征            AUC={full_auc:.4f}  在线复杂度=9.8")
    print(f"  FSCD Top-4        AUC={fscd_auc:.4f}  在线复杂度={fscd_cost:.1f}   {chosen}")
    print(f"  高复杂度 Top-4    AUC={heavy_auc:.4f}  在线复杂度={heavy_cost:.1f}   {heavy}")

    print("\n[4] 第二阶段：仅用筛选后的特征微调（目标退化为纯 CTR 损失）")
    model.train_stage2(samples, epochs=3)
    final_auc = _auc([(model.score(f), y) for f, y in samples])
    print(f"  微调后 FSCD Top-4 AUC：{final_auc:.4f}"
          f"（≈ 全特征效果，但在线复杂度只有 {fscd_cost / 9.8:.0%}）")

    uid = 0
    print(f"\n[5] 用户 {uid} 粗排 Top-5（仅使用 {len(chosen)} 个被保留的特征域）")
    for iid, s in model.pre_rank(uid, list(range(n_items)), top_k=5):
        print(f"  item {iid:2d}: {s:.4f}")
