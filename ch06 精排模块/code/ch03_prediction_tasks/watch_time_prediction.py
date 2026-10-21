"""
时长预估的四种建模范式（第 6 章 §6.3.1 时长预估）

对应正文概念：
正文指出用户播放时长本质是连续值、天然属于回归问题，但直接回归有四个痛点：
    (1) 时长分布严重长尾，大量样本集中在小时长区间；
    (2) 用户行为噪声大，同一内容在不同曝光场景波动大；
    (3) MSE 等回归损失容易被异常样本带偏，训练不稳定；
    (4) 回归任务通常比分类任务更难收敛。
因此工业界「通常不会直接对原始时长做回归，而是采用一些特殊的建模技巧，把连续
时长目标转换为分类问题或者半分类问题进行学习」（正文 §6.3.1 原话）。

本实现做了什么：
用同一个 MLP 主干、同一份合成长尾时长数据，并排实现四种建模方案并打印误差对比表：
    A. 直接回归（MSE）              —— 正文说的「直接对原始时长回归」基线
    B. 对数域回归（log1p + MSE）    —— 压长尾，预测时 expm1 还原
    C. 分桶分类（Bucketize + CE）   —— 正文说的「转换为分类问题」，期望值解码
    D. 序回归 / 累积分类（Ordinal） —— 正文说的「半分类问题」：K-1 个
       「是否超过阈值 t_k」的二分类头，正好对应正文 §6.3 给的阈值事件标签
           label = 1 if watch_time > 5s else 0
       把单一阈值推广成一组阈值后，时长预估退化为一串有效播放率预测。
另外打印各方案在「长尾头部/腰部/尾部」三段上的分段误差，直观展示长尾的影响。

关键公式：
    直接回归：   L = (1/N) * sum_i (f(x_i) - t_i)^2
    对数回归：   L = (1/N) * sum_i (f(x_i) - log(1+t_i))^2,  t_hat = exp(f)-1
    分桶分类：   L = -sum_i log p_{i, b(t_i)},  t_hat = sum_k p_k * v_k
                 其中 b(.) 为分桶函数，v_k 为第 k 桶的代表时长（桶内均值）
    序回归：     y_{i,k} = 1[t_i > tau_k],  k = 1..K-1
                 L = -sum_{i,k} [ y log sigma(f_k) + (1-y) log(1-sigma(f_k)) ]
                 t_hat = tau_0 + sum_k sigma(f_k) * (tau_k - tau_{k-1})
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

FEAT_DIM = 12
N_TRAIN = 6000
N_TEST = 2000
# 分桶 / 序回归共用的时长边界（秒）。按正文长尾特征，桶宽随时长指数增长。
BOUNDARIES = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
N_BUCKET = len(BOUNDARIES)          # 最后一桶是 [160, inf)


# --------------------------------------------------------------------------
# 合成数据：长尾时长分布
# --------------------------------------------------------------------------


def make_data(n: int, gen: torch.Generator):
    """返回 (X, t)。t 为播放时长（秒），呈对数正态长尾分布。"""
    X = torch.randn(n, FEAT_DIM, generator=gen)
    w = torch.linspace(-0.8, 0.8, FEAT_DIM)
    # 对数正态：log t ~ N(mu(x), sigma)，天然右偏长尾
    mu = 1.3 + X @ w * 0.55
    noise = torch.randn(n, generator=gen) * 0.85      # 正文说的「行为噪声较大」
    t = torch.expm1(mu + noise).clamp(min=0.0, max=600.0)
    # 5% 异常样本（挂机/自动播放），正文说的「MSE 易受异常样本影响」
    outlier = torch.rand(n, generator=gen) < 0.05
    t = torch.where(outlier, t + torch.rand(n, generator=gen) * 400.0, t)
    return X, t


def make_backbone() -> nn.Module:
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(FEAT_DIM, 48), nn.ReLU(),
                         nn.Linear(48, 32), nn.ReLU())


def bucket_of(t: torch.Tensor) -> torch.Tensor:
    b = torch.bucketize(t, torch.tensor(BOUNDARIES[1:]), right=False)
    return b.clamp(max=N_BUCKET - 1)


# --------------------------------------------------------------------------
# 四种建模方案
# --------------------------------------------------------------------------


class DirectRegression(nn.Module):
    """A. 直接回归原始时长。"""

    name = "A.直接回归(MSE)"

    def __init__(self):
        super().__init__()
        self.body = make_backbone()
        self.head = nn.Linear(32, 1)

    def loss(self, X, t):
        return ((self.head(self.body(X)).squeeze(-1) - t) ** 2).mean()

    def predict(self, X):
        return self.head(self.body(X)).squeeze(-1).clamp(min=0.0)


class LogRegression(nn.Module):
    """B. 对数域回归，预测后 expm1 还原。"""

    name = "B.对数回归(log1p)"

    def __init__(self):
        super().__init__()
        self.body = make_backbone()
        self.head = nn.Linear(32, 1)

    def loss(self, X, t):
        return ((self.head(self.body(X)).squeeze(-1) - torch.log1p(t)) ** 2).mean()

    def predict(self, X):
        return torch.expm1(self.head(self.body(X)).squeeze(-1)).clamp(min=0.0)


class BucketClassification(nn.Module):
    """C. 分桶分类：把回归转成 N_BUCKET 分类，用期望值解码成时长。"""

    name = "C.分桶分类(CE)"

    def __init__(self, bucket_value):
        super().__init__()
        self.body = make_backbone()
        self.head = nn.Linear(32, N_BUCKET)
        self.register_buffer("bucket_value", bucket_value)

    def loss(self, X, t):
        logits = self.head(self.body(X))
        return nn.functional.cross_entropy(logits, bucket_of(t))

    def predict(self, X):
        p = torch.softmax(self.head(self.body(X)), dim=-1)
        return p @ self.bucket_value


class OrdinalRegression(nn.Module):
    """D. 序回归 / 累积分类：K-1 个「t > tau_k」二分类头（半分类）。"""

    name = "D.序回归(Ordinal)"

    def __init__(self, tail_excess: float = 160.0):
        """tail_excess: 最后一个阈值以上的平均超出时长 E[t-tau_K | t>tau_K]，
        由训练集统计给出；没有它生存函数积分会系统性低估长尾。"""
        super().__init__()
        self.body = make_backbone()
        self.head = nn.Linear(32, N_BUCKET - 1)
        self.register_buffer("tau", torch.tensor(BOUNDARIES[1:]))
        widths = [BOUNDARIES[i + 1] - BOUNDARIES[i]
                  for i in range(len(BOUNDARIES) - 1)] + [tail_excess]
        self.register_buffer("width", torch.tensor(widths))

    def loss(self, X, t):
        logits = self.head(self.body(X))                   # [B, K-1]
        y = (t.unsqueeze(1) > self.tau).float()            # 阈值事件标签
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def predict(self, X):
        p = torch.sigmoid(self.head(self.body(X)))         # P(t > tau_k)
        # t_hat = tau_0/2 + sum_k P(t > tau_k) * 区间宽度（生存函数积分离散形式）
        return (p * self.width[1:]).sum(dim=-1) + self.width[0] * 0.5

    def threshold_rates(self, X):
        return torch.sigmoid(self.head(self.body(X)))


# --------------------------------------------------------------------------
# 训练与评估
# --------------------------------------------------------------------------


def fit(model, X, t, epochs=12, bs=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for st in range(0, n, bs):
            idx = perm[st:st + bs]
            loss = model.loss(X[idx], t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def metrics(pred, t):
    mae = (pred - t).abs().mean().item()
    rmse = ((pred - t) ** 2).mean().sqrt().item()
    # 对数域 MAE：长尾场景下更能反映相对误差
    lmae = (torch.log1p(pred) - torch.log1p(t)).abs().mean().item()
    # 排序质量：时长预估最终是拿来排序的，Spearman 比绝对误差更重要
    rp = _spearman(pred, t)
    return mae, rmse, lmae, rp


def _spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm())).item()


def segment_mae(pred, t, lo, hi):
    m = (t >= lo) & (t < hi)
    if m.sum() == 0:
        return float("nan"), 0
    return (pred[m] - t[m]).abs().mean().item(), int(m.sum())


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    Xtr, ttr = make_data(N_TRAIN, gen)
    Xte, tte = make_data(N_TEST, gen)

    print("=" * 86)
    print("§6.3.1 时长预估：四种建模范式对比")
    print("=" * 86)
    q = torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99])
    qs = torch.quantile(ttr, q)
    print(f"训练样本 {N_TRAIN} / 测试样本 {N_TEST}，时长单位：秒")
    print(f"时长分布  mean={ttr.mean():.2f}  median={ttr.median():.2f}  "
          f"max={ttr.max():.1f}  mean/median={ttr.mean() / ttr.median():.2f}"
          "（>1 即右偏长尾）")
    print("分位点    " + "  ".join(f"P{int(a * 100)}={b:.1f}"
                                   for a, b in zip(q.tolist(), qs.tolist())))

    cnt = torch.bincount(bucket_of(ttr), minlength=N_BUCKET)
    print("\n时长分桶（正文：大量样本集中于较小时长区间）")
    print(f"{'桶':<4}{'区间(s)':<16}{'样本数':>8}{'占比':>9}")
    print("-" * 86)
    for k in range(N_BUCKET):
        lo = BOUNDARIES[k]
        hi = BOUNDARIES[k + 1] if k + 1 < N_BUCKET else float("inf")
        rng = f"[{lo:g}, {hi:g})" if math.isfinite(hi) else f"[{lo:g}, inf)"
        print(f"{k:<4}{rng:<16}{int(cnt[k]):>8}{cnt[k] / N_TRAIN:>9.2%}")

    # 分桶代表值：用训练集桶内均值（而非桶中点），避免最后一桶无界
    bvals = []
    btr = bucket_of(ttr)
    for k in range(N_BUCKET):
        m = btr == k
        bvals.append(ttr[m].mean().item() if m.sum() > 0
                     else BOUNDARIES[min(k, N_BUCKET - 1)])
    bucket_value = torch.tensor(bvals)
    print("\n各桶代表时长 v_k（训练集桶内均值，用于分桶模型的期望值解码）：")
    print("  " + "  ".join(f"v{k}={v:.1f}" for k, v in enumerate(bvals)))
    tail = ttr[ttr > BOUNDARIES[-1]]
    tail_excess = (tail - BOUNDARIES[-1]).mean().item() if len(tail) else 160.0
    print(f"尾部超出量 E[t-{BOUNDARIES[-1]:g} | t>{BOUNDARIES[-1]:g}] = "
          f"{tail_excess:.1f}s（序回归解码最后一段宽度）")

    models = [DirectRegression(), LogRegression(),
              BucketClassification(bucket_value), OrdinalRegression(tail_excess)]

    print("\n" + "=" * 86)
    print(f"{'方案':<20}{'MAE(s)':>10}{'RMSE(s)':>10}{'log-MAE':>10}"
          f"{'Spearman':>11}")
    print("=" * 86)
    results = []
    for m in models:
        torch.manual_seed(42)
        fit(m, Xtr, ttr)
        m.eval()
        with torch.no_grad():
            pred = m.predict(Xte)
        mae, rmse, lmae, rp = metrics(pred, tte)
        results.append((m, pred, mae, rmse, lmae, rp))
        print(f"{m.name:<20}{mae:>10.3f}{rmse:>10.3f}{lmae:>10.4f}{rp:>11.4f}")
    print("=" * 86)
    best_l = min(results, key=lambda r: r[4])
    best_s = max(results, key=lambda r: r[5])
    print(f"log-MAE 最优：{best_l[0].name}（{best_l[4]:.4f}）；"
          f"排序相关性最优：{best_s[0].name}（{best_s[5]:.4f}）")

    # 分段误差：头部（短时长，样本最多）/腰部/尾部
    print("\n分段 MAE（展示长尾对各方案的影响）")
    segs = [("短时长 [0,5)", 0.0, 5.0), ("中时长 [5,40)", 5.0, 40.0),
            ("长时长 [40,inf)", 40.0, float("inf"))]
    header = f"{'方案':<20}" + "".join(f"{s[0]:>18}" for s in segs)
    print(header)
    print("-" * 86)
    for m, pred, *_ in results:
        row = f"{m.name:<20}"
        for _, lo, hi in segs:
            v, c = segment_mae(pred, tte, lo, hi)
            row += f"{v:>12.2f}(n={c})"
        print(row)

    # 序回归天然给出正文 §6.3 的阈值事件率（有效播放率/长播率）
    ordm = results[-1][0]
    with torch.no_grad():
        rates = ordm.threshold_rates(Xte).mean(dim=0)
    print("\n序回归额外产出：正文 §6.3 的阈值事件率 label=1[watch_time>tau]")
    print(f"{'阈值 tau(s)':<14}{'预估 P(t>tau)':>16}{'真实比例':>12}{'偏差':>10}")
    print("-" * 86)
    for k, tau in enumerate(BOUNDARIES[1:]):
        real = (tte > tau).float().mean().item()
        print(f"{tau:<14g}{rates[k].item():>16.4f}{real:>12.4f}"
              f"{rates[k].item() - real:>10.4f}")
    print("\n结论：直接回归被长尾与异常样本主导，误差集中在长时长段；对数回归与"
          "分桶/序回归\n      把目标压回分类或半分类空间，排序相关性更稳，且序回归"
          "的每个头\n      天然就是正文所说的「有效播放率 / 完播率 / 长播率」。")
