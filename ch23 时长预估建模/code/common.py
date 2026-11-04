"""
第23章 时长预估建模 —— 公共工具

本模块为 ch01–ch06 各脚本提供：

1. 合成观看数据：用户的潜在兴趣 r、视频物理时长 duration、观看时长 watch_time，
   按正文 §23.4 的 D2Co 分解生成：

       W = r · W⁺(D) + (1 − r) · W⁻(D)

   其中 W⁺ 是「感兴趣时的观看时长」（随 Duration 增长，体现 Duration Bias），
   W⁻ 是「不感兴趣时的噪声观看」（主要是几秒钟的误触/滑走）。这样 D2Q 的分位数
   变换、DVR 的 WTG、D2Co 的两成分 GMM 都有真实可恢复的结构，而不是拟合噪声。

2. 线性模型：Logistic Regression（牛顿法 / 梯度下降）与最小二乘回归，numpy 实现，
   不依赖 sklearn。

3. 指标：AUC、MAE、RMSE、分位数、经验 CDF 与逆 CDF。

所有脚本固定 np.random.default_rng(2026)，结果可复现。
"""

from __future__ import annotations

import numpy as np

SEED = 2026


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def make_watch_data(n: int = 20000, dim: int = 16, seed: int = SEED
                    ) -> dict[str, np.ndarray]:
    """
    生成一次曝光的数据：用户特征 x、视频物理时长 d、真实兴趣 r、观看时长 w。

    生成过程（与正文的因果图一致）：
        r     ~ Bernoulli(sigmoid(β·u + γ·v))      潜在兴趣，不可观测
        d     ~ LogNormal，截断到 [12, 600] 秒     视频物理时长（下限高于 10 秒阈值）
        c     ~ Bernoulli(σ(2.5 − 0.008·d))        感兴趣时是否看完（短视频更易看完）
        W⁺(d) = c·d + (1−c)·min(d, Exp(mean=4s))   看完则看满，否则几秒后离开
        W⁻(d) ~ Exponential(mean=2.5s) 截断到 d    不感兴趣：几秒的噪声观看
        w     = r·W⁺ + (1−r)·W⁻，再以 12% 概率置 0（曝光但没播放）
        x     = [u, v, log(d)]                     用户/上下文特征 + Duration

    为什么用「是否看完」而不是「看 d 的固定比例」：固定比例意味着 120 秒视频
    看 75% 就是 90 秒，它在任何绝对阈值下都比短视频更容易为正，Duration Bias
    的方向就反了。真实行为是「兴趣决定要不要看下去」，看完概率随物理时长缓慢
    下降；没看完的人几秒后就滑走，观看时长不随 d 放大。这样固定 10 秒阈值下，
    短视频桶的正样本率确实更高，与正文 §23.1「短视频天然具有更高的有效播放率、
    长播率以及完播率」一致。

    Duration 下限设成 12 秒，是因为 d < 10 秒的视频物理上不可能满足 W > 10，
    把它们混在最短桶里会让该桶的正样本率被稀释，掩盖真正想展示的偏差。

    两个刻意的设计：
        · 12% 的曝光根本没有播放（列表里划过去，w=0）。没有这批样本，
          Weighted LR 和 MSE 的区别就无从谈起——正文 §23.2 的前提正是
          「大量曝光但未产生观看行为的样本」。
        · x 里带 log(d)。Duration 在线上是已知的 Item 属性，模型必须能看见它，
          否则对长视频只能输出全局均值，讨论 WLR 与 MSE 谁拟合得更好就没意义。
          注意这不等于「把 duration 当强特征学出 Duration Bias」：bias 说的是
          Label 定义让短视频天然更容易为正，与特征里有没有 duration 是两件事。

    注意 Duration Bias 的两个来源都在里面：
        · 看完概率随 d 下降 → 固定绝对阈值下各桶正样本率不同；
        · 曝光并不均匀（这里用兴趣生成，真实系统里是推荐反馈循环）。
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, dim // 2)).astype(np.float32)
    v = rng.normal(size=(n, dim - dim // 2)).astype(np.float32)

    beta = rng.normal(0, 0.6, size=dim // 2)
    gamma = rng.normal(0, 0.4, size=dim - dim // 2)
    z = u @ beta + v @ gamma
    p = 1.0 / (1.0 + np.exp(-z))
    r = (rng.random(n) < p).astype(np.float32)

    # 物理时长：对数正态，裁剪到 [12, 600]（下限见上面的说明）
    d = np.clip(rng.lognormal(mean=np.log(45), sigma=0.7, size=n), 12, 600)

    # 兴趣还受视频时长影响：同样的用户，长视频更难留住注意力
    z = z - 0.5 * (np.log(d) - np.log(45))
    p = 1.0 / (1.0 + np.exp(-z))
    r = (rng.random(n) < p).astype(np.float32)

    # 感兴趣时：先决定「看没看完」，看完则看满 d；没看完就几秒后离开
    p_done = 1.0 / (1.0 + np.exp(-(2.5 - 0.008 * d)))
    done = (rng.random(n) < p_done).astype(np.float32)
    partial = np.minimum(d, rng.exponential(4.0, size=n))
    w_pos = done * d + (1.0 - done) * partial
    # 不感兴趣：平均 2.5 秒的噪声观看，且不超过视频长度
    w_neg = np.minimum(rng.exponential(2.5, size=n), d)
    w = np.where(r == 1, w_pos, w_neg)
    # 「点开但立刻走」：即使感兴趣，也有一小部分因为加载/手滑只看了 1~2 秒
    slip = rng.random(n) < 0.03
    w = np.where(slip, np.minimum(rng.exponential(1.5, size=n), d), w)
    # 曝光但没播放：连点开都没有，观看时长严格为 0
    no_play = rng.random(n) < 0.12
    w = np.where(no_play, 0.0, w)
    w = np.clip(w, 0.0, d)

    # 模型可见的特征：用户侧 u/v + 已知的 Item Duration（对数尺度）
    x = np.concatenate([u, v, np.log(d).astype(np.float32)[:, None]], axis=1)

    return {
        "x": x, "d": d.astype(np.float32), "r": r,
        "w": w.astype(np.float32), "p": p.astype(np.float32),
    }


def split(idx: np.ndarray, n: int, frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """按给定顺序切分训练/验证下标。"""
    cut = int(n * frac)
    return idx[:cut], idx[cut:]


# --------------------------------------------------------------------------
# 线性模型（numpy）
# --------------------------------------------------------------------------
def standardize(Xtr: np.ndarray, Xte: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用训练集统计量标准化——验证集复用同一组均值/方差（避免泄露）。"""
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_logistic(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
                 l2: float = 1e-3, iters: int = 400, lr: float = 0.5,
                 seed: int = SEED) -> np.ndarray:
    """
    加权 Logistic Regression，梯度下降。

    加权对数似然：L = Σ_i w_i [y_i log p_i + (1−y_i) log(1−p_i)]
    梯度：        ∂L/∂θ = Xᵀ (w ⊙ (y − p)) − l2·θ

    加权二分类正是 Weighted LR 的形式——把 (y=1, w=t) 与 (y=0, w=1) 两条
    虚拟样本合起来，恰好就是这里的 sample_weight 与 y。
    """
    n, dim = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    theta = np.zeros(dim + 1)
    rng = np.random.default_rng(seed)
    theta[:-1] = rng.normal(0, 1e-3, size=dim)
    if sample_weight is None:
        sample_weight = np.ones(n)
    sw = sample_weight.astype(np.float64)
    for _ in range(iters):
        p = sigmoid(Xb @ theta)
        grad = Xb.T @ (sw * (y - p)) - l2 * np.concatenate([theta[:-1], [0.0]])
        theta += lr * grad / n
    return theta


def predict_logistic(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return Xb @ theta


def fit_linear(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """岭回归闭式解，只用来做「MSE 回归」对照。"""
    n = len(X)
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    A = Xb.T @ Xb + l2 * np.eye(Xb.shape[1])
    A[-1, -1] -= l2                       # 截距不惩罚
    return np.linalg.solve(A, Xb.T @ y)


def predict_linear(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((len(X), 1))], axis=1) @ theta


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def auc(y: np.ndarray, score: np.ndarray) -> float:
    """带并列处理（平均秩）的 AUC。"""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # 并列样本取平均秩
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            ranks[order[i:j]] = avg
        i = j
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).mean())


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(((y - pred) ** 2).mean()))


def quantile(values: np.ndarray, q: np.ndarray | float) -> np.ndarray:
    """经验分位数（线性插值，与 numpy.quantile 的默认口径一致）。"""
    return np.quantile(values, q)


def empirical_cdf(ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    """用参考样本的经验 CDF 给 values 打分（范围 [0,1]）。"""
    ref_sorted = np.sort(ref)
    pos = np.searchsorted(ref_sorted, values, side="right")
    return pos / len(ref_sorted)


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def table(header: list[str], rows: list[list], widths: list[int] | None = None) -> None:
    """打印等宽表格（中文按 2 个字符宽度计算）。"""
    def w(s: str) -> int:
        return sum(2 if ord(ch) > 127 else 1 for ch in str(s))

    def pad(s: str, width: int) -> str:
        s = str(s)
        return s + " " * max(0, width - w(s))

    cells = [[str(c) for c in r] for r in rows]
    if widths is None:
        widths = [max([w(h)] + [w(r[j]) for r in cells])
                  for j, h in enumerate(header)]
    print("  ".join(pad(h, widths[j]) for j, h in enumerate(header)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in cells:
        print("  ".join(pad(c, widths[j]) for j, c in enumerate(r)))
