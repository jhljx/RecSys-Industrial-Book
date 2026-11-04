"""
特征筛选实验的公共部分（第15章 §15.6）

提供两部分内容：
1) 一份"有真值"的合成 CTR 数据：12 个特征里只有一部分真正影响点击，
   另一些是纯噪声或高度冗余，便于验证各种筛选指标能不能把它们认出来；
2) 一个不依赖 sklearn 的小型 Logistic Regression（批量 SGD），
   以及 AUC、置换重要性等工具函数。

特征筛选关注的是"这个特征值不值得留"，不是模型结构，所以这里刻意用最简单的
模型，避免模型能力掩盖特征本身的信息量。

仅依赖 numpy。
"""

from __future__ import annotations

import numpy as np

# 12 个特征：前 9 个连续/计数类，后 3 个离散类
FEATURE_NAMES = [
    "user_activity",   # 用户活跃度（隐变量，连续）
    "clicks_7d",       # 近 7 天点击数
    "clicks_30d",      # 近 30 天点击数（与 7d 高度相关）
    "item_ctr",        # 物品历史 CTR
    "item_popularity", # 物品热度（真值：无用）
    "cat_match",       # 主兴趣类目是否匹配（强信号）
    "hist_len",        # 用户历史行为长度
    "follow_ratio",    # 关注比例（真值：无用）
    "is_new_user",     # 是否新用户
    "hour_bucket",     # 时段分桶
    "device",          # 设备类型
    "price_band",      # 价格带（真值：无用）
]

# 真值里真正起作用的特征（用于最后对照：筛选方法能不能找到它们）
TRUE_SIGNAL = {
    "user_activity", "clicks_7d", "item_ctr", "cat_match",
    "is_new_user", "hour_bucket", "device",
}

N_FEATURES = len(FEATURE_NAMES)


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std() or 1.0
    return (x - x.mean()) / sd


def make_ctr_data(n: int = 30_000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X [n, 12], y [n])。列顺序与 FEATURE_NAMES 一致。"""
    rng = np.random.default_rng(seed)

    activity = rng.normal(0, 1, n)
    clicks7 = rng.poisson(np.exp(0.8 * activity + 1.0))
    clicks30 = 4 * clicks7 + rng.poisson(2.0, n)          # 与 7d 强相关
    item_ctr = rng.beta(2, 18, n)
    popularity = rng.pareto(1.8, n)
    cat_match = (rng.random(n) < 0.35).astype(float)
    hist_len = rng.poisson(np.exp(0.7 * activity + 2.0))
    follow_ratio = rng.beta(2, 8, n)
    is_new = (hist_len == 0).astype(float)
    hour = rng.integers(0, 6, n).astype(float)
    device = rng.choice([0.0, 1.0, 2.0], n, p=[0.6, 0.3, 0.1])
    price = rng.integers(0, 5, n).astype(float)

    logit = (
        -1.2
        + 0.70 * _z(activity)
        + 0.45 * _z(np.log1p(clicks7))
        + 0.60 * _z(item_ctr)
        + 0.90 * cat_match
        - 0.30 * is_new
        + 0.30 * (hour >= 4)
        + 0.20 * (device == 1)
        + rng.normal(0, 0.3, n)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float64)

    X = np.stack(
        [activity, clicks7, clicks30, item_ctr, popularity, cat_match,
         hist_len, follow_ratio, is_new, hour, device, price],
        axis=1,
    )
    return X, y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """带并列处理的 AUC。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def train_logreg(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    steps: int = 400,
    lr: float = 0.5,
    l2: float = 1e-3,
    batch: int = 1024,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """
    批量 SGD 训练 Logistic Regression。返回 (w, b, best_val_auc)。
    X 需事先标准化（见 standardize）。
    """
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    w = np.zeros(d)
    b = 0.0
    best = 0.0
    for step in range(steps):
        idx = rng.integers(0, n, batch)
        xb, yb = Xtr[idx], ytr[idx]
        p = sigmoid(xb @ w + b)
        err = p - yb
        w -= lr * (xb.T @ err / batch + l2 * w)
        b -= lr * err.mean()
        if step % 20 == 0 or step == steps - 1:
            auc = auc_score(Xva @ w + b, yva)
            best = max(best, auc)
    return w, b, best


def standardize(*arrays: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    用第一个数组（训练集）的均值方差标准化所有数组。

    注意：这正是 §15.6.6 提到的泄露点之一——如果拿全量数据（含未来）算均值方差，
    验证集的信息就会漏进训练特征。脚本里统一用训练集统计量，就是为了不犯这个错。
    """
    train = arrays[0]
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return [(a - mu) / sd for a in arrays], mu, sd


def grade(auc: float) -> str:
    """把 AUC 高低映射成一句人话，方便表格阅读。"""
    if auc < 0.53:
        return "≈无信号"
    if auc < 0.58:
        return "弱"
    if auc < 0.65:
        return "中等"
    return "强"
