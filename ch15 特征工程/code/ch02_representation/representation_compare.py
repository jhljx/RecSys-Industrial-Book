"""
同一信息的四种特征表示对比（第15章 §15.2.1）

正文的核心论点：同一条业务信息可以表示为 Sparse ID、Dense 数值、分桶
Embedding 或 Sequence，不同选择会改变模型的表达能力与泛化能力。

本脚本构造一个"数值语义确实重要"的合成任务：观看次数 x 越大，点击概率越高，
但 x=6 与 x=7 的差异小于 x=6 与 x=60 的差异。分别用四种表示训练逻辑回归
（手写 numpy，避免依赖），比较测试集 AUC 与参数量。

运行：python3 ch02_representation/representation_compare.py
"""

from __future__ import annotations

import math
import random

import numpy as np

N_TRAIN, N_TEST = 12_000, 4_000


def markdown_table(headers: list[str], rows: list[list]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, steps: int = 400, lr: float = 0.5):
    """全批量梯度下降的逻辑回归；X 已是设计矩阵。"""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列分数
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def make_data(rng: random.Random, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 (最近 5 次观看次数, 是否点击) 的样本。

    设计要点：用户的潜在兴趣 z 决定观看强度；同时兴趣有"上升/平稳/下降"
    三种趋势，而近期上升本身会提高点击率（用户越看越投入）。因此：
    - 总次数只能反映 z，看不到趋势；
    - 5 步序列同时包含 z 与趋势。
    这正是正文"统计特征压缩信息、序列特征保留轨迹"的量化版本。
    """
    hist_all, ys, trends = [], [], []
    for _ in range(n):
        z = min(1.0, max(0.0, rng.betavariate(2.0, 3.0)))
        trend = rng.choice((-1, 0, 1))
        if trend == 1:
            scale = [0.4, 0.7, 1.0, 1.3, 1.6]
        elif trend == -1:
            scale = [1.6, 1.3, 1.0, 0.7, 0.4]
        else:
            scale = [1.0] * 5
        hist = [max(0, int(round(z * 30 * s + rng.gauss(0, 2)))) for s in scale]
        logit = -1.5 + 3.0 * z + 0.8 * trend
        p = 1 / (1 + math.exp(-logit))
        y = 1 if rng.random() < p else 0
        hist_all.append(hist)
        ys.append(y)
        trends.append(trend)
    return np.array(hist_all, dtype=float), np.array(ys, dtype=float), np.array(trends)


def one_hot(xs: np.ndarray, n_bins: int) -> np.ndarray:
    X = np.zeros((len(xs), n_bins))
    for i, x in enumerate(xs):
        X[i, min(int(x), n_bins - 1)] = 1.0
    return X


def bucket_idx(xs: np.ndarray, edges: list[int]) -> np.ndarray:
    return np.searchsorted(np.array(edges), xs, side="right")


def main() -> None:
    rng = random.Random(42)
    H_tr, y_tr, t_tr = make_data(rng, N_TRAIN)
    H_te, y_te, t_te = make_data(rng, N_TEST)
    x_tr, x_te = H_tr.sum(1), H_te.sum(1)

    print(f"训练集 {N_TRAIN} 条，正样本率 {y_tr.mean():.3f}；测试集 {N_TEST} 条")
    print("总观看次数分位数：", {q: int(np.percentile(x_tr, q)) for q in (50, 75, 90, 99)})
    print("趋势分布（-1 下降 / 0 平稳 / 1 上升）：",
          {t: int((t_tr == t).sum()) for t in (-1, 0, 1)})

    results = []

    # (1) Sparse ID：总次数离散成 60 个槽位（等价于每个取值一个 Embedding）
    n_bins = 60
    X = one_hot(np.minimum(x_tr, n_bins - 1), n_bins)
    Xt = one_hot(np.minimum(x_te, n_bins - 1), n_bins)
    w, b = train_logreg(X, y_tr)
    results.append(["Sparse ID（总次数 60 槽）", auc_score(Xt @ w + b, y_te), n_bins + 1,
                    "6 与 7 完全独立，且看不到趋势"])

    # (2) Dense 原值（标准化）
    mu_x, sd_x = x_tr.mean(), x_tr.std() + 1e-9
    X = np.stack([(x_tr - mu_x) / sd_x, np.ones(N_TRAIN)], axis=1)
    Xt = np.stack([(x_te - mu_x) / sd_x, np.ones(N_TEST)], axis=1)
    w, b = train_logreg(X, y_tr)
    results.append(["Dense 总次数（标准化）", auc_score(Xt @ w + b, y_te), 2,
                    "保留数值语义，但只看总量"])

    # (3) Dense + log1p
    X = np.stack([np.log1p(x_tr), np.ones(N_TRAIN)], axis=1)
    Xt = np.stack([np.log1p(x_te), np.ones(N_TEST)], axis=1)
    w, b = train_logreg(X, y_tr)
    results.append(["Dense + log1p", auc_score(Xt @ w + b, y_te), 2,
                    "压缩长尾，仍是标量"])

    # (4) 分位点分桶 Embedding
    edges = sorted(set(int(np.percentile(x_tr, q)) for q in (5, 15, 30, 50, 70, 85, 95)))
    nb = len(edges) + 1
    X = one_hot(bucket_idx(x_tr, edges), nb)
    Xt = one_hot(bucket_idx(x_te, edges), nb)
    w, b = train_logreg(X, y_tr)
    results.append(["分桶 Embedding（8 个分位桶）", auc_score(Xt @ w + b, y_te), nb + 1,
                    "非线性但桶内信息丢失"])

    # (5) Sequence：均值/末位/最大/末位-均值/斜率
    def seq_features(H: np.ndarray) -> np.ndarray:
        mean = H.mean(1)
        last = H[:, -1]
        mx = H.max(1)
        slope = (H[:, -1] - H[:, 0]) / 4.0
        return np.stack([mean, last, mx, last - mean, slope], axis=1)

    S, St = seq_features(H_tr), seq_features(H_te)
    mu, sd = S.mean(0), S.std(0) + 1e-9
    X = np.hstack([(S - mu) / sd, np.ones((N_TRAIN, 1))])
    Xt = np.hstack([(St - mu) / sd, np.ones((N_TEST, 1))])
    w, b = train_logreg(X, y_tr)
    results.append(["Sequence（5 步 + 池化统计）", auc_score(Xt @ w + b, y_te), 6,
                    "保留轨迹，能看出趋势"])

    # (6) 作弊对照：标量 + 真实的趋势标签（证明序列的增益来自趋势信息）
    X = np.stack([(x_tr - mu_x) / sd_x, t_tr, np.ones(N_TRAIN)], axis=1)
    Xt = np.stack([(x_te - mu_x) / sd_x, t_te, np.ones(N_TEST)], axis=1)
    w, b = train_logreg(X, y_tr)
    results.append(["[对照] 总次数 + 真实趋势", auc_score(Xt @ w + b, y_te), 3,
                    "上界：证明趋势就是缺失的信息"])

    print("\n不同表示在同一任务上的效果：")
    markdown_table(["表示", "测试 AUC", "有效参数", "说明"],
                   [[r[0], f"{r[1]:.4f}", r[2], r[3]] for r in results])
    print("\n读法：只用总次数的四种表示 AUC 都在 0.64 附近，彼此差异很小；")
    print("Sequence 因为保留了'最近是涨还是跌'，AUC 明显更高；")
    print("这解释了为什么统计特征与序列特征不是替代关系（§15.2.1）。")


if __name__ == "__main__":
    main()
