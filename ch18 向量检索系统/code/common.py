"""
第18章 向量检索系统 —— 公共工具

本模块提供 ch01–ch06 各脚本共用的几件事：

1. 合成向量库：带簇结构的高斯混合数据（模拟"用户/物品 Embedding 聚集"）；
2. 距离计算：L2、余弦、批量平方距离（用 |x|² + |y|² - 2xᵀy 展开，避免构造
   [Q, N, D] 的三维张量）；
3. 暴力精确检索：作为所有 ANN 实验的 Ground Truth；
4. Recall@K、访问节点占比等评估工具。

所有脚本固定 np.random.default_rng(seed)，结果可复现；不依赖 sklearn / faiss，
只用 NumPy（§18 的算法本身就是几十行能讲清楚的，用库反而看不到细节）。
"""

from __future__ import annotations

import time

import numpy as np

SEED = 2026


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def make_vectors(n: int = 2000, dim: int = 16, n_clusters: int = 20,
                 spread: float = 0.35, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    生成带簇结构的合成向量。

    真实 Embedding 并不是均匀散布在整个球面上的：同一类目的物品、兴趣相近的用户
    会在空间里聚成一团。簇结构会显著影响 ANN 的难度——KD 树/球树擅长处理局部
    聚集，随机投影树和 HNSW 则依赖图的连通性。用高斯混合来近似这种结构。

    返回 (X, labels)，X 为 float32、shape=[n, dim]。
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, (n_clusters, dim)) * 2.0
    labels = rng.integers(0, n_clusters, n)
    X = centers[labels] + rng.normal(0.0, spread, (n, dim))
    return X.astype(np.float32), labels


def make_queries(X: np.ndarray, n_queries: int = 100, jitter: float = 0.15,
                 seed: int = SEED + 1) -> np.ndarray:
    """
    查询向量 = 随机选一个库内向量再加小扰动。

    这样每个查询都能在库里找到"语义近邻"，Recall 才有意义；如果查询是纯随机点，
    近邻距离和随机距离差别很小，任何 ANN 都会表现得差不多。
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), n_queries)
    return (X[idx] + rng.normal(0.0, jitter, (n_queries, X.shape[1]))).astype(np.float32)


# --------------------------------------------------------------------------
# 距离
# --------------------------------------------------------------------------
def l2(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(axis=axis))


def cosine(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    num = (a * b).sum(axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-12
    return num / den


def sq_dists(base: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """
    批量平方欧氏距离：||q||² + ||x||² - 2 qᵀx。

    直接写 (q[:, None, :] - base[None, :, :]) ** 2 会构造 [Q, N, D] 的张量，
    128 维、6000 个库向量、200 个查询就是 600 MB，所以按查询逐条展开。
    """
    base_sq = (base * base).sum(axis=1)                    # [N]
    out = np.empty((len(queries), len(base)), dtype=np.float32)
    for i, q in enumerate(queries):
        out[i] = base_sq - 2.0 * (base @ q) + float(q @ q)
    return np.maximum(out, 0.0)


def topk_from_sq(d: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """从平方距离矩阵取 Top-K，返回 (索引, 距离)。"""
    k = min(k, d.shape[1])
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(d, idx, axis=1)
    order = np.argsort(vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return idx, np.sqrt(vals)


def knn_exact(base: np.ndarray, queries: np.ndarray, k: int = 10):
    """暴力精确检索（Ground Truth）。"""
    return topk_from_sq(sq_dists(base, queries), k)


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int | None = None) -> float:
    """
    Recall@K：预测集合与真实 Top-K 集合的平均重合比例。

    pred / truth 形状均为 [n_queries, K]；k 用于在 pred 更宽时（例如 ANN 返回
    search_k 个候选）只比较前 K 个。
    """
    k = k or truth.shape[1]
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(truth) * k)


def timing(fn, *args, repeat: int = 3, **kwargs) -> tuple[float, object]:
    """跑 repeat 次取最快的一次，返回 (秒, 结果)。"""
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, out


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
