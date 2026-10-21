"""
对应正文第 4 章 §4.1「为什么需要召回」。

正文论点：排序模型理论上可以给全库物品打分并取 Top-K，得到全局最优序；
但工业级推荐系统整条链路只有数百毫秒预算，而内容库规模是千万级、亿级，
因此必须用「级联式（Cascade）架构」把候选规模先压缩到万级甚至千级。

本脚本把这段文字变成两组可验证的数字：

实验一｜算力预算推演（延迟数字全部来自真实测量）
    真实测出三级打分函数的单物品耗时（numpy 前向）：
        召回 = 纯内积；粗排 = 单隐层 MLP；精排 = 3 层 MLP + 交叉项
    然后推演
        T_full     = N * c_rank
        T_cascade  = N * c_retrieve + M_prerank * c_prerank + M_rank * c_rank
    对照 300ms 链路预算，看哪种方案能落地。

实验二｜级联漏斗的效果损失
    把精排打分视为「真值」，上游各级建模为真值的带噪近似：
        s_stage = s_rank + sigma_stage * eps,  eps ~ N(0, 1)
    （sigma 越大代表该级与精排的一致性越差，这正是第 22 章链路一致性的议题。）
    统计级联 Top-K 对全量精排 Top-K 的
        Recall@K = |TopK(cascade) ∩ TopK(full scan)| / K
    说明「漏斗开口越大 / 上下游越一致，损失越小」，而精排算力只需全量的百分之几。

注意：实验二的 sigma 是显式的建模假设（为了让漏斗各级相关，符合工业实际），
      不是正文给出的数值；正文只给出候选规模量级（50000 -> 5000 -> 500）。
"""

import random
import time

import numpy as np


# ---------------------------------------------------------------- 打分函数族
def make_rank_model(dim: int, hidden: int, rng: np.random.Generator):
    """精排级打分：3 层 MLP，输入是 [用户向量, 物品向量, 交叉项] 的拼接。"""
    w1 = rng.normal(0, 0.3, size=(3 * dim, hidden))
    w2 = rng.normal(0, 0.3, size=(hidden, hidden // 2))
    w3 = rng.normal(0, 0.3, size=(hidden // 2, 1))

    def score(u: np.ndarray, items: np.ndarray) -> np.ndarray:
        n = items.shape[0]
        ub = np.broadcast_to(u, (n, u.shape[0]))
        x = np.concatenate([ub, items, ub * items], axis=1)  # 交叉项不可分解为内积
        h = np.maximum(x @ w1, 0.0)
        h = np.maximum(h @ w2, 0.0)
        return (h @ w3).ravel()

    return score


def make_prerank_model(dim: int, hidden: int, rng: np.random.Generator):
    """粗排级打分：单隐层小 MLP，代价远低于精排。"""
    w1 = rng.normal(0, 0.3, size=(2 * dim, hidden))
    w2 = rng.normal(0, 0.3, size=(hidden, 1))

    def score(u: np.ndarray, items: np.ndarray) -> np.ndarray:
        n = items.shape[0]
        ub = np.broadcast_to(u, (n, u.shape[0]))
        x = np.concatenate([ub, items], axis=1)
        h = np.maximum(x @ w1, 0.0)
        return (h @ w2).ravel()

    return score


def retrieve_score(u: np.ndarray, items: np.ndarray) -> np.ndarray:
    """召回级打分：纯内积（线上可交给 ANN 索引，不必逐物品前向）。"""
    return items @ u


# ---------------------------------------------------------------- 单次打分耗时
def measure_cost_us(fn, u: np.ndarray, items: np.ndarray, repeat: int = 3) -> float:
    """返回单物品平均耗时（微秒），取多次测量的最小值以降低抖动。"""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(u, items)
        best = min(best, time.perf_counter() - t0)
    return best / items.shape[0] * 1e6


def budget_table(n_corpus: int, c_retrieve: float, c_prerank: float, c_rank: float,
                 m_prerank: int, m_rank: int, budget_ms: float = 300.0) -> None:
    """按实测单物品代价推演全量打分与级联链路的延迟。"""
    t_full = n_corpus * c_rank / 1000.0  # ms
    t_r = n_corpus * c_retrieve / 1000.0
    t_p = m_prerank * c_prerank / 1000.0
    t_k = m_rank * c_rank / 1000.0
    t_cascade = t_r + t_p + t_k
    print(f"  内容库规模 N = {n_corpus:,}")
    print(f"  方案 A 全量精排:  {n_corpus:,} × {c_rank:.3f}us = {t_full:,.1f} ms"
          f"   -> {'超出' if t_full > budget_ms else '满足'} {budget_ms:.0f}ms 预算")
    print("  方案 B 级联链路:")
    print(f"    召回  {n_corpus:>11,} × {c_retrieve:.4f}us = {t_r:8.2f} ms  (线性扫描上界；ANN 可降到对数级)")
    print(f"    粗排  {m_prerank:>11,} × {c_prerank:.4f}us = {t_p:8.2f} ms")
    print(f"    精排  {m_rank:>11,} × {c_rank:.4f}us = {t_k:8.2f} ms")
    print(f"    合计 = {t_cascade:.2f} ms"
          f"   -> {'超出' if t_cascade > budget_ms else '满足'} {budget_ms:.0f}ms 预算")
    print(f"  加速比 = {t_full / t_cascade:,.1f}x")


# ---------------------------------------------------------------- 效果损失
def cascade_recall(truth: np.ndarray, rng: np.random.Generator,
                   m_retrieve: int, m_prerank: int, top_k: int,
                   sigma_retrieve: float, sigma_prerank: float) -> float:
    """
    在一次请求上跑「召回 -> 粗排 -> 精排」漏斗，返回 Recall@top_k。

    truth: 全量精排分（作为理想排序的依据）
    上游各级分 = truth + sigma * N(0,1)，sigma 越大表示与精排越不一致。
    """
    n = truth.shape[0]
    scale = truth.std()
    ideal = set(np.argsort(-truth)[:top_k].tolist())

    s_ret = truth + sigma_retrieve * scale * rng.normal(size=n)
    cand1 = np.argsort(-s_ret)[:m_retrieve]

    s_pre = truth[cand1] + sigma_prerank * scale * rng.normal(size=m_retrieve)
    cand2 = cand1[np.argsort(-s_pre)[:m_prerank]]

    final = cand2[np.argsort(-truth[cand2])[:top_k]]
    return len(ideal & set(final.tolist())) / top_k


def main() -> None:
    random.seed(42)
    rng = np.random.default_rng(42)
    dim = 32

    rank_fn = make_rank_model(dim, hidden=128, rng=rng)
    prerank_fn = make_prerank_model(dim, hidden=16, rng=rng)

    print("=" * 84)
    print("§4.1 实验一：算力预算推演（单物品打分代价为真实测量值）")
    print("=" * 84)
    probe_items = rng.normal(0, 1, size=(20000, dim))
    probe_u = rng.normal(0, 1, size=dim)
    c_rank = measure_cost_us(rank_fn, probe_u, probe_items)
    c_prerank = measure_cost_us(prerank_fn, probe_u, probe_items)
    c_retrieve = measure_cost_us(retrieve_score, probe_u, probe_items)
    print(f"  实测单物品耗时: 召回(内积) {c_retrieve:.4f}us | "
          f"粗排(单隐层MLP) {c_prerank:.4f}us | 精排(3层MLP) {c_rank:.4f}us")
    print(f"  代价比 精排/粗排 = {c_rank / c_prerank:.1f}x,  精排/召回 = {c_rank / c_retrieve:.1f}x")
    print()
    for n in (1_000_000, 10_000_000, 100_000_000):
        budget_table(n, c_retrieve, c_prerank, c_rank, m_prerank=5000, m_rank=500)
        print()
    print("  注意 N 越大，级联链路的耗时几乎全部由「召回侧线性扫描」贡献（上表 N=1e8 一行最明显）。")
    print("  这正是第 18 章向量检索系统（HNSW/IVF 等 ANN 索引）存在的理由：把 O(N) 降到近似对数级。")
    print()

    print("=" * 84)
    print("§4.1 实验二：级联漏斗的效果损失（全库 50000，Top-500，20 次请求平均）")
    print("=" * 84)
    n_corpus, top_k, n_req = 50000, 500, 20
    items = rng.normal(0, 1, size=(n_corpus, dim))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    users = rng.normal(0, 1, size=(n_req, dim))
    users /= np.linalg.norm(users, axis=1, keepdims=True)
    truths = [rank_fn(users[i], items) for i in range(n_req)]

    print(f"{'召回规模':>9} {'粗排规模':>9} {'一致性(σ召回/σ粗排)':>22} "
          f"{'Recall@500':>12} {'精排打分占全量':>15}")
    print("-" * 74)
    for m_ret, m_pre, s_ret, s_pre in (
        (5000, 500, 0.60, 0.30),
        (10000, 1000, 0.60, 0.30),
        (20000, 2000, 0.60, 0.30),
        (10000, 1000, 0.30, 0.15),
        (10000, 1000, 1.20, 0.60),
    ):
        rec = float(np.mean([
            cascade_recall(truths[i], rng, m_ret, m_pre, top_k, s_ret, s_pre)
            for i in range(n_req)
        ]))
        print(f"{m_ret:>9,} {m_pre:>9,} {f'{s_ret:.2f} / {s_pre:.2f}':>22} "
              f"{rec:>12.4f} {m_pre / n_corpus:>14.2%}")
    print()
    print("  结论：")
    print("  1) 同样的一致性水平下，漏斗开口越大 Recall 越高，但精排算力同比上升；")
    print("  2) 把 σ 从 0.60/0.30 降到 0.30/0.15（上下游更一致）比把召回规模翻倍更划算；")
    print("  3) 任何一档的精排打分次数都只有全量方案的百分之几 —— 正文所说的效果/效率平衡。")


if __name__ == "__main__":
    main()
