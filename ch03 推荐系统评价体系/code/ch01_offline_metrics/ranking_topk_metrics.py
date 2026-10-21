"""对应正文第 3 章 §3.1.2「排序指标」中的 Top-K 列表指标。

本实现做了什么
---------------
按正文公式实现 NDCG@K，并补齐正文在 §3.1.3 小结中列出的、用于召回模型评估的
Top-K 指标：MAP / MRR / HitRate@K / Recall@K / Precision@K。全部为纯标准库实现。

关键公式
---------
NDCG（正文式）：
    DCG@K  = sum_{i=1..K} g(rel_i) / log2(i + 1)        （i 从 1 开始）
    NDCG@K = DCG@K / IDCG@K
其中 g(.) 为增益函数，正文给出两种：指数形式 g(rel)=2^rel（本实现另减 1，
即 2^rel - 1，使 rel=0 时增益为 0，这是 Burges 以来的通行写法，见代码注释）
和线性形式 g(rel)=rel；1/log2(i+1) 为位置折损；IDCG@K 为按真实相关性标签
从高到低理想排序后的最大 DCG。

正文还给出论文常用的 0-based 记号：
    G_R = sum_{r=0..R-1} g(r) D(r) / G_{R,max},  D(r) = 1 / log2(r + 2)
由于 r = i - 1，1/log2(r+2) 与 1/log2(i+1) 完全等价，G_{R,max} 即 IDCG@R。
本实现用 `ndcg_at_k_zero_based` 单独写了这个版本并与 1-based 结果逐位比对。

其余指标（二值相关性，推荐系统常规定义）：
    Precision@K = |relevant ∩ topK| / K
    Recall@K    = |relevant ∩ topK| / |relevant|
    HitRate@K   = 1[ |relevant ∩ topK| > 0 ]（用户级命中，再对用户取平均）
    AP@K        = (1/min(|relevant|,K)) * sum_{i=1..K} 1[rel_i=1] * Precision@i
    MAP@K       = mean_u AP@K(u)
    RR          = 1 / rank(第一个相关物品)，无命中记 0
    MRR         = mean_u RR(u)
"""

from __future__ import annotations

import math
import random


# --------------------------------------------------------------------------
# NDCG：正文主公式
# --------------------------------------------------------------------------
def dcg_at_k(rels: list[float], k: int, gain: str = "exp") -> float:
    """DCG@K = sum_{i=1..K} g(rel_i) / log2(i+1)。

    rels 必须已经按**模型预测分数降序**排列（正文：预测分数决定排序）。
    gain="exp"    -> g(rel) = 2^rel - 1（指数增益，放大高相关性等级的权重）
    gain="linear" -> g(rel) = rel
    """
    total = 0.0
    for i, rel in enumerate(rels[:k], start=1):   # i 从 1 开始
        g = (2.0 ** rel - 1.0) if gain == "exp" else float(rel)
        total += g / math.log2(i + 1)
    return total


def idcg_at_k(rels: list[float], k: int, gain: str = "exp") -> float:
    """IDCG@K：把相关性标签从高到低理想排序后的最大 DCG。"""
    return dcg_at_k(sorted(rels, reverse=True), k, gain)


def ndcg_at_k(rels_in_pred_order: list[float], k: int, gain: str = "exp") -> float:
    """NDCG@K = DCG@K / IDCG@K，取值 [0, 1]，越接近 1 越接近理想排序。"""
    idcg = idcg_at_k(rels_in_pred_order, k, gain)
    if idcg == 0.0:
        return 0.0   # 候选里全是 rel=0，排序无从评价
    return dcg_at_k(rels_in_pred_order, k, gain) / idcg


def ndcg_at_k_zero_based(rels_in_pred_order: list[float], k: int, gain: str = "exp") -> float:
    """正文给出的 0-based 论文记号版：D(r) = 1/log2(r+2)，r = 0..R-1。

    与 `ndcg_at_k` 数学等价，这里单独实现以便逐位验证「r=i-1 时两种折损等价」。
    """
    def _g(rel: float) -> float:
        return (2.0 ** rel - 1.0) if gain == "exp" else float(rel)

    g_r = sum(_g(rel) * (1.0 / math.log2(r + 2))
              for r, rel in enumerate(rels_in_pred_order[:k]))
    ideal = sorted(rels_in_pred_order, reverse=True)
    g_max = sum(_g(rel) * (1.0 / math.log2(r + 2))
                for r, rel in enumerate(ideal[:k]))
    return g_r / g_max if g_max else 0.0


def rank_by_score(scores: list[float], rels: list[float]) -> list[float]:
    """按预测分数降序重排相关性标签，得到「预测排序下的 rel 序列」。

    正文强调的链路：pXTR 预测分数 -> Rank 排序 -> 真实相关性标签 rel
                   -> Gain x Discount -> DCG -> NDCG
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [rels[i] for i in order]


# --------------------------------------------------------------------------
# 召回模型常用的 Top-K 指标（二值相关性）
# --------------------------------------------------------------------------
def precision_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for it in ranked_items[:k] if it in relevant)
    return hits / k if k else 0.0


def recall_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for it in ranked_items[:k] if it in relevant)
    return hits / len(relevant)


def hit_rate_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    """单用户命中与否（0/1）；对用户取平均即 HitRate@K。"""
    return 1.0 if any(it in relevant for it in ranked_items[:k]) else 0.0


def average_precision_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    """AP@K：每个命中位置上的 Precision@i 求和，再除以 min(|relevant|, K)。"""
    if not relevant:
        return 0.0
    hits = 0
    acc = 0.0
    for i, it in enumerate(ranked_items[:k], start=1):
        if it in relevant:
            hits += 1
            acc += hits / i          # 此位置的 Precision@i
    return acc / min(len(relevant), k)


def reciprocal_rank(ranked_items: list[int], relevant: set[int]) -> float:
    """RR = 1 / 第一个相关物品的排名；全列表无命中记 0。"""
    for i, it in enumerate(ranked_items, start=1):
        if it in relevant:
            return 1.0 / i
    return 0.0


# --------------------------------------------------------------------------
# 合成数据
# --------------------------------------------------------------------------
def make_user_lists(n_users: int, n_cand: int, n_relevant: int, quality: float):
    """为每个用户造一份候选集。

    quality 越大，模型打分与真实相关性越一致（quality=0 为随机排序）。
    返回 [(ranked_items, relevant_set, rels_in_pred_order)]。
    rel 取 0/1/2/3 四级：0 无行为，1 点击，2 长播，3 点赞+长播。
    """
    out = []
    for _ in range(n_users):
        items = list(range(n_cand))
        relevant = set(random.sample(items, n_relevant))
        rels = {}
        for it in items:
            if it in relevant:
                rels[it] = random.choice([1, 1, 2, 3])   # 相关物品给多级标签
            else:
                rels[it] = 0
        scores = {it: quality * rels[it] + random.gauss(0.0, 1.0) for it in items}
        ranked = sorted(items, key=lambda it: -scores[it])
        out.append((ranked, relevant, [float(rels[it]) for it in ranked]))
    return out


def main() -> None:
    random.seed(42)

    print("=" * 78)
    print("§3.1.2 排序指标（二）：NDCG / MAP / MRR / HitRate@K / Recall@K")
    print("=" * 78)

    # ---------- (1) NDCG 手算例子，逐位展示 Gain x Discount ----------
    print("\n[NDCG 逐位展开] 一个 6 物品的小例子（rel 为真实相关性标签）")
    pred_scores = [0.91, 0.72, 0.65, 0.40, 0.33, 0.11]
    true_rels = [3.0, 0.0, 2.0, 1.0, 0.0, 1.0]
    ordered = rank_by_score(pred_scores, true_rels)
    print(f"  预测分数 = {pred_scores}")
    print(f"  真实 rel = {[int(r) for r in true_rels]}（已按分数降序，顺序不变）")
    print(f"  {'位置i':>5} | {'rel':>4} | {'g=2^rel-1':>10} | {'1/log2(i+1)':>12} | {'贡献':>8}")
    print("  " + "-" * 56)
    for i, rel in enumerate(ordered, start=1):
        g = 2.0 ** rel - 1.0
        d = 1.0 / math.log2(i + 1)
        print(f"  {i:>5d} | {int(rel):>4d} | {g:>10.3f} | {d:>12.4f} | {g*d:>8.4f}")
    k = 6
    print(f"  DCG@{k}  = {dcg_at_k(ordered, k):.4f}")
    print(f"  理想排序 rel = {[int(r) for r in sorted(true_rels, reverse=True)]}")
    print(f"  IDCG@{k} = {idcg_at_k(ordered, k):.4f}")
    print(f"  NDCG@{k} = {ndcg_at_k(ordered, k):.4f}")

    print("\n  [记号等价性] 正文 0-based 论文记号 D(r)=1/log2(r+2) 与 1-based 1/log2(i+1)：")
    for kk in (1, 2, 3, 4, 5, 6):
        a = ndcg_at_k(ordered, kk)
        b = ndcg_at_k_zero_based(ordered, kk)
        print(f"    K={kk}: 1-based={a:.6f}  0-based={b:.6f}  |差|={abs(a-b):.2e}")

    print("\n  [增益函数对比] 指数增益 2^rel-1 更强调高相关性等级：")
    for kk in (1, 3, 6):
        print(f"    NDCG@{kk}: exp={ndcg_at_k(ordered, kk, 'exp'):.4f}  "
              f"linear={ndcg_at_k(ordered, kk, 'linear'):.4f}")

    # ---------- (2) NDCG 的极端取值 ----------
    print("\n[NDCG 取值] 理想排序 / 完全反序 / 随机排序：")
    rels = [3.0, 2.0, 1.0, 1.0, 0.0, 0.0]
    print(f"  理想排序 NDCG@6 = {ndcg_at_k(rels, 6):.4f}")
    print(f"  完全反序 NDCG@6 = {ndcg_at_k(rels[::-1], 6):.4f}")
    shuffled = rels[:]
    random.shuffle(shuffled)
    print(f"  随机一次 NDCG@6 = {ndcg_at_k(shuffled, 6):.4f}  (序列 {[int(r) for r in shuffled]})")

    # ---------- (3) 位置折损的作用：只换头部两位 ----------
    print("\n[头部敏感性] 把最高相关的物品从位置 1 挪到位置 K，NDCG@6 变化：")
    for pos in range(6):
        seq = [0.0] * 6
        seq[pos] = 3.0
        print(f"  rel=3 在位置 {pos+1}: NDCG@6 = {ndcg_at_k(seq, 6):.4f}")
    print("  >>> 位置折损让头部位置的收益远大于尾部，这是 NDCG 比 AUC 更贴合")
    print("      Top-K 展示场景（首页推荐、搜索排序、重排）的原因。")

    # ---------- (4) 多用户 Top-K 指标汇总 ----------
    print("\n" + "-" * 78)
    print("[多用户汇总] 100 用户 x 200 候选，每人 8 个相关物品")
    print("-" * 78)
    for quality, tag in ((0.0, "随机模型"), (0.8, "弱模型"), (2.5, "强模型")):
        data = make_user_lists(n_users=100, n_cand=200, n_relevant=8, quality=quality)
        print(f"\n  {tag} (quality={quality})")
        print(f"  {'K':>4} | {'Prec@K':>7} | {'Recall@K':>9} | {'HitRate@K':>10} | "
              f"{'MAP@K':>7} | {'NDCG@K':>7}")
        print("  " + "-" * 62)
        for kk in (5, 10, 20, 50):
            p = sum(precision_at_k(r, rel, kk) for r, rel, _ in data) / len(data)
            rc = sum(recall_at_k(r, rel, kk) for r, rel, _ in data) / len(data)
            hr = sum(hit_rate_at_k(r, rel, kk) for r, rel, _ in data) / len(data)
            mp = sum(average_precision_at_k(r, rel, kk) for r, rel, _ in data) / len(data)
            nd = sum(ndcg_at_k(rels_o, kk) for _, _, rels_o in data) / len(data)
            print(f"  {kk:>4d} | {p:>7.4f} | {rc:>9.4f} | {hr:>10.4f} | "
                  f"{mp:>7.4f} | {nd:>7.4f}")
        mrr = sum(reciprocal_rank(r, rel) for r, rel, _ in data) / len(data)
        print(f"  MRR (全列表) = {mrr:.4f}")

    print("\n  >>> 正文口径：GAUC 衡量整体排序能力，NDCG 评估 Top-K 推荐质量，")
    print("      Precision / Recall / F1（及这里的 HitRate@K、MAP、MRR）更多用于召回模型评估。")
    print("  >>> 注意 Recall@K 随 K 单调升、Precision@K 随 K 单调降，")
    print("      召回阶段通常在较大的 K 上看 Recall@K（允许脏，不允许漏）。")


if __name__ == "__main__":
    main()
