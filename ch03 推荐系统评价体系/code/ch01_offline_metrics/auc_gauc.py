"""对应正文第 3 章 §3.1.2「排序指标」中的 AUC / UAUC / GAUC。

本实现做了什么
---------------
1. 三种等价（或近似等价）的 AUC 算法，互相校验：
   a. 定义式：枚举所有 (正, 负) 样本对，统计 P(s_pos > s_neg)，O(n_pos * n_neg)；
   b. rank 公式（工业界最常用）：
          AUC = ( sum_{i in pos} rank_i - M*(M+1)/2 ) / (M * N)
      其中 rank 为升序排名（从 1 开始），M = 正样本数，N = 负样本数，
      同分样本取平均排名（tie 处理），复杂度 O(n log n)；
   c. ROC 曲线梯形积分：按打分降序扫描，累计 (FPR, TPR) 后求曲线下面积。
2. 按正文式实现 UAUC 与 GAUC：
          UAUC = (1/N) * sum_u AUC_u                （用户等权，强调用户公平性）
          GAUC = sum_u n_u * AUC_u / sum_u n_u      （样本量加权，强调流量真实性）
3. 构造一个「全局 AUC 与 GAUC 结论相反」的例子：高活跃用户（1000 曝光）
   与普通用户（10 曝光）打分量级不同，全局 AUC 被高活用户主导。
   正文原话：普通用户每天只产生 10 次曝光，高活用户每天能产生 1000 次曝光。

注意：UAUC/GAUC 要求用户内同时存在正负样本，否则该用户 AUC 无定义，按工业惯例丢弃。
"""

from __future__ import annotations

import random
from collections import defaultdict


# --------------------------------------------------------------------------
# 三种 AUC 实现
# --------------------------------------------------------------------------
def auc_pairwise(labels: list[int], scores: list[float]) -> float:
    """定义式 AUC = P(s_pos > s_neg)，同分记 0.5。复杂度 O(M*N)。"""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    win = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                win += 1.0
            elif p == n:
                win += 0.5
    return win / (len(pos) * len(neg))


def average_ranks(scores: list[float]) -> list[float]:
    """升序平均排名（1-based），同分取平均。O(n log n)。"""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # 并列区间 [i, j] 的 1-based 排名均值
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc_rank(labels: list[int], scores: list[float]) -> float:
    """rank 公式 AUC，O(n log n)，工业界标准实现。

        AUC = ( sum_{i in pos} rank_i - M(M+1)/2 ) / (M * N)
    """
    m = sum(labels)
    n = len(labels) - m
    if m == 0 or n == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum = sum(r for y, r in zip(labels, ranks) if y == 1)
    return (rank_sum - m * (m + 1) / 2.0) / (m * n)


def auc_roc_trapezoid(labels: list[int], scores: list[float]) -> tuple[float, list[tuple[float, float]]]:
    """按打分降序扫描得到 ROC 曲线，梯形法求面积。返回 (auc, roc 曲线点)。"""
    m = sum(labels)
    n = len(labels) - m
    if m == 0 or n == 0:
        return float("nan"), []
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    tp = fp = 0
    curve = [(0.0, 0.0)]
    area = 0.0
    prev_fpr = prev_tpr = 0.0
    i = 0
    while i < len(pairs):
        j = i
        # 同分必须一起处理，否则 ROC 会受样本顺序影响
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                tp += 1
            else:
                fp += 1
        fpr, tpr = fp / n, tp / m
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        curve.append((fpr, tpr))
        prev_fpr, prev_tpr = fpr, tpr
        i = j + 1
    return area, curve


# --------------------------------------------------------------------------
# UAUC / GAUC
# --------------------------------------------------------------------------
def group_aucs(uids: list[int], labels: list[int], scores: list[float]) -> dict[int, tuple[float, int]]:
    """按用户分组算 AUC。返回 {uid: (auc_u, n_u)}，只保留正负样本齐全的用户。"""
    buckets: dict[int, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for u, y, s in zip(uids, labels, scores):
        buckets[u][0].append(y)
        buckets[u][1].append(s)
    out: dict[int, tuple[float, int]] = {}
    for u, (ys, ss) in buckets.items():
        if 0 < sum(ys) < len(ys):  # 同时含正负样本
            out[u] = (auc_rank(ys, ss), len(ys))
    return out


def uauc(uids: list[int], labels: list[int], scores: list[float]) -> float:
    """UAUC = (1/N) * sum_u AUC_u（用户等权）。"""
    g = group_aucs(uids, labels, scores)
    return sum(a for a, _ in g.values()) / len(g) if g else float("nan")


def gauc(uids: list[int], labels: list[int], scores: list[float]) -> float:
    """GAUC = sum_u n_u * AUC_u / sum_u n_u（样本量加权，亦称 WUAUC）。"""
    g = group_aucs(uids, labels, scores)
    num = sum(a * n for a, n in g.values())
    den = sum(n for _, n in g.values())
    return num / den if den else float("nan")


# --------------------------------------------------------------------------
# 合成数据
# --------------------------------------------------------------------------
def simple_dataset(n: int, signal: float) -> tuple[list[int], list[float]]:
    labels, scores = [], []
    for _ in range(n):
        y = 1 if random.random() < 0.3 else 0
        labels.append(y)
        scores.append(random.gauss(0.0, 1.0) + signal * y)
    return labels, scores


def heavy_user_dataset() -> tuple[list[int], list[int], list[float]]:
    """构造「全局 AUC 高、GAUC 低」的流量结构。

    - 1 个高活跃用户：1000 条曝光，模型在他身上排序能力很强；
    - 60 个普通用户：每人 10 条曝光，模型在他们身上几乎随机；
    - 两类用户的打分量级刻意错开（高活用户整体打分偏高且其正样本打分最高），
      于是跨用户比较时产生大量「虚假的正确序对」，把全局 AUC 抬高。
    """
    uids, labels, scores = [], [], []
    # 高活跃用户 uid=0
    for _ in range(1000):
        y = 1 if random.random() < 0.30 else 0
        uids.append(0)
        labels.append(y)
        scores.append(0.60 + 0.30 * y + random.gauss(0.0, 0.05))
    # 普通用户 uid=1..60：打分整体偏低，用户内排序接近随机
    for u in range(1, 61):
        for _ in range(10):
            y = 1 if random.random() < 0.30 else 0
            uids.append(u)
            labels.append(y)
            scores.append(0.10 + 0.01 * y + random.gauss(0.0, 0.08))
    return uids, labels, scores


def two_model_dataset() -> tuple[list[int], list[int], dict[str, list[float]]]:
    """构造「全局 AUC 与 GAUC 选出不同赢家」的对比数据。

    流量结构：2 个高活用户各 300 条曝光 + 60 个普通用户各 10 条曝光，
    两类人群总样本量各占一半，于是 GAUC 的权重在两类人群间大致 1:1。

    每个用户有自己的真实点击倾向 p_u，打分写成
        score = offset(p_u) + signal * y + noise
    其中 offset 是**跨用户的整体量级**，signal 是**用户内的区分能力**。
    全局 AUC 同时吃这两部分（跨用户序对占了绝大多数），GAUC 只吃 signal。

    - 模型 A：offset 与 p_u 正相关（高倾向用户整体打分更高），用户内只在高活
      用户上区分得好。跨用户序对大量「碰对」，把全局 AUC 顶上去。
    - 模型 B：offset 与 p_u 负相关（量级没校准好），但用户内区分能力普遍更强。
      跨用户序对大量「碰错」，全局 AUC 被拉低。

    线上同一次请求只在单个用户内部排序，因此 B 才是真正更好的模型。
    """
    uids: list[int] = []
    labels: list[int] = []
    score_a: list[float] = []
    score_b: list[float] = []

    # (uid, 曝光数, 真实点击倾向 p_u, 模型A用户内区分强度, 模型B用户内区分强度)
    groups = [(0, 300, 0.45, 2.0, 1.0), (1, 300, 0.40, 2.0, 1.0)]
    for u in range(2, 62):
        groups.append((u, 10, 0.05, 0.05, 1.4))

    for uid, n, p_u, sig_a, sig_b in groups:
        off_a = +4.0 * p_u   # 模型 A：量级与点击倾向同向
        off_b = -4.0 * p_u   # 模型 B：量级与点击倾向反向（跨用户不可比）
        for _ in range(n):
            y = 1 if random.random() < p_u else 0
            uids.append(uid)
            labels.append(y)
            score_a.append(off_a + sig_a * y + random.gauss(0.0, 1.0))
            score_b.append(off_b + sig_b * y + random.gauss(0.0, 1.0))
    return uids, labels, {"模型A": score_a, "模型B": score_b}


def main() -> None:
    random.seed(42)

    print("=" * 78)
    print("§3.1.2 排序指标（一）：AUC / UAUC / GAUC")
    print("=" * 78)

    # ---------- (1) 三种 AUC 实现互相校验 ----------
    labels, scores = simple_dataset(n=600, signal=1.2)
    a1 = auc_pairwise(labels, scores)
    a2 = auc_rank(labels, scores)
    a3, curve = auc_roc_trapezoid(labels, scores)
    print(f"\n[实现校验] 样本 {len(labels)} 条，正样本 {sum(labels)} 条")
    print(f"  定义式  P(s_pos>s_neg)  O(M*N)     = {a1:.6f}")
    print(f"  rank 公式              O(n log n) = {a2:.6f}")
    print(f"  ROC 梯形积分           O(n log n) = {a3:.6f}")
    print(f"  最大两两偏差 = {max(abs(a1-a2), abs(a1-a3), abs(a2-a3)):.2e}  -> 三者等价")
    print(f"  ROC 曲线点数 = {len(curve)}；抽样展示 (FPR, TPR)：")
    step = max(1, len(curve) // 6)
    print("    " + "  ".join(f"({f:.3f},{t:.3f})" for f, t in curve[::step]))

    # ---------- (2) AUC 的极端取值 ----------
    print("\n[取值含义] 正文三条结论的数值验证：")
    perfect_y = [1] * 50 + [0] * 50
    perfect_s = [0.9] * 50 + [0.1] * 50
    print(f"  完美排序      AUC = {auc_rank(perfect_y, perfect_s):.4f}  （接近 1，排序能力强）")
    print(f"  完全反序      AUC = {auc_rank(perfect_y, perfect_s[::-1]):.4f}  （接近 0，排序几乎全错）")
    rnd_y, rnd_s = simple_dataset(n=4000, signal=0.0)
    print(f"  随机打分      AUC = {auc_rank(rnd_y, rnd_s):.4f}  （≈0.5，与随机排序等价）")
    tie_y = [1, 0, 1, 0]
    print(f"  全部同分      AUC = {auc_rank(tie_y, [0.5]*4):.4f}  （tie 平均排名 -> 恰好 0.5）")

    # ---------- (3) 样本不平衡鲁棒性 ----------
    print("\n[不平衡鲁棒性] 固定模型能力，只改变正样本率：")
    print(f"  {'正样本率':>8} | {'样本数':>7} | {'AUC':>7}")
    print("  " + "-" * 30)
    for rate in (0.5, 0.1, 0.01, 0.001):
        ys, ss = [], []
        for _ in range(60000):
            y = 1 if random.random() < rate else 0
            ys.append(y)
            ss.append(random.gauss(0.0, 1.0) + 1.2 * y)
        print(f"  {rate:>8.3f} | {len(ys):>7d} | {auc_rank(ys, ss):>7.4f}")
    print("  >>> AUC 基本不随正负比例变化，这正是它取代 Accuracy 的关键原因。")

    # ---------- (4) AUC 与 GAUC 结论不一致 ----------
    print("\n" + "-" * 78)
    print("[关键演示] 全局 AUC 与 UAUC/GAUC 结论不一致")
    print("-" * 78)
    uids, labels, scores = heavy_user_dataset()
    g = group_aucs(uids, labels, scores)
    print(f"  流量结构：高活用户 uid=0 贡献 {uids.count(0)} 条曝光，"
          f"另有 {len(set(uids))-1} 个普通用户各 10 条，共 {len(uids)} 条")
    print(f"  全局 AUC = {auc_rank(labels, scores):.4f}")
    print(f"  UAUC     = {uauc(uids, labels, scores):.4f}   （{len(g)} 个有效用户等权平均）")
    print(f"  GAUC     = {gauc(uids, labels, scores):.4f}   （按 n_u 加权平均）")
    print(f"  其中 高活用户 AUC_0 = {g[0][0]:.4f} (n_0={g[0][1]})")
    normal = [(a, n) for u, (a, n) in g.items() if u != 0]
    print(f"      普通用户 AUC 均值 = {sum(a for a, _ in normal)/len(normal):.4f} "
          f"(共 {len(normal)} 人，每人 n_u={normal[0][1]})")
    print("  >>> 全局 AUC 很高，但它主要来自『高活用户打分普遍高于普通用户』这种")
    print("      跨用户序对——而线上同一次请求只在同一个用户内部排序，这些序对没有业务意义。")
    print("  >>> UAUC 按用户等权，被大量普通用户拉低；GAUC 按曝光量加权，")
    print("      介于两者之间但仍远低于全局 AUC。若只看全局 AUC 会误判模型质量。")
    print("  >>> 正文口径：UAUC 强调用户公平性，GAUC 强调流量真实性，")
    print("      GAUC 已成为 CTR 预估领域应用最广泛的离线指标之一（亦称 WUAUC）。")

    # ---------- (5) 模型选择结论反转 ----------
    print("\n" + "-" * 78)
    print("[关键演示] 模型选择结论反转：全局 AUC 选 A，GAUC 选 B")
    print("-" * 78)
    uids2, labels2, models = two_model_dataset()
    heavy = sorted(set(uids2))[:2]
    print(f"  流量结构：{len(heavy)} 个高活用户各 {uids2.count(heavy[0])} 条曝光 + "
          f"{len(set(uids2))-len(heavy)} 个普通用户各 10 条，共 {len(uids2)} 条")
    print(f"  {'模型':>6} | {'全局AUC':>8} | {'UAUC':>7} | {'GAUC':>7} | "
          f"{'高活AUC均值':>11} | {'普通AUC均值':>11}")
    print("  " + "-" * 72)
    summary = {}
    for name, sc in models.items():
        g2 = group_aucs(uids2, labels2, sc)
        hv = [a for u, (a, _) in g2.items() if u in heavy]
        nm = [a for u, (a, _) in g2.items() if u not in heavy]
        summary[name] = (auc_rank(labels2, sc), uauc(uids2, labels2, sc),
                         gauc(uids2, labels2, sc))
        print(f"  {name:>6} | {summary[name][0]:>8.4f} | {summary[name][1]:>7.4f} | "
              f"{summary[name][2]:>7.4f} | {sum(hv)/len(hv):>11.4f} | "
              f"{sum(nm)/len(nm):>11.4f}")
    da = summary["模型B"][0] - summary["模型A"][0]
    dg = summary["模型B"][2] - summary["模型A"][2]
    print(f"\n  ΔAUC(B-A)  = {da:+.4f}  -> 全局 AUC 判定 "
          f"{'A 更好' if da < 0 else 'B 更好'}")
    print(f"  ΔGAUC(B-A) = {dg:+.4f}  -> GAUC 判定 "
          f"{'A 更好' if dg < 0 else 'B 更好'}")
    print("  >>> 两个指标给出相反的上线建议。模型 A 的全局 AUC 高，只是因为它把")
    print("      『高点击倾向用户的打分整体抬高』，赚到了大量跨用户序对；模型 B 在")
    print("      每个用户内部排序都更准，但打分量级跨用户不可比，被全局 AUC 惩罚。")
    print("  >>> 线上推荐只在单次请求（同一用户）内排序，跨用户序对不产生任何曝光决策，")
    print("      因此应以 GAUC 为准选择模型 B。这就是 GAUC 成为工业界主流指标的原因。")


if __name__ == "__main__":
    main()
