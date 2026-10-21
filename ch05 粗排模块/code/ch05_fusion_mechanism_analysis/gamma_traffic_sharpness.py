"""
幂指数 γ 与流量陡峭程度(第 5 章 §5.5 多目标融合机制分析)

对应正文概念
------------
正文分析广义乘法公式里的幂指数超参 γ:考虑 Score = (β + α p)^γ,取对数得

    log score = γ log(β + α p)

当 γ > 1 时高分样本被进一步放大,正文给的算例是 0.8² = 0.64、0.2² = 0.04,
「二者差距显著增大」,并给出定义:

    「在多目标融合机制的策略算法中,乘法公式较大的 γ 会强化头部内容的竞争优势,
      使流量进一步向高质量内容集中。」

当 0 < γ < 1 时,0.8^0.5 = 0.894、0.2^0.5 = 0.447,「二者差距被缩小」,流量分布更平缓。
因此 γ 本质上控制着流量分布的陡峭程度(Traffic Sharpness)。

本实现做了什么
--------------
1. 复刻正文算例:γ ∈ {0.5, 1, 2} 下 0.8 与 0.2 的取值、绝对差与比值。
2. 单目标 γ 扫描:用「分数经 softmax/归一化后的流量份额」量化陡峭程度,打印头部份额、
   基尼系数、归一化熵、HHI 等集中度指标随 γ 的变化。
3. 多目标情形:∏_i (β_i + α_i p_i)^{γ_i},只放大某一个目标的 γ_i,观察该目标在 Top-K
   候选里的平均值如何被抬高(γ_i 起到「目标重要性旋钮」的作用)。
4. 排序不变性边界:单目标时 γ > 0 对单调变换不改排序(Spearman = 1.0);多目标时
   不同 γ_i 会真正改变排序 -- 打印两类情形的 Spearman 与 Top-K Jaccard 作对照。

关键公式
--------
    Score = (β + α p)^γ,  log score = γ log(β + α p)
    Score = ∏_i (β_i + α_i p_i)^{γ_i}
    流量份额 share_j = score_j / Σ_j score_j
    Gini = Σ_j (2j - n - 1) share_(j) / n   (share 升序)
    归一化熵 = -Σ share log share / log n,  HHI = Σ share²
"""

from __future__ import annotations

import math
import random

BETA_DEFAULT = 0.0   # 正文算例 Score = p^γ 即 β = 0, α = 1
ALPHA_DEFAULT = 1.0


def powered(p: float, gamma: float,
            beta: float = BETA_DEFAULT, alpha: float = ALPHA_DEFAULT) -> float:
    """(β + α p)^γ。"""
    return (beta + alpha * p) ** gamma


def product_score(ps, alphas, betas, gammas) -> float:
    """∏_i (β_i + α_i p_i)^{γ_i}。"""
    s = 1.0
    for k, p in enumerate(ps):
        s *= (betas[k] + alphas[k] * p) ** gammas[k]
    return s


def shares(scores) -> list[float]:
    tot = sum(scores)
    return [s / tot for s in scores]


def gini(sh) -> float:
    n = len(sh)
    xs = sorted(sh)
    return sum((2 * (j + 1) - n - 1) * x for j, x in enumerate(xs)) / (n * sum(xs))


def norm_entropy(sh) -> float:
    n = len(sh)
    h = -sum(x * math.log(x) for x in sh if x > 0)
    return h / math.log(n)


def hhi(sh) -> float:
    return sum(x * x for x in sh)


def spearman(a, b) -> float:
    n = len(a)
    ra = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: -a[i]))}
    rb = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: -b[i]))}
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def topk_jaccard(a, b, k) -> float:
    sa = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    sb = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(sa & sb) / len(sa | sb)


def mock_pool(n: int, rng: random.Random) -> list[tuple[float, float]]:
    """[MOCK] 合成候选池:p_ctr 走 Beta 形状(多数中低分、少数头部高分),p_cvr 弱相关。"""
    out = []
    for _ in range(n):
        p_ctr = min(0.98, max(0.02, rng.betavariate(2.0, 5.0)))
        p_cvr = min(0.98, max(0.02, 0.35 * p_ctr + 0.65 * rng.betavariate(2.0, 4.0)))
        out.append((p_ctr, p_cvr))
    return out


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    # ---- 1. 复刻正文算例 ----
    print("=" * 94)
    print("正文算例:Score = p^γ 下 0.8 与 0.2 的差距如何被 γ 改写")
    print("=" * 94)
    print(f"{'γ':>8}{'0.8^γ':>12}{'0.2^γ':>12}{'绝对差':>12}{'比值 hi/lo':>14}"
          f"{'log 空间差 γ·log(0.8/0.2)':>26}")
    for g in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        hi, lo = powered(0.8, g), powered(0.2, g)
        print(f"{g:>8.2f}{hi:>12.4f}{lo:>12.4f}{hi - lo:>12.4f}{hi / lo:>14.4f}"
              f"{g * math.log(0.8 / 0.2):>26.4f}")
    print("\nγ=2:0.64 vs 0.04,差距显著放大(与正文一致);γ=0.5:0.894 vs 0.447,差距被压缩。")
    print("log score = γ log(β+αp) 说明 γ 就是对数空间上的线性缩放因子。")

    # ---- 2. 流量陡峭程度 ----
    print("\n" + "=" * 94)
    print("流量陡峭程度(Traffic Sharpness):N=2000 合成候选,share_j = score_j / Σ score")
    print("=" * 94)
    pool = mock_pool(2000, rng)
    p_ctrs = [p for p, _ in pool]
    print(f"{'γ':>8}{'Top1% 流量占比':>16}{'Top5% 流量占比':>16}{'Top20% 流量占比':>17}"
          f"{'Gini':>9}{'归一化熵':>11}{'HHI×1e3':>10}")
    for g in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        sc = [powered(p, g) for p in p_ctrs]
        sh = sorted(shares(sc), reverse=True)
        n = len(sh)
        t1 = sum(sh[: max(1, n // 100)])
        t5 = sum(sh[: max(1, n // 20)])
        t20 = sum(sh[: max(1, n // 5)])
        print(f"{g:>8.2f}{t1:>16.4f}{t5:>16.4f}{t20:>17.4f}"
              f"{gini(sh):>9.4f}{norm_entropy(sh):>11.4f}{hhi(sh) * 1e3:>10.4f}")
    print("\nγ 越大,头部流量占比/Gini/HHI 单调上升,熵单调下降 -- 流量向高质量内容集中;")
    print("γ < 1 时分布被拉平,长尾内容拿到更多流量。")

    # ---- 3. 单目标 γ 的排序不变性 vs 多目标 γ_i 的排序改写 ----
    print("\n" + "=" * 94)
    print("排序影响:单目标改 γ(保序)对比多目标只改某个 γ_i(改序)")
    print("=" * 94)
    base_single = [powered(p, 1.0) for p in p_ctrs]
    print(f"{'情形':>34}{'Spearman vs 基线':>18}{'Top100 Jaccard':>16}")
    for g in (0.5, 2.0, 8.0):
        sc = [powered(p, g) for p in p_ctrs]
        print(f"{f'单目标 p_ctr^{g}':>34}{spearman(base_single, sc):>18.4f}"
              f"{topk_jaccard(base_single, sc, 100):>16.4f}")
    alphas, betas = (1.0, 1.0), (1.0, 1.0)
    base_multi = [product_score(p, alphas, betas, (1.0, 1.0)) for p in pool]
    for g1 in (0.5, 2.0, 4.0):
        sc = [product_score(p, alphas, betas, (g1, 1.0)) for p in pool]
        print(f"{f'双目标 γ=({g1},1.0)':>34}{spearman(base_multi, sc):>18.4f}"
              f"{topk_jaccard(base_multi, sc, 100):>16.4f}")
    print("\n单目标时 (β+αp)^γ 是严格单调变换,Spearman 恒为 1.0,只改变分数尺度不改排序;")
    print("多目标时 γ_i 改变各目标的相对权重,排序被真正改写。")

    # ---- 4. γ_i 作为目标重要性旋钮 ----
    print("\n" + "=" * 94)
    print("γ_i 调节目标重要性:Score = (1+p_ctr)^{γ_1}(1+p_cvr)^{γ_2},看 Top-100 的目标均值")
    print("=" * 94)
    print(f"{'(γ_1, γ_2)':>16}{'Top100 平均 p_ctr':>20}{'Top100 平均 p_cvr':>20}"
          f"{'p_ctr/p_cvr':>14}{'vs (1,1) 重叠':>14}")
    ref = set(sorted(range(len(pool)),
                     key=lambda i: -product_score(pool[i], alphas, betas, (1.0, 1.0)))[:100])
    for gs in ((1.0, 1.0), (3.0, 1.0), (1.0, 3.0), (6.0, 1.0), (1.0, 6.0)):
        order = sorted(range(len(pool)),
                       key=lambda i: -product_score(pool[i], alphas, betas, gs))[:100]
        m_ctr = sum(pool[i][0] for i in order) / 100
        m_cvr = sum(pool[i][1] for i in order) / 100
        print(f"{str(gs):>16}{m_ctr:>20.4f}{m_cvr:>20.4f}{m_ctr / m_cvr:>14.4f}"
              f"{len(ref & set(order)) / 100:>14.2f}")
    print("\n抬高 γ_1 时 Top-100 的平均 p_ctr 上升、平均 p_cvr 下降,反之亦然;")
    print("γ_i 因此既控制流量陡峭程度,也控制各目标在融合里的实际话语权。")
