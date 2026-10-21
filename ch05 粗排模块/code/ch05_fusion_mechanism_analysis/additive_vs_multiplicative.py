"""
加法公式与乘法公式的机制对比(第 5 章 §5.5 多目标融合机制分析)

对应正文概念
------------
正文 §5.5 以两目标为例,对比两种融合机制:

    加法:  score = w_1 p_ctr + w_2 p_cvr
    乘法:  score = p_ctr^α p_cvr^β

并用表 ctr_cvr_example 的三个 Item(A=(0.9,0.1)、B=(0.5,0.5)、C=(0.1,0.9))说明:
取 w_1=w_2=1 时加法下 score_A = score_B = score_C = 1.0 完全同分;而乘法下
score_A = 0.09、score_B = 0.25、score_C = 0.09,B 分数最大。由此得到定义:

    「加法融合是一种补偿型(Compensatory)排序机制,高目标可以补偿低目标;
      而乘法融合是一种短板约束型(Bottleneck-Constrained)排序机制,
      任何一个目标过低都会显著拉低最终得分。」

正文还做了等值线分析:
  - p_ctr + p_cvr = 1 在加法下是一条直线,在乘法下 p_ctr·p_cvr = 1 退化为孤点 (1.0,1.0);
  - 取 c = 0.5 时,加法下单因子最大只需 0.5,乘法下单因子最小要 0.5 且另一个必须取 1.0,
    说明「达到相同 c 的目标下,乘法公式对融合因子的取值大小要求更高」;
  - (0.9,0.1) 与 (0.99,0.01) 在加法下同一条等值线,在乘法下后者的等值线更接近 0,
    即乘法对单因子极端取值有很强的惩罚 -- 短板效应(Bottleneck Effect);
  - 乘法公式可看成对数空间的加法公式,而 pxtr ∈ [0,1] 做对数变换会显著放大区间内差异,
    所以电商场景常把 CTR 与 CVR 相乘(需要两者都高),而不是相加。

本实现做了什么
--------------
1. 复刻正文表 ctr_cvr_example 的三个 Item,逐列打印加法/乘法分数,核对 1.0/1.0/1.0 与
   0.09/0.25/0.09。
2. 等值线数值表:对给定 c,分别解出加法等值线 p_cvr = c - p_ctr 与乘法等值线
   p_cvr = c / p_ctr,打印同一 p_ctr 下两者所需的 p_cvr,量化「乘法要求更高」。
3. 极端点惩罚表:沿 p_ctr + p_cvr = 1 这条加法等值线扫描,打印乘积值随「偏科程度」
   衰减的全过程,并给出正文点 (0.9,0.1) 与 (0.99,0.01) 的具体对比。
4. 对数空间验证:log(p_ctr p_cvr) = log p_ctr + log p_cvr,打印 pxtr 与 log pxtr 的
   差异放大倍数,说明乘法是对数空间的加法。
5. 偏科生 vs 六边形战士:在合成候选集上统计两种机制选出的 Top-K 的「均衡度」
   (min(p)/max(p)、两目标标准差)与重叠度,验证正文结论。

关键公式
--------
    score_add  = w_1 p_ctr + w_2 p_cvr
    score_mul  = p_ctr^α p_cvr^β
    log score_mul = α log p_ctr + β log p_cvr
"""

from __future__ import annotations

import math
import random

W1, W2 = 1.0, 1.0        # 正文设定 w_1 = w_2 = 1
ALPHA, BETA = 1.0, 1.0   # 正文最简乘法公式 α = β = 1


def score_add(p_ctr: float, p_cvr: float, w1: float = W1, w2: float = W2) -> float:
    """加法融合 score = w_1 p_ctr + w_2 p_cvr。"""
    return w1 * p_ctr + w2 * p_cvr


def score_mul(p_ctr: float, p_cvr: float, a: float = ALPHA, b: float = BETA) -> float:
    """乘法融合 score = p_ctr^α p_cvr^β。"""
    return p_ctr ** a * p_cvr ** b


def mock_candidates(n: int, rng: random.Random) -> list[tuple[int, float, float]]:
    """[MOCK] 合成候选集:一半「偏科型」(一高一低),一半「均衡型」。"""
    out = []
    for i in range(n):
        if i % 2 == 0:                      # 偏科型
            hi = rng.uniform(0.70, 0.99)
            lo = rng.uniform(0.01, 0.30)
            p_ctr, p_cvr = (hi, lo) if rng.random() < 0.5 else (lo, hi)
        else:                               # 均衡型
            mid = rng.uniform(0.35, 0.62)
            p_ctr = min(0.99, max(0.01, mid + rng.gauss(0, 0.04)))
            p_cvr = min(0.99, max(0.01, mid + rng.gauss(0, 0.04)))
        out.append((i, round(p_ctr, 4), round(p_cvr, 4)))
    return out


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    # ---- 1. 复刻正文表 ctr_cvr_example ----
    print("=" * 92)
    print("正文表 ctr_cvr_example:加法 w1=w2=1,乘法 α=β=1")
    print("=" * 92)
    table = [("A", 0.9, 0.1), ("B", 0.5, 0.5), ("C", 0.1, 0.9)]
    print(f"{'Item':>6}{'p_ctr':>9}{'p_cvr':>9}{'加法 score':>13}{'乘法 score':>13}"
          f"{'加法排名':>10}{'乘法排名':>10}")
    adds = [score_add(a, b) for _, a, b in table]
    muls = [score_mul(a, b) for _, a, b in table]
    for idx, (name, a, b) in enumerate(table):
        ra = 1 + sum(1 for v in adds if v > adds[idx] + 1e-12)
        rm = 1 + sum(1 for v in muls if v > muls[idx] + 1e-12)
        print(f"{name:>6}{a:>9.2f}{b:>9.2f}{adds[idx]:>13.4f}{muls[idx]:>13.4f}"
              f"{ra:>10}{rm:>10}")
    print(f"\n加法下三者同分(都等于 {adds[0]:.1f}),完全无法区分;"
          f"乘法下 B 最高({muls[1]:.2f} > {muls[0]:.2f} = {muls[2]:.2f})。")

    # ---- 2. 等值线对比 ----
    print("\n" + "=" * 92)
    print("等值线对比:达到同一个目标值 c 时,两种公式对因子取值的要求")
    print("=" * 92)
    for c in (1.0, 0.5):
        print(f"\nc = {c}:  加法等值线 p_cvr = c - p_ctr   |   乘法等值线 p_cvr = c / p_ctr")
        print(f"{'p_ctr':>9}{'加法所需 p_cvr':>18}{'乘法所需 p_cvr':>18}{'是否可行(<=1)':>16}")
        for p in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            add_need = c - p
            mul_need = c / p
            add_s = f"{add_need:.3f}" if add_need >= 0 else "无需(已达标)"
            feasible = "可行" if mul_need <= 1.0 + 1e-12 else "不可行"
            print(f"{p:>9.2f}{add_s:>18}{mul_need:>18.3f}{feasible:>16}")
    print("\nc=1.0 时乘法只有 p_ctr=p_cvr=1.0 这一个孤点可行(正文:退化成孤点);")
    print("c=0.5 时加法单因子最大只要 0.5,乘法则要求单因子最小 0.5 且另一个取到 1.0。")

    # ---- 3. 极端取值的惩罚:沿加法等值线 p_ctr + p_cvr = 1 扫描 ----
    print("\n" + "=" * 92)
    print("短板效应:沿加法等值线 p_ctr + p_cvr = 1.0 扫描(加法分恒为 1.0)")
    print("=" * 92)
    print(f"{'p_ctr':>9}{'p_cvr':>9}{'加法 score':>13}{'乘法 score':>13}"
          f"{'乘法/峰值':>12}{'min/max 均衡度':>16}")
    peak = score_mul(0.5, 0.5)
    for p in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999):
        q = 1.0 - p
        m = score_mul(p, q)
        print(f"{p:>9.3f}{q:>9.3f}{score_add(p, q):>13.4f}{m:>13.6f}"
              f"{m / peak:>12.4f}{min(p, q) / max(p, q):>16.4f}")
    a1, a2 = score_mul(0.9, 0.1), score_mul(0.99, 0.01)
    print(f"\n正文两点:(0.9,0.1) 乘法分 {a1:.4f},(0.99,0.01) 乘法分 {a2:.4f},"
          f"后者只有前者的 {a2 / a1:.2f} 倍 -- 更接近 0;")
    print(f"而加法下两点同在 score = {score_add(0.9, 0.1):.1f} 的等值线上,完全无差别。")

    # ---- 4. 乘法 = 对数空间的加法 ----
    print("\n" + "=" * 92)
    print("乘法是对数空间的加法:log(p_ctr·p_cvr) = log p_ctr + log p_cvr")
    print("=" * 92)
    print(f"{'p':>9}{'log p':>11}{'相邻比值 Δp':>14}{'相邻比值 Δlog p':>18}{'差异放大倍数':>14}")
    grid = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
    prev = None
    for p in grid:
        lp = math.log(p)
        if prev is None:
            print(f"{p:>9.3f}{lp:>11.4f}{'-':>14}{'-':>18}{'-':>14}")
        else:
            dp, dl = prev[0] - p, abs(lp - prev[1])
            print(f"{p:>9.3f}{lp:>11.4f}{dp:>14.4f}{dl:>18.4f}{dl / dp:>14.2f}")
        prev = (p, lp)
    chk_l = math.log(score_mul(0.3, 0.6))
    chk_r = math.log(0.3) + math.log(0.6)
    print(f"\n校验 log(0.3×0.6) = {chk_l:.6f},log0.3 + log0.6 = {chk_r:.6f},"
          f"相等? {abs(chk_l - chk_r) < 1e-12}")
    print("pxtr 越小,同样的数值间隔在对数空间被放大得越厉害 -- 这就是乘法惩罚低分因子的根源。")

    # ---- 5. 偏科生 vs 六边形战士 ----
    print("\n" + "=" * 92)
    print("偏科生 vs 六边形战士:两种机制选出的 Top-K 结构对比(N=600 合成候选)")
    print("=" * 92)
    cands = mock_candidates(600, rng)
    print(f"{'K':>5}{'机制':>8}{'偏科型占比':>12}{'均衡型占比':>12}"
          f"{'平均 min/max':>14}{'平均|p_ctr-p_cvr|':>18}{'Top-K 重叠':>12}")
    for K in (10, 30, 100):
        sel = {}
        for tag, fn in (("加法", score_add), ("乘法", score_mul)):
            top = sorted(cands, key=lambda t: -fn(t[1], t[2]))[:K]
            sel[tag] = {i for i, _, _ in top}
            skew = sum(1 for i, _, _ in top if i % 2 == 0) / K
            bal = sum(min(a, b) / max(a, b) for _, a, b in top) / K
            gap = sum(abs(a - b) for _, a, b in top) / K
            print(f"{K:>5}{tag:>8}{skew:>12.3f}{1 - skew:>12.3f}"
                  f"{bal:>14.4f}{gap:>18.4f}", end="")
            if tag == "乘法":
                ov = len(sel["加法"] & sel["乘法"]) / K
                print(f"{ov:>12.3f}")
            else:
                print(f"{'':>12}")
    print("\n加法 Top-K 里偏科型占比高、min/max 均衡度低;乘法 Top-K 几乎全是均衡型。")
    print("正文定义:加法=补偿型(Compensatory),乘法=短板约束型(Bottleneck-Constrained)。")
