"""对应正文第 3 章 §3.3.2「A/B实验平台」中的统计检验部分。

本实现做了什么
---------------
1. AA 实验（空跑实验）：两组都用同一策略，检验分流系统与指标计算是否存在系统性偏差。
   重复 2000 次 AA 实验，统计 p-value 的分布——理论上应服从 U(0,1)，
   且 alpha=0.05 下的假阳性率应接近 5%；
2. 双样本 t 检验（Welch's t-test）：返回 t 统计量、自由度、p-value、
   以及正文强调的**置信区间（Confidence Interval）**；
3. 比例类指标（CTR）的两比例 z 检验与相对提升的置信区间（delta 方法）；
4. 所需样本量估算：给定 MDE（最小可检测效应）、alpha、power，
   反推每组需要多少样本——解释正文的「实验流量过小或周期过短会导致真实收益无法被检测」；
5. 多重检验校正（Multiple Testing Correction）：Bonferroni 与
   Benjamini-Hochberg FDR，演示同时看 20 个指标时假阳性是怎么冒出来的。

关键公式
---------
    Welch t     = (x1 - x2) / sqrt(s1^2/n1 + s2^2/n2)
    Welch df    = (s1^2/n1 + s2^2/n2)^2 / [ (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) ]
    95% CI      = (x1 - x2) ± t_{0.975, df} * sqrt(s1^2/n1 + s2^2/n2)
    两比例 z    = (p1 - p2) / sqrt( p_pool*(1-p_pool)*(1/n1 + 1/n2) )
    样本量      n_per_group = 2 * (z_{1-a/2} + z_{power})^2 * sigma^2 / delta^2
    Bonferroni  拒绝 H0 当 p_i < alpha / m
    BH-FDR      按 p 升序，找最大的 i 使 p_(i) <= i/m * alpha，拒绝前 i 个
"""

from __future__ import annotations

import math
import random


# --------------------------------------------------------------------------
# 分布函数（纯标准库实现）
# --------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """标准正态分位数，Acklam 有理逼近（|误差| < 1.15e-9）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _betacf(a: float, b: float, x: float) -> float:
    """不完全 beta 函数的连分式（Numerical Recipes）。"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log(1 - x) + a * math.log(x)) * _betacf(b, a, 1 - x) / b


def t_sf(t: float, df: float) -> float:
    """t 分布上尾概率 P(T > t)。"""
    x = df / (df + t * t)
    half = 0.5 * betainc(df / 2.0, 0.5, x)
    return half if t > 0 else 1.0 - half


def t_ppf(p: float, df: float) -> float:
    """t 分布分位数，二分法（p in (0,1)）。"""
    lo, hi = -300.0, 300.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1.0 - t_sf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------
# 检验
# --------------------------------------------------------------------------
def welch_ttest(x: list[float], y: list[float], alpha: float = 0.05) -> dict:
    """Welch 双样本 t 检验（不假设方差齐性）。x=Exp 组，y=Base 组。"""
    n1, n2 = len(x), len(y)
    m1, m2 = sum(x) / n1, sum(y) / n2
    v1 = sum((a - m1) ** 2 for a in x) / (n1 - 1)
    v2 = sum((a - m2) ** 2 for a in y) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se if se > 0 else 0.0
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = 2.0 * t_sf(abs(t), df)
    tcrit = t_ppf(1 - alpha / 2, df)
    diff = m1 - m2
    return {
        "mean_exp": m1, "mean_base": m2, "diff": diff,
        "lift_pct": 100.0 * diff / m2 if m2 else float("nan"),
        "t": t, "df": df, "p": p, "se": se,
        "ci_low": diff - tcrit * se, "ci_high": diff + tcrit * se,
        "significant": p < alpha,
    }


def two_proportion_ztest(c1: int, n1: int, c2: int, n2: int,
                         alpha: float = 0.05) -> dict:
    """两比例 z 检验（CTR 类指标）。1=Exp，2=Base。相对提升 CI 用 delta 方法。"""
    p1, p2 = c1 / n1, c2 / n2
    pp = (c1 + c2) / (n1 + n2)
    se_pool = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool if se_pool > 0 else 0.0
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    se_unpool = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = norm_ppf(1 - alpha / 2)
    diff = p1 - p2
    # 相对提升 r = p1/p2 - 1；Var(log(p1/p2)) ≈ (1-p1)/(n1 p1) + (1-p2)/(n2 p2)
    se_log = math.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))
    log_r = math.log(p1 / p2)
    return {
        "p1": p1, "p2": p2, "diff": diff, "z": z, "p": p,
        "ci_low": diff - zc * se_unpool, "ci_high": diff + zc * se_unpool,
        "lift_pct": 100.0 * (p1 / p2 - 1),
        "lift_ci_low": 100.0 * (math.exp(log_r - zc * se_log) - 1),
        "lift_ci_high": 100.0 * (math.exp(log_r + zc * se_log) - 1),
        "significant": p < alpha,
    }


def required_sample_size(sigma: float, mde_abs: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """均值类指标每组所需样本量：n = 2*(z_{1-a/2}+z_{power})^2 * sigma^2 / delta^2。"""
    za = norm_ppf(1 - alpha / 2)
    zb = norm_ppf(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde_abs ** 2)


def required_sample_size_proportion(p_base: float, mde_rel: float,
                                    alpha: float = 0.05, power: float = 0.8) -> int:
    """比例类指标（CTR）每组所需样本量。mde_rel 为相对提升，如 0.02 表示 +2%。"""
    p1 = p_base * (1 + mde_rel)
    pbar = (p_base + p1) / 2
    za = norm_ppf(1 - alpha / 2)
    zb = norm_ppf(power)
    num = (za * math.sqrt(2 * pbar * (1 - pbar))
           + zb * math.sqrt(p_base * (1 - p_base) + p1 * (1 - p1))) ** 2
    return math.ceil(num / (p1 - p_base) ** 2)


def bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    return [p < alpha / m for p in pvals]


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    k = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= rank / m * alpha:
            k = rank
    rej = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= k:
            rej[idx] = True
    return rej


# --------------------------------------------------------------------------
# [MOCK] 指标采样
# --------------------------------------------------------------------------
def sample_watch_time(n: int, mean: float) -> list[float]:
    """[MOCK] 人均时长样本：对数正态，重尾（真实分布亦然），均值校准到 mean。"""
    sigma = 1.0
    mu = math.log(mean) - sigma ** 2 / 2
    return [random.lognormvariate(mu, sigma) for _ in range(n)]


def main() -> None:
    random.seed(42)
    print("=" * 84)
    print("§3.3.2 显著性检验：AA 实验 / t 检验 / 置信区间 / 样本量 / 多重检验校正")
    print("=" * 84)

    # ---------- (1) AA 实验 ----------
    print("\n[1] AA 实验（空跑）：两组跑完全相同的策略，H0 为真")
    n_trials, n_per = 2000, 3000
    pvals = []
    for _ in range(n_trials):
        a = sample_watch_time(n_per, 40.0)
        b = sample_watch_time(n_per, 40.0)
        pvals.append(welch_ttest(a, b)["p"])
    fp = sum(1 for p in pvals if p < 0.05)
    print(f"  重复 {n_trials} 次 AA 实验（每组 {n_per} 用户），alpha=0.05")
    print(f"  假阳性次数 = {fp}，假阳性率 = {100.0*fp/n_trials:.2f}%（理论值 5.00%）")
    print("  p-value 分布（理论上应服从 U(0,1)，每格约 10%）：")
    bins = [0] * 10
    for p in pvals:
        bins[min(9, int(p * 10))] += 1
    for i, c in enumerate(bins):
        bar = "#" * int(c / n_trials * 200)
        print(f"    [{i/10:.1f},{(i+1)/10:.1f}) {c:>4d} ({100.0*c/n_trials:5.2f}%) {bar}")
    print("  >>> AA 实验是上线任何 AB 实验前的必做体检：若假阳性率显著偏离 5%，")
    print("      或 p-value 分布明显非均匀，说明分流系统或指标计算存在系统性偏差。")

    # ---------- (2) AB 实验 + t 检验 + 置信区间 ----------
    print("\n[2] AB 实验：人均时长（Welch t 检验 + 95% 置信区间）")
    base = sample_watch_time(20000, 40.0)
    exp = sample_watch_time(20000, 41.2)        # 真实提升 +3%
    r = welch_ttest(exp, base)
    print(f"  Base 均值 = {r['mean_base']:.4f}s（n={len(base)}）")
    print(f"  Exp  均值 = {r['mean_exp']:.4f}s（n={len(exp)}）")
    print(f"  绝对差 = {r['diff']:+.4f}s，相对提升 = {r['lift_pct']:+.3f}%")
    print(f"  t = {r['t']:.4f}, df = {r['df']:.1f}, p-value = {r['p']:.6f}")
    print(f"  95% CI(绝对差) = [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] s")
    print(f"  结论：{'显著正向' if r['significant'] and r['diff'] > 0 else '不显著'}"
          f"（CI 不含 0 <=> p < 0.05）")

    # ---------- (3) CTR 的两比例 z 检验 ----------
    print("\n[3] AB 实验：CTR（两比例 z 检验）")
    n1 = n2 = 400_000
    p_base_true, p_exp_true = 0.100, 0.102      # 真实相对提升 +2%
    c2 = sum(1 for _ in range(n2) if random.random() < p_base_true)
    c1 = sum(1 for _ in range(n1) if random.random() < p_exp_true)
    z = two_proportion_ztest(c1, n1, c2, n2)
    print(f"  Base: {c2}/{n2} = {100*z['p2']:.4f}%")
    print(f"  Exp : {c1}/{n1} = {100*z['p1']:.4f}%")
    print(f"  z = {z['z']:.4f}, p-value = {z['p']:.3e}")
    print(f"  绝对差 95% CI = [{100*z['ci_low']:+.4f}%, {100*z['ci_high']:+.4f}%]")
    print(f"  相对提升 = {z['lift_pct']:+.3f}%, "
          f"95% CI = [{z['lift_ci_low']:+.3f}%, {z['lift_ci_high']:+.3f}%]")
    print(f"  结论：{'显著' if z['significant'] else '不显著'}")

    # ---------- (4) 样本量估算 ----------
    print("\n[4] 所需样本量估算（alpha=0.05 双侧，power=80%）")
    sigma = math.sqrt(sum((x - 40.0) ** 2 for x in base) / (len(base) - 1))
    print(f"  人均时长：基线 40.00s，样本标准差 sigma = {sigma:.3f}s（重尾分布，CV≈{sigma/40:.2f}）")
    print(f"  {'目标MDE(相对)':>14} | {'MDE(绝对,s)':>12} | {'每组所需样本量':>14} | "
          f"{'按 100 万 DAU 需跑天数(5%流量)':>22}")
    print("  " + "-" * 76)
    for mde_rel in (0.005, 0.01, 0.02, 0.03, 0.05):
        mde_abs = 40.0 * mde_rel
        n = required_sample_size(sigma, mde_abs)
        days = n / (1_000_000 * 0.05)
        print(f"  {mde_rel*100:>13.1f}% | {mde_abs:>12.3f} | {n:>14,d} | {days:>22.2f}")
    print("\n  CTR（基线 10%）：")
    print(f"  {'目标MDE(相对)':>14} | {'每组所需曝光数':>16}")
    print("  " + "-" * 34)
    for mde_rel in (0.005, 0.01, 0.02, 0.05):
        print(f"  {mde_rel*100:>13.1f}% | "
              f"{required_sample_size_proportion(0.10, mde_rel):>16,d}")
    print("  >>> 正文：若实验流量过小或实验周期过短，则可能导致真实收益无法被检测出来。")
    print("      上表给出定量答案：想检出 0.5% 的相对提升，样本量是检出 2% 的 16 倍"
          "（n ∝ 1/delta^2）。")

    # ---------- (5) 多重检验校正 ----------
    print("\n[5] 多重检验校正：同时观察 20 个指标时的假阳性风险")
    metric_names = [f"metric_{i:02d}" for i in range(20)]
    # 前 17 个无真实效应，后 3 个有 +3% 真实提升
    raw_p = []
    for i in range(20):
        lift = 1.03 if i >= 17 else 1.0
        a = sample_watch_time(6000, 40.0 * lift)
        b = sample_watch_time(6000, 40.0)
        raw_p.append(welch_ttest(a, b)["p"])
    naive = [p < 0.05 for p in raw_p]
    bon = bonferroni(raw_p, 0.05)
    bh = benjamini_hochberg(raw_p, 0.05)
    print(f"  同时检验 m=20 个指标；其中 metric_17/18/19 存在 +3% 真实效应，其余为 H0 真。")
    print(f"  未校正下至少一个假阳性的理论概率 = 1-(1-0.05)^20 = "
          f"{1-0.95**20:.3f}")
    print(f"  {'指标':>10} | {'p-value':>10} | {'裸 p<0.05':>9} | "
          f"{'Bonferroni':>11} | {'BH-FDR':>7} | {'真实效应':>8}")
    print("  " + "-" * 70)
    for i, name in enumerate(metric_names):
        truth = "有" if i >= 17 else "无"
        print(f"  {name:>10} | {raw_p[i]:>10.5f} | {'✓' if naive[i] else '·':>9} | "
              f"{'✓' if bon[i] else '·':>11} | {'✓' if bh[i] else '·':>7} | {truth:>8}")
    fp_naive = sum(1 for i in range(17) if naive[i])
    fp_bon = sum(1 for i in range(17) if bon[i])
    fp_bh = sum(1 for i in range(17) if bh[i])
    tp_naive = sum(1 for i in range(17, 20) if naive[i])
    tp_bon = sum(1 for i in range(17, 20) if bon[i])
    tp_bh = sum(1 for i in range(17, 20) if bh[i])
    print(f"\n  假阳性数：裸检验 {fp_naive} / Bonferroni {fp_bon} / BH-FDR {fp_bh}")
    print(f"  真阳性数：裸检验 {tp_naive} / Bonferroni {tp_bon} / BH-FDR {tp_bh}（共 3 个）")
    print("  >>> Bonferroni 控制 FWER（族错误率），最保守，功效损失大；")
    print("      BH 控制 FDR（错误发现率），在指标很多时是更常用的折中方案。")
    print("  >>> 正文：当同时观察多个指标时，必须进行多重检验校正以降低假阳性风险。")


if __name__ == "__main__":
    main()
