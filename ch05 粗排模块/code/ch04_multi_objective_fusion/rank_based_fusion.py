"""
基于序关系的融合(第 5 章 §5.4.2 基于序关系的融合 / Rank-Based Fusion)

对应正文概念
------------
正文 §5.4.2 的出发点:「虽然基于值的融合方法具有较强的表达能力,但其对 pxtr 的具体
数值分布高度敏感。在工业实践中经常会出现这样一种情况:模型升级之后,排序能力实际上
有所提升,但由于输出分布发生变化,导致原有融合策略失效,从而使线上 A/B 实验效果持平
甚至下降。」

于是转而「不再关注具体数值,而只关注排序位置」:先对各目标分别排序得到
Rank_ctr, Rank_cvr, ..., Rank_n,再用排序位置构造融合分数。正文列出的全部 5 种形式
本文件都实现了:

    基础倒数:      s_agg = Σ_i w_i / Rank_i
    幂函数:        S_i = w_i / (Rank_i^λ + b)
    归一化幂函数:   S_i = w_i * (1 - (Rank_i / N)^λ)
    指数形式:      S_i = w_i * (e^{k(2 Rank_i/N - 1)} - e^{-k(2 Rank_i/N - 1)})
    带平滑项衰减:   S_i = w_i * (1 - (Rank_i / (N + b))^λ)

正文同时指出纯 Rank 融合的缺陷:「由于只保留了排序位置,而丢失了原始 pxtr 的数值信息,
因此无法感知不同物品之间真实的分数差异。例如排名第一和第二的物品可能分数极其接近
(0.99, 0.98),也可能存在数量级差异(0.99, 0.50),但在 Rank 融合中都会被视为相邻位置。」

本实现做了什么
--------------
1. 五种 Rank 变换函数 + 权重聚合,给出同一候选集下的排序差异对比表。
2. 鲁棒性实验:对 pxtr 施加三种「分布变化但序不变」的变换(×3 乘性漂移、
   平方压缩、sigmoid 重标定),对比线性值融合与各 Rank 融合的排序稳定性。
   由于 Rank 只依赖序,所有 Rank 融合在三种变换下 ρ 恒为 1.0。
3. 信息损失实验:构造两个 pxtr 分布 -- 「密集型」(相邻分差 0.01)与
   「长尾型」(头部断层),Rank 融合给出完全相同的结果,而值融合不同,
   定量复现正文指出的信息损失缺陷。
4. 位置衰减曲线数值表(代替画图):打印各变换在 Rank=1,2,5,10,...,N 处的取值,
   展示 λ / k / b 如何控制头部陡峭程度。

关键公式:见上方 5 种 S_i 定义;聚合统一为 s_agg = Σ_i S_i(w_i)。
"""

from __future__ import annotations

import math
import random

OBJECTIVES = ("pctr", "pcvr", "pltr")
WEIGHTS = {"pctr": 0.5, "pcvr": 0.3, "pltr": 0.2}


# ----------------------------------------------------------------------------
# Rank 变换(正文 §5.4.2 给出的 5 种形式)
# ----------------------------------------------------------------------------
def g_reciprocal(rank: int, n: int) -> float:
    """s = 1 / Rank  (正文最基础形式)。"""
    return 1.0 / rank


def g_power(rank: int, n: int, lam: float = 0.7, b: float = 2.0) -> float:
    """S = 1 / (Rank^λ + b)  (幂函数形式)。"""
    return 1.0 / (rank ** lam + b)


def g_norm_power(rank: int, n: int, lam: float = 1.5) -> float:
    """S = 1 - (Rank / N)^λ  (归一化幂函数形式)。"""
    return 1.0 - (rank / n) ** lam


def g_exp(rank: int, n: int, k: float = 1.5) -> float:
    """S = e^{k(2 Rank/N - 1)} - e^{-k(2 Rank/N - 1)}  (指数形式)。

    注意:正文该式随 Rank 增大而递增(t = 2Rank/N - 1 从 -1 到 +1,而
    e^{kt} - e^{-kt} = 2 sinh(kt) 单调递增),因此作为「越靠前越高」的融合分时
    需要取负号。这里保留正文原式,返回值在聚合时取相反数,并在输出中注明。
    """
    t = 2.0 * rank / n - 1.0
    return math.exp(k * t) - math.exp(-k * t)


def g_smooth_decay(rank: int, n: int, lam: float = 1.5, b: float = 50.0) -> float:
    """S = 1 - (Rank / (N + b))^λ  (带平滑项的排序衰减形式)。"""
    return 1.0 - (rank / (n + b)) ** lam


RANK_TRANSFORMS = {
    "1/Rank": (g_reciprocal, False),
    "幂函数 1/(R^0.7+2)": (g_power, False),
    "归一化幂 1-(R/N)^1.5": (g_norm_power, False),
    "指数 2sinh(k t)": (g_exp, True),          # True = 聚合时取负(见 g_exp docstring)
    "平滑衰减 1-(R/(N+b))^1.5": (g_smooth_decay, False),
}


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def ranks(values: list[float]) -> list[int]:
    """1 = 最大值。"""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0] * len(values)
    for pos, i in enumerate(order):
        out[i] = pos + 1
    return out


def rank_fusion(pxtr: dict[str, list[float]], transform_name: str) -> list[float]:
    """s_agg = Σ_i w_i * g(Rank_i)。"""
    g, negate = RANK_TRANSFORMS[transform_name]
    n = len(next(iter(pxtr.values())))
    rk = {k: ranks(v) for k, v in pxtr.items()}
    sign = -1.0 if negate else 1.0
    return [sum(WEIGHTS[k] * sign * g(rk[k][i], n) for k in pxtr) for i in range(n)]


def value_fusion(pxtr: dict[str, list[float]]) -> list[float]:
    n = len(next(iter(pxtr.values())))
    return [sum(WEIGHTS[k] * pxtr[k][i] for k in pxtr) for i in range(n)]


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    m = (n + 1) / 2
    num = sum((x - m) * (y - m) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - m) ** 2 for x in ra))
    db = math.sqrt(sum((y - m) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def topk_jaccard(a: list[float], b: list[float], k: int) -> float:
    sa = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    sb = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(sa & sb) / len(sa | sb)


def mock_pxtr(n: int, rng: random.Random) -> dict[str, list[float]]:
    """[MOCK] 三路 pxtr,量纲各异且两两弱相关。"""
    latent = [rng.gauss(0, 1) for _ in range(n)]

    def gen(scale: float, offset: float, corr: float) -> list[float]:
        return [round(offset + scale / (1 + math.exp(-(corr * latent[i]
                                                       + (1 - corr) * rng.gauss(0, 1)))), 6)
                for i in range(n)]

    return {"pctr": gen(0.30, 0.02, 0.6),
            "pcvr": gen(0.04, 0.002, 0.3),
            "pltr": gen(0.50, 0.05, 0.4)}


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    N, TOPK = 400, 50
    pxtr = mock_pxtr(N, rng)

    # ---- 1. 位置衰减曲线数值表 ----
    print("=" * 100)
    print(f"五种 Rank 变换的位置衰减数值表(N={N};指数形式按 docstring 取负后展示)")
    print("=" * 100)
    probe = (1, 2, 3, 5, 10, 20, 50, 100, 200, 400)
    print(f"{'变换':>26}" + "".join(f"{f'R={r}':>10}" for r in probe))
    for name, (g, neg) in RANK_TRANSFORMS.items():
        sign = -1.0 if neg else 1.0
        print(f"{name:>26}" + "".join(f"{sign * g(r, N):>10.4f}" for r in probe))
    print("1/Rank 从 R=1 到 R=10 衰减 10 倍,头部极陡;归一化幂/平滑衰减近似线性,头部平缓。")
    print("λ 越大头部越陡;b 越大对头部名次的区分越钝 -- 这正是调控流量集中度的旋钮。")

    # ---- 2. 同一候选集下各融合公式的排序差异 ----
    v = value_fusion(pxtr)
    fused = {"值融合(线性)": v}
    for name in RANK_TRANSFORMS:
        fused[name] = rank_fusion(pxtr, name)

    print("\n" + "=" * 100)
    print(f"同一候选集下各融合公式 vs 线性值融合的排序差异(N={N}, Top-{TOPK})")
    print("=" * 100)
    print(f"{'融合公式':>26}{'ρ(vs 值融合)':>16}{'Top50 Jaccard':>16}{'Top1 物品id':>14}")
    for name, s in fused.items():
        top1 = max(range(N), key=lambda i: s[i])
        print(f"{name:>26}{spearman(s, v):>+16.4f}{topk_jaccard(s, v, TOPK):>16.3f}{top1:>14}")

    print(f"\n各 Rank 变换之间的 ρ 矩阵:")
    rnames = list(RANK_TRANSFORMS)
    short = {n: n.split()[0] for n in rnames}
    print(f"{'':>14}" + "".join(f"{short[n]:>14}" for n in rnames))
    for a in rnames:
        row = f"{short[a]:>14}"
        for b in rnames:
            row += ("-".rjust(14) if a == b
                    else f"{spearman(fused[a], fused[b]):>+14.4f}")
        print(row)

    # ---- 3. 鲁棒性:分布变化但序不变 ----
    print("\n" + "=" * 100)
    print("鲁棒性实验:三种「保序但改变分布」的 pxtr 变换(模拟模型升级/重校准)")
    print("=" * 100)
    transforms = {
        "×3 乘性漂移": lambda x: min(0.9999, x * 3.0),
        "平方压缩 x^2": lambda x: x ** 2,
        "sigmoid 重标定": lambda x: 1.0 / (1.0 + math.exp(-8.0 * (x - 0.15))),
    }
    print("每格为「ρ / Top-50 Jaccard」;变换只施加在 pctr 这一路上(单目标模型被升级)")
    print(f"{'融合公式':>26}" + "".join(f"{t:>22}" for t in transforms))
    drift_rows: dict[str, list[tuple[float, float]]] = {}
    for name, s_old in fused.items():
        row = f"{name:>26}"
        cells = []
        for _, fn in transforms.items():
            p2 = {k: ([fn(x) for x in vals] if k == "pctr" else list(vals))
                  for k, vals in pxtr.items()}
            s_new = (value_fusion(p2) if name.startswith("值融合")
                     else rank_fusion(p2, name))
            cells.append((spearman(s_old, s_new), topk_jaccard(s_old, s_new, TOPK)))
            row += f"{cells[-1][0]:>+13.4f} /{cells[-1][1]:>6.2f}"
        drift_rows[name] = cells
        print(row)

    vr = drift_rows["值融合(线性)"]
    print(f"\n结论:所有 Rank 融合在三种保序变换下 ρ 与 Top-50 Jaccard 恒为 1.0 -- 正文说的")
    print("「对于 pxtr 的具体取值范围、分布形态以及校准程度并不敏感,仅依赖于排序顺序本身」。")
    print(f"线性值融合则被改写:三种变换下 ρ = "
          f"{vr[0][0]:+.4f}/{vr[1][0]:+.4f}/{vr[2][0]:+.4f},"
          f"Top-50 Jaccard 仅 {vr[0][1]:.2f}/{vr[1][1]:.2f}/{vr[2][1]:.2f}。")
    print("即「模型升级后排序能力可能提升,但原有融合策略失效」(§5.4.2)。")

    # ---- 4. 信息损失:密集型 vs 长尾型分布 ----
    print("\n" + "=" * 100)
    print("信息损失实验:两种 pxtr 分布在 Rank 融合下完全不可区分(§5.4.2 的缺陷)")
    print("=" * 100)
    n_small = 5
    dense = {"pctr": [0.99, 0.98, 0.97, 0.96, 0.95],
             "pcvr": [0.050, 0.049, 0.048, 0.047, 0.046],
             "pltr": [0.20, 0.21, 0.22, 0.23, 0.24]}
    gapped = {"pctr": [0.99, 0.50, 0.20, 0.10, 0.05],
              "pcvr": [0.050, 0.010, 0.005, 0.003, 0.001],
              "pltr": [0.20, 0.60, 0.65, 0.70, 0.90]}
    # pltr 在两组里都是递增的,因此两组的三路 Rank 完全一致
    for tag, p in (("密集型(相邻分差 0.01)", dense), ("长尾型(头部断层)", gapped)):
        print(f"\n{tag}")
        print(f"{'item':>6}{'pctr':>9}{'pcvr':>9}{'pltr':>9}{'R_ctr':>8}{'R_cvr':>8}"
              f"{'R_ltr':>8}{'值融合':>11}{'1/Rank融合':>13}")
        rk = {k: ranks(p[k]) for k in OBJECTIVES}
        vs = value_fusion(p)
        rs = rank_fusion(p, "1/Rank")
        for i in range(n_small):
            print(f"{i:>6}{p['pctr'][i]:>9.3f}{p['pcvr'][i]:>9.3f}{p['pltr'][i]:>9.3f}"
                  f"{rk['pctr'][i]:>8}{rk['pcvr'][i]:>8}{rk['pltr'][i]:>8}"
                  f"{vs[i]:>11.4f}{rs[i]:>13.4f}")
    r_dense = rank_fusion(dense, "1/Rank")
    r_gapped = rank_fusion(gapped, "1/Rank")
    same = all(abs(a - b) < 1e-12 for a, b in zip(r_dense, r_gapped))
    v_dense, v_gapped = value_fusion(dense), value_fusion(gapped)
    print(f"\n两组的 1/Rank 融合分逐项相同? {same}")
    print(f"两组的值融合分之差: "
          f"{[round(a - b, 4) for a, b in zip(v_dense, v_gapped)]}")
    print("Rank 融合看不到「0.99 vs 0.98」与「0.99 vs 0.50」的区别 -- 这就是正文指出的")
    print("信息损失:它「无法感知不同物品之间真实的分数差异」,因此才需要值序混合融合。")
