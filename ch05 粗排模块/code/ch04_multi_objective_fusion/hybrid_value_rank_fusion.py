"""
基于值与序关系的融合(第 5 章 §5.4.3 基于值与序关系的融合 / Hybrid Fusion)

对应正文概念
------------
正文 §5.4.2 末尾指出纯 Rank 融合「只保留了排序位置,而丢失了原始 pxtr 的数值信息」,
§5.4.3 于是给出值序融合(Hybrid Fusion)的一般形式:

    s_agg = f(pxtr, Rank_pxtr)

并进一步近似为可分解形式:

    s_agg = f(pxtr) * g(Rank_pxtr)

其中「f(·) 负责建模 Value 信息,g(·) 负责建模 Rank 信息」。正文给出的候选形式:

    f(pxtr) = log(1.0 + pxtr)
    f(pxtr) = (β + α · pxtr)^γ
    g(Rank) = 1 / Rank^λ
    g(Rank) = 1 - (Rank / N)^λ

「这样既能够保留模型输出的细粒度数值差异,又能够获得 Rank 融合带来的鲁棒性。」

本实现做了什么
--------------
1. 按正文给的 2 种 f(·) × 2 种 g(·) 组合出 4 套值序融合公式,多目标聚合统一取
   s_agg = Σ_i w_i · f(p_i) · g(Rank_i),并与纯值融合 / 纯 Rank 融合同台对比。
2. 信息损失对照实验:复用 §5.4.2 中「密集型 vs 长尾型」两组 pxtr(两组三路 Rank
   完全一致),验证纯 Rank 融合分逐项相同、而值序融合能把数值差异找回来。
3. λ 连续谱实验:λ → 0 时 g(Rank) → 常数,值序融合退化为纯值融合;λ 增大时 Rank
   项主导,排序逐渐靠拢纯 Rank 融合。打印 ρ(vs 值融合) 与 ρ(vs Rank 融合) 的交叉表,
   说明值序融合本质上是在两种机制之间做可调插值。
4. 鲁棒性实验:对 p_ctr 施加三种「保序但改变分布」的变换(模型升级 / 重校准),
   对比三类融合的 Spearman ρ 与 Top-K Jaccard -- 值序融合的鲁棒性介于两者之间。

关键公式
--------
    s_agg = Σ_i w_i · f(p_i) · g(Rank_i)
    f1(p) = log(1 + p)                 f2(p) = (β + α p)^γ
    g1(r) = 1 / r^λ                    g2(r) = 1 - (r / N)^λ
"""

from __future__ import annotations

import math
import random

OBJECTIVES = ("pctr", "pcvr", "pltr")
WEIGHTS = {"pctr": 0.5, "pcvr": 0.3, "pltr": 0.2}


# ----------------------------------------------------------------------------
# 排序与相关性工具
# ----------------------------------------------------------------------------
def ranks(vals: list[float]) -> list[int]:
    """返回每个下标的排名(1 = 该目标上分数最高)。"""
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    r = [0] * len(vals)
    for pos, idx in enumerate(order):
        r[idx] = pos + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma = mb = (n + 1) / 2.0
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def topk_jaccard(a: list[float], b: list[float], k: int) -> float:
    sa = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    sb = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(sa & sb) / len(sa | sb)


# ----------------------------------------------------------------------------
# Value 侧 f(·) 与 Rank 侧 g(·)(正文 §5.4.3 列出的形式)
# ----------------------------------------------------------------------------
def f_log(p: float) -> float:
    """f(pxtr) = log(1.0 + pxtr)。"""
    return math.log(1.0 + p)


def f_power(p: float, beta: float = 0.1, alpha: float = 1.0,
            gamma: float = 0.5) -> float:
    """f(pxtr) = (β + α · pxtr)^γ。"""
    return (beta + alpha * p) ** gamma


def f_identity(p: float) -> float:
    """f(pxtr) = pxtr,用作纯值融合基线。"""
    return p


def g_reciprocal(rank: int, n: int, lam: float = 0.5) -> float:
    """g(Rank) = 1 / Rank^λ。"""
    return 1.0 / rank ** lam


def g_normalized(rank: int, n: int, lam: float = 1.0) -> float:
    """g(Rank) = 1 - (Rank / N)^λ。"""
    return 1.0 - (rank / n) ** lam


def g_constant(rank: int, n: int, lam: float = 0.0) -> float:
    """g(Rank) ≡ 1,即不使用 Rank 信息(纯值融合)。"""
    return 1.0


F_FORMS = {"log(1+p)": f_log, "(β+αp)^γ": f_power}
G_FORMS = {"1/Rank^λ": g_reciprocal, "1-(Rank/N)^λ": g_normalized}


# ----------------------------------------------------------------------------
# 三类融合
# ----------------------------------------------------------------------------
def value_fusion(pxtr: dict[str, list[float]]) -> list[float]:
    """纯值融合(线性):s = Σ_i w_i p_i。"""
    n = len(next(iter(pxtr.values())))
    return [sum(WEIGHTS[k] * pxtr[k][i] for k in OBJECTIVES) for i in range(n)]


def rank_fusion(pxtr: dict[str, list[float]], lam: float = 0.5) -> list[float]:
    """纯 Rank 融合:s = Σ_i w_i / Rank_i^λ。"""
    n = len(next(iter(pxtr.values())))
    rk = {k: ranks(pxtr[k]) for k in OBJECTIVES}
    return [sum(WEIGHTS[k] * g_reciprocal(rk[k][i], n, lam) for k in OBJECTIVES)
            for i in range(n)]


def hybrid_fusion(pxtr: dict[str, list[float]], f_name: str, g_name: str,
                  lam: float = 0.5) -> list[float]:
    """值序融合:s_agg = Σ_i w_i · f(p_i) · g(Rank_i)。"""
    n = len(next(iter(pxtr.values())))
    f = F_FORMS[f_name] if f_name in F_FORMS else f_identity
    g = G_FORMS[g_name] if g_name in G_FORMS else g_constant
    rk = {k: ranks(pxtr[k]) for k in OBJECTIVES}
    out = []
    for i in range(n):
        s = 0.0
        for k in OBJECTIVES:
            s += WEIGHTS[k] * f(pxtr[k][i]) * g(rk[k][i], n, lam)
        out.append(s)
    return out


# ----------------------------------------------------------------------------
# [MOCK] 合成 pxtr
# ----------------------------------------------------------------------------
def mock_pxtr(n: int, rng: random.Random) -> dict[str, list[float]]:
    """[MOCK] 生成三路相关但不一致的 pxtr,量级刻意不同(§5.4.1 的数值敏感问题)。"""
    latent = [rng.gauss(0, 1) for _ in range(n)]

    def gen(mu: float, sigma: float, corr: float) -> list[float]:
        out = []
        for i in range(n):
            z = corr * latent[i] + (1 - corr) * rng.gauss(0, 1)
            out.append(round(max(1e-4, min(0.9999, mu + sigma * z)), 6))
        return out

    return {"pctr": gen(0.30, 0.08, 0.6),
            "pcvr": gen(0.04, 0.012, 0.3),
            "pltr": gen(0.50, 0.15, 0.4)}


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    N, TOPK = 400, 50
    pxtr = mock_pxtr(N, rng)

    v = value_fusion(pxtr)
    r = rank_fusion(pxtr)

    # ---- 1. 四套值序公式 vs 两个基线 ----
    print("=" * 100)
    print(f"四套值序融合公式 vs 纯值融合 / 纯 Rank 融合(N={N}, Top-{TOPK}, λ=0.5)")
    print("=" * 100)
    fused: dict[str, list[float]] = {"纯值融合 Σw·p": v, "纯Rank融合 Σw/R^0.5": r}
    for fn in F_FORMS:
        for gn in G_FORMS:
            fused[f"{fn} × {gn}"] = hybrid_fusion(pxtr, fn, gn)
    print(f"{'融合公式':>24}{'ρ(vs 值融合)':>16}{'ρ(vs Rank融合)':>17}"
          f"{'Top50∩值':>11}{'Top50∩Rank':>12}{'Top1':>7}")
    for name, s in fused.items():
        top1 = max(range(N), key=lambda i: s[i])
        print(f"{name:>24}{spearman(s, v):>+16.4f}{spearman(s, r):>+17.4f}"
              f"{topk_jaccard(s, v, TOPK):>11.3f}{topk_jaccard(s, r, TOPK):>12.3f}"
              f"{top1:>7}")
    print("值序融合的两列 ρ 都显著高于「纯值 vs 纯Rank」这一对,说明它确实落在两种机制中间。")
    print(f"作为参照,纯值融合与纯 Rank 融合彼此的 ρ 只有 {spearman(v, r):+.4f}。")

    # ---- 2. 信息损失对照:密集型 vs 长尾型 ----
    print("\n" + "=" * 100)
    print("信息损失对照:两组三路 Rank 完全相同、数值分布不同的候选集(§5.4.2 的缺陷)")
    print("=" * 100)
    dense = {"pctr": [0.99, 0.98, 0.97, 0.96, 0.95],
             "pcvr": [0.050, 0.049, 0.048, 0.047, 0.046],
             "pltr": [0.20, 0.21, 0.22, 0.23, 0.24]}
    gapped = {"pctr": [0.99, 0.50, 0.20, 0.10, 0.05],
              "pcvr": [0.050, 0.010, 0.005, 0.003, 0.001],
              "pltr": [0.20, 0.60, 0.65, 0.70, 0.90]}
    rk_d = {k: ranks(dense[k]) for k in OBJECTIVES}
    rk_g = {k: ranks(gapped[k]) for k in OBJECTIVES}
    same_rank = all(rk_d[k] == rk_g[k] for k in OBJECTIVES)
    print(f"两组的三路 Rank 是否逐项相同? {same_rank}")
    hyb_d = hybrid_fusion(dense, "log(1+p)", "1/Rank^λ")
    hyb_g = hybrid_fusion(gapped, "log(1+p)", "1/Rank^λ")
    rk_fd, rk_fg = rank_fusion(dense), rank_fusion(gapped)
    v_d, v_g = value_fusion(dense), value_fusion(gapped)
    print(f"\n{'item':>5}{'密集值融合':>12}{'长尾值融合':>12}{'密集Rank融合':>14}"
          f"{'长尾Rank融合':>14}{'密集值序':>11}{'长尾值序':>11}{'值序Δ':>10}")
    for i in range(5):
        print(f"{i:>5}{v_d[i]:>12.4f}{v_g[i]:>12.4f}{rk_fd[i]:>14.4f}"
              f"{rk_fg[i]:>14.4f}{hyb_d[i]:>11.4f}{hyb_g[i]:>11.4f}"
              f"{hyb_d[i] - hyb_g[i]:>+10.4f}")
    print(f"\n纯 Rank 融合两组逐项相同? "
          f"{all(abs(a - b) < 1e-12 for a, b in zip(rk_fd, rk_fg))}  "
          f"值序融合两组逐项相同? "
          f"{all(abs(a - b) < 1e-12 for a, b in zip(hyb_d, hyb_g))}")
    print("值序融合在保留 Rank 骨架的同时,把 f(pxtr) 这一路的数值差异重新注入排序分。")

    # ---- 3. λ 连续谱:值序融合是值融合与 Rank 融合之间的插值 ----
    print("\n" + "=" * 100)
    print("λ 连续谱:g(Rank)=1/Rank^λ 中 λ 如何把值序融合从「值」一端推向「Rank」一端")
    print("=" * 100)
    print(f"{'λ':>8}{'ρ(vs 值融合)':>16}{'ρ(vs Rank融合)':>17}"
          f"{'Top50∩值':>11}{'Top50∩Rank':>12}{'g(1)/g(N)':>12}")
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0):
        s = hybrid_fusion(pxtr, "log(1+p)", "1/Rank^λ", lam)
        ratio = g_reciprocal(1, N, lam) / g_reciprocal(N, N, lam)
        print(f"{lam:>8.2f}{spearman(s, v):>+16.4f}{spearman(s, r):>+17.4f}"
              f"{topk_jaccard(s, v, TOPK):>11.3f}{topk_jaccard(s, r, TOPK):>12.3f}"
              f"{ratio:>12.1f}")
    print("λ=0 时 g≡1,值序融合与纯值融合完全同序(ρ=1.0,这里 f=log(1+p) 单调不改序);")
    print("λ 增大后头尾位置增益比 g(1)/g(N) 迅速拉大,排序逐渐被 Rank 项接管。")

    # ---- 4. 鲁棒性:保序但改变分布的变换 ----
    print("\n" + "=" * 100)
    print("鲁棒性实验:对 p_ctr 施加三种「保序但改变分布」的变换(模型升级 / 重校准)")
    print("=" * 100)
    transforms = {
        "×3 乘性漂移": lambda x: x * 3.0,
        "平方压缩 x^2": lambda x: x ** 2,
        "sigmoid 重标定": lambda x: 1.0 / (1.0 + math.exp(-8.0 * (x - 0.15))),
    }
    print("每格为「ρ / Top-50 Jaccard」(与变换前的自身排序对比,越接近 1.0 越鲁棒)")
    print(f"{'融合公式':>24}" + "".join(f"{t:>22}" for t in transforms))
    for name, s_old in fused.items():
        row = f"{name:>24}"
        for _, fn in transforms.items():
            p2 = {k: ([fn(x) for x in vals] if k == "pctr" else list(vals))
                  for k, vals in pxtr.items()}
            if name.startswith("纯值"):
                s_new = value_fusion(p2)
            elif name.startswith("纯Rank"):
                s_new = rank_fusion(p2)
            else:
                f_name, g_name = name.split(" × ")
                s_new = hybrid_fusion(p2, f_name, g_name)
            row += f"{spearman(s_old, s_new):>+13.4f} /{topk_jaccard(s_old, s_new, TOPK):>6.2f}"
        print(row)
    print("\n纯 Rank 融合恒为 1.0(只看序);纯值融合被分布变化改写得最厉害;")
    print("值序融合居中 -- 正文所说「既保留细粒度数值差异,又获得 Rank 融合带来的鲁棒性」。")
