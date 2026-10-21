"""
广义乘法公式的泰勒展开与偏置项 β 分析(第 5 章 §5.5 多目标融合机制分析)

对应正文概念
------------
正文指出工业界用的是通用乘法公式 ∏_{i=1}^n (β_i + α_i p_i)^{γ_i},并回答:
「为什么工业界广泛采用的广义乘法公式,在某些情况下会表现出类似加法公式的行为?」

正文的泰勒(乘积展开)推导如下:

    Score = ∏_{i=1}^n (β_i + α_i p_i)
          = ∏_i β_i · ∏_i (1 + α_i p_i / β_i)
    记 Const = ∏_i β_i, x_i = α_i p_i / β_i
    则 Score = Const · ∏_i (1 + x_i)

n = 3 时:
    ∏_{i=1}^3 (1+x_i) = 1 + (x_1+x_2+x_3) + (x_1x_2+x_1x_3+x_2x_3) + x_1x_2x_3

代回得:
    Score = Const [ 1 + Σ_i α_i p_i/β_i + Σ_{i<j} α_iα_j p_ip_j/(β_iβ_j)
                      + Σ_{i<j<k} α_iα_jα_k p_ip_jp_k/(β_iβ_jβ_k) + ... ]

其中一阶项=线性加权求和,二阶项=两两协同,三阶项=三目标同时优秀的协同奖励。
当 β_i ≫ α_i p_i 时 x_i ≪ 1,由等价无穷小 x_i x_j = o(x_i)、x_i x_j x_k = o(x_i),
高阶项可忽略:

    Score ≈ Const (1 + Σ_i α_i p_i / β_i) = Const + Σ_i w_i p_i,
    其中 w_i = Const · α_i / β_i

即 β_i → ∞ 时乘法融合退化为线性加法融合;β_i → 0 时高阶交互项主导,强调多目标同时优秀。
正文给出的数值锚点:对 (0.9,0.1) 与 (0.5,0.5),β=0、β=1 时乘法取值差异很大,而 β=10 时
两者分别为 110.09 和 110.25,「几乎没有什么差别」。

正文的梯度视角:对 Score = (β_1+α_1p_1)(β_2+α_2p_2),有
    ∂Score/∂p_1 ∝ (β_2 + α_2 p_2)
即一个目标越优秀,另一个目标的提升价值越高(正向协同 Positive Synergy);
而加法 Score = w_1p_1 + w_2p_2 的偏导恒为常数 w_1、w_2,目标之间无协同增益。

本实现做了什么
--------------
1. 复刻 β ∈ {0, 1, 10} 下 (0.9,0.1) 与 (0.5,0.5) 的乘法取值,核对 110.09 / 110.25。
2. 按正文展开式逐阶计算 n=3 的一阶/二阶/三阶项,打印各阶项占 Score 的比例随 β 的变化,
   直接看到「β 大 → 高阶项消失」。
3. 把精确 Score 与一阶截断近似 Const + Σ w_i p_i 作对比,打印相对误差与 Top-K 排序一致性
   (Spearman / Jaccard),验证 β→∞ 时两者等价。
4. 梯度表:数值与解析两种方式给出 ∂Score/∂p_1,展示它随 p_2 单调上升(乘法)或恒定(加法)。

关键公式
--------
    Score = Const · ∏_i (1 + x_i),  x_i = α_i p_i / β_i,  Const = ∏_i β_i
    Score ≈ Const + Σ_i w_i p_i,    w_i = Const · α_i / β_i
    ∂Score/∂p_1 ∝ (β_2 + α_2 p_2)
"""

from __future__ import annotations

import itertools
import random

N_OBJ = 3                       # 正文展开取 n = 3
ALPHAS = (1.0, 1.0, 1.0)        # α_i,正文示例取 1


def product_score(ps, alphas, betas, gammas=None) -> float:
    """广义乘法公式 ∏_i (β_i + α_i p_i)^{γ_i}。"""
    s = 1.0
    for k, p in enumerate(ps):
        base = betas[k] + alphas[k] * p
        g = 1.0 if gammas is None else gammas[k]
        s *= base ** g
    return s


def taylor_terms(ps, alphas, betas) -> dict[int, float]:
    """按正文展开式计算各阶项(已乘 Const):order -> Const · Σ ∏ x。"""
    const = 1.0
    for b in betas:
        const *= b
    xs = [alphas[k] * ps[k] / betas[k] for k in range(len(ps))]
    out = {0: const}
    for order in range(1, len(ps) + 1):
        acc = 0.0
        for comb in itertools.combinations(range(len(ps)), order):
            t = 1.0
            for k in comb:
                t *= xs[k]
            acc += t
        out[order] = const * acc
    return out


def first_order_weights(alphas, betas) -> tuple[float, list[float]]:
    """一阶截断:Const 与 w_i = Const · α_i / β_i。"""
    const = 1.0
    for b in betas:
        const *= b
    return const, [const * alphas[k] / betas[k] for k in range(len(betas))]


def spearman(a, b) -> float:
    """Spearman 秩相关(无并列值时的简化公式)。"""
    n = len(a)
    ra = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: -a[i]))}
    rb = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: -b[i]))}
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def topk_jaccard(a, b, k) -> float:
    sa = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    sb = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(sa & sb) / len(sa | sb)


def mock_pxtr(n: int, rng: random.Random) -> list[tuple[float, float, float]]:
    """[MOCK] 合成三目标 pxtr,含偏科型与均衡型两类。"""
    out = []
    for i in range(n):
        if i % 3 == 0:
            ps = (rng.uniform(0.7, 0.98), rng.uniform(0.02, 0.2), rng.uniform(0.02, 0.2))
        elif i % 3 == 1:
            m = rng.uniform(0.3, 0.6)
            ps = tuple(min(0.98, max(0.02, m + rng.gauss(0, 0.05))) for _ in range(3))
        else:
            ps = tuple(rng.uniform(0.05, 0.9) for _ in range(3))
        out.append(ps)
    return out


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    # ---- 1. 复刻正文 β = 0 / 1 / 10 的数值锚点(两目标) ----
    print("=" * 96)
    print("正文数值锚点:两目标 ∏(β + α p),α = 1,对比 (0.9,0.1) 与 (0.5,0.5)")
    print("=" * 96)
    print(f"{'β':>6}{'Score(0.9,0.1)':>18}{'Score(0.5,0.5)':>18}"
          f"{'差值':>12}{'相对差 |Δ|/均值':>16}{'谁更高':>16}")
    for beta in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0):
        s1 = product_score((0.9, 0.1), (1.0, 1.0), (beta, beta))
        s2 = product_score((0.5, 0.5), (1.0, 1.0), (beta, beta))
        rel = abs(s1 - s2) / ((s1 + s2) / 2) if s1 + s2 > 0 else 0.0
        who = "偏科(0.9,0.1)" if s1 > s2 else "均衡(0.5,0.5)"
        print(f"{beta:>6.1f}{s1:>18.4f}{s2:>18.4f}{s1 - s2:>+12.4f}"
              f"{rel:>16.6f}{who:>16}")
    print("\nβ=0 时退化成因子直乘:0.09 vs 0.25,均衡内容优势巨大;")
    print("β=10 时为 110.09 vs 110.25(与正文一致),相对差已降到 1e-3 量级,近似加法公式。")

    # ---- 2. 各阶项占比随 β 变化(n = 3) ----
    print("\n" + "=" * 96)
    print("泰勒展开各阶项占比(n=3,p=(0.8,0.5,0.2),α=(1,1,1),β_1=β_2=β_3=β)")
    print("=" * 96)
    ps = (0.8, 0.5, 0.2)
    print(f"{'β':>8}{'Const':>14}{'一阶项':>14}{'二阶项':>14}{'三阶项':>14}"
          f"{'Score':>14}{'高阶占比':>12}")
    for beta in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
        betas = (beta,) * N_OBJ
        t = taylor_terms(ps, ALPHAS, betas)
        exact = product_score(ps, ALPHAS, betas)
        high = (t[2] + t[3]) / exact
        print(f"{beta:>8.2f}{t[0]:>14.4f}{t[1]:>14.4f}{t[2]:>14.4f}{t[3]:>14.6f}"
              f"{exact:>14.4f}{high:>12.6f}")
    t = taylor_terms(ps, ALPHAS, (1.0,) * N_OBJ)
    print(f"\n校验:各阶项求和 = {sum(t.values()):.6f},"
          f"精确 Score = {product_score(ps, ALPHAS, (1.0,)*N_OBJ):.6f}(展开是恒等式,无截断误差)")
    print("β 越大,二阶+三阶占比越小 -- 正文「β_i → ∞ 时高阶交互项消失」。")

    # ---- 3. 一阶截断近似 vs 精确:排序一致性 ----
    print("\n" + "=" * 96)
    print("一阶截断 Score ≈ Const + Σ w_i p_i(w_i = Const·α_i/β_i)与精确乘法的一致性")
    print("=" * 96)
    cands = mock_pxtr(400, rng)
    print(f"{'β':>8}{'w_i':>22}{'平均相对误差':>14}{'最大相对误差':>14}"
          f"{'Spearman':>11}{'Top50 Jaccard':>15}")
    for beta in (0.2, 1.0, 2.0, 5.0, 20.0, 100.0):
        betas = (beta,) * N_OBJ
        const, ws = first_order_weights(ALPHAS, betas)
        exact = [product_score(p, ALPHAS, betas) for p in cands]
        approx = [const + sum(ws[k] * p[k] for k in range(N_OBJ)) for p in cands]
        errs = [abs(a - e) / e for a, e in zip(approx, exact)]
        wstr = "(" + ",".join(f"{w:.3g}" for w in ws) + ")"
        print(f"{beta:>8.2f}{wstr:>22}{sum(errs)/len(errs):>14.6f}{max(errs):>14.6f}"
              f"{spearman(exact, approx):>11.4f}{topk_jaccard(exact, approx, 50):>15.4f}")
    print("\nβ 从 0.2 升到 100,一阶近似的相对误差从百分量级降到 1e-4,Spearman→1.0,")
    print("Top-50 集合完全重合 -- 广义乘法公式在大 β 下与线性加法等价。")

    # ---- 4. 梯度视角:协同 vs 独立 ----
    print("\n" + "=" * 96)
    print("梯度视角:∂Score/∂p_1 随 p_2 的变化(β_1=β_2=1, α_1=α_2=1;加法取 w_1=w_2=1)")
    print("=" * 96)
    print(f"{'p_2':>8}{'乘法解析 (β_2+α_2p_2)':>24}{'乘法数值差分':>16}"
          f"{'加法 ∂/∂p_1 = w_1':>20}{'乘法/加法':>12}")
    eps, b1, b2 = 1e-6, 1.0, 1.0
    for p2 in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        ana = b2 + 1.0 * p2
        num = (product_score((0.5 + eps, p2), (1.0, 1.0), (b1, b2))
               - product_score((0.5 - eps, p2), (1.0, 1.0), (b1, b2))) / (2 * eps)
        print(f"{p2:>8.1f}{ana:>24.6f}{num:>16.6f}{1.0:>20.6f}{ana / 1.0:>12.4f}")
    print("\n乘法下 p_1 的边际价值随 p_2 线性上升(正向协同);加法下恒为 w_1,目标间彼此独立。")
    print("正文结论:加法=补偿型机制,乘法=协同增强型(Synergistic)机制。")
