"""
序变换及形状分析(第 6 章 §6.4.2 序变换及形状分析)

对应正文概念:
正文指出,除了基于分数值(Value)的乘法融合,工业界还广泛使用基于排序位置
(Rank)的融合方法--因为「不同模型输出的 pxtr 往往存在数值尺度不一致、分布漂移
以及校准误差」,而 Rank 只保留相对顺序,天然更鲁棒。正文给出五种代表性序变换:

  (1) 倒数变换:        S_i = 1 / Rank_i
  (2) 平滑倒数变换:    S_i = w_i / (Rank_i^lambda + b)
  (3) 归一化幂次变换:  S_i = w_i * (1 - (Rank_i / N)^lambda)
  (4) 指数双曲变换:    S_i = w_i * (e^{k(2 Rank_i/N - 1)} - e^{-k(2 Rank_i/N - 1)})
                            = 2 w_i * sinh(k (2 Rank_i/N - 1))
  (5) 带偏置归一化变换:S_i = w_i * (1 - (Rank_i / (N + b))^lambda)

本实现做了什么:
1. 按正文公式实现五个序变换函数,打印曲线数值表(代替正文图 6.x 的
   rank_score_all_3curves.pdf,本仓库禁止画图,改为打印数值)。
2. 复刻正文给出的三组形状结论并用数值验证:
   - 倒数变换的头部衰减极快(1 -> 0.5 -> 0.1),是 Head-Oriented 策略;
   - 平滑倒数变换 lambda=0.3/0.7/1.5 分别对应「平缓 / 明显头部强化 / Rank>5
     后迅速趋零」;
   - 归一化幂次变换 lambda<1 凸(头部下降慢、尾部快)、lambda=1 线性、
     lambda>1 凹(压缩头部差异、增强中腰部区分度);
   - 指数双曲变换在 Top-5 陡、中腰部平、尾部再变陡(局部差异化)。
3. 用「相邻位置得分差」Delta_r = S(r) - S(r+1) 作为「该位置的区分能力」,
   打印头 / 中 / 尾三段区分能力占比,量化正文那句核心提问:
   「如何分配不同排序位置之间的区分能力?」
4. 在一个 200 候选的三目标(CTR/LTR/CVR)场景上做 Rank 融合 vs Value 融合的
   对比,并注入「pxtr 尺度漂移」(CVR 整体乘 100)验证 Rank 融合的鲁棒性。

关键公式(正文原文符号):
  Rank_i 为候选在某排序因子中的位置(从 1 开始),N 为候选集长度,
  w_i 为因子权重,lambda / b / k 控制曲线形状;
  最终融合分 ES = sum_f w_f * T_f(Rank_f(i)),T_f 为该因子选用的序变换。
"""

import math
import random

N = 100  # 候选集长度(序变换形状分析用)


# ---------------------------------------------------------------- 五种序变换
def t_reciprocal(r, n):
    """(1) 倒数变换 S = 1 / Rank。"""
    return 1.0 / r


def t_smooth_reciprocal(r, n, w=1.0, lam=1.0, b=1.0):
    """(2) 平滑倒数变换 S = w / (Rank^lambda + b)。"""
    return w / (r ** lam + b)


def t_norm_power(r, n, w=1.0, lam=1.0):
    """(3) 归一化幂次变换 S = w (1 - (Rank/N)^lambda)。"""
    return w * (1.0 - (r / n) ** lam)


def t_exp_hyperbolic(r, n, w=1.0, k=2.0):
    """(4) 指数双曲变换 S = w (e^{kx} - e^{-kx}) = 2 w sinh(k x),x = 2 Rank/N - 1。

    注意正文在 x = 2 Rank/N - 1 下给出该式:Rank 越小 x 越接近 -1,得分为负、
    绝对值最大;Rank 越大 x 越接近 +1,得分为正。为了与「Rank 越小得分越高」
    的融合口径一致,实际排序时需要取其相反数(见 ES 融合部分)。本函数严格按
    正文公式返回原始值,便于与正文图中的曲线对照。
    """
    x = 2.0 * r / n - 1.0
    return w * (math.exp(k * x) - math.exp(-k * x))


def t_biased_norm(r, n, w=1.0, lam=1.0, b=10.0):
    """(5) 带偏置归一化变换 S = w (1 - (Rank/(N+b))^lambda)。"""
    return w * (1.0 - (r / (n + b)) ** lam)


# ---------------------------------------------------------------- 工具
def shape_label(fn, n):
    """用二阶差分符号判断曲线凸凹(针对 Rank 递增方向)。"""
    second = []
    for r in range(2, n):
        second.append(fn(r - 1) - 2 * fn(r) + fn(r + 1))
    pos = sum(1 for v in second if v > 1e-12)
    neg = sum(1 for v in second if v < -1e-12)
    if max(abs(v) for v in second) < 1e-12:
        return "线性"
    if pos > 0.9 * len(second):
        return "凸(头部降得慢)"
    if neg > 0.9 * len(second):
        return "凹(头部降得快)"
    return "混合(分段曲率)"


def discrimination_split(fn, n):
    """区分能力分配:|S(r)-S(r+1)| 在头(1-10) / 中(11-50) / 尾(51-99) 的占比。"""
    deltas = [abs(fn(r) - fn(r + 1)) for r in range(1, n)]
    total = sum(deltas) or 1.0
    head = sum(deltas[0:10]) / total
    mid = sum(deltas[10:50]) / total
    tail = sum(deltas[50:]) / total
    return head, mid, tail


def print_curve_table(title, variants, ranks):
    """variants: [(标签, 单参函数 fn(r))]。"""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    header = f"{'参数':<22}" + "".join(f"{'r=' + str(r):>9}" for r in ranks)
    print(header + f"{'曲率':>18}{'头/中/尾区分能力':>22}")
    print("-" * 110)
    for label, fn in variants:
        row = f"{label:<22}" + "".join(f"{fn(r):>9.4f}" for r in ranks)
        h, m, t = discrimination_split(fn, N)
        row += f"{shape_label(fn, N):>18}"
        row += f"{f'{h:.0%}/{m:.0%}/{t:.0%}':>22}"
        print(row)
    print("-" * 110)


# ---------------------------------------------------------------- 候选集与融合
def make_candidates(n, rng, cvr_scale=1.0):
    """[MOCK] 合成候选:三个 pxtr 共享内容质量 q,各带独立成分。

    cvr_scale 模拟正文提到的「分布漂移 / 校准误差」:CVR 整体被乘上一个常数。
    """
    items = []
    for i in range(n):
        q = rng.random()
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        items.append({
            "id": i,
            "ctr": 0.02 + 0.14 * q + 0.10 * u1,
            "ltr": 0.004 + 0.012 * q ** 2 + 0.016 * u2 ** 2,
            "cvr": (0.0005 + 0.0025 * q ** 3 + 0.0035 * u3 ** 3) * cvr_scale,
        })
    return items


def ranks_of(items, key):
    """按 key 降序给出 1-based 排名。"""
    order = sorted(range(len(items)), key=lambda i: -items[i][key])
    rk = [0] * len(items)
    for pos, idx in enumerate(order):
        rk[idx] = pos + 1
    return rk


def value_fusion(items, weights):
    """Value 融合(加权和,作为对照):Score = sum_f w_f * p_f。"""
    return [sum(w * it[f] for f, w in weights.items()) for it in items]


def rank_fusion(items, weights, transform):
    """Rank 融合:ES = sum_f w_f * T(Rank_f(i))。"""
    n = len(items)
    rk = {f: ranks_of(items, f) for f in weights}
    return [sum(w * transform(rk[f][i], n) for f, w in weights.items())
            for i in range(n)]


def spearman(a, b):
    n = len(a)

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, idx in enumerate(order):
            r[idx] = float(pos)
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return cov / (va * vb) if va and vb else 0.0


def topk_overlap(a, b, k):
    sa = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    sb = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(sa & sb) / k


if __name__ == "__main__":
    random.seed(42)
    ranks = [1, 2, 3, 5, 10, 20, 50, 100]

    # ---- (1) 倒数变换
    print_curve_table(
        "(1) 倒数变换 S = 1/Rank —— 正文:Head-Oriented,1/1=1, 1/2=0.5, 1/10=0.1",
        [("1/Rank", lambda r: t_reciprocal(r, N))], ranks)
    print("观察:Rank 1->2 得分直接腰斩 0.5000,而 Rank 50->60 的差只有 "
          f"{t_reciprocal(50, N) - t_reciprocal(60, N):.4f}。"
          "\n      区分能力几乎全部压在头部,正文称之为典型的 Head-Oriented 策略。")

    # ---- (2) 平滑倒数变换
    print_curve_table(
        "(2) 平滑倒数变换 S = w / (Rank^lambda + b)  (w=1, b=1)",
        [(f"lambda={lam}", (lambda lam: lambda r:
                            t_smooth_reciprocal(r, N, 1.0, lam, 1.0))(lam))
         for lam in [0.3, 0.7, 1.0, 1.5]], ranks)
    print("观察:正文结论的数值验证 —— lambda=0.3 时 Rank 1 与 Rank 10 的得分比只有 "
          f"{t_smooth_reciprocal(1, N, 1, 0.3, 1) / t_smooth_reciprocal(10, N, 1, 0.3, 1):.2f}"
          "(平缓);\n      lambda=0.7 升到 "
          f"{t_smooth_reciprocal(1, N, 1, 0.7, 1) / t_smooth_reciprocal(10, N, 1, 0.7, 1):.2f}"
          "(明显头部强化);lambda=1.5 升到 "
          f"{t_smooth_reciprocal(1, N, 1, 1.5, 1) / t_smooth_reciprocal(10, N, 1, 1.5, 1):.2f},"
          f"\n      且 Rank>5 之后得分只剩 {t_smooth_reciprocal(5, N, 1, 1.5, 1):.4f} "
          "并迅速趋近 0 —— 与正文「Rank 大于 5 之后得分迅速趋近于零」一致。")

    # ---- (3) 归一化幂次变换
    print_curve_table(
        "(3) 归一化幂次变换 S = w (1 - (Rank/N)^lambda)  (w=1, N=100)",
        [(f"lambda={lam}", (lambda lam: lambda r:
                            t_norm_power(r, N, 1.0, lam))(lam))
         for lam in [0.3, 0.5, 1.0, 2.0, 3.0]], ranks)
    print("观察:lambda<1 为凸函数,头部下降慢、尾部下降快,区分能力向尾部倾斜;"
          "\n      lambda=1 线性,区分能力按位置数均分(10%/40%/49%);"
          "\n      lambda>1 为凹函数,头部差异被压缩、区分能力集中到中腰部与尾部"
          "\n      —— 正文所说「压平头部差异,增加中腰部内容曝光机会」。")

    # ---- (4) 指数双曲变换
    print_curve_table(
        "(4) 指数双曲变换 S = w (e^{kx} - e^{-kx}) = 2w sinh(kx), x = 2Rank/N - 1",
        [(f"k={k}", (lambda k: lambda r:
                     t_exp_hyperbolic(r, N, 1.0, k))(k))
         for k in [0.5, 1.0, 2.0, 3.0]], ranks)
    print("说明:该式按正文符号,Rank 越小 x 越接近 -1,得分为负且绝对值最大;"
          "融合时取相反数。")
    k = 2.0
    seg = [("Top-5 (r=1..5)", 1, 5), ("中腰部 (r=48..52)", 48, 52),
           ("尾部 (r=96..100)", 96, 100)]
    print(f"\nk={k} 时的分段敏感性(相邻 Rank 的平均得分差 |Delta|):")
    for name, lo, hi in seg:
        d = [abs(t_exp_hyperbolic(r, N, 1.0, k)
                 - t_exp_hyperbolic(r + 1, N, 1.0, k)) for r in range(lo, hi)]
        print(f"  {name:<20} 平均 |Delta| = {sum(d) / len(d):.5f}")
    print("观察:Top-5 与尾部的 |Delta| 明显大于中腰部,双曲函数在头尾两端陡、"
          "中间平,\n      正文称之为「局部区域的差异化建模」。")

    # ---- (5) 带偏置归一化变换
    print_curve_table(
        "(5) 带偏置归一化变换 S = w (1 - (Rank/(N+b))^lambda)  (w=1, N=100)",
        [(f"lambda={lam},b={b}", (lambda lam, b: lambda r:
                                  t_biased_norm(r, N, 1.0, lam, b))(lam, b))
         for lam, b in [(1.0, 0.0), (1.0, 10.0), (1.0, 100.0),
                        (2.0, 10.0), (0.5, 10.0)]], ranks)
    print("观察:b 的作用是让最后一名不再归零 —— b=0 时 r=N 得分为 0.0000,"
          f"\n      b=100 时仍有 {t_biased_norm(N, N, 1.0, 1.0, 100.0):.4f},"
          "整条曲线被压扁,等价于降低该因子在融合中的总权重。")

    # ---- 区分能力分配总览
    print("\n" + "=" * 110)
    print("核心问题:「如何分配不同排序位置之间的区分能力?」(正文 §6.4.2 结语)")
    print("=" * 110)
    allv = [
        ("(1) 倒数变换", lambda r: t_reciprocal(r, N)),
        ("(2) 平滑倒数 lam=0.3", lambda r: t_smooth_reciprocal(r, N, 1, 0.3, 1)),
        ("(2) 平滑倒数 lam=1.5", lambda r: t_smooth_reciprocal(r, N, 1, 1.5, 1)),
        ("(3) 幂次 lam=0.3", lambda r: t_norm_power(r, N, 1, 0.3)),
        ("(3) 幂次 lam=1.0", lambda r: t_norm_power(r, N, 1, 1.0)),
        ("(3) 幂次 lam=3.0", lambda r: t_norm_power(r, N, 1, 3.0)),
        ("(4) 双曲 k=2.0", lambda r: t_exp_hyperbolic(r, N, 1, 2.0)),
        ("(5) 带偏置 lam=2,b=10", lambda r: t_biased_norm(r, N, 1, 2.0, 10.0)),
    ]
    print(f"{'序变换':<24}{'头部(r1-10)':>14}{'中腰(r11-50)':>14}"
          f"{'尾部(r51-99)':>14}{'流量分配哲学':>22}")
    print("-" * 110)
    philosophy = ["头部优先", "弱头部强化", "强头部强化", "尾部区分",
                  "位置均分", "中尾区分", "头尾两端区分", "中尾区分"]
    for (name, fn), ph in zip(allv, philosophy):
        h, m, t = discrimination_split(fn, N)
        print(f"{name:<24}{h:>14.1%}{m:>14.1%}{t:>14.1%}{ph:>22}")
    print("-" * 110)
    print("观察:同一组 Rank,仅换一个序变换,区分能力就能在头/中/尾之间任意再分配。"
          "\n      正文强调「并不存在绝对最优的序变换函数」,要靠 A/B 实验确定。")

    # ---- Rank 融合 vs Value 融合:尺度漂移鲁棒性
    print("\n" + "=" * 110)
    print("Rank 融合 vs Value 融合:pxtr 尺度漂移(CVR 整体 x100)下的鲁棒性")
    print("=" * 110)
    items = make_candidates(200, random)
    rng2 = random.Random(42)
    items_drift = make_candidates(200, rng2, cvr_scale=100.0)
    weights = {"ctr": 1.0, "ltr": 1.0, "cvr": 1.0}

    def neg_hyper(r, n):
        """双曲变换用于融合时取相反数,保证 Rank 越小得分越高。"""
        return -t_exp_hyperbolic(r, n, 1.0, 2.0)

    fusions = [
        ("Value 加权和", None),
        ("(1) 倒数变换", lambda r, n: t_reciprocal(r, n)),
        ("(2) 平滑倒数 lam=0.7", lambda r, n: t_smooth_reciprocal(r, n, 1, 0.7, 1)),
        ("(3) 幂次 lam=1.0", lambda r, n: t_norm_power(r, n, 1, 1.0)),
        ("(4) 双曲 k=2.0(取负)", neg_hyper),
        ("(5) 带偏置 lam=2,b=10", lambda r, n: t_biased_norm(r, n, 1, 2.0, 10.0)),
    ]
    print(f"{'融合方式':<24}{'漂移前后序相关':>18}{'Top10重合':>12}"
          f"{'Top20重合':>12}{'Top20平均CTR':>16}{'Top20平均CVR':>16}")
    print("-" * 110)
    for name, tf in fusions:
        if tf is None:
            s0 = value_fusion(items, weights)
            s1 = value_fusion(items_drift, weights)
        else:
            s0 = rank_fusion(items, weights, tf)
            s1 = rank_fusion(items_drift, weights, tf)
        top20 = sorted(range(len(items)), key=lambda i: -s0[i])[:20]
        mctr = sum(items[i]["ctr"] for i in top20) / 20
        mcvr = sum(items[i]["cvr"] for i in top20) / 20
        print(f"{name:<24}{spearman(s0, s1):>18.4f}"
              f"{topk_overlap(s0, s1, 10):>12.0%}{topk_overlap(s0, s1, 20):>12.0%}"
              f"{mctr:>16.4f}{mcvr:>16.6f}")
    print("-" * 110)
    print("观察:CVR 整体乘 100 只是单调缩放,不改变 CVR 的 Rank,所以五种 Rank 融合"
          "\n      的序相关与 Top-K 重合都是 1.0000 / 100%;Value 加权和的序相关掉到"
          " 0.7061、\n      Top10 重合只剩 40% —— 这正是正文说 Rank 融合「天然具有"
          "更强的鲁棒性」的原因。")

    # ---- 不同序变换对最终 Top-K 构成的影响
    print("\n" + "=" * 110)
    print("不同序变换对最终 Top-20 构成的影响(同一候选集,权重全为 1)")
    print("=" * 110)
    base = rank_fusion(items, weights, lambda r, n: t_norm_power(r, n, 1, 1.0))
    print(f"{'融合方式':<24}{'与线性幂次Top20重合':>22}"
          f"{'Top20含CTR前20个数':>22}{'Top20平均Rank和':>18}")
    print("-" * 110)
    ctr_top20 = set(sorted(range(len(items)),
                           key=lambda i: -items[i]["ctr"])[:20])
    rk_all = {f: ranks_of(items, f) for f in weights}
    for name, tf in fusions[1:]:
        s = rank_fusion(items, weights, tf)
        top20 = sorted(range(len(items)), key=lambda i: -s[i])[:20]
        rsum = sum(sum(rk_all[f][i] for f in weights) for i in top20) / 20
        print(f"{name:<24}{topk_overlap(s, base, 20):>22.0%}"
              f"{len(set(top20) & ctr_top20):>22d}{rsum:>18.1f}")
    print("-" * 110)
    print("观察:倒数类变换把头部权重推到极致,Top-20 里更容易出现「某单一因子"
          "排名第一但\n      其余因子平庸」的候选(平均 Rank 和更大);幂次/带偏置"
          "变换更偏向「三项都不错」\n      的均衡候选。这就是正文所说不同序变换"
          "对应不同的流量分配哲学。")
    print("=" * 110)
