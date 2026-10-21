"""
乘法融合公式及形状分析（第 6 章 §6.4.1 乘法公式及形状分析）

对应正文概念：
正文给出精排多目标融合的乘法公式（Multiplicative Formula）：
    s_i = (beta_i + alpha_i * p_i) ^ gamma_i
整体融合分为各因子相乘：
    Score = prod_{i=1..N} (beta_i + alpha_i * p_i) ^ gamma_i
三个超参数分别对应三个控制维度（正文原话）：
    alpha_i 决定区分度（Scaling Factor）；
    beta_i  决定基础贡献（Bias Term）；
    gamma_i 决定流量分配形状（Exponent Term）。
并指出：gamma > 1 时为凸函数，强化头部效应（马太效应）；0 < gamma < 1 时为凹
函数，压缩头尾差距、给中腰部更多曝光；gamma = 1 时退化为线性变换。

本实现做了什么（正文的图 6-x 是曲线图，这里按要求改为打印数值表）：
 1) 形状表：固定其余两参数，分别扫 alpha / beta / gamma，打印 p=0.0~1.0 上的
    s 值表，并给出「归一化后的头尾比」量化区分度；
 2) 排序影响：构造 CTR/LTR/CVR 三因子的候选集，扫 gamma 观察 Top-10 的
    流量集中度（基尼系数、Top-1 占比、与各单因子排序的 Spearman 相关）；
 3) 退化验证：数值验证 gamma=1 时 (beta + alpha*p)^1 == beta + alpha*p；
 4) 稀疏因子的 beta 作用：演示正文说的「稀疏目标直接相乘导致得分迅速衰减」，
    以及加偏置项后数值分布如何被稳定下来。

关键公式：
    s_i   = (beta_i + alpha_i * p_i) ^ gamma_i
    Score = prod_i s_i
    基尼系数 G = sum_i (2i - n - 1) * x_(i) / (n * sum_i x_i)
"""

import math
import random


def shape(p, alpha, beta, gamma):
    """乘法公式单因子变换 s = (beta + alpha * p)^gamma。"""
    base = beta + alpha * p
    if base <= 0:
        return 0.0
    return base ** gamma


def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    tot = sum(xs)
    if tot <= 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * tot)


def spearman(a, b):
    """a、b 是两组分数，返回其秩相关。"""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def print_shape_table(title, param_name, values, make_params):
    """打印 p 网格上的 s 值表；每行一组超参数。"""
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"\n{title}")
    print(f"{param_name:>10}" + "".join(f"{f'p={g:g}':>10}" for g in grid)
          + f"{'头尾比':>12}{'曲率':>10}")
    print("-" * 104)
    for v in values:
        a, b, g = make_params(v)
        row = [shape(p, a, b, g) for p in grid]
        # 归一化头尾比：s(1.0)/s(0.1)，衡量该变换把高分与低分拉开多少
        ratio = row[-1] / row[1] if row[1] > 0 else float("inf")
        # 曲率判别：二阶差分符号
        mid = shape(0.5, a, b, g)
        lin = (shape(0.0, a, b, g) + shape(1.0, a, b, g)) / 2
        curv = "凸(头部强化)" if mid < lin - 1e-9 else (
            "凹(压平头部)" if mid > lin + 1e-9 else "线性")
        print(f"{v:>10.2f}" + "".join(f"{x:>10.4f}" for x in row)
              + f"{ratio:>12.3f}{curv:>10}")


def make_candidates(n, rng):
    """[MOCK] 合成候选集：每个候选有 CTR / LTR / CVR 三个 pxtr。

    三个 pxtr 共享一部分内容质量 q，但各自带有独立成分——这与真实系统一致：
    点击率高的内容未必转化率高，否则任何单调融合公式都会给出同一个排序。
    """
    items = []
    for i in range(n):
        q = rng.random()                      # 共享的内容质量
        u1, u2, u3 = rng.random(), rng.random(), rng.random()
        ctr = min(0.95, max(0.001, 0.02 + 0.14 * q + 0.10 * u1))
        ltr = min(0.95, max(0.0001, 0.004 + 0.012 * q ** 2 + 0.016 * u2 ** 2))
        cvr = min(0.95, max(0.00001,
                            0.0005 + 0.0025 * q ** 3 + 0.0035 * u3 ** 3))
        items.append({"id": f"item{i:03d}", "ctr": ctr, "ltr": ltr, "cvr": cvr})
    return items


FACTORS = ["ctr", "ltr", "cvr"]


def fuse(items, params):
    """params: {factor: (alpha, beta, gamma)}；返回每个候选的融合分。"""
    out = []
    for it in items:
        s = 1.0
        for f in FACTORS:
            a, b, g = params[f]
            s *= shape(it[f], a, b, g)
        out.append(s)
    return out


if __name__ == "__main__":
    random.seed(42)

    print("=" * 104)
    print("§6.4.1 乘法公式 s_i = (beta_i + alpha_i * p_i) ^ gamma_i 形状分析")
    print("=" * 104)

    print_shape_table(
        "(a) 乘法系数 alpha 的影响（beta=0.5, gamma=1.0）—— 控制区分度",
        "alpha", [0.2, 0.5, 1.0, 2.0, 5.0],
        lambda v: (v, 0.5, 1.0))
    print("观察：alpha 越大，同样的 pxtr 差异被放大得越多（头尾比 1.346 -> 5.500），"
          "\n      排序更集中于头部；alpha 小则得分趋于平滑、探索性更强。")

    print_shape_table(
        "(b) 偏置项 beta 的影响（alpha=1.0, gamma=1.0）—— 控制基础贡献",
        "beta", [0.0, 0.1, 0.5, 1.0, 5.0],
        lambda v: (1.0, v, 1.0))
    print("观察：beta 抬高基准水平，等于稀释该因子的相对影响力（头尾比单调下降）；"
          "\n      beta=0 时 p 接近 0 的候选得分直接归零，这正是正文说的"
          "「稀疏目标直接\n      参与乘法会让整体得分迅速衰减」。")

    print_shape_table(
        "(c) 指数项 gamma 的影响（alpha=1.0, beta=0.5）—— 控制流量分配形状",
        "gamma", [0.3, 0.5, 1.0, 2.0, 3.0],
        lambda v: (1.0, 0.5, v))
    print("观察：gamma<1 为凹函数，压缩头尾差距、给中腰部更多机会；gamma=1 退化为"
          "\n      线性；gamma>1 为凸函数，强化头部效应（马太效应）。gamma 是正文"
          "\n      强调的「最需要重点调优的参数」。")

    # ---------------- gamma=1 退化验证 ----------------
    print("\n" + "=" * 104)
    print("退化验证：gamma=1 时 (beta + alpha*p)^1 == beta + alpha*p")
    print("=" * 104)
    max_err = max(abs(shape(p / 20, 1.3, 0.4, 1.0) - (0.4 + 1.3 * p / 20))
                  for p in range(21))
    print(f"p 在 [0,1] 上取 21 个点，最大绝对误差 = {max_err:.3e}（数值等价）")

    # ---------------- 排序影响 ----------------
    items = make_candidates(200, random)
    print("\n" + "=" * 104)
    print("gamma 对最终排序与流量分配的影响（200 个候选，CTR/LTR/CVR 三因子乘法融合）")
    print("=" * 104)
    print(f"{'gamma(三因子同值)':>18}{'Top1占比':>12}{'Top10占比':>12}{'基尼系数':>12}"
          f"{'与CTR序相关':>14}{'与CVR序相关':>14}{'Top10与基线重合':>18}")
    print("-" * 104)
    base_top10 = None
    ctr_scores = [it["ctr"] for it in items]
    cvr_scores = [it["cvr"] for it in items]
    for g in [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]:
        params = {"ctr": (1.0, 0.1, g), "ltr": (1.0, 0.1, g),
                  "cvr": (1.0, 0.1, g)}
        sc = fuse(items, params)
        tot = sum(sc)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        top10 = set(order[:10])
        if base_top10 is None:
            base_top10 = top10
        top1 = sc[order[0]] / tot
        t10 = sum(sc[i] for i in order[:10]) / tot
        print(f"{g:>18.2f}{top1:>12.4%}{t10:>12.4%}{gini(sc):>12.4f}"
              f"{spearman(sc, ctr_scores):>14.4f}"
              f"{spearman(sc, cvr_scores):>14.4f}"
              f"{len(top10 & base_top10) / 10:>18.0%}")
    print("-" * 104)
    print("（「与基线重合」以 gamma=0.2 的 Top10 为基线）")
    print("观察 1：三个因子取同一个 gamma 时，Score = (prod_i (beta+alpha*p_i))^gamma，"
          "是基线分的\n        单调幂变换，因此排序完全不变（Top10 重合恒为 100%，"
          "序相关恒为 0.9792/0.4515）。\n        gamma 同值只改变「分数形状」而不改变"
          "「顺序」。")
    print("观察 2：但分数形状本身就是流量形状——gamma 0.2 -> 8 时基尼系数从 0.0291 "
          "升到 0.7243、\n        Top1 流量占比从 0.55% 升到 5.96%。若下游按分数做"
          "概率抽样 / 分数配额，gamma\n        就直接决定马太效应强度，这正是正文说"
          "的 gamma 决定「流量分配形状」。真正改变\n        排序的是「各因子取不同 "
          "gamma」，见下一张表。")

    # ---------------- 不同因子设不同 gamma ----------------
    print("\n" + "=" * 104)
    print("业务调控：对单个因子加权（其余因子固定 alpha=1,beta=0.1,gamma=1）")
    print("=" * 104)
    base_params = {f: (1.0, 0.1, 1.0) for f in FACTORS}
    base_sc = fuse(items, base_params)
    base_order = sorted(range(len(items)), key=lambda i: -base_sc[i])
    base_set = set(base_order[:20])
    print(f"{'调控':<22}{'Top20平均CTR':>14}{'Top20平均LTR':>14}"
          f"{'Top20平均CVR':>14}{'与基线Top20重合':>18}")
    print("-" * 104)
    configs = [("基线 (全 gamma=1)", {}),
               ("cvr gamma=3 (强化转化)", {"cvr": (1.0, 0.1, 3.0)}),
               ("cvr gamma=0.3 (弱化转化)", {"cvr": (1.0, 0.1, 0.3)}),
               ("ctr alpha=5 (放大点击)", {"ctr": (5.0, 0.1, 1.0)}),
               ("ltr beta=2 (稀释互动)", {"ltr": (1.0, 2.0, 1.0)})]
    for name, override in configs:
        pr = dict(base_params)
        pr.update(override)
        sc = fuse(items, pr)
        order = sorted(range(len(items)), key=lambda i: -sc[i])[:20]
        mctr = sum(items[i]["ctr"] for i in order) / 20
        mltr = sum(items[i]["ltr"] for i in order) / 20
        mcvr = sum(items[i]["cvr"] for i in order) / 20
        ov = len(set(order) & base_set) / 20
        print(f"{name:<22}{mctr:>14.4f}{mltr:>14.5f}{mcvr:>14.6f}{ov:>18.0%}")
    print("-" * 104)

    # ---------------- 稀疏因子数值稳定性 ----------------
    print("\n" + "=" * 104)
    print("稀疏因子的 beta 作用：CVR 量级在 1e-4~1e-2，直接相乘导致得分数值塌陷")
    print("=" * 104)
    print(f"{'CVR 的 beta':>14}{'融合分最小值':>18}{'融合分最大值':>18}"
          f"{'最大/最小':>14}{'与CVR序相关':>14}{'Top20含CVR头部':>18}")
    print("-" * 104)
    cvr_top20 = set(sorted(range(len(items)),
                           key=lambda i: -items[i]["cvr"])[:20])
    for b in [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]:
        pr = dict(base_params)
        pr["cvr"] = (1.0, b, 1.0)
        sc = fuse(items, pr)
        lo, hi = min(sc), max(sc)
        ratio = hi / lo if lo > 0 else float("inf")
        top20 = set(sorted(range(len(sc)), key=lambda i: -sc[i])[:20])
        print(f"{b:>14.3f}{lo:>18.3e}{hi:>18.3e}{ratio:>14.2f}"
              f"{spearman(sc, cvr_scores):>14.4f}"
              f"{len(top20 & cvr_top20):>18d}")
    print("-" * 104)
    print("观察：beta=0 时融合分动态范围最大（最大/最小 22.94）、该因子影响力也最强"
          "\n      （与 CVR 序相关 0.9455，Top20 里有 13 个 CVR 头部物品）；beta 增到 10"
          " 后\n      相关性降到 0.4211、Top20 只剩 6 个，动态范围收敛到 3.08。beta"
          "既稳定数值分布，\n      又单调稀释该因子的基础影响力——正文所说 beta「调节"
          "不同目标在融合中的基础影响力」。")
    print("=" * 104)
