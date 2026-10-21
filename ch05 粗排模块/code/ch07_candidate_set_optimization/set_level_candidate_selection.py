"""
候选集集合优化:覆盖度 / 聚集度 / 探索率约束下的集合选择(第 5 章 §5.7 粗排候选集集合优化)

对应正文概念
------------
正文指出粗排真正优化的不是 P(click | u, i) 而是

    P(Value | CandidateSet)

即「候选集优化(Candidate Set Optimization)」问题。正文给出的集合级诉求与约束:

1. 兴趣覆盖(Interest Coverage):
       Coverage(CandidateSet) -> max
   正文算例的用户兴趣分布是 篮球 40% / 足球 30% / 游戏 20% / 科技 10%,
   若候选集全来自篮球,则「大量潜在兴趣方向被提前过滤」。
2. 候选集聚集度(Candidate Concentration),用两类约束压制:
       Similarity(C_i, C_j) < τ
       Count(Category_i) ≤ K
3. 探索注入(避免贪心利用导致创作者生态固化):
       CandidateSet = (1 - ε) TopK + ε Explore,  ε = 5% ~ 20%
4. 评价指标(正文):
       Recall@K = |Relevant ∩ CandidateSet| / |Relevant|
   以及多样性、聚集度、探索率、有效兴趣数、用户偏好垂类数量。

本实现做了什么
--------------
1. [MOCK] 合成召回池:每个 item 有一个「下游真实价值 val」和一个「粗排分 score」,
   score = val + bias[category] + 噪声。bias 让篮球被系统性高估、科技被低估,
   于是逐点 TopK(score) 会把候选集塞满篮球 —— 正文所说的 Candidate Concentration。
   Relevant(下游精排真正偏好的 60 个 item)按用户兴趣分布在 4 个垂类间分配,
   每个垂类内取 val 最高者,因此冷门垂类的优质内容只能靠集合级约束保住。
2. 实现 5 种候选集构造策略并用同一套集合级指标横评:
     A. 逐点 TopK(score)               —— 正文的「贪心利用」基线
     B. 兴趣配额 Quota(按兴趣分布分配)
     C. 类目上限 Count(Category_i) ≤ K
     D. 相似度约束 Similarity(C_i,C_j) < τ(贪心去重)
     E. 集合级贪心(submodular 风格):相关性 + 覆盖缺口 − 相似度惩罚,
        再按 (1-ε)TopK + ε Explore 注入探索
3. 指标:Recall@K、集合平均分、Coverage、有效兴趣数、最大类目占比(聚集度)、
   平均对内相似度、探索率、与兴趣分布的总变差距离。
4. 两组敏感性:探索比例 ε(正文 5%~20%)、集合价值函数里的覆盖权重 λ_cov。

关键公式
--------
    Coverage(S) = Σ_c w_c · 1[ |{i ∈ S : cat_i = c}| ≥ m ]     (w_c = 用户兴趣权重)
    集合价值 F(S) = λ_rel Σ_{i∈S} score_i
                    + λ_cov · CovGap(S)                        (Coverage -> max 的平滑代理)
                    − λ_sim Σ_{i,j ∈ S} Sim(i, j)
    贪心:S ← S ∪ { argmax_{i ∉ S} [ F(S ∪ {i}) − F(S) ] },直到 |S| = K
    CandidateSet = (1 - ε) TopK + ε Explore
"""

from __future__ import annotations

import random

N_POOL = 3000            # [MOCK] 召回候选池大小
K_SET = 300              # 候选集容量(正文粗排输出 300~500)
N_RELEVANT = 60          # [MOCK] 下游精排真正偏好的 item 数(Relevant 集合)
EPSILON = 0.10           # 探索比例,正文 ε = 5% ~ 20%
TAU = 0.80               # Similarity(C_i, C_j) < τ
CAT_CAP_RATIO = 0.40     # Count(Category_i) ≤ K 中的 K 取 40% · K_SET
MIN_PER_INTEREST = 10    # 判定「该兴趣被覆盖」所需的最少 item 数

# 正文算例的用户兴趣分布
INTERESTS = {"篮球": 0.40, "足球": 0.30, "游戏": 0.20, "科技": 0.10}
CLUSTERS_PER_CAT = 5     # [MOCK] 每个垂类下的内容簇数,用于定义 Similarity
# [MOCK] 粗排分对各垂类的系统性偏置(链路不一致的来源,见 §5.3)
SCORE_BIAS = {"篮球": 0.030, "足球": 0.010, "游戏": 0.000, "科技": -0.008}


class Item:
    """[MOCK] 一个召回候选。

    val   : 下游精排视角的真实价值(粗排看不到)
    score : 粗排打分 = val + bias[cat] + 噪声
    cluster: 内容簇,决定 Similarity
    """

    __slots__ = ("idx", "cat", "cluster", "val", "score", "is_new", "relevant")

    def __init__(self, idx: int, cat: str, cluster: int, val: float,
                 score: float, is_new: bool):
        self.idx = idx
        self.cat = cat
        self.cluster = cluster
        self.val = val
        self.score = score
        self.is_new = is_new
        self.relevant = False


def similarity(a: Item, b: Item) -> float:
    """[MOCK] 内容相似度:同簇 0.9,同垂类不同簇 0.5,跨垂类 0.1。"""
    if a.cat != b.cat:
        return 0.1
    return 0.9 if a.cluster == b.cluster else 0.5


def build_pool(rng: random.Random) -> list[Item]:
    """[MOCK] 构造召回池,并标注 Relevant 集合。"""
    pool: list[Item] = []
    cats = list(INTERESTS)
    for i in range(N_POOL):
        cat = cats[i % len(cats)]
        cluster = rng.randrange(CLUSTERS_PER_CAT)
        val = max(1e-4, rng.gauss(0.050, 0.014))
        score = max(1e-4, val + SCORE_BIAS[cat] + rng.gauss(0.0, 0.004))
        pool.append(Item(i, cat, cluster, val, score, is_new=rng.random() < 0.18))
    # Relevant 按用户兴趣分布在垂类间分配,各垂类内取 val 最高者
    for cat, w in INTERESTS.items():
        quota = round(N_RELEVANT * w)
        cand = sorted((it for it in pool if it.cat == cat), key=lambda it: -it.val)
        for it in cand[:quota]:
            it.relevant = True
    return pool


# ---------------------------------------------------------------- 集合级指标
def cat_counts(sel: list[Item]) -> dict[str, int]:
    cnt: dict[str, int] = {}
    for it in sel:
        cnt[it.cat] = cnt.get(it.cat, 0) + 1
    return cnt


def coverage(sel: list[Item]) -> float:
    cnt = cat_counts(sel)
    return sum(w for c, w in INTERESTS.items() if cnt.get(c, 0) >= MIN_PER_INTEREST)


def effective_interests(sel: list[Item]) -> int:
    cnt = cat_counts(sel)
    return sum(1 for c in INTERESTS if cnt.get(c, 0) >= MIN_PER_INTEREST)


def max_cat_ratio(sel: list[Item]) -> float:
    return max(cat_counts(sel).values()) / len(sel)


def avg_pairwise_sim(sel: list[Item]) -> float:
    """集合内平均两两相似度(聚集度),按类目/簇计数闭式求和,无需枚举 pair。"""
    n = len(sel)
    cnt = cat_counts(sel)
    clu: dict[tuple[str, int], int] = {}
    for it in sel:
        key = (it.cat, it.cluster)
        clu[key] = clu.get(key, 0) + 1
    same_cluster = sum(c * (c - 1) // 2 for c in clu.values())
    same_cat = sum(c * (c - 1) // 2 for c in cnt.values()) - same_cluster
    total = n * (n - 1) // 2
    cross = total - same_cat - same_cluster
    return (0.9 * same_cluster + 0.5 * same_cat + 0.1 * cross) / total


def tv_distance(sel: list[Item]) -> float:
    """候选集类目分布与用户兴趣分布的总变差距离。"""
    cnt = cat_counts(sel)
    return 0.5 * sum(abs(cnt.get(c, 0) / len(sel) - w) for c, w in INTERESTS.items())


def metrics(sel: list[Item]) -> dict[str, float]:
    rel = sum(1 for it in sel if it.relevant)
    return {
        "Recall@K": rel / N_RELEVANT,
        "平均分": sum(it.score for it in sel) / len(sel),
        "真值均值": sum(it.val for it in sel) / len(sel),
        "Coverage": coverage(sel),
        "有效兴趣数": float(effective_interests(sel)),
        "最大类目占比": max_cat_ratio(sel),
        "对内相似度": avg_pairwise_sim(sel),
        "探索率": sum(1 for it in sel if it.is_new) / len(sel),
        "TV距离": tv_distance(sel),
    }


# ---------------------------------------------------------------- 五种策略
def strat_topk(pool: list[Item], k: int) -> list[Item]:
    """A. CandidateSet = TopK(score):正文的纯贪心利用。"""
    return sorted(pool, key=lambda it: -it.score)[:k]


def strat_interest_quota(pool: list[Item], k: int) -> list[Item]:
    """B. 覆盖度配额:按用户兴趣分布给每个垂类分 Quota,各自取垂类内 TopK。"""
    sel: list[Item] = []
    for cat, w in INTERESTS.items():
        quota = round(k * w)
        cand = sorted((it for it in pool if it.cat == cat), key=lambda it: -it.score)
        sel.extend(cand[:quota])
    return sel[:k]


def strat_category_cap(pool: list[Item], k: int, cap_ratio: float) -> list[Item]:
    """C. Count(Category_i) ≤ K:按分数贪心但每个类目设硬上限。"""
    cap = int(k * cap_ratio)
    used: dict[str, int] = {}
    sel: list[Item] = []
    for it in sorted(pool, key=lambda i: -i.score):
        if used.get(it.cat, 0) >= cap:
            continue
        sel.append(it)
        used[it.cat] = used.get(it.cat, 0) + 1
        if len(sel) == k:
            break
    return sel


def strat_similarity_cap(pool: list[Item], k: int, tau: float) -> list[Item]:
    """D. Similarity(C_i, C_j) < τ:贪心加入时拒绝与已选任一 item 相似度 ≥ τ 的候选。

    同簇相似度 0.9 ≥ τ=0.8,因此每个(垂类,簇)最多留 1 条;候选用尽后放宽约束补齐,
    保证 |CandidateSet| = K(工程上的兜底逻辑)。
    """
    sel: list[Item] = []
    rest: list[Item] = []
    seen: set[tuple[str, int]] = set()
    for it in sorted(pool, key=lambda i: -i.score):
        key = (it.cat, it.cluster)
        if key in seen:
            rest.append(it)
            continue
        seen.add(key)
        sel.append(it)
        if len(sel) == k:
            return sel
    sel.extend(rest[:k - len(sel)])
    return sel


def strat_set_greedy(pool: list[Item], k: int, eps: float,
                     lam_rel: float = 1.0, lam_cov: float = 10.0,
                     lam_sim: float = 0.0015) -> list[Item]:
    """E. 集合级贪心 + 探索注入(§5.7 的最小可运行集合优化版)。

    边际增益 = λ_rel · score_i
             + λ_cov · covgap_i / K       (covgap_i = 1 若垂类 cat_i 的名额还没达到 w_c·K)
             − λ_sim · Σ_{j∈S} Sim(i, j)
    其中覆盖缺口项是 Coverage(S) -> max 的平滑代理(indicator 版本梯度几乎全为 0,
    工程上用「离目标配额还差多少」代替);相似度惩罚项对应 Similarity 约束的软化形式。
    两项都是集合级信号 —— 同一个 item 的边际价值取决于已选集合 S。
    最后按 CandidateSet = (1-ε)TopK + ε Explore 用新内容/长尾填充 ε 份额。
    """
    k_exploit = int(k * (1.0 - eps))
    # 先用分数取一个工作集(工程上的「多路队列」做法),控制贪心复杂度
    ranked = sorted(pool, key=lambda it: -it.score)
    work = ranked[:min(len(pool), 8 * k)]
    target = {c: w * k for c, w in INTERESTS.items()}
    cnt: dict[str, int] = {c: 0 for c in INTERESTS}
    clu: dict[tuple[str, int], int] = {}
    sel: list[Item] = []
    chosen: set[int] = set()
    for _ in range(k_exploit):
        best, best_gain = None, -1e18
        n_sel = len(sel)
        for it in work:
            if it.idx in chosen:
                continue
            d_cov = 1.0 if cnt[it.cat] < target[it.cat] else 0.0
            same_cluster = clu.get((it.cat, it.cluster), 0)
            same_cat = cnt[it.cat] - same_cluster
            cross = n_sel - cnt[it.cat]
            d_sim = 0.9 * same_cluster + 0.5 * same_cat + 0.1 * cross
            gain = lam_rel * it.score + lam_cov * d_cov / k - lam_sim * d_sim
            if gain > best_gain:
                best, best_gain = it, gain
        assert best is not None
        sel.append(best)
        chosen.add(best.idx)
        cnt[best.cat] += 1
        key = (best.cat, best.cluster)
        clu[key] = clu.get(key, 0) + 1
    # ε Explore:新内容 / 长尾内容,按分数取最优者
    sel.extend([it for it in ranked
                if it.is_new and it.idx not in chosen][:k - len(sel)])
    if len(sel) < k:
        sel.extend([it for it in ranked if it.idx not in chosen][:k - len(sel)])
    return sel


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)
    pool = build_pool(rng)

    print("=" * 112)
    print("候选集集合优化([MOCK] 召回池 3000 -> 候选集 300):"
          "优化 P(Value|CandidateSet) 而非 P(click|u,i)")
    print("=" * 112)
    print("用户兴趣分布(正文算例):" + "  ".join(f"{c} {w:.0%}" for c, w in INTERESTS.items()))
    print(f"\n{'垂类':>8}{'池内数量':>10}{'粗排分偏置':>12}{'平均粗排分':>12}"
          f"{'平均真值':>11}{'Relevant 数':>13}")
    for c in INTERESTS:
        grp = [it for it in pool if it.cat == c]
        print(f"{c:>8}{len(grp):>10}{SCORE_BIAS[c]:>+12.3f}"
              f"{sum(i.score for i in grp) / len(grp):>12.5f}"
              f"{sum(i.val for i in grp) / len(grp):>11.5f}"
              f"{sum(1 for i in grp if i.relevant):>13}")
    print("各垂类真值同分布,但粗排分被「篮球」系统性高估、「科技」低估;"
          "Relevant 按兴趣分布分配 ——")
    print("因此逐点 TopK(score) 会漏掉冷门垂类里的优质内容,这是集合级约束要解决的问题。")

    strategies = {
        "A 逐点 TopK(score)": lambda: strat_topk(pool, K_SET),
        "B 兴趣配额 Quota": lambda: strat_interest_quota(pool, K_SET),
        f"C Count(Cat)≤{int(K_SET * CAT_CAP_RATIO)}": lambda: strat_category_cap(
            pool, K_SET, CAT_CAP_RATIO),
        f"D Sim(Ci,Cj)<{TAU}": lambda: strat_similarity_cap(pool, K_SET, TAU),
        f"E 集合贪心+{EPSILON:.0%}探索": lambda: strat_set_greedy(pool, K_SET, EPSILON),
    }
    res, sels = {}, {}
    for name, fn in strategies.items():
        sel = fn()
        assert len(sel) == K_SET, (name, len(sel))
        sels[name] = sel
        res[name] = metrics(sel)

    cols = ["Recall@K", "平均分", "Coverage", "有效兴趣数", "最大类目占比",
            "对内相似度", "探索率", "TV距离"]
    print("\n" + "=" * 112)
    print("五种候选集构造策略的集合级指标横评(K = 300)")
    print("=" * 112)
    print(f"{'策略':>22}" + "".join(f"{c:>13}" for c in cols))
    for name, m in res.items():
        print(f"{name:>22}" + "".join(
            f"{m[c]:>13.4f}" if c != "有效兴趣数" else f"{int(m[c]):>13}"
            for c in cols))

    print(f"\n{'策略':>22}" + "".join(f"{c:>10}" for c in INTERESTS)
          + f"{'(各垂类入选数)':>16}")
    for name, sel in sels.items():
        cnt = cat_counts(sel)
        print(f"{name:>22}" + "".join(f"{cnt.get(c, 0):>10}" for c in INTERESTS))

    a = res["A 逐点 TopK(score)"]
    e = res[f"E 集合贪心+{EPSILON:.0%}探索"]
    b = res["B 兴趣配额 Quota"]
    print(f"\n逐点 TopK:最大类目占比 {a['最大类目占比']:.4f}、有效兴趣数 "
          f"{int(a['有效兴趣数'])}/{len(INTERESTS)}、Recall@K={a['Recall@K']:.4f} —— "
          "即正文的 Candidate Concentration;")
    print(f"兴趣配额把 Recall@K 提到 {b['Recall@K']:.4f};"
          f"集合贪心 Recall@K={e['Recall@K']:.4f}、有效兴趣数 "
          f"{int(e['有效兴趣数'])}/{len(INTERESTS)}、"
          f"对内相似度 {a['对内相似度']:.4f} -> {e['对内相似度']:.4f}、"
          f"探索率 {a['探索率']:.4f} -> {e['探索率']:.4f};")
    print(f"代价是集合平均粗排分下降 "
          f"{(a['平均分'] - e['平均分']) / a['平均分']:.2%},"
          f"但集合真值均值 {a['真值均值']:.5f} -> {e['真值均值']:.5f} 反而更高 —— "
          "粗排分高只是偏置带来的幻觉。")

    print("\n" + "=" * 112)
    print("探索比例 ε 扫描(正文 ε = 5% ~ 20%)")
    print("=" * 112)
    print(f"{'ε':>8}{'Recall@K':>12}{'平均分':>11}{'探索率':>10}"
          f"{'Coverage':>11}{'对内相似度':>13}{'TV距离':>10}")
    for eps in (0.0, 0.05, 0.10, 0.15, 0.20):
        m = metrics(strat_set_greedy(pool, K_SET, eps))
        print(f"{eps:>8.2f}{m['Recall@K']:>12.4f}{m['平均分']:>11.5f}"
              f"{m['探索率']:>10.4f}{m['Coverage']:>11.4f}"
              f"{m['对内相似度']:>13.4f}{m['TV距离']:>10.4f}")
    print("ε 增大换来更高探索率(新内容/新作者的曝光机会):探索率 0.2067 -> 0.3633;")
    print("代价是 ε=0.20 时 Recall@K 从 1.0000 掉到 0.9833、类目分布偏离兴趣分布"
          "(TV 0.0067 -> 0.0833),")
    print("这正是正文所说「探索机制本质上是在短期收益与长期收益之间进行平衡」。")

    print("\n" + "=" * 112)
    print("集合价值函数 F(S) 的覆盖权重 λ_cov 敏感性(λ_sim = 0.0015, ε = 10%)")
    print("=" * 112)
    print(f"{'λ_cov':>8}{'Recall@K':>12}{'平均分':>11}{'Coverage':>11}"
          f"{'有效兴趣数':>12}{'最大类目占比':>14}{'TV距离':>10}")
    for lam in (0.0, 1.0, 5.0, 10.0, 30.0):
        m = metrics(strat_set_greedy(pool, K_SET, EPSILON, lam_cov=lam))
        print(f"{lam:>8.1f}{m['Recall@K']:>12.4f}{m['平均分']:>11.5f}"
              f"{m['Coverage']:>11.4f}{int(m['有效兴趣数']):>12}"
              f"{m['最大类目占比']:>14.4f}{m['TV距离']:>10.4f}")
    print("λ_cov 控制 Coverage(CandidateSet) -> max 这一项的话语权。本例中 λ_sim 的"
          "相似度惩罚已足够把 4 个兴趣都覆盖住,")
    print("所以 Coverage 在 λ_cov=0 时就已是 1.0;λ_cov 的作用体现在类目配比上 —— "
          "TV 距离 0.0700 -> 0.0400,")
    print("即候选集的类目分布被进一步拉向用户兴趣分布;λ_cov ≥ 10 后配额项已饱和,结果不再变化。")
