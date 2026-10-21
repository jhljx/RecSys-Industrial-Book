"""
粗精排链路一致性度量与漏斗分析（第 5 章 §5.3 链路一致性模型，模块层）

对应正文概念
------------
正文 §5.3 指出级联架构的核心问题是「链路不一致（Pipeline Inconsistency）」：各阶段
特征体系、样本、目标、模型结构不同，对「优质内容」的定义存在差异，于是一个能获得
精排高分的物品可能在粗排被提前过滤 —— 这就是「漏斗效应（Funnel Effect）」。正文
同时给出三个重要判断：
  (a) 一致性优化手段是「将精排 Top-K 物品构造为正样本，未进入 Top-K 的候选构造为
      负样本」的蒸馏 / 级联学习；
  (b) 一致性是系统内部指标（System Metric），与业务指标无严格因果关系，收益会
      「快速衰减」；
  (c) 若粗排与精排完全一致，链路就退化为「两个功能相同但复杂度不同的排序器串联」，
      即「精排复读机（Ranking Mimicry）」，继续优化一致性失去意义。

本实现做了什么（模块层，不重复模型层）
--------------------------------------
§5.3 的**模型层**实现（RankTower 等一致性模型）在 `ch22 链路一致性建模/code/`，本文件
只做**模块层**的度量与仿真：
1. 三个一致性度量：Spearman 序相关 ρ、Kendall τ、Top-K 命中率 Hit@K。
2. 链路漏斗分析：召回 N -> 粗排截断 K -> 精排 TopM，统计每一级的「漏斗损失」
   （精排最终想要的内容有多少在粗排被丢掉）。
3. 一致性-业务收益曲线：把粗排打分建成「α·精排分 + (1-α)·粗排自有信号」的插值族，
   扫 α ∈ [0,1]，同时打印一致性指标与业务指标（最终曝光内容的真实期望收益），
   数值复现正文 (b)(c) 两个结论 —— 一致性继续上升时业务收益先升后平、边际收益衰减。

关键公式
--------
    ρ = 1 - 6Σd_i² / (n(n²-1))                      （Spearman）
    τ = (C - D) / (n(n-1)/2)                         （Kendall）
    Hit@K = |TopK(prerank) ∩ TopK(rank)| / K
    FunnelLoss@M = 1 - |TopM(rank_on_full) ∩ TopK(prerank)| / M

[MOCK] 组件
-----------
`MockRankModel` 为教学用假精排（高维线性打分器），`MockPreRankSignal` 为粗排自有
信号；均不访问网络、不读数据集。
"""

import math
import random


# [MOCK] 特征体系与两级排序器
# ----------------------------------------------------------------------------
# 正文 §5.3：「各阶段采用的特征体系、训练样本、优化目标以及模型结构并不完全相同」。
# 这里用特征可见性切片来刻画这一点：
#   dims [0, 8)   共享特征（画像、类目等），粗排与精排都能看到
#   dims [8, 32)  精排独有的重特征（序列交叉、长周期统计等）
#   dims [32, 40) 粗排独有的实时特征（召回侧实时行为，精排侧因时延拿不到）
# 用户真实长期价值 true_value 依赖全部 40 维 —— 因此精排分只是业务目标的一个
# 「更好但不完美」的代理，而粗排也握有精排看不到的信息。
SHARED_END = 8
RANK_ONLY_END = 32
DIM = 40


class MockBusinessValue:
    """[MOCK] 用户真实长期价值（线上不可观测，本仿真里充当「业务指标」裁判）。"""

    def __init__(self, rng: random.Random):
        self.w = [rng.gauss(0, 1) for _ in range(DIM)]

    def score(self, x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, x))


class MockRankModel:
    """[MOCK] 精排：拟合 true_value，但只能看到 [0,32) 共 32 维（更丰富的特征体系）。"""

    def __init__(self, value: MockBusinessValue):
        self.w = [w if i < RANK_ONLY_END else 0.0 for i, w in enumerate(value.w)]

    def score(self, x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, x))


class MockPreRankSignal:
    """[MOCK] 粗排自有信号：拟合 true_value，但只能看到 [0,8) ∪ [32,40) 共 16 维。

    与精排共享 8 维 => 两者天然有部分一致性（ρ > 0）；
    各自又有独有特征 => 「不同阶段对优质内容的定义不同」，且粗排并非单纯更弱。
    """

    def __init__(self, value: MockBusinessValue):
        self.w = [
            w if (i < SHARED_END or i >= RANK_ONLY_END) else 0.0
            for i, w in enumerate(value.w)
        ]

    def score(self, x: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, x))


# ----------------------------------------------------------------------------
# 一致性度量
# ----------------------------------------------------------------------------
def _ranks(values: list[float]) -> list[float]:
    """降序名次（1 最好），同分取平均名次。"""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    """ρ = 1 - 6Σd² / (n(n²-1))"""
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def kendall_tau(a: list[float], b: list[float]) -> float:
    """τ = (C - D) / (n(n-1)/2)，O(n²)，仅用于本文件的小规模演示集。"""
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = n * (n - 1) / 2
    return (conc - disc) / total if total else 0.0


def hit_at_k(prerank: list[float], rank: list[float], k: int) -> float:
    """Hit@K：两个排序器各自 TopK 的重叠率。"""
    tp = set(sorted(range(len(prerank)), key=lambda i: prerank[i], reverse=True)[:k])
    tr = set(sorted(range(len(rank)), key=lambda i: rank[i], reverse=True)[:k])
    return len(tp & tr) / k


def ndcg_at_k(prerank: list[float], gain: list[float], k: int) -> float:
    """以精排分为增益的 NDCG@K，衡量粗排排序对下游的可用程度。"""
    order = sorted(range(len(prerank)), key=lambda i: prerank[i], reverse=True)[:k]
    dcg = sum(gain[idx] / math.log2(r + 2) for r, idx in enumerate(order))
    ideal = sorted(gain, reverse=True)[:k]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------------
# 漏斗分析
# ----------------------------------------------------------------------------
def funnel_analysis(prerank: list[float], rank: list[float],
                    k_prerank: int, m_expose: int) -> dict[str, float]:
    """召回 N -> 粗排 TopK -> （在 TopK 内）精排 TopM 的漏斗损失。

    理想链路（oracle）：精排直接在全量 N 上取 TopM。
    真实链路：精排只能在粗排留下的 K 条里取 TopM。
    """
    n = len(prerank)
    idx = list(range(n))
    kept = sorted(idx, key=lambda i: prerank[i], reverse=True)[:k_prerank]
    real_top = sorted(kept, key=lambda i: rank[i], reverse=True)[:m_expose]
    oracle_top = sorted(idx, key=lambda i: rank[i], reverse=True)[:m_expose]
    overlap = len(set(real_top) & set(oracle_top)) / m_expose
    real_gain = sum(rank[i] for i in real_top) / m_expose
    oracle_gain = sum(rank[i] for i in oracle_top) / m_expose
    return {
        "recall_n": n,
        "prerank_k": k_prerank,
        "expose_m": m_expose,
        "top_m_overlap": overlap,
        "funnel_loss": 1.0 - overlap,
        "real_avg_rank_score": real_gain,
        "oracle_avg_rank_score": oracle_gain,
        "gain_retention": real_gain / oracle_gain if oracle_gain else 0.0,
    }


if __name__ == "__main__":
    random.seed(42)

    N = 1000
    rng = random.Random(42)
    feats = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
    biz_value = MockBusinessValue(rng)
    rank_model = MockRankModel(biz_value)
    prerank_model = MockPreRankSignal(biz_value)

    rank_s = [rank_model.score(x) for x in feats]
    prerank_s = [prerank_model.score(x) for x in feats]
    true_value = [biz_value.score(x) for x in feats]

    print("=" * 100)
    print("粗精排一致性度量（§5.3 模块层；模型层实现见 ch22 链路一致性建模/code/）")
    print("=" * 100)
    print(f"候选量 N={N}, 总特征 {DIM} 维：共享 [0,{SHARED_END}), "
          f"精排独有 [{SHARED_END},{RANK_ONLY_END}), 粗排独有 [{RANK_ONLY_END},{DIM})")
    print(f"精排与真实业务价值的 ρ = {spearman(rank_s, true_value):+.4f}（更好但不完美的代理）")
    print(f"粗排与真实业务价值的 ρ = {spearman(prerank_s, true_value):+.4f}")
    sub = list(range(300))  # Kendall τ 用 O(n²)，取 300 条子集
    print(f"\nSpearman ρ      = {spearman(prerank_s, rank_s):+.4f}")
    print(f"Kendall  τ(n=300) = {kendall_tau([prerank_s[i] for i in sub], [rank_s[i] for i in sub]):+.4f}")
    for k in (10, 50, 100, 300):
        print(f"Hit@{k:<4}        = {hit_at_k(prerank_s, rank_s, k):.4f}")

    # ---- 漏斗分析 ----
    print("\n" + "=" * 100)
    print("链路漏斗分析：优质内容在粗排被提前过滤的比例（§5.3 漏斗效应）")
    print("=" * 100)
    print(f"{'粗排K':>7}{'压缩比':>9}{'TopM重叠':>11}{'漏斗损失':>11}"
          f"{'真实均分':>11}{'理想均分':>11}{'收益保留':>10}")
    loss50 = None
    for k in (50, 100, 200, 300, 500, 1000):
        f = funnel_analysis(prerank_s, rank_s, k_prerank=k, m_expose=20)
        if k == 50:
            loss50 = f["funnel_loss"]
        print(f"{k:>7}{k / N:>9.1%}{f['top_m_overlap']:>11.3f}{f['funnel_loss']:>11.3f}"
              f"{f['real_avg_rank_score']:>11.3f}{f['oracle_avg_rank_score']:>11.3f}"
              f"{f['gain_retention']:>10.3f}")
    print(f"\n结论：粗排与精排不一致时，截断越狠漏斗损失越大；K=50 时 {loss50:.0%} 的精排 "
          "Top20 在粗排阶段就被永久丢弃\n（§5.3「一旦优质内容在粗排阶段被过滤……无法再重新召回」）。")

    # ---- 一致性 vs 业务收益：边际收益衰减与「精排复读机」 ----
    print("\n" + "=" * 100)
    print("一致性强度 α 扫描：系统内部指标 vs 业务指标（§5.3 结论 b/c 的数值验证）")
    print("=" * 100)
    print("粗排分 = α·(标准化精排分) + (1-α)·(标准化粗排自有信号)；α=1 即完全复读精排")
    print("每个 α 在 R=200 次独立请求上取平均（每次请求 N=300 候选，粗排截断 K=60，曝光 M=10）")

    def z(v: list[float]) -> list[float]:
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
        return [(x - m) / sd for x in v]

    # 预生成 R 次请求的候选集，所有 α 共用，保证对比公平
    R, N_REQ, K_REQ, M_REQ = 200, 300, 60, 10
    requests = []
    for _ in range(R):
        xs = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N_REQ)]
        requests.append((
            z([rank_model.score(x) for x in xs]),
            z([prerank_model.score(x) for x in xs]),
            [biz_value.score(x) for x in xs],
        ))

    print(f"\n{'α':>6}{'ρ(粗,精)':>11}{'Hit@60':>9}{'NDCG@60':>10}"
          f"{'曝光真实收益':>14}{'Δ业务/Δα':>12}")
    prev_biz = None
    rows = []
    for i in range(11):
        a = i / 10
        rho = hit = ndcg = biz = 0.0
        for zr, zp, tv in requests:
            fused = [a * x + (1 - a) * y for x, y in zip(zr, zp)]
            rho += spearman(fused, zr)
            hit += hit_at_k(fused, zr, K_REQ)
            ndcg += ndcg_at_k(fused, zr, K_REQ)
            kept = sorted(range(N_REQ), key=lambda j: fused[j], reverse=True)[:K_REQ]
            exposed = sorted(kept, key=lambda j: zr[j], reverse=True)[:M_REQ]
            biz += sum(tv[j] for j in exposed) / M_REQ
        rho, hit, ndcg, biz = rho / R, hit / R, ndcg / R, biz / R
        delta = "-" if prev_biz is None else f"{(biz - prev_biz) / 0.1:+.3f}"
        rows.append((a, rho, biz))
        print(f"{a:>6.1f}{rho:>11.4f}{hit:>9.4f}{ndcg:>10.4f}{biz:>14.4f}{delta:>12}")
        prev_biz = biz

    best_a, _, best_biz = max(rows, key=lambda r: r[2])
    last_biz = rows[-1][2]
    print(f"\n结论：业务收益在 α={best_a:.1f} 达到峰值 {best_biz:.4f}；"
          f"α 继续升到 1.0 时一致性指标仍单调变好（ρ→1.0, Hit→1.0），")
    print(f"但业务收益回落到 {last_biz:.4f}（相对峰值 {(last_biz - best_biz) / best_biz:+.2%}）。")
    print("即「一致性属于系统内部指标，与业务指标无严格因果关系」且收益快速衰减（结论 b）；")
    print("α=1 时粗排完全复读精排，链路退化为同质排序器串联即「精排复读机」（结论 c）。")
