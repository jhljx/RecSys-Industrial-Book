"""
基于值的融合:线性融合与乘法融合(第 5 章 §5.4.1 基于值的融合)

对应正文概念
------------
正文 §5.4.1 把「基于值(Value-Based)的融合」分为两类:

线性融合(Linear Fusion):
    s_agg = w1*p_ctr + w2*p_cvr + ... + wn*p_cascade
    优点是「简单、稳定且易于解释」,工程师直接调权重就能做流量调控;
    缺点是(a)只能表达加性关系 f(a,b)=a+b,无法刻画 p_ctr×p_ltr 这类协同关系,
    于是工业界把交叉因子作为新项加进公式 -- 这个过程叫「加队列(Add Queue)」;
    (b)「对于 pxtr 的绝对数值高度敏感」,p_evtr >> p_cmtr 或两个模型校准程度
    不同都会让原有权重失效。

乘法融合(Multiplicative Fusion):
    s_agg = ∏_{i=1..n} (β_i + α_i * p_i)^{γ_i}
    「p_i -> 0 会导致整体得分显著下降」,天然具有短板效应(Bottleneck Effect)。

本实现做了什么
--------------
1. `LinearFusion`:线性融合 + 显式的「加队列」接口(把 p_ctr*p_ltr 等交叉项
   当成新的一路因子追加进公式),并打印加队列前后的 Top-10 变动。
2. `MultiplicativeFusion`:严格按正文通用形式 ∏(β_i + α_i p_i)^{γ_i} 实现。
3. 校准漂移实验:对 p_cvr 做 ×3 的乘性漂移(模型升级后分布变化,序不变),
   对比线性融合与乘法融合的排序稳定性 -- 验证正文「原有权重失效」的论断。
4. 同一候选集下给出四套公式的排序差异对比表(Spearman ρ / Top-50 Jaccard /
   头部构成),与 `rank_based_fusion.py`、`hybrid_value_rank_fusion.py` 呼应。

关键公式
--------
    线性:   s = Σ_i w_i p_i                (梯度 ∂s/∂p_i = w_i,恒定)
    乘法:   s = ∏_i (β_i + α_i p_i)^{γ_i}  (∂s/∂p_1 ∝ (β_2+α_2 p_2),互相耦合)
    对数形式: log s = Σ_i γ_i log(β_i + α_i p_i)

参考:根目录 `code/粗排通用乘法公式.py`、`code/粗排加法与乘法公式.py`。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

OBJECTIVES = ("pctr", "pcvr", "pltr", "pevtr")


# ----------------------------------------------------------------------------
# [MOCK] 候选物品与粗排多目标打分
# ----------------------------------------------------------------------------
@dataclass
class Item:
    """[MOCK] 候选物品。pxtr 由确定性随机过程生成,不来自任何真实模型。"""

    iid: int
    kind: str                       # balanced / ctr_spike / cvr_spike / mediocre
    pxtr: dict[str, float] = field(default_factory=dict)


def mock_candidates(n: int, rng: random.Random) -> list[Item]:
    """[MOCK] 生成 n 个候选,刻意混入正文讨论的三类内容。

    - ctr_spike / cvr_spike:单目标极高、其余很低的「偏科生」;
    - balanced:多目标同时中上的「六边形战士」;
    - mediocre:普通内容。
    各 pxtr 量纲刻意不同(正文:p_evtr >> p_cmtr),用于暴露线性融合的尺度敏感性。
    """
    items: list[Item] = []
    for i in range(n):
        r = rng.random()
        if r < 0.12:
            kind = "ctr_spike"
            base = {"pctr": 0.30 + 0.12 * rng.random(), "pcvr": 0.002 + 0.004 * rng.random(),
                    "pltr": 0.02 + 0.04 * rng.random(), "pevtr": 0.45 + 0.15 * rng.random()}
        elif r < 0.24:
            kind = "cvr_spike"
            base = {"pctr": 0.02 + 0.03 * rng.random(), "pcvr": 0.060 + 0.040 * rng.random(),
                    "pltr": 0.03 + 0.05 * rng.random(), "pevtr": 0.30 + 0.15 * rng.random()}
        elif r < 0.44:
            kind = "balanced"
            u = 0.55 + 0.35 * rng.random()
            base = {"pctr": 0.10 + 0.14 * u, "pcvr": 0.018 + 0.026 * u,
                    "pltr": 0.10 + 0.22 * u, "pevtr": 0.55 + 0.30 * u}
        else:
            kind = "mediocre"
            u = rng.random()
            base = {"pctr": 0.03 + 0.10 * u, "pcvr": 0.004 + 0.016 * u,
                    "pltr": 0.02 + 0.12 * u, "pevtr": 0.30 + 0.35 * u}
        items.append(Item(iid=i, kind=kind, pxtr={k: round(v, 6) for k, v in base.items()}))
    return items


# ----------------------------------------------------------------------------
# 线性融合(含「加队列」)
# ----------------------------------------------------------------------------
class LinearFusion:
    """s_agg = Σ_i w_i p_i, 可追加交叉因子(加队列 Add Queue)。"""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = dict(weights)
        self.cross_terms: list[tuple[tuple[str, ...], float]] = []

    def add_queue(self, factors: tuple[str, ...], weight: float) -> "LinearFusion":
        """加队列:把 p_ctr*p_ltr 这类高阶交叉因子作为新的一路加入融合公式。

        正文 §5.4.1:「工业界经常引入高阶交叉因子 p_ctr p_ltr、p_ctr p_cvr、
        p_ltr p_evtr 并将这些交叉项加入融合公式中......其本质是在现有排序体系中
        引入新的信息来源,从而寻找增量收益。」
        """
        self.cross_terms.append((factors, weight))
        return self

    def score(self, it: Item) -> float:
        s = sum(w * it.pxtr[k] for k, w in self.weights.items())
        for factors, w in self.cross_terms:
            prod = 1.0
            for k in factors:
                prod *= it.pxtr[k]
            s += w * prod
        return s

    def name(self) -> str:
        tail = "+加队列" if self.cross_terms else ""
        return f"线性融合{tail}"


# ----------------------------------------------------------------------------
# 乘法融合(通用形式)
# ----------------------------------------------------------------------------
class MultiplicativeFusion:
    """s_agg = ∏_i (β_i + α_i p_i)^{γ_i}  (正文 §5.4.1 乘法融合的典型形式)。"""

    def __init__(self, params: dict[str, tuple[float, float, float]]) -> None:
        # params[key] = (alpha, beta, gamma)
        self.params = dict(params)

    def score(self, it: Item) -> float:
        # 在对数空间累加,避免多因子连乘下溢;正文:「乘法公式可以看做是对数空间的加法公式」
        log_s = 0.0
        for k, (alpha, beta, gamma) in self.params.items():
            base = beta + alpha * it.pxtr[k]
            if base <= 0.0:
                return 0.0
            log_s += gamma * math.log(base)
        return math.exp(log_s)

    def name(self) -> str:
        betas = {k: p[1] for k, p in self.params.items()}
        return f"乘法融合(β={list(betas.values())[0]:g}..)"


# ----------------------------------------------------------------------------
# 排序对比工具
# ----------------------------------------------------------------------------
def rank_of(scores: list[float]) -> list[int]:
    """返回每个下标的排名(1 = 最高分)。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    r = [0] * len(scores)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank_of(a), rank_of(b)
    n = len(a)
    ma, mb = (n + 1) / 2, (n + 1) / 2
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def topk_jaccard(a: list[float], b: list[float], k: int) -> float:
    sa = set(sorted(range(len(a)), key=lambda i: -a[i])[:k])
    sb = set(sorted(range(len(b)), key=lambda i: -b[i])[:k])
    return len(sa & sb) / len(sa | sb)


def kind_mix(items: list[Item], scores: list[float], k: int) -> dict[str, int]:
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    mix: dict[str, int] = {}
    for i in top:
        mix[items[i].kind] = mix.get(items[i].kind, 0) + 1
    return mix


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    N, TOPK = 500, 50
    items = mock_candidates(N, rng)

    # ---- 0. 正文表 5.2 的三点示例:A(0.9,0.1) B(0.5,0.5) C(0.1,0.9) ----
    print("=" * 100)
    print("§5.5 表 5.2 三点示例:加法 w1=w2=1 vs 最简乘法 p_ctr*p_cvr")
    print("=" * 100)
    print(f"{'Item':>6}{'p_ctr':>9}{'p_cvr':>9}{'加法分':>10}{'乘法分':>10}")
    for tag, c, v in (("A", 0.9, 0.1), ("B", 0.5, 0.5), ("C", 0.1, 0.9)):
        print(f"{tag:>6}{c:>9.2f}{v:>9.2f}{c + v:>10.2f}{c * v:>10.4f}")
    print("加法下 A=B=C=1.00(完全同分);乘法下 B=0.2500 最大,A=C=0.0900。")
    print("-> 加法偏好「偏科生」,乘法偏好「六边形战士」(§5.5)。")

    # ---- 1. pxtr 量纲差异 ----
    print("\n" + "=" * 100)
    print(f"候选集 N={N}:各目标 pxtr 的量纲差异(正文:线性融合「对 pxtr 的绝对数值高度敏感」)")
    print("=" * 100)
    print(f"{'目标':>8}{'均值':>12}{'最小':>12}{'最大':>12}{'量纲倍数(相对pcvr)':>22}")
    means = {k: sum(it.pxtr[k] for it in items) / N for k in OBJECTIVES}
    for k in OBJECTIVES:
        vs = [it.pxtr[k] for it in items]
        print(f"{k:>8}{means[k]:>12.4f}{min(vs):>12.4f}{max(vs):>12.4f}"
              f"{means[k] / means['pcvr']:>22.1f}x")

    # ---- 2. 四套公式 ----
    lin = LinearFusion({"pctr": 0.50, "pcvr": 0.30, "pltr": 0.15, "pevtr": 0.05})
    lin_q = LinearFusion({"pctr": 0.50, "pcvr": 0.30, "pltr": 0.15, "pevtr": 0.05})
    lin_q.add_queue(("pctr", "pcvr"), 3.0).add_queue(("pctr", "pltr"), 1.5)
    mul0 = MultiplicativeFusion({k: (1.0, 0.0, g) for k, g in
                                 (("pctr", 1.0), ("pcvr", 1.0), ("pltr", 0.5), ("pevtr", 0.5))})
    mul1 = MultiplicativeFusion({k: (1.0, 1.0, g) for k, g in
                                 (("pctr", 1.0), ("pcvr", 1.0), ("pltr", 0.5), ("pevtr", 0.5))})

    formulas = [("线性融合", lin), ("线性+加队列", lin_q),
                ("乘法 β=0", mul0), ("乘法 β=1", mul1)]
    scores = {name: [f.score(it) for it in items] for name, f in formulas}

    print("\n" + "=" * 100)
    print(f"同一候选集下四套公式的 Top-{TOPK} 头部构成(单位:条)")
    print("=" * 100)
    kinds = ("balanced", "ctr_spike", "cvr_spike", "mediocre")
    print(f"{'公式':>14}" + "".join(f"{k:>12}" for k in kinds))
    for name, _ in formulas:
        mix = kind_mix(items, scores[name], TOPK)
        print(f"{name:>14}" + "".join(f"{mix.get(k, 0):>12}" for k in kinds))
    print("balanced = 多目标均衡;ctr_spike/cvr_spike = 单目标极高的偏科生。")

    # ---- 3. 两两排序差异 ----
    print("\n" + "=" * 100)
    print(f"两两排序差异:Spearman ρ / Top-{TOPK} Jaccard")
    print("=" * 100)
    names = [n for n, _ in formulas]
    print(f"{'':>14}" + "".join(f"{n:>16}" for n in names))
    for a in names:
        row = f"{a:>14}"
        for b in names:
            if a == b:
                row += f"{'-':>16}"
            else:
                row += (f"{spearman(scores[a], scores[b]):>+7.3f}/"
                        f"{topk_jaccard(scores[a], scores[b], TOPK):>5.2f}  ")
        print(row)

    # ---- 4. 校准漂移:模型升级后分布变化,序不变 ----
    print("\n" + "=" * 100)
    print("校准漂移实验:p_cvr 整体 ×3(序完全不变,仅分布变化)后的排序稳定性")
    print("=" * 100)
    drifted = [Item(it.iid, it.kind, dict(it.pxtr)) for it in items]
    for it in drifted:
        it.pxtr["pcvr"] = min(0.999, it.pxtr["pcvr"] * 3.0)
    # 验证漂移前后 p_cvr 自身的序不变
    rho_pcvr = spearman([it.pxtr["pcvr"] for it in items],
                        [it.pxtr["pcvr"] for it in drifted])
    print(f"漂移前后 p_cvr 自身 Spearman ρ = {rho_pcvr:+.4f}(序未变,只是数值放大了 3 倍)")
    print(f"\n{'公式':>14}{'ρ(漂移前,漂移后)':>20}{'Top50 Jaccard':>16}{'头部balanced变化':>20}")
    for name, f in formulas:
        new = [f.score(it) for it in drifted]
        old = scores[name]
        b_old = kind_mix(items, old, TOPK).get("balanced", 0)
        b_new = kind_mix(drifted, new, TOPK).get("balanced", 0)
        print(f"{name:>14}{spearman(old, new):>+20.4f}"
              f"{topk_jaccard(old, new, TOPK):>16.3f}{f'{b_old} -> {b_new}':>20}")
    print("\n结论:乘法融合中 β=0 时 p_cvr 的乘性漂移只给所有物品乘同一常数,排序完全不变(ρ=1);")
    print("线性融合的权重 w_cvr 实际变成了 3*w_cvr,排序显著改变 -- 正文「原有权重失效,")
    print("从而增加调参成本」。β=1 的乘法融合处于两者之间。")

    # ---- 5. 短板效应的定量刻画 ----
    print("\n" + "=" * 100)
    print("短板效应(Bottleneck Effect)定量刻画:等和点 (0.9,0.1) vs (0.99,0.01)")
    print("=" * 100)
    print(f"{'(p1,p2)':>14}{'加法 p1+p2':>13}{'乘法 p1*p2':>13}{'乘法(β=1)':>14}")
    for p1, p2 in ((0.9, 0.1), (0.99, 0.01), (0.5, 0.5), (0.7, 0.3)):
        print(f"{f'({p1},{p2})':>14}{p1 + p2:>13.2f}{p1 * p2:>13.4f}"
              f"{(1 + p1) * (1 + p2):>14.4f}")
    print("加法下 (0.9,0.1) 与 (0.99,0.01) 同在 c=1.0 等值线上;")
    print("乘法下前者 0.0900、后者 0.0099,相差 9.1 倍 -- 单因子极端取值被强惩罚(§5.5)。")
