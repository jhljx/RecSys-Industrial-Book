"""
粗排模块完整链路骨架（第 5 章 §5.1 为什么需要粗排 / 图 5.1 粗排模块内部处理流程）

对应正文概念
------------
正文 §5.1 指出粗排位于召回与精排之间，职责是「在保证较高召回率的前提下，利用轻量级
模型对候选集进行进一步筛选，将候选规模从成千上万压缩到数百的量级」，同时还要承担
多目标融合、流量调控（Quota / 保量）、候选集优化等任务。图 5.1「粗排模块内部处理
流程」给出的五段式链路是本文件的实现对象：

    多路召回候选集  ->  粗排模型打分(多 pxtr)  ->  多目标融合  ->
    Quota 调控 + 保量  ->  截断 TopK  ->  输出给精排

本实现做了什么
--------------
1. 用一个 `PreRankPipeline` 把五个阶段串成可运行的 Python 链路，每一阶段都记录
   「进入候选量级 / 退出候选量级 / 耗时(ms) / 占用预算比例」，最后打印链路日志表，
   复现正文「粗排阶段能够获得的推理预算通常最多只有 50 毫秒」（§5.2）的预算视角。
2. 打分、融合、Quota、保量、截断全部按正文公式实现：
   - 融合（§5.4.1 线性融合）：s_agg = w1*p_ctr + w2*p_cvr + w3*p_ltr
   - 保量（§5.6.2）：Exposure_i < Target_i  =>  Score_i = Score_i + Δ
   - Quota（§5.6.1）：CandidateSet = ∪_i TopK_i，配额按正文表 5.3 的 40/30/20/10
   - 截断（§5.6.3）：TopK(CandidateSet)，K=300
3. 统计正文 §5.7 关心的候选集质量指标：类目覆盖数、最大类目占比（Candidate
   Concentration）、新内容（探索）占比。

[MOCK] 组件
-----------
`MockRecallService`（多路召回）、`MockPreRankModel`（三塔 pxtr 打分）、
`MockRankService`（精排）、`MockExposureStore`（保量用的曝光计数存储）均为教学用假
实现，不访问网络、不读数据集，用确定性伪随机数生成分数。

关键公式
--------
    s_agg = Σ_i w_i p_i                     （线性融合，§5.4.1）
    Score_i <- Score_i + Δ  if Exposure_i < Target_i   （保量，§5.6.2）
    CandidateSet = ∪_i TopK_i               （Quota，§5.6.1）
    FinalSet = TopK(CandidateSet), K = 300  （截断，§5.6.3）
"""

import random
import time
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------
@dataclass
class Item:
    """候选物品。pxtr 由粗排模型填充，score 由多目标融合填充。"""

    item_id: int
    category: str                 # game / entertainment / ecommerce
    is_new_anchor: bool           # 是否新主播（新内容），用于新主播池与保量
    recall_source: str            # 来自哪一路召回
    pctr: float = 0.0
    pcvr: float = 0.0
    pltr: float = 0.0
    score: float = 0.0
    guarantee_bonus: float = 0.0  # 保量补偿项 Δ
    quota_pool: str = ""          # 最终被哪个 Quota 池选中


@dataclass
class StageLog:
    """链路单阶段日志：候选量级 + 耗时。"""

    name: str
    n_in: int
    n_out: int
    cost_ms: float
    note: str = ""


# ----------------------------------------------------------------------------
# [MOCK] 外部依赖
# ----------------------------------------------------------------------------
class MockRecallService:
    """[MOCK] 多路召回服务。按正文 §5.1「多路召回策略」造 5 路候选并做去重融合。"""

    SOURCES = {
        "u2i_cf": 1600,
        "i2i_swing": 1400,
        "vector_ann": 1200,
        "hot_pool": 500,
        "cold_start": 400,
    }
    CATEGORIES = ["game", "entertainment", "ecommerce"]

    def __init__(self, rng: random.Random):
        self.rng = rng

    def retrieve(self) -> tuple[list[Item], int]:
        """返回 (去重后的候选集, 去重前的召回总条数)。"""
        raw_total = 0
        pool: dict[int, Item] = {}
        for source, n in self.SOURCES.items():
            raw_total += n
            for _ in range(n):
                # 物品 id 空间故意小于召回总量，从而产生多路重叠
                iid = self.rng.randrange(0, 6000)
                if iid in pool:
                    continue
                cat = self.CATEGORIES[iid % 3]
                is_new = (source == "cold_start") or (iid % 17 == 0)
                pool[iid] = Item(iid, cat, is_new, source)
        return list(pool.values()), raw_total


class MockPreRankModel:
    """[MOCK] 粗排多任务模型（§5.2）：一次前向同时输出 pctr / pcvr / pltr 三个 pxtr。

    真实实现是双塔 + 多任务塔（代码见 `ch11 粗排模型/code/`），这里用物品 id 的
    确定性哈希生成分数，使同一 seed 下结果可复现；并刻意做两件事：
    (a) 三个 pxtr 量纲不同（pctr ~ 0.07、pcvr ~ 0.017、pltr ~ 0.25 量级），以呈现
        正文 §5.4.1「线性融合对于 pxtr 的绝对数值高度敏感」；
    (b) 给 game 类目一个正向偏置（模拟正文 §5.6.1 的重度游戏用户 P(Game|User)=0.9），
        使纯 TopK(score) 的候选集高度集中于游戏类，复现正文表 5.2「无 Quota 约束下
        的候选集示例」中 Top5 全是游戏的现象。
    """

    # 单条候选的打分开销（微秒）。双塔只需一次低维点积，故极小。
    US_PER_ITEM = 4.0
    # 用户的类目兴趣先验，直接作用在 pctr 上（倍数）
    CATEGORY_PRIOR = {"game": 1.8, "entertainment": 1.0, "ecommerce": 0.7}

    def predict_batch(self, items: list[Item]) -> None:
        for it in items:
            h = (it.item_id * 2654435761) % 1000003
            u = ((h % 997) + 0.5) / 997.0
            v = ((h // 997 % 991) + 0.5) / 991.0
            w = ((h // 7919 % 983) + 0.5) / 983.0
            prior = self.CATEGORY_PRIOR[it.category]
            it.pctr = round(min(0.99, (0.02 + 0.10 * u) * prior), 6)  # 点击率
            it.pcvr = round(0.002 + 0.03 * v, 6)                      # 转化率
            it.pltr = round(0.05 + 0.40 * w, 6)                       # 点赞率


class MockExposureStore:
    """[MOCK] 保量用的曝光计数存储（线上通常是 Redis / 实时指标服务）。"""

    def __init__(self, rng: random.Random, target: int = 500):
        self.rng = rng
        self.target = target
        self._exposure: dict[int, int] = {}

    def exposure_of(self, item_id: int) -> int:
        if item_id not in self._exposure:
            self._exposure[item_id] = self.rng.randrange(0, 900)
        return self._exposure[item_id]

    def under_delivered(self, item_id: int) -> bool:
        """Exposure_i < Target_i 即判定为「欠投」，需要保量补偿。"""
        return self.exposure_of(item_id) < self.target


class MockRankService:
    """[MOCK] 下游精排服务。只用来呈现「精排单条开销远高于粗排」的预算差异。"""

    US_PER_ITEM = 180.0  # 精排单条开销 ≈ 粗排的 45 倍

    def estimated_cost_ms(self, n: int) -> float:
        return n * self.US_PER_ITEM / 1000.0


# ----------------------------------------------------------------------------
# 粗排链路
# ----------------------------------------------------------------------------
class PreRankPipeline:
    """图 5.1 的五段式粗排链路。"""

    # §5.4.1 线性融合权重（与正文代码清单 5.1 的伪代码一致）
    FUSION_WEIGHTS = {"pctr": 0.5, "pcvr": 0.3, "pltr": 0.2}
    # §5.6.1 Quota 配置：正文假设粗排最终输出 100 个候选时为 40/30/20/10
    QUOTA_RATIO = {
        "game": 0.40,
        "entertainment": 0.30,
        "ecommerce": 0.20,
        "new_anchor": 0.10,
    }
    GUARANTEE_DELTA = 0.010   # §5.6.2 保量补偿项 Δ
    BUDGET_MS = 50.0          # §5.2「粗排阶段能够获得的推理预算通常最多只有 50 毫秒」

    def __init__(self, seed: int = 42, truncate_k: int = 300):
        rng = random.Random(seed)
        self.recall = MockRecallService(rng)
        self.model = MockPreRankModel()
        self.exposure = MockExposureStore(rng)
        self.rank = MockRankService()
        self.truncate_k = truncate_k
        self.logs: list[StageLog] = []

    # -- 阶段 1：候选集进入 ---------------------------------------------------
    def stage_recall(self) -> list[Item]:
        t0 = time.perf_counter()
        items, raw_total = self.recall.retrieve()
        cost = (time.perf_counter() - t0) * 1000.0
        self.logs.append(
            StageLog("1 多路召回融合", raw_total, len(items), cost,
                     f"5 路召回, 去重率 {1 - len(items) / raw_total:.1%}")
        )
        return items

    # -- 阶段 2：粗排模型打分 ------------------------------------------------
    def stage_score(self, items: list[Item]) -> list[Item]:
        t0 = time.perf_counter()
        self.model.predict_batch(items)
        cost = (time.perf_counter() - t0) * 1000.0
        model_cost = len(items) * self.model.US_PER_ITEM / 1000.0
        self.logs.append(
            StageLog("2 粗排模型打分", len(items), len(items), cost,
                     f"3 个 pxtr, 等效线上开销 {model_cost:.1f}ms")
        )
        return items

    # -- 阶段 3：多目标融合 --------------------------------------------------
    def stage_fuse(self, items: list[Item]) -> list[Item]:
        t0 = time.perf_counter()
        w = self.FUSION_WEIGHTS
        for it in items:
            it.score = w["pctr"] * it.pctr + w["pcvr"] * it.pcvr + w["pltr"] * it.pltr
        cost = (time.perf_counter() - t0) * 1000.0
        self.logs.append(
            StageLog("3 多目标融合(线性)", len(items), len(items), cost,
                     "s=0.5pctr+0.3pcvr+0.2pltr")
        )
        return items

    # -- 阶段 4：保量 + Quota ------------------------------------------------
    def stage_guarantee(self, items: list[Item]) -> list[Item]:
        t0 = time.perf_counter()
        boosted = 0
        for it in items:
            if it.is_new_anchor and self.exposure.under_delivered(it.item_id):
                it.guarantee_bonus = self.GUARANTEE_DELTA
                it.score += it.guarantee_bonus
                boosted += 1
        cost = (time.perf_counter() - t0) * 1000.0
        self.logs.append(
            StageLog("4 保量补偿", len(items), len(items), cost,
                     f"{boosted} 条欠投新内容 Score+Δ({self.GUARANTEE_DELTA})")
        )
        return items

    def stage_quota(self, items: list[Item], out_size: int) -> list[Item]:
        t0 = time.perf_counter()
        pools: dict[str, list[Item]] = {k: [] for k in self.QUOTA_RATIO}
        for it in items:
            if it.is_new_anchor:
                pools["new_anchor"].append(it)
            else:
                pools[it.category].append(it)
        selected: dict[int, Item] = {}
        detail = []
        for pool_name, ratio in self.QUOTA_RATIO.items():
            quota = int(round(out_size * ratio))
            bucket = sorted(pools[pool_name], key=lambda x: x.score, reverse=True)
            taken = bucket[:quota]
            for it in taken:
                it.quota_pool = pool_name
                selected[it.item_id] = it
            detail.append(f"{pool_name}:{len(taken)}/{quota}(池内{len(bucket)})")
        out = list(selected.values())
        cost = (time.perf_counter() - t0) * 1000.0
        self.logs.append(
            StageLog("5 Quota 调控", len(items), len(out), cost, " ".join(detail))
        )
        return out

    # -- 阶段 5：截断 + 输出 -------------------------------------------------
    def stage_truncate(self, items: list[Item]) -> list[Item]:
        t0 = time.perf_counter()
        out = sorted(items, key=lambda x: x.score, reverse=True)[: self.truncate_k]
        cost = (time.perf_counter() - t0) * 1000.0
        self.logs.append(
            StageLog("6 截断 TopK", len(items), len(out), cost,
                     f"K={self.truncate_k}, 精排预估开销 "
                     f"{self.rank.estimated_cost_ms(len(out)):.0f}ms")
        )
        return out

    def run(self, out_size: int = 400) -> list[Item]:
        items = self.stage_recall()
        items = self.stage_score(items)
        items = self.stage_fuse(items)
        items = self.stage_guarantee(items)
        items = self.stage_quota(items, out_size=out_size)
        return self.stage_truncate(items)


# ----------------------------------------------------------------------------
# 候选集质量指标（§5.7）
# ----------------------------------------------------------------------------
def candidate_set_report(items: list[Item]) -> dict[str, float]:
    n = max(1, len(items))
    cat_cnt: dict[str, int] = {}
    for it in items:
        cat_cnt[it.category] = cat_cnt.get(it.category, 0) + 1
    max_share = max(cat_cnt.values()) / n
    explore_rate = sum(1 for it in items if it.is_new_anchor) / n
    src_cnt = len({it.recall_source for it in items})
    return {
        "size": len(items),
        "category_coverage": len(cat_cnt),
        "max_category_share": max_share,
        "explore_rate": explore_rate,
        "recall_source_coverage": src_cnt,
        "avg_score": sum(it.score for it in items) / n,
    }


def print_pipeline_logs(logs: list[StageLog], budget_ms: float) -> None:
    total = sum(g.cost_ms for g in logs)
    print(f"{'阶段':<18}{'入':>7}{'出':>7}{'耗时ms':>9}{'占预算':>8}  说明")
    print("-" * 110)
    for g in logs:
        print(f"{g.name:<18}{g.n_in:>7}{g.n_out:>7}{g.cost_ms:>9.3f}"
              f"{g.cost_ms / budget_ms:>7.1%}  {g.note}")
    print("-" * 110)
    print(f"{'合计':<18}{'':>7}{'':>7}{total:>9.3f}{total / budget_ms:>7.1%}"
          f"  预算 {budget_ms:.0f}ms -> {'通过' if total <= budget_ms else '超预算'}")


if __name__ == "__main__":
    random.seed(42)

    print("=" * 110)
    print("粗排完整链路：召回 -> 打分 -> 融合 -> 保量 -> Quota -> 截断 -> 精排（§5.1 图 5.1）")
    print("=" * 110)

    pipe = PreRankPipeline(seed=42, truncate_k=300)
    final = pipe.run(out_size=300)
    print_pipeline_logs(pipe.logs, PreRankPipeline.BUDGET_MS)

    print("\n[候选集质量指标 §5.7]")
    rep = candidate_set_report(final)
    for k, v in rep.items():
        print(f"  {k:<24}{v:.4f}" if isinstance(v, float) else f"  {k:<24}{v}")

    print("\n[输出给精排的 Top-10]")
    print(f"{'rank':>5}{'item':>7}{'category':>15}{'pool':>14}"
          f"{'pctr':>9}{'pcvr':>9}{'pltr':>9}{'score':>10}{'Δ':>8}")
    for i, it in enumerate(final[:10], 1):
        print(f"{i:>5}{it.item_id:>7}{it.category:>15}{it.quota_pool:>14}"
              f"{it.pctr:>9.4f}{it.pcvr:>9.4f}{it.pltr:>9.4f}{it.score:>10.5f}"
              f"{it.guarantee_bonus:>8.3f}")

    # ---- 对照：不做 Quota / 保量，纯 TopK(score) ----
    print("\n" + "=" * 110)
    print("对照实验：纯 TopK(score) vs 带 Quota+保量 的候选集结构（正文表 5.2 的问题）")
    print("=" * 110)
    plain = PreRankPipeline(seed=42, truncate_k=300)
    raw = plain.stage_recall()
    raw = plain.stage_score(raw)
    raw = plain.stage_fuse(raw)
    plain_top = sorted(raw, key=lambda x: x.score, reverse=True)[:300]
    rep_plain = candidate_set_report(plain_top)
    print(f"{'指标':<26}{'纯TopK':>12}{'Quota+保量':>14}")
    for k in ("category_coverage", "max_category_share", "explore_rate", "avg_score"):
        a, b = rep_plain[k], rep[k]
        print(f"{k:<26}{a:>12.4f}{b:>14.4f}")
    print("\n结论：纯 TopK 下候选集平均分更高但新内容占比与类目结构失衡；"
          "Quota+保量以少量平均分为代价换回生态结构（§5.6）。")

    # ---- 截断 K 对精排算力的影响 ----
    print("\n" + "=" * 110)
    print("截断条数 K 与精排算力预算（§5.6.3）")
    print("=" * 110)
    rank_svc = MockRankService()
    print(f"{'K':>6}{'精排开销ms':>14}{'相对K=300':>12}")
    for k in (100, 300, 500, 1000, 5000):
        c = rank_svc.estimated_cost_ms(k)
        print(f"{k:>6}{c:>14.1f}{c / rank_svc.estimated_cost_ms(300):>12.2f}x")
    print("\n结论：若不截断直接把 5000 条召回送精排，精排开销 900ms，"
          "远超单次请求数百毫秒的总预算 —— 这正是粗排存在的根本原因（§5.1）。")
