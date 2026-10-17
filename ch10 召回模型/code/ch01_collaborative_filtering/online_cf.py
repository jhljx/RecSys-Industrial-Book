"""
Online CF 召回 —— 实时协同过滤（第10章 §10.1.3）

核心思想：
    传统 ItemCF / UserCF 依赖离线全量统计，相似度矩阵天级更新，无法响应热点。
    Online CF 直接消费用户行为流（Kafka -> Flink -> Redis），行为到达后只更新
    「受影响的 Item Pair」，并对历史共现分数做时间衰减，从而分钟级甚至秒级地
    刷新 Item 之间的共现关系。

关键公式（与正文一致）：
    1) 共现分数增量更新：Score(i, j) <- Decay(Score(i, j)) + w(j)
    2) 行为权重：        w(j) = Σ_k α_k · x_k(j)      （点击1/长观看3/点赞5/关注8/分享10）
    3) 时间衰减：        Decay(x) = x · exp(-λ·Δt)
    4) 在线召回：        Retrieve(B_u) = merge( TopK(b) for b in B_u )

工程对应关系：
    pair_score  <->  Redis Key: item_A_item_B -> (score, timestamp)   长过期时间
    topk_cache  <->  Redis Key: item_A        -> [(item_1,s_1), ...]  短过期时间

本文件为纯 Python 标准库实现，不依赖任何第三方库。
"""

import math
import random
from collections import defaultdict, deque

# 行为权重表（对应正文表「Online CF召回行为权重设置示例」）
BEHAVIOR_WEIGHTS: dict[str, float] = {
    "click": 1.0,      # 点击
    "long_view": 3.0,  # 长观看
    "like": 5.0,       # 点赞
    "follow": 8.0,     # 关注
    "share": 10.0,     # 分享
}


class OnlineCF:
    """
    流式 Online ItemCF（Online I2I）。

    三个核心机制：
      1. 增量更新：一条行为到达时，只更新 "历史序列 H_u × 当前物品 j" 这 |H_u| 对共现，
                   而不是重算整张共现矩阵；
      2. 惰性重算：Top-K 相似索引只在该物品被标记为 dirty（或缓存过期）时才重新排序；
      3. 时间衰减：共现分数按 exp(-λΔt) 衰减，并配合滑动时间窗口淘汰过老的行为。
    """

    def __init__(
        self,
        behavior_weights: dict[str, float] | None = None,
        history_len: int = 20,
        window_sec: float = 600.0,
        decay_lambda: float = 0.004,
        min_weight: float = 1.0,
        top_k: int = 10,
        cache_ttl: float = 120.0,
    ):
        self.behavior_weights = behavior_weights or dict(BEHAVIOR_WEIGHTS)
        self.history_len = history_len        # m：用户历史行为序列截断长度（正文取 20 或 30）
        self.window_sec = window_sec          # 滑动时间窗口大小
        self.decay_lambda = decay_lambda      # 时间衰减系数 λ
        self.min_weight = min_weight          # 行为权重阈值，低于该值不参与共现统计
        self.top_k = top_k
        self.cache_ttl = cache_ttl            # Top-K 索引缓存过期时间（短过期）

        self.pair_score: dict[str, dict[str, float]] = defaultdict(dict)  # Score(i, j)
        self.pair_ts: dict[str, dict[str, float]] = defaultdict(dict)     # 该 pair 上次更新时间
        self.user_hist: dict[str, deque] = defaultdict(deque)             # H_u: [(item, ts), ...]

        self.dirty: set[str] = set()                                      # Top-K 索引失效的物品
        self.topk_cache: dict[str, tuple[float, list[tuple[str, float]]]] = {}

        # 统计量：用于演示增量更新 / 惰性重算带来的计算节省
        self.n_pair_update = 0
        self.n_recompute = 0
        self.n_cache_hit = 0
        self.n_filtered = 0

    # ── 共现分数读写 ─────────────────────────────────────────────────────────
    def _decayed_score(self, i: str, j: str, now: float) -> float:
        """读取共现分数并补上时间衰减：Decay(x) = x · exp(-λΔt)。"""
        score = self.pair_score[i].get(j, 0.0)
        if score == 0.0:
            return 0.0
        dt = max(0.0, now - self.pair_ts[i].get(j, now))
        return score * math.exp(-self.decay_lambda * dt)

    def _bump_pair(self, i: str, j: str, w: float, now: float) -> None:
        """Score(i, j) <- Decay(Score(i, j)) + w(j)，并标记 i 的 Top-K 索引失效。"""
        self.pair_score[i][j] = self._decayed_score(i, j, now) + w
        self.pair_ts[i][j] = now
        self.dirty.add(i)
        self.n_pair_update += 1

    # ── 实时更新流程 ─────────────────────────────────────────────────────────
    def observe(self, user: str, item: str, behavior: str, now: float) -> int:
        """
        消费一条用户行为事件，增量更新共现关系。

        Returns:
            本次实际更新的 Item Pair 数量（= 2 × |H_u|，共现矩阵对称）。
        """
        w = self.behavior_weights.get(behavior, 0.0)
        if w < self.min_weight:  # 低价值行为直接丢弃，减少噪声
            self.n_filtered += 1
            return 0

        hist = self.user_hist[user]
        # 滑动时间窗口：淘汰窗口外的历史行为
        while hist and now - hist[0][1] > self.window_sec:
            hist.popleft()

        touched = 0
        for prev_item, _ in hist:
            if prev_item == item:
                continue
            self._bump_pair(prev_item, item, w, now)
            self._bump_pair(item, prev_item, w, now)
            touched += 2

        hist.append((item, now))
        while len(hist) > self.history_len:  # 截断到最近 m 个物品
            hist.popleft()
        return touched

    # ── 惰性重算 Top-K 索引 ───────────────────────────────────────────────────
    def top_similar(self, item: str, now: float, k: int | None = None) -> list[tuple[str, float]]:
        """
        查询物品的 Top-K 相似物品；只有在 dirty 或缓存过期时才重新排序。
        """
        k = k or self.top_k
        cached = self.topk_cache.get(item)
        if cached is not None and item not in self.dirty and now - cached[0] <= self.cache_ttl:
            self.n_cache_hit += 1
            return cached[1][:k]

        neighbors = [(j, self._decayed_score(item, j, now)) for j in self.pair_score.get(item, {})]
        neighbors = [(j, s) for j, s in neighbors if s > 1e-6]
        neighbors.sort(key=lambda x: (-x[1], x[0]))
        self.topk_cache[item] = (now, neighbors[: self.top_k])
        self.dirty.discard(item)
        self.n_recompute += 1
        return neighbors[:k]

    # ── 在线召回 ─────────────────────────────────────────────────────────────
    def recall(self, user: str, now: float, top_n: int = 10) -> list[tuple[str, float]]:
        """
        在线召回：取用户最近行为集合 B_u，逐个查 TopK(b)，再合并去重排序。
        触发物品本身的新鲜度也参与加权（越近期的 trigger 权重越大）。
        """
        hist = list(self.user_hist.get(user, []))
        seen = {it for it, _ in hist}
        scores: dict[str, float] = defaultdict(float)
        for trigger, ts in hist[-5:]:  # 只用最近 5 个行为作为 trigger
            trigger_w = math.exp(-self.decay_lambda * max(0.0, now - ts))
            for cand, sim in self.top_similar(trigger, now):
                if cand not in seen:
                    scores[cand] += trigger_w * sim
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]


def _fmt(pairs: list[tuple[str, float]]) -> str:
    return ", ".join(f"{it}:{s:.2f}" for it, s in pairs) or "(空)"


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)

    n_users, n_items = 8, 20
    users = [f"u{idx}" for idx in range(n_users)]
    items = [f"i{idx:02d}" for idx in range(n_items)]
    behaviors = ["click", "click", "click", "long_view", "like", "follow", "share"]

    model = OnlineCF(history_len=20, window_sec=600.0, decay_lambda=0.004, top_k=10, cache_ttl=120.0)

    # ── 阶段一：常态行为流（t ∈ [0, 600]）─────────────────────────────────
    print("【阶段一】消费常态行为流 ...")
    now = 0.0
    for step in range(150):
        now += random.uniform(1.0, 5.0)
        user = random.choice(users)
        item = random.choice(items[:12])  # 常态流量集中在前 12 个物品
        model.observe(user, item, random.choice(behaviors), now)
        if step % 50 == 0:
            print(f"  t={now:7.1f}s  已更新 Pair 次数={model.n_pair_update:5d}  被过滤低价值行为={model.n_filtered}")

    probe = "i03"
    before = model.top_similar(probe, now, k=5)
    print(f"\n  热点注入前 {probe} 的 Top-5 相似物品：{_fmt(before)}")
    print(f"  索引重算次数={model.n_recompute}  缓存命中={model.n_cache_hit}")

    # ── 阶段二：热点突发（新物品 i17 与 i03 在短时间内被大量共同消费）────────
    print("\n【阶段二】注入热点行为流（i17 与 i03 高频共现）...")
    hot_item = "i17"
    for _ in range(40):
        now += random.uniform(0.5, 2.0)
        user = random.choice(users)
        model.observe(user, probe, "long_view", now)
        now += random.uniform(0.5, 2.0)
        model.observe(user, hot_item, random.choice(["like", "share", "follow"]), now)

    after = model.top_similar(probe, now, k=5)
    print(f"  热点注入后 {probe} 的 Top-5 相似物品：{_fmt(after)}")

    rank_before = {it: r for r, (it, _) in enumerate(before, 1)}
    rank_after = {it: r for r, (it, _) in enumerate(after, 1)}
    print("\n  Top-5 变化对比（名次：前 -> 后）：")
    for it in sorted(set(rank_before) | set(rank_after)):
        print(f"    {it}: {rank_before.get(it, '-')} -> {rank_after.get(it, '-')}")
    print(f"  热点物品 {hot_item} 的名次：{rank_before.get(hot_item, '未进榜')} -> {rank_after.get(hot_item, '未进榜')}")

    # ── 阶段三：时间衰减演示（静默 1 小时后分数自然衰减）──────────────────
    print("\n【阶段三】静默 3600s 后观察时间衰减：")
    s_now = model._decayed_score(probe, hot_item, now)
    s_later = model._decayed_score(probe, hot_item, now + 3600.0)
    print(f"  Score({probe}, {hot_item}): {s_now:.4f} -> {s_later:.4f}  (衰减比 {s_later / s_now:.6f})")

    # ── 阶段四：在线召回 ─────────────────────────────────────────────────
    target = "u0"
    recs = model.recall(target, now, top_n=5)
    print(f"\n【阶段四】用户 {target} 的 Online CF 召回 Top-5：")
    for rank, (iid, score) in enumerate(recs, 1):
        print(f"  {rank}. {iid}  score={score:.4f}")

    total_pairs = sum(len(v) for v in model.pair_score.values())
    print(
        f"\n统计：共现 Pair 总数={total_pairs}  增量更新次数={model.n_pair_update}  "
        f"Top-K 重算={model.n_recompute}  缓存命中={model.n_cache_hit}"
    )
