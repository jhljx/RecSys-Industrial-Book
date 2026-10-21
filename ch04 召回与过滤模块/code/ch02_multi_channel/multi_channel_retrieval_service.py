"""
对应正文第 4 章 §4.2「多路召回架构」—— 完整的多路召回服务链路。

正文把多路召回划分为三个阶段，本脚本按这个结构逐阶段实现：
  1) 召回前处理：构建触发信号 Trigger（近期点击/点赞/收藏 + 兴趣标签）、
     获取辅助特征（画像/上下文/设备）、访问外部存储（Redis、索引服务、画像服务）；
  2) 多路召回执行：8 路召回源（热门/I2I/U2I/垂类Tag/U2U2I/U2I2I/实时行为/图网络）
     用线程池并发触发，每路有独立超时；超时或异常的召回源走降级兜底；
  3) 召回后处理：按 Item ID 去重（保留召回来源 reason）、轻量级过滤
     （下架/违规/用户屏蔽）、Quota 配额控制、整体截断。

关键机制与公式：
  - 并发召回总耗时 ≈ max_k(T_k)，而非 Σ_k T_k（正文：异步 RPC / 多线程调度）
  - 单路超时保护：T_k > timeout_k 时丢弃该路结果并计入超时率
  - 兜底：若最终候选数 < min_candidates，用热门池补齐（正文 §4.4 兜底机制）
  - Quota 截断：对召回源 k 保留 min(|R_k|, quota_k)，再整体截断到 max_candidates

[MOCK] 本文件中的 MockRedis、MockForwardIndex、MockInvertedIndex、MockVectorIndex、
       MockProfileService 全部是教学用的内存假实现，用 time.sleep 模拟 RPC 网络延迟，
       不访问任何真实存储；延迟与失败率由构造参数注入，便于复现超时/降级场景。
"""

import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field


# ============================================================ [MOCK] 存储与索引层
class MockLatency:
    """[MOCK] 统一的延迟/故障注入器：按名字给出一次调用的模拟耗时与是否失败。"""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.lock = threading.Lock()
        self.calls: dict[str, int] = defaultdict(int)

    def hit(self, name: str, base_ms: float, jitter_ms: float, fail_rate: float = 0.0) -> None:
        with self.lock:
            self.calls[name] += 1
            ms = base_ms + self.rng.random() * jitter_ms
            failed = self.rng.random() < fail_rate
        time.sleep(ms / 1000.0)
        if failed:
            raise RuntimeError(f"[MOCK] {name} RPC failed")


class MockRedis:
    """[MOCK] 内存 KV，模拟 Redis：兴趣 Tag -> 物品列表、用户近期行为等。"""

    def __init__(self, lat: MockLatency):
        self.store: dict[str, object] = {}
        self.lat = lat

    def set(self, key: str, value: object) -> None:
        self.store[key] = value

    def get(self, key: str, default=None):
        self.lat.hit("redis.get", base_ms=1.0, jitter_ms=1.0)
        return self.store.get(key, default)


class MockForwardIndex:
    """[MOCK] 正排索引：Item ID -> 物品完整特征（标题/类目/标签/统计/状态/向量）。"""

    def __init__(self, items: dict[str, dict], lat: MockLatency):
        self.items = items
        self.lat = lat

    def mget(self, item_ids: list[str]) -> dict[str, dict]:
        self.lat.hit("forward_index.mget", base_ms=2.0, jitter_ms=2.0)
        return {i: self.items[i] for i in item_ids if i in self.items}


class MockInvertedIndex:
    """[MOCK] 倒排索引：触发项（tag/类目/地域/item）-> 倒排拉链（关联物品列表）。"""

    def __init__(self, lists: dict[str, list[str]], lat: MockLatency, fail_rate: float = 0.0):
        self.lists = lists
        self.lat = lat
        self.fail_rate = fail_rate

    def seek(self, key: str, top_n: int = 200) -> list[str]:
        self.lat.hit("inverted_index.seek", base_ms=3.0, jitter_ms=3.0, fail_rate=self.fail_rate)
        return self.lists.get(key, [])[:top_n]


class MockVectorIndex:
    """[MOCK] ANN 向量索引服务：用户向量 -> Top-N 近邻物品（这里用暴力内积代替 ANN）。"""

    def __init__(self, vectors: dict[str, list[float]], lat: MockLatency,
                 base_ms: float = 12.0, fail_rate: float = 0.0):
        self.vectors = vectors
        self.lat = lat
        self.base_ms = base_ms
        self.fail_rate = fail_rate

    def search(self, query: list[float], top_n: int) -> list[tuple[str, float]]:
        self.lat.hit("vector_index.search", base_ms=self.base_ms, jitter_ms=8.0,
                     fail_rate=self.fail_rate)
        scored = [(iid, sum(a * b for a, b in zip(query, vec)))
                  for iid, vec in self.vectors.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]


class MockProfileService:
    """[MOCK] 用户画像服务：长期兴趣标签、人口属性、地域。"""

    def __init__(self, profiles: dict[str, dict], lat: MockLatency):
        self.profiles = profiles
        self.lat = lat

    def get(self, uid: str) -> dict:
        self.lat.hit("profile.get", base_ms=4.0, jitter_ms=3.0)
        return self.profiles.get(uid, {})


# ============================================================ 数据结构
@dataclass
class RetrieveRequest:
    """一次召回请求：用户、上下文、设备信息。"""
    uid: str
    scene: str = "home_feed"
    city: str = "beijing"
    device: str = "ios"
    hour: int = 20


@dataclass
class TriggerContext:
    """召回前处理产出：触发信号 + 辅助特征。"""
    uid: str
    recent_clicks: list[str] = field(default_factory=list)
    recent_likes: list[str] = field(default_factory=list)
    realtime_clicks: list[str] = field(default_factory=list)
    interest_tags: list[str] = field(default_factory=list)
    user_vector: list[float] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    city: str = ""
    pre_ms: float = 0.0


@dataclass
class ChannelResult:
    """单路召回结果与可观测指标。"""
    name: str
    items: list[tuple[str, float]]
    cost_ms: float
    status: str  # ok / timeout / error / fallback
    quota: int


# ============================================================ 召回前处理
def pre_retrieval(req: RetrieveRequest, redis: MockRedis, profile_svc: MockProfileService,
                  pool: ThreadPoolExecutor) -> TriggerContext:
    """
    §4.2 召回前处理：三件事并发做——取行为序列（Redis）、取画像（画像服务）、取用户向量。
    注意正文强调「不直接使用全部用户行为」，这里对点击序列只截取最近 N 条作为 Trigger。
    """
    t0 = time.perf_counter()
    f_behavior = pool.submit(redis.get, f"behavior:{req.uid}", {})
    f_profile = pool.submit(profile_svc.get, req.uid)
    f_vector = pool.submit(redis.get, f"user_vec:{req.uid}", [])

    behavior = f_behavior.result() or {}
    profile = f_profile.result() or {}
    user_vec = f_vector.result() or []

    ctx = TriggerContext(
        uid=req.uid,
        recent_clicks=behavior.get("clicks", [])[:10],      # Trigger: 近期点击
        recent_likes=behavior.get("likes", [])[:5],         # Trigger: 近期点赞
        realtime_clicks=behavior.get("clicks", [])[:2],     # Trigger: 当前会话即时行为
        interest_tags=profile.get("tags", [])[:4],          # Trigger: 兴趣标签
        user_vector=user_vec,
        profile=profile,
        city=req.city,
    )
    ctx.pre_ms = (time.perf_counter() - t0) * 1000
    return ctx


# ============================================================ 8 路召回源
def ch_hot(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """热门召回：全站热度 Top-N，保障基础覆盖与新用户冷启动。"""
    hot = res["redis"].get("hot:global", [])
    return [(iid, float(sc)) for iid, sc in hot]


def ch_i2i(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """I2I 召回：对每个 Trigger 物品查 ItemCF 倒排拉链，分数按 trigger 位置衰减。"""
    out: dict[str, float] = {}
    inv: MockInvertedIndex = res["i2i_index"]
    for pos, trig in enumerate(ctx.recent_clicks[:5]):
        decay = 0.9 ** pos
        for rank, iid in enumerate(inv.seek(f"i2i:{trig}", top_n=30)):
            out[iid] = max(out.get(iid, 0.0), decay / (1 + rank) ** 0.5)
    return sorted(out.items(), key=lambda x: -x[1])


def ch_u2i(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """U2I 召回：双塔用户向量过 ANN 索引（正文：双塔 + ANN 检索）。"""
    if not ctx.user_vector:
        return []
    return res["vector_index"].search(ctx.user_vector, top_n=120)


def ch_tag(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """垂类/Tag 召回：按用户兴趣标签查 Tag 倒排拉链。"""
    out: dict[str, float] = {}
    inv: MockInvertedIndex = res["tag_index"]
    for pos, tag in enumerate(ctx.interest_tags):
        w = 1.0 / (1 + pos)
        for rank, iid in enumerate(inv.seek(f"tag:{tag}", top_n=40)):
            out[iid] = max(out.get(iid, 0.0), w * 0.98 ** rank)
    return sorted(out.items(), key=lambda x: -x[1])


def ch_u2u2i(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """U2U2I 召回：先查相似用户拉链，再聚合相似用户的交互物品（UserCF 思想）。"""
    inv: MockInvertedIndex = res["u2u_index"]
    redis: MockRedis = res["redis"]
    out: dict[str, float] = defaultdict(float)
    for sim_uid in inv.seek(f"u2u:{ctx.uid}", top_n=5):
        peer = redis.get(f"behavior:{sim_uid}", {}) or {}
        for rank, iid in enumerate(peer.get("clicks", [])[:20]):
            out[iid] += 1.0 / (1 + rank)
    return sorted(out.items(), key=lambda x: -x[1])


def ch_u2i2i(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """U2I2I 召回：在历史物品上做两跳扩展（trigger -> 相似物品 -> 再相似物品）。"""
    inv: MockInvertedIndex = res["i2i_index"]
    out: dict[str, float] = defaultdict(float)
    for trig in ctx.recent_likes[:3]:
        hop1 = inv.seek(f"i2i:{trig}", top_n=8)
        for a in hop1:
            for rank, b in enumerate(inv.seek(f"i2i:{a}", top_n=8)):
                out[b] += 0.5 / (1 + rank)
    return sorted(out.items(), key=lambda x: -x[1])


def ch_realtime(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """实时行为召回：只用当前会话刚刚发生的行为触发，权重最高。"""
    inv: MockInvertedIndex = res["i2i_index"]
    out: dict[str, float] = {}
    for trig in ctx.realtime_clicks:
        for rank, iid in enumerate(inv.seek(f"i2i:{trig}", top_n=25)):
            out[iid] = max(out.get(iid, 0.0), 1.5 / (1 + rank) ** 0.5)
    return sorted(out.items(), key=lambda x: -x[1])


def ch_graph(ctx: TriggerContext, res: dict) -> list[tuple[str, float]]:
    """图网络召回：GNN 产出的图 Embedding 索引（此处复用一个独立的慢索引服务）。"""
    if not ctx.user_vector:
        return []
    return res["graph_index"].search(ctx.user_vector, top_n=60)


# ============================================================ 多路召回执行
def _timed(fn, ctx: TriggerContext, res: dict, box: dict):
    """在工作线程里记录该路召回自身的真实耗时（不含主线程排队等待）。"""
    def run():
        t0 = time.perf_counter()
        try:
            return fn(ctx, res)
        finally:
            box["cost_ms"] = (time.perf_counter() - t0) * 1000
    return run


def execute_channels(ctx: TriggerContext, res: dict, channels: list[dict],
                     pool: ThreadPoolExecutor) -> tuple[list[ChannelResult], float]:
    """
    §4.2 多路召回执行：线程池并发提交所有召回源，每路独立 timeout。

    两个容易写错的细节，这里都按线上正确做法实现：
      1) 超时必须用「绝对 deadline」判定（submit 时刻 + timeout），
         否则按列表顺序逐个 result(timeout=...) 时，前面的等待会把后面的超时吃掉；
      2) 单路耗时要在工作线程内部测量，否则测到的是「排队 + 执行」的混合值。
    超时 / 异常的召回源标记 status 并返回空结果，后续由兜底机制补齐。
    """
    t0 = time.perf_counter()
    futures, deadlines, boxes = {}, {}, {}
    for cfg in channels:
        name = cfg["name"]
        boxes[name] = {"cost_ms": 0.0}
        deadlines[name] = time.perf_counter() + cfg["timeout_ms"] / 1000.0
        futures[name] = pool.submit(_timed(cfg["fn"], ctx, res, boxes[name]))

    results: list[ChannelResult] = []
    for cfg in channels:
        name = cfg["name"]
        left = max(0.0, deadlines[name] - time.perf_counter())
        try:
            items = futures[name].result(timeout=left)
            status = "ok" if items else "empty"
        except FutureTimeout:
            items, status = [], "timeout"
            futures[name].cancel()
            # 超时时 worker 还在跑，box 里没有耗时；对外记为「至少用掉了超时预算」
            boxes[name]["cost_ms"] = cfg["timeout_ms"]
        except Exception:
            items, status = [], "error"
        results.append(ChannelResult(name, items, boxes[name]["cost_ms"], status, cfg["quota"]))
    return results, (time.perf_counter() - t0) * 1000


# ============================================================ 召回后处理
def post_retrieval(results: list[ChannelResult], res: dict, ctx: TriggerContext,
                   max_candidates: int, min_candidates: int) -> dict:
    """
    §4.2 召回后处理 + §4.4 融合：
      去重（保留 reason）-> 轻量级过滤 -> Quota 配额 -> 整体截断 -> 兜底补齐
    reason 采用「召回源优先级」口径（正文给出的两种做法之一，实现最简单）。
    """
    priority = {c.name: idx for idx, c in enumerate(results)}  # 列表顺序即优先级

    # ---- Quota 配额控制：每路先按自身分数截到 quota
    quota_kept: list[tuple[str, str, float]] = []
    quota_stat = []
    for c in results:
        kept = sorted(c.items, key=lambda x: -x[1])[:c.quota]
        quota_stat.append((c.name, len(c.items), len(kept)))
        quota_kept.extend((iid, c.name, sc) for iid, sc in kept)

    # ---- 去重：同一 item 命中多路，reason 取优先级最高的那一路
    best: dict[str, tuple[str, float]] = {}
    hit_sources: dict[str, set[str]] = defaultdict(set)
    for iid, src, sc in quota_kept:
        hit_sources[iid].add(src)
        if iid not in best or priority[src] < priority[best[iid][0]]:
            best[iid] = (src, sc)
    dedup_in, dedup_out = len(quota_kept), len(best)

    # ---- 轻量级过滤：下架 / 违规 / 用户屏蔽作者（正文：简单且计算成本低的规则）
    fwd: MockForwardIndex = res["forward_index"]
    blocked_authors = set(ctx.profile.get("blocked_authors", []))
    feats = fwd.mget(list(best.keys()))
    filtered, drop_reason = {}, defaultdict(int)
    for iid, (src, sc) in best.items():
        f = feats.get(iid)
        if f is None:
            drop_reason["no_feature"] += 1
            continue
        if f["status"] != "online":
            drop_reason[f"status={f['status']}"] += 1
            continue
        if f["author"] in blocked_authors:
            drop_reason["blocked_author"] += 1
            continue
        filtered[iid] = (src, sc)

    # ---- 整体截断：跨源按「源权重 × 源内分数」排序后截断
    src_weight = {c.name: 1.0 / (1 + priority[c.name] * 0.1) for c in results}
    ordered = sorted(filtered.items(), key=lambda kv: -src_weight[kv[1][0]] * kv[1][1])
    truncated = ordered[:max_candidates]

    # ---- 兜底：候选不足时用热门池补齐（§4.4 召回源兜底机制）
    backup_added = 0
    if len(truncated) < min_candidates:
        have = {iid for iid, _ in truncated}
        hot = [(iid, sc) for iid, sc in res["redis"].get("hot:global", []) if iid not in have]
        # 一次批量取特征，而不是逐个物品打 RPC（线上同理：兜底也要走 batch 接口）
        hot_feats = fwd.mget([iid for iid, _ in hot if iid not in feats])
        for iid, sc in hot:
            if len(truncated) >= min_candidates:
                break
            f = feats.get(iid) or hot_feats.get(iid)
            if f is None or f["status"] != "online" or f["author"] in blocked_authors:
                continue
            truncated.append((iid, ("backup_hot", float(sc))))
            have.add(iid)
            backup_added += 1

    return {
        "candidates": [(iid, src, sc) for iid, (src, sc) in truncated],
        "quota_stat": quota_stat,
        "dedup_in": dedup_in,
        "dedup_out": dedup_out,
        "drop_reason": dict(drop_reason),
        "backup_added": backup_added,
        "multi_hit": sum(1 for s in hit_sources.values() if len(s) > 1),
    }


# ============================================================ 服务编排
def retrieve_service(req: RetrieveRequest, res: dict, channels: list[dict],
                     pool: ThreadPoolExecutor, max_candidates: int = 3000,
                     min_candidates: int = 60) -> dict:
    """一次完整的召回服务调用：前处理 -> 并发多路召回 -> 后处理。"""
    t0 = time.perf_counter()
    ctx = pre_retrieval(req, res["redis"], res["profile"], pool)
    results, exec_ms = execute_channels(ctx, res, channels, pool)
    post = post_retrieval(results, res, ctx, max_candidates, min_candidates)
    post.update({
        "ctx": ctx,
        "channel_results": results,
        "exec_ms": exec_ms,
        "total_ms": (time.perf_counter() - t0) * 1000,
    })
    return post


# ============================================================ 合成数据 [MOCK]
def build_world(rng: random.Random, n_items: int = 600, n_users: int = 40,
                dim: int = 8, u2i_fail: float = 0.0, graph_slow_ms: float = 12.0):
    """[MOCK] 造一个小世界：物品库、正排、倒排拉链、向量索引、用户行为与画像。"""
    lat = MockLatency(rng)
    tags = ["tech", "sports", "food", "travel", "game", "movie"]
    authors = [f"a{i}" for i in range(30)]

    items, vectors = {}, {}
    for i in range(n_items):
        iid = f"i{i:04d}"
        status = "online"
        r = rng.random()
        if r < 0.05:
            status = "offline"
        elif r < 0.08:
            status = "illegal"
        items[iid] = {
            "item_id": iid,
            "tag": rng.choice(tags),
            "category": rng.choice(["c1", "c2", "c3", "c4"]),
            "author": rng.choice(authors),
            "status": status,
            "ctr": round(rng.betavariate(2, 20), 4),
        }
        vectors[iid] = [rng.gauss(0, 1) for _ in range(dim)]

    all_ids = list(items.keys())
    # 倒排拉链：i2i 相似物品、tag 拉链、u2u 相似用户
    i2i_lists = {f"i2i:{iid}": rng.sample(all_ids, 40) for iid in all_ids}
    tag_lists = {}
    for t in tags:
        pool_t = [iid for iid, f in items.items() if f["tag"] == t]
        tag_lists[f"tag:{t}"] = sorted(pool_t, key=lambda x: -items[x]["ctr"])[:60]

    redis = MockRedis(lat)
    uids = [f"u{i:03d}" for i in range(n_users)]
    for uid in uids:
        redis.set(f"behavior:{uid}", {
            "clicks": rng.sample(all_ids, 24),
            "likes": rng.sample(all_ids, 8),
        })
        redis.set(f"user_vec:{uid}", [rng.gauss(0, 1) for _ in range(dim)])
    hot = sorted(items.values(), key=lambda f: -f["ctr"])[:200]
    redis.set("hot:global", [(f["item_id"], f["ctr"]) for f in hot])

    u2u_lists = {f"u2u:{uid}": rng.sample([x for x in uids if x != uid], 8) for uid in uids}
    profiles = {
        uid: {
            "tags": rng.sample(tags, 4),
            "age": rng.randint(18, 50),
            "blocked_authors": rng.sample(authors, 2),
        }
        for uid in uids
    }

    res = {
        "lat": lat,
        "redis": redis,
        "forward_index": MockForwardIndex(items, lat),
        "i2i_index": MockInvertedIndex(i2i_lists, lat),
        "tag_index": MockInvertedIndex(tag_lists, lat),
        "u2u_index": MockInvertedIndex(u2u_lists, lat),
        "vector_index": MockVectorIndex(vectors, lat, base_ms=12.0, fail_rate=u2i_fail),
        "graph_index": MockVectorIndex(vectors, lat, base_ms=graph_slow_ms, fail_rate=0.0),
        "profile": MockProfileService(profiles, lat),
        "items": items,
    }
    return res, uids


def default_channels() -> list[dict]:
    """8 路召回源配置：顺序即 reason 优先级，quota 对应正文表 4.3 的配额思想。"""
    return [
        {"name": "hot",      "fn": ch_hot,      "timeout_ms": 20.0, "quota": 500},
        {"name": "realtime", "fn": ch_realtime, "timeout_ms": 30.0, "quota": 300},
        {"name": "i2i",      "fn": ch_i2i,      "timeout_ms": 40.0, "quota": 1000},
        {"name": "u2i",      "fn": ch_u2i,      "timeout_ms": 35.0, "quota": 1500},
        {"name": "tag",      "fn": ch_tag,      "timeout_ms": 40.0, "quota": 600},
        {"name": "u2u2i",    "fn": ch_u2u2i,    "timeout_ms": 40.0, "quota": 400},
        {"name": "u2i2i",    "fn": ch_u2i2i,    "timeout_ms": 60.0, "quota": 400},
        {"name": "graph",    "fn": ch_graph,    "timeout_ms": 25.0, "quota": 1000},
    ]


def print_report(out: dict) -> None:
    print(f"  前处理耗时 {out['ctx'].pre_ms:6.2f} ms | 多路并发执行 {out['exec_ms']:6.2f} ms "
          f"| 服务总耗时 {out['total_ms']:6.2f} ms")
    print(f"  {'召回源':<10}{'状态':<9}{'耗时(ms)':>10}{'返回数':>8}{'quota':>7}{'保留数':>8}")
    print("  " + "-" * 54)
    kept_by_src = defaultdict(int)
    for iid, src, _ in out["candidates"]:
        kept_by_src[src] += 1
    quota_map = {n: (raw, kept) for n, raw, kept in out["quota_stat"]}
    seq_sum = 0.0
    for c in out["channel_results"]:
        raw, kept = quota_map[c.name]
        seq_sum += c.cost_ms
        print(f"  {c.name:<10}{c.status:<9}{c.cost_ms:>10.2f}{raw:>8}{c.quota:>7}"
              f"{kept_by_src[c.name]:>8}")
    print(f"  串行累加耗时 {seq_sum:.2f} ms  vs  并发耗时 {out['exec_ms']:.2f} ms "
          f"(并发收益 {seq_sum / max(out['exec_ms'], 1e-9):.2f}x)")
    print(f"  去重: {out['dedup_in']} -> {out['dedup_out']} "
          f"(去重率 {1 - out['dedup_out'] / max(out['dedup_in'], 1):.2%}, "
          f"多源命中 {out['multi_hit']} 个)")
    print(f"  轻量级过滤丢弃: {out['drop_reason']}")
    print(f"  兜底补充: {out['backup_added']} 个 | 最终候选集: {len(out['candidates'])}")


def main() -> None:
    random.seed(42)
    rng = random.Random(42)

    print("=" * 78)
    print("场景 1：全部召回源健康")
    print("=" * 78)
    res, uids = build_world(rng)
    with ThreadPoolExecutor(max_workers=12) as pool:
        out = retrieve_service(RetrieveRequest(uid=uids[0]), res, default_channels(), pool)
        print_report(out)
        print("  候选集前 8 条 (item, reason, score):")
        for iid, src, sc in out["candidates"][:8]:
            print(f"    {iid}  reason={src:<9} score={sc:.4f}  tag={res['items'][iid]['tag']}")

    print()
    print("=" * 78)
    print("场景 2：U2I 向量索引服务 100% 故障 + 图召回变慢（30ms > 25ms 超时阈值）")
    print("=" * 78)
    rng2 = random.Random(42)
    res2, uids2 = build_world(rng2, u2i_fail=1.0, graph_slow_ms=30.0)
    with ThreadPoolExecutor(max_workers=12) as pool:
        out2 = retrieve_service(RetrieveRequest(uid=uids2[0]), res2, default_channels(), pool)
        print_report(out2)

    print()
    print("=" * 78)
    print("场景 3：极端降级——只剩热门召回可用（min_candidates 兜底生效）")
    print("=" * 78)
    rng3 = random.Random(42)
    res3, uids3 = build_world(rng3)
    only_hot = [c for c in default_channels() if c["name"] == "hot"]
    only_hot[0] = dict(only_hot[0], quota=30)  # 故意把配额压到低于 min_candidates
    with ThreadPoolExecutor(max_workers=4) as pool:
        out3 = retrieve_service(RetrieveRequest(uid=uids3[0]), res3, only_hot, pool,
                                min_candidates=60)
        print_report(out3)

    print()
    print("=" * 78)
    print("场景 4：吞吐压测——20 个并发请求的 P50/P99 耗时")
    print("=" * 78)
    rng4 = random.Random(7)
    res4, uids4 = build_world(rng4)
    chans = default_channels()
    lat_list = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        for uid in uids4[:20]:
            o = retrieve_service(RetrieveRequest(uid=uid), res4, chans, pool)
            lat_list.append((o["total_ms"], len(o["candidates"])))
    costs = sorted(c for c, _ in lat_list)
    sizes = [n for _, n in lat_list]
    print(f"  请求数 {len(costs)} | 耗时 P50={costs[len(costs) // 2]:.2f}ms "
          f"P90={costs[int(len(costs) * 0.9)]:.2f}ms max={costs[-1]:.2f}ms")
    print(f"  候选集规模 min={min(sizes)} avg={sum(sizes) / len(sizes):.1f} max={max(sizes)}")
    print(f"  [MOCK] 存储调用次数统计: {dict(res4['lat'].calls)}")


if __name__ == "__main__":
    main()
