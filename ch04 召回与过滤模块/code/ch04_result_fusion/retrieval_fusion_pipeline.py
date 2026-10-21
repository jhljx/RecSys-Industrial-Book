"""
对应正文第 4 章 §4.4「召回结果融合」—— 去重、Quota 配额控制、截断、兜底的完整链路。

正文把融合模块的流程定义为:去重 -> 配额控制 -> 截断 -> 兜底,核心目标是
「保证候选集规模满足排序要求 / 保证各召回源合理流量分配 / 提升覆盖率与多样性 /
提高系统稳定性与鲁棒性」。本脚本按这个顺序逐步实现并打印每一步的候选规模变化。

实现要点(全部对齐正文):
  1) 去重(§4.4 召回结果去重)
     - 按 Item ID 唯一化;
     - 为每个物品保留一个「召回来源标识 retrieval reason」。正文明确指出 reason
       只记录其中一路,工业界两种做法:
         (a) 按召回源优先级选择(热门 > CF > 双塔 ...);
         (b) 按召回源内部得分选择(但不同源打分不可比)。
       脚本两种都实现,并打印二者给出的 reason 分布差异;
     - 跨路同物品的分数归并:正文指出各路分数不可比,所以归并前先做
       「组内排名归一化」 norm_k(i) = 1 - rank_k(i) / |R_k| ,再用
         merged(i) = max_k  w_k · norm_k(i)              (取最大,保守)
         merged(i) = 1 - Π_k (1 - w_k · norm_k(i))        (概率或,多路命中加分)
       两种口径都算出来并对比,w_k 为召回源权重。
  2) Quota 配额控制(§4.4 召回源配额控制)
     - 复刻正文表 4.x 的例子:热门 10000 -> 500,Item-CF 3000 -> 1000,
       双塔 5000 -> 1500,图召回 1000 -> 1000,目标总量 <= 5000;
     - 静态 Quota 之外实现「动态配额调整」:按各路近期下游通过率(粗排通过率)
       与独占贡献率重新分配配额,quota_k ∝ (base_k)^(1-β) · (gain_k)^β 后归一化。
  3) 候选集截断(§4.4 候选集截断)
     - 四种口径都实现:按召回源 Quota 截断、按归并分数全局截断、
       按业务优先级截断、按动态流量分配截断,并对比截断后各路占比。
  4) 兜底机制(§4.4 召回源兜底机制)
     - 注入「双塔索引异常返回空 / 图召回 RPC 超时」故障,触发多层兜底:
       类目热门 -> 全局热门 -> Redis 缓存上次结果 -> 历史推荐结果;
     - 保证最终候选数不低于 min_candidates。
  5) 召回监控与质量评估(§4.4 召回监控与质量评估)
     - 正文列出的监控指标:各路召回数量、覆盖率、去重率、CTR、观看时长、
       超时率、空结果率,全部统计并打印成监控表;
     - 另加两个工业界常用的质量指标:
         独占贡献率 unique_k = |只被 k 召回且进入最终候选| / |最终候选|
         下游通过率 pass_k   = |k 的候选进入粗排 Top N| / |k 的候选数|
         两路重合度 jaccard(k, l) = |R_k ∩ R_l| / |R_k ∪ R_l|

[MOCK] MockRetrievalChannel 用合成分布生成各路候选(可注入超时/空结果/延迟),
       MockRedisCache 与 MockHistoryStore 是内存假存储,MockPreRanker 是一个
       线性打分器充当下游粗排,用于计算下游通过率。全部不访问真实服务。
"""

import random
from collections import Counter

N_ITEMS = 60000            # 教学规模的物品库(真实场景为亿级)
TOP_N_PRERANK = 500        # 粗排输出规模(正文:精排输入 500)
MAX_CANDIDATES = 5000      # 粗排输入规模上限(正文:粗排输入 5000)
MIN_CANDIDATES = 3000      # 兜底阈值


# ============================================================ [MOCK] 召回通道
class MockRetrievalChannel:
    """[MOCK] 教学用召回源:按配置好的「物品池 + 偏好」生成候选与内部得分。

    fail_mode: None / "empty"(索引失效返回空) / "timeout"(RPC 超时)
    内部得分刻意使用不同量纲(例如热门用播放量、双塔用 0~1 余弦),以复现
    正文指出的「不同召回源打分不太可比」问题。
    """

    def __init__(self, name, reason_id, priority, raw_size, quota, pool,
                 score_fn, weight=1.0, fail_mode=None, latency_ms=0.0):
        self.name = name
        self.reason_id = reason_id
        self.priority = priority          # 数字越小优先级越高
        self.raw_size = raw_size
        self.quota = quota
        self.pool = pool
        self.score_fn = score_fn
        self.weight = weight
        self.fail_mode = fail_mode
        self.latency_ms = latency_ms

    def retrieve(self, rng):
        """返回 (items, scores, status)。status ∈ ok / empty / timeout。"""
        if self.fail_mode == "timeout":
            return [], [], "timeout"
        if self.fail_mode == "empty":
            return [], [], "empty"
        items = rng.sample(self.pool, min(self.raw_size, len(self.pool)))
        scores = [self.score_fn(rng, k) for k in range(len(items))]
        order = sorted(range(len(items)), key=lambda k: -scores[k])
        return [items[k] for k in order], [scores[k] for k in order], "ok"


class MockRedisCache:
    """[MOCK] 兜底用的 Redis 缓存:存上一次请求的融合结果。"""

    def __init__(self):
        self.store = {}

    def put(self, uid, items):
        self.store[uid] = list(items)

    def get(self, uid):
        return list(self.store.get(uid, []))


class MockHistoryStore:
    """[MOCK] 兜底用的历史推荐结果存储。"""

    def __init__(self, rng, pool, n=1500):
        self.items = rng.sample(pool, n)

    def get(self, n):
        return self.items[:n]


class MockPreRanker:
    """[MOCK] 下游粗排:对候选做一次线性打分,用于统计各路「下游通过率」。"""

    def __init__(self, rng, n_items):
        self.q = [rng.random() for _ in range(n_items)]     # 物品内在质量

    def rank(self, cand_scores, top_n):
        s = {i: 0.7 * self.q[i] + 0.3 * sc for i, sc in cand_scores.items()}
        return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])[:top_n]]


# ============================================================ 1. 去重 + 分数归并
def rank_normalize(items):
    """组内排名归一化:norm(i) = 1 - rank(i)/|R|,把不可比的原始分拉到同一量纲。"""
    n = max(len(items), 1)
    return {it: 1.0 - k / n for k, it in enumerate(items)}


def dedup_and_merge(results):
    """results: {name: (channel, items, scores)}。

    返回 merged = {item: {"sources": [...], "norm": {name: v},
                          "score_max": float, "score_or": float,
                          "reason_prio": name, "reason_score": name}}
    """
    merged = {}
    for name, (ch, items, _scores) in results.items():
        norm = rank_normalize(items)
        for it in items:
            e = merged.setdefault(it, {"sources": [], "norm": {}})
            e["sources"].append(name)
            e["norm"][name] = ch.weight * norm[it]
    return merged


def finalize_scores(merged, channels):
    prio = {c.name: c.priority for c in channels}
    for it, e in merged.items():
        e["score_max"] = max(e["norm"].values())
        p = 1.0
        for v in e["norm"].values():
            p *= (1.0 - min(v, 0.999))
        e["score_or"] = 1.0 - p
        # reason 口径 (a):按召回源优先级
        e["reason_prio"] = min(e["sources"], key=lambda n: prio[n])
        # reason 口径 (b):按召回源内部归一化得分
        e["reason_score"] = max(e["norm"].items(), key=lambda kv: kv[1])[0]
    return merged


# ============================================================ 2. Quota 配额控制
def apply_quota(results, quotas):
    """对召回源 k 保留 min(|R_k|, quota_k) 个候选(正文表:10000->500 等)。"""
    kept = {}
    for name, (ch, items, scores) in results.items():
        q = quotas[name]
        kept[name] = (ch, items[:q], scores[:q])
    return kept


def dynamic_quota(base_quota, gain, beta=0.6, total=MAX_CANDIDATES):
    """动态配额:quota_k ∝ base_k^(1-β) · gain_k^β,再归一化到 total。

    gain_k 取「独占贡献率 × 下游通过率」,即这一路带来的、别人带不来的有效候选。
    β=0 退化为静态配额,β=1 完全由效果驱动。
    """
    raw = {k: (base_quota[k] ** (1 - beta)) * (max(gain.get(k, 0.0), 1e-4) ** beta)
           for k in base_quota}
    s = sum(raw.values())
    out = {k: max(int(total * v / s), 50) for k, v in raw.items()}
    return out


# ============================================================ 3. 截断
def truncate_by_quota(kept, limit):
    """按召回源 Quota 截断:各路轮转取(round-robin),保证小召回源不被挤死。"""
    pools = {n: list(v[1]) for n, v in kept.items()}
    out, seen = [], set()
    while len(out) < limit and any(pools.values()):
        for n in list(pools):
            if not pools[n]:
                continue
            it = pools[n].pop(0)
            if it not in seen:
                seen.add(it)
                out.append((it, n))
            if len(out) >= limit:
                break
    return out


def truncate_by_score(merged, limit, key="score_or"):
    """按归并后的全局分数截断。"""
    order = sorted(merged.items(), key=lambda kv: -kv[1][key])[:limit]
    return [(it, e["reason_prio"]) for it, e in order]


def truncate_by_priority(kept, channels, limit):
    """按业务优先级截断:高优先级召回源先灌满,剩余额度给低优先级。"""
    out, seen = [], set()
    for ch in sorted(channels, key=lambda c: c.priority):
        if ch.name not in kept:
            continue
        for it in kept[ch.name][1]:
            if it in seen:
                continue
            seen.add(it)
            out.append((it, ch.name))
            if len(out) >= limit:
                return out
    return out


def truncate_by_traffic(kept, merged, weights, limit):
    """按动态流量分配截断:先按权重把 limit 拆成各路名额,各路内部按分数取。"""
    s = sum(weights.values())
    out, seen = [], set()
    for name, (ch, items, _) in kept.items():
        n_k = max(int(limit * weights[name] / s), 1)
        picked = 0
        for it in sorted(items, key=lambda x: -merged[x]["norm"].get(name, 0.0)):
            if it in seen:
                continue
            seen.add(it)
            out.append((it, name))
            picked += 1
            if picked >= n_k:
                break
    return out[:limit]


# ============================================================ 4. 兜底
def backup_fill(cand, min_n, cat_hot, global_hot, cache, history, uid):
    """多层兜底:类目热门 -> 全局热门 -> Redis 缓存 -> 历史推荐结果。"""
    seen = {it for it, _ in cand}
    out = list(cand)
    layers = [("backup_cat_hot", cat_hot),
              ("backup_global_hot", global_hot),
              ("backup_redis_cache", cache.get(uid)),
              ("backup_history", history.get(2000))]
    trace = []
    for name, pool in layers:
        if len(out) >= min_n:
            break
        added = 0
        for it in pool:
            if len(out) >= min_n:
                break
            if it not in seen:
                seen.add(it)
                out.append((it, name))
                added += 1
        trace.append((name, added, len(out)))
    return out, trace


# ============================================================ 5. 监控与质量评估
def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def monitor_table(results, merged, final, prerank_top, ctr_prior, dur_prior):
    """正文列出的监控指标 + 独占贡献率 / 下游通过率。"""
    rows = []
    for name, (ch, items, _) in results.items():
        n = len(items)
        dup = sum(1 for it in items if len(merged[it]["sources"]) > 1)
        uniq_final = sum(1 for it, r in final if r == name
                         and len(merged[it]["sources"]) == 1)
        in_final = sum(1 for it, r in final if r == name)
        passed = sum(1 for it in items if it in prerank_top)
        rows.append({
            "name": name,
            "n": n,
            "latency": ch.latency_ms,
            "coverage": n / N_ITEMS,
            "dup_rate": dup / max(n, 1),
            "ctr": ctr_prior[name],
            "duration": dur_prior[name],
            "timeout": 1.0 if ch.fail_mode == "timeout" else 0.0,
            "empty": 1.0 if ch.fail_mode == "empty" or n == 0 else 0.0,
            "in_final": in_final,
            "unique_rate": uniq_final / max(len(final), 1),
            "pass_rate": passed / max(n, 1),
        })
    return rows


# ============================================================ 主流程
def build_channels(rng):
    """构造正文表「召回源 Quota 示例」中的 4 路 + 2 路垂类/实时召回。"""
    hot_pool = list(range(0, 15000))                # 热门池偏头部
    cf_pool = list(range(5000, 35000))
    dssm_pool = list(range(10000, 55000))
    graph_pool = list(range(20000, 40000))
    tag_pool = list(range(30000, 50000))
    rt_pool = list(range(45000, 60000))
    return [
        MockRetrievalChannel("热门召回", 5501, 1, 10000, 500, hot_pool,
                             lambda r, k: 1e6 * (0.999 ** k), weight=0.6,
                             latency_ms=4.0),
        MockRetrievalChannel("ItemCF召回", 5512, 2, 3000, 1000, cf_pool,
                             lambda r, k: 0.8 * (0.998 ** k), weight=1.0,
                             latency_ms=12.0),
        MockRetrievalChannel("双塔召回", 5536, 3, 5000, 1500, dssm_pool,
                             lambda r, k: 0.95 - 0.00005 * k, weight=1.2,
                             latency_ms=18.0),
        MockRetrievalChannel("图召回", 5548, 4, 1000, 1000, graph_pool,
                             lambda r, k: 10.0 / (k + 3), weight=1.1,
                             latency_ms=25.0),
        MockRetrievalChannel("垂类Tag召回", 5560, 5, 2000, 600, tag_pool,
                             lambda r, k: 1.0 - k / 2000, weight=0.8,
                             latency_ms=6.0),
        MockRetrievalChannel("实时行为召回", 5571, 6, 800, 400, rt_pool,
                             lambda r, k: 2.0 * (0.99 ** k), weight=1.3,
                             latency_ms=9.0),
    ]


def main():
    random.seed(42)
    rng = random.Random(42)

    channels = build_channels(rng)
    ctr_prior = {"热门召回": 0.0312, "ItemCF召回": 0.0487, "双塔召回": 0.0561,
                 "图召回": 0.0529, "垂类Tag召回": 0.0405, "实时行为召回": 0.0683}
    dur_prior = {"热门召回": 23.4, "ItemCF召回": 31.8, "双塔召回": 36.2,
                 "图召回": 34.5, "垂类Tag召回": 27.1, "实时行为召回": 41.6}

    print("=" * 86)
    print("§4.4 召回结果融合:去重 -> Quota 配额控制 -> 截断 -> 兜底 -> 监控评估")
    print("=" * 86)

    # ---------------- 多路召回执行(正常场景)
    results = {}
    for ch in channels:
        items, scores, status = ch.retrieve(rng)
        results[ch.name] = (ch, items, scores)
    raw_total = sum(len(v[1]) for v in results.values())

    print("\n[Step 0] 多路召回原始结果")
    print(f"{'召回源':<16}{'reason_id':>10}{'原始候选数':>12}{'静态Quota':>11}{'源权重':>9}")
    print("-" * 60)
    for ch in channels:
        print(f"{ch.name:<16}{ch.reason_id:>10d}{len(results[ch.name][1]):>12d}"
              f"{ch.quota:>11d}{ch.weight:>9.1f}")
    print(f"{'合计':<16}{'':>10}{raw_total:>12d}{sum(c.quota for c in channels):>11d}")

    # ---------------- Step 1 去重 + 分数归并
    merged_all = finalize_scores(dedup_and_merge(results), channels)
    dedup_total = len(merged_all)
    hit_dist = Counter(len(e["sources"]) for e in merged_all.values())
    print(f"\n[Step 1] 去重:{raw_total} -> {dedup_total} "
          f"(整体去重率 {1 - dedup_total / raw_total:.2%})")
    print("  被命中召回源个数分布:", dict(sorted(hit_dist.items())))
    pa = Counter(e["reason_prio"] for e in merged_all.values())
    pb = Counter(e["reason_score"] for e in merged_all.values())
    print("\n  召回 reason 两种设置口径的分布差异(正文:只记录其中一路)")
    print(f"{'召回源':<16}{'按优先级(a)':>14}{'按内部得分(b)':>16}{'差值':>10}")
    print("-" * 58)
    for ch in channels:
        print(f"{ch.name:<16}{pa[ch.name]:>14d}{pb[ch.name]:>16d}"
              f"{pb[ch.name] - pa[ch.name]:>+10d}")

    multi = [it for it, e in merged_all.items() if len(e["sources"]) >= 3][:5]
    if multi:
        print("\n  跨路同物品的分数归并示例(多路命中 >= 3 的物品):")
        print(f"{'item':>8}{'命中源数':>9}{'max归并':>10}{'概率或归并':>12}   命中召回源")
        for it in multi:
            e = merged_all[it]
            print(f"{it:>8d}{len(e['sources']):>9d}{e['score_max']:>10.4f}"
                  f"{e['score_or']:>12.4f}   {'+'.join(e['sources'])}")

    # ---------------- Step 2 Quota 控制
    base_quota = {c.name: c.quota for c in channels}
    kept = apply_quota(results, base_quota)
    after_quota = len({it for v in kept.values() for it in v[1]})
    print("\n[Step 2] 静态 Quota 控制(正文表 4.x 口径)")
    print(f"{'召回源':<16}{'原始候选数':>12}{'保留数量':>10}{'保留比例':>10}")
    print("-" * 50)
    for ch in channels:
        n0, n1 = len(results[ch.name][1]), len(kept[ch.name][1])
        print(f"{ch.name:<16}{n0:>12d}{n1:>10d}{(n1 / max(n0, 1)):>10.1%}")
    print(f"{'合计(去重后)':<16}{dedup_total:>12d}{after_quota:>10d}")

    # ---------------- Step 3 四种截断口径对比
    prerank = MockPreRanker(rng, N_ITEMS)
    cand_scores = {it: e["score_or"] for it, e in merged_all.items()}
    prerank_top = set(prerank.rank(cand_scores, TOP_N_PRERANK))

    weights = {c.name: c.weight for c in channels}
    strategies = {
        "按召回源Quota轮转": truncate_by_quota(kept, MAX_CANDIDATES),
        "按归并得分全局": truncate_by_score(merged_all, MAX_CANDIDATES),
        "按业务优先级": truncate_by_priority(kept, channels, MAX_CANDIDATES),
        "按动态流量分配": truncate_by_traffic(kept, merged_all, weights, MAX_CANDIDATES),
    }
    print(f"\n[Step 3] 候选集截断到 {MAX_CANDIDATES}:四种口径下各召回源占比")
    hdr = f"{'截断策略':<20}{'候选数':>8}"
    for ch in channels:
        hdr += f"{ch.name:>14}"
    print(hdr)
    print("-" * (28 + 14 * len(channels)))
    for sname, out in strategies.items():
        cnt = Counter(r for _, r in out)
        line = f"{sname:<20}{len(out):>8d}"
        for ch in channels:
            line += f"{cnt[ch.name] / max(len(out), 1):>14.1%}"
        print(line)

    final = strategies["按动态流量分配"]

    # ---------------- Step 4 质量评估与监控(正常场景)
    rows = monitor_table(results, merged_all, final, prerank_top,
                         ctr_prior, dur_prior)
    print("\n[Step 4] 召回监控与质量评估(正文列出的监控指标)")
    print(f"{'召回源':<16}{'召回数':>8}{'耗时/ms':>9}{'覆盖率':>9}{'去重率':>9}"
          f"{'CTR':>8}{'时长/s':>8}{'超时率':>8}{'空率':>7}{'独占贡献':>10}{'下游通过':>10}")
    print("-" * 104)
    for r in rows:
        print(f"{r['name']:<16}{r['n']:>8d}{r['latency']:>9.1f}{r['coverage']:>9.2%}"
              f"{r['dup_rate']:>9.1%}{r['ctr']:>8.4f}{r['duration']:>8.1f}"
              f"{r['timeout']:>8.0%}{r['empty']:>7.0%}{r['unique_rate']:>10.2%}"
              f"{r['pass_rate']:>10.2%}")

    print("\n  两两召回源重合度 Jaccard(|R_k ∩ R_l| / |R_k ∪ R_l|):")
    names = [c.name for c in channels]
    print(" " * 16 + "".join(f"{n:>14}" for n in names))
    for a in names:
        line = f"{a:<16}"
        for b in names:
            line += f"{jaccard(results[a][1], results[b][1]):>14.3f}"
        print(line)

    # ---------------- Step 5 动态配额调整
    gain = {r["name"]: max(r["unique_rate"], 1e-4) * max(r["pass_rate"], 1e-4)
            for r in rows}
    dq = dynamic_quota(base_quota, gain, beta=0.6)
    print("\n[Step 5] 动态配额调整  quota_k ∝ base_k^(1-β)·gain_k^β  "
          "(β=0.6, gain = 独占贡献率 × 下游通过率)")
    print(f"{'召回源':<16}{'静态Quota':>11}{'gain':>11}{'动态Quota':>11}{'变化':>10}")
    print("-" * 60)
    for ch in channels:
        b, d = base_quota[ch.name], dq[ch.name]
        print(f"{ch.name:<16}{b:>11d}{gain[ch.name]:>11.5f}{d:>11d}"
              f"{(d - b) / b:>+10.1%}")

    kept2 = apply_quota(results, dq)
    final2 = truncate_by_traffic(kept2, merged_all, weights, MAX_CANDIDATES)
    top2 = set(prerank.rank({it: merged_all[it]["score_or"] for it, _ in final2},
                            TOP_N_PRERANK))
    base_top = set(prerank.rank({it: merged_all[it]["score_or"] for it, _ in final},
                                TOP_N_PRERANK))
    good = {i for i in range(N_ITEMS) if prerank.q[i] > 0.9}
    print(f"\n  静态配额最终候选 {len(final)} 个,其中优质物品(内在质量>0.9) "
          f"{len({it for it, _ in final} & good)} 个;")
    print(f"  动态配额最终候选 {len(final2)} 个,其中优质物品 "
          f"{len({it for it, _ in final2} & good)} 个;")
    print(f"  两者粗排 Top{TOP_N_PRERANK} 的重合度 "
          f"{len(top2 & base_top) / TOP_N_PRERANK:.1%}")

    # ---------------- Step 6 故障注入 + 兜底
    print("\n" + "=" * 86)
    print("[Step 6] 故障注入与兜底机制(正文:索引失效 / RPC 超时 -> 多层兜底)")
    print("=" * 86)
    cache = MockRedisCache()
    cache.put("u_demo", [it for it, _ in final][:2500])
    history = MockHistoryStore(rng, list(range(N_ITEMS)))
    cat_hot = list(range(0, 1200))
    global_hot = list(range(0, 2500))

    for ch in channels:
        if ch.name == "双塔召回":
            ch.fail_mode = "empty"        # 向量索引服务异常
        if ch.name == "图召回":
            ch.fail_mode = "timeout"      # 图召回 RPC 超时

    results_f = {}
    for ch in channels:
        items, scores, status = ch.retrieve(rng)
        results_f[ch.name] = (ch, items, scores)
    merged_f = finalize_scores(dedup_and_merge(results_f), channels)
    kept_f = apply_quota(results_f, base_quota)
    final_f = truncate_by_traffic(kept_f, merged_f, weights, MAX_CANDIDATES)
    print(f"\n  故障后融合候选数 {len(final_f)} "
          f"(正常场景 {len(final)}),低于兜底阈值 {MIN_CANDIDATES}:"
          f" {len(final_f) < MIN_CANDIDATES}")
    filled, trace = backup_fill(final_f, MIN_CANDIDATES, cat_hot, global_hot,
                                cache, history, "u_demo")
    print(f"{'兜底层':<22}{'补充数量':>10}{'补充后候选数':>14}")
    print("-" * 48)
    for name, added, tot in trace:
        print(f"{name:<22}{added:>10d}{tot:>14d}")
    print(f"  兜底后最终候选数 {len(filled)},达到阈值: {len(filled) >= MIN_CANDIDATES}")

    rows_f = monitor_table(results_f, merged_f, filled, prerank_top,
                           ctr_prior, dur_prior)
    print("\n  故障场景监控表(超时率/空结果率告警)")
    print(f"{'召回源':<16}{'召回数':>8}{'超时率':>9}{'空结果率':>10}{'进入最终候选':>14}")
    print("-" * 58)
    for r in rows_f:
        print(f"{r['name']:<16}{r['n']:>8d}{r['timeout']:>9.0%}{r['empty']:>10.0%}"
              f"{r['in_final']:>14d}")
    bk = Counter(r for _, r in filled if r.startswith("backup_"))
    print("  兜底来源构成:", dict(bk))

    print("\n结论(正文 §4.4 要点回顾):")
    print("  1) 去重必须保留 reason,但 reason 只记一路;按优先级与按内部得分两种口径")
    print("     会给出明显不同的来源分布(见 Step 1 表),而正文指出 reason 不带来 AB 收益,")
    print("     因此工业界通常选实现最简单的优先级口径;")
    print("  2) 各路内部得分量纲不同(播放量 vs 余弦),直接比大小没有意义,")
    print("     融合前必须做组内排名归一化;")
    print("  3) Quota 的作用是防止热门召回这类大候选源独占候选集,从而维持多样性;")
    print("  4) 截断口径的选择直接改变候选集的来源构成,需要与业务目标对齐;")
    print("  5) 任何一路召回都会挂,兜底必须分层且保证最低候选规模,")
    print("     监控表的超时率/空结果率是发现这类故障的第一道信号。")


if __name__ == "__main__":
    main()
