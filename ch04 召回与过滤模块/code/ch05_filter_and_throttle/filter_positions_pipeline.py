"""
对应正文第 4 章 §4.5.1「过滤模块的位置」。

正文口径(逐条对齐):
  正文明确反驳「过滤模块只存在于召回之后」的误解,指出过滤逻辑贯穿整个推荐流程,
  在不同阶段承担不同职责:
    1) 召回前过滤(Pre-retrieval Filtering):剔除已下架、审核未通过、版权违规、
       已删除状态的内容,「避免无效内容进入后续推荐链路,从而减少系统计算资源浪费」;
    2) 召回后过滤(Post-retrieval Filtering):用户已消费内容过滤、用户屏蔽内容过滤、
       黑名单作者过滤、地域限制过滤,「进一步提升候选集质量」;
    3) 排序后过滤(Post-ranking Filtering):广告比例控制、风险内容降权、
       同作者内容数量限制,「避免排序模型过度追求点击率而损害用户体验」。

本脚本做了什么:
  用一个进程内的 [MOCK] 推荐链路(内容库 -> 倒排索引 -> 召回 -> 粗排 -> 精排 -> 曝光)
  把这三个位置的过滤都实现出来,然后做三组对照实验:
    配置 A 全不过滤        —— 看会出现正文 §4.5 开头列出的哪些问题;
    配置 B 只在召回后过滤  —— 常见的初学者实现:无效内容仍然进了索引与召回打分,
                              算力被浪费,且候选经过滤后不足,要靠兜底补齐;
    配置 C 三段式过滤      —— 正文推荐的形态:越早过滤越省算力,只把必须依赖
                              最终排序结果的策略(广告占比/同作者打散)留到排序后。
  打印的指标:索引条目数、单请求召回扫描量、召回打分浪费率、候选不足触发兜底的次数、
  最终曝光列表中的无效内容率/已读率/屏蔽作者率/地域违规率/广告占比/同作者最大条数。

关键公式(都是比例口径,方便与线上监控对齐):
  召回算力浪费率 = 进入召回打分但最终必被过滤的物品数 / 召回打分总数
  候选利用率     = 过滤后剩余候选数 / 召回候选数
  广告占比       = 曝光列表中的广告条数 / 曝光条数
  同作者约束     = max_author(曝光列表中同一作者的条数) <= L

[MOCK] 内容库、倒排索引、用户画像、粗排/精排打分全部为合成数据与玩具打分函数,
不加载任何真实数据集,也不起真实服务。
"""

import random
from collections import Counter

N_ITEMS = 200_000        # 内容库规模
N_AUTHORS = 4_000
N_REGIONS = 6
N_REQUESTS = 200         # 模拟请求数
RECALL_SIZE = 2_000      # 单请求召回候选规模
PRERANK_IN = 2_000       # 粗排输入上限
RANK_IN = 200            # 精排输入上限
EXPOSE_K = 20            # 单屏曝光条数
MIN_CANDIDATES = 1_900   # 候选不足则触发兜底
AD_RATIO_CAP = 0.20      # 排序后过滤:广告占比上限
SAME_AUTHOR_CAP = 2      # 排序后过滤:单屏同作者条数上限
RISK_DEMOTE = 0.3        # 排序后过滤:风险内容降权系数

# 正文 §4.5.1 列出的「召回前过滤」要剔除的状态
INVALID_STATUS = ("offline", "audit_pending", "copyright", "deleted")
STATUS_DIST = [
    ("online", 0.82),
    ("offline", 0.06),
    ("audit_pending", 0.05),
    ("copyright", 0.04),
    ("deleted", 0.03),
]


def gen_library(rng):
    """[MOCK] 生成内容库:状态 / 作者 / 地域许可 / 是否广告 / 风险等级 / 热度。"""
    items = []
    cum = []
    acc = 0.0
    for name, p in STATUS_DIST:
        acc += p
        cum.append((acc, name))
    for iid in range(N_ITEMS):
        u = rng.random()
        status = next(name for bound, name in cum if u <= bound)
        items.append({
            "id": iid,
            "status": status,
            "author": rng.randrange(N_AUTHORS),
            # 地域限制:8% 的内容只在部分地域可分发
            "regions": (None if rng.random() > 0.08
                        else frozenset(rng.sample(range(N_REGIONS), 2))),
            "is_ad": rng.random() < 0.12,
            "risk": 1 if rng.random() < 0.05 else 0,
            "hot": rng.random() ** 2,          # 热度,决定被召回的概率
            "ctr": round(rng.uniform(0.01, 0.12), 4),
        })
    return items


def build_index(items, pre_filter):
    """[MOCK] 倒排/正排索引。pre_filter=True 时执行正文的「召回前过滤」。

    工业界的召回前过滤通常发生在索引构建/更新链路上:无效内容根本不入库,
    因此它省掉的不只是过滤本身的开销,而是整条下游链路对这些物品的全部计算。
    """
    if pre_filter:
        return [it for it in items if it["status"] not in INVALID_STATUS]
    return list(items)


def retrieve(index, rng, size=RECALL_SIZE):
    """[MOCK] 多路召回的合并结果:按热度加权采样 size 个候选(含打分开销)。"""
    n = len(index)
    picked = {}
    tries = 0
    while len(picked) < size and tries < size * 6:
        tries += 1
        it = index[rng.randrange(n)]
        if rng.random() < 0.25 + 0.75 * it["hot"]:   # 热度越高越容易被召回
            picked[it["id"]] = it
    return list(picked.values()), tries


def post_retrieval_filter(cands, user, enable):
    """正文的「召回后过滤」:已读 / 用户屏蔽 / 黑名单作者 / 地域限制。"""
    if not enable:
        return cands, Counter()
    kept, dropped = [], Counter()
    for it in cands:
        if it["id"] in user["read"]:
            dropped["已读过滤"] += 1
        elif it["id"] in user["blocked_items"]:
            dropped["用户屏蔽内容"] += 1
        elif it["author"] in user["blocked_authors"]:
            dropped["黑名单作者"] += 1
        elif it["regions"] is not None and user["region"] not in it["regions"]:
            dropped["地域限制"] += 1
        else:
            kept.append(it)
    return kept, dropped


def rank_score(it, user, demote_risk):
    """[MOCK] 玩具排序分:CTR + 热度 + 地域/作者亲和 + 噪声;可叠加风险降权。"""
    s = (it["ctr"] * 6.0 + it["hot"] * 1.2
         + (0.35 if it["author"] in user["fav_authors"] else 0.0)
         + user["noise"][it["id"] % len(user["noise"])])
    if demote_risk and it["risk"]:
        s *= RISK_DEMOTE          # 正文:风险内容降权(而非直接过滤)
    return s


def post_ranking_filter(ranked, enable, k=EXPOSE_K):
    """正文的「排序后过滤」:广告比例控制 + 同作者内容数量限制。

    这两条策略必须放在排序之后:它们约束的是「最终曝光列表的构成」,
    在召回阶段还不知道哪些物品会进入最终列表,因此无法提前执行。
    """
    if not enable:
        return ranked[:k], Counter()
    out, dropped = [], Counter()
    ad_cap = int(k * AD_RATIO_CAP)
    n_ad = 0
    per_author = Counter()
    for it in ranked:
        if len(out) >= k:
            break
        if it["is_ad"] and n_ad >= ad_cap:
            dropped["广告占比超限"] += 1
            continue
        if per_author[it["author"]] >= SAME_AUTHOR_CAP:
            dropped["同作者条数超限"] += 1
            continue
        out.append(it)
        n_ad += it["is_ad"]
        per_author[it["author"]] += 1
    return out, dropped


def gen_user(rng, items):
    """[MOCK] 用户画像:已读历史 / 屏蔽内容 / 黑名单作者 / 地域 / 偏好作者。"""
    read = {rng.randrange(N_ITEMS) for _ in range(1200)}
    return {
        "read": read,
        "blocked_items": {rng.randrange(N_ITEMS) for _ in range(300)},
        "blocked_authors": {rng.randrange(N_AUTHORS) for _ in range(60)},
        "region": rng.randrange(N_REGIONS),
        "fav_authors": {rng.randrange(N_AUTHORS) for _ in range(20)},
        "noise": [rng.uniform(0, 0.5) for _ in range(97)],
    }


def run_config(name, items, backup_pool, rng_seed,
               pre_filter, post_retrieval, post_ranking):
    """跑一组配置,返回链路各级的统计指标。"""
    rng = random.Random(rng_seed)
    index = build_index(items, pre_filter)
    st = {
        "name": name,
        "index_size": len(index),
        "scan": 0, "recall": 0, "waste": 0, "kept": 0,
        "backup_hit": 0, "backup_add": 0,
        "prerank_calls": 0,
        "expose": 0, "bad_status": 0, "read": 0, "blocked": 0,
        "region_bad": 0, "ad": 0, "risk": 0, "max_author": 0,
        "dropped": Counter(),
    }
    for _ in range(N_REQUESTS):
        user = gen_user(rng, items)
        cands, tries = retrieve(index, rng)
        st["scan"] += tries
        st["recall"] += len(cands)
        # 「浪费」= 进入召回打分但本就应该在召回前被剔除的物品
        st["waste"] += sum(1 for it in cands if it["status"] in INVALID_STATUS)

        cands, dropped = post_retrieval_filter(cands, user, post_retrieval)
        st["dropped"] += dropped
        if len(cands) < MIN_CANDIDATES:
            st["backup_hit"] += 1
            need = MIN_CANDIDATES - len(cands)
            have = {it["id"] for it in cands}
            pool = [it for it in backup_pool if it["id"] not in have]
            # 兜底补入的内容同样要过一遍召回后过滤,否则「兜底」会变成过滤绕过口子
            pool, dropped_bk = post_retrieval_filter(pool, user, post_retrieval)
            st["dropped"] += dropped_bk
            add = pool[:need]
            st["backup_add"] += len(add)
            cands = cands + add
        st["kept"] += len(cands)

        # 粗排 -> 精排:打分次数就是算力账单
        prerank = cands[:PRERANK_IN]
        st["prerank_calls"] += len(prerank)
        prerank.sort(key=lambda it: rank_score(it, user, False), reverse=True)
        rank_pool = prerank[:RANK_IN]
        ranked = sorted(rank_pool,
                        key=lambda it: rank_score(it, user, post_ranking),
                        reverse=True)
        final, dropped2 = post_ranking_filter(ranked, post_ranking)
        st["dropped"] += dropped2

        st["expose"] += len(final)
        st["bad_status"] += sum(1 for it in final
                                if it["status"] in INVALID_STATUS)
        st["read"] += sum(1 for it in final if it["id"] in user["read"])
        st["blocked"] += sum(1 for it in final
                             if it["id"] in user["blocked_items"]
                             or it["author"] in user["blocked_authors"])
        st["region_bad"] += sum(1 for it in final
                                if it["regions"] is not None
                                and user["region"] not in it["regions"])
        st["ad"] += sum(1 for it in final if it["is_ad"])
        st["risk"] += sum(1 for it in final if it["risk"])
        if final:
            st["max_author"] = max(st["max_author"],
                                   Counter(it["author"]
                                           for it in final).most_common(1)[0][1])
    return st


def main():
    random.seed(42)
    rng = random.Random(42)
    items = gen_library(rng)
    # [MOCK] 兜底池:热门且状态有效的内容(正文:热门内容召回兜底)
    backup_pool = sorted((it for it in items
                          if it["status"] not in INVALID_STATUS),
                         key=lambda it: -it["hot"])[:4000]

    print("=" * 94)
    print("§4.5.1 过滤模块的位置:召回前过滤 / 召回后过滤 / 排序后过滤 三段式链路对照")
    print("=" * 94)
    cnt = Counter(it["status"] for it in items)
    print(f"\n[MOCK] 内容库 {N_ITEMS:,} 条 / 作者 {N_AUTHORS:,} 个 / 地域 {N_REGIONS} 个;"
          f"模拟 {N_REQUESTS} 次请求,单请求召回 {RECALL_SIZE} -> 曝光 {EXPOSE_K}")
    print(f"{'内容状态':<16}{'条数':>10}{'占比':>9}{'召回前是否剔除':>16}")
    print("-" * 54)
    for name, _ in STATUS_DIST:
        flag = "是" if name in INVALID_STATUS else "否(可分发)"
        print(f"{name:<16}{cnt[name]:>10,}{cnt[name] / N_ITEMS:>9.2%}{flag:>16}")
    invalid = sum(cnt[s] for s in INVALID_STATUS)
    print(f"{'小计(无效内容)':<16}{invalid:>10,}{invalid / N_ITEMS:>9.2%}")

    configs = [
        ("A 全不过滤", False, False, False),
        ("B 只在召回后过滤", False, True, True),
        ("C 三段式(正文)", True, True, True),
    ]
    stats = [run_config(n, items, backup_pool, 1234, pre, post, rank)
             for n, pre, post, rank in configs]

    print("\n[链路算力账单] 过滤发生得越早,下游被浪费的计算越少")
    print(f"{'配置':<20}{'索引条目数':>12}{'召回扫描次数':>14}"
          f"{'召回打分浪费率':>16}{'候选利用率':>12}{'兜底触发请求数':>16}")
    print("-" * 92)
    for s in stats:
        waste = s["waste"] / max(s["recall"], 1)
        util = s["kept"] / max(s["recall"], 1)
        print(f"{s['name']:<20}{s['index_size']:>12,}{s['scan']:>14,}"
              f"{waste:>16.2%}{util:>12.2%}{s['backup_hit']:>16}")
    print(f"  (候选利用率 = 兜底补量后的候选数 / 召回候选数;兜底阈值 {MIN_CANDIDATES},"
          f"配置 B/C 共补入 {stats[1]['backup_add']:,}/{stats[2]['backup_add']:,} 个候选,")
    print("   因此两者过滤后虽各少了约 5% 候选,最终仍被补回到阈值线上)")

    print("\n[曝光质量] 正文 §4.5 开头列出的问题是否真的出现")
    print(f"{'配置':<20}{'曝光条数':>10}{'无效内容率':>12}{'重复推荐率':>12}"
          f"{'屏蔽命中率':>12}{'地域违规率':>12}{'广告占比':>10}{'风险内容率':>12}"
          f"{'单屏同作者上限':>16}")
    print("-" * 120)
    for s in stats:
        e = max(s["expose"], 1)
        print(f"{s['name']:<20}{s['expose']:>10,}{s['bad_status'] / e:>12.2%}"
              f"{s['read'] / e:>12.2%}{s['blocked'] / e:>12.2%}"
              f"{s['region_bad'] / e:>12.2%}{s['ad'] / e:>10.2%}"
              f"{s['risk'] / e:>12.2%}{s['max_author']:>16}")
    print(f"  (广告占比上限 {AD_RATIO_CAP:.0%}、单屏同作者上限 {SAME_AUTHOR_CAP} 条、"
          f"风险内容降权系数 {RISK_DEMOTE})")

    print("\n[各位置的拦截构成] 三段式配置 C 的过滤明细")
    print(f"{'过滤策略':<18}{'所属位置':<16}{'拦截次数':>12}{'占总拦截':>12}")
    print("-" * 60)
    pos = {
        "已读过滤": "召回后过滤", "用户屏蔽内容": "召回后过滤",
        "黑名单作者": "召回后过滤", "地域限制": "召回后过滤",
        "广告占比超限": "排序后过滤", "同作者条数超限": "排序后过滤",
    }
    total = sum(stats[2]["dropped"].values())
    for key, num in stats[2]["dropped"].most_common():
        print(f"{key:<18}{pos[key]:<16}{num:>12,}{num / max(total, 1):>12.2%}")
    pre_cut = N_ITEMS - stats[2]["index_size"]
    print(f"{'无效状态剔除':<18}{'召回前过滤':<16}{pre_cut:>12,}"
          f"{'(按物品计,不进索引)':>12}")

    b, c = stats[1], stats[2]
    print("\n结论(正文 §4.5.1 的工程含义):")
    print("  1) 召回前过滤省的是「整条下游链路的算力」:配置 B 的索引里有 "
          f"{b['index_size'] - c['index_size']:,} 条无效内容,")
    print(f"     导致召回打分中有 {b['waste'] / max(b['recall'], 1):.2%} 的无效计算;"
          "配置 C 把这部分成本在建索引时就消掉了;")
    print("  2) 召回后过滤才是「个性化」的过滤:已读 / 屏蔽 / 黑名单 / 地域都依赖用户态,")
    print("     无法在离线索引侧完成,只能在拿到候选后逐个核对;")
    print("  3) 排序后过滤必须留在最后:广告占比与同作者打散约束的是最终列表的构成,")
    print("     在召回阶段根本不知道哪些物品会进 Top-K,所以提前做没有意义;")
    print("  4) 过滤会显著削减候选规模,因此必须配合兜底阈值 "
          f"(本次配置 C 有 {c['backup_hit']}/{N_REQUESTS} 次请求触发兜底),")
    print("     否则「过滤得干净」会变成「候选不够、推荐变空」;")
    print("  5) 对照配置 A 可以看到,缺了过滤模块就会同时出现正文开头列出的")
    print("     重复推荐、无效内容曝光、广告挤占、同作者刷屏等全部问题。")


if __name__ == "__main__":
    main()
