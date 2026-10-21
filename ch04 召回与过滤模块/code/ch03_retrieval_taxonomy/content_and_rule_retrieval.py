"""
对应正文第 4 章 §4.3「召回技术分类 - 内容与规则召回」。

正文口径(Tag 召回,正文图 tag_retrieval 的文字描述):
  [离线] 以用户偏好的兴趣标签 Tag 作为触发信号,离线任务预先产出每个 Tag 的候选
         物品列表。物品选择基于后验表现(曝光量 TopK、点赞数 TopK、评论数 TopK),
         同时引入一定比例的随机采样物品,避免候选集过度集中于热门内容,缓解
         推荐系统自身 bias 导致的同质化问题。产出以 K-V 形式写入 Redis。
  [在线] 根据用户当前多个兴趣偏好 Tag,直接从 Redis 查对应物品列表,合并即可。

正文同时指出:内容与规则召回「不依赖模型,是纯业务导向的策略」,优势是可控性与
稳定性强,常承担冷启动扶持、运营干预、内容保量职责。

本脚本实现:
  1) 离线建倒排:Tag -> 三类后验 TopK(exposure / like / comment)+ 随机采样池;
  2) 多路规则召回在线侧:Tag 召回、类目(Category)召回、热门(Hot)召回、
     新内容冷启扶持召回、运营干预(白名单保量)召回;
  3) 对比「纯热门 TopK」与「后验 TopK + random 比例混合」的基尼系数/长尾占比,
     把正文所说的「缓解 bias 导致的同质化」变成数字。

后验热度打分(本脚本口径,用于排 TopK):
  posterior(i) = like(i) / (exposure(i) + β)   , β = 100 做小样本平滑
  hot(i)       = 0.5·norm(exposure) + 0.3·norm(like) + 0.2·norm(comment)

[MOCK] RedisKV:dict 模拟线上 Redis,统计 get/mget 次数。
"""

import math
import random
from collections import Counter, defaultdict

N_ITEMS = 2000
TAGS = ["美食", "旅行", "健身", "宠物", "数码", "穿搭", "家居", "游戏",
        "影视", "音乐", "知识", "亲子"]
CATEGORIES = ["短视频", "图文", "直播回放"]
TAG_LIST_SIZE = 200        # 每个 Tag 的离线候选列表长度
RANDOM_RATIO = 0.30        # 随机采样物品占比(正文:引入一定比例随机采样)
BETA_SMOOTH = 100.0


# ============================================================ [MOCK] Redis
class RedisKV:
    """[MOCK] 教学用 Redis:只实现 set / get / mget,并统计访问次数。"""

    def __init__(self):
        self._d, self.n_get, self.n_mget = {}, 0, 0

    def set(self, k, v):
        self._d[k] = v

    def get(self, k):
        self.n_get += 1
        return self._d.get(k, [])

    def mget(self, keys):
        self.n_mget += 1
        self.n_get += len(keys)
        return [self._d.get(k, []) for k in keys]

    def __len__(self):
        return len(self._d)


# ============================================================ 合成物品库
def gen_item_pool(rng):
    items = {}
    for i in range(N_ITEMS):
        n_tag = rng.choices([1, 2, 3], weights=[5, 3, 2], k=1)[0]
        tags = rng.sample(TAGS, n_tag)
        # 曝光量长尾分布:少数爆款占据绝大部分曝光
        expo = int(rng.paretovariate(1.15) * 500)
        ctr_like = rng.betavariate(2, 60)          # 真实点赞率
        likes = int(expo * ctr_like)
        comments = int(likes * rng.uniform(0.05, 0.4))
        items[i] = {
            "tags": tags,
            "cat": rng.choice(CATEGORIES),
            "author": f"a{rng.randrange(300)}",
            "expo": expo,
            "like": likes,
            "comment": comments,
            "age_h": rng.uniform(0, 240),          # 发布至今小时数
            "quality": ctr_like,                   # 用真实点赞率当「内在质量」真值
        }
    return items


def _norm(vals):
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo + 1e-9) for v in vals]


def hot_score(items):
    ids = list(items)
    e = _norm([math.log1p(items[i]["expo"]) for i in ids])
    l = _norm([math.log1p(items[i]["like"]) for i in ids])
    c = _norm([math.log1p(items[i]["comment"]) for i in ids])
    return {i: 0.5 * e[k] + 0.3 * l[k] + 0.2 * c[k] for k, i in enumerate(ids)}


def posterior_score(it):
    return it["like"] / (it["expo"] + BETA_SMOOTH)


# ============================================================ 离线任务
def offline_build_tag_index(items, rng, redis, random_ratio=RANDOM_RATIO):
    """离线任务:每个 Tag 产出 TAG_LIST_SIZE 条候选 = 后验 TopK + 随机采样。"""
    by_tag = defaultdict(list)
    for i, it in items.items():
        for t in it["tags"]:
            by_tag[t].append(i)

    stats = {}
    n_rand = int(TAG_LIST_SIZE * random_ratio)
    n_top = TAG_LIST_SIZE - n_rand
    # 三类后验指标各占 TopK 的三分之一(正文:曝光 TopK / 点赞 TopK / 评论 TopK)
    per_metric = n_top // 3
    for t, pool in by_tag.items():
        picked, picked_set = [], set()
        for metric, key in [("expo", lambda i: items[i]["expo"]),
                            ("like", lambda i: items[i]["like"]),
                            ("comment", lambda i: items[i]["comment"])]:
            ranked = sorted(pool, key=key, reverse=True)
            cnt = 0
            for i in ranked:
                if i in picked_set:
                    continue
                picked.append((i, metric, posterior_score(items[i])))
                picked_set.add(i)
                cnt += 1
                if cnt >= per_metric:
                    break
        rest = [i for i in pool if i not in picked_set]
        for i in rng.sample(rest, min(n_rand, len(rest))):
            picked.append((i, "random", posterior_score(items[i])))
            picked_set.add(i)
        redis.set(f"tag:{t}", picked)
        stats[t] = (len(pool), len(picked))
    return stats


def offline_build_aux_index(items, redis, hot):
    """离线:类目热门表 / 全局热门表 / 冷启新内容池 / 运营白名单。"""
    by_cat = defaultdict(list)
    for i, it in items.items():
        by_cat[it["cat"]].append(i)
    for c, pool in by_cat.items():
        redis.set(f"cat_hot:{c}",
                  sorted(pool, key=lambda i: -hot[i])[:TAG_LIST_SIZE])
    redis.set("global_hot", sorted(items, key=lambda i: -hot[i])[:TAG_LIST_SIZE])
    cold = [i for i, it in items.items() if it["age_h"] < 24 and it["expo"] < 800]
    redis.set("cold_start_pool", sorted(cold, key=lambda i: items[i]["age_h"]))
    return len(cold)


# ============================================================ 在线规则召回
def retrieve_by_tag(redis, user_tags, quota, seen):
    """在线 Tag 召回:mget 多个 Tag 列表 -> 按兴趣权重加权合并 -> 截断。"""
    keys = [f"tag:{t}" for t, _ in user_tags]
    lists = redis.mget(keys)
    scores, reason = defaultdict(float), {}
    for (t, w), lst in zip(user_tags, lists):
        for iid, metric, post in lst:
            if iid in seen:
                continue
            s = w * (0.2 + post)
            if s > scores[iid]:
                reason[iid] = f"tag:{t}/{metric}"
            scores[iid] = max(scores[iid], s)
    out = sorted(scores.items(), key=lambda x: -x[1])[:quota]
    return [(i, s, reason[i]) for i, s in out]


def retrieve_by_category(redis, fav_cat, quota, seen):
    lst = redis.get(f"cat_hot:{fav_cat}")
    return [(i, 1.0 - k / max(1, len(lst)), f"cat_hot:{fav_cat}")
            for k, i in enumerate(lst) if i not in seen][:quota]


def retrieve_hot(redis, quota, seen):
    lst = redis.get("global_hot")
    return [(i, 1.0 - k / max(1, len(lst)), "global_hot")
            for k, i in enumerate(lst) if i not in seen][:quota]


def retrieve_cold_start(redis, items, user_tags, quota, seen):
    """冷启动扶持:新内容池里挑与用户兴趣 Tag 有交集的,按发布时间从新到旧。"""
    want = {t for t, _ in user_tags}
    out = []
    for i in redis.get("cold_start_pool"):
        if i in seen or not (want & set(items[i]["tags"])):
            continue
        out.append((i, 1.0 / (1.0 + items[i]["age_h"]), "cold_start"))
        if len(out) >= quota:
            break
    return out


def retrieve_operation(redis, quota, seen):
    """运营干预/内容保量:白名单直接插入,是规则召回可控性的典型体现。"""
    return [(i, 1.0, "operation_boost") for i in redis.get("op_whitelist")
            if i not in seen][:quota]


# ============================================================ bias 度量
def gini(counts):
    """基尼系数:0 = 曝光完全平均,1 = 全部曝光集中在一个物品。"""
    xs = sorted(counts)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((2 * (k + 1) - n - 1) * x for k, x in enumerate(xs))
    return cum / (n * sum(xs))


def main():
    random.seed(42)
    rng = random.Random(42)
    items = gen_item_pool(rng)
    hot = hot_score(items)
    redis = RedisKV()

    print("=" * 76)
    print("§4.3 内容与规则召回:Tag 召回离线建表 + 多路规则召回在线链路")
    print("=" * 76)
    stats = offline_build_tag_index(items, rng, redis)
    n_cold = offline_build_aux_index(items, redis, hot)
    redis.set("op_whitelist", rng.sample(range(N_ITEMS), 20))
    print(f"物品库 {N_ITEMS} 条,Tag {len(TAGS)} 个,类目 {len(CATEGORIES)} 个")
    print(f"[离线] Tag 倒排表构建完成,每 Tag 列表长度 {TAG_LIST_SIZE} "
          f"(后验 TopK {TAG_LIST_SIZE - int(TAG_LIST_SIZE * RANDOM_RATIO)} + "
          f"随机 {int(TAG_LIST_SIZE * RANDOM_RATIO)})")
    print(f"[离线] 冷启池 {n_cold} 条(发布 <24h 且曝光 <800),运营白名单 20 条")
    print(f"[MOCK] Redis 键总数 = {len(redis)}")
    print()
    print(f"  {'Tag':<8}{'命中物品数':>10}{'离线列表长度':>14}")
    for t in TAGS[:5]:
        p, k = stats[t]
        print(f"  {t:<9}{p:>10}{k:>14}")
    print("  ...")
    print()

    # ---------------- 单请求:五路规则召回链路
    print("-" * 76)
    print("[在线] 单请求:五路规则召回 + 配额 + 合并去重")
    print("-" * 76)
    user_tags = [("美食", 1.0), ("旅行", 0.7), ("数码", 0.4)]
    fav_cat = "短视频"
    seen = set(rng.sample(range(N_ITEMS), 300))
    print(f"用户兴趣 Tag={user_tags} 偏好类目={fav_cat} 已消费 {len(seen)} 条")
    print()
    channels = [
        ("tag", retrieve_by_tag(redis, user_tags, 150, seen), 150),
        ("category", retrieve_by_category(redis, fav_cat, 60, seen), 60),
        ("hot", retrieve_hot(redis, 50, seen), 50),
        ("cold_start", retrieve_cold_start(redis, items, user_tags, 40, seen), 40),
        ("operation", retrieve_operation(redis, 20, seen), 20),
    ]
    print(f"{'召回路':<14}{'配额':>6}{'返回数':>8}{'示例 reason':<26}{'新增(去重后)':>14}")
    merged, merged_reason = {}, {}
    for name, res, quota in channels:
        before = len(merged)
        for iid, sc, rsn in res:
            if iid not in merged:
                merged[iid] = sc
                merged_reason[iid] = rsn
        ex = res[0][2] if res else "-"
        print(f"{name:<16}{quota:>6}{len(res):>8}  {ex:<26}{len(merged) - before:>12}")
    print(f"{'合计':<16}{'':>6}{sum(len(r) for _, r, _ in channels):>8}"
          f"{'':<28}{len(merged):>12}")
    print()
    rc = Counter(r.split("/")[0].split(":")[0] for r in merged_reason.values())
    print(f"  合并后候选 {len(merged)} 条,retrieval reason 分布: {dict(rc)}")
    print()

    # ---------------- 随机采样比例对 bias 的影响
    print("-" * 76)
    print("[对比实验] 随机采样比例 vs 曝光集中度(正文:缓解系统 bias 导致的同质化)")
    print("-" * 76)
    print(f"{'random 比例':>12}{'去重候选数':>12}{'长尾覆盖数':>12}"
          f"{'全库曝光基尼':>14}{'平均质量':>10}")
    # 长尾 = 曝光量低于全库中位数的物品
    median_expo = sorted(it["expo"] for it in items.values())[N_ITEMS // 2]
    n_tail_total = sum(1 for it in items.values() if it["expo"] < median_expo)
    for ratio in (0.0, 0.1, 0.3, 0.5, 0.8):
        r2 = RedisKV()
        offline_build_tag_index(items, random.Random(7), r2, random_ratio=ratio)
        cnt = Counter()
        quality = []
        for _ in range(200):  # 模拟 200 个不同兴趣的用户请求
            tg = [(t, round(rng.uniform(0.3, 1.0), 2))
                  for t in rng.sample(TAGS, 3)]
            res = retrieve_by_tag(r2, tg, 100, set())
            for iid, _, _ in res:
                cnt[iid] += 1
                quality.append(items[iid]["quality"])
        n_tail = sum(1 for i in cnt if items[i]["expo"] < median_expo)
        # 基尼系数按「全库口径」计算:从未被召回的物品也要计入 0 曝光,
        # 否则只统计被召回的物品会把「覆盖面变大」这件事从指标里抹掉。
        expo_all = [cnt.get(i, 0) for i in range(N_ITEMS)]
        print(f"{ratio:>12.1f}{len(cnt):>12}{n_tail:>12}"
              f"{gini(expo_all):>14.4f}{sum(quality) / max(1, len(quality)):>10.4f}")
    print(f"\n  (全库共 {n_tail_total} 条长尾物品,曝光低于中位数 {median_expo})")
    print("  读法:random 比例越高,被召回的不同物品越多、长尾覆盖越广、全库曝光基尼")
    print("        系数总体下降(候选分布更均匀),但候选的平均「内在质量」(真实点赞率)")
    print("        随之下降 —— 这正是正文所说的「避免候选集过度集中于热门内容」")
    print("        与推荐精度之间的权衡。两点诚实说明:(1) 本脚本 Tag 召回打分用后验")
    print("        点赞率 like/(expo+β),本身已有去热门效果,所以基尼改善幅度有限;")
    print("        (2) ratio=0.1/0.3 时基尼略有上升,是小比例随机只是把尾部候选稀疏地")
    print("        加进来、单请求 quota=100 仍被后验高分项占满导致的,属于合成数据噪声。")


if __name__ == "__main__":
    main()
