"""
对应正文第 4 章 §4.3「召回技术分类 - 协同过滤召回」。

正文口径:协同过滤完全依赖用户行为共现关系,不关心物品语义;按建模对象分为
User-CF(U2U2I 路径)、Item-CF(I2I 路径)、Swing(Item-CF 的工业级改进)。

本脚本聚焦的是「召回链路」而不是模型本身(算法原理见第 10 章 ch10 召回模型/code/
ch01_collaborative_filtering/)。这里实现的是工业界 CF 召回的两段式工程形态:

  [离线] 全量行为日志 -> 倒排索引(item -> 用户集 / user -> 物品集)
                      -> 相似度计算 -> 截断成 TopN 相似表 -> 写入 [MOCK] KV 存储
  [在线] 用户 trigger(最近 N 个正反馈物品) -> 查 KV 拿每个 trigger 的 TopN
                      -> 加权聚合(trigger 权重 × 相似度) -> 过滤已交互 -> 截断

三种相似度口径(与正文一致):
  Item-CF  w(i,j) = |U_i ∩ U_j| / sqrt(|U_i| · |U_j|)
  User-CF  w(u,v) = |I_u ∩ I_v| / sqrt(|I_u| · |I_v|)
  Swing    w(i,j) = Σ_{u,v ∈ U_i∩U_j} 1 / (α + |I_u ∩ I_v|)
           (Swing 原始论文:Yang et al., "Large Scale Product Graph Construction
            for Recommendation in E-commerce", arXiv:2010.05525)

在线聚合打分(三种召回路共用同一套链路代码):
  score(j) = Σ_{i ∈ trigger} w_trigger(i) · sim(i, j)
其中 w_trigger(i) 为行为权重 × 时间衰减 0.5^(Δt / halflife)。

[MOCK] KVStore:用 dict 模拟线上的 Redis / 相似度表存储,带调用计数。
"""

import math
import random
import time
from collections import defaultdict

N_USERS = 400
N_ITEMS = 300
N_GROUPS = 6          # 潜在兴趣群组,用来造出有共现结构的行为日志
TOPN_SIM = 30         # 离线相似表每个 key 保留的条数
ALPHA_SWING = 1.0     # Swing 平滑系数
HALF_LIFE_H = 24.0    # 行为时间衰减半衰期(小时)

BEHAVIOR_WEIGHT = {"click": 1.0, "like": 2.0, "collect": 3.0}


# ============================================================ [MOCK] 存储
class KVStore:
    """[MOCK] 模拟线上存放离线相似表的 KV 存储(真实系统里是 Redis / 自研 KV)。"""

    def __init__(self, name):
        self.name = name
        self._d = {}
        self.calls = 0

    def mset(self, mapping):
        self._d.update(mapping)

    def get(self, key, default=None):
        self.calls += 1
        return self._d.get(key, default)

    def __len__(self):
        return len(self._d)


# ============================================================ 合成行为日志
def gen_behavior_log(rng):
    """造一份有群组结构的行为日志:同组用户偏好同一批物品,于是存在真实共现。"""
    user_group = {u: rng.randrange(N_GROUPS) for u in range(N_USERS)}
    item_group = {i: rng.randrange(N_GROUPS) for i in range(N_ITEMS)}
    # 物品热度服从长尾,用来检验相似度公式对热门物品的抑制能力
    hot = [rng.paretovariate(1.3) for _ in range(N_ITEMS)]

    log = []  # (user, item, behavior, hours_ago)
    for u in range(N_USERS):
        g = user_group[u]
        n_act = rng.randint(8, 30)
        for _ in range(n_act):
            if rng.random() < 0.75:
                pool = [i for i in range(N_ITEMS) if item_group[i] == g]
            else:
                pool = list(range(N_ITEMS))
            weights = [hot[i] for i in pool]
            item = rng.choices(pool, weights=weights, k=1)[0]
            beh = rng.choices(list(BEHAVIOR_WEIGHT), weights=[6, 3, 1], k=1)[0]
            log.append((u, item, beh, rng.uniform(0, 72)))
    return log, user_group, item_group


def build_inverted_index(log):
    u2i, i2u = defaultdict(dict), defaultdict(dict)
    for u, i, beh, hrs in log:
        w = BEHAVIOR_WEIGHT[beh]
        if w > u2i[u].get(i, 0.0):
            u2i[u][i] = w
            i2u[i][u] = w
    return u2i, i2u


# ============================================================ 离线:三种相似度
def offline_itemcf(i2u):
    co = defaultdict(lambda: defaultdict(float))
    # 以用户为中心扫描,复杂度 Σ_u |I_u|^2,这是工业界 MapReduce 的标准做法
    u2i = defaultdict(list)
    for i, users in i2u.items():
        for u in users:
            u2i[u].append(i)
    for u, items in u2i.items():
        for a in items:
            for b in items:
                if a != b:
                    co[a][b] += 1.0
    sim = {}
    for a, row in co.items():
        scored = [(b, c / math.sqrt(len(i2u[a]) * len(i2u[b]))) for b, c in row.items()]
        scored.sort(key=lambda x: -x[1])
        sim[a] = scored[:TOPN_SIM]
    return sim


def offline_usercf(u2i):
    i2u = defaultdict(list)
    for u, items in u2i.items():
        for i in items:
            i2u[i].append(u)
    co = defaultdict(lambda: defaultdict(float))
    for i, users in i2u.items():
        for a in users:
            for b in users:
                if a != b:
                    co[a][b] += 1.0
    sim = {}
    for a, row in co.items():
        scored = [(b, c / math.sqrt(len(u2i[a]) * len(u2i[b]))) for b, c in row.items()]
        scored.sort(key=lambda x: -x[1])
        sim[a] = scored[:TOPN_SIM]
    return sim


def offline_swing(u2i, i2u, alpha=ALPHA_SWING):
    """Swing:对 item 对的每一组共同用户 (u, v),按其共同物品数做惩罚。"""
    sets = {u: set(items) for u, items in u2i.items()}
    u2i_list = {u: list(items) for u, items in u2i.items()}
    # 先找出所有有共现的 item 对,避免 O(N_ITEMS^2) 全枚举
    pair_users = defaultdict(list)
    for u, items in u2i_list.items():
        for a in items:
            for b in items:
                if a < b:
                    pair_users[(a, b)].append(u)
    sim_raw = defaultdict(lambda: defaultdict(float))
    for (a, b), users in pair_users.items():
        if len(users) < 2:
            continue
        s = 0.0
        for x in range(len(users)):
            for y in range(x + 1, len(users)):
                inter = len(sets[users[x]] & sets[users[y]])
                s += 1.0 / (alpha + inter)
        sim_raw[a][b] += s
        sim_raw[b][a] += s
    sim = {}
    for a, row in sim_raw.items():
        scored = sorted(row.items(), key=lambda x: -x[1])
        sim[a] = scored[:TOPN_SIM]
    return sim


# ============================================================ 在线召回链路
def build_trigger(u2i_log, uid, max_trigger=5):
    """trigger = 最近若干个互不重复的正反馈物品,权重 = 行为权重 × 时间衰减。"""
    trig, used = [], set()
    for item, hrs, beh in sorted(u2i_log[uid], key=lambda x: x[1]):  # hours_ago 升序
        if item in used:
            continue
        used.add(item)
        trig.append((item, BEHAVIOR_WEIGHT[beh] * 0.5 ** (hrs / HALF_LIFE_H)))
        if len(trig) >= max_trigger:
            break
    return trig


def online_i2i_retrieve(kv, triggers, seen, topk):
    """I2I 在线链路:逐 trigger 查相似表 -> 加权聚合 -> 过滤已消费 -> 截断。"""
    scores, reason = defaultdict(float), {}
    for item, w in triggers:
        for cand, s in kv.get(f"i2i:{item}", []):
            if cand in seen:
                continue
            scores[cand] += w * s
            if cand not in reason or w * s > reason[cand][1]:
                reason[cand] = (item, w * s)
    out = sorted(scores.items(), key=lambda x: -x[1])[:topk]
    return [(i, s, reason[i][0]) for i, s in out]


def online_u2u2i_retrieve(kv_user, u2i, uid, seen, topk, topn_user=20):
    """U2U2I 在线链路:先查相似用户,再取其正反馈物品加权聚合。"""
    scores, reason = defaultdict(float), {}
    for v, s in kv_user.get(f"u2u:{uid}", [])[:topn_user]:
        for item, w in u2i.get(v, {}).items():
            if item in seen:
                continue
            scores[item] += s * w
            if item not in reason or s * w > reason[item][1]:
                reason[item] = (v, s * w)
    out = sorted(scores.items(), key=lambda x: -x[1])[:topk]
    return [(i, sc, reason[i][0]) for i, sc in out]


# ============================================================ 评估
def hit_rate(recs, truth):
    if not recs:
        return 0.0
    return len([i for i, *_ in recs if i in truth]) / len(recs)


def main():
    random.seed(42)
    rng = random.Random(42)

    log, user_group, item_group = gen_behavior_log(rng)
    # 留出法:每个用户最近 20% 行为作为「未来兴趣」真值,其余做训练
    by_user = defaultdict(list)
    for u, i, beh, hrs in log:
        by_user[u].append((i, hrs, beh))
    train, test = [], defaultdict(set)
    for u, acts in by_user.items():
        acts.sort(key=lambda x: x[1])  # hours_ago 升序 = 时间倒序
        k = max(1, len(acts) // 5)
        for i, hrs, beh in acts[:k]:
            test[u].add(i)
        for i, hrs, beh in acts[k:]:
            train.append((u, i, beh, hrs))

    u2i, i2u = build_inverted_index(train)
    train_by_user = defaultdict(list)
    for u, i, beh, hrs in train:
        train_by_user[u].append((i, hrs, beh))

    print("=" * 76)
    print("§4.3 协同过滤召回:离线相似表构建 + 在线 I2I / U2U2I 召回链路")
    print("=" * 76)
    print(f"合成日志: users={N_USERS} items={N_ITEMS} 行为数={len(log)} "
          f"(训练 {len(train)} / 评测 {sum(len(v) for v in test.values())})")
    print()

    print("-" * 76)
    print("[离线阶段] 三种相似度口径各自建表并截断到 TopN =", TOPN_SIM)
    print("-" * 76)
    tables, costs = {}, {}
    for name, fn in [("Item-CF", lambda: offline_itemcf(i2u)),
                     ("Swing", lambda: offline_swing(u2i, i2u)),
                     ("User-CF", lambda: offline_usercf(u2i))]:
        t0 = time.perf_counter()
        tables[name] = fn()
        costs[name] = (time.perf_counter() - t0) * 1000
    print(f"{'相似表':<10}{'key 数':>8}{'平均每 key 条数':>16}{'离线构建耗时(ms)':>18}")
    for name in ("Item-CF", "Swing", "User-CF"):
        tb = tables[name]
        avg = sum(len(v) for v in tb.values()) / max(1, len(tb))
        print(f"{name:<12}{len(tb):>8}{avg:>16.1f}{costs[name]:>18.1f}")
    print()

    # ---- 写入 [MOCK] KV
    kv_i2i = KVStore("i2i")
    kv_i2i.mset({f"i2i:{k}": v for k, v in tables["Item-CF"].items()})
    kv_swing = KVStore("swing")
    kv_swing.mset({f"i2i:{k}": v for k, v in tables["Swing"].items()})
    kv_u2u = KVStore("u2u")
    kv_u2u.mset({f"u2u:{k}": v for k, v in tables["User-CF"].items()})
    print(f"[MOCK] 写入 KV: i2i={len(kv_i2i)} 键 / swing={len(kv_swing)} 键 / u2u={len(kv_u2u)} 键")
    print()

    # ---- 单用户链路 trace
    demo_u = max(train_by_user, key=lambda u: len(train_by_user[u]))
    trig = build_trigger(train_by_user, demo_u)
    seen = set(u2i[demo_u])
    print("-" * 76)
    print(f"[在线阶段] 单请求 trace, user={demo_u} 已消费 {len(seen)} 个物品")
    print("-" * 76)
    print("  trigger(最近正反馈 + 时间衰减权重):")
    for it, w in trig:
        print(f"    item={it:<4} group={item_group[it]}  trigger_weight={w:.4f}")
    rec = online_i2i_retrieve(kv_i2i, trig, seen, topk=8)
    print(f"  Item-CF I2I 召回 Top8 (KV 访问 {kv_i2i.calls} 次):")
    for it, sc, src in rec:
        tag = "命中真值" if it in test[demo_u] else ""
        print(f"    item={it:<4} group={item_group[it]} score={sc:.4f} "
              f"reason=i2i<-{src} {tag}")
    print()

    # ---- 三路召回的整体效果对比
    print("-" * 76)
    print("[批量评测] 对全部用户各召回 Top20,统计命中率 / 覆盖率 / 平均在线耗时")
    print("-" * 76)
    print(f"{'召回路':<14}{'路径':<10}{'HitRate@20':>12}{'物品覆盖率':>12}"
          f"{'空结果率':>10}{'平均耗时(ms)':>14}")
    rows = []
    for name, kind in [("Item-CF", "i2i"), ("Swing", "i2i"), ("User-CF", "u2u2i")]:
        kv = {"Item-CF": kv_i2i, "Swing": kv_swing, "User-CF": kv_u2u}[name]
        kv.calls = 0
        hits, covered, empty, t0 = [], set(), 0, time.perf_counter()
        for u in range(N_USERS):
            if not train_by_user[u]:
                continue
            s = set(u2i[u])
            if kind == "i2i":
                r = online_i2i_retrieve(kv, build_trigger(train_by_user, u), s, 20)
            else:
                r = online_u2u2i_retrieve(kv, u2i, u, s, 20)
            if not r:
                empty += 1
            hits.append(hit_rate(r, test[u]))
            covered.update(i for i, *_ in r)
        dt = (time.perf_counter() - t0) * 1000 / max(1, len(hits))
        path = "I2I" if kind == "i2i" else "U2U2I"
        rows.append((name, path, sum(hits) / len(hits), len(covered) / N_ITEMS,
                     empty / N_USERS, dt))
        print(f"{name:<16}{path:<10}{rows[-1][2]:>11.4f}{rows[-1][3]:>12.1%}"
              f"{rows[-1][4]:>10.1%}{dt:>14.3f}")
    print()
    cov = {r[0]: r[3] for r in rows}
    print("  说明:Item-CF 的 sqrt 归一化抑制热门;Swing 用「共同用户的共同物品数」进一步")
    print("        惩罚那些仅因为用户行为泛滥而共现的 item 对,相似表更锐利但也更集中 ——")
    print(f"        本次合成数据上 Swing 覆盖率 {cov['Swing']:.1%} 低于 Item-CF {cov['Item-CF']:.1%};")
    print("        User-CF 走 U2U2I,需要先查相似用户再展开物品,在线放大系数更大。")
    print()
    print("  注:相似度算法的完整推导与变体见第 10 章 ch10 召回模型/code/"
          "ch01_collaborative_filtering/,本脚本只负责链路形态。")


if __name__ == "__main__":
    main()
