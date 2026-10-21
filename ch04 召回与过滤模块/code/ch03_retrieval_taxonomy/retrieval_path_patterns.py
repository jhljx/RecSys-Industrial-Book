"""
对应正文第 4 章 §4.3「召回技术分类」开头关于**关系路径**的那一段。

正文口径(逐条对齐):
  正文指出「技术路线」与「关系路径」是两种不交叉的划分方式:前者关注召回策略
  用什么技术实现(协同过滤 / 内容规则 / 向量 / 图),后者关注召回结果沿着
  什么关系路径产生。正文列出的常见链路为:
    U2I   (User-to-Item)        基于用户兴趣直接找物品;
    I2I   (Item-to-Item)        基于已交互物品找相似物品;
    A2A   (Author-to-Author)    基于创作者关系找相似创作者;
    U2U2I (User-to-User-to-Item) 先找相似用户,再推荐相似用户喜欢的物品;
    U2I2I (User-to-Item-to-Item) 先找用户历史物品,再扩展到相似物品;
    I2U2U (Item-to-User-to-User) 先找与物品交互过的用户,再扩展到相似用户;
    I2I2U (Item-to-Item-to-User) 先找相似物品,再找与这些物品交互的用户。
  其中 U2U2I 与 U2I2I 是用户侧推荐最常见的两类;I2U2U 与 I2I2U 更多用于
  作者侧推荐(Creator Recommendation),为内容寻找潜在感兴趣用户。
  正文还强调:这些路径里的「相似用户」「相似物品」模块**往往复用同一套
  相似度计算服务**,因此它们体现的是信号传播路径的差异,而非底层技术差异。

本脚本做了什么:
  1) 用一份合成的 user-item 交互日志建一个 [MOCK] 统一相似度服务 SimilarityService,
     只提供三张表(u2u / i2i / a2a)与一次正排查询,所有 7 条路径都只调这一套服务;
  2) 把 7 条路径分别实现成「几跳 + 每跳查哪张表」的组合,打印每条路径的
     跳数、调用的相似表、扇出规模、候选量、在线耗时;
  3) 用户侧 4 条路径(U2I / I2I / U2U2I / U2I2I)做留出集命中率评测,
     作者侧 2 条路径(I2U2U / I2I2U)做「为物品找目标用户」的命中率评测,
     A2A 做作者维度的 Top-K 相似作者展示;
  4) 做一个「同一相似表被不同路径复用」的统计表,验证正文的那句话。

关键公式(与第 10 章的算法推导无关,这里只写路径展开的形式):
  一跳展开:  score(x) = Σ_{t ∈ Trigger} w(t) · sim(t, x)
  两跳展开:  score(z) = Σ_{t ∈ Trigger} Σ_{y ∈ TopK(t)} w(t) · sim(t, y) · sim(y, z)
  其中 w(t) 为 trigger 的时间衰减权重 w(t) = 0.9^rank(t)。
  扇出规模(fanout) = |Trigger| × K_1 × K_2,是两跳路径在线成本的主要来源。

[MOCK] 交互日志、相似表、正排索引均为合成数据与内存实现,不加载真实数据集,
也不起真实服务;相似度算法本身的推导与变体见第 10 章。
"""

import math
import random
import time
from collections import Counter, defaultdict

N_USERS = 300
N_ITEMS = 240
N_AUTHORS = 40
N_GROUPS = 4          # 兴趣簇:让合成数据具备可召回的结构
TOPN = 30             # 相似表截断
K1 = 10               # 第一跳取 TopK
K2 = 10               # 第二跳取 TopK
N_TRIGGER = 5         # trigger 个数
REC_K = 20            # 每条路径返回 Top-K


def pad(s, width, align="<"):
    """按「中日韩全角字符占 2 列」做等宽对齐,避免中文表头错位。"""
    s = str(s)
    w = sum(2 if ord(c) > 0x2E7F else 1 for c in s)
    fill = " " * max(width - w, 0)
    return fill + s if align == ">" else s + fill


# ======================================================== 合成数据 [MOCK]
def gen_log(rng):
    """[MOCK] 生成带兴趣簇结构的交互日志,并划分训练/留出集。"""
    user_group = [rng.randrange(N_GROUPS) for _ in range(N_USERS)]
    item_group = [rng.randrange(N_GROUPS) for _ in range(N_ITEMS)]
    item_author = [rng.randrange(N_AUTHORS) for _ in range(N_ITEMS)]
    # 作者也有兴趣倾向:作者的主簇 = 其作品的多数簇
    log = []
    for u in range(N_USERS):
        n = rng.randint(12, 30)
        for _ in range(n):
            if rng.random() < 0.75:                       # 75% 同簇内消费
                pool = [i for i in range(N_ITEMS)
                        if item_group[i] == user_group[u]]
                it = pool[rng.randrange(len(pool))]
            else:
                it = rng.randrange(N_ITEMS)
            log.append((u, it))
    rng.shuffle(log)
    cut = int(len(log) * 0.82)
    return (log[:cut], log[cut:], user_group, item_group, item_author)


# ======================================================== [MOCK] 统一相似度服务
class SimilarityService:
    """[MOCK] 正文所说「可以复用同一套相似度计算服务」的进程内实现。

    对外只暴露四个接口:
      sim_items(i)   -> i2i 相似物品拉链
      sim_users(u)   -> u2u 相似用户拉链
      sim_authors(a) -> a2a 相似作者拉链
      items_of(u) / users_of(i) -> 正排/倒排的交互关系
    7 条关系路径全部由这四个接口拼出来,调用次数被逐路统计。
    """

    def __init__(self, train, item_author):
        self.item_author = item_author
        self.u_items = defaultdict(set)
        self.i_users = defaultdict(set)
        for u, i in train:
            self.u_items[u].add(i)
            self.i_users[i].add(u)
        self.a_users = defaultdict(set)
        self.a_items = defaultdict(set)
        for i, a in enumerate(item_author):
            self.a_items[a].add(i)
            for u in self.i_users[i]:
                self.a_users[a].add(u)
        self.calls = Counter()
        self.i2i = self._cosine_table(self.i_users, N_ITEMS)
        self.u2u = self._cosine_table(self.u_items, N_USERS)
        self.a2a = self._cosine_table(self.a_users, N_AUTHORS)

    @staticmethod
    def _cosine_table(sets, n):
        """余弦相似度: sim(x,y) = |S_x ∩ S_y| / sqrt(|S_x|·|S_y|),截断到 TOPN。"""
        inv = defaultdict(list)
        for x, s in sets.items():
            for e in s:
                inv[e].append(x)
        pair = defaultdict(float)
        for e, xs in inv.items():
            if len(xs) > 200:              # 超长拉链降权采样,防止 O(n^2) 爆炸
                continue
            for a in range(len(xs)):
                for b in range(a + 1, len(xs)):
                    pair[(xs[a], xs[b])] += 1.0
                    pair[(xs[b], xs[a])] += 1.0
        table = defaultdict(list)
        for (x, y), c in pair.items():
            denom = math.sqrt(len(sets[x]) * len(sets[y])) or 1.0
            table[x].append((y, c / denom))
        for x in list(table):
            table[x].sort(key=lambda t: -t[1])
            table[x] = table[x][:TOPN]
        return table

    def sim_items(self, i, k=K1):
        self.calls["i2i表"] += 1
        return self.i2i.get(i, [])[:k]

    def sim_users(self, u, k=K1):
        self.calls["u2u表"] += 1
        return self.u2u.get(u, [])[:k]

    def sim_authors(self, a, k=K1):
        self.calls["a2a表"] += 1
        return self.a2a.get(a, [])[:k]

    def items_of(self, u, k=K2):
        self.calls["正排:用户->物品"] += 1
        return sorted(self.u_items.get(u, ()))[:k]

    def users_of(self, i, k=K2):
        self.calls["倒排:物品->用户"] += 1
        return sorted(self.i_users.get(i, ()))[:k]

    def author_of(self, i):
        self.calls["正排:物品->作者"] += 1
        return self.item_author[i]


# ======================================================== 用户向量(U2I 用)
def build_vectors(train, dim, rng):
    """[MOCK] 极简矩阵分解替身:用「用户消费过的物品簇分布」当作向量。

    真正的双塔/MF 训练见第 10 章;此处只需要一个能跑 U2I 路径的向量即可。
    """
    u_vec = defaultdict(lambda: [0.0] * dim)
    i_vec = defaultdict(lambda: [0.0] * dim)
    for u, i in train:
        h = (i * 2654435761) % dim
        u_vec[u][h] += 1.0
        i_vec[i][h] += 1.0
    for d in (u_vec, i_vec):
        for k in list(d):
            nrm = math.sqrt(sum(x * x for x in d[k])) or 1.0
            d[k] = [x / nrm + rng.uniform(-1e-6, 1e-6) for x in d[k]]
    return u_vec, i_vec


# ======================================================== 7 条关系路径
def trigger_weights(items):
    """trigger 时间衰减权重 w(t) = 0.9^rank(t)。"""
    return [(it, 0.9 ** r) for r, it in enumerate(items[:N_TRIGGER])]


def path_u2i(u, u_vec, i_vec):
    """U2I:用户向量直接与物品向量做内积检索(一跳,不经过中间实体)。"""
    uv = u_vec.get(u)
    if uv is None:
        return {}, 1
    return {i: sum(a * b for a, b in zip(uv, iv)) for i, iv in i_vec.items()}, 1


def path_i2i(svc, triggers):
    """I2I:对每个 trigger 查 i2i 相似表(一跳)。"""
    out = defaultdict(float)
    for t, w in triggers:
        for j, s in svc.sim_items(t):
            out[j] += w * s
    return out, len(triggers) * K1


def path_u2u2i(svc, u):
    """U2U2I:先查相似用户(u2u 表),再展开相似用户的物品(正排,两跳)。"""
    out = defaultdict(float)
    fan = 0
    for v, su in svc.sim_users(u):
        for j in svc.items_of(v):
            out[j] += su
            fan += 1
    return out, fan


def path_u2i2i(svc, triggers):
    """U2I2I:trigger -> 相似物品 -> 再相似物品(两跳,都查同一张 i2i 表)。"""
    out = defaultdict(float)
    fan = 0
    for t, w in triggers:
        for y, s1 in svc.sim_items(t):
            for z, s2 in svc.sim_items(y, K2):
                out[z] += w * s1 * s2
                fan += 1
    return out, fan


def path_a2a(svc, author):
    """A2A:创作者关系路径,给定作者找相似作者(作者侧的一跳)。"""
    return svc.sim_authors(author, REC_K), K1


def path_i2u2u(svc, item):
    """I2U2U(作者侧):物品 -> 交互过的用户(倒排) -> 相似用户(u2u 表),两跳。

    用途:为一条内容寻找潜在感兴趣用户,做内容冷启分发。
    """
    out = defaultdict(float)
    fan = 0
    seeds = svc.users_of(item, K1)
    for u in seeds:
        for v, s in svc.sim_users(u, K2):
            out[v] += s
            fan += 1
    return out, fan


def path_i2i2u(svc, item):
    """I2I2U(作者侧):物品 -> 相似物品(i2i 表) -> 交互过的用户(倒排),两跳。"""
    out = defaultdict(float)
    fan = 0
    for y, s in svc.sim_items(item, K1):
        for v in svc.users_of(y, K2):
            out[v] += s
            fan += 1
    return out, fan


def topk(scores, k, exclude=()):
    if isinstance(scores, list):
        return scores[:k]
    ex = set(exclude)
    return sorted(((x, s) for x, s in scores.items() if x not in ex),
                  key=lambda t: -t[1])[:k]


# ======================================================== 评测
def eval_user_side(svc, train_items, holdout, u_vec, i_vec):
    """用户侧 4 条路径:命中率@K / 物品覆盖率 / 扇出 / 在线耗时。"""
    names = ["U2I", "I2I", "U2U2I", "U2I2I"]
    hit = Counter()
    tot = Counter()
    cov = defaultdict(set)
    fanout = Counter()
    cost = defaultdict(float)
    for u, truth in holdout.items():
        hist = train_items.get(u)
        if not hist or not truth:
            continue
        triggers = trigger_weights(hist)
        for name in names:
            t0 = time.perf_counter()
            if name == "U2I":
                sc, fan = path_u2i(u, u_vec, i_vec)
            elif name == "I2I":
                sc, fan = path_i2i(svc, triggers)
            elif name == "U2U2I":
                sc, fan = path_u2u2i(svc, u)
            else:
                sc, fan = path_u2i2i(svc, triggers)
            rec = topk(sc, REC_K, exclude=set(hist))
            cost[name] += (time.perf_counter() - t0) * 1000
            fanout[name] += fan
            tot[name] += 1
            cov[name].update(x for x, _ in rec)
            if any(x in truth for x, _ in rec):
                hit[name] += 1
    return names, hit, tot, cov, fanout, cost


def eval_creator_side(svc, holdout_items):
    """作者侧 2 条路径:给物品找用户,命中留出集中真正交互过该物品的用户。"""
    names = ["I2U2U", "I2I2U"]
    hit, tot, fanout, cost = Counter(), Counter(), Counter(), defaultdict(float)
    cov = defaultdict(set)
    for item, truth_users in holdout_items.items():
        if not truth_users:
            continue
        for name in names:
            t0 = time.perf_counter()
            sc, fan = (path_i2u2u(svc, item) if name == "I2U2U"
                       else path_i2i2u(svc, item))
            rec = topk(sc, REC_K)
            cost[name] += (time.perf_counter() - t0) * 1000
            fanout[name] += fan
            tot[name] += 1
            cov[name].update(x for x, _ in rec)
            if any(v in truth_users for v, _ in rec):
                hit[name] += 1
    return names, hit, tot, cov, fanout, cost


def main():
    random.seed(42)
    rng = random.Random(42)
    dim = 16

    train, test, _, _, item_author = gen_log(rng)
    svc = SimilarityService(train, item_author)
    u_vec, i_vec = build_vectors(train, dim, rng)

    train_items = defaultdict(list)
    for u, i in train:
        train_items[u].append(i)
    holdout = defaultdict(set)
    holdout_items = defaultdict(set)
    for u, i in test:
        holdout[u].add(i)
        holdout_items[i].add(u)

    print("=" * 96)
    print("§4.3 召回的「关系路径」视角:U2I / I2I / A2A / U2U2I / U2I2I / I2U2U / I2I2U")
    print("=" * 96)
    print(f"\n[MOCK] 合成日志: 用户 {N_USERS} / 物品 {N_ITEMS} / 作者 {N_AUTHORS} / "
          f"兴趣簇 {N_GROUPS};交互 {len(train) + len(test)} 条"
          f"(训练 {len(train)} / 留出 {len(test)})")
    print(f"统一相似度服务已建三张表:i2i {len(svc.i2i)} 键 / u2u {len(svc.u2u)} 键 / "
          f"a2a {len(svc.a2a)} 键,每键截断 TopN={TOPN}")

    # ---------------- 路径形态表
    print("\n[路径形态] 每条路径 = 几跳 + 每跳查哪张表(正文:复用同一套相似度服务)")
    print(pad("路径", 8) + pad("侧", 8) + pad("跳数", 6, ">") + "  "
          + pad("第一跳查", 20) + pad("第二跳查", 20) + pad("语义", 24))
    print("-" * 86)
    shape = [
        ("U2I", "用户侧", 1, "向量索引(内积)", "-", "用户兴趣直接找物品"),
        ("I2I", "用户侧", 1, "i2i 相似表", "-", "已交互物品找相似物品"),
        ("A2A", "作者侧", 1, "a2a 相似表", "-", "创作者关系找相似创作者"),
        ("U2U2I", "用户侧", 2, "u2u 相似表", "正排:用户->物品", "相似用户喜欢的物品"),
        ("U2I2I", "用户侧", 2, "i2i 相似表", "i2i 相似表", "历史物品的两跳扩展"),
        ("I2U2U", "作者侧", 2, "倒排:物品->用户", "u2u 相似表", "为内容找潜在用户"),
        ("I2I2U", "作者侧", 2, "i2i 相似表", "倒排:物品->用户", "相似物品的受众"),
    ]
    for name, side, hops, s1, s2, mean in shape:
        print(pad(name, 8) + pad(side, 8) + pad(hops, 6, ">") + "  "
              + pad(s1, 20) + pad(s2, 20) + pad(mean, 24))

    # ---------------- A2A 展示(一跳的作者侧路径)
    svc.calls.clear()
    print("\n[A2A 单例] 作者维度的相似作者 Top5(作者侧生态常用:相似创作者推荐)")
    seed_author = max(range(N_AUTHORS), key=lambda a: len(svc.a_users[a]))
    sims, a2a_fan = path_a2a(svc, seed_author)
    print(f"  种子作者 a{seed_author:02d}(受众 {len(svc.a_users[seed_author])} 人,"
          f"作品 {len(svc.a_items[seed_author])} 个),扇出 {a2a_fan}")
    print("  " + pad("相似作者", 12) + pad("相似度", 10, ">")
          + pad("受众数", 9, ">") + pad("作品数", 9, ">"))
    for a, s in sims[:5]:
        print("  " + pad(f"a{a:02d}", 12) + f"{s:>10.4f}"
              + f"{len(svc.a_users[a]):>9}" + f"{len(svc.a_items[a]):>9}")
    a2a_calls = dict(svc.calls)

    # ---------------- 用户侧评测
    svc.calls.clear()
    names, hit, tot, cov, fanout, cost = eval_user_side(
        svc, train_items, holdout, u_vec, i_vec)
    print(f"\n[用户侧评测] 每用户召回 Top{REC_K},命中留出集即算命中"
          f"(评测用户 {tot[names[0]]} 个)")
    print(pad("路径", 10) + pad("HitRate@20", 13, ">") + pad("物品覆盖率", 14, ">")
          + pad("平均扇出", 11, ">") + pad("平均耗时/ms", 15, ">"))
    print("-" * 63)
    for n in names:
        print(pad(n, 10) + f"{hit[n] / max(tot[n], 1):>13.4f}"
              + f"{len(cov[n]) / N_ITEMS:>14.1%}"
              + f"{fanout[n] / max(tot[n], 1):>11.1f}"
              + f"{cost[n] / max(tot[n], 1):>15.3f}")
    user_calls = dict(svc.calls)

    # ---------------- 作者侧评测
    svc.calls.clear()
    cnames, chit, ctot, ccov, cfan, ccost = eval_creator_side(svc, holdout_items)
    print(f"\n[作者侧评测] 每个物品召回 Top{REC_K} 个目标用户"
          f"(评测物品 {ctot[cnames[0]]} 个,正文:Creator Recommendation)")
    print(pad("路径", 10) + pad("HitRate@20", 13, ">") + pad("用户覆盖率", 14, ">")
          + pad("平均扇出", 11, ">") + pad("平均耗时/ms", 15, ">"))
    print("-" * 63)
    for n in cnames:
        print(pad(n, 10) + f"{chit[n] / max(ctot[n], 1):>13.4f}"
              + f"{len(ccov[n]) / N_USERS:>14.1%}"
              + f"{cfan[n] / max(ctot[n], 1):>11.1f}"
              + f"{ccost[n] / max(ctot[n], 1):>15.3f}")
    creator_calls = dict(svc.calls)

    # ---------------- 相似表复用统计
    print("\n[相似表复用] 正文:「相似用户」「相似物品」模块往往复用同一套相似度服务")
    print(pad("被调用的服务接口", 22) + pad("用户侧 4 路调用", 17, ">")
          + pad("作者侧 2 路调用", 17, ">") + pad("A2A 调用", 11, ">")
          + pad("被几条路径复用", 17, ">"))
    print("-" * 86)
    reuse = {
        "i2i表": ["I2I", "U2I2I", "I2I2U"],
        "u2u表": ["U2U2I", "I2U2U"],
        "a2a表": ["A2A"],
        "正排:用户->物品": ["U2U2I"],
        "倒排:物品->用户": ["I2U2U", "I2I2U"],
    }
    for api, paths in reuse.items():
        print(pad(api, 22) + f"{user_calls.get(api, 0):>17,}"
              + f"{creator_calls.get(api, 0):>17,}"
              + f"{a2a_calls.get(api, 0):>11,}"
              + f"{len(paths):>17}  {'/'.join(paths)}")

    # ---------------- 技术路线 × 关系路径 交叉表
    print("\n[两种划分方式的交叉] 正文:「技术路线」与「关系路径」并不冲突")
    print(pad("技术路线", 16) + pad("典型算法", 30) + pad("可实现的关系路径", 34))
    print("-" * 80)
    cross = [
        ("协同过滤召回", "Item-CF / Swing", "I2I、U2I2I、I2I2U"),
        ("协同过滤召回", "User-CF", "U2U2I、I2U2U"),
        ("内容与规则召回", "Tag 匹配 / 热门 / 运营规则", "Tag、Hot、Category(无实体跳转)"),
        ("向量召回", "双塔 / MIND / 多兴趣", "U2I(也可做 I2I)"),
        ("图召回", "GNN / 随机游走 / 图 Embedding", "U2I、I2I、U2U2I、U2I2I 均可"),
    ]
    for tech, algo, paths in cross:
        print(pad(tech, 16) + pad(algo, 30) + pad(paths, 34))

    print("\n结论(正文 §4.3 要点回顾):")
    print("  1) 7 条路径在本脚本里只用了 5 个服务接口,其中 i2i 表被 3 条路径复用、")
    print("     u2u 表被 2 条、倒排被 2 条复用 -- 这正是正文说的「可以复用同一套")
    print("     相似度计算服务」:维护少数几张相似表,即可派生出一整族召回路;")
    print("  2) 跳数直接决定在线成本:一跳 I2I 扇出 = |Trigger|×K1,")
    print(f"     两跳 U2I2I 扇出 = |Trigger|×K1×K2,本次实测"
          f" {fanout['I2I'] / max(tot['I2I'], 1):.0f} vs"
          f" {fanout['U2I2I'] / max(tot['U2I2I'], 1):.0f}(相差"
          f" {fanout['U2I2I'] / max(fanout['I2I'], 1):.0f} 倍),耗时也同步放大;")
    print(f"  3) 两跳路径靠「多跳证据累加」换命中率:I2I"
          f" {hit['I2I'] / max(tot['I2I'], 1):.4f} ->"
          f" U2I2I {hit['U2I2I'] / max(tot['U2I2I'], 1):.4f};")
    print(f"     但本次实测两跳的物品覆盖率反而更低(I2I"
          f" {len(cov['I2I']) / N_ITEMS:.1%} -> U2I2I"
          f" {len(cov['U2I2I']) / N_ITEMS:.1%}),")
    print("     因为两跳得分是相似度乘积、会把分数进一步堆到少数枢纽物品上 --")
    print("     所以工业界用的是「多路并存 + 融合配额」,而不是指望单路同时")
    print("     做到高精度与高覆盖(配额机制见 §4.4 的融合脚本);")
    print("  4) I2U2U / I2I2U 的输出是「用户列表」而不是「物品列表」,因此只能用于")
    print("     作者侧的内容冷启分发,不能直接塞进用户侧的候选集;")
    print("  5) 关系路径描述的是信号怎么传播,技术路线描述的是相似度怎么算出来 --")
    print("     同一条 I2I 既可以由 Item-CF 实现,也可以由双塔或 GNN 实现;")
    print("  6) U2I 这一路命中率最低,是因为本脚本只用「哈希桶计数向量」当作")
    print("     向量召回的占位实现(真正的双塔/多兴趣模型见第 10 章),")
    print("     这里只演示路径形态,不代表向量召回的真实效果。")


if __name__ == "__main__":
    main()
