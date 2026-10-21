"""对应正文第 7 章 §7.2「重排模块架构」中的四个环节：
重排前处理（Pre-processing）→ 序列生成（List Generation）→ 序列评估（List Evaluation）
→ 多样性打散（Diversification）。

本实现做了什么
----------------
用纯 Python 串出一条端到端的重排链路，各环节都是可替换的组件：
1. [MOCK] `ForwardIndexService` / `RedisFeatureStore`：模拟正排索引服务与 Redis 在线特征存储，
   提供物品属性（垂类/作者/时长/内容类型）与 item embedding；
2. `PreProcessor`：特征补全 + 属性初始化 + 数据清洗（丢弃缺 embedding、缺垂类、时长异常的候选）；
3. `ListGenerator`：高斯采样多目标融合权重，贪心生成多条候选序列（§7.3）；
4. `ListEvaluator`：按「是否继续下滑」口径做 listwise 序列打分（§7.4）；
5. `Diversifier`：MMR 贪心打散（§7.5）；
6. `PostProcessor`：相邻去重 + 窗口配额 + 强插（§7.6）。

关键公式（§7.2 定义 7.1）
--------------------------
    max_pi F(pi) = sum_{t=1}^{T} Rel(i_t, u) + lambda * Div(pi) + mu * Constrain(pi)
    可行解空间 O(P(N, T)) = O(N! / (N-T)!)

每个环节都打印输入/输出规模与耗时占比，便于对照正文的流程图。
"""

import math
import random
import time

CATEGORIES = ("影视", "美食", "萌宠", "知识", "游戏")
CONTENT_TYPES = ("短视频", "短剧", "直播", "广告")
TARGETS = ("p_evr", "p_lvr", "p_ltr", "p_cmr", "p_ftr")
FUSION_MEAN = (0.2, 0.3, 0.2, 0.2, 0.1)

EMB_DIM = 8
LIST_LEN = 10


# --------------------------------------------------------------------------
# [MOCK] 外部依赖
# --------------------------------------------------------------------------
# 真实候选池是「头部效应」的：热门垂类占比高，同垂类物品 embedding 聚簇、
# 精排分也整体偏高。只有这样打散环节才有实际作用。
CATEGORY_WEIGHTS = (0.42, 0.26, 0.14, 0.10, 0.08)
CATEGORY_SCORE_BONUS = {"影视": 0.10, "美食": 0.07, "萌宠": 0.03, "知识": 0.0, "游戏": 0.0}


def build_catalog(n_items):
    """[MOCK] 物品底表：垂类按头部效应抽样，作者从该垂类的小作者池中抽。"""
    centroids = {}
    for c in CATEGORIES:
        v = [random.gauss(0.0, 1.0) for _ in range(EMB_DIM)]
        norm = math.sqrt(sum(x * x for x in v))
        centroids[c] = [x / norm for x in v]

    catalog = {}
    for i in range(n_items):
        cat = random.choices(CATEGORIES, weights=CATEGORY_WEIGHTS)[0]
        catalog["item_%d" % i] = {
            "cat": cat,
            # 每个垂类只有 4 个头部作者，容易出现同作者相邻
            "author_id": "author_%s_%d" % (cat, random.randint(0, 3)),
            "duration": round(random.uniform(5.0, 300.0), 1),
            "content_type": random.choices(CONTENT_TYPES, weights=(0.70, 0.12, 0.10, 0.08))[0],
            # 同垂类物品围绕同一个质心，cos 相似度高
            "emb": [0.85 * centroids[cat][d] + 0.35 * random.gauss(0.0, 1.0)
                    for d in range(EMB_DIM)],
        }
    return catalog


class ForwardIndexService:
    """[MOCK] 正排索引服务：按 item_id 批量返回静态属性。"""

    def __init__(self, catalog, missing_rate=0.06):
        self._table = {
            k: {f: v[f] for f in ("cat", "author_id", "duration", "content_type")}
            for k, v in catalog.items()
            if random.random() >= missing_rate   # 模拟索引缺失，前处理需清洗
        }
        self.qps = 0

    def mget(self, item_ids):
        self.qps += 1
        return {i: self._table.get(i) for i in item_ids}


class RedisFeatureStore:
    """[MOCK] Redis 在线特征存储：返回 item embedding 与近期曝光计数。"""

    def __init__(self, catalog, missing_rate=0.04):
        self._emb = {k: v["emb"] for k, v in catalog.items() if random.random() >= missing_rate}
        self._exposure = {
            k: random.choices([0, 1, 2, 5], weights=(0.8, 0.1, 0.06, 0.04))[0] for k in catalog
        }
        self.qps = 0

    def mget(self, item_ids):
        self.qps += 1
        return {
            i: {"emb": self._emb.get(i), "recent_exposure": self._exposure.get(i, 0)}
            for i in item_ids
        }


def mock_rank_output(catalog):
    """[MOCK] 精排模块输出：item_id + 各目标 pxtr 预估值。热门垂类整体分更高。"""
    out = []
    for item_id, meta in catalog.items():
        bonus = CATEGORY_SCORE_BONUS[meta["cat"]]
        pxtr = {t: round(min(0.999, random.random() * 0.74 + bonus), 6) for t in TARGETS}
        out.append({"item_id": item_id, "pxtr": pxtr})
    return out


# --------------------------------------------------------------------------
# 1. 重排前处理
# --------------------------------------------------------------------------
class PreProcessor:
    """特征补全 + 属性初始化 + 数据清洗。"""

    def __init__(self, index_service, feature_store):
        self.index = index_service
        self.store = feature_store
        self.drop_stats = {}

    def run(self, rank_items):
        ids = [it["item_id"] for it in rank_items]
        attrs = self.index.mget(ids)       # 一次批量 RPC，而不是逐条
        feats = self.store.mget(ids)
        self.drop_stats = {"no_attr": 0, "no_emb": 0, "bad_duration": 0}

        cleaned = []
        for it in rank_items:
            a = attrs.get(it["item_id"])
            if a is None:
                self.drop_stats["no_attr"] += 1
                continue
            f = feats.get(it["item_id"]) or {}
            emb = f.get("emb")
            if emb is None:
                self.drop_stats["no_emb"] += 1
                continue
            if not (1.0 <= a["duration"] <= 600.0):
                self.drop_stats["bad_duration"] += 1
                continue
            norm = math.sqrt(sum(v * v for v in emb)) or 1.0
            cleaned.append(
                {
                    "item_id": it["item_id"],
                    "pxtr": it["pxtr"],
                    "cat": a["cat"],
                    "author_id": a["author_id"],
                    "duration": a["duration"],
                    "content_type": a["content_type"],
                    "emb": [v / norm for v in emb],   # 归一化，供相似度计算
                    "recent_exposure": f.get("recent_exposure", 0),
                }
            )
        return cleaned


# --------------------------------------------------------------------------
# 2. 序列生成（§7.3）
# --------------------------------------------------------------------------
class ListGenerator:
    """高斯采样融合权重 + 贪心排序生成多条候选序列。

    每组权重同时产出两个东西：
      - seq ：前 list_len 个物品，作为交给评估器的「候选序列」；
      - pool：前 pool_len 个物品，作为后续多样性打散的选择空间
              （打散要有挑选余地，否则只能在固定集合里换顺序）。
    """

    def __init__(self, mean=FUSION_MEAN, std=0.1, n_samples=8,
                 list_len=LIST_LEN, pool_len=30):
        self.mean = mean
        self.std = std
        self.n_samples = n_samples
        self.list_len = list_len
        self.pool_len = pool_len

    def fuse(self, item, w):
        return sum(wi * item["pxtr"][t] for wi, t in zip(w, TARGETS))

    def run(self, items):
        seqs = []
        for k in range(self.n_samples):
            # 第 0 组用基准权重（线上主策略），其余围绕基准做高斯扰动
            w = (
                list(self.mean)
                if k == 0
                else [m + random.gauss(0.0, self.std) for m in self.mean]
            )
            ranked = sorted(items, key=lambda it: -self.fuse(it, w))
            seqs.append(
                {
                    "weights": w,
                    "seq": ranked[: self.list_len],
                    "pool": ranked[: self.pool_len],
                    "fuse": {it["item_id"]: self.fuse(it, w) for it in ranked[: self.pool_len]},
                }
            )
        # 去掉完全重复的序列
        uniq, seen = [], set()
        for s in seqs:
            key = tuple(it["item_id"] for it in s["seq"])
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq


# --------------------------------------------------------------------------
# 3. 序列评估（§7.4）
# --------------------------------------------------------------------------
class ListEvaluator:
    """listwise 评估：按「是否继续下滑」建模曝光概率后加权求和。"""

    def __init__(self, p0=1.0, fatigue=0.75):
        self.p0 = p0
        self.fatigue = fatigue

    @staticmethod
    def item_value(item):
        """单点分 f(x_i)：取预估观看时长（秒）。"""
        return item["duration"] * (0.25 + 0.55 * item["pxtr"]["p_lvr"])

    def p_slide(self, item, prev):
        p = 0.50 + 0.45 * item["pxtr"]["p_evr"]
        if prev is not None and prev["cat"] == item["cat"]:
            p *= self.fatigue
        if prev is not None and prev["author_id"] == item["author_id"]:
            p *= self.fatigue
        return min(p, 0.99)

    def score(self, seq):
        total, prob, prev = 0.0, self.p0, None
        for item in seq:
            total += prob * self.item_value(item)
            prob *= self.p_slide(item, prev)
            prev = item
        return total

    def run(self, candidate_seqs):
        scored = [(self.score(s["seq"]), s) for s in candidate_seqs]
        scored.sort(key=lambda x: -x[0])
        return scored


# --------------------------------------------------------------------------
# 4. 多样性打散（§7.5，MMR）
# --------------------------------------------------------------------------
class Diversifier:
    """MMR 贪心打散：lambda*rel - (1-lambda)*max_sim。"""

    def __init__(self, lam=0.7):
        self.lam = lam

    @staticmethod
    def cos(a, b):
        return sum(x * y for x, y in zip(a["emb"], b["emb"]))

    def run(self, pool, rel_scores, k):
        """从 pool 中按 MMR 贪心选出 k 个物品（同时决定集合与顺序）。"""
        remaining = list(pool)
        selected = [max(remaining, key=lambda it: rel_scores[it["item_id"]])]
        remaining.remove(selected[0])
        while remaining and len(selected) < k:
            best, best_v = None, -1e18
            for cand in remaining:
                max_sim = max(self.cos(cand, s) for s in selected)
                v = self.lam * rel_scores[cand["item_id"]] - (1.0 - self.lam) * max_sim
                if v > best_v:
                    best, best_v = cand, v
            selected.append(best)
            remaining.remove(best)
        return selected


# --------------------------------------------------------------------------
# 5. 重排后处理（§7.6）
# --------------------------------------------------------------------------
class PostProcessor:
    """相邻去重（垂类/作者）+ 内容类型窗口配额 + 运营强插。"""

    def __init__(self, max_same_cat_in_window=3, window=5, ad_quota=1, force_slot=3):
        self.max_same_cat = max_same_cat_in_window
        self.window = window
        self.ad_quota = ad_quota
        self.force_slot = force_slot
        self.actions = []

    def run(self, seq, forced_item):
        out = []
        pool = list(seq)
        self.actions = []
        while pool:
            picked = None
            for cand in pool:
                if out and (out[-1]["cat"] == cand["cat"] or out[-1]["author_id"] == cand["author_id"]):
                    continue
                win = out[-(self.window - 1):] if self.window > 1 else []
                if sum(1 for x in win if x["cat"] == cand["cat"]) >= self.max_same_cat:
                    continue
                if cand["content_type"] == "广告" and \
                        sum(1 for x in out if x["content_type"] == "广告") >= self.ad_quota:
                    continue
                picked = cand
                break
            if picked is None:                   # 约束无法满足，放宽为取队首（兜底）
                picked = pool[0]
                self.actions.append("relax@pos%d:%s" % (len(out) + 1, picked["item_id"]))
            pool.remove(picked)
            out.append(picked)

        if forced_item is not None:
            slot = min(self.force_slot - 1, len(out))
            out.insert(slot, forced_item)
            self.actions.append("force_insert@pos%d:%s" % (slot + 1, forced_item["item_id"]))
            out = out[:LIST_LEN]
        return out


# --------------------------------------------------------------------------
def fmt_list(seq):
    lines = []
    for i, it in enumerate(seq, 1):
        lines.append("  %2d. %-9s %-5s %-11s %-9s dur=%6.1fs"
                     % (i, it["item_id"], it["cat"], it["author_id"], it["content_type"], it["duration"]))
    return "\n".join(lines)


def avg_pairwise_cos(seq):
    pairs = [(a, b) for i, a in enumerate(seq) for b in seq[i + 1:]]
    if not pairs:
        return 0.0
    return sum(sum(x * y for x, y in zip(a["emb"], b["emb"])) for a, b in pairs) / len(pairs)


def diversity_report(seq, name):
    cats = {it["cat"] for it in seq}
    authors = {it["author_id"] for it in seq}
    adj_cat = sum(1 for a, b in zip(seq, seq[1:]) if a["cat"] == b["cat"])
    adj_author = sum(1 for a, b in zip(seq, seq[1:]) if a["author_id"] == b["author_id"])
    print("  %-10s 垂类覆盖=%d/%d 作者覆盖=%2d/%d 相邻同垂类=%d 相邻同作者=%d 平均两两cos=%.4f"
          % (name, len(cats), len(CATEGORIES), len(authors), len(seq),
             adj_cat, adj_author, avg_pairwise_cos(seq)))


def main():
    random.seed(42)

    n_candidates = 300
    print("=" * 96)
    print("§7.2 重排模块四段链路：前处理 -> 序列生成 -> 序列评估 -> 多样性打散 -> 后处理")
    print("=" * 96)

    catalog = build_catalog(n_candidates)
    index = ForwardIndexService(catalog)
    store = RedisFeatureStore(catalog)
    rank_items = mock_rank_output(catalog)
    cat_hist = {}
    for m in catalog.values():
        cat_hist[m["cat"]] = cat_hist.get(m["cat"], 0) + 1
    print("[精排输出] 候选 %d 个（每个带 %d 个目标的 pxtr 预估）" % (len(rank_items), len(TARGETS)))
    print("           候选池垂类分布（头部效应）：%s"
          % ", ".join("%s=%d" % (c, cat_hist.get(c, 0)) for c in CATEGORIES))

    timings = {}

    # ---- 前处理 ----
    t0 = time.perf_counter()
    pre = PreProcessor(index, store)
    items = pre.run(rank_items)
    timings["前处理"] = time.perf_counter() - t0
    print("\n[1/5 前处理] %d -> %d 个候选；RPC 次数：正排 %d 次 / Redis %d 次"
          % (len(rank_items), len(items), index.qps, store.qps))
    print("           清洗丢弃：%s" % pre.drop_stats)

    # ---- 序列生成 ----
    t0 = time.perf_counter()
    gen = ListGenerator()
    cand_seqs = gen.run(items)
    timings["序列生成"] = time.perf_counter() - t0
    print("\n[2/5 序列生成] 采样 %d 组融合权重 -> %d 条互不相同的候选序列（每条长度 %d）"
          % (gen.n_samples, len(cand_seqs), LIST_LEN))
    for k, s in enumerate(cand_seqs[:3]):
        print("           序列%d 权重=[%s] 头部=%s"
              % (k, ", ".join("%.3f" % w for w in s["weights"]),
                 "/".join(it["item_id"] for it in s["seq"][:4])))

    # ---- 序列评估 ----
    t0 = time.perf_counter()
    evaluator = ListEvaluator()
    scored = evaluator.run(cand_seqs)
    timings["序列评估"] = time.perf_counter() - t0
    print("\n[3/5 序列评估] listwise 打分（曝光概率 x 单点分 求和）：")
    for k, (v, s) in enumerate(scored):
        tag = "  <== 最优" if k == 0 else ""
        print("           序列%d value=%8.4f（点加和=%8.4f）%s"
              % (k, v, sum(evaluator.item_value(it) for it in s["seq"]), tag))
    best = scored[0][1]
    best_seq = best["seq"]

    # ---- 多样性打散 ----
    t0 = time.perf_counter()
    # 相关性分 rel 用该组融合分归一化到 [0, 1]，和相似度量级可比
    fmax = max(best["fuse"].values())
    rel = {k: v / fmax for k, v in best["fuse"].items()}
    div = Diversifier(lam=0.7)
    diversified = div.run(best["pool"], rel, LIST_LEN)
    timings["多样性打散"] = time.perf_counter() - t0
    print("\n[4/5 多样性打散] MMR(lambda=%.1f)，从 %d 个候选中选 %d 个："
          % (div.lam, len(best["pool"]), LIST_LEN))
    diversity_report(best_seq, "打散前")
    diversity_report(diversified, "打散后")

    # ---- 后处理 ----
    t0 = time.perf_counter()
    forced = {
        "item_id": "op_video_0",
        "pxtr": {t: 0.5 for t in TARGETS},
        "cat": "知识",
        "author_id": "author_official",
        "duration": 60.0,
        "content_type": "运营",
        "emb": [0.0] * EMB_DIM,
        "recent_exposure": 0,
    }
    post = PostProcessor()
    final = post.run(diversified, forced)
    timings["后处理"] = time.perf_counter() - t0
    print("\n[5/5 后处理] 规则命中：%s" % (post.actions or "无"))
    diversity_report(final, "最终曝光")

    print("\n最终曝光列表（长度 %d）：" % len(final))
    print(fmt_list(final))

    print("\n各环节耗时占比：")
    total = sum(timings.values())
    for k, v in timings.items():
        print("  %-12s %8.3f ms  (%5.1f%%)" % (k, v * 1000.0, v / total * 100.0))
    print("  %-12s %8.3f ms" % ("合计", total * 1000.0))

    final_value = evaluator.score(final)
    print("\n[目标函数 F(pi)] 最优生成序列 value=%.4f -> 打散后 value=%.4f -> 后处理后 value=%.4f"
          % (scored[0][0], evaluator.score(diversified), final_value))
    print("                 相关性被让渡 %.2f%%，换来垂类覆盖 %d->%d、相邻同垂类 %d->%d。"
          % ((scored[0][0] - final_value) / scored[0][0] * 100.0,
             len({it['cat'] for it in best_seq}), len({it['cat'] for it in final}),
             sum(1 for a, b in zip(best_seq, best_seq[1:]) if a["cat"] == b["cat"]),
             sum(1 for a, b in zip(final, final[1:]) if a["cat"] == b["cat"])))


if __name__ == "__main__":
    main()
