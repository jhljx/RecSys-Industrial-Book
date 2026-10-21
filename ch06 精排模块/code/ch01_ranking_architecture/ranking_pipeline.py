"""
精排模块端到端链路（第 6 章 §6.1 精排模块架构）

对应正文概念：
正文把精排模块内部拆成三个串行阶段（图 fig_fr_module）：
    (1) 精排主模型 + LTR 模型 → 输出各类 pxtr；
    (2) 多目标融合（Multi-objective Fusion）→ 统一排序分；
    (3) 流量调控与业务约束（Traffic Control）→ 最终 Top-K。
输入是粗排给出的 100~500 个候选，输出 50~100 个物品送往重排。

本实现做了什么：
用纯标准库把上述四步（打分 → 融合 → 调控 → 输出）串成一条可运行链路，
并在每一步打印候选集规模、分数分布与业务指标的变化，让读者看清
「模型分」如何一步步被业务约束改写成「最终曝光顺序」。

关键公式：
    多目标融合（正文 §6.4.1 乘法公式）：
        Score_i = prod_k ( beta_k + alpha_k * p_{i,k} ) ^ gamma_k
    流量调控（正文 §6.5.1 级联式伪代码）：
        Score_i <- Score_i * prod_j boost_j(item_i)
    保量约束（正文 §6.5.5）：
        Exposure_target >= Q_target

[MOCK] 组件说明：
    - MockRankingModelService：假的精排主模型 RPC 服务，用确定性哈希+特征线性组合
      伪造 pctr / pevtr / pltr / pcmtr / pcvr / pwatch_time，不训练、不联网。
    - MockLTRService：假的 LTR 模型服务，输出一个业务 Reward 排序分。
    - MockFeatureStore：假的特征存储，按 item_id 返回物品画像。
    - MockExposureLog：假的曝光流水日志，用于统计调控效果。
"""

import math
import random

# --------------------------------------------------------------------------
# [MOCK] 特征存储
# --------------------------------------------------------------------------


class MockFeatureStore:
    """[MOCK] 假的物品特征存储。线上应为 KV 存储（Redis / 自研 Feature Store）。"""

    CATEGORIES = ["搞笑", "娱乐", "知识", "生活", "体育"]

    def __init__(self, num_items: int, rng: random.Random):
        self.table: dict[int, dict] = {}
        for item_id in range(num_items):
            cat = self.CATEGORIES[item_id % len(self.CATEGORIES)]
            # 搞笑/娱乐天然高点击，知识/体育偏长尾 —— 复现正文说的生态失衡成因
            hot = cat in ("搞笑", "娱乐")
            self.table[item_id] = {
                "item_id": item_id,
                "category": cat,
                "duration": rng.choice([15.0, 30.0, 60.0, 120.0]),
                "age_hours": rng.choice([0.5, 2.0, 8.0, 48.0, 240.0]),
                "author_fans": int(10 ** rng.uniform(1.5, 6.0)),
                "in_campaign": rng.random() < 0.12,
                "quality": rng.betavariate(5, 2) if not hot else rng.betavariate(3, 3),
                "hot_prior": 0.55 if hot else 0.25,
            }

    def get(self, item_id: int) -> dict:
        return self.table[item_id]

    def is_new_video(self, item_id: int) -> bool:
        return self.table[item_id]["age_hours"] < 6.0


# --------------------------------------------------------------------------
# [MOCK] 精排主模型与 LTR 模型服务
# --------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class MockRankingModelService:
    """[MOCK] 假的精排主模型推理服务。

    正文 §6.2：精排主模型把用户/物品/上下文特征统一拼接后送入 DNN，
    用多任务学习（MTL）框架联合预估多个行为目标。这里不训练网络，
    直接用「用户兴趣向量 · 物品属性向量」的确定性线性打分来伪造 pxtr，
    保证同一 (uid, item_id) 多次调用结果一致（模拟线上幂等推理）。
    """

    # 正文 §6.3 列出的典型稀疏度量级：CTR 3%~10%，LTR <1%，CMTR <1‰，FTR <1‱
    TASK_BASE = {
        "pctr": 0.06,
        "pevtr": 0.35,
        "pltr": 0.008,
        "pcmtr": 0.0008,
        "pcvr": 0.004,
    }

    def __init__(self, store: MockFeatureStore):
        self.store = store
        self.call_count = 0

    def _affinity(self, user: dict, feat: dict) -> float:
        """用户兴趣与物品类目的匹配度，取值约 [-1, 1]。"""
        pref = user["cat_pref"].get(feat["category"], 0.0)
        return 2.0 * pref - 1.0 + 0.6 * (feat["quality"] - 0.5)

    def predict(self, user: dict, item_id: int) -> dict:
        """返回该候选物品的全部 pxtr 预估值。"""
        self.call_count += 1
        feat = self.store.get(item_id)
        a = self._affinity(user, feat)
        out = {}
        for task, base in self.TASK_BASE.items():
            logit = math.log(base / (1.0 - base)) + 2.2 * a + 1.4 * feat["hot_prior"]
            out[task] = _sigmoid(logit)
        # 时长预估是回归目标（正文 §6.3.1），单位秒，受视频时长上限约束
        out["pwatch_time"] = feat["duration"] * _sigmoid(1.8 * a + 0.3)
        return out


class MockLTRService:
    """[MOCK] 假的 LTR 模型服务。

    正文 §6.2：LTR 模型不直接拟合单个行为标签，而是基于业务经验构造排序
    Reward（综合满意度 / 长周期价值），直接优化排序效果。这里用一个
    人工 Reward 公式充当「已训练好的 LTR 模型输出」。
    """

    def __init__(self, store: MockFeatureStore):
        self.store = store

    def predict(self, user: dict, item_id: int, pxtr: dict) -> float:
        feat = self.store.get(item_id)
        # 归一化时长 + 互动加权 + 内容质量，映射到 (0,1)
        norm_wt = pxtr["pwatch_time"] / max(feat["duration"], 1e-6)
        raw = (
            0.5 * norm_wt
            + 20.0 * pxtr["pltr"]
            + 120.0 * pxtr["pcmtr"]
            + 0.4 * feat["quality"]
        )
        return _sigmoid(2.5 * (raw - 0.6))


class MockExposureLog:
    """[MOCK] 假的曝光流水日志，仅在内存里累加统计量。"""

    def __init__(self):
        self.records: list[dict] = []

    def write(self, stage: str, items: list[int], store: MockFeatureStore) -> None:
        for pos, item_id in enumerate(items):
            self.records.append({"stage": stage, "pos": pos, "item_id": item_id})

    def category_share(self, stage: str, store: MockFeatureStore) -> dict[str, float]:
        rows = [r for r in self.records if r["stage"] == stage]
        total = len(rows) or 1
        share: dict[str, float] = {}
        for r in rows:
            cat = store.get(r["item_id"])["category"]
            share[cat] = share.get(cat, 0.0) + 1.0 / total
        return share


# --------------------------------------------------------------------------
# Stage 2：多目标融合（乘法公式，正文 §6.4.1）
# --------------------------------------------------------------------------


class MultiplicativeFusion:
    """Score_i = prod_k (beta_k + alpha_k * p_{i,k}) ^ gamma_k"""

    def __init__(self, params: dict[str, tuple[float, float, float]]):
        self.params = params  # factor -> (alpha, beta, gamma)

    def fuse(self, factors: dict[str, float]) -> float:
        score = 1.0
        for name, (alpha, beta, gamma) in self.params.items():
            base = beta + alpha * factors[name]
            score *= max(base, 1e-9) ** gamma
        return score


# --------------------------------------------------------------------------
# Stage 3：流量调控（级联式，正文 §6.5.1）
# --------------------------------------------------------------------------


class CascadeTrafficController:
    """正文 §6.5.1 的级联式调控：融合 → 初排 → boost → 再排 → Top-K。"""

    def __init__(self, store: MockFeatureStore, quota: dict[str, int],
                 cat_floor: dict[str, int] | None = None):
        self.store = store
        self.quota = quota  # 保量目标，正文 §6.5.5 Exposure_target >= Q_target
        self.cat_floor = cat_floor or {}  # 垂类保底坑位，正文 §6.5.3

    def boost(self, item_id: int) -> tuple[float, list[str]]:
        feat = self.store.get(item_id)
        mul, hits = 1.0, []
        if self.store.is_new_video(item_id):      # 新内容冷启动扶持
            mul *= 1.10
            hits.append("new_video×1.10")
        if feat["author_fans"] < 1000:            # 新作者/低粉创作者保护
            mul *= 1.05
            hits.append("new_author×1.05")
        if feat["in_campaign"]:                   # 电商活动流量倾斜
            mul *= 1.15
            hits.append("campaign×1.15")
        return mul, hits

    def force_insert(self, ranked: list[tuple[int, float]], top_k: int):
        """保量强插（正文 §6.5.5）：配额未达成则从池内未入选物品按分数补位。

        同时实现正文 §6.5.3 的「各垂类内容分发均衡管控」：为每个类目设置
        最低曝光坑位 cat_floor，避免搞笑/娱乐等高点击类目垄断全部曝光。
        """
        selected = [i for i, _ in ranked[:top_k]]
        pool_of = {
            "new_video": lambda i: self.store.is_new_video(i),
            "campaign": lambda i: self.store.get(i)["in_campaign"],
        }
        inserted: list[tuple[str, int]] = []

        def evict_one(protected) -> int | None:
            """从队尾找一个可被替换的坑位（不属于任何保量池、且其类目仍超额）。"""
            cat_cnt: dict[str, int] = {}
            for i in selected:
                c = self.store.get(i)["category"]
                cat_cnt[c] = cat_cnt.get(c, 0) + 1
            for pos in range(len(selected) - 1, -1, -1):
                cur = selected[pos]
                if any(p(cur) for p in pool_of.values()):
                    continue
                cur_cat = self.store.get(cur)["category"]
                if cat_cnt.get(cur_cat, 0) <= self.cat_floor.get(cur_cat, 0):
                    continue
                if cur in protected:
                    continue
                return pos
            return None

        # ---- (a) 类目保底 ----
        for cat, floor in self.cat_floor.items():
            have = sum(1 for i in selected if self.store.get(i)["category"] == cat)
            spare = [i for i, _ in ranked[top_k:]
                     if self.store.get(i)["category"] == cat]
            for item_id in spare[: max(0, floor - have)]:
                pos = evict_one(protected=set())
                if pos is None:
                    break
                selected[pos] = item_id
                inserted.append((f"cat:{cat}", item_id))

        # ---- (b) 业务池保量 ----
        floor_ids = {i for _, i in inserted}
        for pool, pred in pool_of.items():
            need = self.quota.get(pool, 0) - sum(1 for i in selected if pred(i))
            if need <= 0:
                continue
            spare = [i for i, _ in ranked[top_k:] if pred(i)]
            for item_id in spare[:need]:
                pos = evict_one(protected=floor_ids)
                if pos is None:
                    break
                selected[pos] = item_id
                inserted.append((pool, item_id))
        return selected, inserted


# --------------------------------------------------------------------------
# 完整链路
# --------------------------------------------------------------------------


def gini(values: list[float]) -> float:
    """基尼系数，用于衡量流量集中度（0 完全均匀，1 完全垄断）。"""
    xs = sorted(values)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs))
    return cum / (n * sum(xs))


class RankingModule:
    """精排模块：打分 → 多目标融合 → 流量调控 → 输出 Top-K。"""

    def __init__(self, store, model, ltr, fusion, controller, log):
        self.store, self.model, self.ltr = store, model, ltr
        self.fusion, self.controller, self.log = fusion, controller, log

    def rank(self, user: dict, candidates: list[int], top_k: int, verbose=True):
        # ---- Step 1 主模型 + LTR 模型打分 ----
        scored = []
        for item_id in candidates:
            pxtr = self.model.predict(user, item_id)
            pxtr["pltr_reward"] = self.ltr.predict(user, item_id, pxtr)
            scored.append((item_id, pxtr))
        if verbose:
            print(f"[Step1 打分] 候选 {len(candidates)} 个，"
                  f"主模型 RPC {self.model.call_count} 次")
            keys = ["pctr", "pevtr", "pltr", "pcmtr", "pcvr", "pwatch_time",
                    "pltr_reward"]
            print(f"  {'因子':<12}{'min':>10}{'mean':>10}{'max':>10}")
            for k in keys:
                vs = [p[k] for _, p in scored]
                print(f"  {k:<12}{min(vs):>10.5f}{sum(vs)/len(vs):>10.5f}"
                      f"{max(vs):>10.5f}")

        # ---- Step 2 多目标融合 ----
        fused = []
        for item_id, pxtr in scored:
            factors = {
                "pctr": pxtr["pctr"],
                "pevtr": pxtr["pevtr"],
                "pltr": pxtr["pltr"],
                "pwatch_time_norm": pxtr["pwatch_time"]
                / self.store.get(item_id)["duration"],
                "ltr_reward": pxtr["pltr_reward"],
            }
            fused.append((item_id, self.fusion.fuse(factors)))
        fused.sort(key=lambda t: -t[1])
        if verbose:
            print(f"\n[Step2 多目标融合] Top-5 融合分："
                  + ", ".join(f"item{i}:{s:.4f}" for i, s in fused[:5]))

        base_topk = [i for i, _ in fused[:top_k]]
        self.log.write("before_control", base_topk, self.store)

        # ---- Step 3 流量调控（boost + 再排） ----
        boosted = []
        boost_hits = 0
        for item_id, s in fused:
            mul, hits = self.controller.boost(item_id)
            boost_hits += 1 if hits else 0
            boosted.append((item_id, s * mul))
        boosted.sort(key=lambda t: -t[1])
        if verbose:
            print(f"[Step3 流量调控] {boost_hits}/{len(fused)} 个候选命中 boost 规则")

        # ---- Step 4 保量强插 + 输出 Top-K ----
        final, inserted = self.controller.force_insert(boosted, top_k)
        self.log.write("after_control", final, self.store)
        if verbose:
            if inserted:
                print(f"[Step4 保量强插] 强插 {len(inserted)} 个："
                      + ", ".join(f"{p}->item{i}" for p, i in inserted))
            else:
                print("[Step4 保量强插] 配额已自然满足，无需强插")
        return final


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    NUM_ITEMS, NUM_CANDIDATES, TOP_K = 400, 300, 60

    store = MockFeatureStore(NUM_ITEMS, rng)
    model = MockRankingModelService(store)
    ltr = MockLTRService(store)
    log = MockExposureLog()

    # 乘法公式超参：(alpha, beta, gamma)，gamma 控制头部/中腰部流量形状
    fusion = MultiplicativeFusion({
        "pctr": (1.0, 0.05, 1.0),
        "pevtr": (1.0, 0.20, 0.8),
        "pltr": (10.0, 0.50, 0.6),
        "pwatch_time_norm": (1.0, 0.10, 1.2),
        "ltr_reward": (1.0, 0.30, 1.0),
    })
    controller = CascadeTrafficController(
        store,
        quota={"new_video": 8, "campaign": 6},
        cat_floor={"知识": 6, "生活": 5, "体育": 4},
    )
    user = {
        "uid": 1001,
        "cat_pref": {"搞笑": 0.70, "娱乐": 0.55, "知识": 0.45,
                     "生活": 0.35, "体育": 0.20},
    }
    candidates = rng.sample(range(NUM_ITEMS), NUM_CANDIDATES)

    print("=" * 74)
    print("精排模块链路演示：粗排候选 300 → 精排输出 Top-60")
    print("=" * 74)
    module = RankingModule(store, model, ltr, fusion, controller, log)
    final = module.rank(user, candidates, TOP_K)

    print("\n" + "=" * 74)
    print("调控前后对比")
    print("=" * 74)
    before = log.category_share("before_control", store)
    after = log.category_share("after_control", store)
    print(f"{'类目':<8}{'调控前占比':>12}{'调控后占比':>12}{'变化':>10}")
    for cat in MockFeatureStore.CATEGORIES:
        b, a = before.get(cat, 0.0), after.get(cat, 0.0)
        print(f"{cat:<8}{b:>12.3f}{a:>12.3f}{a - b:>+10.3f}")

    before_ids = [r["item_id"] for r in log.records if r["stage"] == "before_control"]
    after_ids = [r["item_id"] for r in log.records if r["stage"] == "after_control"]
    for tag, ids, share in (("调控前", before_ids, before),
                            ("调控后", after_ids, after)):
        n_new = sum(1 for i in ids if store.is_new_video(i))
        n_camp = sum(1 for i in ids if store.get(i)["in_campaign"])
        n_small = sum(1 for i in ids if store.get(i)["author_fans"] < 1000)
        g = gini([share.get(c, 0.0) for c in MockFeatureStore.CATEGORIES])
        print(f"\n{tag}：新内容 {n_new}/{TOP_K}（配额 8），"
              f"活动商品 {n_camp}/{TOP_K}（配额 6），低粉作者 {n_small}/{TOP_K}")
        print(f"       覆盖类目数 = "
              f"{sum(1 for c in MockFeatureStore.CATEGORIES if share.get(c, 0) > 0)}/5"
              f"，类目分发基尼系数 = {g:.4f}")

    overlap = len(set(before_ids) & set(after_ids))
    print(f"\nTop-{TOP_K} 集合重合度：{overlap}/{TOP_K} = {overlap / TOP_K:.1%}"
          f"（调控换掉 {TOP_K - overlap} 个坑位换来生态约束达成，"
          f"其余坑位仍由模型收益决定）")
    print(f"全链路主模型调用次数：{model.call_count}")
