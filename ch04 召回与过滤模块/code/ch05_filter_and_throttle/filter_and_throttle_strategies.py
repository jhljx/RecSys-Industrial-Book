"""
对应正文第 4 章 §4.5.2「常见过滤与限流策略」(表「过滤限流策略示例」filter_scope)。

正文口径:过滤与限流策略来源于内容安全、用户体验、生态治理、业务规则等多个维度。
正文表中给出的 7 个维度与对应策略原样实现如下:
    封面/画面合规   -> 随机过滤 80% 的请求
    标题/文本违规   -> 随机过滤 80% 的请求
    内容主题风险     -> 100% 完全过滤,内容仅创作者可见
    内容质量低下     -> 随机过滤 75% 的请求,总曝光不能超过 X 次
    原创性存疑       -> 全域随机过滤 60% 的请求
    用户行为反馈     -> 不同活跃度人群不同的随机过滤比例
    发布者信誉       -> 大流量公域页面 100% 完全过滤,小流量页面随机 50% 的请求
正文还强调:「相比完全过滤,限流能够在降低风险的同时减少误伤」,即限流 = 概率性
降低曝光配额,而不是硬删除。

本脚本实现一个可运行的策略引擎:
  1) RandomDropRule      随机过滤 r 比例的请求(按 request_id 哈希做稳定随机,
                          同一请求重复评估结果一致,这是工业界灰度/限流的常规做法);
  2) HardFilterRule      100% 过滤(仅创作者可见:作者本人请求放过);
  3) ExposureCapRule     随机过滤 + 总曝光上限 X 次(正文「总曝光不能超过X次」);
  4) AudienceRateRule    按用户活跃度分层给不同过滤比例(正文「不同活跃度人群不同比例」);
  5) PageAwareRule       按页面流量大小给不同策略(公域 100% / 小流量 50%);
  6) 频控类限流(正文 §4.5.2 的限流机制在工业界最常见的落地形态):
       - 作者频控:一刷内同一作者最多 n 条;
       - 类目频控:一刷内同一类目最多占 q 比例;
       - 疲劳度:用户对某作者/类目近 d 天曝光未点击累计 >= f 次则降权/过滤;
       - 滑动时间窗限流:单内容在 T 秒内全站曝光不超过 C 次(令牌桶 + 滑动窗两种)。

全部策略在一条合成请求流上评估,打印每条规则的命中量、过滤量、实际过滤比例
(用以验证「随机 80%」真的落在 80% 附近)以及最终候选规模变化。

依赖:仅 Python 3 标准库。[MOCK] 风险标签、曝光计数器、时间戳都是进程内合成数据,
不读取任何真实数据集,也不起真实服务。
"""

import hashlib
import random
from collections import Counter, defaultdict, deque

N_ITEMS = 800               # 教学规模的物品库
N_AUTHORS = 40
N_CATS = 8
N_USERS = 40                # 用户少一些,疲劳度才能在多刷之间累积
N_REQUESTS = 600            # 模拟 600 次推荐请求
PER_REQUEST_CAND = 60       # 每次请求的召回候选数(一刷)
HEAD_SIZE = 150             # 头部热门池大小(召回天然偏头部,限流才有意义)
HEAD_PROB = 0.65            # 候选落在头部池的概率


# ============================================================ 稳定随机
def stable_rand(*parts) -> float:
    """用哈希把 (请求, 物品, 规则) 映射到 [0,1),保证同一组合判定可复现。

    工业界做随机过滤时不用 random(),因为同一请求重试/多副本要给一致结果,
    否则用户刷新一次就能绕过限流。
    """
    key = "|".join(str(p) for p in parts).encode()
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


# ============================================================ 规则基类与实现
class Rule:
    name = "rule"
    dim = "-"
    policy = "-"

    def applies(self, item) -> bool:
        raise NotImplementedError

    def drop(self, item, ctx) -> bool:
        raise NotImplementedError


class RandomDropRule(Rule):
    """随机过滤 ratio 比例的请求。正文:封面/画面合规、标题/文本违规 -> 80%。"""

    def __init__(self, name, dim, tag, ratio):
        self.name, self.dim, self.tag, self.ratio = name, dim, tag, ratio
        self.policy = f"随机过滤{ratio:.0%}"

    def applies(self, item):
        return item["tags"].get(self.tag, False)

    def drop(self, item, ctx):
        return stable_rand(ctx["req_id"], item["id"], self.name) < self.ratio


class HardFilterRule(Rule):
    """100% 完全过滤,内容仅创作者可见。正文:内容主题风险。"""

    def __init__(self, name, dim, tag):
        self.name, self.dim, self.tag = name, dim, tag
        self.policy = "100%过滤(仅创作者可见)"

    def applies(self, item):
        return item["tags"].get(self.tag, False)

    def drop(self, item, ctx):
        return item["author"] != ctx["uid"]        # 作者本人可见


class ExposureCapRule(Rule):
    """随机过滤 ratio + 总曝光不超过 cap 次。正文:内容质量低下 -> 75% 且总曝光<=X。"""

    def __init__(self, name, dim, tag, ratio, cap):
        self.name, self.dim, self.tag = name, dim, tag
        self.ratio, self.cap = ratio, cap
        self.policy = f"随机过滤{ratio:.0%}+总曝光<={cap}"
        self.exposed = Counter()                   # [MOCK] 全站曝光计数器
        self.cap_hits = 0

    def applies(self, item):
        return item["tags"].get(self.tag, False)

    def drop(self, item, ctx):
        if self.exposed[item["id"]] >= self.cap:
            self.cap_hits += 1
            return True
        return stable_rand(ctx["req_id"], item["id"], self.name) < self.ratio

    def on_expose(self, item_id):
        self.exposed[item_id] += 1


class AudienceRateRule(Rule):
    """按用户活跃度分层的过滤比例。正文:用户行为反馈 -> 不同活跃度人群不同比例。"""

    def __init__(self, name, dim, tag, ratios):
        self.name, self.dim, self.tag, self.ratios = name, dim, tag, ratios
        self.policy = "分层比例 " + "/".join(
            f"{k}:{v:.0%}" for k, v in ratios.items())

    def applies(self, item):
        return item["tags"].get(self.tag, False)

    def drop(self, item, ctx):
        r = self.ratios[ctx["activity"]]
        return stable_rand(ctx["req_id"], item["id"], self.name) < r


class PageAwareRule(Rule):
    """按页面流量大小区分策略。正文:发布者信誉 -> 公域 100% / 小流量 50%。"""

    def __init__(self, name, dim, tag, page_ratios):
        self.name, self.dim, self.tag = name, dim, tag
        self.page_ratios = page_ratios
        self.policy = " / ".join(f"{k}:{v:.0%}" for k, v in page_ratios.items())

    def applies(self, item):
        return item["tags"].get(self.tag, False)

    def drop(self, item, ctx):
        r = self.page_ratios[ctx["page"]]
        return stable_rand(ctx["req_id"], item["id"], self.name) < r


# ============================================================ 频控 / 疲劳 / 时间窗
def author_quota_filter(cands, items, max_per_author):
    """作者频控:一刷内同一作者最多 max_per_author 条。"""
    cnt = Counter()
    out, dropped = [], 0
    for it in cands:
        a = items[it]["author"]
        if cnt[a] >= max_per_author:
            dropped += 1
            continue
        cnt[a] += 1
        out.append(it)
    return out, dropped


def category_quota_filter(cands, items, max_ratio):
    """类目频控:一刷内同一类目占比不超过 max_ratio。"""
    cap = max(1, int(len(cands) * max_ratio))
    cnt = Counter()
    out, dropped = [], 0
    for it in cands:
        c = items[it]["cat"]
        if cnt[c] >= cap:
            dropped += 1
            continue
        cnt[c] += 1
        out.append(it)
    return out, dropped


class FatigueFilter:
    """疲劳度过滤:用户对某作者/类目「曝光未点击」累计达到阈值则不再分发。

    fatigue(u, key) = Σ_t 1[曝光] · decay^(Δ刷次) - w_click · Σ_t 1[点击] · decay^(Δ刷次)
    decay 按「该用户自己的刷次间隔」做指数衰减(正文 §4.5.3 提到负向信号需随时间衰减);
    衰减是惰性的 -- 只在读写该 key 时按 Δ 补算,避免每次请求遍历全量计数器,
    这也是工业界疲劳度服务的常规实现(值 + 时间戳,读时衰减)。
    """

    def __init__(self, threshold, decay=0.85, w_click=3.0):
        self.threshold = threshold
        self.decay = decay
        self.w_click = w_click
        self.score = {}                 # (uid, key) -> [值, 该用户的刷次序号]

    def _get(self, uid, key, step) -> float:
        rec = self.score.get((uid, key))
        if rec is None:
            return 0.0
        v = rec[0] * self.decay ** (step - rec[1])
        rec[0], rec[1] = v, step
        return v

    def is_fatigued(self, uid, key, step) -> bool:
        return self._get(uid, key, step) >= self.threshold

    def on_expose(self, uid, key, step, clicked: bool):
        v = self._get(uid, key, step)
        v += -self.w_click if clicked else 1.0
        self.score[(uid, key)] = [max(v, 0.0), step]

    def size(self) -> int:
        return len(self.score)


class SlidingWindowLimiter:
    """滑动时间窗限流:单物品在 window 秒内全站曝光不超过 cap 次。"""

    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.q = defaultdict(deque)

    def allow(self, item_id, now) -> bool:
        dq = self.q[item_id]
        while dq and now - dq[0] > self.window:
            dq.popleft()
        if len(dq) >= self.cap:
            return False
        dq.append(now)
        return True


class TokenBucketLimiter:
    """令牌桶限流:速率 rate 个/秒,桶容量 burst,允许短时突发。"""

    def __init__(self, rate, burst):
        self.rate, self.burst = rate, burst
        self.tokens = defaultdict(lambda: float(burst))
        self.last = {}

    def allow(self, item_id, now) -> bool:
        t0 = self.last.get(item_id, now)
        self.tokens[item_id] = min(self.burst,
                                   self.tokens[item_id] + (now - t0) * self.rate)
        self.last[item_id] = now
        if self.tokens[item_id] >= 1.0:
            self.tokens[item_id] -= 1.0
            return True
        return False


# ============================================================ 合成数据
def gen_items(rng):
    """[MOCK] 给每个物品随机挂上正文表里的 7 类风险标签。"""
    rates = {
        "cover_risk": 0.05, "text_risk": 0.05, "topic_risk": 0.01,
        "low_quality": 0.12, "originality_risk": 0.08,
        "neg_feedback": 0.07, "bad_publisher": 0.03,
    }
    items = {}
    for i in range(N_ITEMS):
        items[i] = {
            "id": i,
            "author": rng.randrange(N_AUTHORS),
            "cat": rng.randrange(N_CATS),
            "tags": {t: rng.random() < p for t, p in rates.items()},
            "ctr": rng.betavariate(2, 40),
        }
    return items


def build_rules():
    """严格按正文表 filter_scope 的 7 行构造规则。"""
    return [
        RandomDropRule("封面/画面合规", "内容安全", "cover_risk", 0.80),
        RandomDropRule("标题/文本违规", "内容安全", "text_risk", 0.80),
        HardFilterRule("内容主题风险", "内容安全", "topic_risk"),
        ExposureCapRule("内容质量低下", "用户体验", "low_quality", 0.75, cap=8),
        RandomDropRule("原创性存疑", "生态治理", "originality_risk", 0.60),
        AudienceRateRule("用户行为反馈", "用户体验", "neg_feedback",
                         {"高活": 0.30, "中活": 0.60, "低活": 0.90}),
        PageAwareRule("发布者信誉", "生态治理", "bad_publisher",
                      {"公域大流量": 1.00, "小流量": 0.50}),
    ]


def main():
    random.seed(42)
    rng = random.Random(42)

    items = gen_items(rng)
    rules = build_rules()

    print("=" * 96)
    print("§4.5.2 常见过滤与限流策略:正文表「过滤限流策略示例」的可运行实现")
    print("=" * 96)
    print(f"\n合成物品库 {N_ITEMS} 个 / 作者 {N_AUTHORS} 个 / 类目 {N_CATS} 个 / "
          f"用户 {N_USERS} 个;模拟 {N_REQUESTS} 次请求,"
          f"每次召回 {PER_REQUEST_CAND} 个候选(65% 落在 150 个热门物品内)")

    print(f"\n[规则表] {'规则(正文维度)':<16}{'维度':<10}{'正文策略':<28}{'标签占比':>10}")
    print("-" * 70)
    for r in rules:
        share = sum(1 for it in items.values() if r.applies(it)) / N_ITEMS
        print(f"{r.name:<16}{r.dim:<10}{r.policy:<28}{share:>10.2%}")

    # ---------------- 主循环:逐请求跑过滤 + 频控 + 疲劳 + 时间窗
    fatigue = FatigueFilter(threshold=4.0)
    sw = SlidingWindowLimiter(window=60.0, cap=15)
    tb = TokenBucketLimiter(rate=0.1, burst=3)
    cap_rule = next(r for r in rules if isinstance(r, ExposureCapRule))

    hit = Counter()
    drop = Counter()
    stage = Counter()
    sw_drop = tb_drop = fat_drop = author_drop = cat_drop = 0
    user_step = Counter()               # 每个用户自己的刷次序号,驱动疲劳度衰减
    now = 0.0

    for req in range(N_REQUESTS):
        uid = rng.randrange(N_USERS)
        user_step[uid] += 1
        step = user_step[uid]
        ctx = {
            "req_id": req, "uid": uid,
            "activity": rng.choice(["高活", "中活", "低活"]),
            "page": "公域大流量" if rng.random() < 0.7 else "小流量",
        }
        # 召回天然偏头部:65% 的候选来自 150 个热门物品
        cands = []
        seen = set()
        while len(cands) < PER_REQUEST_CAND:
            it = (rng.randrange(HEAD_SIZE) if rng.random() < HEAD_PROB
                  else rng.randrange(N_ITEMS))
            if it not in seen:
                seen.add(it)
                cands.append(it)
        stage["召回候选"] += len(cands)

        # 1) 正文表规则
        kept = []
        for it in cands:
            item = items[it]
            blocked = False
            for r in rules:
                if not r.applies(item):
                    continue
                hit[r.name] += 1
                if r.drop(item, ctx):
                    drop[r.name] += 1
                    blocked = True
                    break
            if not blocked:
                kept.append(it)
        stage["规则过滤后"] += len(kept)

        # 2) 疲劳度
        kept2 = []
        for it in kept:
            if fatigue.is_fatigued(uid, f"a{items[it]['author']}", step):
                fat_drop += 1
                continue
            kept2.append(it)
        stage["疲劳度过滤后"] += len(kept2)

        # 3) 作者 / 类目频控
        kept3, d_a = author_quota_filter(kept2, items, max_per_author=2)
        kept4, d_c = category_quota_filter(kept3, items, max_ratio=0.25)
        author_drop += d_a
        cat_drop += d_c
        stage["作者频控后"] += len(kept3)
        stage["类目频控后"] += len(kept4)

        # 4) 单物品全站限流(滑动窗 + 令牌桶)
        final = []
        for it in kept4:
            now += 0.01
            if not sw.allow(it, now):
                sw_drop += 1
                continue
            if not tb.allow(it, now):
                tb_drop += 1
                continue
            final.append(it)
        stage["限流后(最终曝光)"] += len(final)

        # 5) 回写曝光/点击,驱动疲劳与曝光上限
        for it in final:
            clicked = rng.random() < items[it]["ctr"]
            fatigue.on_expose(uid, f"a{items[it]['author']}", step, clicked)
            cap_rule.on_expose(it)

    print("\n[规则命中与实际过滤比例] 验证「随机过滤 X%」是否真的落在 X% 附近")
    print(f"{'规则':<16}{'命中次数':>10}{'过滤次数':>10}{'实际过滤比例':>14}"
          f"{'正文目标':>12}")
    print("-" * 64)
    target = {"封面/画面合规": "80%", "标题/文本违规": "80%", "内容主题风险": "100%",
              "内容质量低下": "75%+上限", "原创性存疑": "60%",
              "用户行为反馈": "30/60/90%", "发布者信誉": "100%/50%"}
    for r in rules:
        h = hit[r.name]
        d = drop[r.name]
        print(f"{r.name:<16}{h:>10d}{d:>10d}{d / max(h, 1):>14.2%}"
              f"{target[r.name]:>12}")
    print(f"  其中「内容质量低下」有 {cap_rule.cap_hits} 次是撞到总曝光上限 "
          f"{cap_rule.cap} 被拦(而非随机过滤)")

    order = ["召回候选", "规则过滤后", "疲劳度过滤后", "作者频控后",
             "类目频控后", "限流后(最终曝光)"]
    print("\n[链路漏斗] 各阶段累计候选量(600 次请求合计)")
    print(f"{'阶段':<20}{'累计候选':>10}{'相对上一阶段':>14}{'相对召回':>12}")
    print("-" * 58)
    base = stage["召回候选"]
    prev = base
    for s in order:
        v = stage[s]
        print(f"{s:<20}{v:>10d}{v / max(prev, 1):>14.2%}{v / base:>12.2%}")
        prev = v
    print(f"  疲劳度拦截 {fat_drop} 次(活跃疲劳计数器 {fatigue.size()} 个),"
          f"作者频控拦截 {author_drop} 次,类目频控拦截 {cat_drop} 次,")
    print(f"  滑动窗限流拦截 {sw_drop} 次,令牌桶限流拦截 {tb_drop} 次")

    # ---------------- 硬过滤 vs 软限流的误伤对比(正文 §4.5.2 结尾的论点)
    print("\n[硬过滤 vs 软限流] 正文:「相比完全过滤,限流能在降低风险的同时减少误伤」")
    risky = [it for it in items.values() if it["tags"]["low_quality"]]
    # [MOCK] 假设风险标签准确率 70%,即 30% 是误伤的优质内容
    mislabeled = {it["id"] for it in risky
                  if stable_rand("label", it["id"]) < 0.30}
    n_good = len(mislabeled)
    n_risk = len(risky) - n_good
    trials = 200                      # 每个物品模拟 200 次请求,统计曝光存活比例
    print(f"{'策略':<20}{'风险内容残留曝光率':>20}{'优质内容被误杀率':>20}")
    print("-" * 60)
    for label, ratio in [("硬过滤(100%)", 1.00), ("软限流(75%)", 0.75),
                         ("软限流(50%)", 0.50), ("不处理(0%)", 0.00)]:
        kept_risk = kept_good = 0
        for it in risky:
            live = sum(1 for t in range(trials)
                       if stable_rand("eval", it["id"], ratio, t) >= ratio)
            if it["id"] in mislabeled:
                kept_good += live
            else:
                kept_risk += live
        residual = kept_risk / (n_risk * trials)
        hurt = 1.0 - kept_good / (n_good * trials)
        print(f"{label:<20}{residual:>20.1%}{hurt:>20.1%}")
    print(f"  (标签池 {len(risky)} 个低质内容,其中 {n_good} 个是 "
          f"[MOCK] 误标的优质内容;每个内容模拟 {trials} 次请求求均值)")
    print("  硬过滤把风险残留压到 0,代价是误标的优质内容 100% 被封死;")
    print("  限流让风险与误伤同比例下降,给了「错标内容仍有机会被验证」的空间 -- ")
    print("  这正是正文说「限流能在降低风险的同时减少误伤」的含义。")

    print("\n结论(正文 §4.5.2 的工程含义):")
    print("  1) 随机过滤必须用「稳定随机」(请求 ID + 物品 ID 哈希),否则用户下拉刷新")
    print("     几次就能把被限流的内容刷出来,策略形同虚设;")
    print("  2) 过滤链路是串联漏斗,上游比例乘起来衰减很快,")
    print("     本脚本 600 次请求的候选从 36000 衰减到最终曝光,需要和召回规模联动设计;")
    print("  3) 单物品全站限流同时需要滑动窗(控总量)与令牌桶(控速率、允许突发);")
    print("  4) 限流比例是「风险残留」与「误伤优质内容」之间的直接权衡,")
    print("     对照表说明为什么正文中绝大多数维度用随机比例而非 100% 硬过滤。")


if __name__ == "__main__":
    main()
