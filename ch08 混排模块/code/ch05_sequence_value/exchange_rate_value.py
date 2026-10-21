"""业务价值兑换关系(Exchange Rate)与统一价值空间。

对应正文第 8 章 §8.5「序列价值评估体系」中的核心机制:
不同业务的价值定义/统计口径/数值分布完全不同,因此业务分天然不可比;
工业界的做法不是简单 Min-Max / Z-Score 缩放,而是通过长期实验建立
**不同业务价值之间的兑换关系**,把一切折算到同一个价值单位(正文选「用户消费时长」)。

本实现做了什么:
1. 逐行复刻正文表「不同业务价值的兑换关系示例」,并由「平台收益 / 单位时长价值」
   自动推导兑换系数,验证正文给出的 20 / 10 / 2 三个系数;
2. 复刻正文表「统一价值计算示例」,把 Video-A/Ad-B/Live-C/Ecom-D 的原始价值
   折算为统一价值分 {3, 4, 5, 3};
3. 复刻正文的序列对比:Seq1=[Live-C, Ad-B, Video-A] 得 10.0,
   Seq2=[Ad-B, Live-C, Video-A] 得 9.8,因此混排选 Seq1;
4. 对照实验:Min-Max 归一化 / Z-Score 标准化 / 兑换关系三种口径,
   在合成请求上统计「选出的序列真实平台收益」,说明只有兑换关系口径
   优化的是平台整体价值(Platform Value)。

关键公式(全部取自正文):
    1 分钟时长价值 = 0.05 元  =>  1 元平台收益 = 20 分钟时长价值
    Value_ad   = 20 x Revenue
    Value_live = 10 x Gift
    Value_ec   =  2 x GMV
    Score_seq  = sum_{i=1..L} gamma_i * Value(item_i)
"""

import itertools
import random
import statistics

# ------------------------------------------------------------ 正文给定常量

MINUTE_VALUE_YUAN = 0.05          # 用户消费 1 分钟短视频 = 0.05 元平台收益

# 正文表「不同业务价值的兑换关系示例」:业务 -> (单位原始量描述, 平台收益/单位)
# 「平台收益」列的单位是元,除以 MINUTE_VALUE_YUAN 即得折算时长价值(分钟)
EXCHANGE_TABLE = [
    ("短视频 video", "用户消费 1 分钟", 0.05),
    ("广告 ad",      "广告收入 1 元",   1.00),
    ("直播 live",    "直播打赏 1 元",   0.50),
    ("电商 ec",      "电商 GMV 1 元",   0.10),
]

# 正文表「位置衰减权重示例」
POS_WEIGHT = [1.0, 0.8, 0.6, 0.4]


def exchange_rate(platform_revenue_per_unit, minute_value=MINUTE_VALUE_YUAN):
    """兑换系数 = 该业务单位原始量带来的平台收益 / 单位时长价值。"""
    return platform_revenue_per_unit / minute_value


def gamma(i):
    """位置衰减权重 gamma_i(i 从 1 开始);超出正文给定表则线性外推到 0。"""
    if i <= len(POS_WEIGHT):
        return POS_WEIGHT[i - 1]
    return max(0.0, POS_WEIGHT[-1] - 0.2 * (i - len(POS_WEIGHT)))


def score_seq(seq):
    """Score_seq = sum_i gamma_i * Value(item_i)。"""
    return sum(gamma(i + 1) * it["value"] for i, it in enumerate(seq))


# ------------------------------------------------------------ 演示 1:兑换关系表

def demo_exchange_table():
    print("=" * 78)
    print("【1】兑换关系推导(正文表「不同业务价值的兑换关系示例」)")
    print("=" * 78)
    print(f"统一价值单位:用户消费时长(分钟),1 分钟 = {MINUTE_VALUE_YUAN} 元平台收益")
    print(f"=> 1 元平台收益 = {1 / MINUTE_VALUE_YUAN:.0f} 分钟时长价值")
    print()
    print(f"{'业务':<14}{'用户行为':<18}{'平台收益(元)':>13}{'折算时长(分钟)':>15}"
          f"{'兑换系数':>10}")
    print("-" * 78)
    for biz, behavior, revenue in EXCHANGE_TABLE:
        r = exchange_rate(revenue)
        print(f"{biz:<14}{behavior:<18}{revenue:>13.2f}{r:>15.1f}{r:>10.1f}")
    print("-" * 78)
    print("与正文结论一致:Value_ad = 20 x Revenue,Value_live = 10 x Gift,"
          "Value_ec = 2 x GMV。")


# ------------------------------------------------------------ 演示 2:统一价值分

def textbook_items():
    """正文表「统一价值计算示例」的四个 Item(原始价值 -> 统一价值分)。"""
    return [
        {"id": "Video-A", "biz": "video", "raw_desc": "预计观看 3 分钟",
         "raw": 3.00, "value": 3.00 * exchange_rate(0.05)},
        {"id": "Ad-B", "biz": "ad", "raw_desc": "eCPM=0.20 元",
         "raw": 0.20, "value": 0.20 * exchange_rate(1.00)},
        {"id": "Live-C", "biz": "live", "raw_desc": "预计打赏 0.50 元",
         "raw": 0.50, "value": 0.50 * exchange_rate(0.50)},
        {"id": "Ecom-D", "biz": "ec", "raw_desc": "GMV=1.50 元",
         "raw": 1.50, "value": 1.50 * exchange_rate(0.10)},
    ]


def demo_unified_value():
    items = textbook_items()
    print()
    print("=" * 78)
    print("【2】统一价值映射(正文表「统一价值计算示例」)")
    print("=" * 78)
    print(f"{'Item':<10}{'业务':<8}{'原始价值':<18}{'兑换系数':>10}{'统一价值分':>12}")
    print("-" * 78)
    for it in items:
        rate = {"video": 1.0, "ad": 20.0, "live": 10.0, "ec": 2.0}[it["biz"]]
        print(f"{it['id']:<10}{it['biz']:<8}{it['raw_desc']:<18}"
              f"{rate:>10.1f}{it['value']:>12.2f}")
    print("-" * 78)
    vals = [round(it["value"], 2) for it in items]
    print(f"Value(item) = {{{', '.join(f'{v:.0f}' for v in vals)}}} "
          "-> 与正文一致,四个异质业务终于可直接比较。")
    return items


# ------------------------------------------------------------ 演示 3:序列对比

def demo_sequence_choice(items):
    by_id = {it["id"]: it for it in items}
    seq1 = [by_id["Live-C"], by_id["Ad-B"], by_id["Video-A"]]
    seq2 = [by_id["Ad-B"], by_id["Live-C"], by_id["Video-A"]]
    print()
    print("=" * 78)
    print("【3】序列价值对比(正文 Seq1 vs Seq2)")
    print("=" * 78)
    print(f"{'位置':<6}{'gamma_i':>9}  {'Seq1':<22}{'Seq2':<22}")
    print("-" * 78)
    for i in range(3):
        a, b = seq1[i], seq2[i]
        ca = f"{a['id']}(v={a['value']:.0f})"
        cb = f"{b['id']}(v={b['value']:.0f})"
        print(f"{i + 1:<6}{gamma(i + 1):>9.1f}  {ca:<22}{cb:<22}")
    print("-" * 78)
    s1, s2 = score_seq(seq1), score_seq(seq2)
    print(f"Score(Seq1) = 1.0x5 + 0.8x4 + 0.6x3 = {s1:.1f}")
    print(f"Score(Seq2) = 1.0x4 + 0.8x5 + 0.6x3 = {s2:.1f}")
    print(f"Score(Seq1) - Score(Seq2) = {s1 - s2:.1f} > 0 -> 混排选择 Seq1,与正文一致。")
    print("直觉:高价值 Item 放前排能吃到更大的 gamma_i,位置分配本身就是价值分配。")


# ------------------------------------------------------------ 演示 4:口径对照实验

def make_request(n_per_biz=2):
    """[MOCK] 合成一次混排请求的候选池,各业务的原始分尺度刻意差异巨大。

    video: 预计观看分钟数 ~ 1-6 分钟
    ad:    eCPM(元/次) ~ 0.05-0.45
    live:  预计打赏(元) ~ 0.0-1.2
    ec:    GMV(元)     ~ 0.0-8.0
    """
    pool = []
    for _ in range(n_per_biz):
        pool.append({"biz": "video", "raw": random.uniform(1.0, 6.0)})
        pool.append({"biz": "ad", "raw": random.uniform(0.05, 0.45)})
        pool.append({"biz": "live", "raw": random.uniform(0.0, 1.2)})
        pool.append({"biz": "ec", "raw": random.uniform(0.0, 8.0)})
    rate = {"video": 1.0, "ad": 20.0, "live": 10.0, "ec": 2.0}
    for it in pool:
        it["true_value"] = it["raw"] * rate[it["biz"]]   # 真实统一价值(评估用)
    return pool


def minmax_scores(pool):
    """Min-Max 归一化:只做尺度对齐,不解决口径问题。"""
    out = []
    for biz in ("video", "ad", "live", "ec"):
        xs = [it["raw"] for it in pool if it["biz"] == biz]
        lo, hi = min(xs), max(xs)
        span = hi - lo if hi > lo else 1.0
        for it in pool:
            if it["biz"] == biz:
                out.append((it, (it["raw"] - lo) / span))
    return dict((id(it), s) for it, s in out)


def zscore_scores(pool):
    """Z-Score 标准化:同样只是缩放。"""
    out = []
    for biz in ("video", "ad", "live", "ec"):
        xs = [it["raw"] for it in pool if it["biz"] == biz]
        mu = statistics.fmean(xs)
        sd = statistics.pstdev(xs) or 1.0
        for it in pool:
            if it["biz"] == biz:
                out.append((it, (it["raw"] - mu) / sd))
    return dict((id(it), s) for it, s in out)


def exchange_scores(pool):
    """兑换关系口径:折算到统一时长价值。"""
    rate = {"video": 1.0, "ad": 20.0, "live": 10.0, "ec": 2.0}
    return dict((id(it), it["raw"] * rate[it["biz"]]) for it in pool)


def best_seq_by(pool, table, length=4):
    """在候选池里枚举长度为 length 的序列,按给定打分口径取最优。"""
    best, best_s = None, float("-inf")
    for comb in itertools.permutations(pool, length):
        s = sum(gamma(i + 1) * table[id(it)] for i, it in enumerate(comb))
        if s > best_s:
            best, best_s = comb, s
    return best


def demo_caliber_comparison(n_trial=60):
    print()
    print("=" * 78)
    print("【4】对照实验:三种价值口径选出的序列,真实平台收益谁更高")
    print("=" * 78)
    gains = {"minmax": [], "zscore": [], "exchange": []}
    mix = {"minmax": [], "zscore": [], "exchange": []}
    for _ in range(n_trial):
        pool = make_request()
        truth = dict((id(it), it["true_value"]) for it in pool)
        for key, fn in (("minmax", minmax_scores), ("zscore", zscore_scores),
                        ("exchange", exchange_scores)):
            seq = best_seq_by(pool, fn(pool))
            gains[key].append(sum(gamma(i + 1) * truth[id(it)]
                                  for i, it in enumerate(seq)))
            mix[key].append(tuple(it["biz"] for it in seq))
    base = statistics.fmean(gains["exchange"])
    print(f"{'价值口径':<26}{'真实序列价值均值':>18}{'相对兑换关系':>14}")
    print("-" * 78)
    for key, name in (("minmax", "Min-Max 归一化"),
                      ("zscore", "Z-Score 标准化"),
                      ("exchange", "兑换关系(正文方案)")):
        m = statistics.fmean(gains[key])
        print(f"{name:<26}{m:>18.3f}{m / base:>14.4f}")
    print("-" * 78)
    print("首位业务分布(选出的 Top-1 坑位给了谁):")
    for key, name in (("minmax", "Min-Max"), ("zscore", "Z-Score"),
                      ("exchange", "兑换关系")):
        cnt = {}
        for seq in mix[key]:
            cnt[seq[0]] = cnt.get(seq[0], 0) + 1
        dist = "  ".join(f"{b}:{cnt.get(b, 0) / n_trial:.0%}"
                         for b in ("video", "ad", "live", "ec"))
        print(f"  {name:<12}{dist}")
    print("-" * 78)
    print("归一化口径把「各业务内部的相对名次」当成跨业务价值,等于默认四个业务同等重要,")
    print("于是常把低价值 Item 放到首位;兑换关系口径优化的是平台整体价值,因此最优。")
    print("这正是正文强调的:归一化只是缩放,没有解决价值口径不一致的问题。")


if __name__ == "__main__":
    random.seed(42)
    demo_exchange_table()
    items = demo_unified_value()
    demo_sequence_choice(items)
    demo_caliber_comparison()
