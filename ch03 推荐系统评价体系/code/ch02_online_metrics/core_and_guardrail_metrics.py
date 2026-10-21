"""对应正文第 3 章 §3.2「在线评价指标」与 §3.3.1 中「核心指标 / 诊断指标」的划分。

本实现做了什么
---------------
1. 从 `[MOCK]` 的曝光/行为日志出发，计算正文列举的**用户级核心指标
   （North Star Metrics）**：人均消费时长 Watch Time、人均播放次数、
   人均互动次数、互动率、次日/7 日留存 Retention、人均 GMV；
2. 计算正文列举的**诊断指标（Guardrail Metrics）**：内容多样性（类目熵、
   基尼系数、聚集度 HHI）、冷启动内容曝光率、负反馈 hate 率、系统延迟 P99、
   客户端崩溃率；
3. 演示正文反复强调的一个权衡：一个「只追 CTR」的策略可以把 CTR 推上去，
   却让人均时长、多样性、冷启动曝光率、hate 率全面恶化——这正是正文所说的
   「高 CTR 并不一定意味着用户真正喜欢内容（标题党）」。

关键公式
---------
    人均时长     = 总播放时长 / DAU
    人均播放     = 总播放次数 / DAU
    互动率       = (点赞+评论+分享+关注) / 曝光次数
    次日留存 D1  = |D0 活跃且 D1 仍活跃的用户| / |D0 活跃用户|
    类目熵       = -sum_c p_c * log2(p_c)，归一化熵 = H / log2(C)
    基尼系数     = 1 - 2 * sum_i (cum_share_i) * w_i   （按曝光量排序的洛伦兹曲线）
    HHI 聚集度   = sum_c p_c^2
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass


CATEGORIES = ["游戏", "美食", "搞笑", "知识", "音乐", "影视", "体育", "宠物"]


@dataclass
class Event:
    """[MOCK] 一条「曝光-反馈」合并记录（由 req_id 关联曝光日志与行为日志后得到）。"""

    user_id: int
    item_id: int
    category: str
    is_cold_start: bool          # 冷启动内容（发布时间短、曝光量低）
    clicked: bool
    watch_time_s: float
    interactions: int            # 点赞+评论+分享+关注
    hated: bool                  # 负反馈（不感兴趣/举报）
    gmv: float
    latency_ms: float
    crashed: bool


def simulate_day(n_users: int, imps_per_user: int, strategy: str) -> list[Event]:
    """[MOCK] 模拟一天的线上流量。

    strategy="balanced"  : 线上稳定版本（Base 组），多样性与消费深度平衡；
    strategy="ctr_only"  : 只优化 CTR 的激进策略（Exp 组），偏向标题党 + 热门头部内容。
    """
    events: list[Event] = []
    for u in range(n_users):
        for _ in range(imps_per_user):
            if strategy == "ctr_only":
                # 流量高度集中在 2 个热门类目，几乎不给冷启动内容机会
                cat = random.choices(CATEGORIES, weights=[40, 35, 6, 4, 4, 4, 4, 3])[0]
                cold = random.random() < 0.02
                item = random.randint(0, 800) if random.random() < 0.8 else random.randint(0, 50000)
                p_click = 0.16                 # 标题党点击高
                base_watch = 4.0               # 但看不久
                p_hate = 0.030
                p_inter = 0.012
            else:
                cat = random.choices(CATEGORIES, weights=[16, 14, 13, 12, 12, 11, 11, 11])[0]
                cold = random.random() < 0.12
                item = random.randint(0, 50000)
                p_click = 0.10
                base_watch = 16.0
                p_hate = 0.008
                p_inter = 0.030

            clicked = random.random() < p_click
            wt = max(0.0, random.gauss(base_watch, base_watch * 0.5)) if clicked else 0.0
            inter = sum(1 for _ in range(4) if random.random() < p_inter) if clicked else 0
            events.append(Event(
                user_id=u, item_id=item, category=cat, is_cold_start=cold,
                clicked=clicked, watch_time_s=wt, interactions=inter,
                hated=random.random() < p_hate, gmv=(random.expovariate(1 / 3.5) if clicked and random.random() < 0.03 else 0.0),
                latency_ms=random.lognormvariate(math.log(28.0), 0.45 if strategy == "balanced" else 0.62),
                crashed=random.random() < (0.0004 if strategy == "balanced" else 0.0011),
            ))
    return events


def simulate_retention(events: list[Event], n_users: int, base_retention: float):
    """[MOCK] 基于当日消费深度生成次日/7 日留存。

    留存概率 = base_retention + 消费深度带来的增量，体现「人均时长影响留存」的业务直觉。
    返回 (d1_retention, d7_retention)。
    """
    per_user_watch: dict[int, float] = defaultdict(float)
    for e in events:
        per_user_watch[e.user_id] += e.watch_time_s
    d1 = d7 = 0
    for u in range(n_users):
        w = per_user_watch.get(u, 0.0)
        lift = min(0.25, w / 600.0)                  # 10 分钟消费 -> +25% 留存上限
        p1 = min(0.95, base_retention + lift)
        if random.random() < p1:
            d1 += 1
            if random.random() < p1 * 0.72:          # D7 在 D1 基础上再衰减
                d7 += 1
    return d1 / n_users, d7 / n_users


# --------------------------------------------------------------------------
# 核心指标
# --------------------------------------------------------------------------
def core_metrics(events: list[Event], n_users: int) -> dict[str, float]:
    imps = len(events)
    clicks = sum(e.clicked for e in events)
    watch = sum(e.watch_time_s for e in events)
    inter = sum(e.interactions for e in events)
    gmv = sum(e.gmv for e in events)
    return {
        "DAU": float(n_users),
        "CTR(%)": 100.0 * clicks / imps,
        "人均时长(s)": watch / n_users,
        "人均播放次数": clicks / n_users,
        "人均互动次数": inter / n_users,
        "互动率(%)": 100.0 * inter / imps,
        "人均GMV(元)": gmv / n_users,
        "_imps": float(imps),
        "_clicks": float(clicks),
    }


# --------------------------------------------------------------------------
# 诊断指标
# --------------------------------------------------------------------------
def shannon_entropy(counts: Counter) -> tuple[float, float]:
    """返回 (熵 bit, 归一化熵)。归一化熵 = H / log2(类别数)，取值 [0,1]。"""
    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h, h / math.log2(len(counts)) if len(counts) > 1 else 0.0


def gini_coefficient(counts: Counter) -> float:
    """曝光量在 item 上的基尼系数：0 = 完全均匀，接近 1 = 极度集中。"""
    vals = sorted(counts.values())
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    cum = 0.0
    weighted = 0.0
    for i, v in enumerate(vals, start=1):
        cum += v
        weighted += cum
    # 等价于 1 - 2 * (洛伦兹曲线下面积)
    return 1.0 - 2.0 * (weighted / (n * total)) + 1.0 / n


def hhi(counts: Counter) -> float:
    """赫芬达尔指数（聚集度）= sum p_c^2，越大越集中。均匀分布下 = 1/C。"""
    total = sum(counts.values())
    return sum((c / total) ** 2 for c in counts.values())


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数，q in [0,1]。"""
    if not values:
        return 0.0
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def guardrail_metrics(events: list[Event]) -> dict[str, float]:
    imps = len(events)
    cats = Counter(e.category for e in events)
    items = Counter(e.item_id for e in events)
    lat = [e.latency_ms for e in events]
    h, h_norm = shannon_entropy(cats)
    return {
        "类目熵(bit)": h,
        "归一化类目熵": h_norm,
        "类目HHI聚集度": hhi(cats),
        "item基尼系数": gini_coefficient(items),
        "去重item数": float(len(items)),
        "冷启动曝光率(%)": 100.0 * sum(e.is_cold_start for e in events) / imps,
        "负反馈hate率(%)": 100.0 * sum(e.hated for e in events) / imps,
        "平均延迟(ms)": sum(lat) / len(lat),
        "P99延迟(ms)": percentile(lat, 0.99),
        "超时率(%,>100ms)": 100.0 * sum(1 for x in lat if x > 100.0) / imps,
        "崩溃率(%)": 100.0 * sum(e.crashed for e in events) / imps,
    }


def main() -> None:
    random.seed(42)

    print("=" * 80)
    print("§3.2 + §3.3.1：核心指标（North Star） vs 诊断指标（Guardrail）")
    print("=" * 80)

    n_users, imps_per_user = 4000, 25
    base_ev = simulate_day(n_users, imps_per_user, "balanced")
    exp_ev = simulate_day(n_users, imps_per_user, "ctr_only")

    base_core = core_metrics(base_ev, n_users)
    exp_core = core_metrics(exp_ev, n_users)
    b_d1, b_d7 = simulate_retention(base_ev, n_users, base_retention=0.42)
    e_d1, e_d7 = simulate_retention(exp_ev, n_users, base_retention=0.42)
    base_core["次日留存D1(%)"] = 100.0 * b_d1
    base_core["7日留存D7(%)"] = 100.0 * b_d7
    exp_core["次日留存D1(%)"] = 100.0 * e_d1
    exp_core["7日留存D7(%)"] = 100.0 * e_d7

    print(f"\n[MOCK 流量] Base 组 / Exp 组各 {n_users} 用户 x {imps_per_user} 次曝光 "
          f"= {len(base_ev)} 条曝光")
    print("  Base 组 = 当前线上稳定版本（多目标平衡）")
    print("  Exp  组 = 只优化 CTR 的激进策略（偏向标题党与头部热门内容）")

    print("\n【核心指标 North Star Metrics】")
    print(f"  {'指标':>16} | {'Base':>11} | {'Exp':>11} | {'相对变化':>10}")
    print("  " + "-" * 56)
    keys_core = ["CTR(%)", "人均时长(s)", "人均播放次数", "人均互动次数",
                 "互动率(%)", "人均GMV(元)", "次日留存D1(%)", "7日留存D7(%)"]
    for k in keys_core:
        b, e = base_core[k], exp_core[k]
        d = (e - b) / b * 100.0 if b else float("nan")
        flag = "↑" if d > 0 else "↓"
        print(f"  {k:>16} | {b:>11.4f} | {e:>11.4f} | {d:>+9.2f}% {flag}")

    base_g = guardrail_metrics(base_ev)
    exp_g = guardrail_metrics(exp_ev)
    print("\n【诊断指标 Guardrail Metrics】")
    print(f"  {'指标':>18} | {'Base':>11} | {'Exp':>11} | {'相对变化':>10}")
    print("  " + "-" * 58)
    for k in base_g:
        b, e = base_g[k], exp_g[k]
        d = (e - b) / b * 100.0 if b else float("nan")
        flag = "↑" if d > 0 else "↓"
        print(f"  {k:>18} | {b:>11.4f} | {e:>11.4f} | {d:>+9.2f}% {flag}")

    ctr_lift = (exp_core["CTR(%)"] - base_core["CTR(%)"]) / base_core["CTR(%)"] * 100
    wt_lift = (exp_core["人均时长(s)"] - base_core["人均时长(s)"]) / base_core["人均时长(s)"] * 100
    print(f"\n  >>> Exp 组 CTR 提升 {ctr_lift:+.2f}%，但人均时长 {wt_lift:+.2f}%、"
          f"互动率 "
          f"{(exp_core['互动率(%)']-base_core['互动率(%)'])/base_core['互动率(%)']*100:+.2f}%、"
          f"次日留存 "
          f"{(exp_core['次日留存D1(%)']-base_core['次日留存D1(%)'])/base_core['次日留存D1(%)']*100:+.2f}%。")
    print("  >>> 诊断指标同时报警：归一化类目熵下降、HHI 聚集度上升、item 基尼系数上升、")
    print("      冷启动曝光率骤降、hate 率上升、P99 延迟与崩溃率恶化。")
    print("  >>> 正文结论：高 CTR 并不一定意味着用户真正喜欢内容（标题党问题）。")
    print("      这正是 Watch Time、EVR、LVR 等消费深度指标取代 CTR 成为核心目标，")
    print("      以及 A/B 实验必须同时看核心指标与诊断指标的原因（§3.3.1）。")
    print("  >>> 按正文 §3.3.4 的扩量规则：Exp 组虽然 CTR 显著正向，")
    print("      但诊断指标明显恶化，不应进入 5% -> 10% 的扩量流程。")

    # ---------- 人均时长的分布，不只看均值 ----------
    print("\n[不要只看均值] 人均时长的分位数分布：")
    for name, ev in (("Base", base_ev), ("Exp", exp_ev)):
        per_user: dict[int, float] = defaultdict(float)
        for e in ev:
            per_user[e.user_id] += e.watch_time_s
        vals = [per_user.get(u, 0.0) for u in range(n_users)]
        qs = [percentile(vals, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)]
        print(f"  {name:>4}: mean={sum(vals)/len(vals):7.2f}s  "
              + "  ".join(f"P{int(q*100)}={v:.1f}s"
                          for q, v in zip((0.10, 0.25, 0.50, 0.75, 0.90, 0.99), qs)))
    print("  >>> 均值易被重度用户拉动，上线前应同时检查中位数与分位数分布。")


if __name__ == "__main__":
    main()
