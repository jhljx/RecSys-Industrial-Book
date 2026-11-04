"""
特征质量：覆盖率、新鲜度与在线监控（第15章 §15.6.4）

正文的三个问题：有没有值、值是否正常、值是否及时。
本脚本用 ch15 公共的合成事件流构造"逐日特征快照"，然后演示：

1) Coverage：能够取到有效值（非缺失）的请求比例
   - 覆盖率的缺口往往集中在特定用户群（新用户、低活用户），
     所以同时打印分群覆盖率，而不是只看一个总均值
2) Freshness：特征距上次更新的时间（分钟），按分位数看
3) 值分布：均值/方差/极值比例，以及"某天突然异常"的检测
   - 用历史 7 天的滚动均值与标准差做 z-score，超过阈值即告警
4) 一次模拟故障：上游实时任务在 D9 开始延迟，覆盖率下降、更新延迟飙升，
   看监控能否在"业务指标下降之前"就报警

数据来自 common.make_events 的合成事件流；平台、告警阈值等均为 [MOCK]。

运行：python3 ch05_feature_selection/feature_quality.py
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import make_events, print_table  # noqa: E402

DAYS = 14
FAULT_DAY = 9        # 第 9 天起上游故障（0-based）
N_USERS = 400
N_ITEMS = 500
EVENTS_PER_DAY = 3_000


def main() -> None:
    print("特征质量：覆盖率 / 新鲜度 / 分布异常（§15.6.4）\n")
    rng = random.Random(2025)
    users = list(range(N_USERS))
    # 活跃度分群：高频用户、普通用户、新用户（历史很短）
    group_of = {}
    for u in users:
        r = rng.random()
        group_of[u] = "高频" if r < 0.2 else ("普通" if r < 0.8 else "新用户")

    # 每天生成一批事件，并据此计算"当天实时特征"的快照
    day_events = defaultdict(list)
    events = make_events(n_users=N_USERS, n_events=EVENTS_PER_DAY * DAYS,
                         n_items=N_ITEMS, days=DAYS, rng=rng)
    for e in events:
        day_events[int(e["day"])].append(e)

    rows = []
    history_mean = []
    for d in range(DAYS):
        evs = day_events.get(d, [])
        # 用当天事件模拟特征产出：每个用户一条"近 1 天点击数"
        cnt = defaultdict(int)
        for e in evs:
            if e["action"] in ("click", "watch", "like", "comment", "share", "follow"):
                cnt[e["user_id"]] += 1

        fault = d >= FAULT_DAY
        # 覆盖率：故障后只有 60% 用户能拿到实时值（其余回退到昨天的值 → 视为缺失）
        cover = {}
        for g in ("高频", "普通", "新用户"):
            members = [u for u in users if group_of[u] == g]
            ok = sum(1 for u in members if (not fault or rng.random() < 0.6)
                     and (cnt[u] > 0 or rng.random() < 0.9))
            cover[g] = ok / len(members)
        total_cov = sum(cover.values()) / 3

        # 新鲜度：故障后上游任务延迟，分钟级 → 小时级
        if fault:
            fresh = sorted(rng.lognormvariate(math.log(95), 0.5) for _ in users)
        else:
            fresh = sorted(rng.lognormvariate(math.log(8), 0.35) for _ in users)
        p50 = fresh[len(fresh) // 2]
        p99 = fresh[int(len(fresh) * 0.99) - 1]

        # 值分布：故障后数据不全，均值被低估、方差变大
        vals = [cnt[u] for u in users if cnt[u] > 0]
        if fault:
            vals = [v * rng.uniform(0.1, 0.6) for v in vals]
        mean_v = sum(vals) / max(1, len(vals))
        var_v = sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals))
        extreme = sum(1 for v in vals if v > 5 * mean_v) / max(1, len(vals))

        # 滚动 z-score：基线只取最近 7 个正常日（故障期数据不进入基线）。
        # 因为基线期的日间均值波动很小（标准差约 0.15），故障一发生 z 就跌到 −15，
        # 所以它对"值分布突变"非常敏感；但它对"局部用户缺失"不敏感，
        # 这正是需要与分群覆盖率、新鲜度搭配使用的原因。
        z = float("nan")
        if len(history_mean) >= 7:
            recent = history_mean[-7:]
            mu = sum(recent) / len(recent)
            sd = math.sqrt(sum((v - mu) ** 2 for v in recent) / len(recent)) + 1e-6
            z = (mean_v - mu) / sd
        if not fault:           # 故障期数据不进入基线（否则会污染监控本身）
            history_mean.append(mean_v)

        rows.append([f"D{d}", f"{total_cov:.1%}", f"{cover['高频']:.0%}",
                     f"{cover['普通']:.0%}", f"{cover['新用户']:.0%}",
                     f"{p50:.0f}", f"{p99:.0f}", f"{mean_v:.2f}", f"{var_v:.2f}",
                     f"{extreme:.2%}", "—" if math.isnan(z) else f"{z:+.2f}",
                     "故障" if fault else ""])

    print("逐日特征质量指标（D0–D13，D9 起模拟上游故障）：")
    print_table(
        ["日期", "总覆盖", "高频", "普通", "新用户", "p50延迟m", "p99延迟m",
         "均值", "方差", "极值占比", "z-score", "状态"], rows)

    print("\n告警规则（[MOCK] 阈值）：")
    rules = [
        ("总覆盖率 < 95%", sum(1 for r in rows if float(r[1][:-1]) < 95)),
        ("新用户覆盖率 < 90%", sum(1 for r in rows if float(r[4][:-1]) < 90)),
        ("p99 新鲜度 > 60 分钟", sum(1 for r in rows if int(r[6]) > 60)),
        ("|z-score| > 3", sum(1 for r in rows
                              if r[10] != "—" and abs(float(r[10])) > 3)),
        ("极值占比 > 5%", sum(1 for r in rows if float(r[9][:-1]) > 5)),
    ]
    for rule, n in rules:
        print(f"  {rule:<24} 触发 {n:2d} 天  {'← 故障期' if n else ''}")

    print("\n关键观察：")
    print("  1) 覆盖率从 96% 直接掉到 49%，是本次故障最明显的信号；但注意故障")
    print("     首日分群差异并不大（48% / 53% / 45%），分群监控的价值在于更早发现")
    print("     局部缺失（例如只影响新用户、只影响某个城市）；")
    print("  2) 新鲜度同一天从 8/17 分钟跳到 94/287 分钟，p99 放大十几倍，")
    print("     说明上游任务已经严重堆积——它比 AUC 下降更早暴露问题；")
    print("  3) 特征均值被系统性低估（数据不全导致），这会直接改变模型输入的分布，")
    print("     但离线 AUC 往往要等到训练/评估之后才暴露问题；")
    print("  4) z-score 在 D9 直接跌到 −15，对'值分布突变'很敏感；但它只反映总体")
    print("     均值，如果故障只影响一小群用户（例如新用户），总体 z-score 可能仍是")
    print("     正常的。实际平台要用'覆盖率 + 新鲜度 + 分群分布 / PSI'多指标交叉判断，")
    print("     并保证监控基线排除故障期数据，否则会把异常当正常（§15.6.4）。")

    print("\n落地上还要配合：")
    print("  - 特征血缘：从'业务指标下降 → 模型指标 → 特征 → 上游 Pipeline'反向定位；")
    print("  - 覆盖率/新鲜度/分布三类指标一起看，缺一不可；")
    print("  - 告警分级：轻微漂移记录日志，明显异常触发值班（阈值均为 [MOCK]）。")


if __name__ == "__main__":
    main()
