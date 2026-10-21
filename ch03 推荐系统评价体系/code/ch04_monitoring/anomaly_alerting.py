"""对应正文第 3 章 §3.4「推荐系统监控体系」里的异常发现环节。

正文强调：「监控体系的核心目标并非简单记录数据，而是帮助工程师及时发现问题、
快速定位问题并持续优化系统。」正文列出的监控量既有均值类（CTR、平均响应时间），
也有分位数类（P50/P90/P99），还要求能识别「模型发布后 pCTR 整体抬高 50%」这类
分布漂移。本脚本实现的就是从原始指标流到分级告警的那一段。

本实现做了什么
---------------
1. `[MOCK]` 生成 3 天、每 5 分钟一个窗口（每天 288 个窗口）的三条指标流：
   CTR（有明显日内周期）、P99 延迟、pCTR 均值；并注入三类异常：
   - 突降：CTR 在第 3 天某时刻被打掉 18%（典型策略/数据事故）；
   - 尖刺：P99 延迟连续 4 个窗口放大到 3 倍（容量抖动）；
   - 慢漂移：pCTR 均值在第 3 天后半段以每窗口 +0.35% 的速度缓慢抬升
     （对应正文「模型输出值分布漂移」，阈值类告警抓不到，需要同比）。
2. 三种检测量，都是工业监控盘的标配：
   - 环比（WoW, window-over-window）：r_t = (x_t - x_{t-1}) / x_{t-1}；
   - 同比（DoD, day-over-day）：y_t = (x_t - x_{t-S}) / x_{t-S}，S = 288 个窗口
     为一个日周期，用于抵消日内波动；
   - 季节性残差的稳健 z 分数：先取同一时刻在历史各天的中位数作为季节基线
     b_t = median{x_{t-S}, x_{t-2S}, ...}，残差 e_t = x_t - b_t，再用近期残差的
     MAD 做标准化
         z_t = e_t / (1.4826 * MAD(E))，E 为最近若干窗口的残差集合，
     其中 1.4826 = 1 / Phi^{-1}(0.75) 使 MAD 在正态下成为标准差的一致估计。
     用中位数 / MAD 而不是均值 / 标准差，是因为故障样本本身会污染均值与方差。
3. 分级告警：按 |同比| 幅度定 P0/P1/P2，并加两条工业界必备的降噪规则：
   - 持续确认：必须连续 `CONFIRM` 个窗口命中才升级为告警（压掉单点抖动）；
   - 冷却抑制：同一指标在 `COOLDOWN` 个窗口内不重复告警（压掉告警风暴）。
4. 打印指标流摘要表、逐条告警表，以及「只用阈值 / 只用环比 / 同比+稳健 z」
   三种策略在同一数据流上的检出对比，说明慢漂移为什么必须靠同比。

全部指标流为进程内 `[MOCK]` 合成数据，不读外部监控系统、不联网。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

SLOTS_PER_DAY = 288          # 5 分钟一个窗口
DAYS = 3
SEASON = SLOTS_PER_DAY       # 同比周期 S
MAD_WINDOW = 36              # 稳健 z 分数使用最近 36 个残差（3 小时）
CONFIRM = 2                  # 连续命中窗口数
COOLDOWN = 24                # 告警冷却窗口数（2 小时）


# ==========================================================================
# 统计工具
# ==========================================================================
def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mad(xs: list[float]) -> float:
    """中位绝对偏差 MAD = median(|x_i - median(x)|)。"""
    if not xs:
        return float("nan")
    m = median(xs)
    return median([abs(x - m) for x in xs])


def quantile(xs: list[float], q: float) -> float:
    """线性插值分位数，用于打印 P50/P90/P99（正文要求的分位数监控）。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# ==========================================================================
# [MOCK] 指标流
# ==========================================================================
def diurnal(slot: int) -> float:
    """日内周期因子：早高峰 / 午间 / 晚高峰三峰，夜间低谷。"""
    h = 24.0 * slot / SLOTS_PER_DAY
    return (1.0
            + 0.18 * math.sin(2 * math.pi * (h - 7.0) / 24.0)
            + 0.07 * math.sin(4 * math.pi * (h - 3.0) / 24.0))


@dataclass
class Stream:
    """一条指标时间序列。direction=-1 表示「下降是坏事」，+1 表示「上升是坏事」。"""
    name: str
    values: list[float] = field(default_factory=list)
    direction: int = -1
    p0: float = 0.12         # |同比| 超过该幅度 -> P0
    p1: float = 0.06
    p2: float = 0.03
    hard_threshold: float | None = None   # 静态阈值（只用来做对比实验）
    hard_op: str = ">"


def build_streams() -> tuple[list[Stream], dict[str, int]]:
    """[MOCK] 合成三条指标流并注入三类异常，返回故障注入起始窗口。"""
    n = DAYS * SLOTS_PER_DAY
    ctr, p99, pctr = [], [], []

    drop_at = 2 * SLOTS_PER_DAY + 120          # 第 3 天 10:00 起 CTR 突降
    spike_at = 2 * SLOTS_PER_DAY + 48          # 第 3 天 04:00 起 P99 尖刺 4 个窗口
    drift_at = 2 * SLOTS_PER_DAY + 192         # 第 3 天 16:00 起 pCTR 慢漂移

    for t in range(n):
        slot = t % SLOTS_PER_DAY
        f = diurnal(slot)

        c = 0.062 * f * random.gauss(1.0, 0.012)
        if t >= drop_at:
            c *= 0.82                          # 突降 18%
        ctr.append(c)

        lat = 78.0 / f * random.gauss(1.0, 0.05)
        if spike_at <= t < spike_at + 4:
            lat *= 3.0                         # 连续 4 窗口尖刺
        p99.append(lat)

        p = 0.065 * f * random.gauss(1.0, 0.010)
        if t >= drift_at:
            p *= (1.0035 ** (t - drift_at))     # 每窗口 +0.35% 的慢漂移
        pctr.append(p)

    streams = [
        Stream("ctr", ctr, direction=-1, hard_threshold=0.045, hard_op="<"),
        Stream("latency_p99_ms", p99, direction=+1, p0=0.60, p1=0.30, p2=0.15,
               hard_threshold=150.0, hard_op=">"),
        Stream("pctr_mean", pctr, direction=+1, p0=0.20, p1=0.10, p2=0.05,
               hard_threshold=0.12, hard_op=">"),
    ]
    return streams, {"ctr 突降": drop_at, "p99 尖刺": spike_at, "pctr 慢漂移": drift_at}


# ==========================================================================
# 检测量
# ==========================================================================
def wow(xs: list[float], t: int) -> float | None:
    """环比：与上一个窗口比。"""
    if t < 1 or xs[t - 1] == 0:
        return None
    return xs[t] / xs[t - 1] - 1.0


def dod(xs: list[float], t: int) -> float | None:
    """同比：与一个日周期前的同一时刻比。"""
    if t < SEASON or xs[t - SEASON] == 0:
        return None
    return xs[t] / xs[t - SEASON] - 1.0


def seasonal_baseline(xs: list[float], t: int) -> float | None:
    """季节基线 b_t = median{x_{t-S}, x_{t-2S}, ...}（历史同时刻中位数）。"""
    hist = [xs[t - k * SEASON] for k in range(1, DAYS + 1) if t - k * SEASON >= 0]
    return median(hist) if hist else None


def robust_z(xs: list[float], t: int) -> float | None:
    """季节性残差的稳健 z 分数（MAD 标准化）。

    残差 e_t = x_t - b_t 在无故障时应围绕 0 波动，因此不再做二次中心化，
    直接用近期残差的 MAD 估计尺度：z_t = e_t / (1.4826 * MAD(E))。
    这样残差的符号被保留下来，可以和「坏的方向」做一致性校验。
    """
    b = seasonal_baseline(xs, t)
    if b is None:
        return None
    resid = []
    for u in range(max(SEASON, t - MAD_WINDOW), t + 1):
        bu = seasonal_baseline(xs, u)
        if bu is not None:
            resid.append(xs[u] - bu)
    if len(resid) < 8:
        return None
    s = 1.4826 * mad(resid)
    if s <= 1e-12:
        return None
    return (xs[t] - b) / s


def grade(st: Stream, dod_val: float, z: float) -> str | None:
    """分级：同比幅度定级别，稳健 z 用作「确实朝坏方向偏离季节基线」的门禁。"""
    sign = -1 if st.direction < 0 else 1
    signed = dod_val * sign      # 转成「坏的方向为正」
    signed_z = z * sign
    if signed <= 0 or signed_z < 3.0:
        return None
    if signed >= st.p0:
        return "P0"
    if signed >= st.p1:
        return "P1"
    if signed >= st.p2:
        return "P2"
    return None


# ==========================================================================
# 告警引擎：持续确认 + 冷却抑制
# ==========================================================================
@dataclass
class Alert:
    t: int
    metric: str
    level: str
    value: float
    baseline: float
    dod: float
    wow: float
    z: float


def run_alerting(streams: list[Stream]) -> tuple[list[Alert], int]:
    """返回 (告警列表, 被冷却抑制掉的候选数)。"""
    alerts: list[Alert] = []
    suppressed = 0
    streak = {st.name: 0 for st in streams}
    last_fire = {st.name: -10 ** 9 for st in streams}
    n = len(streams[0].values)

    for t in range(n):
        for st in streams:
            d, w, z = dod(st.values, t), wow(st.values, t), robust_z(st.values, t)
            lvl = None if (d is None or z is None) else grade(st, d, z)
            if lvl is None:
                streak[st.name] = 0
                continue
            streak[st.name] += 1
            if streak[st.name] < CONFIRM:
                continue
            if t - last_fire[st.name] < COOLDOWN:
                suppressed += 1
                continue
            last_fire[st.name] = t
            b = seasonal_baseline(st.values, t) or float("nan")
            alerts.append(Alert(t, st.name, lvl, st.values[t], b, d,
                                w if w is not None else float("nan"), z))
    return alerts, suppressed


def count_hits(st: Stream, mode: str) -> int:
    """对比实验：三种检测策略各自命中多少个窗口（不含确认 / 抑制逻辑）。"""
    hits = 0
    for t in range(len(st.values)):
        if mode == "threshold":
            if st.hard_threshold is None:
                continue
            v = st.values[t]
            if (st.hard_op == ">" and v > st.hard_threshold) or\
               (st.hard_op == "<" and v < st.hard_threshold):
                hits += 1
        elif mode == "wow":
            w = wow(st.values, t)
            if w is not None and w * (-1 if st.direction < 0 else 1) >= st.p1:
                hits += 1
        else:  # dod + robust z
            d, z = dod(st.values, t), robust_z(st.values, t)
            if d is not None and z is not None and grade(st, d, z) is not None:
                hits += 1
    return hits


def fmt_slot(t: int) -> str:
    day = t // SLOTS_PER_DAY + 1
    mins = 5 * (t % SLOTS_PER_DAY)
    return f"D{day} {mins // 60:02d}:{mins % 60:02d}"


# ==========================================================================
def main() -> None:
    random.seed(42)

    print("=" * 92)
    print("§3.4 监控体系：指标采集 -> 同比/环比/分位数 -> 突变检测 -> 分级告警")
    print("=" * 92)
    streams, faults = build_streams()
    n = len(streams[0].values)
    print(f"[MOCK] {DAYS} 天 x {SLOTS_PER_DAY} 窗口（5 min/窗口）= {n} 个窗口，"
          f"3 条指标流")
    print("[MOCK] 注入故障：" + "，".join(
        f"{k}@{fmt_slot(v)}" for k, v in faults.items()))

    print("\n[1] 指标流分位数概览（正文：pxtr 与延迟都要看 P50/P90/P99）")
    print(f"     {'metric':<16}{'P50':>10}{'P90':>10}{'P99':>10}"
          f"{'D1 均值':>12}{'D3 均值':>12}")
    for st in streams:
        d1 = st.values[:SLOTS_PER_DAY]
        d3 = st.values[2 * SLOTS_PER_DAY:]
        print(f"     {st.name:<16}{quantile(st.values, .5):>10.4f}"
              f"{quantile(st.values, .9):>10.4f}{quantile(st.values, .99):>10.4f}"
              f"{sum(d1) / len(d1):>12.4f}{sum(d3) / len(d3):>12.4f}")

    print("\n[2] 分级告警（连续确认 >= %d 窗口，冷却 %d 窗口）" % (CONFIRM, COOLDOWN))
    alerts, suppressed = run_alerting(streams)
    print(f"     {'窗口':<9}{'级别':<5}{'metric':<16}{'当前值':>10}"
          f"{'季节基线':>10}{'同比':>9}{'环比':>9}{'稳健z':>9}")
    for a in alerts:
        print(f"     {fmt_slot(a.t):<9}{a.level:<5}{a.metric:<16}{a.value:>10.4f}"
              f"{a.baseline:>10.4f}{100 * a.dod:>8.1f}%{100 * a.wow:>8.1f}%"
              f"{a.z:>9.2f}")
    print(f"     共 {len(alerts)} 条告警；被冷却规则抑制掉的重复候选 {suppressed} 个")
    by_level: dict[str, int] = {}
    for a in alerts:
        by_level[a.level] = by_level.get(a.level, 0) + 1
    print("     级别分布：" + "，".join(f"{k}={v}" for k, v in sorted(by_level.items())))

    print("\n[3] 三种检测策略在同一数据流上的命中窗口数对比")
    print(f"     {'metric':<16}{'静态阈值':>10}{'仅环比':>10}{'同比+稳健z':>12}"
          f"{'故障真窗口数':>14}")
    truth = {"ctr": n - faults["ctr 突降"],
             "latency_p99_ms": 4,
             "pctr_mean": n - faults["pctr 慢漂移"]}
    for st in streams:
        print(f"     {st.name:<16}{count_hits(st, 'threshold'):>10d}"
              f"{count_hits(st, 'wow'):>10d}{count_hits(st, 'dod'):>12d}"
              f"{truth[st.name]:>14d}")

    print("\n[4] 结论")
    print("     >>> 慢漂移（pctr_mean）完全逃过静态阈值与环比：每窗口只涨 0.35%，")
    print("         环比永远看不出来，静态阈值要等绝对值涨到 0.12 才响；")
    print("         同比把日内周期约掉之后，累计涨幅立刻暴露出来，这正是正文说的")
    print("         「pCTR 整体抬高而真实 CTR 未同步提升」的发现方式。")
    print("     >>> 尖刺（latency_p99_ms）靠环比也能发现，但环比在正常日内波动上")
    print("         会反复误报，所以需要稳健 z 门禁 + 连续确认 + 冷却抑制降噪。")
    print("     >>> 分级的意义在于分流处置：P0 立即止损（回滚/降级），P1 当班跟进，")
    print("         P2 只记录待复盘；否则告警越多，工程师越不看。")


if __name__ == "__main__":
    main()
