"""对应正文第 3 章 §3.4「降级与兜底监控」。

正文原话：「推荐系统通常会设计多层容灾机制。例如：特征服务降级；模型降级；
召回降级；热榜兜底；内容池兜底。因此需要持续监控：降级触发次数；降级流量占比；
兜底内容占比；降级持续时长等。」

本实现做了什么
---------------
1. 把正文列的五层容灾按「损失从小到大」排成一条降级梯度（L0 正常 -> L4 内容池兜底），
   每一级都给出该级能承受的单请求延迟预算、以及相对正常链路的预期效果折损：

       L0 正常全链路          -> 效果基准 1.000
       L1 特征服务降级        -> 丢实时特征，用离线静态特征
       L2 模型降级            -> 精排大模型换成轻量 LR / 双塔
       L3 召回降级            -> 只保留本地缓存的少数几路召回
       L4 热榜 / 内容池兜底   -> 直接下发热榜，无个性化

2. 熔断器（Circuit Breaker）按「滑动窗口错误率 / P99 延迟」决定是否降级，状态机为
   CLOSED -> OPEN -> HALF_OPEN -> CLOSED：
   - CLOSED：窗口内 error_rate > `ERR_OPEN` 或 p99 > `P99_OPEN` 则 OPEN 并下沉一级；
   - OPEN：冷却 `OPEN_HOLD` 个窗口内不做任何尝试（避免反复打死下游）；
   - HALF_OPEN：放 `PROBE_RATIO` 的探测流量回上一级，连续 `RECOVER_OK` 个窗口
     健康才真正恢复一级（逐级恢复，不一次跳回 L0），否则退回 OPEN 重新冷却。
   这里的「连续健康才恢复 + 逐级恢复」是防止恢复瞬间再次把下游打垮的标准做法。

3. `[MOCK]` 构造一条 120 个窗口（每窗口 10 秒）的故障注入时间线：
   特征服务超时 -> 恢复 -> GPU 推理队列打满 -> 召回引擎抖动 -> 全面恢复，
   逐窗口跑熔断器，打印状态流水表。

4. 最后统计正文要求的四个降级监控指标：
       降级触发次数 = 状态机从 CLOSED 进入 OPEN 的次数
       降级流量占比 = sum_{t: level>0} qps_t / sum_t qps_t
       兜底内容占比 = sum_t qps_t * fallback_ratio(level_t) / sum_t qps_t
       降级持续时长 = 处于 level>0 的窗口数 * 窗口长度（并给出每段的时长）
   并对比「有降级 / 无降级（硬扛）」两种情况下的请求成功率与累计效果损失，
   说明降级的本质是「主动付出可控的效果损失，换取可用性」。

全链路组件（特征服务、模型服务、召回引擎、熔断器、热榜）均为进程内 `[MOCK]`，
不起真实服务、不做网络调用。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

WINDOW_SEC = 10          # 每个监控窗口 10 秒
N_WINDOW = 120           # 共 120 个窗口 = 20 分钟
REQ_PER_WINDOW = 400     # 每窗口采样 400 个请求做仿真

ERR_OPEN = 0.05          # 窗口错误率超过 5% -> 熔断降级
P99_OPEN = 180.0         # 窗口 P99 延迟超过 180ms -> 熔断降级
ERR_OK = 0.01            # 恢复判定：错误率低于 1%
P99_OK = 120.0           # 且 P99 低于 120ms
OPEN_HOLD = 3            # OPEN 冷却窗口数
RECOVER_OK = 2           # HALF_OPEN 连续健康窗口数才恢复一级
PROBE_RATIO = 0.10       # HALF_OPEN 阶段的探测流量比例

CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"


# ==========================================================================
# 降级梯度（正文列出的五层容灾）
# ==========================================================================
@dataclass(frozen=True)
class Level:
    idx: int
    name: str
    base_latency_ms: float      # 正常状态下该级的平均延迟
    quality: float              # 相对 L0 的效果（如人均时长）系数
    fallback_ratio: float       # 下发内容中「兜底内容」的占比
    uses: tuple[str, ...]       # 该级依赖哪些下游（决定它会被哪些故障影响）


LEVELS = [
    Level(0, "L0 正常全链路", 42.0, 1.000, 0.00, ("feature", "model", "recall")),
    Level(1, "L1 特征服务降级", 33.0, 0.962, 0.00, ("model", "recall")),
    Level(2, "L2 模型降级", 21.0, 0.915, 0.00, ("recall",)),
    Level(3, "L3 召回降级", 14.0, 0.848, 0.25, ()),
    Level(4, "L4 热榜/内容池兜底", 6.0, 0.690, 1.00, ()),
]


# ==========================================================================
# [MOCK] 故障时间线
# ==========================================================================
@dataclass(frozen=True)
class Incident:
    name: str
    dep: str                 # 受影响的下游：feature / model / recall
    start: int
    end: int                 # 左闭右开
    err: float               # 该下游的失败概率
    lat_mult: float          # 该下游的延迟放大倍数


INCIDENTS = [
    Incident("特征服务大面积超时", "feature", 10, 28, 0.30, 6.0),
    Incident("GPU 推理队列打满", "model", 45, 70, 0.22, 7.0),
    Incident("召回引擎索引抖动", "recall", 84, 96, 0.35, 4.0),
]


def dep_state(t: int) -> dict[str, tuple[float, float]]:
    """[MOCK] 返回窗口 t 时各下游的 （失败概率，延迟放大倍数）。"""
    st = {d: (0.0005, 1.0) for d in ("feature", "model", "recall")}
    for inc in INCIDENTS:
        if inc.start <= t < inc.end:
            st[inc.dep] = (inc.err, inc.lat_mult)
    return st


def qps_of(t: int) -> float:
    """[MOCK] 窗口 QPS，带轻微波动。"""
    return 12000.0 * (1.0 + 0.08 * math.sin(2 * math.pi * t / 40.0))


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def serve_window(t: int, level: Level, probe_level: Level | None) -> tuple[float, float, float]:
    """[MOCK] 在窗口 t 以 `level` 服务（可带 PROBE_RATIO 的 `probe_level` 探测流量）。

    返回 (错误率, 平均延迟, P99 延迟)。只要该级仍依赖的下游在故障中，
    请求就会按该下游的失败概率报错、按其倍数放大延迟；下沉一级之后不再
    依赖该下游，错误率与延迟立刻回落——这就是降级的作用。
    """
    st = dep_state(t)
    errs, lats = 0, []
    for _ in range(REQ_PER_WINDOW):
        lv = level
        if probe_level is not None and random.random() < PROBE_RATIO:
            lv = probe_level
        lat = lv.base_latency_ms * random.gauss(1.0, 0.12)
        failed = False
        for dep in lv.uses:
            p_err, mult = st[dep]
            if random.random() < p_err:
                failed = True
                lat *= mult
                break
            lat *= 1.0 + 0.25 * (mult - 1.0) * random.random()
        if failed:
            errs += 1
        lats.append(max(lat, 1.0))
    return errs / REQ_PER_WINDOW, sum(lats) / len(lats), quantile(lats, 0.99)


# ==========================================================================
# 熔断器状态机
# ==========================================================================
@dataclass
class Breaker:
    level: int = 0
    state: str = CLOSED
    hold: int = 0            # OPEN 剩余冷却窗口
    ok_streak: int = 0       # HALF_OPEN 已连续健康窗口
    trips: int = 0           # 降级触发次数（CLOSED -> OPEN）
    recoveries: int = 0      # 成功恢复一级的次数

    def unhealthy(self, err: float, p99: float) -> bool:
        return err > ERR_OPEN or p99 > P99_OPEN

    def healthy(self, err: float, p99: float) -> bool:
        return err < ERR_OK and p99 < P99_OK

    def step(self, err: float, p99: float) -> str:
        """消费本窗口观测，返回处置动作字符串；self.level 为下一窗口服务等级。"""
        if self.state == CLOSED:
            if self.unhealthy(err, p99):
                if self.level < len(LEVELS) - 1:
                    self.level += 1
                self.state, self.hold, self.trips = OPEN, OPEN_HOLD, self.trips + 1
                return f"熔断 -> 下沉到 L{self.level}"
            return "--"

        if self.state == OPEN:
            if self.unhealthy(err, p99) and self.level < len(LEVELS) - 1:
                self.level += 1
                self.hold = OPEN_HOLD
                return f"仍不健康 -> 继续下沉到 L{self.level}"
            self.hold -= 1
            if self.hold <= 0:
                self.state, self.ok_streak = HALF_OPEN, 0
                return "冷却结束 -> HALF_OPEN 放探测流量"
            return f"冷却中（{self.hold}）"

        # HALF_OPEN
        if not self.healthy(err, p99):
            self.state, self.hold, self.ok_streak = OPEN, OPEN_HOLD, 0
            return "探测失败 -> 退回 OPEN 重新冷却"
        self.ok_streak += 1
        if self.ok_streak < RECOVER_OK:
            return f"探测健康（{self.ok_streak}/{RECOVER_OK}）"
        self.level -= 1
        self.recoveries += 1
        self.ok_streak = 0
        if self.level == 0:
            self.state = CLOSED
            return "恢复到 L0，熔断器关闭"
        return f"上浮一级 -> L{self.level}(继续 HALF_OPEN)"


# ==========================================================================
@dataclass
class Row:
    t: int
    level: int
    state: str
    err: float
    avg: float
    p99: float
    action: str
    qps: float
    quality: float
    fb: float


def simulate(with_degrade: bool) -> tuple[list[Row], int, int]:
    """跑完整条时间线，返回 (逐窗口记录, 降级触发次数, 逐级恢复次数)。

    with_degrade=False 时禁用降级能力（永远停在 L0，硬扛故障），用作对照。
    """
    br = Breaker()
    rows: list[Row] = []
    for t in range(N_WINDOW):
        lv = LEVELS[br.level]
        probe = LEVELS[br.level - 1] if (br.state == HALF_OPEN and br.level > 0) else None
        err, avg, p99 = serve_window(t, lv, probe)
        action = br.step(err, p99) if with_degrade else "--"
        rows.append(Row(t, lv.idx, br.state if with_degrade else CLOSED,
                        err, avg, p99, action, qps_of(t),
                        lv.quality, lv.fallback_ratio))
    return rows, br.trips, br.recoveries


def segments(rows: list[Row]) -> list[tuple[int, int, int]]:
    """把 level>0 的连续窗口聚成降级区间，返回 (start, end, 最深级别)。"""
    segs: list[tuple[int, int, int]] = []
    start, deepest = None, 0
    for r in rows:
        if r.level > 0 and start is None:
            start, deepest = r.t, r.level
        elif r.level > 0:
            deepest = max(deepest, r.level)
        elif start is not None:
            segs.append((start, r.t, deepest))
            start, deepest = None, 0
    if start is not None:
        segs.append((start, N_WINDOW, deepest))
    return segs


def main() -> None:
    random.seed(42)

    print("=" * 96)
    print("§3.4 降级与兜底监控：熔断触发 -> 逐级下沉 -> 半开探测 -> 逐级恢复")
    print("=" * 96)
    print(f"[MOCK] {N_WINDOW} 个窗口 x {WINDOW_SEC}s = {N_WINDOW * WINDOW_SEC}s，"
          f"每窗口采样 {REQ_PER_WINDOW} 个请求")
    print("[MOCK] 注入故障：" + "；".join(
        f"{i.name}[{i.start},{i.end}) dep={i.dep} err={i.err:.2f} lat x{i.lat_mult:.0f}"
        for i in INCIDENTS))

    print("\n[1] 降级梯度（正文：特征服务降级 / 模型降级 / 召回降级 / 热榜兜底 / 内容池兜底）")
    print(f"     {'级别':<20}{'均值延迟ms':>12}{'效果系数':>10}{'兜底内容占比':>14}{'仍依赖':>22}")
    for lv in LEVELS:
        print(f"     {lv.name:<20}{lv.base_latency_ms:>12.1f}{lv.quality:>10.3f}"
              f"{lv.fallback_ratio:>14.2f}{','.join(lv.uses) or '-':>22}")

    rows, trips, recoveries = simulate(with_degrade=True)

    print("\n[2] 状态流水（只打印发生状态变化的窗口）")
    print(f"     {'窗口':>5}{'服务级别':>10}{'熔断状态':>11}{'错误率':>9}"
          f"{'均值ms':>9}{'P99ms':>9}  处置")
    for r in rows:
        if r.action != "--" and not r.action.startswith("冷却中"):
            print(f"     {r.t:>5d}{('L' + str(r.level)):>10}{r.state:>11}"
                  f"{100 * r.err:>8.2f}%{r.avg:>9.1f}{r.p99:>9.1f}  {r.action}")

    print("\n[3] 正文要求的四个降级监控指标")
    tot_q = sum(r.qps for r in rows)
    deg_q = sum(r.qps for r in rows if r.level > 0)
    fb_q = sum(r.qps * r.fb for r in rows)
    deg_w = sum(1 for r in rows if r.level > 0)
    print(f"     降级触发次数（CLOSED->OPEN）        {trips:d} 次，"
          f"逐级恢复 {recoveries:d} 次")
    print(f"     降级流量占比                        {100 * deg_q / tot_q:.2f}%")
    print(f"     兜底内容占比                        {100 * fb_q / tot_q:.2f}%")
    print(f"     降级持续时长                        {deg_w} 个窗口 = "
          f"{deg_w * WINDOW_SEC}s(占时间线 {100 * deg_w / N_WINDOW:.1f}%)")
    print(f"     {'降级区间':<14}{'起止窗口':>14}{'时长s':>8}{'最深级别':>10}")
    for i, (a, b, d) in enumerate(segments(rows), 1):
        print(f"     {'段#' + str(i):<14}{f'[{a},{b})':>14}"
              f"{(b - a) * WINDOW_SEC:>8d}{('L' + str(d)):>10}")

    print("\n[4] 各级别的驻留时长与其间的实际服务质量")
    print(f"     {'级别':<20}{'窗口数':>8}{'驻留s':>8}{'平均错误率':>12}"
          f"{'平均P99ms':>12}")
    for lv in LEVELS:
        sub = [r for r in rows if r.level == lv.idx]
        if not sub:
            continue
        print(f"     {lv.name:<20}{len(sub):>8d}{len(sub) * WINDOW_SEC:>8d}"
              f"{100 * sum(r.err for r in sub) / len(sub):>11.2f}%"
              f"{sum(r.p99 for r in sub) / len(sub):>12.1f}")

    print("\n[5] 对照实验：关掉降级能力，硬扛同一组故障")
    random.seed(42)
    hard, _, _ = simulate(with_degrade=False)
    print(f"     {'方案':<16}{'请求成功率':>12}{'平均P99ms':>12}"
          f"{'最差窗口错误率':>16}{'累计效果（相对L0全程）':>22}")
    for tag, rs in (("有降级", rows), ("无降级硬扛", hard)):
        succ = sum(r.qps * (1 - r.err) for r in rs) / sum(r.qps for r in rs)
        p99 = sum(r.p99 for r in rs) / len(rs)
        worst = max(r.err for r in rs)
        # 效果 = 成功请求量 x 该级效果系数，归一化到「全程 L0 且零故障」
        eff = sum(r.qps * (1 - r.err) * r.quality for r in rs) / sum(r.qps for r in rs)
        print(f"     {tag:<16}{100 * succ:>11.2f}%{p99:>12.1f}"
              f"{100 * worst:>15.2f}%{eff:>22.4f}")

    print("\n[6] 结论")
    print("     >>> 降级的本质是主动用可控的效果损失换可用性：上表「有降级」的")
    print("         累计效果系数低于零故障理想值，但请求成功率明显高于硬扛方案，")
    print("         最差窗口错误率也被压住——用户看到的是略差的推荐，而不是白屏。")
    print("     >>> 恢复必须「逐级 + 连续健康确认 + 半开小流量探测」：一次性跳回 L0")
    print("         会把尚未痊愈的下游再次打死，监控上表现为熔断器反复 OPEN。")
    print("     >>> 正文要求监控的四个量（触发次数 / 降级流量占比 / 兜底内容占比 /")
    print("         降级持续时长）缺一不可：只看触发次数会漏掉「长期低水位降级」，")
    print("         只看时长会漏掉「频繁抖动」，兜底内容占比则直接对应用户体验。")


if __name__ == "__main__":
    main()
