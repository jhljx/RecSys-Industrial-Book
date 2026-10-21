"""对应正文第 3 章 §3.3.4「A/B实验迭代流程」。

正文原话：「一个新策略通常不会直接全量上线，而是经历逐步扩量的过程」，典型流程为

    5% -> 10% -> 20% -> 50% -> 100%

「首先通过 5% 左右的小流量验证策略安全性。若核心指标获得统计显著提升，且诊断指标
无明显恶化，则逐步扩大实验流量。当实验在更大流量规模下依然保持稳定收益，并持续
运行多个用户行为周期后，才会执行全量发布（Full Rollout）。与此同时，许多公司还会
保留约 1% 的长期 Holdout 流量作为永久对照组。」

本实现做了什么
---------------
1. **最小样本量 / MDE 估算**：给定 DAU 与各阶段流量占比，反解每个阶段「跑 N 天能
   检出多大的提升」（MDE），从而说明为什么 5% 阶段只能发现大收益、放量后才能确认
   小收益；
2. **逐步放量决策引擎（RolloutController）**：严格实现正文的放量规则——
   核心指标显著正向 + 诊断指标无明显恶化 -> 进入下一阶段；核心指标显著负向或
   诊断指标破线 -> 回滚；不显著 -> 延长观察（直到达到该阶段的最小样本量/天数上限）；
3. 对三个 `[MOCK]` 策略跑完整的放量流水线：
   - `strategy_good`：真实 +2.5% 时长收益，应一路放量到全量；
   - `strategy_tiny`：真实 +0.6% 收益，小于 5% 阶段的 MDE，只能靠放量换统计功效；
   - `strategy_risky`：时长 +3% 但 hate 率 +40%，应在小流量阶段就被护栏拦下回滚；
4. **新奇效应（Novelty Effect）**：首日提升虚高、随天数衰减，演示为什么正文要求
   「持续运行多个用户行为周期」才允许全量；
5. **1% 长期 Holdout**：全量发布后保留 1% 永久对照组，演示多个策略的收益叠加
   不等于各自收益之和（收益重叠 / 衰减），以及 Holdout 如何度量「系统级累计收益」。

关键公式
---------
    每组最小样本量 n = 2 (z_{1-a/2} + z_{power})^2 sigma^2 / delta^2
    反解 MDE         delta = (z_{1-a/2} + z_{power}) sigma sqrt(2/n)
    阶段可用样本     n = DAU * 流量占比 / 2 * 天数      （实验/对照各半）
    Welch t          t = (x1 - x2) / sqrt(s1^2/n1 + s2^2/n2)
    累计收益(Holdout) = 全量组指标 / Holdout 组指标 - 1
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


Z_ALPHA = 1.959964          # z_{0.975}，双侧 alpha=0.05
Z_POWER = 0.841621          # z_{0.80}，power=80%


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_ttest(a: list[float], b: list[float]) -> dict:
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    return {"mean_exp": m1, "mean_base": m2, "t": t,
            "p": 2 * (1 - norm_cdf(abs(t))),
            "lift_pct": 100.0 * (m1 / m2 - 1),
            "ci_low_pct": 100.0 * ((m1 - m2) - Z_ALPHA * se) / m2,
            "ci_high_pct": 100.0 * ((m1 - m2) + Z_ALPHA * se) / m2}


def two_proportion_ztest(c1: int, n1: int, c2: int, n2: int) -> dict:
    p1, p2 = c1 / n1, c2 / n2
    pp = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return {"p1": p1, "p2": p2, "z": z, "p": 2 * (1 - norm_cdf(abs(z))),
            "lift_pct": 100.0 * (p1 / p2 - 1) if p2 else 0.0}


def min_sample_size(sigma: float, mean: float, mde_rel: float) -> int:
    """每组最小样本量：n = 2 (z_a + z_b)^2 sigma^2 / delta^2。"""
    delta = mean * mde_rel
    return math.ceil(2 * (Z_ALPHA + Z_POWER) ** 2 * sigma ** 2 / delta ** 2)


def mde_from_n(sigma: float, mean: float, n_per_group: int) -> float:
    """反解 MDE（相对值）：delta = (z_a + z_b) sigma sqrt(2/n)。"""
    delta = (Z_ALPHA + Z_POWER) * sigma * math.sqrt(2.0 / n_per_group)
    return delta / mean


# ==========================================================================
# [MOCK] 线上指标采样
# ==========================================================================
BASE_WATCH_MEAN = 42.0      # 基线人均时长（秒）
BASE_WATCH_CV = 1.30        # 重尾：对数正态，CV 约 1.3
BASE_HATE_RATE = 0.0048


@dataclass
class Strategy:
    """[MOCK] 一个待上线策略的「真实」效应（上帝视角，实验只能观测带噪估计）。"""

    name: str
    watch_lift: float            # 稳态时长相对提升
    hate_lift: float = 0.0       # hate 率相对变化
    novelty_boost: float = 0.0   # 第 1 天额外虚高的提升（按天指数衰减）
    novelty_halflife: float = 2.5


def effective_watch_lift(s: Strategy, day: int) -> float:
    """新奇效应：day=1 时提升被抬高 novelty_boost，按半衰期衰减到稳态。"""
    decay = 0.5 ** ((day - 1) / s.novelty_halflife)
    return s.watch_lift + s.novelty_boost * decay


def sample_watch(n: int, mean: float) -> list[float]:
    """[MOCK] 人均时长样本：对数正态（重尾），均值校准到 mean，CV 固定。"""
    sigma = math.sqrt(math.log(1 + BASE_WATCH_CV ** 2))
    mu = math.log(mean) - sigma ** 2 / 2
    return [random.lognormvariate(mu, sigma) for _ in range(n)]


def sample_hate(n: int, rate: float) -> int:
    """[MOCK] hate 事件数（二项抽样，正态近似加速）。"""
    mean = n * rate
    sd = math.sqrt(n * rate * (1 - rate))
    return max(0, round(random.gauss(mean, sd)))


# ==========================================================================
# 放量决策引擎
# ==========================================================================
STAGES = (0.05, 0.10, 0.20, 0.50, 1.00)         # 正文：5% -> 10% -> 20% -> 50% -> 100%
STAGE_MIN_DAYS = {0.05: 3, 0.10: 3, 0.20: 4, 0.50: 7}
STAGE_MAX_DAYS = {0.05: 7, 0.10: 7, 0.20: 10, 0.50: 14}
HATE_GUARDRAIL_PCT = 5.0                        # hate 率相对恶化超过 +5% 即破线


@dataclass
class StageReport:
    stage: float
    days: int
    n_per_group: int
    mde_pct: float
    watch_lift_pct: float
    watch_ci: tuple[float, float]
    watch_p: float
    hate_lift_pct: float
    hate_p: float
    decision: str


class RolloutController:
    """正文 §3.3.4 的放量决策规则（核心指标=人均时长，诊断指标=hate 率）。"""

    def __init__(self, dau: int = 2_000_000, imps_per_user: int = 25) -> None:
        self.dau = dau
        self.imps_per_user = imps_per_user

    def run_stage(self, s: Strategy, stage: float, days: int) -> StageReport:
        # 实验/对照各占该阶段流量的一半（1.00 阶段留 1% Holdout，见 main 第 (5) 部分）
        n_real = max(600, int(self.dau * stage / 2 * days))
        n_per = min(n_real, 60_000)               # [MOCK] 采样上限，保证脚本 20s 内跑完
        scale = n_real / n_per                    # 真实样本量相对采样量的放大倍数

        base = sample_watch(n_per, BASE_WATCH_MEAN)
        lift = effective_watch_lift(s, days)
        exp = sample_watch(n_per, BASE_WATCH_MEAN * (1 + lift))
        r = welch_ttest(exp, base)
        # 真实样本量是采样量的 scale 倍 => 估计量对真值的偏离缩小 sqrt(scale) 倍，
        # 即 lift_hat = lift_true + (lift_sample - lift_true) / sqrt(scale)，
        # 这样点估计与置信区间同步收紧，分布与「真的跑 n_real 个样本」一致。
        true_pct = 100 * lift
        lift_pct = true_pct + (r["lift_pct"] - true_pct) / math.sqrt(scale)
        se_pct = (r["ci_high_pct"] - r["ci_low_pct"]) / 2 / 1.96 / math.sqrt(scale)
        t_real = lift_pct / se_pct if se_pct > 0 else 0.0
        p_real = 2 * (1 - norm_cdf(abs(t_real)))
        ci = (lift_pct - 1.96 * se_pct, lift_pct + 1.96 * se_pct)

        imps = int(n_per * self.imps_per_user)
        h_base = sample_hate(imps, BASE_HATE_RATE)
        h_exp = sample_hate(imps, BASE_HATE_RATE * (1 + s.hate_lift))
        hz = two_proportion_ztest(h_exp, imps, h_base, imps)
        hate_true_pct = 100 * s.hate_lift
        hate_pct = hate_true_pct + (hz["lift_pct"] - hate_true_pct) / math.sqrt(scale)
        hate_p = 2 * (1 - norm_cdf(abs(hz["z"] * math.sqrt(scale))))

        sigma = BASE_WATCH_MEAN * BASE_WATCH_CV
        mde = 100 * mde_from_n(sigma, BASE_WATCH_MEAN, n_real)

        # ---- 正文决策规则 ----
        guard_broken = hate_pct > HATE_GUARDRAIL_PCT and hate_p < 0.05
        core_sig_pos = p_real < 0.05 and lift_pct > 0
        core_sig_neg = p_real < 0.05 and lift_pct < 0
        if guard_broken:
            decision = "回滚（诊断指标破线）"
        elif core_sig_neg:
            decision = "回滚（核心指标显著负向）"
        elif core_sig_pos and days >= STAGE_MIN_DAYS.get(stage, 7):
            decision = "放量" if stage < 1.00 else "全量发布"
        elif days >= STAGE_MAX_DAYS.get(stage, 14):
            # 观察期满仍不显著：若真实收益小于本阶段 MDE，小流量永远等不出结论。
            # 此时只要护栏安全、置信区间下界未显著为负，就放量以换取统计功效。
            if stage < 0.50 and ci[0] > -mde:
                decision = f"放量以提升功效（收益 < 本阶段 MDE {mde:.3f}%）"
            else:
                decision = "终止（观察期满仍不显著，收益小于 MDE）"
        else:
            decision = "延长观察"
        return StageReport(stage, days, n_real, mde, lift_pct, ci, p_real,
                           hate_pct, hate_p, decision)

    def rollout(self, s: Strategy) -> list[StageReport]:
        reports: list[StageReport] = []
        for stage in STAGES:
            days = STAGE_MIN_DAYS.get(stage, 7)
            while True:
                rep = self.run_stage(s, stage, days)
                reports.append(rep)
                if rep.decision == "延长观察":
                    days += 2
                    continue
                break
            if rep.decision.startswith("回滚") or rep.decision.startswith("终止"):
                return reports
        return reports


def print_rollout(name: str, reports: list[StageReport]) -> None:
    print(f"\n  >> 策略 {name}")
    print(f"  {'阶段':>6} | {'天数':>4} | {'每组样本':>12} | {'该阶段MDE':>9} | "
          f"{'时长提升':>9} | {'95% CI':>20} | {'p':>9} | "
          f"{'hate相对':>9} | 决策")
    print("  " + "-" * 118)
    for r in reports:
        ci = f"[{r.watch_ci[0]:+.3f}%,{r.watch_ci[1]:+.3f}%]"
        print(f"  {100 * r.stage:>5.0f}% | {r.days:>4d} | {r.n_per_group:>12,d} | "
              f"{r.mde_pct:>8.3f}% | {r.watch_lift_pct:>+8.3f}% | {ci:>20} | "
              f"{r.watch_p:>9.2e} | {r.hate_lift_pct:>+8.2f}% | {r.decision}")


def main() -> None:
    random.seed(42)
    print("=" * 92)
    print("§3.3.4 A/B实验迭代流程：5% -> 10% -> 20% -> 50% -> 100% 的放量决策与 1% 长期 Holdout")
    print("=" * 92)

    dau = 2_000_000
    sigma = BASE_WATCH_MEAN * BASE_WATCH_CV

    # ---------- (1) 最小样本量 / MDE ----------
    print(f"\n[1] 最小样本量与 MDE（人均时长基线 {BASE_WATCH_MEAN:.1f}s，"
          f"sigma = {sigma:.2f}s，CV = {BASE_WATCH_CV:.2f}；alpha=0.05 双侧，power=80%）")
    print(f"  {'目标MDE(相对)':>13} | {'每组最小样本量':>16} | "
          f"{'5%流量需跑天数':>14} | {'20%流量':>9} | {'50%流量':>9}")
    print("  " + "-" * 74)
    for mde in (0.002, 0.005, 0.01, 0.02, 0.05):
        n = min_sample_size(sigma, BASE_WATCH_MEAN, mde)
        d5 = n / (dau * 0.05 / 2)
        d20 = n / (dau * 0.20 / 2)
        d50 = n / (dau * 0.50 / 2)
        print(f"  {100 * mde:>12.1f}% | {n:>16,d} | {d5:>14.2f} | {d20:>9.2f} | {d50:>9.2f}")
    print(f"\n  反解：各放量阶段跑到最小天数时能检出的最小提升（DAU={dau:,}）")
    print(f"  {'阶段':>6} | {'最小天数':>8} | {'每组样本量':>13} | {'可检出MDE':>10}")
    print("  " + "-" * 48)
    for stage in STAGES[:-1]:
        d = STAGE_MIN_DAYS[stage]
        n = int(dau * stage / 2 * d)
        print(f"  {100 * stage:>5.0f}% | {d:>8d} | {n:>13,d} | "
              f"{100 * mde_from_n(sigma, BASE_WATCH_MEAN, n):>9.4f}%")
    print("  >>> 5% 阶段的定位是「验证安全性」而非「确认收益」：它只能检出 "
          f"{100 * mde_from_n(sigma, BASE_WATCH_MEAN, int(dau * 0.05 / 2 * 3)):.3f}% "
          "以上的提升，")
    print("      更小的真实收益必须放量后才可能被统计显著地检出（n ∝ 1/delta^2）。")

    # ---------- (2)(3)(4) 三个策略的放量流水线 ----------
    print("\n[2] 逐步放量决策（规则=正文 §3.3.4：核心指标显著正向 且 诊断指标无明显恶化 -> 放量）")
    print(f"    护栏：hate 率相对恶化 > {HATE_GUARDRAIL_PCT:.0f}% 且显著 -> 立即回滚")
    ctrl = RolloutController(dau=dau)
    strategies = [
        Strategy("strategy_good  (真实 +2.5% 时长)", watch_lift=0.025, hate_lift=-0.02),
        Strategy("strategy_tiny  (真实 +0.6% 时长)", watch_lift=0.006, hate_lift=0.00),
        Strategy("strategy_risky (时长 +3%, hate +40%)", watch_lift=0.030, hate_lift=0.40),
    ]
    finals = {}
    for s in strategies:
        reps = ctrl.rollout(s)
        print_rollout(s.name, reps)
        finals[s.name] = reps[-1].decision

    print("\n  放量结论汇总：")
    for name, dec in finals.items():
        print(f"    {name:<42} -> {dec}")

    # ---------- (4) 新奇效应 ----------
    print("\n[3] 新奇效应（Novelty Effect）：为什么正文要求「持续运行多个用户行为周期」")
    nov = Strategy("strategy_novelty", watch_lift=0.004, novelty_boost=0.030,
                   novelty_halflife=2.5)
    print(f"  该策略稳态真实提升仅 {100 * nov.watch_lift:.1f}%，"
          f"但首日被新奇效应抬高 +{100 * nov.novelty_boost:.1f}%（半衰期 "
          f"{nov.novelty_halflife} 天）")
    print(f"  {'观察天数':>8} | {'真实提升':>9} | {'观测提升':>9} | {'95% CI':>22} | "
          f"{'p':>9} | 若此刻决策")
    print("  " + "-" * 86)
    for d in (1, 2, 3, 5, 7, 10, 14):
        rep = ctrl.run_stage(nov, 0.10, d)
        truth = 100 * effective_watch_lift(nov, d)
        ci = f"[{rep.watch_ci[0]:+.3f}%,{rep.watch_ci[1]:+.3f}%]"
        act = "判定大幅正向(高估)" if rep.watch_lift_pct > 1.5 else (
            "判定小幅正向" if rep.watch_p < 0.05 and rep.watch_lift_pct > 0 else "判定不显著")
        print(f"  {d:>8d} | {truth:>+8.3f}% | {rep.watch_lift_pct:>+8.3f}% | {ci:>22} | "
              f"{rep.watch_p:>9.2e} | {act}")
    print("  >>> 第 1~2 天的收益主要来自「用户对新东西好奇」，不可持续；")
    print("      只看首日数据会严重高估收益，这是正文要求跑多个行为周期的根本原因。")

    # ---------- (5) 1% 长期 Holdout ----------
    print("\n[4] 1% 长期 Holdout（永久对照组）：度量系统级累计收益")
    print("  [MOCK] 一个季度内先后全量了 6 个策略；Holdout 组始终停留在季度初的版本。")
    quarter = [
        ("Q1-W02 多目标融合公式重构", 0.022),
        ("Q1-W04 召回新增 i2i_v3 通道", 0.011),
        ("Q1-W06 排序模型 MMoE 升级", 0.018),
        ("Q1-W08 重排多样性打散", 0.004),
        ("Q1-W10 时长目标校准修正", 0.009),
        ("Q1-W12 冷启动扶持策略", -0.002),
    ]
    naive_sum = 0.0
    cum_true = 1.0
    overlap = 0.65          # [MOCK] 收益重叠系数：后上线策略的边际收益被前者吃掉一部分
    print(f"  {'全量批次':>26} | {'AB实验测得':>10} | {'累乘预期':>10} | "
          f"{'Holdout实测累计':>14} | {'兑现率':>8}")
    print("  " + "-" * 82)
    for i, (name, lift) in enumerate(quarter):
        naive_sum += lift
        cum_true *= 1 + lift * (overlap ** i if lift > 0 else 1.0)
        # [MOCK] Holdout 观测：真实累计收益 + 抽样噪声（1% 流量，样本少故噪声更大）
        n_hold = int(dau * 0.01 / 2 * 7)
        se = BASE_WATCH_CV * math.sqrt(2.0 / n_hold)   # 两组均值之差的相对标准误
        obs = (cum_true - 1) + random.gauss(0, se)
        print(f"  {name:>26} | {100 * lift:>+9.2f}% | "
              f"{100 * (math.prod(1 + l for _, l in quarter[:i + 1]) - 1):>+9.2f}% | "
              f"{100 * obs:>+13.2f}% | "
              f"{100 * obs / naive_sum:>7.1f}%")
    print(f"  各实验收益朴素相加 = {100 * naive_sum:+.2f}%，"
          f"Holdout 实测季度累计 = {100 * (cum_true - 1):+.2f}%")
    print(f"  收益兑现率 = {100 * (cum_true - 1) / naive_sum:.1f}%")
    print("  >>> 正文：保留约 1% 的长期 Holdout 流量作为永久对照组，用于观察系统长期")
    print("      演化趋势以及进行跨实验横向比较。单实验收益之和几乎总是大于系统真实累计")
    print("      收益（收益重叠、新奇效应衰减、季节性），Holdout 是唯一能揭穿这件事的手段。")
    print("  >>> 代价：1% 流量永久停留在旧版本，且 1% 流量下 MDE 很大"
          f"（7 天每组 {int(dau * 0.01 / 2 * 7):,} 人，MDE ≈ "
          f"{100 * mde_from_n(sigma, BASE_WATCH_MEAN, int(dau * 0.01 / 2 * 7)):.3f}%），")
    print("      因此 Holdout 只用于看「大盘累计收益」，不用于判定单个策略。")


if __name__ == "__main__":
    main()
