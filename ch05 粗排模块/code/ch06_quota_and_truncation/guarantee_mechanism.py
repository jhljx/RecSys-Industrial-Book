"""
粗排保量机制:分数补偿 Δ 与预留位置两种实现(第 5 章 §5.6.2 粗排保量机制)

对应正文概念
------------
正文列出需要保量(Guarantee)的对象:新主播保量、新视频保量、广告保量、商业化内容保量。
并给出两种实现:

实现一(分数补偿):若曝光进度落后于目标,则抬分
    Exposure_i < Target_i  =>  Score_i = Score_i + Δ
其中 Δ 为保量补偿项。

实现二(位置预留):在 Quota 层面直接保留固定曝光位置 ——
正文原话是「在粗排最终输出的候选集 Top100 中预留比如 5% 的流量给冷启动的新内容」。

本实现做了什么
--------------
把保量写成一天 T 个时间片的在线仿真,[MOCK] 流量与 [MOCK] 候选供给逐片到来:
1. 基线 no_guarantee:纯 TopK(score),观察保量对象的欠曝光缺口。
2. 固定 Δ 补偿:对所有落后对象统一加同一个 Δ,扫描 Δ ∈ {0.005, 0.01, 0.02, 0.05},
   打印完成率与对大盘融合分的侵蚀。
3. 自适应 Δ(PID 型进度追踪):Δ_t = Δ_max * clip(进度缺口 / 容忍带, 0, 1),其中
       进度缺口 = Target_i * (t / T) - Exposure_i(t)
   落后越多补得越多,追上后自动停补,避免固定 Δ 的「超投」。
4. 位置预留:Top100 里固定留 5 个位置给冷启内容(正文实现二),与分数补偿做对比。
打印「保量完成率 / 超投率 / 大盘平均融合分损失 / 保量对象实际曝光曲线」数值表。

关键公式
--------
    Exposure_i < Target_i  =>  Score_i = Score_i + Δ
    Δ_t = Δ_max · clip( (Target_i · t/T − Exposure_i(t)) / (Target_i · band), 0, 1 )
    CandidateSet = TopK_{100−r}(score) ∪ TopR_r(保量池),  r = 5% × 100
"""

from __future__ import annotations

import random
from dataclasses import dataclass

T_SLOTS = 24                 # 一天 24 个时间片
TOPK = 100                   # 正文:粗排输出 Top100
REQ_PER_SLOT = 40            # [MOCK] 每片请求数
POOL_PER_REQ = 600           # [MOCK] 每次请求的候选规模
RESERVE_RATIO = 0.05         # 正文实现二:预留 5% 流量给冷启新内容
GUARANTEE_TARGET = 0.05      # 保量对象的目标曝光占比(全天),与预留 5% 位置对齐


@dataclass
class Cand:
    score: float
    is_guarantee: bool       # 是否属于保量池(新主播/新视频/商业化内容)


def gen_request(rng: random.Random) -> list[Cand]:
    """[MOCK] 单次请求的候选池:保量内容占 12%,但因缺后验其 score 分布明显偏低。"""
    out = []
    for _ in range(POOL_PER_REQ):
        g = rng.random() < 0.12
        mu, sd = (0.042, 0.016) if g else (0.062, 0.020)
        out.append(Cand(score=max(1e-4, rng.gauss(mu, sd)), is_guarantee=g))
    return out


def select_plain(pool: list[Cand], bonus: float) -> list[Cand]:
    """Score_i = Score_i + Δ 后取 TopK(实现一)。"""
    keyed = [(c.score + (bonus if c.is_guarantee else 0.0), c) for c in pool]
    keyed.sort(key=lambda kv: -kv[0])
    return [c for _, c in keyed[:TOPK]]


def select_reserved(pool: list[Cand], reserve: int) -> list[Cand]:
    """TopK 中预留 reserve 个位置给保量池(实现二)。"""
    g = sorted([c for c in pool if c.is_guarantee], key=lambda c: -c.score)[:reserve]
    rest = sorted([c for c in pool if not c.is_guarantee], key=lambda c: -c.score)
    return g + rest[: TOPK - len(g)]


def adaptive_delta(exposure: int, total: int, t: int, delta_max: float,
                   band: float = 0.35) -> float:
    """按进度缺口自适应补偿:落后越多 Δ 越大,追平即归零。"""
    pace = GUARANTEE_TARGET * total * (t / T_SLOTS)      # 到 t 时刻应完成的曝光
    gap = pace - exposure
    if gap <= 0:
        return 0.0
    ref = max(1.0, GUARANTEE_TARGET * total * band)
    return delta_max * min(1.0, gap / ref)


def run(strategy: str, param: float, seed: int = 2024) -> dict[str, float]:
    """跑一天的仿真,返回保量完成率 / 超投率 / 大盘平均分。"""
    rng = random.Random(seed)
    total_slots_exposure = T_SLOTS * REQ_PER_SLOT * TOPK
    target = GUARANTEE_TARGET * total_slots_exposure
    exposure, score_sum, shown = 0, 0.0, 0
    curve = []
    for t in range(1, T_SLOTS + 1):
        if strategy == "fixed":
            bonus = param if exposure < target else 0.0
        elif strategy == "adaptive":
            bonus = adaptive_delta(exposure, total_slots_exposure, t - 1, param)
        else:
            bonus = 0.0
        for _ in range(REQ_PER_SLOT):
            pool = gen_request(rng)
            sel = (select_reserved(pool, int(RESERVE_RATIO * TOPK))
                   if strategy == "reserve" else select_plain(pool, bonus))
            exposure += sum(1 for c in sel if c.is_guarantee)
            score_sum += sum(c.score for c in sel)
            shown += len(sel)
        curve.append(exposure / target)
    return {
        "完成率": exposure / target,
        "超投率": max(0.0, exposure / target - 1.0),
        "大盘平均分": score_sum / shown,
        "保量曝光占比": exposure / shown,
        "curve": curve,
    }


if __name__ == "__main__":
    random.seed(42)

    print("=" * 104)
    print("保量仿真设定([MOCK] 流量与候选供给)")
    print("=" * 104)
    print(f"时间片 T = {T_SLOTS},每片 {REQ_PER_SLOT} 次请求,每次请求候选 {POOL_PER_REQ} 个,"
          f"粗排输出 Top{TOPK}")
    print(f"保量对象在候选池占比 12%,但 score ~ N(0.042, 0.016) 低于普通内容 N(0.062, 0.020);"
          f"\n全天保量目标 Target = {GUARANTEE_TARGET:.0%} × 总曝光"
          f" = {int(GUARANTEE_TARGET * T_SLOTS * REQ_PER_SLOT * TOPK)} 次曝光")

    base = run("none", 0.0)
    print("\n" + "=" * 104)
    print("实现一:Exposure_i < Target_i  =>  Score_i = Score_i + Δ(固定 Δ 扫描)")
    print("=" * 104)
    print(f"{'策略':>22}{'Δ':>9}{'保量完成率':>12}{'超投率':>10}"
          f"{'保量曝光占比':>14}{'大盘平均分':>12}{'大盘分损失':>12}")
    rows = [("无保量 TopK(score)", 0.0, base)]
    for d in (0.005, 0.010, 0.020, 0.050):
        rows.append(("固定 Δ 补偿", d, run("fixed", d)))
    for name, d, r in rows:
        loss = (base["大盘平均分"] - r["大盘平均分"]) / base["大盘平均分"]
        print(f"{name:>22}{d:>9.3f}{r['完成率']:>12.3f}{r['超投率']:>10.3f}"
              f"{r['保量曝光占比']:>14.4f}{r['大盘平均分']:>12.5f}{loss:>12.2%}")
    print(f"\n无保量时完成率只有 {base['完成率']:.3f}(保量池 score 系统性偏低,"
          f"在 TopK 里被挤出);Δ 越大完成得越快,但 Δ=0.050 已经把大盘分压低"
          f" {(base['大盘平均分'] - run('fixed', 0.05)['大盘平均分']) / base['大盘平均分']:.2%}"
          f",且出现超投。")

    print("\n" + "=" * 104)
    print("实现一改进:自适应 Δ_t(按进度缺口)  vs  实现二:Top100 预留 5% 位置")
    print("=" * 104)
    ad = run("adaptive", 0.050)
    rs = run("reserve", 0.0)
    fx = run("fixed", 0.020)
    print(f"{'策略':>26}{'保量完成率':>12}{'超投率':>10}{'保量曝光占比':>14}"
          f"{'大盘平均分':>12}{'大盘分损失':>12}")
    for name, r in (("固定 Δ=0.020", fx), ("自适应 Δ_max=0.050", ad),
                    ("预留 5 个位置(正文实现二)", rs)):
        loss = (base["大盘平均分"] - r["大盘平均分"]) / base["大盘平均分"]
        print(f"{name:>26}{r['完成率']:>12.3f}{r['超投率']:>10.3f}"
              f"{r['保量曝光占比']:>14.4f}{r['大盘平均分']:>12.5f}{loss:>12.2%}")

    print("\n全天保量进度曲线(完成率 = 累计曝光 / Target):")
    print(f"{'时间片':>8}{'无保量':>12}{'固定Δ=0.02':>14}{'自适应Δ':>12}{'预留5位':>12}"
          f"{'理想匀速':>12}")
    for t in range(0, T_SLOTS, 3):
        print(f"{t+1:>8}{base['curve'][t]:>12.3f}{fx['curve'][t]:>14.3f}"
              f"{ad['curve'][t]:>12.3f}{rs['curve'][t]:>12.3f}{(t+1)/T_SLOTS:>12.3f}")
    print(f"{T_SLOTS:>8}{base['curve'][-1]:>12.3f}{fx['curve'][-1]:>14.3f}"
          f"{ad['curve'][-1]:>12.3f}{rs['curve'][-1]:>12.3f}{1.0:>12.3f}")

    print("\n结论:")
    print(f"1) 分数补偿(实现一)用 Δ 换完成率,Δ 需要在完成率与大盘分之间权衡;"
          f"自适应 Δ_t 在完成率 {ad['完成率']:.3f} 下只损失大盘分 "
          f"{(base['大盘平均分'] - ad['大盘平均分']) / base['大盘平均分']:.2%}。")
    print(f"2) 位置预留(实现二)把完成率钉死在 {rs['完成率']:.3f},"
          f"实现最简单且进度曲线最贴近匀速,代价是无条件占用 {int(RESERVE_RATIO*TOPK)} 个坑位,"
          f"大盘分损失 {(base['大盘平均分'] - rs['大盘平均分']) / base['大盘平均分']:.2%}。")
