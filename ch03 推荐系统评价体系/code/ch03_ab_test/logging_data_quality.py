"""对应正文第 3 章 §3.3.3「日志体系与数据质量保障」。

正文原话：推荐系统通常维护两类核心日志——**曝光日志**（用户ID、请求ID、推荐结果
列表、排序位置、用户侧行为历史、实验组信息）与**行为日志**（点击、播放、点赞、
评论、分享、转化）。「通过 req_id 关联两类日志，可以形成完整的『曝光—反馈』闭环。」
并且「工业界通常还会建立：分流比例监控，日志丢失率监控，指标漂移监控，实验异常
告警等，以确保实验结果真实可信。」

本实现做了什么
---------------
1. `[MOCK]` 生成一天的曝光日志与行为日志，并注入四类真实世界里最常见的脏数据：
   **延迟到达**（行为日志晚于统计窗口关闸）、**日志丢失**（曝光/行为各自丢一部分）、
   **重复上报**（客户端重试导致同一事件多条）、**分流信息错误**（配置下发滞后的
   实例把实验组写错）；
2. 实现按 `req_id`（更严格：`(req_id, item_id)`）的**曝光—行为拼接**，并给出
   LEFT JOIN 的 join 率、孤儿行为日志（找不到曝光）比例；
3. 实现正文要求的四类数据质量监控：
   **分流比例监控**（卡方检验 + SRM / Sample Ratio Mismatch 检测）、
   **日志丢失率监控**、**指标漂移监控**（CTR 的 PSI 与同比突变）、**实验异常告警**；
4. 实现**去重**（事件主键 + 幂等）与**延迟到达补偿**（T+1 回溯重算），对比
   「关闸即算」与「T+1 回算」两种口径下的 CTR 偏差；
5. 实现**特征快照一致性校验**：线上打分时落的特征快照 vs 离线用当日特征表重建的
   特征，统计字段级不一致率与由此产生的训练/服务偏差（training-serving skew）。

关键公式
---------
    join 率        = |曝光∩行为(按 req_id,item_id)| / |行为日志|
    日志丢失率     = 1 - 实收条数 / 上报条数
    SRM 卡方       = sum_g (n_g - E_g)^2 / E_g,  E_g = N * 设计占比
    PSI            = sum_b (p_b - q_b) * ln(p_b / q_b)
    快照不一致率   = #{字段值不同} / #{字段数 * 样本数}
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# ==========================================================================
# 两类核心日志的 schema（照抄正文列举的字段）
# ==========================================================================
@dataclass
class ImpressionLog:
    """[MOCK] 曝光日志：正文列举字段 = 用户ID / 请求ID / 推荐结果列表 / 排序位置 /
    用户侧行为历史 / 实验组信息。此处按 (req_id, item_id) 打平成一行一曝光。"""

    req_id: str
    user_id: int
    item_id: int
    position: int                       # 排序位置
    exp_group: str                      # 实验组信息
    hist_len: int                       # 用户侧行为历史长度
    ts_ms: int
    feature_snapshot: dict = field(default_factory=dict)   # 线上打分时的特征快照


@dataclass
class ActionLog:
    """[MOCK] 行为日志：点击/播放/点赞/评论/分享/转化。"""

    req_id: str
    user_id: int
    item_id: int
    action: str                         # click / play / like / comment / share / convert
    value: float                        # 播放时长、成交金额等
    ts_ms: int
    event_id: str                       # 客户端生成的事件主键，用于幂等去重


# ==========================================================================
# [MOCK] 日志生产 + 脏数据注入
# ==========================================================================
FEATURE_KEYS = ("u_ctr_7d", "i_ctr_7d", "i_duration_s", "u_city_id", "i_category_id")

WINDOW_MS = 24 * 3600 * 1000            # 统计窗口：一天
CUTOFF_MS = WINDOW_MS                   # 「关闸即算」的截止时刻
LATE_TOLERANCE_MS = 6 * 3600 * 1000     # T+1 回算额外等待 6 小时


def design_split() -> dict[str, float]:
    """实验设计的分流比例：base 90%、exp_a 5%、exp_b 5%。"""
    return {"base": 0.90, "exp_a": 0.05, "exp_b": 0.05}


def true_group(user_id: int) -> str:
    """md5 分桶得到的「正确」实验组。"""
    b = int(hashlib.md5(f"log_quality:{user_id}".encode()).hexdigest(), 16) % 100
    if b < 90:
        return "base"
    return "exp_a" if b < 95 else "exp_b"


GROUP_CTR = {"base": 0.070, "exp_a": 0.077, "exp_b": 0.0705}


def produce_logs(n_users: int, items_per_user: int,
                 misroute_rate: float = 0.02) -> tuple[list[ImpressionLog], list[ActionLog],
                                                       dict]:
    """[MOCK] 生产原始（上报侧）日志。misroute_rate 模拟配置下发滞后导致的分流信息错误。"""
    imps: list[ImpressionLog] = []
    acts: list[ActionLog] = []
    stats = {"misrouted": 0, "late_actions": 0, "dup_actions": 0}
    for u in range(n_users):
        g = true_group(u)
        logged_g = g
        if random.random() < misroute_rate:         # 旧配置实例写错了实验组
            logged_g = random.choice([x for x in design_split() if x != g])
            stats["misrouted"] += 1
        req_id = f"req_{u:06d}_{random.randrange(1 << 20):05x}"
        base_ts = random.randrange(0, WINDOW_MS)
        hist_len = max(0, int(random.gauss(120, 50)))
        for pos in range(items_per_user):
            item_id = random.randrange(100_000)
            snap = {
                "u_ctr_7d": round(random.betavariate(2, 25), 5),
                "i_ctr_7d": round(random.betavariate(2, 22), 5),
                "i_duration_s": round(random.lognormvariate(3.6, 0.7), 2),
                "u_city_id": u % 340,
                "i_category_id": item_id % 32,
            }
            imps.append(ImpressionLog(req_id, u, item_id, pos, logged_g, hist_len,
                                      base_ts + pos * 50, snap))
            # 位置偏置：越靠后 CTR 越低
            p = GROUP_CTR[g] * math.exp(-0.08 * pos)
            if random.random() < p:
                # 行为时间 = 曝光时间 + 停留，少量行为跨窗口晚到
                delay = int(random.expovariate(1 / 90_000.0))
                ts = base_ts + pos * 50 + delay
                eid = f"ev_{u:06d}_{pos}_{random.randrange(1 << 24):06x}"
                acts.append(ActionLog(req_id, u, item_id, "click",
                                      random.lognormvariate(3.0, 0.8), ts, eid))
                if ts > CUTOFF_MS:
                    stats["late_actions"] += 1
                if random.random() < 0.05:          # 客户端重试 -> 重复上报
                    acts.append(ActionLog(req_id, u, item_id, "click",
                                          random.lognormvariate(3.0, 0.8), ts + 120, eid))
                    stats["dup_actions"] += 1
    return imps, acts, stats


def transport(imps: list[ImpressionLog], acts: list[ActionLog],
              imp_loss: float, act_loss: float) -> tuple[list[ImpressionLog],
                                                         list[ActionLog], dict]:
    """[MOCK] 采集链路（客户端 -> 网关 -> Kafka -> 数仓）丢日志。"""
    kept_i = [x for x in imps if random.random() >= imp_loss]
    kept_a = [x for x in acts if random.random() >= act_loss]
    return kept_i, kept_a, {
        "imp_sent": len(imps), "imp_recv": len(kept_i),
        "act_sent": len(acts), "act_recv": len(kept_a),
        "imp_loss_rate": 1 - len(kept_i) / len(imps),
        "act_loss_rate": 1 - len(kept_a) / len(acts),
    }


# ==========================================================================
# 曝光—反馈拼接（按 req_id 关联）
# ==========================================================================
def dedup_actions(acts: list[ActionLog]) -> tuple[list[ActionLog], int]:
    """按 event_id 幂等去重。"""
    seen: set[str] = set()
    out: list[ActionLog] = []
    for a in acts:
        if a.event_id in seen:
            continue
        seen.add(a.event_id)
        out.append(a)
    return out, len(acts) - len(out)


def join_logs(imps: list[ImpressionLog], acts: list[ActionLog],
              cutoff_ms: int) -> dict:
    """LEFT JOIN：曝光为主表，按 (req_id, item_id) 关联窗口内的行为。"""
    idx: dict[tuple[str, int], ImpressionLog] = {(i.req_id, i.item_id): i for i in imps}
    clicked: set[tuple[str, int]] = set()
    orphan = 0
    dropped_late = 0
    for a in acts:
        if a.ts_ms > cutoff_ms:
            dropped_late += 1
            continue
        k = (a.req_id, a.item_id)
        if k in idx:
            clicked.add(k)
        else:
            orphan += 1                 # 有反馈却找不到曝光：曝光日志丢了
    per_group = defaultdict(lambda: [0, 0])     # group -> [imps, clicks]
    for k, i in idx.items():
        per_group[i.exp_group][0] += 1
        if k in clicked:
            per_group[i.exp_group][1] += 1
    in_window_acts = sum(1 for a in acts if a.ts_ms <= cutoff_ms)
    return {
        "join_rate": len(clicked) / in_window_acts if in_window_acts else 0.0,
        "orphan_actions": orphan,
        "orphan_rate": orphan / in_window_acts if in_window_acts else 0.0,
        "dropped_late": dropped_late,
        "per_group": {g: tuple(v) for g, v in per_group.items()},
    }


# ==========================================================================
# 数据质量监控
# ==========================================================================
def chi2_sf_approx(chi2: float, df: int) -> float:
    """卡方上尾概率（Wilson–Hilferty 立方根正态近似，df>=1 时精度足够演示）。"""
    if chi2 <= 0:
        return 1.0
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))


def srm_check(counts: dict[str, int], design: dict[str, float]) -> dict:
    """分流比例监控 / SRM 检测：观测组样本数是否偏离设计占比。"""
    n = sum(counts.values())
    chi2 = 0.0
    rows = []
    for g, ratio in design.items():
        e = n * ratio
        o = counts.get(g, 0)
        chi2 += (o - e) ** 2 / e
        rows.append((g, o, e, 100.0 * o / n, 100.0 * ratio))
    df = len(design) - 1
    return {"chi2": chi2, "df": df, "p": chi2_sf_approx(chi2, df), "rows": rows, "n": n}


def psi(p_hist: list[float], q_hist: list[float]) -> float:
    """Population Stability Index，用于指标/特征分布漂移监控。"""
    eps = 1e-6
    return sum((p - q) * math.log(max(p, eps) / max(q, eps))
               for p, q in zip(p_hist, q_hist))


def ctr_by_hour(imps: list[ImpressionLog], clicked: set[tuple[str, int]]) -> list[float]:
    num = [0] * 24
    den = [0] * 24
    for i in imps:
        h = min(23, i.ts_ms // 3_600_000)
        den[h] += 1
        if (i.req_id, i.item_id) in clicked:
            num[h] += 1
    return [num[h] / den[h] if den[h] else 0.0 for h in range(24)]


# ==========================================================================
# 特征快照一致性校验
# ==========================================================================
def rebuild_features_offline(imp: ImpressionLog, drift_rate: float) -> dict:
    """[MOCK] 离线用「当日最新特征表」重建特征。
    特征表在曝光之后又被更新过，因此部分字段与线上快照不一致——这正是
    必须落「特征快照」而不能离线重建训练样本的原因。"""
    out = dict(imp.feature_snapshot)
    for k in ("u_ctr_7d", "i_ctr_7d", "i_duration_s"):
        if random.random() < drift_rate:
            out[k] = round(out[k] * random.uniform(0.6, 1.6), 5)
    if random.random() < drift_rate * 0.3:          # 类目被运营改了
        out["i_category_id"] = (out["i_category_id"] + 1) % 32
    return out


def snapshot_consistency(imps: list[ImpressionLog], drift_rate: float) -> dict:
    per_field = Counter()
    rel_err = defaultdict(list)
    n = len(imps)
    bad_rows = 0
    for i in imps:
        rb = rebuild_features_offline(i, drift_rate)
        row_bad = False
        for k in FEATURE_KEYS:
            a, b = i.feature_snapshot[k], rb[k]
            if a != b:
                per_field[k] += 1
                row_bad = True
                if isinstance(a, float) and a:
                    rel_err[k].append(abs(b - a) / a)
        bad_rows += row_bad
    return {
        "n": n,
        "row_inconsistent_rate": bad_rows / n,
        "field_rate": {k: per_field[k] / n for k in FEATURE_KEYS},
        "mean_rel_err": {k: (sum(v) / len(v) if v else 0.0) for k, v in rel_err.items()},
        "overall_field_rate": sum(per_field.values()) / (n * len(FEATURE_KEYS)),
    }


def main() -> None:
    random.seed(42)
    print("=" * 88)
    print("§3.3.3 日志体系与数据质量保障：曝光/行为日志拼接 + 丢失/重复/延迟 + 快照一致性")
    print("=" * 88)

    # ---------- (1) 日志生产 ----------
    print("\n[1] [MOCK] 原始日志生产（上报侧）")
    raw_imps, raw_acts, prod = produce_logs(n_users=9000, items_per_user=8,
                                            misroute_rate=0.02)
    print(f"  曝光日志 {len(raw_imps)} 条，行为日志 {len(raw_acts)} 条")
    print(f"  注入脏数据：实验组写错 {prod['misrouted']} 个用户"
          f"（配置下发滞后）、跨窗口晚到行为 {prod['late_actions']} 条、"
          f"客户端重试重复上报 {prod['dup_actions']} 条")
    print("  曝光日志字段（正文列举）：user_id / req_id / item_id / position / "
          "exp_group / hist_len / ts / feature_snapshot")
    print("  行为日志字段（正文列举）：req_id / user_id / item_id / action / value / "
          "ts / event_id")

    # ---------- (2) 采集链路丢失 ----------
    print("\n[2] 日志丢失率监控（采集链路：客户端 -> 网关 -> Kafka -> 数仓）")
    imps, acts, tp = transport(raw_imps, raw_acts, imp_loss=0.012, act_loss=0.030)
    print(f"  {'日志类型':>10} | {'上报':>9} | {'实收':>9} | {'丢失率':>8} | 阈值 0.5% 告警")
    print("  " + "-" * 62)
    for name, s, r, lr in (("曝光日志", tp["imp_sent"], tp["imp_recv"], tp["imp_loss_rate"]),
                           ("行为日志", tp["act_sent"], tp["act_recv"], tp["act_loss_rate"])):
        flag = "✗ 告警" if lr > 0.005 else "✓"
        print(f"  {name:>10} | {s:>9d} | {r:>9d} | {100 * lr:>7.3f}% | {flag}")
    print("  >>> 行为日志丢失会系统性低估 CTR；曝光日志丢失会产生「孤儿行为」并高估 CTR，")
    print("      两者方向相反，必须分别监控而不能只看总量。")

    # ---------- (3) 去重 ----------
    print("\n[3] 重复上报与幂等去重（按 event_id）")
    deduped, n_dup = dedup_actions(acts)
    print(f"  去重前 {len(acts)} 条，去重后 {len(deduped)} 条，剔除重复 {n_dup} 条"
          f"（{100 * n_dup / len(acts):.2f}%）")
    # 若按「行为条数 / 曝光条数」计算 CTR，重复上报会直接把 CTR 抬高
    n_imp = len(imps)
    ctr_raw = sum(1 for a in acts if a.ts_ms <= CUTOFF_MS) / n_imp
    ctr_dedup = sum(1 for a in deduped if a.ts_ms <= CUTOFF_MS) / n_imp
    print(f"  按条数算 CTR：未去重 {100 * ctr_raw:.4f}% vs 去重后 {100 * ctr_dedup:.4f}%"
          f"（虚高 {100 * (ctr_raw / ctr_dedup - 1):+.3f}%）")
    print("  >>> 指标口径必须是「去重后的 (req_id, item_id) 去重点击数 / 曝光数」，")
    print("      否则一次客户端重试风暴就能把 CTR 指标推上去，形成假的实验收益。")

    # ---------- (4) 曝光—反馈拼接 ----------
    print("\n[4] 按 req_id 关联两类日志，形成「曝光—反馈」闭环")
    j_cut = join_logs(imps, deduped, CUTOFF_MS)
    j_t1 = join_logs(imps, deduped, CUTOFF_MS + LATE_TOLERANCE_MS)
    print(f"  窗口内行为 join 命中率 = {100 * j_cut['join_rate']:.3f}%")
    print(f"  孤儿行为日志（有反馈无曝光）= {j_cut['orphan_actions']} 条"
          f"（{100 * j_cut['orphan_rate']:.3f}%）<- 由曝光日志丢失造成")
    print(f"  被关闸丢弃的晚到行为 = {j_cut['dropped_late']} 条")

    print("\n  延迟到达的影响：『关闸即算』 vs 『T+1 回溯重算（多等 6h）』")
    print(f"  {'实验组':>8} | {'曝光':>9} | {'关闸CTR':>9} | {'T+1 CTR':>9} | {'低估幅度':>9}")
    print("  " + "-" * 56)
    for g in ("base", "exp_a", "exp_b"):
        i0, c0 = j_cut["per_group"][g]
        i1, c1 = j_t1["per_group"][g]
        ctr0, ctr1 = c0 / i0, c1 / i1
        print(f"  {g:>8} | {i0:>9d} | {100 * ctr0:>8.4f}% | {100 * ctr1:>8.4f}% | "
              f"{100 * (ctr0 / ctr1 - 1):>+8.3f}%")

    # 分流信息被写错会把两组样本互相「掺和」，从而稀释真实的实验效果。
    # 这里对同一批拼接结果用两种分组口径各算一次 exp_a 的 CTR 提升：
    #   (a) 直接用曝光日志里记录的 exp_group（含 2% 写错的用户）；
    #   (b) 用分桶函数按 user_id 重算实验组（正确分组）。
    clicked = {(a.req_id, a.item_id) for a in deduped if a.ts_ms <= CUTOFF_MS}
    design_lift = GROUP_CTR["exp_a"] / GROUP_CTR["base"] - 1
    print(f"\n  exp_a 相对 base 的 CTR 提升（设计值 {100 * design_lift:+.2f}%）")
    print(f"  {'分组口径':>16} | {'base CTR':>9} | {'exp_a CTR':>9} | {'相对提升':>9}"
          f" | {'95% 置信区间':>20}")
    print("  " + "-" * 78)
    for title, keyfn in (("日志记录的分组", lambda i: i.exp_group),
                         ("分桶函数重算", lambda i: true_group(i.user_id))):
        agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for i in imps:
            g = keyfn(i)
            agg[g][0] += 1
            if (i.req_id, i.item_id) in clicked:
                agg[g][1] += 1
        n_b, c_b = agg["base"]
        n_e, c_e = agg["exp_a"]
        p_b, p_e = c_b / n_b, c_e / n_e
        ratio = p_e / p_b
        # delta 方法：Var(p_e/p_b) ≈ ratio^2 * (Var(p_e)/p_e^2 + Var(p_b)/p_b^2)
        se = ratio * math.sqrt(p_e * (1 - p_e) / n_e / p_e ** 2
                               + p_b * (1 - p_b) / n_b / p_b ** 2)
        lo, hi = ratio - 1 - 1.96 * se, ratio - 1 + 1.96 * se
        print(f"  {title:>16} | {100 * p_b:>8.4f}% | {100 * p_e:>8.4f}% | "
              f"{100 * (ratio - 1):>+8.3f}% | "
              f"[{100 * lo:>+7.2f}%, {100 * hi:>+7.2f}%]")
    print("  >>> 延迟到达对两组近似等比例，对「相对提升」影响小，但对「绝对指标值」")
    print("      有系统性低估——日报口径必须固定，否则无法跨天比较。")
    print("  >>> 两种分组口径的点估计相差数个百分点，而 5% 小流量 + 单日样本的置信区间宽达")
    print("      ±十几个百分点（同时覆盖 0 与设计值 +10%）：既说明分流信息写错会把点估计")
    print("      拉偏，也说明此量级样本根本无法判定 +10% 的收益，必须按 §3.3.4 放量。")

    # ---------- (5) 分流比例监控 / SRM ----------
    print("\n[5] 分流比例监控（SRM: Sample Ratio Mismatch，卡方检验）")
    design = design_split()
    logged_users = defaultdict(set)
    true_users = defaultdict(set)
    for i in imps:
        logged_users[i.exp_group].add(i.user_id)
        true_users[true_group(i.user_id)].add(i.user_id)
    for title, counts in (("日志中记录的实验组（含写错的）",
                           {g: len(s) for g, s in logged_users.items()}),
                          ("分桶函数计算的正确实验组",
                           {g: len(s) for g, s in true_users.items()})):
        r = srm_check(counts, design)
        print(f"\n  {title}（N={r['n']}）")
        print(f"  {'组':>8} | {'观测':>7} | {'期望':>9} | {'观测占比':>9} | {'设计占比':>9}")
        print("  " + "-" * 54)
        for g, o, e, op, dp in sorted(r["rows"]):
            print(f"  {g:>8} | {o:>7d} | {e:>9.1f} | {op:>8.3f}% | {dp:>8.3f}%")
        verdict = "✗ SRM 告警，实验数据不可用" if r["p"] < 0.001 else "✓ 分流比例正常"
        print(f"  chi2 = {r['chi2']:.3f} (df={r['df']}), p = {r['p']:.3e} -> {verdict}")
    print("  >>> 工业界惯例：SRM 的 p < 0.001 即判定实验失效，必须先修分流/日志再看指标，")
    print("      因为分流偏差会同时污染所有指标，任何「显著结论」都不可信。")

    # ---------- (6) 指标漂移监控 ----------
    print("\n[6] 指标漂移监控：CTR 按小时的分布 PSI + 突变检测")
    idx = {(i.req_id, i.item_id): i for i in imps}
    clicked = {(a.req_id, a.item_id) for a in deduped
               if a.ts_ms <= CUTOFF_MS and (a.req_id, a.item_id) in idx}
    today = ctr_by_hour(imps, clicked)
    # [MOCK] 昨日同期：整体同形状，但 14 点有一次埋点故障导致 CTR 腰斩
    yday = [c * random.uniform(0.97, 1.03) for c in today]
    broken = list(today)
    broken[14] = broken[14] * 0.45
    s_t, s_y = sum(broken), sum(yday)
    psi_val = psi([c / s_t for c in broken], [c / s_y for c in yday])
    print(f"  {'小时':>5} | {'今日CTR':>9} | {'昨日CTR':>9} | {'同比':>9} | 突变判定(|同比|>25%)")
    print("  " + "-" * 62)
    for h in range(0, 24, 2):
        yoy = 100 * (broken[h] / yday[h] - 1)
        flag = "✗ 突变告警" if abs(yoy) > 25 else ""
        print(f"  {h:>5d} | {100 * broken[h]:>8.4f}% | {100 * yday[h]:>8.4f}% | "
              f"{yoy:>+8.2f}% | {flag}")
    print("      (14 点的腰斩来自 [MOCK] 注入的埋点故障：该小时 click 事件未上报)")
    print(f"  今日 vs 昨日 CTR 小时分布 PSI = {psi_val:.5f} "
          f"（经验阈值：<0.1 稳定 / 0.1~0.25 需关注 / >0.25 显著漂移）")

    # ---------- (7) 特征快照一致性 ----------
    print("\n[7] 特征快照一致性校验（线上快照 vs 离线重建）")
    sample = imps[:4000]
    for drift in (0.00, 0.03, 0.15):
        r = snapshot_consistency(sample, drift)
        print(f"\n  离线特征表漂移率 drift_rate = {drift:.2f}（N={r['n']}）")
        print(f"  整体字段不一致率 = {100 * r['overall_field_rate']:.3f}%，"
              f"至少一个字段不一致的样本占比 = {100 * r['row_inconsistent_rate']:.3f}%")
        print(f"  {'字段':>16} | {'不一致率':>9} | {'平均相对误差':>12}")
        print("  " + "-" * 44)
        for k in FEATURE_KEYS:
            print(f"  {k:>16} | {100 * r['field_rate'][k]:>8.3f}% | "
                  f"{100 * r['mean_rel_err'].get(k, 0.0):>11.3f}%")
    print("\n  >>> drift_rate=0 对应「训练样本直接用线上快照」：一致率 100%，无 "
          "training-serving skew；")
    print("      drift_rate>0 对应「离线重建特征」：特征表在曝光后被刷新过，重建值与")
    print("      线上打分值不符，模型离线 AUC 再高，线上也兑现不了。")
    print("  >>> 正文：实验结论的可靠性最终取决于数据质量；需建立分流比例监控、")
    print("      日志丢失率监控、指标漂移监控、实验异常告警，以确保实验结果真实可信。")



if __name__ == "__main__":
    main()
