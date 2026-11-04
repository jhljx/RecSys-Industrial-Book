"""
特征治理：注册表、生命周期与下线决策（第15章 §15.6.7）

正文列出的治理要素：定义、负责人、来源、新鲜度、覆盖率、成本、质量、漂移、
使用情况、生命周期。单个模型开发者记不住这些，所以需要平台化的注册表。

脚本用 [MOCK] 数据模拟一个 12 条记录的特征注册表，把前几节算出来的指标
（重要性、成本、覆盖率、漂移）汇到一张表里，然后：

1) 按生命周期状态分组：上线 → 观察 → 待下线 → 实验（已下线的记录会移出注册表）
2) 用决策规则自动给出处置建议（保留 / 观察 / 下线 / 补文档）
3) 计算"僵尸特征"比例：长期无人使用、指标又不好的特征
4) 打印一次下线评审需要收集的证据清单

所有数据均为 [MOCK]，用于演示治理流程，不对应任何真实系统。

运行：python3 ch05_feature_selection/feature_governance.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# 前几节实测/估算的指标，作为注册表的一部分
DELTA_AUC = {
    "user_activity": 0.2173, "item_ctr": 0.0432, "cat_match": 0.0186,
    "clicks_7d": 0.0062, "clicks_30d": 0.0000, "hour_bucket": -0.0001,
    "is_new_user": -0.0004, "hist_len": -0.0002, "device": 0.0004,
    "price_band": -0.0001, "item_popularity": 0.0000, "follow_ratio": 0.0001,
}
COST = {
    "user_activity": 15.70, "clicks_7d": 7.90, "clicks_30d": 6.65,
    "item_ctr": 4.25, "follow_ratio": 4.10, "item_popularity": 2.35,
    "hist_len": 1.94, "price_band": 1.30, "is_new_user": 0.24,
    "cat_match": 0.33, "hour_bucket": 0.0, "device": 0.0,
}

# [MOCK] 注册表：owner / source / 更新频率 / 覆盖率 / 使用方 / 状态
REGISTRY = [
    # name, owner, source, freshness, coverage, users, state
    ("user_activity", "推荐特征组", "实时流 + 离线聚合", "小时级", 0.98, 6, "上线"),
    ("clicks_7d", "推荐特征组", "离线宽表", "天级", 0.97, 6, "上线"),
    ("clicks_30d", "推荐特征组", "离线宽表", "天级", 0.97, 6, "待下线"),
    ("item_ctr", "内容理解组", "离线统计", "天级", 0.95, 5, "上线"),
    ("item_popularity", "内容理解组", "离线统计", "天级", 0.99, 1, "观察"),
    ("cat_match", "推荐特征组", "在线实时计算", "请求级", 1.00, 5, "上线"),
    ("hist_len", "推荐特征组", "离线宽表", "天级", 0.93, 2, "观察"),
    ("follow_ratio", "社交组", "跨服务 RPC", "小时级", 0.88, 1, "待下线"),
    ("is_new_user", "用户画像组", "用户属性", "天级", 0.60, 4, "观察"),
    ("hour_bucket", "推荐特征组", "请求自带", "请求级", 1.00, 6, "上线"),
    ("device", "推荐特征组", "请求自带", "请求级", 1.00, 6, "上线"),
    ("price_band", "电商组", "离线统计", "周级", 0.72, 1, "实验"),
]

STATE_ORDER = ["上线", "观察", "待下线", "实验"]


def decide(name: str, coverage: float, users: int, state: str) -> tuple[str, str]:
    """
    治理规则（[MOCK]）：综合边际收益、成本、覆盖率、使用方数量。

    已经进入"待下线"评审的特征优先按评审结论走；其余特征按指标判断。
    """
    delta = DELTA_AUC[name]
    cost = COST[name]
    if state == "待下线":
        if delta > 0.001:
            return "复核", "已列入下线但仍有边际收益，需确认是否误判"
        return "下线", "无增量，沿用既有下线排期"
    if delta > 0.001 and coverage >= 0.9:
        return "保留", "有边际收益且覆盖稳定"
    if delta > 0.001:
        return "观察", f"有收益但覆盖率仅 {coverage:.0%}，先补数据源"
    if users >= 4:
        return "保留", "无增量但被多个模型依赖，删之前需评估下游"
    if cost > 3.0:
        return "下线", "无增量、成本高、使用方少"
    return "待讨论", "无增量、低成本，可留在观察区"


def main() -> None:
    print("特征治理：注册表、生命周期与处置建议（§15.6.7）\n")

    print("① 按状态分组：")
    for state in STATE_ORDER:
        names = [r[0] for r in REGISTRY if r[6] == state]
        print(f"  {state:<6}（{len(names):>2}）: {', '.join(names) if names else '（无）'}")

    print("\n② 注册表全貌（按状态 + 边际收益排序）：")
    rows = sorted(REGISTRY, key=lambda r: (STATE_ORDER.index(r[6]), -DELTA_AUC[r[0]]))
    print(f"{'特征':<17}{'状态':<7}{'负责人':<12}{'来源':<18}{'更新':<8}"
          f"{'覆盖':>6}{'成本':>7}{'ΔAUC':>8}{'使用方':>7}")
    print("-" * 96)
    for name, owner, source, fresh, cov, users, state in rows:
        print(f"{name:<17}{state:<7}{owner:<12}{source:<18}{fresh:<8}"
              f"{cov:>6.0%}{COST[name]:>7.2f}{DELTA_AUC[name]:>+8.4f}{users:>7}")

    print("\n③ 自动处置建议：")
    print(f"{'特征':<17}{'建议':<8}{'理由':<44}")
    print("-" * 71)
    suggestions = {}
    for name, owner, source, fresh, cov, users, state in REGISTRY:
        action, why = decide(name, cov, users, state)
        suggestions[name] = action
        print(f"{name:<17}{action:<8}{why:<44}")

    counts = {}
    for a in suggestions.values():
        counts[a] = counts.get(a, 0) + 1
    print("\n  处置分布：" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # 僵尸特征：无增量 + 使用方 <= 1
    zombies = [r[0] for r in REGISTRY
               if DELTA_AUC[r[0]] <= 0.001 and r[5] <= 1]
    print(f"\n④ '僵尸特征'（长期低收益 + 使用方 ≤ 1）：{len(zombies)} 个")
    print(f"  {', '.join(zombies) if zombies else '（无）'}")
    print("  这些特征是特征负债的主要来源：没人敢删，但也说不清价值。")

    # 低覆盖特征：即使当前边际 ΔAUC 小，也可能因为只覆盖少数用户而"看起来没用"
    low_cov = [r[0] for r in REGISTRY if r[4] < 0.9]
    print(f"\n⑤ 覆盖率不足 90% 的特征：{', '.join(low_cov) if low_cov else '（无）'}")
    print("  这类特征不建议直接下线：低覆盖会让它在离线评估里天然吃亏，")
    print("  优先级是补数据源、加兜底、做降级策略，拿回覆盖后再评一次。")

    print("\n⑥ 下线评审证据清单（一个特征走到'下线'需要准备）：")
    checklist = [
        "离线消融：整组删除后的 ΔAUC / LogLoss 变化（本目录 feature_redundancy.py）",
        "成本核算：计算 / 存储 / 网络 / 延迟四项的现值（feature_cost_roi.py）",
        "质量与漂移：近 30 天覆盖率、新鲜度、PSI 记录（feature_quality / data_drift）",
        "使用方确认：所有依赖该特征的模型与规则已排期改造",
        "灰度计划：小流量关闭 → 观测核心指标 → 全量 → 保留回滚开关",
        "血缘归档：数据源、计算任务、下游模型记录留档，便于未来复盘",
    ]
    for i, item in enumerate(checklist, 1):
        print(f"  {i}. {item}")

    print("\n治理的本质是让'加特征'和'删特征'都变成有据可查的流程，")
    print("而不是靠某个人的记忆。平台承担监控与告警，评审会承担决策（§15.6.7）。")


if __name__ == "__main__":
    main()
