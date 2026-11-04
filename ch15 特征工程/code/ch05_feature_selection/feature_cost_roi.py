"""
特征成本与 ROI（第15章 §15.6.3）

正文把特征成本拆成四项：
    C(f) = C_计算 + C_存储 + C_网络 + C_延迟
并强调特征选择看的是"收益是否值得成本"，而不是"能不能提升离线指标"。

本脚本用 [MOCK] 的单价给 12 个特征估一份成本账：
- 计算：离线/实时任务折算到"每千次请求"的 CPU 成本
- 存储：Embedding/统计值占用的内存或磁盘
- 网络：跨服务 RPC 拉取的传输开销
- 延迟：在线请求链路上新增的 p99 耗时（最贵的一项）
再结合上一节测得的边际 ΔAUC，计算每秒延迟预算换来的 AUC 提升（ROI），
按 ROI 排序给出"高收益低成本 / 高成本低收益"四个象限。

所有单价均为 [MOCK]：用于演示量级关系，不代表任何真实系统的报价。

运行：python3 ch05_feature_selection/feature_cost_roi.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _lr_utils import FEATURE_NAMES  # noqa: E402


# 用 [MOCK] 单价，单位：元 / 百万次请求（计算、存储、网络），毫秒（延迟）
MOCK_COST = {
    # 特征: (计算, 存储, 网络, 延迟ms)
    "user_activity":   (12.0, 0.8, 0.2, 0.90),   # 聚合大窗口，实时任务
    "clicks_7d":       (6.0, 0.6, 0.1, 0.40),
    "clicks_30d":      (4.5, 0.7, 0.1, 0.45),
    "item_ctr":        (3.0, 0.3, 0.2, 0.25),
    "item_popularity": (2.0, 0.1, 0.1, 0.05),
    "cat_match":       (0.3, 0.02, 0.0, 0.01),
    "hist_len":        (1.5, 0.1, 0.1, 0.08),
    "follow_ratio":    (2.5, 0.2, 0.5, 0.30),    # 跨服务 RPC
    "is_new_user":     (0.2, 0.01, 0.0, 0.01),
    "hour_bucket":     (0.0, 0.0, 0.0, 0.00),
    "device":          (0.0, 0.0, 0.0, 0.00),
    "price_band":      (1.0, 0.05, 0.1, 0.05),
}

# 上一节（feature_redundancy.py）实测的前向边际 ΔAUC，按同样顺序
MEASURED_DELTA_AUC = {
    "user_activity": 0.2173, "item_ctr": 0.0432, "cat_match": 0.0186,
    "clicks_7d": 0.0062, "clicks_30d": 0.0000, "hour_bucket": -0.0001,
    "is_new_user": -0.0004, "hist_len": -0.0002, "device": 0.0004,
    "price_band": -0.0001, "item_popularity": 0.0000, "follow_ratio": 0.0001,
}

# 延迟的换算：1ms p99 延迟折算成的"单位成本"
# [MOCK] 假设 1ms ≈ 3.0 元/百万次请求（与计算成本同量级，实际通常更贵）
LATENCY_UNIT = 3.0


def main() -> None:
    print("特征成本与 ROI（§15.6.3）\n")
    print("[MOCK] 单价说明：计算/存储/网络 = 元每百万次请求；延迟按 1ms ≈ "
          f"{LATENCY_UNIT} 元/百万次折算\n")

    rows = []
    for name in FEATURE_NAMES:
        calc, store, net, lat = MOCK_COST[name]
        total = calc + store + net + lat * LATENCY_UNIT
        delta = MEASURED_DELTA_AUC[name]
        roi = (delta / total * 1000) if total > 0 else float("inf")   # 每千元成本换来的 ΔAUC
        rows.append((name, calc, store, net, lat, total, delta, roi))

    rows.sort(key=lambda r: -r[-1])

    print(f"{'特征':<17}{'计算':>6}{'存储':>6}{'网络':>6}{'延迟ms':>8}"
          f"{'总成本':>8}{'ΔAUC':>9}{'ROI':>9}")
    print("-" * 76)
    for name, calc, store, net, lat, total, delta, roi in rows:
        print(f"{name:<17}{calc:>6.1f}{store:>6.2f}{net:>6.2f}{lat:>8.2f}"
              f"{total:>8.2f}{delta:>+9.4f}{roi:>9.2f}")

    print("\nROI = ΔAUC / 总成本 × 1000（每千元/百万请求成本换来的 AUC 增益）")
    print("  ROI 越高，越值得保留；负数说明在已有特征下没有边际收益。\n")

    # 四象限
    med_cost = sorted(r[5] for r in rows)[len(rows) // 2]
    print(f"按总成本中位数 {med_cost:.2f} 划分四个象限：")
    quad = {"高收益低/中成本（优先保留）": [], "高收益高成本（按需保留）": [],
            "低收益低成本（可留可删）": [], "低收益高成本（优先下线）": []}
    for name, *_, total, delta, roi in rows:
        if delta > 0.001 and total <= med_cost:
            quad["高收益低/中成本（优先保留）"].append(name)
        elif delta > 0.001:
            quad["高收益高成本（按需保留）"].append(name)
        elif total <= med_cost:
            quad["低收益低成本（可留可删）"].append(name)
        else:
            quad["低收益高成本（优先下线）"].append(name)
    for q, names in quad.items():
        print(f"  {q}：{', '.join(names) if names else '（无）'}")

    # 逐项拆解：看谁最贵
    print("\n成本结构拆解（各特征占总成本的比例）：")
    total_all = sum(r[5] for r in rows)
    for name, calc, store, net, lat, total, delta, roi in sorted(rows, key=lambda r: -r[5])[:5]:
        parts = {
            "计算": calc, "存储": store, "网络": net, "延迟": lat * LATENCY_UNIT,
        }
        desc = "  ".join(f"{k}={v / total:.0%}" for k, v in parts.items() if v > 0)
        print(f"  {name:<17} 总成本 {total:5.2f}（占 {total / total_all:4.1%}）  {desc}")

    print("\n工程读法：")
    print("  1) 延迟通常是最贵的一项——同一份统计离线算好、在线只查表，成本能降一个量级；")
    print("  2) 高成本低收益的特征不一定要直接删，可以先降频（7d → 1d）、降维、")
    print("     或者改成离线特征，观察效果是否保持；")
    print("  3) ROI 为负不代表特征本身无用，而是'在现有特征集合上'没有增量，")
    print("     换一个特征集合或换一个业务目标，结论可能反转（§15.6.2）。")


if __name__ == "__main__":
    main()
