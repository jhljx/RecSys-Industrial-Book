"""
用户/物品/上下文特征的体系化组织（第15章 §15.1）

本脚本把同一批业务数据整理成"特征清单"：每条特征标注来源（用户/物品/上下文/
行为/多模态）、表示形式（Sparse/Dense/Sequence/Embedding）、基数与覆盖率，
用来演示正文的核心观点——特征体系要同时回答"特征来自哪里"和"特征以什么形式
进入模型"两个问题。

运行：python3 ch01_feature_system/user_item_context_features.py
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import make_catalog, make_events, make_users, print_table  # noqa: E402


def feature_inventory(users: list[dict], items: list[dict], events: list[dict]) -> list[list]:
    """按正文的分类整理特征清单，并统计基数/覆盖率。"""
    n_users, n_items, n_events = len(users), len(items), len(events)

    def ratio(pred, seq):
        if not seq:
            return 0.0
        return sum(1 for x in seq if pred(x)) / len(seq)

    # 覆盖率：Dense 特征为"非缺失/非默认值"的比例；Sparse 特征为有取值的比例
    click_events = [e for e in events if e["action"] in ("click", "watch")]
    watch_events = [e for e in events if e["action"] == "watch"]
    interacted = {e["user_id"] for e in click_events}
    popular_items = {e["item_id"] for e in click_events}

    return [
        # 来源, 特征, 表示形式, 基数, 覆盖率, 示例
        ["用户", "user_id", "Sparse", n_users, 1.0, "u_000123"],
        ["用户", "age_band", "Sparse", 5, 0.97, "3"],
        ["用户", "gender", "Sparse", 2, 0.92, "1"],
        ["用户", "city_tier", "Sparse", 4, 0.88, "2"],
        ["用户", "intensity", "Dense", "-", 1.0, "0.7341"],
        ["用户", "ctr_7d", "Dense", "-", f"{len(interacted) / n_users:.2%}", "0.0831"],
        ["用户", "watch_time_30d", "Dense", "-", f"{len(watch_events) and len({e['user_id'] for e in watch_events}) / n_users:.2%}", "412.7s"],
        ["用户", "action_seq", "Sequence", n_users, f"{len(interacted) / n_users:.2%}", "[i12,i88,i31,...]"],
        ["物品", "item_id", "Sparse", n_items, 1.0, "v_000042"],
        ["物品", "category", "Sparse", len({it["category"] for it in items}), 1.0, "sports"],
        ["物品", "author_id", "Sparse", len({it["author"] for it in items}), 1.0, "a_011"],
        ["物品", "duration", "Dense", "-", 1.0, "37.4s"],
        ["物品", "quality", "Dense", "-", 1.0, "0.8123"],
        ["物品", "empirical_ctr", "Dense", "-", f"{len(popular_items) / n_items:.2%}", "0.0612"],
        ["物品", "cover_embedding", "Embedding", "-", 0.63, "d=64"],
        ["上下文", "hour", "Sparse", 24, 1.0, "21"],
        ["上下文", "weekday", "Sparse", 7, 1.0, "5"],
        ["上下文", "network", "Sparse", 4, 1.0, "wifi"],
        ["上下文", "device_price", "Dense", "-", 0.95, "2399"],
        ["上下文", "session_len", "Dense", "-", 1.0, "12"],
        ["上下文", "cand_size", "Dense", "-", 1.0, "200"],
        ["行为", "click_1d/7d/30d", "Dense", "-", 1.0, "3 / 21 / 88"],
        ["行为", "recency_last_click", "Dense", "-", f"{len(interacted) / n_users:.2%}", "0.42h"],
        ["行为", "watch_ratio_7d", "Dense", "-", 0.91, "0.618"],
    ]


def main() -> None:
    rng = random.Random(42)
    users = make_users(200, 8, rng)
    items = make_catalog(400, 8, rng)
    events = make_events(200, 20_000, 400, 8, 30, random.Random(2024))

    rows = feature_inventory(users, items, events)
    # 覆盖率统一成百分比字符串
    for r in rows:
        r[4] = f"{r[4]:.1%}" if isinstance(r[4], float) else r[4]
    print("特征清单（来源 × 表示形式）")
    print_table(["来源", "特征", "表示", "基数", "覆盖率", "示例"], rows)

    by_source: dict[str, int] = {}
    by_repr: dict[str, int] = {}
    for r in rows:
        by_source[r[0]] = by_source.get(r[0], 0) + 1
        by_repr[r[2]] = by_repr.get(r[2], 0) + 1
    print("\n来源分布：", "  ".join(f"{k}={v}" for k, v in by_source.items()))
    print("表示分布：", "  ".join(f"{k}={v}" for k, v in by_repr.items()))

    # ---- 同一信息的不同表示形式：以"用户近7天观看体育类内容次数"为例 ----
    print("\n同一业务信息（7 天内观看体育内容次数 = 6）的四种表示：")
    print_table(
        ["表示形式", "模型输入", "信息损失", "参数量", "备注"],
        [
            ["Sparse ID", "Embedding[6]", "完全丢失数值顺序", "词表大小 × d", "6 与 7 是两个无关类别"],
            ["Dense 原值", "6.0（可标准化）", "无", "0", "保留数值语义，最省参数"],
            ["分桶 Embedding", "Embedding[bucket(6)]", "桶内信息丢失", "桶数 × d", "兼顾非线性与语义"],
            ["Sequence", "[c,c,g,c,c,c]", "无（但更贵）", "序列编码器参数", "保留次序与具体内容"],
        ],
    )

    # ---- 覆盖率与缺失的代价 ----
    missing = 0.38
    usable = 1 - missing
    print(f"\n示例：某商品类 Dense 特征覆盖率 {usable:.0%}，"
          f"若按默认值 0 填充，模型看到的是 {missing:.0%} 的'零'样本，"
          "这部分样本会把该特征的权重向 0 拉偏（正文 §15.6.4 特征质量）。")


if __name__ == "__main__":
    main()
