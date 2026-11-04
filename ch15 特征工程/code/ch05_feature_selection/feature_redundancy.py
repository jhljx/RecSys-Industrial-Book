"""
特征冗余与分批次消融（第15章 §15.6.2）

正文强调：判断一个特征要不要留，看的是它在已有特征集合上的边际收益
    ΔM = M(F ∪ {f}) − M(F)
而不是它单独的重要性。脚本做三件事：

1) 识别冗余
   - 计算特征间相关系数矩阵，|r| 高的对子单独列出
   - 对每个冗余对，比较"留前一个 / 留后一个 / 都留"的验证 AUC

2) 前向边际增益
   - 按重要性从高到低逐个加入特征，打印每一步的 ΔAUC
   - ΔAUC 掉到接近 0 的位置，就是"再加也无意义"的天然停点

3) 分批次消融
   - 把特征按业务含义分组（身份/行为/物品/上下文），整组删除后重训
   - 对照正文的 ΔM_G = M(F) − M(F∖G)

运行：python3 ch05_feature_selection/feature_redundancy.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _lr_utils import (  # noqa: E402
    FEATURE_NAMES, N_FEATURES, auc_score, make_ctr_data, standardize, train_logreg,
)

# 按业务含义分组（对应正文说的"按来源/业务含义/相关性分组"）
GROUPS = {
    "用户身份": ["user_activity", "is_new_user", "device"],
    "用户行为": ["clicks_7d", "clicks_30d", "hist_len", "follow_ratio"],
    "物品属性": ["item_ctr", "item_popularity", "cat_match", "price_band"],
    "上下文": ["hour_bucket"],
}


def fit_eval(cols, data, steps=300):
    Xtr, ytr, Xva, yva = data
    w, b, auc = train_logreg(Xtr[:, cols], ytr, Xva[:, cols], yva, steps=steps)
    return auc, w, b


def main() -> None:
    print("特征冗余：相关性 → 边际增益 → 分组消融（§15.6.2）\n")
    X, y = make_ctr_data(30_000, seed=1)
    Xva_raw, yva = make_ctr_data(6_000, seed=2)
    (Xtr, Xva), _, _ = standardize(X, Xva_raw)
    data = (Xtr, y, Xva, yva)

    base_auc, _, _ = fit_eval(list(range(N_FEATURES)), data)
    print(f"全特征基线：val AUC = {base_auc:.4f}\n")

    # ---------- 1) 相关性 ----------
    print("① 特征相关性矩阵（|r| > 0.5 的对子）：")
    corr = np.corrcoef(Xtr.T)
    pairs = []
    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            if abs(corr[i, j]) > 0.5:
                pairs.append((i, j, corr[i, j]))
    pairs.sort(key=lambda t: -abs(t[2]))
    if not pairs:
        print("  （无）")
    for i, j, r in pairs:
        print(f"  {FEATURE_NAMES[i]:<16} ↔ {FEATURE_NAMES[j]:<16} r = {r:+.3f}")

    print("\n  冗余对的处理对照：")
    print(f"  {'特征组合':<40}{'val AUC':>10}{'vs 全量':>10}")
    print("  " + "-" * 60)
    print(f"  {'全量 12 个特征':<40}{base_auc:>10.4f}{0.0:>+10.4f}")
    all_cols = set(range(N_FEATURES))
    for i, j, r in pairs:
        if abs(r) < 0.7:
            continue
        for keep, drop in ((i, j), (j, i)):
            cols = sorted(all_cols - {drop})
            auc, _, _ = fit_eval(cols, data)
            label = f"去掉 {FEATURE_NAMES[drop]}（留 {FEATURE_NAMES[keep]}）"
            print(f"  {label:<40}{auc:>10.4f}{auc - base_auc:>+10.4f}")
    print("\n  如果删掉其中一个几乎不掉点，说明两者信息高度重叠；")
    print("  保留哪一个要看更新频率、覆盖率与计算成本（§15.6.3），而不是只看 AUC。")

    # ---------- 2) 前向边际增益 ----------
    print("\n② 前向边际增益 ΔM = M(F∪{f}) − M(F)（按单特征 |w| 排序加入）：")
    w_all, _, _ = train_logreg(Xtr, y, Xva, yva, steps=400)
    order = list(np.argsort(-np.abs(w_all)))
    current: list[int] = []
    prev_auc = 0.5
    rows = []
    for step, j in enumerate(order, 1):
        current.append(j)
        auc, _, _ = fit_eval(current, data, steps=250)
        rows.append([step, FEATURE_NAMES[j], len(current), f"{auc:.4f}",
                     f"{auc - prev_auc:+.4f}"])
        prev_auc = auc
    print(f"  {'#':<3}{'新增特征':<18}{'规模':>5}{'AUC':>10}{'ΔAUC':>10}")
    print("  " + "-" * 47)
    for r in rows:
        print(f"  {r[0]:<3}{r[1]:<18}{r[2]:>5}{r[3]:>10}{r[4]:>10}")

    # ---------- 3) 分组消融 ----------
    print("\n③ 分批次消融：整组删除后重训")
    print(f"  {'删除的组':<12}{'剩余特征数':>10}{'val AUC':>10}{'ΔAUC':>10}")
    print("  " + "-" * 44)
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for group, names in GROUPS.items():
        cols = [j for j in range(N_FEATURES) if FEATURE_NAMES[j] not in names]
        auc, _, _ = fit_eval(cols, data)
        print(f"  {group:<12}{len(cols):>10}{auc:>10.4f}{base_auc - auc:>+10.4f}")
    print("\n  ΔAUC 小（<0.002）的组可以进入下一轮精简；但注意两点：")
    print("  1) 组内特征可能有协同，整组删除不掉点不代表单个删除也不掉点；")
    print("  2) 离线 AUC 不是唯一标准，曝光结构、长尾内容等业务指标也要看（§15.6.2）。")
    print("  因此真实流程是：离线分组消融 → 小流量实验 → 全量验证 → 正式下线。")


if __name__ == "__main__":
    main()
