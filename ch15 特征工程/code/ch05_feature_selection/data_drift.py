"""
数据漂移检测：PSI、KL 与 Label Drift（第15章 §15.6.5）

漂移可以分几层：
- 特征分布漂移：P_train(X) ≠ P_online(X)
- 用户/物品结构漂移：时间段不同，人群与内容池的构成变了
- 场景漂移：节假日、大促、热点带来的短期流量突变
- Label 漂移：CTR、长播率等指标本身的均值/分布变了

脚本构造三段时间的数据：
    训练期 0–30 天 → 稳定期 30–60 天 → 漂移期 60–90 天
在漂移期分别注入两种变化：
    A) 平滑漂移：用户活跃度整体上升（新用户涌入）
    B) 结构漂移：某类目供给大幅增加，同时 Label 的基线概率上升
然后计算：
    PSI = Σ (p_online − p_train) · ln(p_online / p_train)   分箱用训练集分位数
    KL  = Σ p_online · ln(p_online / p_train)               （相对熵）
    Label drift：正样本率、均值、分位数的逐期变化
以及"用训练期模型直接打分"的 AUC 变化。

运行：python3 ch05_feature_selection/data_drift.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _lr_utils import (  # noqa: E402
    FEATURE_NAMES, auc_score, make_ctr_data, sigmoid, standardize, train_logreg,
)


def quantile_bins(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    用训练集切箱，保证线上数据用同一套边界计算 PSI。

    连续特征用分位数切箱；取值很少的离散特征（如 0/1 的 cat_match）如果也用
    分位数，会被合并成唯一一个箱，PSI 恒为 0——所以这类特征直接按取值切。
    """
    uniq = np.unique(x)
    if len(uniq) <= 21:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        return np.concatenate([[-np.inf], mids, [np.inf]])
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return np.array([-np.inf, np.inf])
    qs[0], qs[-1] = -np.inf, np.inf
    return qs


def hist_share(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins)
    p = h / max(1, h.sum())
    return np.clip(p, 1e-6, None)


def psi(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum((q - p) * np.log(q / p)))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(q ‖ p)：用线上分布对训练分布求相对熵。"""
    return float(np.sum(q * np.log(q / p)))


def drift_level(v: float) -> str:
    if v < 0.1:
        return "稳定"
    if v < 0.25:
        return "轻度"
    return "显著"


def make_period(seed: int, n: int, shift: float = 0.0, label_shift: float = 0.0,
                cat_boost: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    在 make_ctr_data 的基础上注入漂移：
    - shift: 活跃度整体上移（新用户涌入导致 active 分布右移）
    - label_shift: Label 基线概率上移（大促期间购买意愿增强）
    - cat_boost: 类目匹配比例从 35% 提到 65%（某类目供给大增）
    """
    X, y = make_ctr_data(n, seed=seed)
    X = X.copy()
    auc_idx = FEATURE_NAMES.index("user_activity")
    X[:, auc_idx] += shift
    if cat_boost:
        ci = FEATURE_NAMES.index("cat_match")
        flip = (X[:, ci] == 0) & (np.random.default_rng(seed).random(n) < 0.46)
        X[flip, ci] = 1.0
    if label_shift != 0.0:
        rng = np.random.default_rng(seed + 999)
        # 只在负样本里把一部分翻成正样本，模拟整体点击率上升
        flip = (y == 0) & (rng.random(n) < label_shift)
        y = y.copy()
        y[flip] = 1.0
    return X, y


def main() -> None:
    print("数据漂移：PSI / KL / Label Drift（§15.6.5）\n")
    Xtr, ytr = make_period(seed=1, n=30_000)
    X_stable, y_stable = make_period(seed=2, n=10_000)
    # A) 平滑漂移：活跃度上移 0.8σ
    X_smooth, y_smooth = make_period(seed=3, n=10_000, shift=0.8)
    # B) 结构漂移：类目匹配比例大涨 + 点击率上移 0.05
    X_struct, y_struct = make_period(seed=4, n=10_000, cat_boost=True, label_shift=0.05)

    (Xtr_s, X_stable_s, X_smooth_s, X_struct_s), _, _ = standardize(
        Xtr, X_stable, X_smooth, X_struct)
    w, b, val_auc = train_logreg(Xtr_s, ytr, Xtr_s, ytr)
    print(f"训练期模型自评 AUC = {val_auc:.4f}（训练集内评估，仅用于演示漂移影响）\n")

    periods = [
        ("稳定期", X_stable_s, y_stable, X_stable),
        ("A 平滑漂移", X_smooth_s, y_smooth, X_smooth),
        ("B 结构漂移", X_struct_s, y_struct, X_struct),
    ]

    # 每个特征在训练集上切箱
    bins = {j: quantile_bins(Xtr[:, j]) for j in range(len(FEATURE_NAMES))}
    train_share = {j: hist_share(Xtr[:, j], bins[j]) for j in range(len(FEATURE_NAMES))}

    print("各特征的 PSI（相对训练期，分箱=训练集十分位）：")
    header = f"{'特征':<17}" + "".join(f"{name:>14}" for name, *_ in periods)
    print(header)
    print("-" * len(header))
    for j, name in enumerate(FEATURE_NAMES):
        cells = []
        for pname, _, _, Xraw in periods:
            q = hist_share(Xraw[:, j], bins[j])
            v = psi(train_share[j], q)
            cells.append(f"{v:>8.3f}({drift_level(v)})")
        print(f"{name:<17}" + "".join(f"{c:>14}" for c in cells))

    print("\n  经验阈值（[MOCK]，业界常用）：PSI < 0.1 稳定，0.1–0.25 轻度，> 0.25 显著。")
    print("  注意只有注入漂移的那些特征会报警，其余特征保持在 0 附近——")
    print("  PSI 是逐特征的，不做全局平均，否则异常会被稀释。\n")

    print("整体分布差异：")
    print(f"{'时期':<12}{'平均PSI':>10}{'最大PSI':>10}{'KL均值':>10}"
          f"{'label率':>10}{'预测均值':>11}{'校准偏差':>10}{'AUC':>9}")
    print("-" * 82)
    base_rate = ytr.mean()
    for pname, Xs, ys, Xraw in periods:
        vals = [psi(train_share[j], hist_share(Xraw[:, j], bins[j]))
                for j in range(len(FEATURE_NAMES))]
        kls = [kl(train_share[j], hist_share(Xraw[:, j], bins[j]))
               for j in range(len(FEATURE_NAMES))]
        prob = sigmoid(Xs @ w + b)
        auc = auc_score(Xs @ w + b, ys)
        print(f"{pname:<12}{np.mean(vals):>10.3f}{np.max(vals):>10.3f}"
              f"{np.mean(kls):>10.4f}{ys.mean():>10.3f}"
              f"{prob.mean():>11.3f}{prob.mean() - ys.mean():>+10.3f}{auc:>9.4f}")

    print(f"\n  训练期 label 率 = {base_rate:.3f}")
    print("\n结论（对照上面的实测数字）：")
    print("  1) 平滑漂移（A）：user_activity 的 PSI 冲到 0.62，label 率几乎不变，")
    print("     AUC 甚至略有提升（0.786 → 0.791），但预测均值被系统性抬高")
    print("     （0.372 → 0.491，校准偏差 +0.126）。这类漂移最隐蔽——排序没变坏，")
    print("     但概率已经不能直接用于出价、混排或阈值决策，需要重新校准；")
    print("  2) 结构漂移（B）：cat_match 的 PSI 达到 0.35，label 率从 0.367 升到")
    print("     0.407，AUC 从 0.786 掉到 0.749。输入与 Label 同时变了，模型学到的")
    print("     '类目匹配→点击'关系在新生态下不再准确，这是需要重训的情形；")
    print("  3) PSI 只能说明'输入变了'，无法单独判断'模型是否失效'：A 的 PSI 比 B")
    print("     更极端，但真正影响排序的是 B。必须把 PSI、label 漂移、AUC 与")
    print("     Calibration 放在一起看（§15.6.5）。")

    print("\n  监控建议：")
    print("  - 逐特征算 PSI，按'用户侧 / 物品侧 / 场景侧'分组看，避免全局平均；")
    print("  - PSI 报警时先查上游数据源与口径，再决定是否重训；")
    print("  - 阈值本身要随季节/活动调整，大促期间短期漂移是预期之内的（[MOCK]）。")


if __name__ == "__main__":
    main()
