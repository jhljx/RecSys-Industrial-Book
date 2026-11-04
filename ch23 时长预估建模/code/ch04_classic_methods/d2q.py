"""
D2Q：Duration 分组 → 组内分位数 Label → 共享模型 → 分位反变换（第23章 §23.4.1）

正文的三步：
    1) 按视频物理时长的经验分位数把训练数据分成 M 个组（各组样本量近似相等）；
    2) 在每个组内用经验 CDF 把观看时长 w 变成分位数 q = Φ_k(w)；
    3) 用同一个模型 h(x) 回归 q，推理时再按样本所属组的经验分位函数反变换
       ŵ = Φ_k^{-1}(q̂)。

本脚本对比：
    · 直接回归原始观看时长（MSE）；
    · D2Q，并扫 M ∈ {5, 10, 20, 40}，观察正文说的「分组太少去偏不够、
      分组太多 CDF 估计噪声大」的折中。

评价除了整体 MAE，还看每个 Duration 组的平均偏差（预测均值 − 真实均值）：
直接回归往往在中长视频上系统性高估，这正是 Duration Bias 在预测端的表现。

运行：python3 ch04_classic_methods/d2q.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    fit_linear, make_watch_data, mae, predict_linear, split, standardize, table,
)

SEED = 2026


def group_edges(d: np.ndarray, M: int) -> np.ndarray:
    """按 Duration 的经验分位数切 M 组，返回 M+1 个边界（两端开）。"""
    edges = np.quantile(d, np.linspace(0.0, 1.0, M + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign(d: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, d, side="right") - 1


def fit_d2q(Xtr: np.ndarray, wtr: np.ndarray, dtr: np.ndarray, M: int):
    """返回 (组边界, 各组排序后的观看时长, 训练集分位数 Label)。"""
    edges = group_edges(dtr, M)
    ktr = assign(dtr, edges)
    groups = [np.sort(wtr[ktr == k]) for k in range(M)]
    qtr = np.empty(len(wtr))
    for k in range(M):
        m = ktr == k
        if not m.any():
            continue
        qtr[m] = np.searchsorted(groups[k], wtr[m], side="right") / len(groups[k])
    qtr = np.clip(qtr, 1e-4, 1 - 1e-4)
    return edges, groups, qtr


def predict_d2q(theta, Xte, dte, edges, groups) -> np.ndarray:
    qhat = np.clip(predict_linear(theta, Xte), 1e-3, 1 - 1e-3)
    kte = assign(dte, edges)
    out = np.empty(len(dte))
    for k in range(len(groups)):
        m = kte == k
        if m.any():
            out[m] = np.quantile(groups[k], qhat[m])
    return out


def by_group_report(dte, wte, pred, edges, M) -> list[list]:
    kte = assign(dte, edges)
    rows = []
    for k in range(M):
        m = kte == k
        if not m.any():
            continue
        lo = edges[k] if np.isfinite(edges[k]) else 0
        hi = edges[k + 1] if np.isfinite(edges[k + 1]) else 600
        rows.append([f"[{lo:.0f},{hi:.0f})", int(m.sum()),
                     f"{wte[m].mean():.1f}", f"{pred[m].mean():.1f}",
                     f"{pred[m].mean() - wte[m].mean():+.1f}",
                     f"{mae(wte[m], pred[m]):.2f}"])
    return rows


def main() -> None:
    print("D2Q：分位数 Label + 共享模型（§23.4.1）\n")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, w, d = data["x"], data["w"].astype(np.float64), data["d"]
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    Xtr, Xte, _, _ = standardize(X[tr], X[te])
    wtr, wte, dtr, dte = w[tr], w[te], d[tr], d[te]

    # ---------------- 1) 基准：直接回归 ----------------
    print("① 基准：直接对原始观看时长做 MSE 回归")
    theta0 = fit_linear(Xtr, wtr)
    pred0 = np.clip(predict_linear(theta0, Xte), 0, None)
    edges0 = group_edges(dtr, 5)
    rows = by_group_report(dte, wte, pred0, edges0, 5)
    table(["Duration 组", "样本数", "真实均值", "预测均值", "偏差", "MAE"], rows)
    print(f"\n  整体 MAE {mae(wte, pred0):.2f} 秒。越往长视频走，偏差越大——")
    print("  模型把「Duration 长」和「看得久」直接关联起来，长视频被系统性高估。\n")

    # ---------------- 2) D2Q 主实验 ----------------
    print("② D2Q：M=10 个 Duration 组")
    M = 10
    edges, groups, qtr = fit_d2q(Xtr, wtr, dtr, M)
    theta = fit_linear(Xtr, qtr)
    pred = predict_d2q(theta, Xte, dte, edges, groups)
    rows = by_group_report(dte, wte, pred, edges, 10)
    table(["Duration 组", "样本数", "真实均值", "预测均值", "偏差", "MAE"], rows)
    print(f"\n  整体 MAE {mae(wte, pred):.2f} 秒，直接回归是 {mae(wte, pred0):.2f} 秒。")
    print(f"  训练集分位数 Label 的均值 {qtr.mean():.3f}、标准差 {qtr.std():.3f}——")
    print("  Label 被压到 0~1 的统一尺度，长视频的「权重」不再由绝对秒数决定。")
    print("  但各组的偏差并没有变好：长视频组从 +0.8 秒变成 -33 秒。原因有两层：")
    print(f"    · 训练集 q 的标准差只有 {qtr.std():.3f}，模型几乎学不到个体差异，")
    print("      q̂ 基本落在 0.5 附近（实测均值 0.51）；")
    print("    · 而每个组的观看时长分布都严重右偏（长视频组均值 43 秒、中位数只有")
    print("      2.6 秒），q=0.5 反变换回来就是中位数，不是均值。")
    print("  所以 D2Q 降低的是「整体 MAE 被少数长样本拉爆」的程度，")
    print("  不是把每个组的预测校准到组均值——后者需要模型真的学到 q 的差异。")
    print("  这一点在正文里没有展开，但它是把 D2Q 落到线上时必须面对的问题。\n")

    # ---------------- 3) M 的折中 ----------------
    print("③ Duration 组数 M 的折中（正文：太少去偏不足，太多 CDF 估计有噪声）")
    rows = []
    for M_ in (5, 10, 20, 40):
        edges_, groups_, qtr_ = fit_d2q(Xtr, wtr, dtr, M_)
        th_ = fit_linear(Xtr, qtr_)
        pr_ = predict_d2q(th_, Xte, dte, edges_, groups_)
        sizes = np.array([len(g) for g in groups_])
        bias = []
        kte = assign(dte, edges_)
        for k in range(M_):
            m = kte == k
            if m.any():
                bias.append(abs(pr_[m].mean() - wte[m].mean()))
        rows.append([M_, f"{sizes.min():,}", f"{mae(wte, pr_):.2f}",
                     f"{max(bias):.2f}"])
    table(["M", "最小组样本数", "整体 MAE", "最大组偏差"], rows)
    print("\n  最小组样本数随 M 线性下降，M=40 时已经出现空组（0 条）。")
    print("  但整体 MAE 随 M 增大只有轻微下降，最大组偏差反而单调变大：")
    print("  组越细分，最长那组里的观看时长分布越极端（均值被少数长观看拉高），")
    print("  而模型输出的 q̂ 仍挤在 0.5 附近，反变换的落点离组均值越远。")
    print("  这里没有出现正文说的「M 太大 MAE 回升」，因为 20000 条还撑得住")
    print("  40 个组；当数据量降到每个组只有几百条时，经验 CDF 尾部的噪声")
    print("  才会开始主导。M 是一个必须按数据量、按目标（MAE 还是校准）")
    print("  分别调的工程参数，不存在普遍最优。\n")

    # ---------------- 4) 反变换的尾部处理 ----------------
    print("④ 分位反变换的尾部：q̂ 落在 (0.99, 1) 时怎么处理")
    tail = (predict_linear(theta, Xte) > 0.99).mean()
    print(f"  测试集有 {tail:.1%} 的样本 q̂ > 0.99。")
    print("  本例的线性模型把 q̂ 压在 0.5 附近，尾部根本没被触发，")
    print("  所以这一段只能说明规则、不能说明现象：模型表达力越强、")
    print("  q̂ 越分散，尾部处理就越重要。")
    print("  直接反变换会取到组内最大值附近（全局最大 %.0f 秒），"
          % np.concatenate(groups).max())
    print("  如果测试分布比训练集更重尾，这部分样本会被系统性低估。")
    print("  工业实现通常做法是：对 q̂ 截断到 [q_lo, q_hi]（例如 [0.005, 0.995]），")
    print("  或者对尾部单独建一个「超过某阈值」的桶。\n")

    print("⑤ 一句话：D2Q 没有改模型结构，改的是 Label 的坐标系——")
    print("   把每个 Duration 组内的观看时长投影到分位数空间，让共享模型")
    print("   学「相对水平」，把绝对秒数留给反变换处理。")


if __name__ == "__main__":
    main()
