"""
分位数 Label 与分布漂移（第23章 §23.6.2–§23.6.4）

正文的演进链条：固定阈值 → Duration 相关分段阈值 → 分桶 → 分位数。
本脚本把后两段落到可执行的对照上：

    ① 正样本率是否被 Duration 拉平：固定阈值 vs 分位数 Label；
    ② 不同 Duration 下的播放时长分位曲线（P50/P60/P70/P80）长什么样，
       为什么「固定 12 秒」在这里会被 Duration 扭曲；
    ③ 两种 Label 对真实兴趣 r 的排序能力；
    ④ 分布漂移：视频整体变长之后，固定阈值和分位数阈值各会怎样失效
       ——重点区分「阈值没跟上」「数据换了」两种失效；
    ⑤ 长尾处理：截断 / log 变换 / 不处理，对点回归 MAE、RMSE 的影响。

分位数 Label 的做法：对每个 Duration 桶，在训练集上取播放时长的
P60/P70/P80 作为该桶阈值，Label = 1{w > Q_p(d)}。这样每个桶的正样本率
在训练分布上恒为 1−p，天然与 Duration 无关；代价是新数据分布一变，
阈值就得重估——④ 量化这个代价。

运行：python3 ch06_industrial_practice/quantile_labels.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    auc, fit_linear, make_watch_data, mae, predict_linear, rmse, split,
    standardize, table,
)

SEED = 2026
BINS = [(12, 20), (20, 30), (30, 45), (45, 70), (70, 120), (120, 600)]
PS = (0.50, 0.60, 0.70, 0.80)


def bucketize(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.array([b[0] for b in BINS] + [BINS[-1][1]], dtype=float)
    k = np.searchsorted(edges, d, side="right") - 1
    return np.clip(k, 0, len(BINS) - 1), edges


def quantile_curves(w: np.ndarray, k: np.ndarray, n_b: int) -> np.ndarray:
    """每个 Duration 桶的 P50/P60/P70/P80，形状 [n_b, len(PS)]。"""
    out = np.zeros((n_b, len(PS)))
    for b in range(n_b):
        m = k == b
        out[b] = np.quantile(w[m], PS) if m.any() else 0.0
    return out


def label_by_quantile(w: np.ndarray, k: np.ndarray, q: np.ndarray,
                      p_idx: int) -> np.ndarray:
    """1{w > Q_p(d)}，阈值按桶查表。"""
    return (w > q[k, p_idx]).astype(np.float64)


def within_bin_auc(label: np.ndarray, r: np.ndarray, k: np.ndarray,
                   n_b: int) -> float:
    tot, nsum = 0.0, 0
    for b in range(n_b):
        m = k == b
        if m.sum() > 20 and 0 < r[m].sum() < m.sum():
            tot += auc(r[m], label[m]) * m.sum()
            nsum += m.sum()
    return tot / max(nsum, 1)


def main() -> None:
    print("分位数 Label 与分布漂移（§23.6.2–§23.6.4）\n")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, w, d, r = data["x"], data["w"].astype(np.float64), data["d"], data["r"]
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    ktr, _ = bucketize(d[tr])
    kte, _ = bucketize(d[te])
    n_b = len(BINS)

    q_tr = quantile_curves(w[tr], ktr, n_b)

    # ---------------- 1) 正样本率 ----------------
    print("① 正样本率是否被 Duration 拉平（训练集）")
    rows = []
    for b, (lo, hi) in enumerate(BINS):
        m = ktr == b
        rows.append([f"[{lo},{hi})", int(m.sum()),
                     f"{np.mean(w[tr][m] >= 12):.1%}",
                     f"{np.mean(w[tr][m] > q_tr[b, 2]):.1%}"])
    table(["Duration 桶", "样本数", "固定 EVR(w≥12s)", "分位数 P70"], rows)
    print("\n  分位数 Label 的每一行都是 30%——这不是巧合，是构造出来的：")
    print("  每个桶的阈值就是这个桶自己的 P70。它与 Duration 无关这一点")
    print("  是「训练分布上」的结论，④会说明新数据上它照样会漂移。")
    print("  固定阈值的曲线则随桶下降（46% → 27%），方向见 label_design.py。\n")

    # ---------------- 2) 分位曲线 ----------------
    print("② 不同 Duration 下的播放时长分位曲线（训练集，秒）")
    rows = []
    for b, (lo, hi) in enumerate(BINS):
        rows.append([f"[{lo},{hi})"] + [f"{q_tr[b, j]:.1f}" for j in range(len(PS))])
    table(["Duration 桶", "P50", "P60", "P70", "P80"], rows)
    print("\n  读取这张表的正确方式是**横向比同一分位、纵向看 12 秒落在哪**：")
    print("    · P50 在各个桶都只有两三秒——大量样本是「不感兴趣就滑走」，")
    print("      它把中位数压在了地板上，这是零观看 12% 之外的第二层堆积；")
    print("    · P60 到 P80 随桶抬升，但长桶里是**跳变**：")
    print("      [120,600) 的 P70 只有 8.3 秒、P80 却是 132.6 秒。")
    print("      这不是估计噪声，而是分布本身有两个团——")
    print("      「几秒就滑走」占七成，剩下三成「看完了」直接贴在 d 上，")
    print("      两者中间几乎是空的，所以分位点会从 8 秒一步跨到 130 秒。")
    print("    · 同一个 12 秒，在 [12,20) 桶里落在 P60（12.0）与 P70（13.5）之间；")
    print("      在 [120,600) 桶里高于 P70（8.3）、远低于 P80（132.6），")
    print("      对应的是完全不同的观看程度。")
    print("  所以固定阈值不是「处处等价」，它在不同桶里对应完全不同的")
    print("  观看程度：短桶里接近「看完」，长桶里只是「刚看了一点」。")
    print("  这才是 Duration Bias 的本体——不是正样本率的差异，")
    print("  而是「同一个 Label 在不同桶里意味着不同的事」。\n")

    # ---------------- 3) 排序能力 ----------------
    print("③ 对真实兴趣 r 的排序能力（测试集）")
    rows = []
    lab_fix = (w[te] >= 12).astype(np.float64)
    rows.append(["固定 EVR", f"{lab_fix.mean():.1%}", f"{auc(r[te], lab_fix):.3f}",
                 f"{within_bin_auc(lab_fix, r[te], kte, n_b):.3f}"])
    for j, p in enumerate((0.6, 0.7, 0.8)):
        lab = label_by_quantile(w[te], kte, q_tr, j + 1)
        rows.append([f"分位数 P{int(p * 100)}", f"{lab.mean():.1%}",
                     f"{auc(r[te], lab):.3f}",
                     f"{within_bin_auc(lab, r[te], kte, n_b):.3f}"])
    table(["Label", "正样本率", "整体 AUC(r)", "桶内 AUC(r)"], rows)
    print("\n  这里的排序能力并不是「接近」，而是**随 p 单调下降**：")
    print("  固定 EVR 0.872、P60 0.847、P70 0.792、P80 0.695。")
    print("  原因不是 Label 丢了信息，而是越往高端走，正样本越少、")
    print("  且「看得很久」这件事本身被两类人共享——真感兴趣的，")
    print("  以及滑动中途被别的内容吸引后回来的。P80 只保留最上面 20%，")
    print("  它衡量的是「谁看得最久」，不是「谁最感兴趣」。")
    print("  这也解释了为什么正文的演进不是「分位数越细越好」：")
    print("  P50/P60 适合描述观看深度，P80/P90 更适合做「爆款判定」这类")
    print("  完全不同的问题。换 Label 换的是问题本身，不是同一个问题的更好答案。\n")

    # ---------------- 4) 分布漂移 ----------------
    print("④ 分布漂移：视频整体变长 50% 之后，两种阈值谁先失效")
    rng = np.random.default_rng(SEED)
    d_new = np.clip(d[te] * 1.5, 12, 900)
    k_new, _ = bucketize(d_new)
    # 观看行为按原生成规则重算：同一个用户对同类内容，观看时长随
    # 「是否看完」变化，因此新数据不是简单缩放 w，而是重采样。
    c = (rng.random(len(d_new)) < 1 / (1 + np.exp(-(2.5 - 0.008 * d_new))))
    wp = c * d_new + (1 - c) * np.minimum(d_new, rng.exponential(4.0, len(d_new)))
    wn = np.minimum(rng.exponential(2.5, len(d_new)), d_new)
    w_new = r[te] * wp + (1 - r[te]) * wn
    w_new[rng.random(len(w_new)) < 0.12] = 0.0

    rows = []
    rows.append(["固定 EVR", f"{np.mean(w_new >= 12):.1%}",
                 f"{np.mean(w_new >= 12):.1%}"])
    # 不重估：沿用旧分位数阈值
    lab_old = (w_new > q_tr[k_new, 2]).astype(np.float64)
    rows.append(["分位数 P70（沿用旧阈值）", f"{np.mean(w_new > q_tr[k_new, 2]):.1%}",
                 "—"])
    # 重估：在新数据上重新算分位数阈值
    q_new = quantile_curves(w_new, k_new, n_b)
    lab_new = (w_new > q_new[k_new, 2]).astype(np.float64)
    rows.append(["分位数 P70（重估阈值）", f"{lab_new.mean():.1%}", "—"])
    table(["策略", "新数据上的正样本率", "备注"], rows)
    print("\n  结果和「固定阈值一定先失效」的直觉相反：")
    print("  固定 EVR 只从 38.6% 走到 39.5%，几乎没动；")
    print("  分位数 P70 从 30.1% 漂到 34.4%，反而漂得更多；")
    print("  在新数据上重估分位点，才回到 29.3%。")
    print("  原因在这套生成过程——观看时长不随 d 累积长视频，")
    print("  感兴趣的人要么看完（此时 d 变长，w 也变长）、要么几秒就走，")
    print("  而「看完概率」随 d 下降恰好抵消了一部分。真实系统里")
    print("  长视频有更强的累积效应，固定阈值会明显上漂，")
    print("  这时分位数 Label 的优势才会显出来。")
    print("  但两类 Label 有一个共同的结论：**漂移必须被监控**。")
    print("  固定阈值靠 Review 发现失效，分位数阈值靠定期重估兜底，")
    print("  谁也不是一劳永逸。正文强调 Label 要定期 Review，根因就在这里。\n")

    # ---------------- 5) 长尾处理 ----------------
    print("⑤ 长尾处理对点回归的影响（测试集）")
    Xtr, Xte, _, _ = standardize(X[tr], X[te])
    rows = []
    # 不处理
    th = fit_linear(Xtr, w[tr])
    pred = np.clip(predict_linear(th, Xte), 0, None)
    rows.append(["原始 w", f"{mae(w[te], pred):.2f}", f"{rmse(w[te], pred):.2f}"])
    # 截断到 P99
    cap = float(np.quantile(w[tr], 0.99))
    th = fit_linear(Xtr, np.minimum(w[tr], cap))
    pred = np.clip(predict_linear(th, Xte), 0, None)
    rows.append([f"截断到 P99={cap:.0f}s", f"{mae(w[te], pred):.2f}",
                 f"{rmse(w[te], pred):.2f}"])
    # log1p 目标
    th = fit_linear(Xtr, np.log1p(w[tr]))
    pred = np.clip(np.expm1(predict_linear(th, Xte)), 0, None)
    rows.append(["log1p(w) 目标", f"{mae(w[te], pred):.2f}",
                 f"{rmse(w[te], pred):.2f}"])
    # 截断 + log
    th = fit_linear(Xtr, np.log1p(np.minimum(w[tr], cap)))
    pred = np.clip(np.expm1(predict_linear(th, Xte)), 0, None)
    rows.append(["截断 + log1p", f"{mae(w[te], pred):.2f}",
                 f"{rmse(w[te], pred):.2f}"])
    table(["目标变换", "MAE", "RMSE"], rows)
    print(f"\n  P99 是 {cap:.0f} 秒，长尾很长（训练集均值 {w[tr].mean():.1f} 秒）。")
    print("  四行的取舍各不相同：")
    print("    · 截断到 P99：MAE 略降（19.80 → 19.63）、RMSE 基本不变，")
    print("      因为它只动了 1% 的样本，对线性模型影响很小；")
    print("    · log1p：MAE 明显变好（19.80 → 17.88），RMSE 却从 28.45")
    print("      升到 32.92——log 空间的最优解回变换后偏向「相对误差小」，")
    print("      长尾的平方误差被放大。这和 EGMN 那节的 MAE/RMSE 反向")
    print("      是同一件事；")
    print("    · 截断 + log1p 与单独 log1p 几乎一样，说明在 log 空间里")
    print("      长尾已经被压平，再截断没有额外收益。")
    print("  所以「用哪种变换」取决于线上用 MAE 还是 RMSE 排序：")
    print("  看重中位体验用 log，看重极端样本用原始尺度 + 截断。")
    print("  正文把「截断 / 归一化 / 分桶」并列，正是因为它们服务于不同目标。\n")

    print("⑥ 一句话：分位数 Label 用「按 Duration 分桶估分位点」把")
    print("   正样本率拉平，代价是阈值必须跟着数据分布重估；")
    print("   而长尾处理是另一组取舍，和 Label 无关，与 Loss、排序目标有关。")


if __name__ == "__main__":
    main()
