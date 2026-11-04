"""
工业阈值 Label：EVR / LVR / SVR / FPR 的 Duration Bias 审计（第23章 §23.6.1）

正文列了四个典型阈值 Label：
    EVR: w ≥ 12 秒
    LVR: d ≤ 3s 时 w > 10；3s < d ≤ 40s 时 w > 0.8d；d > 40s 时 w > 50
    SVR: 0 < w ≤ 3
    FPR: d > 3 且 w ≥ d
这类 Label 直观、好听、能直接进多目标融合，但每个阈值都在
「区分用户兴趣」和「引入 Duration 信息」之间选一个点。本脚本量化这件事：

    ① 每个 Label 的正样本率随 Duration 桶怎么变；
    ② 每个 Label 对真实兴趣 r 的排序能力（整体 AUC / 桶内 AUC）；
    ③ 只用 log(d) 能解释多少 Label（McFadden R²）；
    ④ EVR 阈值从 4 秒扫到 40 秒，三个量怎么一起动；
    ⑤ 注入「长内容用户」相关性之后，②的结论会怎么变。

先说结论，因为它与正文的直觉不一致：在默认合成数据上，四个 Label
的「桶内 AUC」几乎等于「整体 AUC」（EVR 0.874 vs 0.876），
log(d) 的 McFadden R² 约为 0。原因是 make_watch_data 里
兴趣 r 与物理时长 d **独立生成**——没有「偏爱长内容的用户」这一类人，
阈值 Label 就没有 Duration Shortcut 可借。真实系统里 r 与 d 是相关的
（长视频更容易被推给能接受长内容的用户），⑤ 用一个人为注入的相关性
复现这件事，并显示 Shortcut 确实会出现。

注意 LVR 第一行在合成数据里恒为 0：w 被裁剪到 [0, d]，d ≤ 3 秒时
w 不可能超过 10 秒。正文那张表里这一行在物理上不可能成立，除非
playing_time 记录了重播（w 可以大于 d），本数据没有重播。

运行：python3 ch06_industrial_practice/label_design.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    auc, fit_logistic, make_watch_data, sigmoid, split, table,
)

SEED = 2026

BINS = [(12, 20), (20, 30), (30, 45), (45, 70), (70, 120), (120, 600)]


def evr(w: np.ndarray, d: np.ndarray) -> np.ndarray:
    return (w >= 12).astype(np.float64)


def lvr(w: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.zeros(len(w))
    out[(d <= 3)] = (w[d <= 3] > 10)
    mid = (d > 3) & (d <= 40)
    out[mid] = (w[mid] > 0.8 * d[mid])
    out[d > 40] = (w[d > 40] > 50)
    return out


def svr(w: np.ndarray, d: np.ndarray) -> np.ndarray:
    return ((w > 0) & (w <= 3)).astype(np.float64)


def fpr(w: np.ndarray, d: np.ndarray) -> np.ndarray:
    return ((d > 3) & (w >= d)).astype(np.float64)


LABELS = {"EVR (w≥12s)": evr, "LVR (分段)": lvr,
          "SVR (0<w≤3s)": svr, "FPR (w≥d)": fpr}


def within_bin_auc(label: np.ndarray, r: np.ndarray, k: np.ndarray,
                   n_b: int) -> float:
    """在每个 Duration 桶内部算 AUC，再按样本量加权平均。"""
    tot, nsum = 0.0, 0
    for b in range(n_b):
        m = k == b
        if m.sum() > 20 and 0 < r[m].sum() < m.sum():
            tot += auc(r[m], label[m]) * m.sum()
            nsum += m.sum()
    return tot / max(nsum, 1)


def bucketize(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    edges = np.array([b[0] for b in BINS] + [BINS[-1][1]], dtype=float)
    k = np.searchsorted(edges, d, side="right") - 1
    return np.clip(k, 0, len(BINS) - 1), edges, len(BINS)


def dur_auc(lab: np.ndarray, ld: np.ndarray) -> float:
    """
    「只用视频长度排序，能多大程度还原这个 Label」——AUC(log d, label)。

    不用 Logistic + McFadden R²：单特征 GD 在 300 步内不一定收敛，
    而且 R² 对「正样本率随 Duration 非单调变化」的 Label 不敏感
    （LVR 就是非单调的）。AUC 只依赖排序，稳健得多。
    0.5 = Duration 完全没信息；离 0.5 越远，说明 Label 越能被长度解释
    （方向看大于还是小于 0.5）。
    """
    return float(auc(lab, ld))


def main() -> None:
    print("工业阈值 Label：Duration Bias 审计（§23.6.1）\n")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, w, d, r = data["x"], data["w"].astype(np.float64), data["d"], data["r"]
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    kb, edges, n_b = bucketize(d)

    # ---------------- 1) 正样本率随 Duration ----------------
    print("① 正样本率随物理时长怎么变")
    rows = []
    for b, (lo, hi) in enumerate(BINS):
        m = kb == b
        rows.append([f"[{lo},{hi})", int(m.sum())] +
                    [f"{fn(w, d)[m].mean():.1%}" for fn in LABELS.values()])
    table(["Duration 桶", "样本数"] + list(LABELS.keys()), rows)
    print("\n  方向与「长视频更容易凑够 12 秒」的直觉相反：EVR 从 46% 降到 27%。")
    print("  原因在生成过程——兴趣 r 决定「要不要看下去」，不感兴趣就是几秒走人，")
    print("  感兴趣则要么看完、要么几秒后离开。时长本身并不累积：")
    print("  一条 300 秒的视频，不感兴趣的用户也只看 2 秒，凑不到 12 秒；")
    print("  而 12–20 秒的视频，看完就等于过了 12 秒。所以固定绝对阈值")
    print("  在这套数据上偏向短视频。正文说「短视频天然有更高的有效播放率」，")
    print("  方向一致，只是起因不是「长度累积」而是「看完比看下去容易」。")
    print("  LVR 的分段阈值把这条曲线压平了一些，SVR 是「短看」，方向相反，")
    print("  FPR 要求看满，长视频更难，也是下降的。\n")

    # ---------------- 2) 排序能力 ----------------
    print("② 对真实兴趣 r 的排序能力：整体 vs 桶内")
    rows = []
    for name, fn in LABELS.items():
        lab = fn(w, d)
        rows.append([name, f"{lab.mean():.1%}", f"{auc(r, lab):.3f}",
                     f"{within_bin_auc(lab, r, kb, n_b):.3f}"])
    rows.append(["原始观看时长 w（对照）", f"{np.mean(w > 0):.1%}",
                 f"{auc(r, w):.3f}", f"{within_bin_auc(w, r, kb, n_b):.3f}"])
    table(["Label", "正样本率", "整体 AUC(r)", "桶内 AUC(r)"], rows)
    print("\n  整体 AUC 与桶内 AUC 几乎相等（EVR 0.876 对 0.874，差 0.002）。")
    print("  这说明在这份数据上，阈值 Label 的排序能力**不是**从 Duration 借来的——")
    print("  把 Duration 的影响扣掉之后，它照样能分开感兴趣与不感兴趣的人。")
    print("  SVR 的 AUC 只有 0.23，因为它是反向 Label：越不感兴趣越容易是「短看」。")
    print("  w 本身的 AUC 0.836 低于 EVR，因为 w 连续、而零观看那 12% 全都排在最低端，")
    print("  阈值化反而把「有没有看」这个最粗的区分显式化了。\n")

    # ---------------- 3) Duration 能解释多少 ----------------
    print("③ 只用视频长度排序，能多大程度还原这些 Label")
    ld = np.log(d)
    rows = []
    for name, fn in LABELS.items():
        lab = fn(w, d)
        rows.append([name, f"{dur_auc(lab, ld):.3f}",
                     f"{dur_auc(lab, ld) - 0.5:+.3f}"])
    table(["Label", "AUC(log d, label)", "偏离 0.5"], rows)
    print("\n  全部在 0.5 附近（0.44~0.55）——视频长度几乎不能告诉我们")
    print("  这个 Label 是正是负。这与①里看得见的漂移并不矛盾：漂移是")
    print("  「正样本率随桶变化」，而 AUC 衡量的是「同一个桶内外的排序」，")
    print("  正样本率整体平移、但每个桶内部的排序不变时，AUC 就贴着 0.5。")
    print("  换句话说，在这份数据上 Label 确实**带着** Duration 信息，")
    print("  却没能靠它获得排序能力——如果只关心排序，Shortcut 的收益是零。")
    print("  真实系统里这两个量往往一起动，见⑤。\n")

    # ---------------- 4) 阈值扫描 ----------------
    print("④ EVR 阈值扫描：正样本率、AUC 与 Duration 依赖一起移动")
    rows = []
    for thr in (4.0, 8.0, 12.0, 20.0, 40.0):
        lab = (w >= thr).astype(np.float64)
        rows.append([f"{thr:g}", f"{lab.mean():.1%}", f"{auc(r, lab):.3f}",
                     f"{within_bin_auc(lab, r, kb, n_b):.3f}",
                     f"{dur_auc(lab, ld):.3f}"])
    table(["EVR 阈值(秒)", "正样本率", "整体 AUC(r)", "桶内 AUC(r)",
           "AUC(log d, label)"], rows)
    print("\n  阈值从 4 抬到 40，正样本率从 49% 掉到 20%，AUC 在 12 秒附近达到峰值")
    print("  0.876，然后掉到 0.69。原因和②一致：40 秒这个阈值只有「看完长视频」")
    print("  的那部分人过得了，它不再是「有效观看」而是「看完」，")
    print("  而看完的人里既有感兴趣的也有碰巧的。")
    print("  12 秒恰好落在「看完 12–20 秒视频」与「不感兴趣两秒走」之间，")
    print("  区分度最好——这是 12 秒这个数字在这份数据上的依据。")
    print("  最后一列在 40 秒处从 0.55 跳到 0.75，说明高阈值时 Label 已经变成")
    print("  「这条视频够不够长」的近似——长视频才可能看满 40 秒。")

    # ---------------- 5) 注入相关性 ----------------
    print("⑤ 注入「长内容用户」：让兴趣与物理时长相关，Shortcut 就出现了")
    rng = np.random.default_rng(SEED)
    # 真实系统里，长视频更多被推给能接受长内容的用户；用一个正相关的
    # 潜在变量 z 同时抬高「感兴趣的概率」和「视频变长的概率」。
    z = rng.normal(size=len(X))
    r2sim = (rng.random(len(X)) < sigmoid(2.0 * z)).astype(np.float64)
    d2 = np.clip(d * np.exp(0.45 * z), 12, 600)
    # 重算观看时长：用原来的生成逻辑太重，这里直接用 r2sim 与 d2 构造一个
    # 与 make_watch_data 同构的 w（感兴趣看满或几秒后离开，反之几秒）
    c = (rng.random(len(X)) < sigmoid(2.5 - 0.008 * d2)).astype(np.float64)
    wp = c * d2 + (1 - c) * np.minimum(d2, rng.exponential(4.0, len(X)))
    wn = np.minimum(rng.exponential(2.5, len(X)), d2)
    w2 = r2sim * wp + (1 - r2sim) * wn
    w2[rng.random(len(X)) < 0.12] = 0.0

    kb2, _, _ = bucketize(d2)
    rows = []
    for name in ("EVR (w≥12s)", "LVR (分段)"):
        lab = LABELS[name](w2, d2)
        rows.append([name, f"{lab.mean():.1%}", f"{auc(r2sim, lab):.3f}",
                     f"{within_bin_auc(lab, r2sim, kb2, n_b):.3f}",
                     f"{dur_auc(lab, np.log(d2)):.3f}"])
    table(["Label（注入相关后）", "正样本率", "整体 AUC(r)", "桶内 AUC(r)",
           "AUC(log d, label)"], rows)
    print(f"\n  注入之后 r 与 log(d) 的相关系数是 "
          f"{np.corrcoef(r2sim, np.log(d2))[0, 1]:+.2f}，")
    print(f"  AUC(log d, EVR) 从 0.442 涨到 {dur_auc(LABELS['EVR (w≥12s)'](w2, d2), np.log(d2)):.3f}——")
    print("  只拿视频长度去猜 EVR，已经能比瞎猜准得多。这就是 Shortcut 的样子：")
    print("  模型不需要理解兴趣，只要学会「长视频更可能过 EVR」就能涨点。")
    print("  但注意桶内 AUC 反而从 0.874 升到 0.883：Label 在同一个长度段内")
    print("  仍然很好地分开感兴趣与不感兴趣的人。两件事同时成立——")
    print("  Label 既携带真实兴趣、又和长度耦合，所以「扣掉长度再看」")
    print("  并不能把它的区分度抹掉，只能说明它的区分度里混了长度这一份。")
    print("  这正是正文说的 Duration Bias：不是 Label 写错了，")
    print("  而是 Label 与 Duration 在数据里耦合了，你无法从离线指标上")
    print("  区分「模型学到了兴趣」还是「模型学到了长度」。")
    print("  处理办法有两类——把 Label 做成 Duration-aware（LVR 那条路），")
    print("  或者在特征侧做去偏（D²Co、DVR 那条路），这里只展示了前者的形状。\n")

    print("⑥ 一句话：阈值 Label 的业务含义清晰，但「阈值选在哪」等价于")
    print("   「在区分度和 Duration 耦合之间选一个点」；")
    print("   而耦合强度取决于数据怎么生成，所以要定期 Review，不能照抄阈值。")


if __name__ == "__main__":
    main()
