"""
阈值化时长 Label：EVTR / LVR / SVR / FPR 的构造与 Duration Bias（第23章 §23.1）

正文的核心链路：
    播放时长 → Duration-aware 阈值 → 二值化 Label → 分类 → P(W > T(D))

三个要点，本脚本逐一验证：
1) 固定绝对阈值 T 会让短视频天然占优（Duration Bias）：同样 T=10s，
   12 秒视频只要看 10 秒就为正，120 秒视频看 10 秒可能只是刚开始；
   这会让模型把「物理时长短」学成「用户兴趣强」。
2) Duration 分桶经验阈值（正文表 23.2）能显著拉平各 Duration 分桶的正样本率。
3) 用分位数阈值 T_k = Q_α(W | D ∈ B_k) 可以精确把每个桶的正样本率钉在 1−α，
   代价是阈值随数据漂移。

另外核对正文 §23.1 的四个工业 Label 定义（EVR / LVR / SVR / FPR）。

运行：python3 ch01_threshold_labels/threshold_labels.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import make_watch_data, table  # noqa: E402

SEED = 2026

# 正文表 23.2 的 Duration 分桶经验阈值（秒）
BUCKETS = [(0, 30, 5, 15), (30, 60, 10, 30), (60, 120, 20, 60),
           (120, 300, 30, 90), (300, 600, 40, 120)]


def bucket_index(d: np.ndarray) -> np.ndarray:
    """返回每条样本所属的分桶编号；超出最后一个桶的归入最后一桶。"""
    edges = np.array([b[1] for b in BUCKETS], dtype=np.float32)
    return np.searchsorted(edges, d, side="right").clip(0, len(BUCKETS) - 1)


def bucket_threshold(d: np.ndarray, which: int = 2) -> np.ndarray:
    """按分桶取阈值：which=2 为 EVTR，3 为 LVTR。"""
    k = bucket_index(d)
    thr = np.array([b[which] for b in BUCKETS], dtype=np.float32)
    return thr[k]


def rate_by_bucket(d: np.ndarray, y: np.ndarray, k: np.ndarray) -> list[list]:
    rows = []
    for i, (lo, hi, t_e, t_l) in enumerate(BUCKETS):
        m = k == i
        rows.append([f"[{lo},{hi})", int(m.sum()),
                     f"{y[m].mean():.3f}" if m.any() else "—",
                     f"{t_e}/{t_l}"])
    return rows


def main() -> None:
    print("阈值化时长 Label：EVTR / LVR / SVR / FPR 与 Duration Bias（§23.1）\n")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    d, w = data["d"], data["w"]
    k = bucket_index(d)

    # ---------------- 1) 固定阈值的 Duration Bias ----------------
    print("① 固定绝对阈值 T=10s：各 Duration 分桶的正样本率")
    print("   （正文命题：相同观看秒数对不同物理时长的内容意义不同）")
    y_fix = (w > 10).astype(np.float32)
    rows = rate_by_bucket(d, y_fix, k)
    table(["Duration 分桶", "样本数", "P(W>10s)", "本桶 EVTR/LVTR 阈值"], rows)
    short = rows[0][2]
    long_ = rows[-1][2]
    print(f"\n  最短桶正样本率 {short}，最长桶只有 {long_}——差 "
          f"{float(short) - float(long_):.2f}。")
    print("  这个差不是用户对短视频更感兴趣，而是短视频更容易达到 10 秒。")
    print("  模型若直接学这个 Label，会把 duration 当成强特征，形成 Duration Bias。\n")

    # ---------------- 2) 分桶经验阈值 ----------------
    print("② Duration 分桶经验阈值（正文表 23.2）：同样看各桶正样本率")
    y_evtr = (w > bucket_threshold(d, 2)).astype(np.float32)
    rows = rate_by_bucket(d, y_evtr, k)
    table(["Duration 分桶", "样本数", "P(W>T_EVTR(D))", "本桶 EVTR/LVTR 阈值"], rows)
    rates = np.array([float(r[2]) for r in rows])
    print(f"\n  分桶后正样本率落在 {rates.min():.3f}~{rates.max():.3f}，极差 "
          f"{rates.max() - rates.min():.3f}")
    print(f"  ——比固定阈值的 {float(short) - float(long_):.3f} 还大。这不奇怪：")
    print("  表 23.2 里最短桶用 5 秒阈值（比 10 秒更松），最长桶用 40 秒（更严），")
    print("  两端同时朝同一方向推，偏差自然被放大。也就是说，「分桶」本身不消除")
    print("  Duration Bias，只有「校准过的分桶阈值」才会。\n")

    print("  按数据校准：为每个桶搜索一个阈值，使该桶正样本率都等于固定阈值下的整体")
    target = float(y_fix.mean())
    calib = []
    for i in range(len(BUCKETS)):
        m = k == i
        if not m.any():
            continue
        cand = np.linspace(0.0, float(np.quantile(w[m], 0.99)), 400)
        err = [abs(float((w[m] > t).mean()) - target) for t in cand]
        t_best = float(cand[int(np.argmin(err))])
        calib.append((i, t_best, float((w[m] > t_best).mean())))
    rows = [[f"[{BUCKETS[i][0]},{BUCKETS[i][1]})", f"{BUCKETS[i][2]}",
             f"{t:.1f}", f"{r:.3f}"] for i, t, r in calib]
    table(["Duration 分桶", "表 23.2 的 EVTR 阈值", "校准后阈值", "校准后正样本率"], rows)
    gap = max(r for _, _, r in calib) - min(r for _, _, r in calib)
    print(f"\n  校准后各桶都落在 {target:.3f} 附近，极差 {gap:.3f}。")
    print("  这才是分桶阈值的正确用法：阈值不是拍脑袋定的经验值，而是对着业务目标")
    print("  （各桶「有效观看」的比例一致）反解出来的。表 23.2 给的是一组起点，")
    print("  真实系统每季度都要用最新数据重新解一次。\n")

    # ---------------- 3) 分位数阈值 ----------------
    print("③ 用分位数阈值 T_k = Q_α(W | D∈B_k) 精确控制各桶正样本率")
    rows = []
    for alpha in (0.5, 0.6, 0.7):
        ok = 0
        for i in range(len(BUCKETS)):
            m = k == i
            if not m.any():
                continue
            t = np.quantile(w[m], alpha)
            r = float((w[m] > t).mean())
            ok += int(abs(r - (1 - alpha)) < 0.02)
        rows.append([f"{alpha:.1f}", f"{1 - alpha:.1f}", f"{ok}/{len(BUCKETS)}",
                     "阈值由数据决定，需随分布漂移重估"])
    table(["α（分位）", "期望正样本率", "达标桶数", "备注"], rows)
    print("\n  分位阈值的优点是「每个桶的正样本率被钉死」，模型不会因为某桶")
    print("  天然正样本多而偏向它；代价是阈值完全跟随数据，一旦观看分布漂移，")
    print("  旧阈值立即失效。工业上通常按季度/半年重新拟合一次。\n")

    # ---------------- 4) 正文的四个工业 Label ----------------
    print("④ 正文 §23.4 的四个工业 Label 定义（EVR/LVR/SVR/FPR）在本数据上的比例")
    lvr_thr = np.where(d <= 3, 10.0,
                       np.where(d <= 40, 0.8 * d, 50.0))
    labels = {
        "EVR（playing_time ≥ 12s）": (w >= 12),
        "LVR（分段阈值）": (w > lvr_thr),
        "SVR（0 < playing_time ≤ 3s）": ((w > 0) & (w <= 3)),
        "FPR（duration>3s 且看完）": ((d > 3) & (w >= d)),
    }
    rows = []
    for name, y in labels.items():
        yf = y.astype(np.float32)
        rows.append([name, f"{yf.mean():.3f}",
                     f"{yf[d < 30].mean():.3f}", f"{yf[d >= 120].mean():.3f}"])
    table(["Label", "整体正样本率", "短视频(<30s)", "长视频(≥120s)"], rows)
    gap_evr = labels["EVR（playing_time ≥ 12s）"][d < 30].mean() \
        - labels["EVR（playing_time ≥ 12s）"][d >= 120].mean()
    gap_lvr = labels["LVR（分段阈值）"][d < 30].mean() \
        - labels["LVR（分段阈值）"][d >= 120].mean()
    print(f"\n  EVR 的短/长差距 {gap_evr:+.3f}，LVR 是 {gap_lvr:+.3f}——")
    print("  分段阈值没有把差距压小，反而略大。看阈值本身：短视频的中位时长约 21 秒，")
    print("  LVR 阈值 0.8d ≈ 17 秒，比 EVR 的 12 秒严 5 秒；长视频中位约 154 秒，")
    print("  但 LVR 在 d>40 后封顶在 50 秒，比 12 秒严 38 秒。")
    print("  两边都变严，但长桶严得多，于是长桶的正样本率掉得更多，差距被放大。")
    print("  这说明分段阈值好不好，取决于阈值函数和真实观看分布是否匹配，")
    print("  而不是「分了桶就一定更好」——这一段用的是正文 §23.1 的示意口径，")
    print("  真实系统里这组参数要按 ② 的方法对着业务目标校准。")
    print("  SVR（≤3 秒）在长视频上反而更高（0.415 vs 0.392）：本数据里短视频的")
    print("  观看时长受物理时长约束，不容易落进 3 秒以内，长视频则有一批「划走」。")
    print("  想用它当辅助信号，必须先按 Duration 校正，否则它衡量的其实是 Duration。")
    print("  FPR 是完播率口径，短桶确实更高，正文提醒的 Metric Hacking 正是指")
    print("  只优化这一条时，模型会系统性地把长视频排在后面。\n")

    # ---------------- 5) 多阈值联合建模 ----------------
    print("⑤ 多阈值联合建模：同一批样本上的 4 个嵌套 Label")
    thresholds = [3.0, 10.0, 30.0, 60.0]
    Y = np.stack([(w > t).astype(np.float32) for t in thresholds], axis=1)
    nested = np.all(np.diff(Y, axis=1) <= 0)            # 是否处处单调不增
    print(f"  阈值 {thresholds} → Label 矩阵形状 {Y.shape}")
    print(f"  正样本率：{[f'{Y[:, j].mean():.3f}' for j in range(Y.shape[1])]}")
    print(f"  单调（y_1 ≥ y_2 ≥ y_3 ≥ y_4）的样本占比：{nested.mean():.1%}")
    print("  这组 Label 合起来就是生存函数 S(t)=P(W>t) 在 4 个点的采样，")
    print("  正文说的「多阈值逐步逼近完整时长分布」在数据上就是逐点变稀疏的 0/1 矩阵。\n")

    print("⑥ 一句话：时长 Label 的起点不是模型，而是「什么算有效观看」。")
    print("   固定阈值简单但带 Duration Bias，分桶阈值校准了难度，")
    print("   分位阈值把正样本率钉死但依赖数据分布——三者是一条连续谱。")


if __name__ == "__main__":
    main()
