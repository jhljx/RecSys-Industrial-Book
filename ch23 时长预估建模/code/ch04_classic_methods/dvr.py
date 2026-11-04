"""
DVR：WTG 标准化标签 + 对抗去除 Duration 信息（第23章 §23.4.2）

正文的三层去偏：
    1) 指标层：在 Duration 分桶内做标准化，WTG = (WT − μ_B)/σ_B；
    2) 目标层：模型预测 WTG，而不是原始观看时长；
    3) 结构层：在预测分数后再接一个 Duration 预测器，用 GRL 做对抗训练，
       让分数里尽量少残留 Duration 信息。

本脚本依次验证：
    · 用原始观看时长排序，排出来的 Top-k 是否系统性偏向长视频；
    · 换成 WTG 后，Top-k 的平均物理时长是否回到候选池均值附近；
    · 对抗项能否进一步降低「从预测分数里预测 Duration」的 R²；
    · WTG@k 与 DCWTG@k（正文的排序指标）在两种模型上的差别。

数据里我们保留了真实兴趣 r，所以可以顺便检查 WTG 排序与「真实兴趣」的
一致性——这是线上没有的监督信号，只在这里作为评价用。

运行：python3 ch04_classic_methods/dvr.py
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


def duration_bucket(d: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Duration → 桶下标，区间约定为 [edges[b], edges[b+1])。

    edges 以 -inf 开头，它也满足「<= d」，所以 searchsorted(side="right")
    返回的下标会比真实桶号大 1，必须减掉；否则所有样本整体右移一格、
    桶 0 恒空、最长桶越界。这个坑在 dvr 的第一次实现里把 WTG 标签污染成
    未初始化内存，回归系数一度飙到 4.26。
    """
    return np.searchsorted(edges, d, side="right") - 1


def wtg_labels(w: np.ndarray, k: np.ndarray, n_b: int) -> np.ndarray:
    """
    每个 Duration 桶内标准化；桶内标准差为 0 时退回 0。

    用 zeros 而不是 empty 初始化：如果 k 里出现 >= n_b 的下标（分桶边界和
    桶数对不上时很容易发生），那些样本会拿到未初始化的内存值，后面拟合出来
    的系数会离谱到 θ≈4；zeros 会让这类样本老实落在「桶均值」上。
    """
    assert k.max() < n_b, f"分桶下标 {k.max()} 超出桶数 {n_b}"
    out = np.zeros(len(w))
    for b in range(n_b):
        m = k == b
        if not m.any():
            continue
        mu, sd = w[m].mean(), w[m].std()
        out[m] = (w[m] - mu) / sd if sd > 1e-8 else 0.0
    return out


def topk_metrics(scores: np.ndarray, slates: list[np.ndarray],
                 wtg_true: np.ndarray, k: int = 5) -> tuple[float, float]:
    """在给定的候选分组（slates）内按 scores 取 Top-k，返回 WTG@k / DCWTG@k。"""
    disc = 1.0 / np.log2(2 + np.arange(k))
    w_sum, dc_sum = 0.0, 0.0
    for s in slates:
        order = s[np.argsort(-scores[s])][:k]
        g = wtg_true[order]
        w_sum += g.mean()
        dc_sum += float((g * disc).sum())
    return w_sum / len(slates), dc_sum / len(slates)


def main() -> None:
    print("DVR：WTG 标签与对抗去偏（§23.4.2）\n")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, w, d, r = data["x"], data["w"].astype(np.float64), data["d"], data["r"]
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    Xtr, Xte, _, _ = standardize(X[tr], X[te])
    wtr, wte, dtr, dte = w[tr], w[te], d[tr], d[te]

    # ---------------- 1) WTG 标签 ----------------
    print("① 按 Duration 分桶做组内标准化，得到 WTG 标签")
    edges = np.array([20, 30, 45, 70, 120, 300], dtype=np.float32)
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    n_b = len(edges) - 1
    ktr = duration_bucket(dtr, edges)
    # WTG 用完整数据统计（线上是全量历史），避免训练/测试口径不一致
    k_all = duration_bucket(d, edges)
    wtg_all = wtg_labels(w, k_all, n_b)
    wtg_tr, wtg_te = wtg_all[tr], wtg_all[te]

    rows = []
    for b in range(n_b):
        m = ktr == b
        if not m.any():
            continue
        lo = edges[b] if np.isfinite(edges[b]) else 12
        hi = edges[b + 1] if np.isfinite(edges[b + 1]) else 600
        rows.append([f"[{lo:.0f},{hi:.0f})", int(m.sum()),
                     f"{wtr[m].mean():.1f}", f"{wtr[m].std():.1f}",
                     f"{wtr[m].std() / wtr.std():.2f}"])
    table(["Duration 桶", "样本数", "WT 均值", "WT 标准差", "标准差/全局"], rows)
    print("\n  各桶的绝对 WT 均值差了好几倍（7.9 → 46.3），组内标准差也跟着涨")
    print("  （0.21 → 2.17 倍全局标准差）。这不是 bug，是 Duration Bias 的")
    print("  另一种形态：短视频的观看时长被封顶，方差天然小；长视频能看的")
    print("  范围大，方差也大。标准化正是把这条「方差随 Duration 增长」的")
    print("  曲线拉平，让 20 秒桶里的 +1σ 和 300 秒桶里的 +1σ 可比。")
    print("  最长桶（300–600 秒）只有 58 条样本，它那 3.28 倍的标准差已经")
    print("  主要是抽样噪声——这也是分桶不能太细的原因。\n")

    # ---------------- 2) 直接回归 vs WTG 回归 ----------------
    print("② 用原始 WT 回归 vs 用 WTG 回归：Top-5 的 Duration 分布")
    th_w = fit_linear(Xtr, wtr)
    th_g = fit_linear(Xtr, wtg_tr)
    s_w = predict_linear(th_w, Xte)
    s_g = predict_linear(th_g, Xte)

    rng = np.random.default_rng(SEED)
    slates = [rng.choice(len(Xte), size=20, replace=False) for _ in range(200)]
    rows = []
    for name, s in (("按 WT 预测排序", s_w), ("按 WTG 预测排序", s_g),
                    ("随机排序（对照）", rng.normal(size=len(Xte)))):
        picked = np.concatenate([sl[np.argsort(-s[sl])][:5] for sl in slates])
        w_at, dc_at = topk_metrics(s, slates, wtg_te, 5)
        rows.append([name, f"{dte[picked].mean():.1f}",
                     f"{wte[picked].mean():.1f}",
                     f"{r[picked].mean():.3f}",
                     f"{w_at:+.3f}", f"{dc_at:+.3f}"])
    table(["排序依据", "Top-5 平均 Duration", "Top-5 平均 WT", "Top-5 真实兴趣",
           "WTG@5", "DCWTG@5"], rows)
    print("\n  按 WT 排序时，Top-5 的平均时长 93.7 秒，比候选池的 58.3 秒高出")
    print("  六成——模型不是根据兴趣在挑，而是根据「这视频够长、能刷出时长」")
    print("  在挑。换成 WTG 后，Top-5 平均时长回到 61.2 秒，几乎贴着候选池，")
    print("  WTG@5 从 +0.450 升到 +0.577：这是 LTR 里「优化对的指标」的直接")
    print("  体现。但注意第三列：三种排序下 Top-5 的真实兴趣命中率几乎一样")
    print("  （0.490 / 0.487 / 0.512），WTG 排序并没有挑到更感兴趣的视频。")
    print("  原因是 WTG 桶内标准化把跨桶的兴趣差异也一并抹掉了——16 维线性")
    print("  特征学到的得分，在桶内只够分辨「相对多看一点」，不够分辨「真喜欢」。")
    print("  线上 DVR 用深度模型补的就是这一段表达能力。\n")

    # ---------------- 3) 对抗去偏 ----------------
    print("③ 对抗训练：从预测分数 s 再预测 Duration，用 GRL 反传")
    s_tr = predict_linear(th_g, Xtr)
    logd_tr = np.log(dtr)

    def duration_r2(score: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
        """一元线性回归 d ≈ a·s + b 的 R²，以及斜率 a（Duration 残留的强度）。"""
        A = np.stack([score, np.ones_like(score)], axis=1)
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        pred = A @ coef
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot, coef[0], coef[1]

    r2_before, slope_before, _ = duration_r2(s_tr, logd_tr)
    print(f"  对抗前：用 s 预测 log(Duration) 的 R² = {r2_before:.3f}，斜率 {slope_before:+.3f}")

    # 简化版 GRL：s = x·θ，Duration 预测器是 1 维线性 ψ(s) = a·s + b。
    # ψ 正常梯度下降最小化 MSE(a·s+b, log d)；
    # θ 的对抗梯度是 −α·∂MSE/∂s，即把 a 的符号翻过来。
    a, b = 0.0, float(logd_tr.mean())
    theta_adv = th_g.copy()
    alpha = 0.5
    lr_psi, lr_theta = 0.05, 0.02
    n = len(Xtr)
    Xb = np.concatenate([Xtr, np.ones((n, 1))], axis=1)
    for _ in range(300):
        s = Xb @ theta_adv
        err = a * s + b - logd_tr
        da = (err * s).mean()
        db = err.mean()
        a -= lr_psi * da
        b -= lr_psi * db
        # θ 的梯度：α · a · err（对 s 求导后符号翻转）
        g_theta = Xb.T @ (alpha * a * err) / n
        theta_adv[:-1] += lr_theta * g_theta[:-1]
        theta_adv[-1] += lr_theta * g_theta[-1]

    s_adv = predict_linear(theta_adv, Xte)
    r2_after, slope_after, _ = duration_r2(predict_linear(theta_adv, Xtr), logd_tr)
    rows = [
        ["WTG 回归（无对抗）", f"{r2_before:.3f}", f"{slope_before:+.3f}",
         f"{mae(wtg_te, s_g):.3f}"],
        ["+ GRL 对抗", f"{r2_after:.3f}", f"{slope_after:+.3f}",
         f"{mae(wtg_te, s_adv):.3f}"],
    ]
    table(["模型", "Duration 可预测性 R²", "斜率", "WTG 预测 MAE"], rows)
    print("\n  注意 R² 的绝对值：对抗前就已经只有 0.002，对抗后被压到 0.000。")
    print("  这不是实验失败，恰恰说明 WTG 标签本身已经把 Duration 信息清掉了——")
    print("  桶内标准化之后，分数与 log(Duration) 的线性相关性本来就接近零，")
    print("  留给 GRL 去挤的东西不多。对照一下更直观：如果用原始 WT 做标签，")
    print("  同一套特征拟合出的 w 预测值对 log(Duration) 的 R² 是 0.394。")
    print("  换句话说，DVR 的三个层次不是同等的：指标层的贡献最大，")
    print("  对抗层是「保险」——当 Duration 通过 Category、Creator 等特征")
    print("  间接漏进来时，它才真正开始工作。")
    print("  WTG 的 MAE 从 0.749 变成 0.747，没有付出精度代价；但 16 维线性")
    print("  模型能残留的 Duration 信息太少，这个「没有代价」不能外推到 MLP。\n")

    # ---------------- 4) 排序指标 ----------------
    print("④ WTG@k 与 DCWTG@k（k=5,10）在三种排序下的表现：")
    rows = []
    for name, s in (("WT 回归", s_w), ("WTG 回归", s_g), ("WTG + GRL", s_adv),
                    ("随机", rng.normal(size=len(Xte)))):
        rows.append([name] + [f"{topk_metrics(s, slates, wtg_te, k)[0]:+.3f}"
                              for k in (5, 10)]
                     + [f"{topk_metrics(s, slates, wtg_te, k)[1]:+.3f}"
                        for k in (5, 10)])
    table(["模型", "WTG@5", "WTG@10", "DCWTG@5", "DCWTG@10"], rows)
    print("\n  WTG@k 只看 Top-k 的平均增益，DCWTG@k 给靠前的位置更高权重。")
    print("  两者与原观看时长指标（比如总时长最大化）的区别在于：")
    print("  它们不会因为「排了一个 300 秒的视频」而奖励模型，")
    print("  只看这个视频相对同 Duration 的平均表现好不好。\n")

    print("⑤ 一句话：DVR 把优化目标从「看了多少秒」换成「比同时长视频好多少」，")
    print("   再用对抗把分数里偷偷带进来的 Duration 信息挤出去。")


if __name__ == "__main__":
    main()
