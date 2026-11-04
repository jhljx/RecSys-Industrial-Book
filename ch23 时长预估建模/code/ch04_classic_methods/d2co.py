"""
D²Co：两成分 GMM 估 w⁺/w⁻，再用校正函数还原兴趣（第23章 §23.4.3）

正文的分解：
    w = p_x^r · w_d⁺ + (1 − p_x^r) · w_d⁻
其中 w_d⁺ 是「感兴趣时的平均观看时长」（含 Duration Bias），
w_d⁻ 是「不感兴趣时的噪声观看」（几秒的误触/滑走，Noisy Watching）。

流程：
    1) 按 Duration 分组，在每组内用两成分 GMM 拟合观看时长分布，
       得到 ŵ_d⁺ 与 ŵ_d⁻；
    2) 用频次加权的双向移动平均抹平相邻 Duration 上的估计噪声；
    3) 线性校正 D²Co(A)：r = (w − w_d⁻) / (w_d⁺ − w_d⁻)；
       指数校正 D²Co(S)：r = (e^{αw} − e^{αw_d⁻}) / (e^{αw_d⁺} − e^{αw_d⁻})。

本例的特殊之处：合成数据里保留了真实兴趣 r，所以可以拿 GMM 估出来的
w⁺/w⁻ 去和「按 r 分组的真实均值」对比——这是线上做不到的，只在这里
用来判断估计误差有多大、又会怎样传到最终 Label 上。

工程取舍：GMM 拟合在 log(1+W) 上做，而不是原始秒数。原始 W 右偏且被人为
封顶在 [0, d]，直接套高斯混合会让两个成分都去追那条长尾。log 空间里
「长视频看满」与「划走」才是两个可分的团。回变换用 E[W]=exp(μ+σ²/2)−1，
带方差修正，否则会系统性低估均值。

运行：python3 ch04_classic_methods/d2co.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    auc, mae, make_watch_data, split, table,
)

SEED = 2026


# --------------------------------------------------------------------------
# 一维两成分 GMM（EM，多次随机重启，取对数似然最大的解）
# --------------------------------------------------------------------------
def _log_norm_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((x - mu) / sd) ** 2


def fit_gmm1d(x: np.ndarray, k: int = 2, iters: int = 300,
              restarts: int = 4, seed: int = SEED):
    """
    返回按均值升序排列的 (mu, sd, pi)。

    第 0 次用分位点初始化（稳定），其余用随机样本初始化（避免所有重启
    都落在同一个局部最优）。EM 的对数似然单调不减，所以「多重启取最大」
    是安全的。
    """
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        if r == 0:
            mu = np.sort(np.quantile(x, np.linspace(0.15, 0.85, k)))
        else:
            mu = np.sort(rng.choice(x, size=k, replace=False))
            # 随机重启可能抽到几乎重合的两点，加一点抖动保证可分
            mu = mu + rng.normal(0, 1e-3, size=k)
        sd = np.full(k, max(x.std(), 1e-3))
        pi = np.full(k, 1.0 / k)
        ll_old = -np.inf
        for _ in range(iters):
            logp = np.stack([_log_norm_pdf(x, mu[j], sd[j]) for j in range(k)], axis=1)
            logp = logp + np.log(pi + 1e-300)
            m = logp.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
            resp = np.exp(logp - lse[:, None])
            nk = resp.sum(axis=0) + 1e-300
            mu = (resp * x[:, None]).sum(axis=0) / nk
            sd = np.sqrt((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-6
            pi = nk / len(x)
            ll = float(lse.sum())
            if ll - ll_old < 1e-6:
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, mu.copy(), sd.copy(), pi.copy())
    _, mu, sd, pi = best
    order = np.argsort(mu)
    return mu[order], sd[order], pi[order]


def gmm_means(x: np.ndarray):
    """在 log(1+W) 空间拟合，回变换成秒：E[W] = exp(μ+σ²/2) − 1。"""
    mu, sd, pi = fit_gmm1d(x)
    means = np.expm1(mu + 0.5 * sd ** 2)
    return means, mu, sd, pi


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    print("D²Co：两成分 GMM 估 w⁺/w⁻ + 校正还原兴趣（§23.4.3）\n")

    data = make_watch_data(n=40000, dim=16, seed=SEED)
    X, w, d, r, p = (data["x"], data["w"].astype(np.float64), data["d"],
                     data["r"], data["p"])
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)

    # ---------------- 1) 分布长什么样 ----------------
    print("① 固定 Duration 区间内的观看时长分布：真的有两个团吗？")
    band = (d >= 40) & (d < 60)
    wb = w[band]
    rows = []
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, 20),
                   (20, 40), (40, 60), (60, 1e9)):
        m = (wb >= lo) & (wb < hi)
        rows.append([f"[{lo:g},{hi:g})" if hi < 1e9 else f"[{lo:g},∞)",
                     int(m.sum()), f"{m.mean():.1%}"])
    table(["观看时长区间（秒）", "样本数", "占比"], rows)
    print(f"\n  {int(band.sum()):,} 条 40–60 秒视频：{np.mean(wb == 0):.1%} 是零观看，")
    print("  另有一大团挤在 1–3 秒（划走/误触），剩下的才是不连续的「看点内容」。")
    print("  两条曲线叠在一起，这就是 w⁺/w⁻ 分解在数据上的样子。\n")

    # ---------------- 2) 按 Duration 分组拟合 GMM ----------------
    print("② 每个 Duration 组内拟合两成分 GMM，与「按真实兴趣分组」的均值对比")
    edges = np.unique(np.quantile(d[tr], np.linspace(0, 1, 13)))
    edges[0], edges[-1] = -np.inf, np.inf
    n_b = len(edges) - 1
    kb = np.searchsorted(edges, d, side="right") - 1

    est_p = np.full(n_b, np.nan)     # GMM 估的 w⁺
    est_n = np.full(n_b, np.nan)     # GMM 估的 w⁻
    cnt = np.zeros(n_b)
    rows = []
    for b in range(n_b):
        m = (kb == b) & (w > 0) & np.isin(idx, tr)
        cnt[b] = ((kb == b) & np.isin(idx, tr)).sum()
        if m.sum() < 50:
            continue
        means, mu, sd, pi = gmm_means(np.log1p(w[m]))
        est_n[b], est_p[b] = means[0], means[-1]
        # 真实分量：直接用生成时的 r。注意 GMM 是在 w>0 的样本上拟合的，
        # 所以对照的基准也取 w>0，否则「12% 的零观看」会把基准压低，
        # 让误差看起来比实际大。
        mt = (kb == b) & np.isin(idx, tr)
        true_p = w[mt & (r == 1) & (w > 0)].mean()
        true_n = w[mt & (r == 0)].mean()
        lo = edges[b] if np.isfinite(edges[b]) else d.min()
        hi = edges[b + 1] if np.isfinite(edges[b + 1]) else d.max()
        rows.append([f"[{lo:.0f},{hi:.0f})", int(cnt[b]),
                     f"{true_p:.1f}", f"{est_p[b]:.1f}",
                     f"{true_n:.1f}", f"{est_n[b]:.1f}",
                     f"{est_p[b] / true_p - 1:+.1%}"])
    table(["Duration 组", "样本数", "真实 w⁺", "GMM w⁺", "真实 w⁻", "GMM w⁻",
           "w⁺ 相对误差"], rows)
    print("\n  w⁻ 估得很准（几秒的噪声观看本来就好分辨），w⁺ 则是系统性高估，")
    print("  短的组 8%、最长的组 36%。原因是高成分抓到的其实是一个「看完」的")
    print("  尖峰：感兴趣的人里有一半以上看到视频结束，w 就贴在 d 上，")
    print("  log 空间里它是一个又窄又高的峰，回变换后的均值自然高于")
    print("  「把部分观看也平均进来」的 E[W|r=1]。这不是 GMM 收敛不好的抖动，")
    print("  而是「高成分 ≠ 条件均值」这个结构性差异，它会直接进分母。\n")

    # ---------------- 3) 频次加权移动平均 ----------------
    print("③ 频次加权双向移动平均（T=2）平滑相邻 Duration 的估计")
    T = 2
    sm_p, sm_n = est_p.copy(), est_n.copy()
    for b in range(n_b):
        lo, hi = max(0, b - T), min(n_b, b + T + 1)
        sl = slice(lo, hi)
        valid = np.isfinite(est_p[sl])
        wts = cnt[sl][valid]
        if wts.sum() == 0:
            continue
        sm_p[b] = (est_p[sl][valid] * wts).sum() / wts.sum()
        sm_n[b] = (est_n[sl][valid] * wts).sum() / wts.sum()
    pick = [b for b in range(n_b) if np.isfinite(est_p[b])][:8]
    rows = []
    for b in pick:
        lo = edges[b] if np.isfinite(edges[b]) else d.min()
        hi = edges[b + 1] if np.isfinite(edges[b + 1]) else d.max()
        rows.append([f"[{lo:.0f},{hi:.0f})", f"{est_p[b]:.1f}", f"{sm_p[b]:.1f}",
                     f"{est_n[b]:.2f}", f"{sm_n[b]:.2f}"])
    table(["Duration 组", "w⁺ 原始", "w⁺ 平滑", "w⁻ 原始", "w⁻ 平滑"], rows)
    print("\n  平滑主要作用在样本量小的组：长 Duration 组只有一两千条，")
    print("  GMM 的单次估计抖动明显；借相邻组的样本量加权后曲线更连续。")
    print("  注意它只能降方差，降不了②里那种系统性低估——偏差要靠模型结构解决。\n")

    # ---------------- 4) 校正与评价 ----------------
    print("④ 三种兴趣 Label 与真实兴趣 r 的一致性（测试集）")
    wte, dte, rte, pte = w[te], d[te], r[te], p[te]
    kte = np.searchsorted(edges, dte, side="right") - 1
    kte = np.clip(kte, 0, n_b - 1)
    wp = sm_p[kte]
    wn = sm_n[kte]

    pcr = np.clip(wte / np.maximum(dte, 1e-6), 0, 1)
    lin = np.clip((wte - wn) / np.maximum(wp - wn, 1e-6), 0, 1)

    def expo_label(alpha: float) -> np.ndarray:
        # (e^{αw} − e^{αw⁻})/(e^{αw⁺} − e^{αw⁻})，公因子 e^{αw⁻} 直接约掉，
        # 用 expm1 计算差值，既避免大数溢出的相减抵消，也不会因 α 小而丢精度。
        num = np.expm1(alpha * (wte - wn))
        den = np.expm1(alpha * (wp - wn))
        return np.clip(num / np.maximum(den, 1e-12), 0, 1)

    # 对照：直接用「真实的」条件均值 E[W|r=1]、E[W|r=0] 做同一套线性校正。
    # 它不是上限——真实条件均值同样不等于 GMM 的高成分，而且它混入了
    # 「感兴趣但只看几秒」的样本，分母反而更像噪声。
    true_p = np.array([w[(kb == b) & (r == 1)].mean() if np.isfinite(est_p[b])
                       else np.nan for b in range(n_b)])
    true_n = np.array([w[(kb == b) & (r == 0)].mean() if np.isfinite(est_p[b])
                       else np.nan for b in range(n_b)])
    tp, tn = true_p[kte], true_n[kte]
    orc = np.clip((wte - tn) / np.maximum(tp - tn, 1e-6), 0, 1)

    # 桶内 AUC：把 Duration 的影响先扣掉，只看组内排序
    def within_bin_auc(label: np.ndarray) -> float:
        tot, nsum = 0.0, 0
        for b in range(n_b):
            m = kte == b
            if m.sum() > 10 and 0 < rte[m].sum() < m.sum():
                tot += auc(rte[m], label[m]) * m.sum()
                nsum += m.sum()
        return tot / max(nsum, 1)

    rows = [
        ["原始观看时长 w（不归一）", f"{auc(rte, wte):.3f}", "—",
         f"{within_bin_auc(wte):.3f}"],
        ["PCR：w/d", f"{auc(rte, pcr):.3f}", f"{mae(pte, pcr):.3f}",
         f"{within_bin_auc(pcr):.3f}"],
        ["D²Co(A)：线性校正", f"{auc(rte, lin):.3f}", f"{mae(pte, lin):.3f}",
         f"{within_bin_auc(lin):.3f}"],
        ["D²Co(S)：指数校正 α=0.05", f"{auc(rte, expo_label(0.05)):.3f}",
         f"{mae(pte, expo_label(0.05)):.3f}",
         f"{within_bin_auc(expo_label(0.05)):.3f}"],
        ["对照：真实条件均值做线性校正", f"{auc(rte, orc):.3f}",
         f"{mae(pte, orc):.3f}", f"{within_bin_auc(orc):.3f}"],
    ]
    table(["兴趣 Label", "AUC(r)", "MAE(p)", "桶内 AUC"], rows)
    print("\n  三个数各自说明一件事：")
    print("    · AUC(r)：D²Co(A) 0.862 vs PCR 0.830，+0.032。改善来自")
    print("      「先减掉 w⁻ 再加权」这个结构——它把「光划走」的样本压到 0 附近，")
    print("      让真正看了内容的人和它们分开；")
    print("    · MAE(p)：0.358 → 0.358，几乎没动。Label 更适合排序，")
    print("      但直接把它当兴趣概率用（比如拿去算 pCTR）并不比 PCR 准，")
    print("      因为分母 w⁺ 高估，Label 整体被压缩向 0；")
    print("    · 桶内 AUC：0.829 → 0.860，和整体 AUC 几乎同幅提升。")
    print("      说明增益不是「Duration 信息被扣掉了」的假象，")
    print("      而是同一 Duration 内的排序确实变好了。")
    print("  对照行更值得注意：用「真实条件均值」做同一套线性校正只有 0.859，")
    print("  还略低于 GMM 版本的 0.862。条件均值把「感兴趣但只看三秒」")
    print("  也平均进来，分母被稀释；GMM 的高成分更接近「看完」的尖峰，")
    print("  恰好把这类样本的贡献压了下去。所以「估得越像 E[W|r=1] 越好」")
    print("  在这里并不成立——高成分不是条件均值，而是条件均值里的一支。")
    print("  作为参照，直接用生成用的兴趣概率 p 当分数，AUC(r) 只有 0.817。")
    print("  这不构成「上限」——r 是按 p 抽样出来的，而 w 又是在 r 的基础上")
    print("  再生成的，所以 w 天然比 p 多带了信息。有意义的比较只有同一行内")
    print("  横向看：w / PCR / D²Co 三者用的是同一份观测，D²Co 多出 0.032。\n")
    print("  正文说 w⁺ 和 w⁻ 接近时分母小、误差被放大；本例里 w⁻ 稳定在 2.5 秒，")
    print("  w⁺ 从十几秒到上百秒，分母并不小。真正放大误差的是反方向：")
    print("  高成分高估 8%~36%，且长视频组更严重，于是长视频的 Label 被压得")
    print("  更靠近 0，排序时它们的相对差异被缩小——但方向没变，所以 AUC 仍涨。\n")

    # ---------------- 5) α 的稳健性 ----------------
    print("⑤ 指数校正的 α：它控制对估计误差的敏感度")
    rows = []
    for alpha in (0.01, 0.02, 0.05, 0.1, 0.2):
        lab = expo_label(alpha)
        rows.append([f"{alpha:.2f}", f"{auc(rte, lab):.3f}", f"{mae(pte, lab):.3f}",
                     f"{lab.std():.3f}", f"{np.mean((lab < 1e-6) | (lab > 1 - 1e-6)):.1%}"])
    table(["α", "AUC(r)", "MAE(p)", "Label 标准差", "落在端点的比例"], rows)
    print("\n  α→0 时 expm1(αx)/α → x，指数校正精确退化成线性校正，")
    print("  所以 0.862 这条基线不会因为换公式而改变。α 变大后，")
    print("  落在两端的样本变多（56% → 64%）、标准差反而下降（0.436 → 0.375）：")
    print("  中间地带被推到 0 或 1，分布从「连续」变成「两端堆积」。")
    print("  代价在 MAE 上：α=0.05 之前 0.356，α=0.2 时劣化到 0.376。")
    print("  也就是说 α 在这里不是提升排序的旋钮，只是改变 Label 的离散程度；")
    print("  论文按数据分布调它，换数据集不能照搬。\n")

    print("⑥ 一句话：D²Co 把观看时长拆成「兴趣 × 偏差 + (1−兴趣) × 噪声」，")
    print("   再用同 Duration 下的大量样本把两个潜在分量估出来反解兴趣。")
    print("   它的收益主要体现在排序（AUC +0.03）而不是概率校准（MAE 不动），")
    print("   而高成分估的是「看完」那一支、不是条件均值——这一点决定了")
    print("   它的 Label 会比「用真实条件均值」更锐利，也更容易受长尾影响。")


if __name__ == "__main__":
    main()
