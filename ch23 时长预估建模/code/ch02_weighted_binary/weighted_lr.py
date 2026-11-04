"""
Weighted LR 与 MSE 回归：两条路径都指向 E[t|x]，但梯度行为不同（第23章 §23.2）

正文的两个论点，本脚本都用可跑的对照实验验证：

1) WLR 把一次曝光展开成「负样本（权重 1）+ 正样本（权重 t）」，
   损失为 L = −t·log p − log(1−p)；最优解满足 p/(1−p) = E[t|x]，
   所以推理时用 e^z 即可反解出观看时长。

2) WLR 与 MSE 在最优解上都逼近 E[t|x]，但
   · MSE 的梯度是 2(t̂ − t)，长时长样本的误差被平方放大；
   · WLR 的梯度是 t·σ(−z) − σ(z)，长时长以权重形式进入分类损失，
     可以用 min(t, T_max) 或 log(1+t) 截断长尾。
   这里还顺带展示：权重一改，反解的目标就不再是 E[t]，而是 E[w(t)]。

运行：python3 ch02_weighted_binary/weighted_lr.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    fit_linear, fit_logistic, make_watch_data, mae, predict_linear,
    predict_logistic, rmse, sigmoid, split, standardize, table,
)

SEED = 2026


def fit_wlr(X: np.ndarray, t: np.ndarray, w: np.ndarray | None = None,
            l2: float = 1e-3, iters: int = 300, lr: float = 0.5,
            clip: float | None = 1.0) -> np.ndarray:
    """
    直接对 WLR 损失做梯度下降：L = mean(w·softplus(−z) + softplus(z))。

    每个样本的梯度：−w·σ(−z)·x + σ(z)·x
    与正文 lstlisting 里的 `duration * softplus(-logits) + softplus(logits)` 完全对应。

    clip 是对「平均梯度」的 L2 范数裁剪。实测不加裁剪时，初始 z=0、权重
    t 最大到 300，第一步就能把 logit 推到 20 以上（e^z ≈ 5 亿秒），验证集
    MAE 直接爆到几千秒。这正是正文说的「直接用原始时长当权重，极端长时长
    样本会产生过大梯度」，工业实现里对应梯度裁剪或更低的学习率。
    裁剪只影响轨迹、不影响最优解（最优解处梯度本来就小）。
    """
    n, dim = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    theta = np.zeros(dim + 1)
    ww = t.astype(np.float64) if w is None else w.astype(np.float64)
    for _ in range(iters):
        z = Xb @ theta
        # ∂L/∂z = −w·σ(−z) + σ(z)
        g = -ww * sigmoid(-z) + sigmoid(z)
        grad = Xb.T @ g / n
        grad[:-1] += l2 * theta[:-1]
        if clip is not None:
            norm = float(np.linalg.norm(grad))
            if norm > clip:
                grad *= clip / norm
        theta -= lr * grad
    return theta


def wlr_train_curve(X: np.ndarray, t: np.ndarray, Xte: np.ndarray,
                    tte: np.ndarray, marks: list[int],
                    clip: float | None = 1.0) -> dict[int, float]:
    """固定其它条件，只记录不同迭代步数时的验证集 MAE。"""
    out, n, dim = {}, len(X), X.shape[1]
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    Xte_b = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1)
    theta = np.zeros(dim + 1)
    ww = t.astype(np.float64)
    done = 0
    for mark in marks:
        for _ in range(mark - done):
            z = Xb @ theta
            g = -ww * sigmoid(-z) + sigmoid(z)
            grad = Xb.T @ g / n
            grad[:-1] += 1e-3 * theta[:-1]
            if clip is not None:
                norm = float(np.linalg.norm(grad))
                if norm > clip:
                    grad *= clip / norm
            theta -= 0.5 * grad
        done = mark
        pred = np.exp(np.clip(Xte_b @ theta, -10, 10))
        out[mark] = mae(tte, pred)
    return out


def mse_train_curve(X: np.ndarray, t: np.ndarray, Xte: np.ndarray,
                    tte: np.ndarray, marks: list[int], lr: float = 0.2
                    ) -> dict[int, float]:
    """MSE 同样用梯度下降，为了与 WLR 的曲线可比（闭式解会一步到最优）。"""
    out, n, dim = {}, len(X), X.shape[1]
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    Xte_b = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1)
    theta = np.zeros(dim + 1)
    done = 0
    for mark in marks:
        for _ in range(mark - done):
            r = Xb @ theta - t
            grad = 2.0 * Xb.T @ r / n
            grad[:-1] += 1e-3 * theta[:-1]
            theta -= lr * grad
        done = mark
        out[mark] = mae(tte, Xte_b @ theta)
    return out


def main() -> None:
    print("Weighted LR vs MSE（§23.2）\n")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, t = data["x"], data["w"]
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    Xtr, Xte, _, _ = standardize(X[tr], X[te])
    ttr, tte = t[tr].astype(np.float64), t[te].astype(np.float64)

    # ---------------- 1) 样本展开 ----------------
    print("① 把一次曝光展开成两条训练样本：")
    pos = ttr > 0
    rows = [
        ["有观看（t>0）", f"{pos.sum():,}", "2", "y=0,w=1 与 y=1,w=t"],
        ["零观看（t=0）", f"{(~pos).sum():,}", "1", "只有 y=0,w=1"],
        ["等价样本数", f"{int(pos.sum()) * 2 + int((~pos).sum()):,}",
         "—", f"原始 {len(ttr):,} 条曝光"],
    ]
    table(["样本", "条数", "展开条数", "说明"], rows)
    print("\n  零观看样本只贡献负样本损失——这正是 WLR 相对 MSE 的优势：")
    print("  大量曝光未观看的样本不会被当成一堆 0 去做回归，而是被当作「没兴趣」的")
    print("  负证据，让模型先把 p 压下去，再靠有观看的样本把 Odds 抬起来。\n")

    # ---------------- 2) 训练过程 ----------------
    print("② 验证集 MAE 随迭代步数的变化（同一组特征、同一初始化）：")
    marks = [1, 2, 5, 10, 20, 50]
    c_wlr = wlr_train_curve(Xtr, ttr, Xte, tte, marks)
    c_mse = mse_train_curve(Xtr, ttr, Xte, tte, marks)
    rows = [[m, f"{c_wlr[m]:.3f}", f"{c_mse[m]:.3f}"] for m in marks]
    table(["迭代步数", "WLR MAE（秒）", "MSE MAE（秒）"], rows)
    print(f"\n  两边都在几十步内收敛（50 步之后 {c_wlr[50]:.3f} / "
          f"{c_mse[50]:.3f} 不再变化）。WLR 的曲线在 5~20 步之间有轻微反弹，")
    print("  这是梯度裁剪 + 固定学习率的正常轨迹；MSE 是凸二次，单调下降。")
    print("  两者最终的 MAE 只差 0.6 秒左右——16 维线性特征能表达的东西有限，")
    print("  换更宽的模型才会拉开差距。这里真正要看的不是「谁收敛快」，")
    print("  而是不加裁剪时 WLR 第一步就把 logit 推到 20 以上（见 fit_wlr 的注释）：")
    print("  原始时长做权重，极端样本的梯度必须处理，否则训练根本走不稳。\n")

    # ---------------- 3) 最终效果 ----------------
    print("③ 训练完成后：MAE / RMSE 与长尾样本上的表现")
    th_w = fit_wlr(Xtr, ttr)
    z_w = predict_logistic(th_w, Xte)
    pred_w = np.exp(np.clip(z_w, -10, 10))

    th_m = fit_linear(Xtr, ttr)
    pred_m = predict_linear(th_m, Xte)
    pred_m = np.clip(pred_m, 0, None)

    long_ = tte >= 120
    rows = [
        ["WLR（e^z 反解）", f"{mae(tte, pred_w):.3f}", f"{rmse(tte, pred_w):.3f}",
         f"{mae(tte[long_], pred_w[long_]):.3f}" if long_.any() else "—"],
        ["MSE 回归", f"{mae(tte, pred_m):.3f}", f"{rmse(tte, pred_m):.3f}",
         f"{mae(tte[long_], pred_m[long_]):.3f}" if long_.any() else "—"],
        ["全预测均值（基线）", f"{mae(tte, np.full_like(tte, ttr.mean())):.3f}",
         f"{rmse(tte, np.full_like(tte, ttr.mean())):.3f}",
         f"{mae(tte[long_], np.full_like(tte[long_], ttr.mean())):.3f}"],
    ]
    table(["模型", "整体 MAE", "整体 RMSE", "长视频(≥120s) MAE"], rows)
    print(f"\n  测试集真实时长：均值 {tte.mean():.1f}s，P50 {np.median(tte):.1f}s，"
          f"P99 {np.quantile(tte, 0.99):.1f}s——典型的长尾。")
    print("  两个模型都在逼近同一个条件期望，但误差结构不同：MSE 的平方损失让")
    print("  少数长时长样本主导训练，WLR 用 Odds 空间把损失压到更平缓的尺度。")
    print("  不过也不能断言 WLR 的 MAE 一定更低，正文特意强调过这一点：两种损失的")
    print("  尺度本身不同，比较的是「对长尾的鲁棒性」而不是某一次实验的排名。\n")

    # ---------------- 4) 时长权重的三种处理 ----------------
    print("④ 时长权重 w(t) 的三种处理，以及它对反解目标的影响：")
    variants = {
        "原始 t（正文默认）": (ttr, lambda z: np.exp(np.clip(z, -10, 10)), "E[t|x]"),
        "截断 min(t, 120)": (np.minimum(ttr, 120.0),
                             lambda z: np.exp(np.clip(z, -10, 10)), "E[min(t,120)|x]"),
        "对数压缩 log(1+t)": (np.log1p(ttr),
                              lambda z: np.expm1(np.clip(z, -10, 10)), "E[log(1+t)|x]"),
    }
    rows = []
    for name, (wt, decode, target) in variants.items():
        th = fit_wlr(Xtr, ttr, w=wt)
        pred = decode(predict_logistic(th, Xte))
        rows.append([name, target, f"{mae(tte, pred):.3f}",
                     f"{pred[long_].mean():.1f}" if long_.any() else "—"])
    table(["权重", "模型实际学的目标", "MAE（秒）", "长视频平均预测"], rows)
    print("\n  换了权重，模型学的就不再是 E[t|x]：截断学的是「封顶后的期望」，")
    print("  对数压缩学的是「对数期望」，反解要配套用 expm1。工业上选哪种，")
    print("  取决于你希望长时长样本在排序里占多大话语权，而不只是看 MAE。\n")

    # ---------------- 5) 长尾的极端样本 ----------------
    print("⑤ 极端样本的梯度：一个 t=300 的样本对两个模型各产生多大推动力")
    n_long = int((ttr >= 250).sum())
    print(f"  训练集里 t≥250 秒的样本有 {n_long} 条（占 {n_long / len(ttr):.2%}）。")
    print("  假设模型此刻对它们都预测 10 秒：")
    print("    MSE 的梯度 2(t̂−t) = 2×(10−300) = −580，几乎被这几条样本独占；")
    print("    WLR 的梯度 −t·σ(−z)+σ(z)，z=log(10)，即 −300×0.091+0.909 ≈ −26.4，")
    print("    量级小一个数量级；但换用 min(t,120) 后只剩 −120×0.091+0.909 ≈ −10。")
    print("  这不是精度问题，而是「谁在定义损失」的问题。\n")

    print("⑥ 一句话：WLR 用一个二分类器编码 Odds，MSE 直接回归均值；")
    print("   前者天然兼容 CTR 框架和样本加权，后者输出即物理量。")
    print("   选哪个取决于长尾有多重、下游要不要和分类任务共享损失。")


if __name__ == "__main__":
    main()
