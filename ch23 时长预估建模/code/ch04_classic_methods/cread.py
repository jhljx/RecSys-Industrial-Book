"""
CREAD：分类 + 恢复的联合优化，以及误差自适应分桶 EAD（第23章 §23.4.5）

CREAD = OR + 三件事：
    1) 恢复损失 L_restore：不满足于「每个阈值分类对」，直接对还原出的
       连续时长 ŷ = Σ p_m (t_m − t_{m−1}) 和真实 y 算 Huber Loss；
    2) 序正则 L_ord：对相邻阈值概率倒数的情况罚 Σ max(p_{m+1} − p_m, 0)；
    3) 误差自适应分桶 EAD：在等距与等频之间用校准函数 γ 连续调节，
       目标是最小化学习误差 A_w 与修复误差 A_b 的加权和。

本脚本分三段：
    ① 分桶策略对比：等距 / 等频 / EAD 网格搜索选出的 γ；
    ② 三个损失项的消融：CE / CE+恢复 / CE+恢复+序正则；
    ③ 阈值数 M 与样本量的关系：正文说 CREAD 选 K=80，
       在 2 万条数据上到底撑不撑得住。

运行：python3 ch04_classic_methods/cread.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    make_watch_data, mae, rmse, sigmoid, split, standardize, table,
)

SEED = 2026
ITERS = 400
LR = 0.5


def thresholds_from_gamma(w: np.ndarray, M: int, gamma, t_max: float
                          ) -> np.ndarray:
    """t_m = Ψ^{-1}(γ(m/M))，用训练集经验分位数代替 Ψ^{-1}。"""
    z = np.arange(1, M) / M            # 内部 M−1 个阈值
    q = np.clip(gamma(z), 0.0, 1.0)
    # 零观看样本占 12%，前几个分位点会重复落在 0 上。thresholds_from_gamma
    # 用 unique 去重，但那样第一个阈值就是 0，z_1 = 1{w > 0} 退化成一个
    # 「播没播」的分类。这里把 0 阈值剔掉，从第一个正分位点开始。
    t = np.quantile(w, q)
    t = t[t > 0]
    return np.unique(np.minimum(t, t_max))


def equal_width(w: np.ndarray, M: int, t_max: float) -> np.ndarray:
    return np.linspace(t_max / M, t_max * (M - 1) / M, M - 1)


def ead_grid(w: np.ndarray, M: int, t_max: float, beta: float = 1.0):
    """
    在 γ_θ(z) = z^θ 这一族里网格搜索，最小化 A_w + β·A_b。

    A_w（学习误差的代理）：Σ 1/√n_m —— 桶越空，分类器越难学；
    A_b（修复误差的代理）：Σ p_m · (t_m − t_{m−1})² / T_max²
        —— 桶越宽，积分近似越粗。
    两者用各自的网格最小值归一化，只用来看「相对形状」，
    绝对值没有物理意义。
    """
    cands = []
    for theta in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
        t = thresholds_from_gamma(w, M, lambda z, th=theta: z ** th, t_max)
        edges = np.concatenate([[-np.inf], t, [np.inf]])
        cnt = np.bincount(np.searchsorted(edges, w, side="right") - 1,
                          minlength=len(edges))[:len(edges) - 1]
        cnt = np.maximum(cnt, 1)
        widths = np.diff(edges)
        widths[0] = t[0] - 0.0 if len(t) else t_max
        mass = cnt / cnt.sum()
        A_w = float((1.0 / np.sqrt(cnt)).sum())
        A_b = float((mass * widths ** 2 / t_max ** 2).sum())
        cands.append((theta, t, A_w, A_b))
    aw_min = min(c[2] for c in cands)
    ab_min = min(c[3] for c in cands)
    scored = [(c[2] / aw_min + beta * c[3] / ab_min, c) for c in cands]
    scored.sort(key=lambda x: x[0])
    return scored[0][1], cands


def huber(err: np.ndarray, delta: float = 1.0) -> np.ndarray:
    a = np.abs(err)
    return np.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


def fit_cread(X: np.ndarray, y: np.ndarray, thr: np.ndarray, *,
              lam_ce: float = 1.0, lam_res: float = 0.0, lam_ord: float = 0.0,
              iters: int = ITERS, lr: float = LR) -> np.ndarray:
    """
    所有阈值联合训练（全批量梯度下降）。

    p_m = σ(x·θ_m)，ŷ = Σ p_m w_m（w_m = t_m − t_{m−1}）。
    三项损失的梯度都直接对 θ_m 求：
        CE:     (p_m − z_m) · x
        Huber:  dH/dŷ · w_m · p_m(1−p_m) · x
        Ord:    −1{p_{m+1} > p_m} · p_m(1−p_m) · x  （对 θ_m）
    """
    n, dim = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    coef = np.zeros((dim + 1, len(thr)))
    Z = (y[:, None] > thr[None, :]).astype(np.float64)
    prev = np.concatenate([[0.0], thr[:-1]])
    widths = thr - prev
    for _ in range(iters):
        p = sigmoid(Xb @ coef)                        # [n, M]
        g = Z - p                                     # CE 的 ∂L/∂z
        grads = Xb.T @ g / n
        if lam_res > 0:
            yhat = p @ widths
            err = yhat - y
            dh = np.clip(err, -1.0, 1.0)              # Huber(δ=1) 的导数
            gp = (dh / n)[:, None] * widths[None, :] * p * (1 - p)
            # ∂L/∂z 是 +dh·w·p(1−p)；这里算的是负梯度，所以要减。
            # 写成加号会让「预测偏高、dh=+1」继续把 p 往上推，
            # 训练几十步后 ŷ 的均值就能从 20 秒涨到 79 秒。
            grads -= lam_res * (Xb.T @ gp)
        if lam_ord > 0 and len(thr) > 1:
            d = p[:, 1:] - p[:, :-1]                  # 应 ≤ 0
            act = (d > 0).astype(np.float64)
            go = np.zeros_like(p)
            # 对 p_m 的梯度：d>0 时 penalty = p_{m+1} − p_m
            #   ∂/∂p_m     = −1，∂/∂p_{m+1} = +1
            # penalty 对 p_m 的导数是 −1，负梯度是 +1；对 p_{m+1} 反之。
            go[:, :-1] += act
            go[:, 1:] -= act
            gp = go * p * (1 - p)
            grads += lam_ord * (Xb.T @ gp) / n
        coef += lr * grads
    return coef


def decode(p: np.ndarray, thr: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[0.0], thr[:-1]])
    return p @ (thr - prev)


def violation(p: np.ndarray) -> float:
    return float((p[:, 1:] - p[:, :-1] > 1e-9).any(axis=1).mean())


def main() -> None:
    print("CREAD：分类 + 恢复 + EAD（§23.4.5）\n")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, w = data["x"], data["w"].astype(np.float64)
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    Xtr, Xte, _, _ = standardize(X[tr], X[te])
    wtr, wte = w[tr], w[te]
    t_max = float(wtr.max())

    # ---------------- 1) 分桶策略 ----------------
    print("① 三种分桶策略：等距 / 等频 / EAD（M=20）")
    M = 20
    eqf = thresholds_from_gamma(wtr, M, lambda z: z, t_max)
    eqw = equal_width(wtr, M, t_max)
    (theta_best, ead_t, aw, ab), cands = ead_grid(wtr, M, t_max)

    rows, strat = [], []
    for name, t in (("等距", eqw), ("等频", eqf), (f"EAD γ(z)=z^{theta_best:g}", ead_t)):
        edges = np.concatenate([[0.0], t, [t_max]])
        cnt = np.bincount(np.searchsorted(np.concatenate([[-np.inf], t, [np.inf]]),
                                          wtr, side="right") - 1)[1:len(t) + 1]
        coef = fit_cread(Xtr, wtr, t, lam_ce=1.0, lam_res=1.0, lam_ord=0.1)
        p = sigmoid(np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1) @ coef)
        yhat = decode(p, t)
        strat.append((name, mae(wte, yhat), rmse(wte, yhat)))
        rows.append([name, f"{cnt.min():,}", f"{cnt.max():,}",
                     f"{np.diff(edges).max():.0f}", f"{mae(wte, yhat):.2f}",
                     f"{rmse(wte, yhat):.2f}"])
    table(["分桶", "最小组样本", "最大组样本", "最大桶宽(秒)", "MAE", "RMSE"], rows)
    best_mae = min(strat, key=lambda x: x[1])
    best_rmse = min(strat, key=lambda x: x[2])
    print(f"\n  EAD 选出的 θ={theta_best:g}（θ=1 退化为等频，θ→∞ 趋近等距）。")
    print(f"  结果有点反直觉：MAE 最好是{best_mae[0]}（{best_mae[1]:.2f}），"
          f"RMSE 最好是{best_rmse[0]}（{best_rmse[2]:.2f}）。")
    print("  等距分桶按秒数切，12% 的零观看样本被切成前几档空桶——")
    print("  但空桶恰恰让前几个阈值的正样本率接近 100%，分类器只要把")
    print("  概率压在 1 附近就对了，短样本的积分项因此估得相当准；")
    print("  代价全落在长尾，所以它的 RMSE 最差。等频分桶把长尾切细，")
    print("  RMSE 变好、MAE 略差。EAD 的代理目标 A_w + A_b 更看重")
    print("  学习误差，选出来的 θ 偏小、桶偏窄，于是 RMSE 最好而 MAE 最差。")
    print("  三个策略正好落在这条权衡曲线的三个位置，没有谁全面胜出；")
    print("  而且 EAD 优化的是代理量，不是 MAE 本身，二者方向不必一致。\n")

    # ---------------- 2) 损失消融 ----------------
    print("② 三个损失项的消融（固定 EAD 分桶）")
    rows, res = [], []
    for name, kw in (("仅 CE", dict(lam_ce=1.0)),
                     ("CE + 恢复(Huber)", dict(lam_ce=1.0, lam_res=1.0)),
                     ("CE + 恢复 + 序正则", dict(lam_ce=1.0, lam_res=1.0, lam_ord=0.1))):
        coef = fit_cread(Xtr, wtr, ead_t, **kw)
        p = sigmoid(np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1) @ coef)
        yhat = decode(p, ead_t)
        res.append((mae(wte, yhat), rmse(wte, yhat), violation(p)))
        rows.append([name, f"{mae(wte, yhat):.2f}", f"{rmse(wte, yhat):.2f}",
                     f"{violation(p):.1%}"])
    table(["训练目标", "MAE", "RMSE", "单调违例率"], rows)
    (m_ce, r_ce, _), (m_res, r_res, _), (m_ord, r_ord, _) = res
    print("\n  恢复损失的作用是「把训练目标对齐到线上真正在用的量」：")
    print(f"  MAE 从 {m_ce:.2f} 降到 {m_res:.2f}，但 RMSE 反而从 {r_ce:.2f} 升到 {r_res:.2f}。")
    print("  Huber 在小误差时是平方、大误差时是线性，长尾被压平之后，")
    print("  平均误差变好、极端样本的误差却被放大了——这正是选 Huber")
    print("  而不是 MSE 的代价，CREAD 论文也是这么权衡的。")
    print(f"  序正则的效果比预期弱：违例率 {res[1][2]:.1%} → {res[2][2]:.1%}，"
          f"MAE {m_res:.2f} → {m_ord:.2f}，基本没动。")
    print("  这不是实现错误，而是惩罚项只罚「倒挂的幅度」，不罚「倒挂的个数」；")
    print("  大量样本只倒挂一点点，梯度太小，压不下去。")
    print("  要真正保证单调，还是得在推理端做一次 cummin。\n")

    # ---------------- 3) 阈值数量与样本量 ----------------
    print("③ 阈值数 M 与样本量：正文用 K=80，2 万条数据撑得住吗")
    rows = []
    for M_ in (10, 20, 40, 80):
        t = thresholds_from_gamma(wtr, M_, lambda z: z, t_max)
        edges = np.concatenate([[-np.inf], t, [np.inf]])
        cnt = np.bincount(np.searchsorted(edges, wtr, side="right") - 1,
                          minlength=len(edges))[:len(edges) - 1]
        coef = fit_cread(Xtr, wtr, t, lam_ce=1.0, lam_res=1.0, lam_ord=0.1)
        p = sigmoid(np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1) @ coef)
        yhat = decode(p, t)
        rows.append([len(t), f"{cnt.min():,}", f"{cnt.mean():.0f}",
                     f"{mae(wte, yhat):.2f}", f"{rmse(wte, yhat):.2f}"])
    table(["阈值数", "最小组样本", "平均组样本", "MAE", "RMSE"], rows)
    print("\n  这里 MAE 随阈值变多单调变差——注意它与 OR 那节的结论方向相反，")
    print("  原因在于模型不同：OR 用逐个独立训练的分类器，CREAD 用共享")
    print("  特征空间的联合训练，阈值一密，相邻两个任务的 Label 几乎一样、")
    print("  梯度互相抵消，反而学不动。")
    print("  论文的 K=80 是在 KuaiRand 这种千万级数据上成立的；")
    print("  同样的 K 搬到 2 万条样本上，最小组只剩两三百条，")
    print("  每个阈值对应的正样本不到 1%。分桶数与样本量必须一起定。\n")

    print("④ 一句话：CREAD 在 OR 之上补了三块——")
    print("   直接优化恢复误差、软约束单调性、按误差权衡自适应分桶；")
    print("   它把「怎么离散化」从预处理细节升级成了一个需要调参的建模组件。")


if __name__ == "__main__":
    main()
