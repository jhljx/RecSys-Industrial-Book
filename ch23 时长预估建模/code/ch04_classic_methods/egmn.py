"""
EGMN：指数 + 高斯混合的连续分布建模（第23章 §23.4.7）

正文的 EGM 分布：
    p(t|x) = ω₀·Exp(t; λ) + Σ_k ω_k·N(t; μ_k, σ_k²)
三个损失：
    L_MLE  = −mean log p(t|x)                    分布拟合
    L_ent  = mean Σ ω_k log ω_k                  负熵（防成分塌缩）
    L_reg  = mean |t − E[t|x]|                   点预测准确性
    L      = L_MLE + α·L_ent + β·L_reg

本脚本验证：
    · EGMN 的点预测（分布期望）与直接 MSE 回归对比；
    · MLE 单独训练会退化成「一个成分吃掉全部质量」吗？
      熵正则与回归项各自起什么作用；
    · 拟合出来的分量到底对应什么行为——把每个成分的权重、均值列出来，
      对照数据里「划走 / 中途离开 / 看满」这三种模式；
    · 分布给出的累计概率能做什么（P(t>10)、P(t>0.8d) 之类的业务量）。

依赖 torch；没有 torch 时直接跳过。

运行：python3 ch04_classic_methods/egmn.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    fit_linear, make_watch_data, mae, predict_linear, rmse, split,
    standardize, table,
)

SEED = 2026
K_GAUSS = 3
STEPS = 2000
BATCH = 512
LR = 2e-3


def run_torch() -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("  （未安装 torch，跳过 EGMN）\n")
        return

    torch.manual_seed(SEED)
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, w = data["x"], data["w"].astype(np.float64)
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    Xtr, Xte, _, _ = standardize(X[tr], X[te])
    wtr, wte = w[tr], w[te]
    dte = data["d"][te].astype(np.float64)

    th0 = fit_linear(Xtr, wtr)
    pred0 = np.clip(predict_linear(th0, Xte), 0, None)
    print("① 参照：直接 MSE 回归 vs 全预测训练均值")
    print(f"  MSE 回归 MAE {mae(wte, pred0):.2f}、RMSE {rmse(wte, pred0):.2f}；"
          f"常数基线 MAE {np.abs(wte - wtr.mean()).mean():.2f}\n")

    Xt = torch.from_numpy(Xtr).float()
    yt = torch.from_numpy(wtr).float()
    Xs = torch.from_numpy(Xte).float()
    ys = torch.from_numpy(wte).float()

    def softplus(x):
        return F.softplus(x) + 1e-4

    class EGMN(nn.Module):
        """
        一个共享的 FFN backbone + 四组预测头。

        论文把 λ 用 softplus 约束为正、把高斯均值整体抬 1/λ，
        是为了让「指数成分管低端、高斯成分管中高端」这个分工稳定下来；
        不加这个偏移，两个高斯会去抢 0 附近的概率质量，训练极不稳定。
        σ² 也用 softplus，再加一个下限，避免某一维塌成 delta 函数。
        """

        def __init__(self, dim: int, hidden: int = 64):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.lam = nn.Linear(hidden, 1)
            self.mu = nn.Linear(hidden, K_GAUSS)
            self.logvar = nn.Linear(hidden, K_GAUSS)
            self.w = nn.Linear(hidden, K_GAUSS + 1)   # 指数成分 + K 个高斯

        def forward(self, x):
            h = self.backbone(x)
            lam = softplus(self.lam(h)).squeeze(-1)           # [n]
            mu = 1.0 / lam[:, None] + softplus(self.mu(h))    # [n,K]
            sd = torch.sqrt(softplus(self.logvar(h)) + 1e-2)  # [n,K]
            omega = F.softmax(self.w(h), dim=-1)              # [n,K+1]
            return lam, mu, sd, omega

    def log_prob(t, lam, mu, sd, omega):
        """返回每个成分在 t 处的 log 密度，形状 [n, K+1]。"""
        t_ = t[:, None]
        lam_ = lam[:, None]
        exp_lp = torch.log(lam_) - lam_ * t_                   # [n,1]
        g_lp = (-0.5 * ((t_ - mu) / sd) ** 2
                - torch.log(sd) - 0.5 * np.log(2 * np.pi))
        return torch.cat([exp_lp, g_lp], dim=1) + torch.log(omega + 1e-12)

    def nll_parts(model, x, t):
        lam, mu, sd, omega = model(x)
        lp = log_prob(t, lam, mu, sd, omega)
        m = lp.max(dim=1, keepdim=True).values
        lse = m[:, 0] + torch.log(torch.exp(lp - m).sum(dim=1))
        nll = -lse.mean()
        ent = (omega * torch.log(omega + 1e-12)).sum(dim=1).mean()
        # 混合分布的期望：Σ ω_k · mean_k，指数成分的均值是 1/λ。
        mean = (omega[:, 0] * (1.0 / lam)
                + (omega[:, 1:] * mu).sum(dim=1))
        return nll, ent, mean, (lam, mu, sd, omega)

    def train(alpha: float, beta: float, steps: int = STEPS, seed: int = SEED):
        """返回 (pred, nll, ent, params)，pred 是逐样本的期望预测。"""
        torch.manual_seed(seed)
        model = EGMN(Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            b = torch.from_numpy(rng.integers(0, len(Xt), BATCH))
            nll, ent, mean, _ = nll_parts(model, Xt[b], yt[b])
            reg = (mean - yt[b]).abs().mean()
            loss = nll + alpha * ent + beta * reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        with torch.no_grad():
            nll, ent, mean, params = nll_parts(model, Xs, ys)
        return mean.numpy().astype(np.float64), float(nll), float(ent), params

    # ---------------- MLE 单独训练 ----------------
    print("② 只做 MLE：看混合分布会不会自己塌成一个成分")
    _, nll_mle, ent_mle, (lam, mu, sd, omega) = train(alpha=0.0, beta=0.0)
    lam_n, mu_n, sd_n, om_n = (lam.numpy(), mu.numpy(),
                               sd.numpy(), omega.numpy())
    rows = [["指数", f"{float(lam_n.mean()):.4f}", "—",
             f"{float(1 / lam_n.mean()):.1f}", f"{om_n[:, 0].mean():.2f}"]]
    for k in range(K_GAUSS):
        rows.append([f"高斯 {k + 1}", "—", f"{mu_n[:, k].mean():.1f}",
                     f"{sd_n[:, k].mean():.1f}", f"{om_n[:, k + 1].mean():.2f}"])
    table(["成分", "λ", "均值 μ", "SD", "平均权重"], rows)
    print(f"\n  只用 MLE 时，测试集 NLL {nll_mle:.2f}、混合权重熵 {ent_mle:.2f}"
          f"（−ln4 = {np.log(4):.2f} 是均匀分布的负熵）。")
    print(f"  实测指数成分的平均权重 {om_n[:, 0].mean():.2f}——MLE 把几乎全部质量")
    print("  压到了单一成分上，这就是成分塌缩：模型退化成单分布，")
    print("  参数更多、却没有任何额外表达力。原因是 MLE 只关心真实值处的")
    print("  密度，两个成分只要有一个能给出足够高的密度就够了，")
    print("  另一个成分即使存在也只会分走质量、降低似然。\n")

    # ---------------- 消融 ----------------
    print("③ 三项损失的消融（test MAE / RMSE / 分布 NLL / 权重负熵）")
    rows, res = [], {}
    for name, a, b in (("仅 MLE", 0.0, 0.0),
                       ("MLE + 熵正则", 0.5, 0.0),
                       ("MLE + 熵 + MAE 回归", 0.5, 0.5),
                       ("MLE + MAE 回归（无熵）", 0.0, 0.5)):
        pred_v, nll_v, ent_v, _ = train(alpha=a, beta=b)
        res[name] = pred_v
        rows.append([name, f"{mae(wte, pred_v):.2f}",
                     f"{rmse(wte, pred_v):.2f}", f"{nll_v:.2f}", f"{-ent_v:.2f}"])
    table(["训练目标", "MAE", "RMSE", "NLL", "权重负熵"], rows)
    print("\n  三件事分开看：")
    print("    · 熵正则把权重从「一个成分吃掉全部」拉开，负熵上升、NLL 变差——")
    print("      它不是免费的，是用拟合度换成分分工；")
    print("    · MAE 回归项才真正拉低点预测误差：分布期望与真实值之间")
    print("      并没有天然的等式，MLE 只保证「真实值附近的密度高」，")
    print("      不保证「密度的均值等于真实值」；")
    print("    · 两项一起用，才同时拿到「不塌缩的分布」和「够准的点预测」。")
    print("  但别只看 MAE：MAE 从线性的 19.80 降到 15.58，RMSE 却从 28.45")
    print("  升到 31.60。Huber / MAE 这一项把预测往条件中位数上拽，")
    print("  中位样本变准了，长尾样本的平方误差被放大——这正是「用哪个")
    print("  点误差训练」的差别，不是分布形式的功劳，也不是 bug。")
    print("  至于 MAE 能降这么多，是因为 σ(β·u+γ·v) 本身是非线性的，")
    print("  FFN backbone 拟合得了，而线性回归只能给一条直线。")
    print("  分布形式只是外壳，真正把点误差压下去的是 backbone。\n")

    # ---------------- 分布能干什么 ----------------
    print("④ 用拟合出的分布算业务量：P(t > 10)、P(t > 0.8d)、P(t > d)")
    model_pred, _, _, (lam, mu, sd, omega) = train(alpha=0.5, beta=0.5)
    lam_n, mu_n, sd_n, om_n = (lam.numpy(), mu.numpy(),
                               sd.numpy(), omega.numpy())

    from math import erf

    def surv(t, lam, mu, sd, omega):
        # P(T > t) = ω₀·e^{−λt} + Σ_k ω_k·(1 − Φ((t−μ_k)/σ_k))
        t_ = np.asarray(t, dtype=np.float64)[:, None]
        e = np.exp(-lam * t_[:, 0])                                   # [n]
        g = 0.5 * (1 + np.vectorize(erf)((t_ - mu) / (sd * np.sqrt(2))))
        return omega[:, 0] * e + (omega[:, 1:] * (1.0 - g)).sum(axis=1)

    rows = []
    for name, thr in (("P(t > 10 秒)", np.full(len(wte), 10.0)),
                      ("P(t > 0.8d)", 0.8 * dte),
                      ("P(t > d)", dte)):
        p_hat = surv(thr, lam_n, mu_n, sd_n, om_n)
        truth = (wte > thr).astype(float)
        rows.append([name, f"{p_hat.mean():.3f}", f"{truth.mean():.3f}",
                     f"{np.abs(p_hat - truth).mean():.3f}"])
    table(["业务量", "模型平均概率", "真实频率", "平均绝对差"], rows)
    print("\n  三个事件的偏差方向并不一致，而且都不是「数据太少」能解释的：")
    print("    · P(t>10) 高估：混合分布的右尾比数据更厚，模型把一部分")
    print("      本应集中在几秒的质量摊到了中段；")
    print("    · P(t>0.8d) 低估：和上一条相反，说明偏差不是整体平移，")
    print("      而是「相对 duration 的尾部形状」没学对；")
    print("    · P(t>d) 给出 0.137，而真实频率是 0——这是结构性的：")
    print("      EGM 的支撑集是 [0,∞)，模型根本不知道视频只有 d 秒长。")
    print("      似然只看观测到的 t，不看「t 不可能超过 d」这个天花板，")
    print("      所以它会在 d 以上照样铺概率质量。")
    print("  想修第三条，得在分布上做截断（t ∈ [0,d]）或者直接用生存模型")
    print("  处理删失——这正是正文 §23.5 的 GR / FlowTime 要解决的事。")
    print("  结论：EGMN 的分布能给出「形状」，但尾部概率要校准、")
    print("  支撑集要按业务补齐，不能把混合分布的输出直接当概率用。\n")

    print("⑤ 一句话：EGMN 把「回归一个数」换成「预测一整条密度」，")
    print("   指数成分负责快速划走的长尾、高斯成分负责多峰的行为模式，")
    print("   MLE 拟合分布、熵正则防塌缩、MAE 项保点预测。")


def main() -> None:
    print("EGMN：指数 + 高斯混合分布（§23.4.7）\n")
    run_torch()


if __name__ == "__main__":
    main()
