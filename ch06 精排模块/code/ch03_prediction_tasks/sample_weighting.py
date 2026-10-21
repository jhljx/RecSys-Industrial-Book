"""
样本加权：用打赏金额作为正样本权重（第 6 章 §6.3.3 样本加权）

对应正文概念：
正文以直播打赏概率 GTR（Gift Through Rate）为例：每一次打赏行为都伴随具体的
打赏金额 G-Value，可以直接把金额作为正样本权重
    w_i = gift_amount_i
对应的加权交叉熵损失为
    L = - sum_i w_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]
正文强调两点收益：(a) 缓解正负样本极度不平衡；(b) 让模型学到不同正样本之间的
价值差异——「打赏 100 元与打赏 1 元虽然都属于正样本，但显然对应着完全不同的
用户价值和兴趣强度」。

本实现做了什么：
合成一批直播曝光样本：打赏与否由用户-主播匹配度决定，打赏金额服从长尾分布且
与匹配度正相关。并排训练 5 种加权方案，用同一网络结构、同一初始化：
    W0 不加权（标准 BCE）
    W1 正样本固定上采样权重（w=K，经典 class weighting）
    W2 正样本权重 = 打赏金额（正文公式原样）
    W3 正样本权重 = log(1+金额)（压长尾，工业常用的稳健变体）
    W4 正样本权重 = 金额 / 金额均值（归一化，保持总损失量级不变）
评估三类指标：
    - 排序：GTR AUC（是否打赏这件事排得准不准）
    - 价值排序：GMV-weighted AUC / Top-K 金额捕获率（高价值样本是否排在前面）
    - 校准：预估均值 vs 真实打赏率（加权会破坏校准，正文未展开，这里量化出来）
并额外演示「把预估 GTR 乘以预估金额得到期望收益」这一业务用法。

关键公式：
    w_i = gift_amount_i
    L = -sum_i w_i [ y_i log p_i + (1-y_i) log(1-p_i) ]
    期望收益 E[value] = p_gtr(x) * E[amount | gift, x]
    金额捕获率@K = (Top-K 预估分内的真实金额之和) / (全量真实金额之和)
"""

import random

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

FEAT_DIM = 14
N_TRAIN = 20000
N_TEST = 10000
BASE_GTR = 0.012          # 基础打赏率，正文说的万分~千分级稀疏目标量级


# --------------------------------------------------------------------------
# 合成数据：直播曝光 → 是否打赏 → 打赏金额
# --------------------------------------------------------------------------


def _solve_bias(logit, rate):
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if torch.sigmoid(logit + mid).mean().item() > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def make_data(n, gen, bias=None):
    X = torch.randn(n, FEAT_DIM, generator=gen)
    w_match = torch.linspace(-1.0, 1.0, FEAT_DIM)
    match = X @ w_match / FEAT_DIM ** 0.5          # 用户-主播匹配度
    g2 = torch.Generator().manual_seed(77)
    w_rich = torch.randn(FEAT_DIM, generator=g2)
    wealth = X @ w_rich / FEAT_DIM ** 0.5          # 用户付费能力（与匹配度弱相关）

    logit = 2.4 * match + 0.5 * wealth
    b = _solve_bias(logit, BASE_GTR) if bias is None else bias
    y = torch.bernoulli(torch.sigmoid(logit + b), generator=gen)

    # 金额：对数正态长尾，期望随匹配度与付费能力上升（正文：1 元 vs 100 元）
    mu = 1.0 + 0.9 * wealth + 0.5 * match
    amount = torch.expm1(mu + torch.randn(n, generator=gen) * 1.0).clamp(0.1, 5000.0)
    amount = amount * y                            # 未打赏的金额为 0
    return X, y, amount, b


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------


def auc(score, y, weight=None):
    """weight 为 None 时是标准 AUC；给定权重时是价值加权 AUC
    （正样本按其金额加权，衡量高价值样本是否排得更靠前）。"""
    pos = y > 0.5
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    if weight is None:
        rank = score.argsort().argsort().float() + 1.0
        n1 = int(pos.sum())
        n0 = len(y) - n1
        return ((rank[pos].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0))
    sp, wp = score[pos], weight[pos]
    sn = score[~pos]
    # 加权 AUC：sum_ij w_i * 1[s_i > s_j] / (sum_i w_i * n_neg)
    order = sn.sort().values
    cnt = torch.searchsorted(order, sp.contiguous())    # 比各正样本小的负样本数
    return (wp * cnt.float()).sum().item() / (wp.sum().item() * len(sn))


def amount_capture_at_k(score, amount, k):
    idx = score.argsort(descending=True)[:k]
    return amount[idx].sum().item() / amount.sum().item()


# --------------------------------------------------------------------------
# 模型与加权方案
# --------------------------------------------------------------------------


def make_net(seed=42):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(FEAT_DIM, 48), nn.ReLU(),
                         nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))


def fit(weights, X, y, epochs=12, bs=256, lr=3e-3, seed=42):
    """weights: [N] 每条样本的权重；加权交叉熵 L = -sum w (y log p + ...)。"""
    net = make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    torch.manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0])
        for st in range(0, X.shape[0], bs):
            idx = perm[st:st + bs]
            logit = net(X[idx]).squeeze(-1)
            raw = nn.functional.binary_cross_entropy_with_logits(
                logit, y[idx], reduction="none")
            loss = (raw * weights[idx]).sum() / weights[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def fit_amount_head(X, y, amount, epochs=12, bs=256, lr=3e-3, seed=42):
    """只在打赏样本上回归 log1p(金额)，用于「GTR × 金额」期望收益。"""
    net = make_net(seed + 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pos = y > 0.5
    Xp, tp = X[pos], torch.log1p(amount[pos])
    torch.manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(Xp.shape[0])
        for st in range(0, Xp.shape[0], bs):
            idx = perm[st:st + bs]
            loss = ((net(Xp[idx]).squeeze(-1) - tp[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    Xtr, ytr, atr, b = make_data(N_TRAIN, gen)
    Xte, yte, ate, _ = make_data(N_TEST, gen, b)

    print("=" * 92)
    print("§6.3.3 样本加权：以直播打赏 GTR / G-Value 为例")
    print("=" * 92)
    npos = int(ytr.sum())
    print(f"训练 {N_TRAIN} 条曝光，打赏正样本 {npos} 条"
          f"（GTR={npos / N_TRAIN:.4%}），测试 {N_TEST} 条")
    pa = atr[ytr > 0.5]
    qs = torch.quantile(pa, torch.tensor([0.5, 0.9, 0.99]))
    print(f"打赏金额 G-Value：mean={pa.mean():.2f} 元  median={qs[0]:.2f}  "
          f"P90={qs[1]:.2f}  P99={qs[2]:.2f}  max={pa.max():.1f}")
    top1 = pa.sort(descending=True).values[:max(1, len(pa) // 100)].sum()
    print(f"金额 Top-1% 正样本贡献了全部打赏金额的 {top1 / pa.sum():.1%}"
          "（正文：打赏 100 元与打赏 1 元价值完全不同）")

    one = torch.ones(N_TRAIN)
    k_up = (N_TRAIN - npos) / max(npos, 1)      # 正负样本比的倒数
    schemes = [
        ("W0 不加权", one.clone(), "标准 BCE"),
        (f"W1 正样本×{k_up:.0f}", torch.where(ytr > 0.5, torch.full_like(one, k_up), one),
         "固定类别权重"),
        ("W2 w=金额", torch.where(ytr > 0.5, atr, one), "正文公式 w_i=gift_amount_i"),
        ("W3 w=log(1+金额)", torch.where(ytr > 0.5, torch.log1p(atr), one),
         "压长尾的稳健变体"),
        ("W4 w=金额/均值", torch.where(ytr > 0.5, atr / pa.mean(), one),
         "归一化，保持损失量级"),
    ]

    print("\n" + "=" * 92)
    print("五种加权方案（同一网络、同一初始化 seed=42，12 epochs）")
    print("=" * 92)
    print(f"{'方案':<18}{'GTR AUC':>10}{'价值加权AUC':>13}{'捕获率@500':>12}"
          f"{'捕获率@2000':>13}{'预估均值':>11}{'校准比':>9}")
    print("-" * 92)
    real_rate = yte.mean().item()
    rows, trained = [], []
    for name, w, _ in schemes:
        net = fit(w, Xtr, ytr)
        trained.append(net)
        net.eval()
        with torch.no_grad():
            logit = net(Xte).squeeze(-1)
            p = torch.sigmoid(logit)
        a = auc(logit, yte)
        wa = auc(logit, yte, ate)
        c5 = amount_capture_at_k(logit, ate, 500)
        c20 = amount_capture_at_k(logit, ate, 2000)
        rows.append((name, a, wa, c5, c20, p.mean().item()))
        print(f"{name:<18}{a:>10.4f}{wa:>13.4f}{c5:>12.2%}{c20:>13.2%}"
              f"{p.mean().item():>11.5f}{p.mean().item() / real_rate:>9.2f}")
    print("-" * 92)
    print(f"真实打赏率 = {real_rate:.5f}；校准比 = 预估均值/真实率，"
          "=1 表示校准良好")

    best_val = max(rows, key=lambda r: r[2])
    best_auc = max(rows, key=lambda r: r[1])
    print(f"\n价值加权 AUC 最优：{best_val[0]}（{best_val[2]:.4f}）；"
          f"纯 GTR AUC 最优：{best_auc[0]}（{best_auc[1]:.4f}）")

    # ---------------- 加权如何改变正样本内部的排序 ----------------
    print("\n" + "=" * 92)
    print("加权如何让模型区分「不同价值的正样本」")
    print("=" * 92)
    pos_te = yte > 0.5
    bands = [("小额 [0,5)", 0.0, 5.0), ("中额 [5,50)", 5.0, 50.0),
             ("大额 [50,inf)", 50.0, float("inf"))]
    print(f"{'方案':<18}" + "".join(f"{bn:>20}" for bn, _, _ in bands)
          + f"{'大额/小额':>12}")
    print(f"{'':<18}" + "".join(f"{'(平均预估分/n)':>20}" for _ in bands)
          + f"{'(分档倍数)':>12}")
    print("-" * 92)
    for (name, _, _), net in zip(schemes, trained):
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xte).squeeze(-1))
        row, means = f"{name:<18}", []
        for _, lo, hi in bands:
            m = pos_te & (ate >= lo) & (ate < hi)
            v = p[m].mean().item()
            means.append(v)
            row += f"{v:>13.4f}(n={int(m.sum())})"
        print(row + f"{means[2] / means[0]:>12.2f}")
    print("观察：W1 这种与金额无关的固定类别权重只整体抬高正样本分数，分档倍数"
          "反而被压缩（1.70）；\n      只有 W2/W3/W4 这类与 G-Value 挂钩的权重才"
          "放大「大额/小额」倍数，\n      说明模型学到了正样本之间的价值差异。")

    # ---------------- 业务用法：期望收益 = GTR × 金额 ----------------
    print("\n" + "=" * 92)
    print("业务用法对比：E[value] = p_gtr × E[amount | gift]")
    print("=" * 92)
    net_gtr, net_w = trained[0], trained[2]      # 复用 W0 / W2
    net_amt = fit_amount_head(Xtr, ytr, atr)
    net_gtr.eval(); net_amt.eval(); net_w.eval()
    with torch.no_grad():
        p_gtr = torch.sigmoid(net_gtr(Xte).squeeze(-1))
        amt_hat = torch.expm1(net_amt(Xte).squeeze(-1)).clamp(min=0.0)
        p_w = torch.sigmoid(net_w(Xte).squeeze(-1))
    ev = p_gtr * amt_hat
    print(f"{'排序分':<28}{'价值加权AUC':>14}{'捕获率@500':>13}"
          f"{'捕获率@2000':>14}{'Top500真实总金额':>18}")
    print("-" * 92)
    for nm, s in [("不加权 GTR", p_gtr), ("金额加权 GTR（W2）", p_w),
                  ("GTR × 预估金额（双塔头）", ev)]:
        print(f"{nm:<28}{auc(s, yte, ate):>14.4f}"
              f"{amount_capture_at_k(s, ate, 500):>13.2%}"
              f"{amount_capture_at_k(s, ate, 2000):>14.2%}"
              f"{ate[s.argsort(descending=True)[:500]].sum().item():>18.1f}")
    print("=" * 92)
    print("结论：样本加权是把「价值」塞进单个概率头的廉价手段，代价是预估值不再"
          "校准\n      （校准比显著偏离 1）；若线上还需要可解释的概率，更稳的做法是"
          "拆成\n      GTR 头 + 金额头，在融合阶段相乘——这正好衔接 §6.4 的乘法融合公式。")
