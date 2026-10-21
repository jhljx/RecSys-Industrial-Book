"""
稀疏互动信号建模：稀疏度对学习难度的影响（第 6 章 §6.3.2 稀疏互动信号建模）

对应正文概念：
正文列出了精排各互动目标的典型正样本率：
    CTR  3%~10%  /  LTR  <1%  /  CMTR  <千分级  /  FTR  <万分级  /  CVR 更稀疏
并指出「随着用户行为价值的不断提升，其对应标签往往也会越来越稀疏」，绝大部分
样本都是负样本，真正有价值的正样本极其有限。

本实现做了什么：
用同一套用户/物品特征、同一个共享底座，生成一组正样本率逐级下降的互动标签
（ctr/ltr/cmtr/ftr/cvr），并对每个目标做三件事：
    (1) 打印正样本数、正样本率、每个 epoch 实际贡献梯度的正样本数；
    (2) 独立训练单任务模型，报告 AUC、预估均值与真实均值的偏差（校准）、
        以及 PR-AUC（稀疏目标下比 AUC 更敏感）；
    (3) 对比「单任务独立训练」与「共享底座多任务训练」，展示稠密目标
        （CTR）如何通过共享表示把梯度信号传给稀疏目标——这是 §6.3.4 的铺垫。
实验重复 3 个随机种子，打印均值 ± 标准差，用于呈现稀疏目标训练方差大的现象。

关键公式：
    正样本率：  r_k = (1/N) * sum_i y_{i,k}
    单任务损失：L_k = -(1/N) * sum_i [ y log p + (1-y) log(1-p) ]
    多任务损失：L   = sum_k L_k，底座 phi(x) 共享，仅塔头独立
    PR-AUC：    对 precision-recall 曲线做梯形积分
"""

import random

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

FEAT_DIM = 16
N_TRAIN = 20000
N_TEST = 10000
# (目标名, 目标正样本率)，按正文给的量级逐级稀疏
TASKS = [("ctr", 0.080), ("ltr", 0.010), ("cmtr", 0.003),
         ("ftr", 0.001), ("cvr", 0.0005)]


# --------------------------------------------------------------------------
# 合成数据：共享的潜在兴趣强度 z，各目标在 z 上取不同阈值形成不同稀疏度
# --------------------------------------------------------------------------


def _solve_bias(logit: torch.Tensor, rate: float) -> float:
    """二分求截距 b，使 mean(sigmoid(logit + b)) == rate。"""
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if torch.sigmoid(logit + mid).mean().item() > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def make_data(n: int, gen: torch.Generator, biases=None):
    X = torch.randn(n, FEAT_DIM, generator=gen)
    w = torch.linspace(-1.0, 1.0, FEAT_DIM)
    # 潜在兴趣强度：所有互动目标共享，这正是多任务能互相帮忙的根因
    z = X @ w / FEAT_DIM ** 0.5
    Y, out_bias = {}, {}
    for ti, (name, rate) in enumerate(TASKS):
        # 每个目标有自己的私有偏好分量，避免退化成完全相同的标签
        g = torch.Generator().manual_seed(1000 + ti)
        priv = (X @ torch.randn(FEAT_DIM, generator=g)) * 0.6 / FEAT_DIM ** 0.5
        logit = 2.2 * z + priv
        b = biases[name] if biases else _solve_bias(logit, rate)
        out_bias[name] = b
        Y[name] = torch.bernoulli(torch.sigmoid(logit + b), generator=gen)
    return X, Y, out_bias


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------


def auc(score: torch.Tensor, y: torch.Tensor) -> float:
    pos, neg = score[y > 0.5], score[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rank = score.argsort().argsort().float() + 1.0
    rpos = rank[y > 0.5].sum().item()
    n1, n0 = len(pos), len(neg)
    return (rpos - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(score: torch.Tensor, y: torch.Tensor) -> float:
    order = score.argsort(descending=True)
    ys = y[order]
    tp = torch.cumsum(ys, dim=0)
    prec = tp / torch.arange(1, len(ys) + 1, dtype=torch.float)
    total = ys.sum()
    if total == 0:
        return float("nan")
    rec = tp / total
    area, prev_r = 0.0, 0.0
    for i in range(len(ys)):
        r = rec[i].item()
        if r > prev_r:
            area += (r - prev_r) * prec[i].item()
            prev_r = r
    return area


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------


class SingleTask(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(FEAT_DIM, 48), nn.ReLU(),
                                 nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)


class SharedBottomMultiTask(nn.Module):
    """共享底座 + 每个目标一个轻塔。稠密目标的梯度会流进共享底座。"""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.body = nn.Sequential(nn.Linear(FEAT_DIM, 48), nn.ReLU(),
                                  nn.Linear(48, 32), nn.ReLU())
        self.heads = nn.ModuleDict({n: nn.Linear(32, 1) for n, _ in TASKS})

    def forward(self, X):
        h = self.body(X)
        return {n: self.heads[n](h).squeeze(-1) for n, _ in TASKS}


def fit_single(model, X, y, epochs=10, bs=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0])
        for st in range(0, X.shape[0], bs):
            idx = perm[st:st + bs]
            loss = bce(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def fit_multi(model, X, Y, epochs=10, bs=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0])
        for st in range(0, X.shape[0], bs):
            idx = perm[st:st + bs]
            out = model(X[idx])
            loss = sum(bce(out[n], Y[n][idx]) for n, _ in TASKS)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    Xtr, Ytr, biases = make_data(N_TRAIN, gen)
    Xte, Yte, _ = make_data(N_TEST, gen, biases)

    print("=" * 88)
    print("§6.3.2 稀疏互动信号建模：目标越有价值、标签越稀疏")
    print("=" * 88)
    print(f"训练 {N_TRAIN} 条 / 测试 {N_TEST} 条，batch=256 "
          f"（共 {N_TRAIN // 256} 个 batch/epoch）")
    print(f"\n{'目标':<8}{'正文量级':<16}{'训练正样本数':>14}{'实际正样本率':>14}"
          f"{'平均每batch正样本':>18}")
    print("-" * 88)
    level = {"ctr": "3%~10%", "ltr": "<1%", "cmtr": "<千分级",
             "ftr": "<万分级", "cvr": "更稀疏"}
    for name, _ in TASKS:
        npos = int(Ytr[name].sum())
        rate = npos / N_TRAIN
        print(f"{name.upper():<8}{level[name]:<16}{npos:>14}{rate:>14.4%}"
              f"{rate * 256:>18.2f}")
    print("注：FTR/CVR 平均每个 batch 不到一个正样本，梯度几乎全部来自负样本。")

    # ---------------- 多种子单任务 vs 多任务 ----------------
    seeds = [42, 7, 2024]
    single_res = {n: [] for n, _ in TASKS}
    multi_res = {n: [] for n, _ in TASKS}
    for sd in seeds:
        torch.manual_seed(sd)
        for name, _ in TASKS:
            m = fit_single(SingleTask(sd), Xtr, Ytr[name])
            m.eval()
            with torch.no_grad():
                s = m(Xte)
            single_res[name].append((auc(s, Yte[name]), pr_auc(s, Yte[name]),
                                     torch.sigmoid(s).mean().item()))
        torch.manual_seed(sd)
        mm = fit_multi(SharedBottomMultiTask(sd), Xtr, Ytr)
        mm.eval()
        with torch.no_grad():
            out = mm(Xte)
        for name, _ in TASKS:
            s = out[name]
            multi_res[name].append((auc(s, Yte[name]), pr_auc(s, Yte[name]),
                                    torch.sigmoid(s).mean().item()))

    def agg(rows, i):
        v = np.array([r[i] for r in rows])
        return v.mean(), v.std()

    print("\n" + "=" * 88)
    print("单任务独立训练（3 个随机种子，均值±标准差）")
    print("=" * 88)
    print(f"{'目标':<8}{'AUC':>18}{'PR-AUC':>18}{'预估均值':>12}"
          f"{'真实均值':>12}{'校准偏差':>12}")
    print("-" * 88)
    for name, _ in TASKS:
        a, asd = agg(single_res[name], 0)
        p, psd = agg(single_res[name], 1)
        mp, _ = agg(single_res[name], 2)
        real = Yte[name].mean().item()
        print(f"{name.upper():<8}{a:>11.4f}±{asd:.4f}{p:>11.4f}±{psd:.4f}"
              f"{mp:>12.5f}{real:>12.5f}{mp - real:>12.5f}")

    print("\n" + "=" * 88)
    print("共享底座多任务训练 vs 单任务（稠密目标为稀疏目标提供额外监督信号）")
    print("=" * 88)
    print(f"{'目标':<8}{'单任务AUC':>14}{'多任务AUC':>14}{'ΔAUC':>10}"
          f"{'单任务PR-AUC':>16}{'多任务PR-AUC':>16}")
    print("-" * 88)
    for name, _ in TASKS:
        a1, _ = agg(single_res[name], 0)
        a2, _ = agg(multi_res[name], 0)
        p1, _ = agg(single_res[name], 1)
        p2, _ = agg(multi_res[name], 1)
        print(f"{name.upper():<8}{a1:>14.4f}{a2:>14.4f}{a2 - a1:>+10.4f}"
              f"{p1:>16.4f}{p2:>16.4f}")
    print("=" * 88)
    print("观察：目标越稀疏，单任务 AUC 的种子间标准差越大、PR-AUC 越低；"
          "共享底座后\n      稀疏目标普遍受益——这正是正文 §6.3.4「稠密目标辅助"
          "稀疏目标学习」的\n      最朴素形式，后两节的样本加权与 ESMM 会进一步"
          "放大这一效果。")
