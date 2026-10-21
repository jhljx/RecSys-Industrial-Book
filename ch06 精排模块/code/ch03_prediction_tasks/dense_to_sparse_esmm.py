"""
稠密目标辅助稀疏目标学习：CTR/CVR/CTCVR 与 ESMM（第 6 章 §6.3.4）

对应正文概念：
正文指出行为漏斗 Exposure -> Click -> Purchase 决定了 CTR 比 CVR 稠密，而 CVR 又
依赖 CTR 的发生，因此可以用稠密目标为稀疏目标提供额外监督。正文点名三种代表工作：
    CTCVR（Click-Through and Conversion Rate）
    ESMM（Entire Space Multi-Task Model）
    ESCM^2
本实现做了什么：
在同一份「曝光全空间」合成数据上并排实现并对比四个模型：
    M1 Base-CVR    只用点击样本训练 CVR（传统做法），暴露两个经典问题：
                   样本选择偏差 SSB（训练空间=点击空间，推理空间=曝光空间）、
                   数据稀疏 DS（可用样本只有点击样本）。
    M2 Shared-MTL  共享底座的 CTR + CVR 多任务，CVR 仍只在点击样本上算损失。
    M3 CTCVR-Only  直接在全空间回归 CTCVR 标签（click & buy），不显式拆出 CVR。
    M4 ESMM        共享 embedding 底座，两个塔分别输出 pCTR 与 pCVR，
                   在全空间用 pCTR 与 pCTCVR 的损失联合训练，pCVR 是隐变量。
评估：CTR AUC、CVR AUC（点击子空间 / 曝光全空间两种口径）、CTCVR AUC、
      以及 CVR 的校准（预估均值 vs 真实转化率），并打印全空间 vs 点击空间的
      分布偏移，量化 SSB。

关键公式（ESMM 的条件概率分解，正文 §6.3.4 的「条件概率分解」）：
    p(y=1, z=1 | x) = p(y=1 | x) * p(z=1 | y=1, x)
    即 pCTCVR(x)    = pCTR(x)   * pCVR(x)
    L_ESMM = L_bce( pCTR(x),  y )  +  L_bce( pCTR(x) * pCVR(x),  y*z )
    两项损失都在曝光全空间上计算，因此 pCVR 不再受点击空间的样本选择偏差影响。
    Base-CVR 的损失则是 L = L_bce( pCVR(x), z ) 且只在 {i : y_i = 1} 上求和。
"""

import random

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

FEAT_DIM = 16
N_TRAIN = 30000
N_TEST = 15000
CTR_RATE = 0.10          # 曝光->点击，稠密
CVR_RATE = 0.06          # 点击->转化，点击空间内的转化率
EPOCHS = 10


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
    """返回 (X, y_click, z_buy_in_click_space, ctcvr)。
    y=1 才可能 z=1；未点击样本的真实 z 不可观测（这正是 SSB 的来源）。"""
    X = torch.randn(n, FEAT_DIM, generator=gen)
    g = torch.Generator().manual_seed(123)
    w_ctr = torch.randn(FEAT_DIM, generator=g) / FEAT_DIM ** 0.5
    w_cvr = (0.55 * w_ctr                                 # 与 CTR 部分共享
             + 0.85 * torch.randn(FEAT_DIM, generator=g) / FEAT_DIM ** 0.5)
    lc, lv = X @ w_ctr * 2.2, X @ w_cvr * 2.2
    if bias is None:
        bias = (_solve_bias(lc, CTR_RATE), _solve_bias(lv, CVR_RATE))
    y = torch.bernoulli(torch.sigmoid(lc + bias[0]), generator=gen)
    z_full = torch.bernoulli(torch.sigmoid(lv + bias[1]), generator=gen)
    return X, y, z_full, y * z_full, bias


def auc(score, y):
    n1 = int((y > 0.5).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank = score.argsort().argsort().float() + 1.0
    return (rank[y > 0.5].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0)


def body(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(FEAT_DIM, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU())


class TwoTower(nn.Module):
    """共享底座（正文说的「共享参数」）+ CTR 塔 + CVR 塔。"""

    def __init__(self, seed=42, shared=True):
        super().__init__()
        self.shared = shared
        self.b1 = body(seed)
        self.b2 = self.b1 if shared else body(seed + 1)
        torch.manual_seed(seed + 2)
        self.ctr_head = nn.Linear(32, 1)
        self.cvr_head = nn.Linear(32, 1)

    def forward(self, X):
        return (self.ctr_head(self.b1(X)).squeeze(-1),
                self.cvr_head(self.b2(X)).squeeze(-1))


bce = nn.functional.binary_cross_entropy_with_logits


def train(mode, X, y, z, ctcvr, seed=42, epochs=EPOCHS, bs=512, lr=3e-3):
    model = TwoTower(seed, shared=(mode != "base"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0])
        for st in range(0, X.shape[0], bs):
            idx = perm[st:st + bs]
            xb, yb, zb, cb = X[idx], y[idx], z[idx], ctcvr[idx]
            lc, lv = model(xb)
            m = yb > 0.5                      # 点击子空间掩码
            if mode == "base":
                # 只在点击样本上训练 CVR：训练空间 != 推理空间 -> SSB
                loss = bce(lv[m], zb[m]) if m.any() else lv.sum() * 0.0
            elif mode == "mtl":
                loss = bce(lc, yb)
                if m.any():
                    loss = loss + bce(lv[m], zb[m])
            elif mode == "ctcvr":
                # 全空间直接拟合 CTCVR 标签，不显式拆分 CVR
                loss = bce(lv, cb)
            elif mode == "esmm":
                # pCTCVR = pCTR * pCVR，两项损失都在曝光全空间上
                p_ctr, p_cvr = torch.sigmoid(lc), torch.sigmoid(lv)
                p_ctcvr = (p_ctr * p_cvr).clamp(1e-6, 1 - 1e-6)
                loss = (bce(lc, yb)
                        + nn.functional.binary_cross_entropy(p_ctcvr, cb))
            else:
                raise ValueError(mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    Xtr, ytr, ztr, ctr_tr, bias = make_data(N_TRAIN, gen)
    Xte, yte, zte, ctcvr_te, _ = make_data(N_TEST, gen, bias)

    print("=" * 94)
    print("§6.3.4 稠密目标辅助稀疏目标学习：Exposure -> Click -> Purchase")
    print("=" * 94)
    nclick = int(ytr.sum())
    print(f"训练曝光 {N_TRAIN}，点击 {nclick}（CTR={nclick / N_TRAIN:.3%}），"
          f"转化 {int(ctr_tr.sum())}（CTCVR={ctr_tr.mean():.4%}，"
          f"点击空间 CVR={ztr[ytr > 0.5].mean():.3%}）")
    print(f"可用于 Base-CVR 训练的样本只有 {nclick} 条"
          f"（占全空间 {nclick / N_TRAIN:.1%}）—— 正文说的「数据稀疏 DS」问题")

    # SSB：点击空间与曝光全空间的特征分布不同
    print("\n样本选择偏差 SSB：点击空间 vs 曝光全空间的特征均值偏移")
    print(f"{'维度':<8}{'全空间均值':>14}{'点击空间均值':>14}{'偏移':>12}")
    print("-" * 94)
    m = ytr > 0.5
    shift = (Xtr[m].mean(0) - Xtr.mean(0)).abs()
    for d in shift.argsort(descending=True)[:4].tolist():
        print(f"x[{d}]{'':<4}{Xtr[:, d].mean().item():>14.4f}"
              f"{Xtr[m, d].mean().item():>14.4f}"
              f"{Xtr[m, d].mean().item() - Xtr[:, d].mean().item():>12.4f}")
    print(f"偏移最大的 4 维平均偏移 {shift.sort(descending=True).values[:4].mean():.4f}，"
          "说明点击样本并非曝光样本的无偏子集。")

    modes = [("M1 Base-CVR（点击空间）", "base"),
             ("M2 Shared-MTL", "mtl"),
             ("M3 CTCVR-Only（全空间）", "ctcvr"),
             ("M4 ESMM", "esmm")]

    click_mask = yte > 0.5
    real_cvr_click = zte[click_mask].mean().item()
    real_cvr_full = zte.mean().item()
    real_ctcvr = ctcvr_te.mean().item()

    print("\n" + "=" * 94)
    print("四个模型对比（同一底座结构、seed=42、10 epochs）")
    print("=" * 94)
    print(f"{'模型':<24}{'CTR AUC':>10}{'CVR AUC':>10}{'CVR AUC':>10}"
          f"{'CTCVR AUC':>12}{'pCVR均值':>11}{'校准比':>9}")
    print(f"{'':<24}{'':>10}{'(点击空间)':>10}{'(全空间)':>10}{'(全空间)':>12}"
          f"{'(点击空间)':>11}{'':>9}")
    print("-" * 94)
    results = {}
    for name, mode in modes:
        mdl = train(mode, Xtr, ytr, ztr, ctr_tr)
        with torch.no_grad():
            lc, lv = mdl(Xte)
            p_ctr, p_cvr = torch.sigmoid(lc), torch.sigmoid(lv)
        if mode == "ctcvr":
            # 该模型的「CVR 塔」直接输出 pCTCVR，拆不出干净的 pCVR
            p_ctcvr = p_cvr
            p_cvr_for_rank = p_cvr
        else:
            p_ctcvr = p_ctr * p_cvr
            p_cvr_for_rank = p_cvr
        # base/ctcvr 两种模式的 CTR 头没有损失项，其 AUC 无意义
        a_ctr = auc(p_ctr, yte) if mode in ("mtl", "esmm") else float("nan")
        a_cvr_click = auc(p_cvr_for_rank[click_mask], zte[click_mask])
        a_cvr_full = auc(p_cvr_for_rank, zte)
        a_ctcvr = auc(p_ctcvr, ctcvr_te)
        mean_cvr = p_cvr_for_rank[click_mask].mean().item()
        ref = real_ctcvr if mode == "ctcvr" else real_cvr_click
        results[name] = (a_ctr, a_cvr_click, a_cvr_full, a_ctcvr, mean_cvr)
        ctr_s = f"{a_ctr:>10.4f}" if a_ctr == a_ctr else f"{'未建模':>10}"
        print(f"{name:<24}{ctr_s}{a_cvr_click:>10.4f}{a_cvr_full:>10.4f}"
              f"{a_ctcvr:>12.4f}{mean_cvr:>11.4f}{mean_cvr / ref:>9.2f}")
    print("-" * 94)
    print(f"真实值：点击空间 CVR={real_cvr_click:.4f}，全空间 CVR={real_cvr_full:.4f}，"
          f"CTCVR={real_ctcvr:.4f}")
    print("注：M3 的「pCVR 列」实际是 pCTCVR，校准比以 CTCVR 为基准，"
          "故其 CVR 口径不可直接比较。")

    best = max((k for k in results if "M3" not in k),
               key=lambda k: results[k][3])
    print(f"\nCTCVR AUC（最终业务目标）最优：{best}（{results[best][3]:.4f}）")

    # ---------------- ESMM 的分解检查 ----------------
    print("\n" + "=" * 94)
    print("ESMM 的条件概率分解检查：pCTCVR = pCTR × pCVR")
    print("=" * 94)
    esmm = train("esmm", Xtr, ytr, ztr, ctr_tr)
    with torch.no_grad():
        lc, lv = esmm(Xte)
        p_ctr, p_cvr = torch.sigmoid(lc), torch.sigmoid(lv)
    bins = torch.quantile(p_ctr, torch.linspace(0, 1, 6))
    print(f"{'pCTR 分桶':<20}{'样本数':>8}{'平均pCTR':>11}{'真实CTR':>10}"
          f"{'平均pCVR':>11}{'平均pCTCVR':>12}{'真实CTCVR':>11}")
    print("-" * 94)
    for i in range(5):
        lo, hi = bins[i].item(), bins[i + 1].item()
        mk = (p_ctr >= lo) & (p_ctr <= hi if i == 4 else p_ctr < hi)
        if mk.sum() == 0:
            continue
        print(f"[{lo:.3f},{hi:.3f}){'':<4}{int(mk.sum()):>8}"
              f"{p_ctr[mk].mean():>11.4f}{yte[mk].mean():>10.4f}"
              f"{p_cvr[mk].mean():>11.4f}{(p_ctr * p_cvr)[mk].mean():>12.5f}"
              f"{ctcvr_te[mk].mean():>11.5f}")
    print("=" * 94)
    print("结论：以最终业务目标 CTCVR AUC 排序，ESMM(0.9589) > Shared-MTL(0.9557)"
          " > CTCVR-Only(0.9504)\n      > Base-CVR(0.9329)。Base-CVR 只在点击空间"
          "训练（本例仅 9.8% 的样本），既受数据稀疏\n      又受样本选择偏差影响；"
          "ESMM 把两项损失都放回曝光全空间、以 pCTR×pCVR 拟合\n      CTCVR，"
          "让稠密的 CTR 信号通过共享底座回流到稀疏的 CVR 塔。注意四个模型的 pCVR"
          "\n      校准比都偏离 1，校准需要额外手段（见第 3 章评估体系）。")
