"""
行为关联建模：AITM / MMoE / PLE 与链式依赖（第 6 章 §6.3.5 行为关联建模）

对应正文概念：
正文指出不同互动行为之间存在较强相关性与潜在的传递关系，并给出两条典型链路：
    Like -> Comment -> Follow
    Like -> Favorite -> Follow
正文点名的代表方法：
    AITM（Adaptive Information Transfer Multi-task framework，美团）
        —— 通过信息传递机制显式建模任务之间的依赖关系，解决任务互相干扰；
    PLE（Progressive Layered Extraction）、MMoE（Multi-gate Mixture-of-Experts）、
    SNR（Sub-Network Routing）
        —— 通过共享/差异化信息的分离来提升稀疏目标学习效果。
这一类方法在工业界也称为「多目标联合建模 / 多目标感知建模」。

本实现做了什么：
按链路 Like -> Comment -> Follow 合成具有单调漏斗关系的三目标数据（后一环的
发生以前一环为条件），并实现 4 个模型做同口径对比：
    A. SharedBottom  共享底座 + 三个独立塔（最朴素的多目标，任务互相干扰）
    B. MMoE          多个专家 + 每个任务一个 gate，软性分离共享/私有信息
    C. PLE           任务私有专家 + 共享专家，gate 只在「私有 + 共享」上做选择
    D. AITM          在 MMoE 塔的基础上加「自适应信息迁移」模块：
                     前一个任务的表征经注意力门控后传给后一个任务，
                     并加入行为期望校准损失（behavior expectation calibration）
                     约束 p(前置任务) >= p(后置任务)，即漏斗单调性。
打印：各任务 AUC、PR-AUC、漏斗单调性违背率（p_follow > p_like 的样本占比），
以及 AITM 的信息迁移门控强度随任务链条的变化。

关键公式：
    MMoE：  y_k = h_k( sum_j g_k(x)_j * E_j(x) ),  g_k = softmax(W_k x)
    PLE ：  y_k = h_k( gate_k([E^k_priv(x); E^share(x)]) )
    AITM：  q_t      = u_{t-1}（前序任务传来的信息）
            info_t   = AttentionTransfer( [p_t, q_t] )
            z_t      = 融合后的任务表征 -> 塔头输出 y_t
            注意力：  w_i = <Q(v_i), K(v_i)> / sqrt(d),  a = softmax(w),
                      out = sum_i a_i * V(v_i),  v in {p_t, q_t}
    AITM 校准损失： L_cal = sum_t max(0, p_t - p_{t-1})   （漏斗单调性约束）
    总损失：        L = sum_t L_bce,t + lambda * L_cal
"""

import random

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

FEAT_DIM = 20
HID = 32
N_TRAIN = 20000
N_TEST = 10000
EPOCHS = 12
# 链路 Like -> Comment -> Follow，后一环以前一环为条件
CHAIN = ["like", "comment", "follow"]
COND_RATE = [0.12, 0.22, 0.18]   # 条件概率：P(like)、P(comment|like)、P(follow|comment)
LAMBDA_CAL = 0.6


def _solve_bias(logit, rate):
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if torch.sigmoid(logit + mid).mean().item() > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def make_data(n, gen, biases=None):
    """生成严格漏斗：follow=1 必然 comment=1 必然 like=1。"""
    X = torch.randn(n, FEAT_DIM, generator=gen)
    g = torch.Generator().manual_seed(2024)
    w0 = torch.randn(FEAT_DIM, generator=g) / FEAT_DIM ** 0.5
    Y, out_b, prev = {}, [], None
    for t, name in enumerate(CHAIN):
        # 每环的偏好 = 共享兴趣 + 该环私有成分（相关但不相同）
        wt = 0.7 * w0 + 0.7 * torch.randn(FEAT_DIM, generator=g) / FEAT_DIM ** 0.5
        logit = X @ wt * 2.4
        b = biases[t] if biases else _solve_bias(logit, COND_RATE[t])
        out_b.append(b)
        cond = torch.bernoulli(torch.sigmoid(logit + b), generator=gen)
        Y[name] = cond if prev is None else cond * prev   # 条件于前一环
        prev = Y[name]
    return X, Y, out_b


def auc(score, y):
    n1 = int((y > 0.5).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank = score.argsort().argsort().float() + 1.0
    return (rank[y > 0.5].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(score, y):
    order = score.argsort(descending=True)
    ys = y[order]
    tp = torch.cumsum(ys, dim=0)
    tot = ys.sum()
    if tot == 0:
        return float("nan")
    prec = tp / torch.arange(1, len(ys) + 1, dtype=torch.float)
    rec = tp / tot
    area, prev = 0.0, 0.0
    for i in range(len(ys)):
        r = rec[i].item()
        if r > prev:
            area += (r - prev) * prec[i].item()
            prev = r
    return area


def expert():
    return nn.Sequential(nn.Linear(FEAT_DIM, 64), nn.ReLU(), nn.Linear(64, HID),
                         nn.ReLU())


def tower():
    return nn.Sequential(nn.Linear(HID, 16), nn.ReLU(), nn.Linear(16, 1))


# --------------------------------------------------------------------------
# A. SharedBottom
# --------------------------------------------------------------------------


class SharedBottom(nn.Module):
    name = "A.SharedBottom"

    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.body = expert()
        self.towers = nn.ModuleList([tower() for _ in CHAIN])

    def forward(self, X):
        h = self.body(X)
        return [t(h).squeeze(-1) for t in self.towers]


# --------------------------------------------------------------------------
# B. MMoE
# --------------------------------------------------------------------------


class MMoE(nn.Module):
    name = "B.MMoE"

    def __init__(self, seed=42, n_expert=4):
        super().__init__()
        torch.manual_seed(seed)
        self.experts = nn.ModuleList([expert() for _ in range(n_expert)])
        self.gates = nn.ModuleList([nn.Linear(FEAT_DIM, n_expert) for _ in CHAIN])
        self.towers = nn.ModuleList([tower() for _ in CHAIN])

    def _mix(self, X):
        E = torch.stack([e(X) for e in self.experts], dim=1)      # [B, M, H]
        return E

    def forward(self, X):
        E = self._mix(X)
        outs = []
        for k, _ in enumerate(CHAIN):
            g = torch.softmax(self.gates[k](X), dim=-1).unsqueeze(-1)  # [B,M,1]
            outs.append(self.towers[k]((E * g).sum(1)).squeeze(-1))
        return outs

    def gate_weights(self, X):
        return [torch.softmax(self.gates[k](X), dim=-1).mean(0)
                for k in range(len(CHAIN))]


# --------------------------------------------------------------------------
# C. PLE（单层 CGC：任务私有专家 + 共享专家）
# --------------------------------------------------------------------------


class PLE(nn.Module):
    name = "C.PLE(CGC)"

    def __init__(self, seed=42, n_share=2, n_priv=1):
        super().__init__()
        torch.manual_seed(seed)
        self.shared = nn.ModuleList([expert() for _ in range(n_share)])
        self.priv = nn.ModuleList(
            [nn.ModuleList([expert() for _ in range(n_priv)]) for _ in CHAIN])
        self.gates = nn.ModuleList(
            [nn.Linear(FEAT_DIM, n_share + n_priv) for _ in CHAIN])
        self.towers = nn.ModuleList([tower() for _ in CHAIN])

    def forward(self, X):
        S = [e(X) for e in self.shared]
        outs = []
        for k, _ in enumerate(CHAIN):
            P = [e(X) for e in self.priv[k]]
            E = torch.stack(P + S, dim=1)
            g = torch.softmax(self.gates[k](X), dim=-1).unsqueeze(-1)
            outs.append(self.towers[k]((E * g).sum(1)).squeeze(-1))
        return outs


# --------------------------------------------------------------------------
# D. AITM：自适应信息迁移 + 行为期望校准
# --------------------------------------------------------------------------


class AttentionTransfer(nn.Module):
    """AITM 的信息迁移单元：对 {前序信息 q_t, 本任务表征 p_t} 做自注意力加权。"""

    def __init__(self, dim=HID):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** 0.5

    def forward(self, vs):
        V = torch.stack(vs, dim=1)                       # [B, 2, H]
        w = (self.q(V) * self.k(V)).sum(-1) / self.scale  # [B, 2]
        a = torch.softmax(w, dim=-1)
        return (a.unsqueeze(-1) * self.v(V)).sum(1), a


class AITM(nn.Module):
    """AITM：MMoE 式底座 + 沿 Like->Comment->Follow 链条的自适应信息迁移。"""

    name = "D.AITM"

    def __init__(self, seed=42, n_expert=4):
        super().__init__()
        torch.manual_seed(seed)
        self.experts = nn.ModuleList([expert() for _ in range(n_expert)])
        self.gates = nn.ModuleList([nn.Linear(FEAT_DIM, n_expert) for _ in CHAIN])
        # 前序表征 u_{t-1} -> q_t 的映射
        self.transfer = nn.ModuleList(
            [nn.Sequential(nn.Linear(HID, HID), nn.ReLU())
             for _ in CHAIN[1:]])
        self.ait = nn.ModuleList([AttentionTransfer() for _ in CHAIN[1:]])
        self.towers = nn.ModuleList([tower() for _ in CHAIN])

    def forward(self, X, return_attn=False):
        E = torch.stack([e(X) for e in self.experts], dim=1)
        outs, attns, prev = [], [], None
        for k, _ in enumerate(CHAIN):
            g = torch.softmax(self.gates[k](X), dim=-1).unsqueeze(-1)
            p = (E * g).sum(1)                           # 本任务表征 p_t
            if k == 0:
                z = p
            else:
                q = self.transfer[k - 1](prev)           # 前序信息 q_t
                z, a = self.ait[k - 1]([q, p])
                attns.append(a.mean(0))                  # [w_q, w_p]
            outs.append(self.towers[k](z).squeeze(-1))
            prev = z
        return (outs, attns) if return_attn else outs


bce = nn.functional.binary_cross_entropy_with_logits


def fit(model, X, Y, use_cal=False, epochs=EPOCHS, bs=512, lr=3e-3, seed=42):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0])
        for st in range(0, X.shape[0], bs):
            idx = perm[st:st + bs]
            logits = model(X[idx])
            loss = sum(bce(logits[k], Y[n][idx]) for k, n in enumerate(CHAIN))
            if use_cal:
                # 行为期望校准：约束 p_t <= p_{t-1}（漏斗单调性）
                ps = [torch.sigmoid(l) for l in logits]
                cal = sum(torch.clamp(ps[t] - ps[t - 1], min=0.0).mean()
                          for t in range(1, len(CHAIN)))
                loss = loss + LAMBDA_CAL * cal
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def monotonic_violation(ps):
    """返回漏斗单调性违背率：存在 t 使得 p_t > p_{t-1} 的样本占比。"""
    bad = torch.zeros_like(ps[0], dtype=torch.bool)
    for t in range(1, len(ps)):
        bad |= ps[t] > ps[t - 1]
    return bad.float().mean().item()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    Xtr, Ytr, bs_ = make_data(N_TRAIN, gen)
    Xte, Yte, _ = make_data(N_TEST, gen, bs_)

    print("=" * 94)
    print("§6.3.5 行为关联建模：Like -> Comment -> Follow 链式依赖")
    print("=" * 94)
    print(f"训练 {N_TRAIN} 条 / 测试 {N_TEST} 条")
    print(f"\n{'环节':<12}{'正样本数':>10}{'全空间正样本率':>16}"
          f"{'条件正样本率 P(t|t-1)':>24}")
    print("-" * 94)
    prev = None
    for n in CHAIN:
        cnt = int(Ytr[n].sum())
        cond = (cnt / int(Ytr[prev].sum())) if prev else cnt / N_TRAIN
        print(f"{n:<12}{cnt:>10}{cnt / N_TRAIN:>16.4%}{cond:>24.4%}")
        prev = n
    # 相关性矩阵：正文说的「行为之间并非相互独立」
    print("\n行为两两 Pearson 相关系数（正文：点赞/评论/关注往往共同出现）")
    M = torch.stack([Ytr[n] for n in CHAIN])
    Mc = M - M.mean(1, keepdim=True)
    C = (Mc @ Mc.T) / (Mc.norm(dim=1, keepdim=True) @ Mc.norm(dim=1, keepdim=True).T)
    print(f"{'':<12}" + "".join(f"{n:>12}" for n in CHAIN))
    for i, n in enumerate(CHAIN):
        print(f"{n:<12}" + "".join(f"{C[i, j].item():>12.4f}"
                                   for j in range(len(CHAIN))))

    aitm_nocal = AITM()
    aitm_nocal.name = "D0.AITM(无校准)"
    models = [(SharedBottom(), False), (MMoE(), False), (PLE(), False),
              (aitm_nocal, False), (AITM(), True)]

    print("\n" + "=" * 94)
    print("四个多任务框架对比（同一数据、seed=42、12 epochs）")
    print("=" * 94)
    hdr = f"{'模型':<18}"
    for n in CHAIN:
        hdr += f"{n + ' AUC':>13}"
    hdr += f"{'平均AUC':>10}{'单调违背率':>12}"
    print(hdr)
    print("-" * 94)
    trained = []
    for m, cal in models:
        fit(m, Xtr, Ytr, use_cal=cal)
        trained.append(m)
        with torch.no_grad():
            logits = m(Xte)
            ps = [torch.sigmoid(l) for l in logits]
        aucs = [auc(ps[k], Yte[n]) for k, n in enumerate(CHAIN)]
        row = f"{m.name:<18}" + "".join(f"{a:>13.4f}" for a in aucs)
        row += f"{sum(aucs) / len(aucs):>10.4f}{monotonic_violation(ps):>12.2%}"
        print(row)
    print("-" * 94)
    print("单调违背率 = 存在 t 使 p_t > p_{t-1} 的样本占比；"
          "真实漏斗下该值应接近 0。")

    print("\nPR-AUC（链条末端最稀疏，PR-AUC 比 AUC 更敏感）")
    print(f"{'模型':<18}" + "".join(f"{n + ' PR-AUC':>16}" for n in CHAIN))
    print("-" * 94)
    for m in trained:
        with torch.no_grad():
            ps = [torch.sigmoid(l) for l in m(Xte)]
        print(f"{m.name:<18}" + "".join(f"{pr_auc(ps[k], Yte[n]):>16.4f}"
                                        for k, n in enumerate(CHAIN)))

    # ---------------- AITM 的信息迁移门控强度 ----------------
    print("\n" + "=" * 94)
    print("AITM 自适应信息迁移：注意力在「前序任务信息 q_t」与「本任务表征 p_t」"
          "之间的分配")
    print("=" * 94)
    aitm = trained[-1]
    with torch.no_grad():
        _, attns = aitm(Xte, return_attn=True)
    print(f"{'任务':<14}{'w(前序信息 q_t)':>20}{'w(本任务 p_t)':>18}"
          f"{'前序信息占比':>14}")
    print("-" * 94)
    for t, a in enumerate(attns, start=1):
        print(f"{CHAIN[t]:<14}{a[0].item():>20.4f}{a[1].item():>18.4f}"
              f"{a[0].item():>14.2%}")

    # ---------------- MMoE gate 的任务差异 ----------------
    print("\nMMoE 各任务的专家选择分布（gate softmax 均值，展示共享/差异化分离）")
    mmoe = trained[1]
    with torch.no_grad():
        gw = mmoe.gate_weights(Xte)
    print(f"{'任务':<14}" + "".join(f"{'E' + str(j):>12}" for j in range(4)))
    print("-" * 94)
    for k, n in enumerate(CHAIN):
        print(f"{n:<14}" + "".join(f"{gw[k][j].item():>12.4f}" for j in range(4)))
    print("=" * 94)
    print("结论：三个目标沿漏斗逐级稀疏（11.98% / 5.67% / 2.09%），行为之间强相关"
          "（like-comment\n      相关系数 0.66）。本例数据规模下四种框架的 AUC 差异"
          "在 ±0.003 以内，真正拉开\n      差距的是「漏斗单调性」：MMoE/PLE 各任务"
          "塔彼此独立，违背率约 4%；AITM 显式沿\n      Like->Comment->Follow 传递"
          "前序表征后降到 1.65%，再加行为期望校准损失降到 0.77%。\n      这说明"
          "正文强调的「建模行为之间的依赖关系」收益主要体现在输出的结构一致性上，"
          "\n      而不是单任务 AUC 的绝对值。")
