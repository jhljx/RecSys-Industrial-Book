r"""对应正文第 9 章 §9.8「生成式推荐讨论」中的梯度冲突(Gradient Conflict)与 PCGrad/GradNorm。

正文要点:
  * One-Model 范式让单一模型同时承担多个目标, 不同任务之间往往产生梯度冲突:
    某一目标的优化方向可能削弱另一目标的性能, 最终收敛到折中的局部最优;
  * 工业推荐同时要兼顾 CTR、Watch Time、留存、生态健康、商业化等目标, 这些目标之间
    并非完全一致, 甚至天然冲突, 且难以统一表达为单一损失函数;
  * 正文提到两类缓解手段: PCGrad(梯度投影) 与 GradNorm(梯度范数自适应配权)。

本实现(严格按两篇方法的定义式复刻):
  * 冲突度量: cos(g_i, g_j) = <g_i, g_j> / (||g_i|| ||g_j||); cos < 0 即为冲突;
  * PCGrad: 若 <g_i, g_j> < 0, 则把 g_i 投影到 g_j 的法平面上
        g_i <- g_i - (<g_i, g_j> / ||g_j||^2) g_j
    对所有 j != i(随机顺序)依次投影, 最终更新方向 g = Σ_i g_i^PC;
  * GradNorm: 以共享层最后一层权重 W 上的梯度范数 G_i = ||∇_W w_i L_i|| 为准,
        G_avg = mean_i G_i,  r_i = L~_i / mean_j L~_j (L~_i = L_i / L_i(0)),
        L_grad = Σ_i | G_i - G_avg * r_i^α |_1
    对 w_i 求梯度更新权重, 再把 w 重新归一化到 Σ w_i = T(任务数)。

本脚本构造一对"显式冲突"的多任务回归问题(两个任务的最优解方向夹角 > 90°),
对比 等权求和 / PCGrad / GradNorm 三种策略的最终各任务损失、梯度冲突比例与权重轨迹,
全部打印为数值表格。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

D_IN = 16
D_HID = 32
D_BOT = 2                         # 共享表示的瓶颈宽度: 小于任务数, 制造容量竞争
N_SAMPLE = 512
N_TASK = 3
EPOCHS = 300
ALPHA = 0.5                       # GradNorm 的不对称超参 α


class Shared(nn.Module):
    """共享 Backbone + 每任务一个独立线性头(正文所述 One-Model 多目标结构)。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(D_IN, D_HID), nn.Tanh(),
                                  nn.Linear(D_HID, D_BOT), nn.Tanh())
        self.heads = nn.ModuleList([nn.Linear(D_BOT, 1) for _ in range(N_TASK)])

    @property
    def last_shared(self):
        """GradNorm 要求在"共享层最后一层"的权重上算梯度范数。"""
        return self.body[2].weight

    def forward(self, x):
        h = self.body(x)
        return [head(h).squeeze(-1) for head in self.heads]


def make_tasks(gen):
    """三个任务各依赖一个不同的非线性特征, 而共享瓶颈只有 D_BOT=2 维。

    y_i = scale_i * tanh(3 * x·u_i), u_i 之间近似正交。三个任务需要三个不同方向的
    非线性特征, 但共享表示只有 2 维 —— 这正是正文所说"多个目标争抢同一组参数"的
    结构性冲突来源, 而非单纯的损失尺度差异。
    """
    x = torch.randn(N_SAMPLE, D_IN, generator=gen)
    us = [torch.randn(D_IN, generator=gen) for _ in range(N_TASK)]
    us = [u / u.norm() for u in us]
    scale = [1.0, 1.0, 6.0]                                   # 任务 3 损失尺度更大
    ys = [scale[i] * torch.tanh(3.0 * (x @ us[i])) for i in range(N_TASK)]
    return x, ys


def task_losses(model, x, ys):
    pred = model(x)
    return [((pred[i] - ys[i]) ** 2).mean() for i in range(N_TASK)]


def flat_grads(model, losses):
    """返回每个任务在全部共享参数上的展平梯度 (N_TASK, P)。"""
    shared = list(model.body.parameters())
    out = []
    for i, l in enumerate(losses):
        g = torch.autograd.grad(l, shared, retain_graph=True, allow_unused=True)
        out.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                              for gi, p in zip(g, shared)]))
    return torch.stack(out), shared


def set_shared_grad(shared, flat):
    off = 0
    for p in shared:
        n = p.numel()
        p.grad = flat[off:off + n].view_as(p).clone()
        off += n


def conflict_ratio(G):
    """两两任务梯度的余弦, 返回 (冲突对数 / 总对数, 余弦列表)。"""
    cos, n_bad = [], 0
    for i in range(N_TASK):
        for j in range(i + 1, N_TASK):
            c = float(torch.dot(G[i], G[j]) /
                      (G[i].norm() * G[j].norm() + 1e-12))
            cos.append(((i, j), c))
            n_bad += int(c < 0)
    return n_bad / len(cos), cos


def pcgrad(G, gen):
    """PCGrad: 对每个任务梯度, 逐个投影掉与其冲突的其他任务梯度分量。"""
    out = []
    for i in range(N_TASK):
        g = G[i].clone()
        order = torch.randperm(N_TASK, generator=gen).tolist()
        for j in order:
            if j == i:
                continue
            dot = torch.dot(g, G[j])
            if dot < 0:                                       # 仅在冲突时投影
                g = g - dot / (G[j].norm() ** 2 + 1e-12) * G[j]
        out.append(g)
    return torch.stack(out).sum(0)


def train(mode, x, ys, gen, lr=0.02):
    torch.manual_seed(42)
    model = Shared()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = torch.ones(N_TASK, requires_grad=(mode == "gradnorm"))
    opt_w = torch.optim.Adam([w], lr=0.03) if mode == "gradnorm" else None
    l0, traj, bad_hist = None, [], []
    for ep in range(EPOCHS):
        losses = task_losses(model, x, ys)
        if l0 is None:
            l0 = torch.tensor([float(l) for l in losses])
        G, shared = flat_grads(model, losses)
        bad, _ = conflict_ratio(G)
        bad_hist.append(bad)

        if mode == "sum":                                     # 等权求和
            set_shared_grad(shared, G.sum(0))
        elif mode == "pcgrad":
            set_shared_grad(shared, pcgrad(G, gen))
        else:                                                 # GradNorm
            W = model.last_shared
            gn = torch.stack([
                w[i] * torch.autograd.grad(losses[i], W, retain_graph=True)[0].norm()
                for i in range(N_TASK)])
            g_avg = gn.mean().detach()
            lt = torch.tensor([float(l) for l in losses]) / l0
            r = lt / lt.mean()
            l_grad = (gn - g_avg * r.pow(ALPHA)).abs().sum()
            opt_w.zero_grad()
            l_grad.backward(retain_graph=True)
            opt_w.step()
            with torch.no_grad():
                w.data = (w.data.clamp(min=1e-3) / w.data.clamp(min=1e-3).sum()
                          * N_TASK)                           # 重归一化 Σw = T
            set_shared_grad(shared, (w.detach().unsqueeze(1) * G).sum(0))

        # 任务头各自只受自身损失影响
        for i in range(N_TASK):
            gh = torch.autograd.grad(losses[i], list(model.heads[i].parameters()),
                                     retain_graph=(i < N_TASK - 1))
            for p, g in zip(model.heads[i].parameters(), gh):
                p.grad = g.clone()
        opt.step()
        opt.zero_grad()
        if ep % 60 == 0 or ep == EPOCHS - 1:
            traj.append((ep, [float(l) for l in losses], w.detach().clone().tolist()))
    final = [float(l) for l in task_losses(model, x, ys)]
    return final, traj, sum(bad_hist) / len(bad_hist)


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)
    x, ys = make_tasks(gen)

    print("=" * 96)
    print(f"配置: 共享 Backbone({D_IN}->{D_HID}->{D_BOT}, 瓶颈仅 {D_BOT} 维) + "
          f"{N_TASK} 个任务头, {N_SAMPLE} 条合成样本, {EPOCHS} 步 Adam")
    print(f"任务设计: 三个任务各需要一个不同方向的非线性特征 tanh(3 x·u_i), 但共享"
          f"表示只有 {D_BOT} 维;\n          任务 3 的标签尺度额外大 6 倍, 同时存在"
          "容量竞争与尺度失衡两种冲突来源")

    print()
    print("表 1  初始化时刻的任务间梯度余弦(cos<0 即为冲突)")
    print("-" * 96)
    torch.manual_seed(42)
    m0 = Shared()
    G0, _ = flat_grads(m0, task_losses(m0, x, ys))
    bad0, cos0 = conflict_ratio(G0)
    for (i, j), c in cos0:
        print(f"  cos(g_任务{i + 1}, g_任务{j + 1}) = {c:+.4f}"
              f"{'   <- 冲突' if c < 0 else ''}")
    print(f"  各任务梯度范数 = [" + ", ".join(f"{float(g.norm()):.3f}" for g in G0) + "]")
    print(f"  冲突任务对占比 = {bad0:.1%}")
    print("-" * 96)

    res = {}
    for mode in ("sum", "pcgrad", "gradnorm"):
        res[mode] = train(mode, x, ys, gen)

    print()
    print("表 2  三种多目标优化策略的最终各任务 MSE(越小越好)")
    print("-" * 96)
    name = {"sum": "等权求和 Σ L_i", "pcgrad": "PCGrad(梯度投影)",
            "gradnorm": "GradNorm(α=0.5)"}
    print(f"{'策略':<22}" + "".join(f"{'任务' + str(i + 1):>12}" for i in range(N_TASK))
          + f"{'平均':>12}{'最差任务':>12}{'训练中冲突对占比':>18}")
    for mode in ("sum", "pcgrad", "gradnorm"):
        f, _, bad = res[mode]
        print(f"{name[mode]:<22}" + "".join(f"{v:>12.4f}" for v in f)
              + f"{sum(f) / N_TASK:>12.4f}{max(f):>12.4f}{bad:>18.1%}")
    print("-" * 96)
    print("解读(本次运行): 等权求和被尺度最大的任务 3 主导, 它的损失最低而任务 1/2 被牺牲;")
    print("PCGrad 投影掉冲突分量后平均损失与最差任务损失都更低, 是三者中最均衡的;")
    print("GradNorm 为了平衡梯度范数把任务 3 的权重压到很小(见表 3), 于是任务 3 反而变差 ——")
    print("说明调权只是重新分配了冲突, 并没有消除它, 这正是正文所强调的 Trade-off。")

    print()
    print("表 3  GradNorm 的任务权重轨迹(每 60 步采样, 约束 Σ w_i = 任务数)")
    print("-" * 96)
    print(f"{'step':>6}" + "".join(f"{'w' + str(i + 1):>10}" for i in range(N_TASK))
          + "".join(f"{'L' + str(i + 1):>12}" for i in range(N_TASK)))
    for ep, ls, w in res["gradnorm"][1]:
        print(f"{ep:>6}" + "".join(f"{v:>10.4f}" for v in w)
              + "".join(f"{v:>12.4f}" for v in ls))
    print("-" * 96)
    print("正文结论: 这类方法只能缓解而非消除梯度冲突; 推荐系统的生态/商业目标往往")
    print("无法统一写成单一损失, 因此 One-Model 更可能是级联架构的补充而非替代。")


if __name__ == "__main__":
    main()
