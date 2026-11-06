"""
MMoE 多任务精排模型（第12章 §12.4.2.1）

Multi-gate Mixture-of-Experts（Google KDD 2018）：

    e_i = f_i(x),            i=1…n   # n 个共享专家（MLP）
    g^(k)(x) = softmax(W_g^(k) x),   # 第 k 个任务的门控向量 ∈ ℝ^n
    f^(k)(x) = Σ_i g_i^(k)(x) e_i   # 任务 k 的加权专家输出
    ŷ_k = h_k(f^(k)(x))              # 任务 k 的独立塔网络

与 Shared-Bottom / OMoE 对比：
  - Shared-Bottom : f^(k)(x) = f(x)              所有任务固定共享同一底层
  - OMoE          : f^(k)(x) = Σ g_i(x) e_i      共享门控，各任务使用相同权重
  - MMoE          : f^(k)(x) = Σ g_i^(k)(x) e_i  每任务独立门控，自适应专家选择

本书依据：《第十二章-精排模型.pdf》§12.4.2.1，PDF 第 30–35 页（多任务部分）。
依赖：PyTorch。运行：python3 mmoe.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _auc(scores, labels) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


# ── Shared-Bottom（对比基线）────────────────────────────────────────────────

class SharedBottom(nn.Module):
    """所有任务固定共享同一底层 MLP，再各自过塔网络。"""

    def __init__(self, in_dim: int, n_tasks: int,
                 bottom_hidden: int = 32, tower_hidden: int = 16):
        super().__init__()
        self.bottom = nn.Sequential(
            nn.Linear(in_dim, bottom_hidden), nn.ReLU(),
            nn.Linear(bottom_hidden, bottom_hidden), nn.ReLU(),
        )
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(bottom_hidden, tower_hidden), nn.ReLU(),
                          nn.Linear(tower_hidden, 1))
            for _ in range(n_tasks)
        ])

    def forward(self, x):
        h = self.bottom(x)
        return torch.cat([t(h) for t in self.towers], dim=1)  # (B, T)


# ── MMoE 模型 ─────────────────────────────────────────────────────────────────

class MMoE(nn.Module):
    """
    MMoE：n 个共享专家 + T 个任务独立门控 + T 个塔网络。

    正文（PDF §12.4.2.1）核心：
      「不同任务可以根据输入特征，自适应地组合不同专家的输出……
        即使输入特征相同，各任务可以生成不同的专家权重，
        从而形成不同的特征组合。」
    """

    def __init__(self, in_dim: int, n_experts: int, n_tasks: int,
                 expert_hidden: int = 32, tower_hidden: int = 16):
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks

        # 共享专家（结构相同，参数独立）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_hidden), nn.ReLU(),
                nn.Linear(expert_hidden, expert_hidden), nn.ReLU(),
            )
            for _ in range(n_experts)
        ])

        # 每个任务独立的门控线性层：x → ℝ^n_experts → softmax
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts) for _ in range(n_tasks)])

        # 每个任务独立的塔网络
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, tower_hidden), nn.ReLU(),
                nn.Linear(tower_hidden, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_dim)
        返回 logits : (B, T)
        """
        # 所有专家并行计算
        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=1)          # (B, n, expert_hidden)

        logits = []
        for k in range(self.n_tasks):
            g = F.softmax(self.gates[k](x), dim=1)        # (B, n)
            # 加权融合专家输出
            f_k = (g.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, expert_hidden)
            logits.append(self.towers[k](f_k))            # (B, 1)

        return torch.cat(logits, dim=1)                    # (B, T)

    def fit(self, x, ys, epochs: int = 40, lr: float = 1e-2,
            task_weights=None):
        """
        ys : (B, T) 多任务标签
        task_weights : 各任务损失权重（默认均等）
        """
        w = task_weights or [1.0] * self.n_tasks
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for ep in range(epochs):
            opt.zero_grad()
            logits = self.forward(x)                      # (B, T)
            loss = sum(
                w[k] * F.binary_cross_entropy_with_logits(
                    logits[:, k], ys[:, k].float())
                for k in range(self.n_tasks))
            loss.backward()
            opt.step()
            if (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:3d}  total_loss={loss.item():.4f}")

    def predict_proba(self, x) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))         # (B, T)


# ── 合成数据：两任务存在显式冲突 ─────────────────────────────────────────────

def make_data(n: int = 2400, in_dim: int = 16, seed: int = 42):
    """
    两任务的输入特征互不重叠：
      任务 0（点击）依赖 x0–x3 的线性与 x1*x2 交互；
      任务 1（完播）依赖 x6–x9 的线性与 x7*x8 交互。
    模拟正文所述「不同任务关注重点存在明显差异」的场景——
    每个任务都有自己的专属信号，Shared-Bottom 被迫用同一底层表示
    同时拟合两组不相关的特征，容易出现负迁移。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    x = torch.randn(n, in_dim)

    l0 = 2.2 * x[:, 0] + 1.8 * (x[:, 1] * x[:, 2]) + 1.0 * x[:, 3] - 0.3
    l1 = 2.2 * x[:, 6] + 1.8 * (x[:, 7] * x[:, 8]) + 1.0 * x[:, 9] - 0.3

    y0 = (torch.sigmoid(l0 + 0.35 * torch.randn(n)) > 0.5).long()
    y1 = (torch.sigmoid(l1 + 0.35 * torch.randn(n)) > 0.5).long()

    return x, torch.stack([y0, y1], dim=1)


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    IN_DIM, N_TASKS = 16, 2
    x, y = make_data(2400, IN_DIM)
    split = 1920
    x_tr, y_tr = x[:split], y[:split]
    x_te, y_te = x[split:], y[split:]

    # ── Shared-Bottom 基线（容量受限：bottom_hidden=8）───────────────────────
    # 正文（PDF §12.4.2.1）指出 Shared-Bottom 的局限：
    # 「所有任务必须使用相同的底层表示。当不同任务的关注重点存在明显差异时，
    #   共享表示可能难以同时满足所有任务的需求。」
    # 用较小的底层容量来凸显这一局限。
    torch.manual_seed(42)
    sb = SharedBottom(IN_DIM, N_TASKS, bottom_hidden=8, tower_hidden=16)
    opt_sb = optim.Adam(sb.parameters(), lr=5e-3, weight_decay=1e-5)
    for ep in range(120):
        opt_sb.zero_grad()
        logits = sb(x_tr)
        loss = sum(F.binary_cross_entropy_with_logits(
            logits[:, k], y_tr[:, k].float()) for k in range(N_TASKS))
        loss.backward()
        opt_sb.step()
    with torch.no_grad():
        p_sb = torch.sigmoid(sb(x_te))
    auc_sb = [_auc(p_sb[:, k].tolist(), y_te[:, k].tolist()) for k in range(N_TASKS)]
    print(f"Shared-Bottom（h=8）  Task0 AUC={auc_sb[0]:.4f}  Task1 AUC={auc_sb[1]:.4f}")

    # ── MMoE ─────────────────────────────────────────────────────────────────
    # 4 个共享专家，每个专家容量与 Shared-Bottom 底层一致（h=8），
    # 但通过独立门控让两个任务选择不同的专家组合。
    torch.manual_seed(42)
    mmoe = MMoE(IN_DIM, n_experts=4, n_tasks=N_TASKS,
                expert_hidden=8, tower_hidden=16)
    print(f"\nMMoE 训练（4 个共享专家 h=8，2 个独立门控）")
    mmoe.fit(x_tr, y_tr, epochs=120, lr=5e-3)

    p_mmoe = mmoe.predict_proba(x_te)
    auc_mmoe = [_auc(p_mmoe[:, k].tolist(), y_te[:, k].tolist())
                for k in range(N_TASKS)]
    print(f"\nMMoE（4 专家 h=8）    Task0 AUC={auc_mmoe[0]:.4f}  Task1 AUC={auc_mmoe[1]:.4f}")

    # ── 门控权重分析 ──────────────────────────────────────────────────────────
    print(f"\n门控权重（前 20 样本平均，正文：不同任务应学到不同专家偏好）：")
    with torch.no_grad():
        for k in range(N_TASKS):
            g = F.softmax(mmoe.gates[k](x_te[:20]), dim=1)
            g_mean = g.mean(dim=0)
            print(f"  Task{k} 专家权重: " +
                  " ".join(f"E{i}={g_mean[i].item():.3f}"
                            for i in range(4)))

    # ── 与 Shared-Bottom 对比 ────────────────────────────────────────────────
    print(f"\n对比（正文 §12.4.2.1）：")
    print(f"  Shared-Bottom avg AUC={sum(auc_sb)/2:.4f}  "
          f"MMoE avg AUC={sum(auc_mmoe)/2:.4f}")
    print("  MMoE 通过任务独立门控，让冲突任务选择不同专家组合，")
    print("  缓解了 Shared-Bottom 中「所有任务被迫使用相同底层表示」的局限。")

    # ── 精排推理演示（任务 0：点击）────────────────────────────────────────────
    scores_t0 = p_mmoe[:20, 0].tolist()
    top5 = sorted(enumerate(scores_t0), key=lambda t: -t[1])[:5]
    print("\n按任务0（点击）精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  p_click={s:.4f}  "
              f"label_click={y_te[idx, 0].item()}")
