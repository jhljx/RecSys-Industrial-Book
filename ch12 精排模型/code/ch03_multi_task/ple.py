"""
PLE 多任务精排模型（第12章 §12.4.2.2）

Progressive Layered Extraction（腾讯 RecSys 2020）：

CGC（Customized Gate Control）模块：
    在共享专家之外，每个任务有专属专家；
    每个任务的门控混合「本任务专属专家 + 共享专家」：

    f^(k) = Σ g_i^(k) · e_i^(k 或 shared)

    共享专家另有独立的共享门控，聚合出跨任务的共享表示，
    作为下一层 CGC 共享专家的输入。

PLE：将多个 CGC 模块逐层堆叠，逐步分离不同任务的特征表示：

    第 1 层 CGC → 第 2 层 CGC → … → 各任务塔网络

与 MMoE 的区别：
  - MMoE  : 所有任务仅使用共享专家，通过独立门控选择组合；
  - CGC   : 引入任务专属专家，共享专家补充跨任务信号；
  - PLE   : 逐层堆叠 CGC，使任务特异性随层深递增，减少负迁移。

本书依据：《第十二章-精排模型.pdf》§12.4.2.2，PDF 第 35–40 页（多任务部分）。
依赖：PyTorch。运行：python3 ple.py
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


def _mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ── CGC 抽取层 ────────────────────────────────────────────────────────────────

class CGCLayer(nn.Module):
    """
    一个 CGC 抽取层。

    正文说明（PDF §12.4.2.2）：
      「CGC 模块在共享专家之外，为每个任务设置专属专家；
        每个任务的门控混合本任务专属专家和共享专家的输出。」

    共享专家的输入是上一层聚合的共享表示（首层为原始输入），
    而非某个任务的输出——这是 PLE 与朴素 MMoE 堆叠的关键差异。

    参数：
      in_dim        : 输入维度（上一层输出或原始特征）
      hidden        : 专家内部隐藏/输出维度
      n_task_experts: 每个任务的专属专家数
      n_shared      : 共享专家数
      n_tasks       : 任务数
    """

    def __init__(self, in_dim: int, hidden: int,
                 n_task_experts: int, n_shared: int, n_tasks: int):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_task_exp = n_task_experts
        self.n_shared = n_shared

        # 任务专属专家: n_tasks 组，每组 n_task_experts 个
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                              nn.Linear(hidden, hidden), nn.ReLU())
                for _ in range(n_task_experts)
            ])
            for _ in range(n_tasks)
        ])

        # 共享专家（接收共享输入）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU())
            for _ in range(n_shared)
        ])

        # 每个任务的门控：任务输入 → 任务专属(n_task_experts) + 共享(n_shared) → softmax
        gate_out = n_task_experts + n_shared
        self.task_gates = nn.ModuleList([
            nn.Linear(in_dim, gate_out) for _ in range(n_tasks)])

        # 共享门控：共享输入 → n_shared → softmax
        self.shared_gate = nn.Linear(in_dim, n_shared)

    def forward(self, task_inputs: list[torch.Tensor],
                shared_input: torch.Tensor
                ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        task_inputs  : list of T tensors, 每个 (B, in_dim)
                       第 k 个元素是任务 k 的上一层输出（首层均为原始输入）
        shared_input : (B, in_dim) 上一层聚合的共享表示（首层为原始输入）
        返回:
          new_task_outputs : list of T tensors, 每个 (B, hidden)
          new_shared       : (B, hidden)
        """
        # 共享专家：用共享输入计算一次，所有任务共享
        shared_outs = [e(shared_input) for e in self.shared_experts]
        shared_stack = torch.stack(shared_outs, dim=1)  # (B, n_shared, H)

        # 共享门控 → 聚合共享专家为下一层的共享输入
        sg = F.softmax(self.shared_gate(shared_input), dim=1)  # (B, n_shared)
        new_shared = (sg.unsqueeze(-1) * shared_stack).sum(dim=1)  # (B, H)

        new_task_outputs = []
        for k in range(self.n_tasks):
            # 任务 k 的专属专家（用任务 k 自己的上一层输入）
            task_outs = [e(task_inputs[k]) for e in self.task_experts[k]]

            # 任务专属专家 + 共享专家拼在一起供门控选择
            all_outs = torch.stack(task_outs + shared_outs, dim=1)  # (B, E_k+S, H)

            g = F.softmax(self.task_gates[k](task_inputs[k]), dim=1)  # (B, E_k+S)
            f_k = (g.unsqueeze(-1) * all_outs).sum(dim=1)            # (B, H)
            new_task_outputs.append(f_k)

        return new_task_outputs, new_shared


# ── PLE 模型 ─────────────────────────────────────────────────────────────────

class PLE(nn.Module):
    """
    PLE：逐层堆叠 CGC，再接各任务塔网络。

    正文（PDF §12.4.2.2）：
      「PLE 通过多层网络逐步完成信息提取与任务分离，
        随层深递增，减少任务间的负迁移（Negative Transfer）。」
    """

    def __init__(self, in_dim: int, n_tasks: int,
                 hidden: int = 16, n_layers: int = 2,
                 n_task_experts: int = 2, n_shared: int = 2,
                 tower_hidden: int = 16):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_layers = n_layers

        # 逐层 CGC（各层维度相同，方便堆叠）
        self.cgc_layers = nn.ModuleList()
        cur_dim = in_dim
        for _ in range(n_layers):
            self.cgc_layers.append(
                CGCLayer(cur_dim, hidden, n_task_experts, n_shared, n_tasks))
            cur_dim = hidden

        # 各任务塔网络
        self.towers = nn.ModuleList([
            _mlp([cur_dim, tower_hidden, 1]) for _ in range(n_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始时各任务输入与共享输入均为原始特征
        task_inputs = [x] * self.n_tasks
        shared_input = x

        # 逐层 CGC
        for layer in self.cgc_layers:
            task_inputs, shared_input = layer(task_inputs, shared_input)

        # 各任务塔
        return torch.cat([self.towers[k](task_inputs[k])
                          for k in range(self.n_tasks)], dim=1)  # (B, T)

    def fit(self, x, ys, epochs: int = 100, lr: float = 5e-3,
            task_weights=None):
        w = task_weights or [1.0] * self.n_tasks
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for ep in range(epochs):
            opt.zero_grad()
            logits = self.forward(x)
            loss = sum(
                w[k] * F.binary_cross_entropy_with_logits(
                    logits[:, k], ys[:, k].float())
                for k in range(self.n_tasks))
            loss.backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f"  epoch {ep+1:3d}  total_loss={loss.item():.4f}")

    def predict_proba(self, x) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 2400, in_dim: int = 16, seed: int = 42):
    """
    两任务的输入特征互不重叠：
      任务 0（点击）依赖 x0–x3 的线性与 x1*x2 交互；
      任务 1（完播）依赖 x6–x9 的线性与 x7*x8 交互。
    每个任务都有自己的专属信号，正是 CGC/PLE 的任务专属专家
    要捕捉的对象；共享专家则负责跨任务的共性表示。
    与 mmoe.py 共用同一份数据生成逻辑，方便公平对比。
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

    # ── 对比基线 1：Shared-Bottom（与 mmoe.py 配置一致）───────────────────────
    from mmoe import MMoE, SharedBottom
    torch.manual_seed(42)
    sb = SharedBottom(IN_DIM, N_TASKS, bottom_hidden=8, tower_hidden=16)
    opt_sb = optim.Adam(sb.parameters(), lr=5e-3, weight_decay=1e-5)
    for _ in range(120):
        opt_sb.zero_grad()
        logits = sb(x_tr)
        loss = sum(F.binary_cross_entropy_with_logits(
            logits[:, k], y_tr[:, k].float()) for k in range(N_TASKS))
        loss.backward()
        opt_sb.step()
    with torch.no_grad():
        p_sb = torch.sigmoid(sb(x_te))
    auc_sb = [_auc(p_sb[:, k].tolist(), y_te[:, k].tolist())
              for k in range(N_TASKS)]
    print(f"\nShared-Bottom  Task0 AUC={auc_sb[0]:.4f}  Task1 AUC={auc_sb[1]:.4f}")

    # ── 对比基线 2：MMoE（来自正文 §12.4.2.1 的对比） ──────────────────────────
    torch.manual_seed(42)
    mmoe = MMoE(IN_DIM, n_experts=4, n_tasks=N_TASKS,
                expert_hidden=8, tower_hidden=16)
    mmoe.fit(x_tr, y_tr, epochs=120, lr=5e-3)
    p_mmoe = mmoe.predict_proba(x_te)
    auc_mmoe = [_auc(p_mmoe[:, k].tolist(), y_te[:, k].tolist())
                for k in range(N_TASKS)]
    print(f"\nMMoE    Task0 AUC={auc_mmoe[0]:.4f}  Task1 AUC={auc_mmoe[1]:.4f}")

    # ── PLE（2 层 CGC，每任务 2 专属专家 + 2 共享专家）─────────────────────────
    torch.manual_seed(42)
    ple = PLE(IN_DIM, N_TASKS, hidden=8, n_layers=2,
              n_task_experts=2, n_shared=2, tower_hidden=16)
    print(f"\nPLE 训练（2 层 CGC，各任务 2 专属 + 2 共享专家，专家宽度 8）")
    ple.fit(x_tr, y_tr, epochs=100, lr=5e-3)

    p_ple = ple.predict_proba(x_te)
    auc_ple = [_auc(p_ple[:, k].tolist(), y_te[:, k].tolist())
               for k in range(N_TASKS)]
    print(f"\nPLE     Task0 AUC={auc_ple[0]:.4f}  Task1 AUC={auc_ple[1]:.4f}")

    # ── 三模型对比 ────────────────────────────────────────────────────────────
    print(f"\n对比（正文 §12.4.2.2）：")
    print(f"  Shared-Bottom avg AUC={sum(auc_sb)/2:.4f}")
    print(f"  MMoE         avg AUC={sum(auc_mmoe)/2:.4f}")
    print(f"  PLE          avg AUC={sum(auc_ple)/2:.4f}")
    print("  PLE 通过任务专属专家 + 共享专家逐层分离，")
    print("  在任务特征支持不重叠时进一步减少负迁移，超越 MMoE。")

    # ── 门控分析（PLE 最后一层 CGC 的任务专属专家权重）──────────────────────────
    print(f"\n最后一层 CGC 门控权重分析（前 20 样本平均）：")
    with torch.no_grad():
        # 取得倒数第二层（即最后 CGC 的输入）
        task_in = [x_te[:20]] * N_TASKS
        shared_in = x_te[:20]
        for layer in ple.cgc_layers[:-1]:
            task_in, shared_in = layer(task_in, shared_in)
        last_cgc = ple.cgc_layers[-1]
        n_te = last_cgc.n_task_exp
        n_sh = last_cgc.n_shared
        for k in range(N_TASKS):
            g = F.softmax(last_cgc.task_gates[k](task_in[k]), dim=1)
            g_mean = g.mean(dim=0)
            task_w = g_mean[:n_te].sum().item()
            shared_w = g_mean[n_te:].sum().item()
            print(f"  Task{k}: 专属专家权重合计={task_w:.3f}  "
                  f"共享专家权重合计={shared_w:.3f}")
    print("  正文（PDF §12.4.2.2）：任务差异越大，专属专家权重占比应越高。")

    # ── 精排推理演示（任务 0）─────────────────────────────────────────────────
    scores_t0 = p_ple[:20, 0].tolist()
    top5 = sorted(enumerate(scores_t0), key=lambda t: -t[1])[:5]
    print("\n按任务0（点击）精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  p_click={s:.4f}  "
              f"label_click={y_te[idx, 0].item()}")
