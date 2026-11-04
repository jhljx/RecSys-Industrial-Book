"""
离散化时长建模：One-Hot / 高斯 Soft Label / 自适应分位点（第23章 §23.3）

正文给出了两个层次的实验，本脚本都做：

1) 受控实验（正文 §23.3.1）：400 个时间桶、128 维随机特征、时长来自
   30 秒与 180 秒两个高斯簇的混合，比较 One-Hot 与高斯 Soft Label（σ=2）
   在 Loss、MAE、预测分布上的差异。正文强调这类实验的数据是人为构造的，
   只能用来解释模型行为，不能当作线上效果。

2) 自适应分位点（正文 §23.3.2）：先用观看时长自己的分布做粗分桶，
   再在每个桶内取分位点作为离散坐标；解码时对比
       · 概率加权平均  t̂ = Σ p_i q_i
       · 生存函数积分  t̂ = q_1 + Σ (q_{i+1} − q_i)(1 − F_i)
   后者显式使用了分位点之间的非均匀间隔。

依赖 torch（正文的模型是 MLP）；没有 torch 时 1) 会跳过，2) 仍可用。
本文实现刻意不做任何 tricks：同种子、同结构、同超参，只换 Label 与 Loss。

运行：python3 ch03_discretization/soft_label.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import make_watch_data, mae, rmse, split, standardize, table  # noqa: E402

SEED = 2026
BUK_NUMS = 400
SIGMA_TRAIN = 2.0


# --------------------------------------------------------------------------
# 1) 受控实验：One-Hot vs 高斯 Soft Label
# --------------------------------------------------------------------------
def gaussian_soft_label(duration: np.ndarray, num_bins: int, sigma: float
                        ) -> np.ndarray:
    """正文的 y_i ∝ exp(−(i−k)²/2σ²)，按行归一化。"""
    bins = np.arange(num_bins, dtype=np.float32)
    diff = bins[None, :] - duration[:, None]
    w = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    return w / w.sum(axis=1, keepdims=True)


def make_toy(n: int = 2000, dim: int = 128, seed: int = SEED):
    """
    正文 §23.3.1 的受控数据：30 秒与 180 秒两簇。

    与正文的列表相比，这里多做了两件事，都是为了让实验可读：
      · 特征不全是噪声。正文用 torch.randn 生成 128 维特征，此时任何模型
        对任何测试样本的输出都只能是训练集先验，两条 Loss 曲线只会一起
        停在「预测均值」上，比较不出监督方式的差别。这里让第 0 维与
        log(1+t) 相关、第 1 维与 cos(t/40) 相关，其余仍是噪声——既保留了
        「高维稀疏信号」的味道，又让「模型有没有学到时长」变得可观测。
      · 训练步数用到 1500（正文图里是 500 步量级）。128→256→128→400 的
        MLP 在这个数据上 500 步还没收敛，图上看到的多半是欠拟合。
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([rng.normal(30, 10, n // 2),
                             rng.normal(180, 30, n - n // 2)])
    labels = np.clip(labels, 0, BUK_NUMS - 1).astype(np.float32)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X[:, 0] = (np.log1p(labels) / np.log(BUK_NUMS) * 2 - 1
               + 0.5 * rng.normal(size=n)).astype(np.float32)
    X[:, 1] = (np.cos(labels / 40.0)
               + 0.3 * rng.normal(size=n)).astype(np.float32)
    return X, labels


def run_torch_experiment() -> None:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("  （未安装 torch，跳过受控实验）\n")
        return

    torch.manual_seed(SEED)
    X, y = make_toy()
    Xtr, Xte = X[:1600], X[1600:]
    ytr, yte = y[:1600], y[1600:]
    bin_vals = torch.arange(BUK_NUMS, dtype=torch.float32)

    # 固定一个测试样本，用来观察预测分布如何随训练变化
    fixed = torch.from_numpy(Xte[:1])
    fixed_true = float(yte[0])

    class DurationModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU())
            self.head = nn.Linear(128, BUK_NUMS)

        def forward(self, x):
            logits = self.head(self.net(x))
            probs = F.softmax(logits, dim=-1)
            pred = (probs * bin_vals).sum(-1)
            return logits, probs, pred

    def train(method: str, steps: int = 1500, batch: int = 128, lr: float = 1e-3):
        torch.manual_seed(SEED)                    # 两个模型从同一初始化出发
        model = DurationModel()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        Xt = torch.from_numpy(Xtr)
        yt = torch.from_numpy(ytr)
        rng = np.random.default_rng(SEED)
        history, snapshots = [], {}
        for step in range(1, steps + 1):
            idx = rng.integers(0, len(Xt), batch)
            logits, _, _ = model(Xt[idx])
            if method == "onehot":
                loss = F.cross_entropy(logits, yt[idx].long().clamp(0, BUK_NUMS - 1))
            else:
                soft = torch.from_numpy(
                    gaussian_soft_label(yt[idx].numpy(), BUK_NUMS, SIGMA_TRAIN))
                loss = -(soft * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == 1:
                with torch.no_grad():
                    _, _, pred = model(torch.from_numpy(Xte))
                    history.append((step, float(loss), mae(yte, pred.numpy())))
            if step in (200, 800, 1500):
                with torch.no_grad():
                    _, probs, pred = model(fixed)
                snapshots[step] = (float(pred[0]), probs[0].numpy())
        return history, snapshots

    print("① 受控实验：400 桶、128 维特征、30s 与 180s 两簇混合")
    print("   同一初始化、同一结构、同一优化器，只换监督方式\n")

    hist_oh, snap_oh = train("onehot")
    hist_gs, snap_gs = train("gaussian")

    # --- Label 形状 ---
    print("② 真实时长 120 秒时两种 Label 的形状（只看附近 7 个桶）：")
    t120 = np.array([120.0])
    oh = np.zeros(BUK_NUMS)
    oh[120] = 1.0
    gs = gaussian_soft_label(t120, BUK_NUMS, SIGMA_TRAIN)[0]
    rows = []
    for b in range(117, 124):
        rows.append([b, f"{oh[b]:.4f}", f"{gs[b]:.4f}"])
    table(["桶编号", "One-Hot", f"高斯 σ={SIGMA_TRAIN}"], rows)
    nz = int((gs > 1e-3).sum())
    print(f"\n  高斯 Soft Label 在 |i−k|≤3 内有 {nz} 个非零项，"
          f"中心权重 {gs[120]:.3f}，相邻桶 {gs[119]:.3f}。")
    print("  σ 越小越接近 One-Hot；σ=2 时概率质量集中在 ±4 个桶内，")
    print("  这正是「把距离信息写进监督信号」的直观含义。\n")

    # --- 训练曲线 ---
    print("③ 训练过程中的 Loss 与测试集 MAE：")
    rows = []
    keys = {s: i for i, (s, _, _) in enumerate(hist_oh)}
    for step in (200, 400, 800, 1200, 1500):
        i = keys[step]
        rows.append([step, f"{hist_oh[i][1]:.3f}", f"{hist_oh[i][2]:.2f}",
                     f"{hist_gs[i][1]:.3f}", f"{hist_gs[i][2]:.2f}"])
    table(["step", "One-Hot Loss", "One-Hot MAE", "Soft Loss", "Soft MAE"], rows)
    oh_final = hist_oh[keys[1500]][2]
    gs_final = hist_gs[keys[1500]][2]
    base = float(np.abs(yte - ytr.mean()).mean())
    print(f"\n  最终 MAE：One-Hot {oh_final:.2f} 秒，Soft Label {gs_final:.2f} 秒，")
    print(f"  「一律预测训练均值」的基线是 {base:.2f} 秒。")
    print("  两种 Loss 的数值不可直接比较（一个是硬标签交叉熵、一个是软标签交叉熵），")
    print("  但 MAE 在同一个尺度上。Soft Label 通常更低：它的监督信号本身就把")
    print("  邻近桶的概率一起抬起来，模型更容易学到「这两簇大致在哪」。\n")

    # --- 固定样本的预测分布 ---
    print(f"④ 固定样本（真实时长 {fixed_true:.1f} 秒）在各训练阶段的预测：")
    rows = []
    for step in (200, 800, 1500):
        p_oh, d_oh = snap_oh[step]
        p_gs, d_gs = snap_gs[step]
        # 预测分布的集中度：概率最大的桶 + 分布熵
        e_oh = float(-(d_oh * np.log(d_oh + 1e-12)).sum())
        e_gs = float(-(d_gs * np.log(d_gs + 1e-12)).sum())
        rows.append([step, f"{p_oh:.1f}", f"{e_oh:.2f}", f"{p_gs:.1f}",
                     f"{e_gs:.2f}"])
    table(["step", "One-Hot 预测", "One-Hot 熵", "Soft 预测", "Soft 熵"], rows)
    print("\n  One-Hot 的预测分布熵随训练下降得更快，概率向少数桶集中；")
    print("  Soft Label 的分布更平缓，因为监督信号本身就要求「邻近桶也有概率」。")
    print("  这个样本的特征向量没有落进任何一簇的中心，两种方法都只能给出")
    print("  一个折中的预测——受控实验要看的正是「分布的平滑程度」，不是这一个点。\n")


# --------------------------------------------------------------------------
# 2) 自适应分位点 + 两种解码
# --------------------------------------------------------------------------
def run_adaptive() -> None:
    print("⑤ 自适应分位点：粗分桶 × 桶内分位点（§23.3.2）")
    data = make_watch_data(n=20000, dim=16, seed=SEED)
    X, w, d = data["x"], data["w"].astype(np.float64), data["d"]
    idx = np.arange(len(X))
    tr, te = split(idx, len(X), 0.8)
    Xtr, Xte, _, _ = standardize(X[tr], X[te])
    wtr, wte = w[tr], w[te]

    # 粗分桶：按物理时长分 6 段（业务上可解释，且每段样本量不必相等）
    coarse_edges = np.array([15, 25, 40, 60, 100])
    def coarse_of(dv):
        return np.searchsorted(coarse_edges, dv, side="right")

    ctr = coarse_of(d[tr])
    K = 12                                     # 每桶 12 个分位点
    # 分位点用 0.05~0.95：下限避开零观看那 12% 的样本质量，否则前几个
    # 分位点会挤在 0 上，桶内坐标退化，Soft Label 只剩一两个有效位置。
    quantiles = np.zeros((len(coarse_edges) + 1, K))
    for b in range(len(coarse_edges) + 1):
        m = ctr == b
        q = np.quantile(wtr[m], np.linspace(0.05, 0.95, K))
        # 有些桶的观看时长被物理时长封顶（例如 [12,15) 秒的视频最多看 15 秒），
        # 分位点可能重复；加一个极小的递增偏移，保证坐标严格单调，解码不退化。
        q = np.maximum.accumulate(q) + np.arange(K) * 1e-3
        quantiles[b] = q
    print(f"  粗分桶边界 {coarse_edges.tolist()}，每桶 {K} 个分位点；")
    print(f"  最短桶的分位点：{[round(float(q), 2) for q in quantiles[0]]}")
    print(f"  最长桶的分位点：{[round(float(q), 1) for q in quantiles[-1]]}")
    print("  短桶的末尾几个坐标几乎一样，是因为该桶的视频最长只有 15 秒，")
    print("  观看时长被物理时长封顶；加 1e-3 的递增偏移只是为了坐标严格单调。")
    print("  同样 12 个坐标，短桶覆盖约 0~20 秒，长桶覆盖几十到几百秒——")
    print("  分辨率跟着数据密度走，这就是「自适应」的含义。\n")

    # 简化模型：线性 softmax，每个粗桶一个 12 类头
    n_b, Kd = len(coarse_edges) + 1, K
    Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    Xte_b = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1)
    W = np.zeros((Xb.shape[1], n_b * Kd))
    rng = np.random.default_rng(SEED)
    lr = 0.05
    for step in range(600):
        logits = Xb @ W
        # 每个样本只在自己所属的桶内做 softmax
        logits3 = logits.reshape(len(Xb), n_b, Kd)
        z = np.take_along_axis(logits3, ctr[:, None, None], axis=1)[:, 0, :]
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        # 软标签：高斯核作用在该桶的分位点上
        q = quantiles[ctr]                      # [n, K]，每个样本自己的桶内分位点
        diff = q - wtr[:, None]
        logits_soft = -(diff ** 2) / (2.0 * 4.0 ** 2)
        # 先减最大值再取 exp，避免长桶里远离真实值的分位点下溢成 0
        logits_soft -= logits_soft.max(axis=1, keepdims=True)
        soft = np.exp(logits_soft)
        soft = soft / soft.sum(axis=1, keepdims=True)
        grad = (p - soft) / len(Xb)             # ∂CE/∂z
        gW = np.zeros_like(W)
        for i in range(len(Xb)):
            gW[:, ctr[i] * Kd:(ctr[i] + 1) * Kd] += np.outer(Xb[i], grad[i])
        W -= lr * gW

    cte = coarse_of(d[te])
    logits3 = (Xte_b @ W).reshape(len(Xte), n_b, Kd)
    z = np.take_along_axis(logits3, cte[:, None, None], axis=1)[:, 0, :]
    z -= z.max(axis=1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    q = quantiles[cte]

    # 解码 1：概率加权平均
    pred_mean = (P * q).sum(axis=1)
    # 解码 2：生存函数积分 t = q_1 + Σ (q_{i+1}−q_i)(1−F_i)
    F = np.cumsum(P, axis=1)
    dq = np.diff(q, axis=1)
    pred_surv = q[:, 0] + (dq * (1.0 - F[:, :-1])).sum(axis=1)

    rows = [
        ["概率加权平均", f"{mae(wte, pred_mean):.3f}", f"{rmse(wte, pred_mean):.3f}"],
        ["生存函数积分", f"{mae(wte, pred_surv):.3f}", f"{rmse(wte, pred_surv):.3f}"],
        ["全预测均值", f"{mae(wte, np.full_like(wte, wtr.mean())):.3f}",
         f"{rmse(wte, np.full_like(wte, wtr.mean())):.3f}"],
    ]
    table(["解码方式", "MAE（秒）", "RMSE（秒）"], rows)
    print("\n  两种解码的结果相同，这不是巧合：对离散分布做分部求和，")
    print("      q_1 + Σ_{i=1}^{K−1}(q_{i+1}−q_i)(1−F_i) = Σ_i p_i q_i")
    print("  是恒等式（q_i 单调、p 归一），所以它并不能「比加权平均更准」。")
    print("  生存积分的价值在于把 E[T]=∫S(t)dt 这个定义显现出来：它告诉你")
    print("  「第 K 个坐标之后还有多少时长」被忽略了——两个公式都只积到 q_K，")
    print("  长尾越重，这个截断误差越大。要修正就得给尾部单独加一个桶")
    print("  （比如 q_K 到某个上限），而不是换一种求和顺序。\n")

    print("⑥ 一句话：离散化的演进是「点 → 分布 → 自适应分布」。")
    print("   One-Hot 只告诉模型「落在哪个桶」，Soft Label 告诉它「离真实值多远」，")
    print("   自适应分位点则让坐标本身跟着数据密度走。")


def main() -> None:
    print("离散化时长建模（§23.3）\n")
    run_torch_experiment()
    run_adaptive()


if __name__ == "__main__":
    main()
