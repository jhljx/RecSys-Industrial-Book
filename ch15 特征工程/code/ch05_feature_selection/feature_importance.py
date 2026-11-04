"""
特征重要性：从静态权重到软性选择（第15章 §15.6.1）

正文的演进脉络是：
    静态重要性 → 自适应贡献 → 特征重加权 → 细粒度动态选择
本脚本用同一个 Logistic 模型演示其中前两步：

1) 静态重要性
   - 标准化后的 |w|：最朴素的线性权重
   - 单特征 AUC：每个特征单独能解释多少点击
   - 置换重要性：打乱一列后验证集 AUC 的下降（能识别交互/冗余）

2) 自适应贡献（LHUC 式的可学习门控）
   - 给每个特征一个门控标量 g_i，梯度 dL/dg_i = Σ x_i · w · (p − y)
     也就是"这个特征在当前模型里贡献了多少残差"
   - 用 SE 式 squeeze-excitation 把 g 交给一个小网络，得到样本级权重
     （这里是 field 级共享，样本级版本在 ch04 的 FiBiNet 中）

3) 软选择 vs 硬删除
   - 门控很小时，相当于把该特征从模型里"软删除"，但不影响其他参数
   - 硬删除是直接把列拿掉，重训一次做对照

运行：python3 ch05_feature_selection/feature_importance.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _lr_utils import (  # noqa: E402
    FEATURE_NAMES, N_FEATURES, TRUE_SIGNAL, auc_score, grade, make_ctr_data,
    sigmoid, standardize, train_logreg,
)


def single_feature_auc(X: np.ndarray, y: np.ndarray) -> list[float]:
    return [auc_score(X[:, j], y) for j in range(N_FEATURES)]


def permutation_importance(w, b, Xva, yva, rounds: int = 5, seed: int = 0) -> list[float]:
    """
    置换重要性：打乱第 j 列后 AUC 掉多少。
    相比 |w|，它能反映"该特征在与其他特征共同作用下"的边际贡献。
    """
    rng = np.random.default_rng(seed)
    base = auc_score(Xva @ w + b, yva)
    out = []
    for j in range(N_FEATURES):
        drops = []
        for _ in range(rounds):
            Xp = Xva.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - auc_score(Xp @ w + b, yva))
        out.append(float(np.mean(drops)))
    return out


class GatedLogistic:
    """
    y = σ(Σ_i g_i w_i x_i + b)，g_i = softmax 前的自由参数经 sigmoid 映射到 (0,1)。

    这就是 LHUC 的形式：为每个"信息单元"（这里是特征）引入幅度参数，
    再经非线性 ξ(·) 映射为贡献权重。梯度会自然把无用的 g 压到接近 0。
    """

    def __init__(self, d: int, gate_lr: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = np.zeros(d)
        self.r = np.zeros(d)          # 门控原始参数
        self.b = 0.0
        self.gate_lr = gate_lr

    def gates(self) -> np.ndarray:
        return sigmoid(self.r)

    def train(self, Xtr, ytr, Xva, yva, steps: int = 500, lr: float = 0.5,
              l2: float = 1e-3, batch: int = 1024, seed: int = 0) -> tuple[float, list]:
        rng = np.random.default_rng(seed)
        n, d = Xtr.shape
        best = 0.0
        hist = []
        for step in range(steps):
            idx = rng.integers(0, n, batch)
            xb, yb = Xtr[idx], ytr[idx]
            g = sigmoid(self.r)
            z = (xb * g) @ self.w + self.b
            p = sigmoid(z)
            err = p - yb
            # 门控的梯度：∂L/∂g_i = Σ_b w_i x_bi (p_b − y_b) · g_i(1−g_i)
            # 特征已标准化、g≈0.5 时该梯度只有 1e-3 量级，直接乘 lr 几乎不动，
            # 所以按该 batch 的平均幅度归一化（等价于对门控用 Adam/符号更新）。
            raw_g = (xb.T @ err / batch) * self.w * g * (1 - g)
            scale = np.abs(raw_g).mean() + 1e-8
            self.w -= lr * ((xb * g).T @ err / batch + l2 * self.w)
            self.b -= lr * err.mean()
            self.r -= self.gate_lr * raw_g / scale
            if step % 25 == 0 or step == steps - 1:
                g = sigmoid(self.r)
                auc = auc_score((Xva * g) @ self.w + self.b, yva)
                best = max(best, auc)
                hist.append((step, auc))
        return best, hist

    def predict(self, X):
        g = sigmoid(self.r)
        return sigmoid((X * g) @ self.w + self.b)


def main() -> None:
    print("特征重要性：静态权重 → 置换 → 门控软选择（§15.6.1）\n")
    X, y = make_ctr_data(30_000, seed=1)
    Xva_raw, yva = make_ctr_data(6_000, seed=2)
    (Xtr, Xva), _, _ = standardize(X, Xva_raw)

    # 1) 全特征基线
    w, b, full_auc = train_logreg(Xtr, y, Xva, yva)
    print(f"全特征 Logistic：val AUC = {full_auc:.4f}\n")

    # 2) 三种静态重要性
    abs_w = np.abs(w)
    sf_auc = single_feature_auc(Xva, yva)
    perm = permutation_importance(w, b, Xva, yva)

    # 按 |w| 排序输出
    order = np.argsort(-abs_w)
    print("静态重要性（按标准化 |w| 排序）：")
    print(f"{'排名':<4}{'特征':<18}{'|w|':>8}{'单特征AUC':>12}{'置换ΔAUC':>12}  {'真值信号':<8}")
    print("-" * 68)
    for rank, j in enumerate(order, 1):
        flag = "✓" if FEATURE_NAMES[j] in TRUE_SIGNAL else "×"
        print(f"{rank:<4}{FEATURE_NAMES[j]:<18}{abs_w[j]:>8.3f}"
              f"{sf_auc[j]:>12.4f}{perm[j]:>12.4f}  {flag:<8}")

    print("\n  单特征 AUC 是'单独用'的信息量；置换 ΔAUC 是'在模型里拿掉'的损失。")
    print("  两者排序不一致时（例如 clicks_30d），说明该特征被别的特征替代了——")
    print("  这正是下一步要处理的冗余问题（§15.6.2）。")

    # 3) LHUC 式门控
    model = GatedLogistic(N_FEATURES)
    gated_auc, hist = model.train(Xtr, y, Xva, yva, steps=600)
    g = model.gates()
    print(f"\nLHUC 式门控 Logistic：val AUC = {gated_auc:.4f}"
          f"（全特征 {full_auc:.4f}）")
    print("\n学到的特征门控 g_i（越小说明模型自己在抑制它）：")
    for j in np.argsort(-g):
        bar = "█" * max(1, int(g[j] * 30))
        flag = "真信号" if FEATURE_NAMES[j] in TRUE_SIGNAL else "噪声"
        print(f"  {FEATURE_NAMES[j]:<18}{g[j]:.3f}  {bar:<30} [{flag}]")
    kept = [FEATURE_NAMES[j] for j in range(N_FEATURES) if g[j] > 0.5]
    dropped = [FEATURE_NAMES[j] for j in range(N_FEATURES) if g[j] <= 0.5]
    print(f"\n  g>0.5 保留 {len(kept)} 个：{', '.join(kept)}")
    print(f"  g≤0.5 抑制 {len(dropped)} 个：{', '.join(dropped)}")

    # 4) 硬删除对照：只保留 g>0.5 的列，重新训练
    keep_idx = [j for j in range(N_FEATURES) if g[j] > 0.5]
    w2, b2, hard_auc = train_logreg(Xtr[:, keep_idx], y, Xva[:, keep_idx], yva)
    print(f"\n硬删除后重训（仅 {len(keep_idx)} 个特征）：val AUC = {hard_auc:.4f}")
    print(f"  全特征 {full_auc:.4f} → 硬删除 {hard_auc:.4f}"
          f"（变化 {hard_auc - full_auc:+.4f}）")
    print("\n  结论：软选择不需要'特征是否保留'的二元决策，而是在训练中持续压低")
    print("  低贡献特征的幅度；硬删除则用一次重训来验证'压掉之后是否真的不掉点'。")
    print("  工业上通常先软选择观察，再分批硬删除并用线上实验兜底（§15.6.2）。")

    print("\n  补充：上表最后一列是合成数据的真值，仅用于验证方法能否识别出信号；")
    print("  真实业务里没有这一列，只能依靠离线消融 + 线上 A/B。")


if __name__ == "__main__":
    main()
