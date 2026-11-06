"""
逻辑回归精排模型（第12章 §12.3.1）

LR 通过 Sigmoid(w·x + b) 预估点击概率，交叉熵损失 + L2 正则化。
支持人工构造交叉特征（Cross Feature），演示 PCOC 校准计算。

本书依据：《第十二章-精排模型.pdf》§12.3.1，PDF 第 4–5 页。
依赖：PyTorch。运行：python3 lr.py
"""

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim


def _auc(pairs: list[tuple[float, int]]) -> float:
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


# ── 特征工程：手工构造交叉特征 ────────────────────────────────────────────────

def cross_product(x: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    """按指定下标对生成二阶交叉特征（两个 0/1 特征的逻辑 AND）。"""
    extras = [(x[:, i] * x[:, j]).unsqueeze(1) for i, j in pairs]
    return torch.cat([x] + extras, dim=1)


# ── 模型 ──────────────────────────────────────────────────────────────────────

class LogisticRegression(nn.Module):
    """
    广义线性模型：z = w·[x; cross(x)] + b，p = σ(z)。

    正文公式：
        z(x) = w_0 + Σ w_i x_i
        p̂(x) = σ(z(x))
    损失：
        L_LR = −Σ [y log p̂ + (1−y) log(1−p̂)] + λ‖w‖²
    """

    def __init__(self, in_dim: int, reg: float = 1e-3):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.reg = reg
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)          # logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def fit(self, x: torch.Tensor, y: torch.Tensor,
            epochs: int = 60, lr: float = 0.1) -> None:
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)
        for _ in range(epochs):
            opt.zero_grad()
            logit = self.forward(x)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, y.float())
            loss.backward()
            opt.step()

    def rank(self, x: torch.Tensor, top_k: int) -> list[tuple[int, float]]:
        """返回 [(item_idx, score), ...] Top-K 排序结果。"""
        with torch.no_grad():
            scores = self.predict_proba(x).tolist()
        ranked = sorted(enumerate(scores), key=lambda t: -t[1])
        return ranked[:top_k]


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 400, seed: int = 42):
    """
    生成二分类排序数据。
    特征 0–3：用户属性（性别/年龄段/地域/活跃度）
    特征 4–7：物品属性（类别/时长段/新鲜度/质量分）
    标签由真实权重 + 噪声决定，确保 CTR ≈ 0.25。
    """
    torch.manual_seed(seed)
    x = torch.zeros(n, 8)
    # 用户属性
    x[:, 0] = (torch.rand(n) < 0.5).float()          # gender
    x[:, 1] = (torch.rand(n) < 0.4).float()          # young
    x[:, 2] = (torch.rand(n) < 0.6).float()          # city
    x[:, 3] = (torch.rand(n) < 0.3).float()          # active
    # 物品属性
    x[:, 4] = (torch.rand(n) < 0.3).float()          # cat_A
    x[:, 5] = (torch.rand(n) < 0.5).float()          # mid_len
    x[:, 6] = (torch.rand(n) < 0.4).float()          # fresh
    x[:, 7] = (torch.rand(n) < 0.35).float()         # high_quality

    true_w = torch.tensor([0.3, 0.5, 0.2, 0.6, 0.4, 0.3, 0.5, 0.7])
    logit = x @ true_w - 1.4 + 0.3 * torch.randn(n)
    y = (torch.sigmoid(logit) > 0.5).long()
    return x, y


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    x_raw, y = make_data(400)
    split = 300
    x_tr, y_tr = x_raw[:split], y[:split]
    x_te, y_te = x_raw[split:], y[split:]

    # ── 实验 1：纯 LR（无交叉特征）────────────────────────────────────────────
    lr_plain = LogisticRegression(in_dim=8)
    lr_plain.fit(x_tr, y_tr, epochs=60)

    with torch.no_grad():
        p_te = lr_plain.predict_proba(x_te)

    auc_plain = _auc(list(zip(p_te.tolist(), y_te.tolist())))
    pcoc_plain = p_te.sum().item() / max(y_te.sum().item(), 1)
    print(f"LR（无交叉特征） AUC={auc_plain:.4f}  PCOC={pcoc_plain:.3f}")

    # ── 实验 2：LR + 交叉特征（gender×cat_A, active×high_quality）──────────────
    # 正文说明：φ_k(x) = Π x_i^{c_{ki}}，c_{ki}∈{0,1}
    cross_pairs = [(0, 4), (3, 7)]    # gender×cat_A，active×high_quality
    x_tr_c = cross_product(x_tr, cross_pairs)   # dim=10
    x_te_c = cross_product(x_te, cross_pairs)

    lr_cross = LogisticRegression(in_dim=10)
    lr_cross.fit(x_tr_c, y_tr, epochs=60)

    with torch.no_grad():
        p_te_c = lr_cross.predict_proba(x_te_c)

    auc_cross = _auc(list(zip(p_te_c.tolist(), y_te.tolist())))
    pcoc_cross = p_te_c.sum().item() / max(y_te.sum().item(), 1)
    print(f"LR（有交叉特征） AUC={auc_cross:.4f}  PCOC={pcoc_cross:.3f}")

    # ── 精排推理演示 ───────────────────────────────────────────────────────────
    # 取测试集前 20 条作为「一次请求的候选集合」
    candidates = x_te_c[:20]
    top5 = lr_cross.rank(candidates, top_k=5)
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, score) in enumerate(top5, 1):
        label = y_te[idx].item()
        print(f"  #{rank_i}  item={idx:2d}  score={score:.4f}  label={label}")

    # ── PCOC 说明 ──────────────────────────────────────────────────────────────
    # 正文公式：PCOC = Σp̂_i / Σy_i；接近 1 表示概率校准良好
    actual_ctr = y_te.float().mean().item()
    pred_ctr   = p_te_c.mean().item()
    print(f"\n实际 CTR={actual_ctr:.4f}  预测平均 CTR={pred_ctr:.4f}  "
          f"PCOC={pcoc_cross:.3f}")
    print("（PCOC 接近 1 说明 LR 概率校准良好）")
