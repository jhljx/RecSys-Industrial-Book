"""
因子分解机（FM）精排模型（第12章 §12.3.3）

FM 通过为每个特征学习低维隐向量，用内积计算二阶特征交互：
    s(x) = w_0 + Σ w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j

利用「和平方 − 平方和」恒等式将计算复杂度从 O(d²k) 降至 O(d·k)：
    Σ_{i<j} <v_i,v_j> x_i x_j = ½ Σ_f [(Σ_i v_{if} x_i)² − Σ_i v²_{if} x²_i]

本书依据：《第十二章-精排模型.pdf》§12.3.3，PDF 第 6–8 页。
依赖：PyTorch。运行：python3 fm.py
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = scores[labels == 1].tolist()
    neg = scores[labels == 0].tolist()
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


# ── FM 模型 ───────────────────────────────────────────────────────────────────

class FM(nn.Module):
    """
    因子分解机。

    正文公式（PDF §12.3.3）：
        s(x) = w_0 + Σ w_i x_i
               + ½ Σ_f [(Σ_i v_{if} x_i)² − Σ_i v²_{if} x²_i]

    参数：
        n_fields : 特征字段数（每字段用 embedding 表示）
        n_values : 每字段的取值数（不同字段统一编号后的总数）
        emb_dim  : 隐向量维度 k
    """

    def __init__(self, n_values: int, emb_dim: int = 8, reg: float = 1e-4):
        super().__init__()
        # 一阶权重
        self.w = nn.Embedding(n_values, 1)
        # 二阶隐向量 v_i ∈ ℝ^k
        self.v = nn.Embedding(n_values, emb_dim)
        self.w0 = nn.Parameter(torch.zeros(1))
        self.reg = reg
        nn.init.zeros_(self.w.weight)
        nn.init.normal_(self.v.weight, std=0.01)

    def forward(self, feat_ids: torch.Tensor,
                feat_vals: torch.Tensor) -> torch.Tensor:
        """
        feat_ids  : (B, F)  每字段的特征 ID
        feat_vals : (B, F)  对应的特征值（连续特征直接传值，离散特征传 1.0）
        返回 logit (B,)。

        二阶项高效计算（正文等式推导）：
          sum_ij = ½ (‖Σ_i v_i x_i‖² − Σ_i ‖v_i‖² x²_i)
        """
        # 一阶：Σ w_i x_i
        first = (self.w(feat_ids).squeeze(-1) * feat_vals).sum(dim=1)  # (B,)

        # 二阶：利用「和平方 − 平方和」恒等式，复杂度 O(B·F·k)
        emb = self.v(feat_ids)                       # (B, F, k)
        xv = emb * feat_vals.unsqueeze(-1)           # (B, F, k)  v_i * x_i
        sum_sq = xv.sum(dim=1).pow(2).sum(dim=1)     # (B,) ‖Σ v_i x_i‖²
        sq_sum = xv.pow(2).sum(dim=(1, 2))           # (B,) Σ ‖v_i x_i‖²
        second = 0.5 * (sum_sq - sq_sum)             # (B,)

        return self.w0 + first + second

    def predict_proba(self, feat_ids: torch.Tensor,
                      feat_vals: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(feat_ids, feat_vals))

    def fit(self, feat_ids: torch.Tensor, feat_vals: torch.Tensor,
            y: torch.Tensor, epochs: int = 60, lr: float = 0.01) -> None:
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)
        for ep in range(epochs):
            opt.zero_grad()
            logit = self.forward(feat_ids, feat_vals)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y.float())
            loss.backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                with torch.no_grad():
                    auc = _auc(torch.sigmoid(logit), y)
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}  train-AUC={auc:.4f}")

    def rank(self, feat_ids: torch.Tensor, feat_vals: torch.Tensor,
             top_k: int) -> list[tuple[int, float]]:
        scores = self.predict_proba(feat_ids, feat_vals).tolist()
        return sorted(enumerate(scores), key=lambda t: -t[1])[:top_k]


# ── 合成数据 ───────────────────────────────────────────────────────────────────

def make_fm_data(n: int = 400, n_fields: int = 6, vals_per_field: int = 5,
                 seed: int = 42):
    """
    生成多字段离散特征数据。
    每个字段有 vals_per_field 个取值，独热编码后总维度 = n_fields * vals_per_field。
    feat_ids  : (n, n_fields)  每字段取值的全局 ID
    feat_vals : (n, n_fields)  全为 1.0（离散 one-hot 等价）
    """
    torch.manual_seed(seed)
    random.seed(seed)
    # 全局 ID = field_idx * vals_per_field + local_val
    feat_ids = torch.zeros(n, n_fields, dtype=torch.long)
    for f in range(n_fields):
        feat_ids[:, f] = (torch.randint(0, vals_per_field, (n,))
                          + f * vals_per_field)
    feat_vals = torch.ones(n, n_fields)

    # 真实信号：字段 0 & 字段 3 的交互对标签有强影响
    true_main = torch.zeros(n)
    for f in range(n_fields):
        true_main += (feat_ids[:, f] % vals_per_field).float() * (0.3 + 0.1 * f)
    cross = ((feat_ids[:, 0] % vals_per_field) *
             (feat_ids[:, 3] % vals_per_field)).float() * 0.5
    logit = true_main + cross - true_main.mean()
    y = (torch.sigmoid(logit + 0.4 * torch.randn(n)) > 0.5).long()
    return feat_ids, feat_vals, y


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    N_FIELDS = 6
    VALS_PER_FIELD = 5
    N_VALUES = N_FIELDS * VALS_PER_FIELD   # 全局特征值空间大小

    feat_ids, feat_vals, y = make_fm_data(400, N_FIELDS, VALS_PER_FIELD)
    split = 300
    ids_tr, vals_tr, y_tr = feat_ids[:split], feat_vals[:split], y[:split]
    ids_te, vals_te, y_te = feat_ids[split:], feat_vals[split:], y[split:]

    # ── FM 训练 ──────────────────────────────────────────────────────────────
    print("=== FM 训练（6 字段 × 5 取值，k=8）===")
    fm = FM(n_values=N_VALUES, emb_dim=8)
    fm.fit(ids_tr, vals_tr, y_tr, epochs=60)

    p_te = fm.predict_proba(ids_te, vals_te)
    auc_te = _auc(p_te, y_te)
    print(f"\n测试集 AUC={auc_te:.4f}")

    # ── 精排推理演示 ──────────────────────────────────────────────────────────
    top5 = fm.rank(ids_te[:20], vals_te[:20], top_k=5)
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, score) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={score:.4f}  label={y_te[idx].item()}")

    # ── 高效计算验证 ──────────────────────────────────────────────────────────
    # 正文：「和平方 − 平方和」恒等式 vs 暴力枚举，结果应完全一致
    with torch.no_grad():
        emb = fm.v(ids_tr[:5])               # (5, F, k)
        xv = emb * vals_tr[:5].unsqueeze(-1) # (5, F, k)
        fast = 0.5 * (xv.sum(1).pow(2).sum(1) - xv.pow(2).sum((1, 2)))

        brute = torch.zeros(5)
        F = N_FIELDS
        for s in range(5):
            for i in range(F):
                for j in range(i + 1, F):
                    brute[s] += (xv[s, i] * xv[s, j]).sum()
        print(f"\n高效计算 vs 暴力枚举（最大误差={((fast - brute).abs().max().item()):.2e}）")
        print("  正文公式：Σ_{i<j} <v_i,v_j> x_i x_j = ½(‖Σv_ix_i‖² − Σ‖v_ix_i‖²)")
