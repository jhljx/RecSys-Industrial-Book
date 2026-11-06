"""
场感知因子分解机（FFM）精排模型（第12章 §12.3.4）

FFM 在 FM 基础上引入特征域（Field）：
  特征 i 与不同特征域的 j 交互时，使用专门面向 f_j 的隐向量 v_{i, f_j}。

    s(x) = w_0 + Σ w_i x_i + Σ_{i<j} <v_{i, f_j}, v_{j, f_i}> x_i x_j

与 FM 的对比：
  - FM  : 每个特征一个隐向量 v_i
  - FFM : 每个特征 × 每个特征域一个隐向量 v_{i,f}，参数为 FM 的 F 倍

本书依据：《第十二章-精排模型.pdf》§12.3.4，PDF 第 7–8 页。
依赖：PyTorch。运行：python3 ffm.py
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim


def _auc(scores, labels) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


# ── FFM 模型 ──────────────────────────────────────────────────────────────────

class FFM(nn.Module):
    """
    场感知因子分解机。

    参数化（正文 PDF §12.3.4）：
        特征 i 面向特征域 f 的隐向量为 v_{i,f} ∈ ℝ^k
        二阶项 <v_{i,f_j}, v_{j,f_i}> x_i x_j

    实现：
        emb_table : (n_values, n_fields, k)  每个特征值 × 每个目标域一个向量
        与 FM 共享一阶项 w_i，二阶项直接枚举非零字段对（单样本 <10 字段，可接受）
    """

    def __init__(self, n_values: int, n_fields: int,
                 emb_dim: int = 4, reg: float = 1e-4):
        super().__init__()
        self.n_fields = n_fields
        self.w = nn.Embedding(n_values, 1)
        # v_{i, f}: n_values 个特征值 × n_fields 个目标域 × k
        self.v = nn.Parameter(torch.randn(n_values, n_fields, emb_dim) * 0.01)
        self.w0 = nn.Parameter(torch.zeros(1))
        self.reg = reg
        nn.init.zeros_(self.w.weight)

    def forward(self, feat_ids: torch.Tensor,
                feat_vals: torch.Tensor) -> torch.Tensor:
        """
        feat_ids  : (B, F)
        feat_vals : (B, F)
        返回 logit (B,)
        """
        B, F = feat_ids.shape
        first = (self.w(feat_ids).squeeze(-1) * feat_vals).sum(dim=1)

        second = torch.zeros(B, device=feat_ids.device)
        for i in range(F):
            for j in range(i + 1, F):
                # 特征 i 用面向「域 j」的向量；特征 j 用面向「域 i」的向量
                vi = self.v[feat_ids[:, i], j]      # (B, k)
                vj = self.v[feat_ids[:, j], i]      # (B, k)
                pair = (vi * vj).sum(dim=1)         # (B,)
                second = second + pair * feat_vals[:, i] * feat_vals[:, j]

        return self.w0 + first + second

    def predict_proba(self, feat_ids, feat_vals) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(feat_ids, feat_vals))

    def fit(self, feat_ids, feat_vals, y, epochs: int = 60, lr: float = 0.02):
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)
        for ep in range(epochs):
            opt.zero_grad()
            logit = self.forward(feat_ids, feat_vals)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y.float())
            loss.backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")

    def rank(self, feat_ids, feat_vals, top_k) -> list[tuple[int, float]]:
        scores = self.predict_proba(feat_ids, feat_vals).tolist()
        return sorted(enumerate(scores), key=lambda t: -t[1])[:top_k]


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 400, n_fields: int = 5, vals_per_field: int = 6,
              seed: int = 42):
    """
    生成带「域间差异」的合成数据：
    - 字段 0（用户年龄）与字段 1（物品类别）的交互为正；
    - 字段 0 与字段 2（上下文位置）的交互为负，且强度不同。
    这让 FFM 有区分不同域交互的动机（FM 只能用同一套隐向量）。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    ids = torch.zeros(n, n_fields, dtype=torch.long)
    for f in range(n_fields):
        ids[:, f] = torch.randint(0, vals_per_field, (n,)) + f * vals_per_field
    vals = torch.ones(n, n_fields)

    local = ids % vals_per_field
    # 主效应 + 域相关交互
    logit = (0.4 * local[:, 0].float() + 0.3 * local[:, 1].float()
             + 0.2 * local[:, 2].float()
             + 0.6 * (local[:, 0] == local[:, 1]).float()      # 同值→正交互
             - 0.5 * (local[:, 0] == local[:, 2]).float()      # 同值→负交互
             - 1.5)
    y = (torch.sigmoid(logit + 0.5 * torch.randn(n)) > 0.5).long()
    return ids, vals, y


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    N_FIELDS, VALS = 5, 6
    N_VALUES = N_FIELDS * VALS

    ids, vals, y = make_data(400, N_FIELDS, VALS)
    split = 300
    ids_tr, vals_tr, y_tr = ids[:split], vals[:split], y[:split]
    ids_te, vals_te, y_te = ids[split:], vals[split:], y[split:]

    # ── FFM 训练 ──────────────────────────────────────────────────────────────
    print("=== FFM 训练（5 字段 × 6 取值，k=4）===")
    ffm = FFM(n_values=N_VALUES, n_fields=N_FIELDS, emb_dim=4)
    ffm.fit(ids_tr, vals_tr, y_tr, epochs=60)

    p_te = ffm.predict_proba(ids_te, vals_te).tolist()
    auc_te = _auc(p_te, y_te.tolist())
    print(f"\n测试集 AUC={auc_te:.4f}")

    # ── 参数量对比 ────────────────────────────────────────────────────────────
    ffm_emb_params = ffm.v.numel()
    fm_emb_params = N_VALUES * 4   # FM 若同样 k=4，隐向量参数 = n_values * k
    print(f"\n参数量对比（k=4）：")
    print(f"  FM  二阶参数: {fm_emb_params}")
    print(f"  FFM 二阶参数: {ffm_emb_params}  (FM 的 {ffm_emb_params / fm_emb_params:.1f} 倍)")
    print("  正文指出：FFM 参数规模明显大于 FM，计算成本随有效特征数快速上升。")

    # ── 精排推理演示 ──────────────────────────────────────────────────────────
    top5 = ffm.rank(ids_te[:20], vals_te[:20], top_k=5)
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, score) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={score:.4f}  label={y_te[idx].item()}")

    # ── 域感知示例 ────────────────────────────────────────────────────────────
    # 同一特征值（如「用户年龄=2」）面向不同域时应学到不同表示
    with torch.no_grad():
        val_id = 2                        # 字段 0 的第 3 个取值
        toward_item = ffm.v[val_id, 1]    # 面向物品域
        toward_ctx  = ffm.v[val_id, 2]    # 面向上下文域
        cos = torch.cosine_similarity(toward_item, toward_ctx, dim=0).item()
    print(f"\n域感知验证：同一特征面向物品域/上下文域向量的余弦相似度={cos:.4f}")
    print("  （FFM 允许两个域学到不同表示；FM 只有单一向量，相似度恒为 1）")
