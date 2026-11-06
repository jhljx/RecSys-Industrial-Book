"""
xDeepFM 精排模型（第12章 §12.4.1.4）

xDeepFM = 线性组件 + CIN 组件 + DNN 组件。

CIN（Compressed Interaction Network）逐层把上一层的交互表示与原始字段嵌入
再做逐元素乘积，显式构造任意阶交互：

    X^(k)_{h,:} = Σ_i^{H_{k-1}} Σ_j^{m} W^(k,h)_{i,j} (X^(k-1)_{i,:} ∘ X^(0)_{j,:})
    p^(k) = sum-pooling(X^(k))
    p⁺ = [p^(1); p^(2); ...; p^(T)]

    ŷ = σ(w_linearᵀx + w_CINᵀp⁺ + z_DNN)

本书依据：《第十二章-精排模型.pdf》§12.4.1.4，PDF 第 24–30 页。
依赖：PyTorch。运行：python3 xdeepfm.py
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


# ── CIN 组件 ──────────────────────────────────────────────────────────────────

class CIN(nn.Module):
    """
    压缩交互网络。

    实现约定：
      X^(0) ∈ ℝ^(m × D)，m = 字段数，D = 嵌入维度
      第 k 层 H_k 个交互向量，H_k 由参数决定（默认各层相同）
      每层的 W^(k,h) ∈ ℝ^(H_{k-1} × m)
    """

    def __init__(self, n_fields: int, emb_dim: int, layer_sizes=(8, 8)):
        super().__init__()
        self.n_fields = n_fields
        self.emb_dim = emb_dim
        self.layer_sizes = list(layer_sizes)

        # 每层的联合权重 (H_k, H_{k-1}, m)
        self.weights = nn.ParameterList()
        prev_h = n_fields                    # H_0 = m
        for h in self.layer_sizes:
            self.weights.append(
                nn.Parameter(torch.randn(h, prev_h, n_fields) * 0.05))
            prev_h = h

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        x0: (B, m, D)
        返回 p⁺: (B, ΣH_k)
        """
        B = x0.size(0)
        x_prev = x0                                    # (B, H_0=m, D)
        pooled = []
        for layer_idx, w in enumerate(self.weights):
            H_k, H_prev, m = w.shape
            # z[i,j] = x_prev[:, i, :] ∘ x0[:, j, :] → (B, H_prev, m, D)
            z = x_prev.unsqueeze(2) * x0.unsqueeze(1)
            # X^(k)_{h} = Σ_{i,j} W[h,i,j] z[i,j]
            # einsum: (h,i,j) × (b,i,j,d) → (b,h,d)
            x_cur = torch.einsum("hij,bijd->bhd", w, z)
            # Sum pooling over embedding dim → p^(k) ∈ ℝ^(B, H_k)
            pooled.append(x_cur.sum(dim=2))
            x_prev = x_cur
        return torch.cat(pooled, dim=1)                # (B, ΣH_k)


# ── xDeepFM 完整模型 ─────────────────────────────────────────────────────────

class xDeepFM(nn.Module):
    def __init__(self, n_fields: int, vocab_size: int, emb_dim: int = 6,
                 cin_layers=(6, 6), hidden: int = 24):
        super().__init__()
        self.n_fields = n_fields
        self.emb = nn.Embedding(vocab_size * n_fields, emb_dim)
        self.linear = nn.Embedding(vocab_size * n_fields, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        self.cin = CIN(n_fields, emb_dim, cin_layers)
        cin_out = sum(cin_layers)
        self.cin_proj = nn.Linear(cin_out, 1)

        self.dnn = nn.Sequential(
            nn.Linear(n_fields * emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.emb.weight, std=0.01)

    def _gids(self, feat_ids):
        offsets = (torch.arange(self.n_fields, device=feat_ids.device)
                   * (self.emb.num_embeddings // self.n_fields))
        return feat_ids + offsets

    def forward(self, feat_ids: torch.Tensor) -> torch.Tensor:
        gids = self._gids(feat_ids)
        x0 = self.emb(gids)                            # (B, m, D)

        z_linear = self.linear(gids).sum(dim=(1, 2))
        p_plus = self.cin(x0)
        z_cin = self.cin_proj(p_plus).squeeze(-1)
        z_dnn = self.dnn(x0.reshape(x0.size(0), -1)).squeeze(-1)
        return self.bias + z_linear + z_cin + z_dnn

    def predict_proba(self, feat_ids):
        with torch.no_grad():
            return torch.sigmoid(self.forward(feat_ids))

    def fit(self, feat_ids, y, epochs: int = 40, lr: float = 1e-2):
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for ep in range(epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                self.forward(feat_ids), y.float())
            loss.backward()
            opt.step()
            if (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")
        return loss.item()

    def rank(self, feat_ids, top_k):
        scores = self.predict_proba(feat_ids).tolist()
        return sorted(enumerate(scores), key=lambda t: -t[1])[:top_k]


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 500, n_fields: int = 5, vocab: int = 8, seed: int = 42):
    """
    含三阶交互的数据：字段 0×1×2 的联合模式对标签影响较大，
    用于观察 CIN 显式建模高阶交互的能力（DNN 只能隐式学习）。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    ids = torch.randint(0, vocab, (n, n_fields), dtype=torch.long)
    local = ids.float()
    triple = (local[:, 0] > 3).float() * (local[:, 1] < 4).float() * \
             (local[:, 2] > 2).float()
    logit = (0.3 * local[:, 0] + 0.2 * local[:, 3]
             + 1.6 * triple                                # 三阶交互
             + 0.7 * (local[:, 0] == local[:, 1]).float()
             - 2.3)
    y = (torch.sigmoid(logit + 0.35 * torch.randn(n)) > 0.5).long()
    return ids, y


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    N_FIELDS, VOCAB = 5, 8
    ids, y = make_data(600, N_FIELDS, VOCAB)
    split = 480
    ids_tr, y_tr = ids[:split], y[:split]
    ids_te, y_te = ids[split:], y[split:]

    print("=== xDeepFM 训练（线性 + CIN + DNN）===")
    model = xDeepFM(N_FIELDS, VOCAB, emb_dim=6, cin_layers=(6, 6))
    model.fit(ids_tr, y_tr, epochs=30)

    p_te = model.predict_proba(ids_te).tolist()
    auc_te = _auc(p_te, y_te.tolist())
    print(f"\n测试集 AUC={auc_te:.4f}")

    # ── 三组件贡献 ────────────────────────────────────────────────────────────
    with torch.no_grad():
        gids = model._gids(ids_te[:20])
        x0 = model.emb(gids)
        z_lin = model.linear(gids).sum(dim=(1, 2))
        z_cin = model.cin_proj(model.cin(x0)).squeeze(-1)
        z_dnn = model.dnn(x0.reshape(x0.size(0), -1)).squeeze(-1)
    print(f"\n三组件得分量级（前 20 样本平均绝对值）：")
    print(f"  |z_linear|={z_lin.abs().mean().item():.4f}   "
          f"|z_CIN|={z_cin.abs().mean().item():.4f}   "
          f"|z_DNN|={z_dnn.abs().mean().item():.4f}")

    # ── CIN 维度流转 ──────────────────────────────────────────────────────────
    with torch.no_grad():
        p_plus = model.cin(model.emb(model._gids(ids_te[:4])))
    print(f"\nCIN 池化输出维度：{tuple(p_plus.shape)} "
          f"(B=4, ΣH_k={sum(model.cin.layer_sizes)})")
    print(f"  第 1 层构造二阶交互（H_0=m={N_FIELDS} → H_1={model.cin.layer_sizes[0]}）")
    print(f"  第 2 层构造三阶交互（H_1 → H_2={model.cin.layer_sizes[1]}）")
    print("  正文 §12.4.1.4：交互阶数随网络深度递增，这是 CIN 与 DNN 的本质区别。")

    # ── 精排推理演示 ──────────────────────────────────────────────────────────
    top5 = model.rank(ids_te[:20], top_k=5)
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={s:.4f}  label={y_te[idx].item()}")
