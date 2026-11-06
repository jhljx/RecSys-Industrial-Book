"""
PNN 精排模型（第12章 §12.4.1.2）

Product-based Neural Network 在 Embedding 之后插入 Product Layer，
显式构造字段之间的乘性二阶交互，再送入 DNN：

    f_i = W_0^(i) x_i
    z = Concat(f_1,...,f_N)
    l_1 = ReLU(l_z + l_p + b_1)
    ŷ = σ(DNN(l_1))

两种乘积形式（正文 PDF 第 14–19 页）：
  - IPNN：内积 p_{i,j}=<f_i,f_j>，用秩一/低秩分解把 O(N²) 降到 O(N)
  - OPNN：外积 p_{i,j}=f_i f_jᵀ，用特征叠加 f_Σ=Σf_i 压缩为 M×M 矩阵

本书依据：《第十二章-精排模型.pdf》§12.4.1.2，PDF 第 12–19 页。
依赖：PyTorch。运行：python3 pnn.py
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


# ── IPNN ──────────────────────────────────────────────────────────────────────

class IPNN(nn.Module):
    """
    内积版 PNN。

    正文关键推导（PDF 第 16–18 页）：
      l_p^(n) = Σ_i Σ_j θ_i^(n) θ_j^(n) <f_i,f_j>
               = ‖Σ_i θ_i^(n) f_i‖²         （秩一分解，O(D1·M·N)）
    低秩推广：
      l_p^(n) = Σ_{r=1}^{K} ‖Σ_i θ_{i,r}^(n) f_i‖²
    """

    def __init__(self, n_fields: int, vocab_size: int, emb_dim: int = 8,
                 d1: int = 24, rank: int = 1, hidden: int = 16):
        super().__init__()
        self.n_fields = n_fields
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for _ in range(n_fields)])
        self.z_dim = n_fields * emb_dim

        # 线性部分 W_z^(n)：z → D1
        self.wz = nn.Linear(self.z_dim, d1)
        # 交互部分 θ^(n) ∈ ℝ^(N × K)：每个隐藏节点 n 一组字段权重
        # 用 (d1, rank, n_fields) 存储，输出 l_p = Σ_r ‖Σ_i θ_{n,r,i} f_i‖²
        self.theta = nn.Parameter(torch.randn(d1, rank, n_fields) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(d1))
        self.mlp = nn.Sequential(
            nn.Linear(d1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def product_layer(self, fields: torch.Tensor) -> torch.Tensor:
        """fields: (B, N, M) → l1: (B, D1)"""
        B, N, M = fields.shape
        z = fields.reshape(B, N * M)

        # l_z^(n) = <W_z^(n), z>
        l_z = self.wz(z)                                  # (B, D1)

        # l_p^(n) = Σ_r ‖Σ_i θ_{n,r,i} f_i‖²   —— O(B·D1·K·N·M)
        # theta: (D1, K, N) → s: (B, D1, K, M)
        s = torch.einsum("dkn,bnm->bdkm", self.theta, fields)
        l_p = s.pow(2).sum(dim=(2, 3))                    # (B, D1)

        return F.relu(l_z + l_p + self.b1)

    def forward(self, feat_ids):
        fields = torch.stack([emb(feat_ids[:, i])
                              for i, emb in enumerate(self.embs)], dim=1)
        l1 = self.product_layer(fields)
        return self.mlp(l1).squeeze(-1)


# ── OPNN ──────────────────────────────────────────────────────────────────────

class OPNN(nn.Module):
    """
    外积版 PNN。

    正文（PDF 第 18 页）用特征叠加把 N²M² 压缩为 M×M：
      f_Σ = Σ_i f_i
      p = f_Σ f_Σᵀ
      l_p^(n) = <W_p^(n), f_Σ f_Σᵀ>
    """

    def __init__(self, n_fields: int, vocab_size: int, emb_dim: int = 8,
                 d1: int = 24, hidden: int = 16):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for _ in range(n_fields)])
        self.z_dim = n_fields * emb_dim
        self.emb_dim = emb_dim

        self.wz = nn.Linear(self.z_dim, d1)
        # W_p^(n) ∈ ℝ^(M×M)，但由于外积对称，只需对称部分
        self.wp = nn.Parameter(torch.randn(d1, emb_dim, emb_dim) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(d1))
        self.mlp = nn.Sequential(
            nn.Linear(d1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def product_layer(self, fields):
        B, N, M = fields.shape
        z = fields.reshape(B, N * M)
        l_z = self.wz(z)                                   # (B, D1)

        f_sigma = fields.sum(dim=1)                        # (B, M)
        outer = f_sigma.unsqueeze(2) * f_sigma.unsqueeze(1)  # (B, M, M)
        # l_p^(n) = <W_p^(n), outer> = Σ_{a,b} W[a,b] * outer[a,b]
        l_p = torch.einsum("bij,nij->bn", outer, self.wp)  # (B, D1)
        return F.relu(l_z + l_p + self.b1)

    def forward(self, feat_ids):
        fields = torch.stack([emb(feat_ids[:, i])
                              for i, emb in enumerate(self.embs)], dim=1)
        return self.mlp(self.product_layer(fields)).squeeze(-1)


# ── 通用训练循环 ──────────────────────────────────────────────────────────────

def train(model, ids_tr, y_tr, ids_te, y_te, epochs: int = 40, lr: float = 1e-2):
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(ids_tr), y_tr.float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.sigmoid(model(ids_te)).tolist()
    return loss.item(), _auc(p, y_te.tolist())


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 500, n_fields: int = 5, vocab: int = 8, seed: int = 42):
    """标签含字段 0×1 与 2×3 的乘性交互，突出 Product Layer 的作用。"""
    torch.manual_seed(seed)
    random.seed(seed)
    ids = torch.randint(0, vocab, (n, n_fields), dtype=torch.long)
    local = ids.float()

    logit = (0.35 * local[:, 0] + 0.3 * local[:, 1] - 0.25 * local[:, 2]
             + 0.9 * (local[:, 0] * local[:, 1]) / vocab      # 乘性交互
             + 0.7 * (local[:, 2] * local[:, 3]) / vocab
             - 2.5)
    y = (torch.sigmoid(logit + 0.4 * torch.randn(n)) > 0.5).long()
    return ids, y


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    N_FIELDS, VOCAB = 5, 8
    ids, y = make_data(500, N_FIELDS, VOCAB)
    split = 380
    ids_tr, y_tr = ids[:split], y[:split]
    ids_te, y_te = ids[split:], y[split:]

    print(f"数据：{N_FIELDS} 字段 × {VOCAB} 取值，训练 {split} / 测试 {500-split}")

    # ── IPNN（秩一）──────────────────────────────────────────────────────────
    torch.manual_seed(42)
    ipnn = IPNN(N_FIELDS, VOCAB, emb_dim=8, d1=24, rank=1)
    loss_i, auc_i = train(ipnn, ids_tr, y_tr, ids_te, y_te)
    print(f"\nIPNN（K=1 秩一分解）  loss={loss_i:.4f}  AUC={auc_i:.4f}")

    # ── IPNN（秩 K=4）────────────────────────────────────────────────────────
    torch.manual_seed(42)
    ipnn_k4 = IPNN(N_FIELDS, VOCAB, emb_dim=8, d1=24, rank=4)
    loss_k, auc_k = train(ipnn_k4, ids_tr, y_tr, ids_te, y_te)
    n_theta_k1 = ipnn.theta.numel()
    print(f"IPNN（K=4 低秩）      loss={loss_k:.4f}  AUC={auc_k:.4f}")
    print(f"  交互参数：K=1 → {n_theta_k1}，K=4 → {ipnn_k4.theta.numel()}"
          f"（{ipnn_k4.theta.numel()/n_theta_k1:.0f} 倍）")

    # ── OPNN ─────────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    opnn = OPNN(N_FIELDS, VOCAB, emb_dim=8, d1=24)
    loss_o, auc_o = train(opnn, ids_tr, y_tr, ids_te, y_te)
    print(f"OPNN（特征叠加外积）  loss={loss_o:.4f}  AUC={auc_o:.4f}")

    # ── 精排推理演示 ──────────────────────────────────────────────────────────
    with torch.no_grad():
        scores = torch.sigmoid(ipnn(ids_te[:20])).tolist()
    top5 = sorted(enumerate(scores), key=lambda t: -t[1])[:5]
    print("\nIPNN 精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={s:.4f}  label={y_te[idx].item()}")

    # ── 复杂度对比 ────────────────────────────────────────────────────────────
    print(f"\n复杂度（正文 §12.4.1.2）：")
    print(f"  IPNN 秩一: O(D1·M·N)         本实现即此形式")
    print(f"  OPNN 叠加: O(D1·M·(M+N))     本实现即此形式")
    print("  Product Layer 是 PNN 区别于 FNN 的关键组件：")
    print("  去掉二阶交互后网络退化为 Embedding + MLP 的普通深度模型。")
