"""
DeepFM 精排模型（第12章 §12.4.1.3）

DeepFM = FM 组件 + Deep 组件，二者共享同一套特征嵌入，端到端联合训练：

    z_FM   = b + Σ w_i x_i + Σ_{i<j} <v_i,v_j> x_i x_j
    z_DNN  = w_DNNᵀ a^(L) + b_DNN,   a^(0) = [e_1; e_2; ...; e_m]
    ŷ      = σ(z_FM + z_DNN)

关键设计（正文 PDF 第 19–24 页）：
  - 用 FM 替换 Wide & Deep 中依赖人工交叉的 Wide 分支；
  - FM 与 DNN 共享 Embedding，无需先在 FM 上预训练（区别于 FNN）。

本书依据：《第十二章-精排模型.pdf》§12.4.1.3，PDF 第 19–24 页。
依赖：PyTorch。运行：python3 deepfm.py
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


class DeepFM(nn.Module):
    """
    FM 组件与 Deep 组件共享 self.emb（Embedding 表）。

    正文（PDF 第 21 页）强调共享的梯度效应：
    「来自 FM 组件和 Deep 组件的梯度会共同作用于同一组嵌入参数」。
    """

    def __init__(self, n_fields: int, vocab_size: int, emb_dim: int = 8,
                 hidden: int = 32, deep_layers: int = 2):
        super().__init__()
        self.n_fields = n_fields
        # 共享嵌入：既是 FM 的隐向量 v_i，也是 Deep 组件的输入嵌入
        self.emb = nn.Embedding(vocab_size * n_fields, emb_dim)

        # 一阶权重（FM 的 w_i）
        self.first_order = nn.Embedding(vocab_size * n_fields, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # DNN
        layers = []
        in_dim = n_fields * emb_dim
        for _ in range(deep_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.dnn = nn.Sequential(*layers)

        nn.init.zeros_(self.first_order.weight)
        nn.init.normal_(self.emb.weight, std=0.01)

    def _global_ids(self, feat_ids: torch.Tensor) -> torch.Tensor:
        """把 (B, F) 的字段内 ID 映射到全局 Embedding 下标。"""
        offsets = torch.arange(self.n_fields, device=feat_ids.device) * \
            self.emb.num_embeddings // self.n_fields
        return feat_ids + offsets

    def fm_part(self, gids: torch.Tensor) -> torch.Tensor:
        """FM 组件：一阶 + 二阶（和平方 − 平方和 恒等式）。"""
        first = self.first_order(gids).sum(dim=(1, 2))
        v = self.emb(gids)                                  # (B, F, k)
        sq_of_sum = v.sum(dim=1).pow(2).sum(dim=1)
        sum_of_sq = v.pow(2).sum(dim=(1, 2))
        second = 0.5 * (sq_of_sum - sum_of_sq)
        return self.bias + first + second

    def dnn_part(self, gids: torch.Tensor) -> torch.Tensor:
        """Deep 组件：a^(0) = [e_1;...;e_m] → MLP。"""
        a0 = self.emb(gids).reshape(gids.size(0), -1)
        return self.dnn(a0).squeeze(-1)

    def forward(self, feat_ids: torch.Tensor) -> torch.Tensor:
        gids = self._global_ids(feat_ids)
        return self.fm_part(gids) + self.dnn_part(gids)

    def predict_proba(self, feat_ids) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(feat_ids))

    def fit(self, feat_ids, y, epochs: int = 40, lr: float = 1e-2):
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for ep in range(epochs):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self.forward(feat_ids), y.float())
            loss.backward()
            opt.step()
            if (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")
        return loss.item()

    def rank(self, feat_ids, top_k) -> list[tuple[int, float]]:
        scores = self.predict_proba(feat_ids).tolist()
        return sorted(enumerate(scores), key=lambda t: -t[1])[:top_k]


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 500, n_fields: int = 5, vocab: int = 8, seed: int = 42):
    """
    同时含线性主效应与两类交互：
      - 字段 0×1 的「同值」指示交互（低阶，FM 可直接表达）
      - 字段 2×3 的乘积项（需非线性，DNN 表达）
    这样可以观察 FM 与 DNN 两分支各自的贡献。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    ids = torch.randint(0, vocab, (n, n_fields), dtype=torch.long)
    local = ids.float()
    logit = (0.3 * local[:, 0] + 0.25 * local[:, 1] + 0.2 * local[:, 4]
             + 0.8 * (local[:, 0] == local[:, 1]).float()
             + 0.5 * (local[:, 2] * local[:, 3]) / vocab
             - 2.2)
    y = (torch.sigmoid(logit + 0.35 * torch.randn(n)) > 0.5).long()
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

    print("=== DeepFM 训练（FM 与 DNN 共享嵌入）===")
    model = DeepFM(N_FIELDS, VOCAB, emb_dim=8, hidden=32)
    model.fit(ids_tr, y_tr, epochs=40)

    p_te = model.predict_proba(ids_te).tolist()
    auc_te = _auc(p_te, y_te.tolist())
    print(f"\n测试集 AUC={auc_te:.4f}")

    # ── 两分支贡献 ────────────────────────────────────────────────────────────
    with torch.no_grad():
        gids = model._global_ids(ids_te[:20])
        z_fm = model.fm_part(gids)
        z_dnn = model.dnn_part(gids)
    print(f"\n两分支得分量级（前 20 样本平均绝对值）：")
    print(f"  |z_FM|={z_fm.abs().mean().item():.4f}   "
          f"|z_DNN|={z_dnn.abs().mean().item():.4f}")

    # ── 共享嵌入验证 ──────────────────────────────────────────────────────────
    # 正文强调：FM 与 DNN 用的不是两份参数，而是同一张 Embedding 表。
    # 检查 Embedding 是否同时收到两个分支的梯度。
    model.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model.fm_part(model._global_ids(ids_tr[:50])), y_tr[:50].float())
    loss.backward()
    grad_from_fm = model.emb.weight.grad.abs().sum().item()
    model.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model.dnn_part(model._global_ids(ids_tr[:50])), y_tr[:50].float())
    loss.backward()
    grad_from_dnn = model.emb.weight.grad.abs().sum().item()
    print(f"\n共享嵌入梯度范数（同一张表）：")
    print(f"  仅 FM 分支反传: {grad_from_fm:.4f}")
    print(f"  仅 DNN 分支反传: {grad_from_dnn:.4f}")
    print("  → 两个组件共同更新同一组嵌入参数（正文 §12.4.1.3）")

    # ── 精排推理演示 ──────────────────────────────────────────────────────────
    top5 = model.rank(ids_te[:20], top_k=5)
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={s:.4f}  label={y_te[idx].item()}")
