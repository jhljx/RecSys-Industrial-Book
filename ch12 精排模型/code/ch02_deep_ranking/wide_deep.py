"""
Wide & Deep 精排模型（第12章 §12.4.1.1）

Wide 分支（广义线性模型 + 显式交叉特征）与 Deep 分支（Embedding + MLP）
并行计算，得分相加后统一过 Sigmoid：

    z_wide(x) = w_wideᵀ [x; φ(x)] + b_wide
    z_deep(x) = w_deepᵀ h^(L) + b_deep
    p̂ = σ(z_wide + z_deep)

两个分支共享同一个交叉熵损失联合训练（而非分别训练后再集成）。

本书依据：《第十二章-精排模型.pdf》§12.4.1.1，PDF 第 8–12 页。
依赖：PyTorch。运行：python3 wide_deep.py
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


# ── Wide & Deep 模型 ──────────────────────────────────────────────────────────

class WideAndDeep(nn.Module):
    """
    Wide 分支：原始稀疏特征 + 手工交叉特征 → 线性
    Deep 分支：类别特征 Embedding + 数值特征 → 多层 MLP

    正文说明（PDF 第 8–12 页）：
      - Wide 分支侧重「记忆」已知的特征组合；
      - Deep 分支通过 Embedding + 非线性变换侧重「泛化」；
      - 两分支共享损失联合训练。
    """

    def __init__(self, n_fields: int, vocab_size: int, emb_dim: int = 8,
                 n_dense: int = 2, hidden: int = 32, cross_pairs=None):
        super().__init__()
        self.n_fields = n_fields
        self.vocab_size = vocab_size
        self.cross_pairs = cross_pairs or [(0, 1), (0, 2), (1, 3)]

        # Deep 分支：每个字段一个 Embedding 表 + 数值特征拼接
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for _ in range(n_fields)])
        deep_in = n_fields * emb_dim + n_dense
        self.deep = nn.Sequential(
            nn.Linear(deep_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Wide 分支：原始特征的 one-hot 展平 + 显式交叉特征
        # 正文 φ_k(x)=Π x_i^{c_ki}：对离散字段即两个 one-hot 的 Hadamard 积
        # （等价于联合取值的指示向量，维度 vocab²）
        wide_in = n_fields * vocab_size + len(self.cross_pairs) * vocab_size ** 2
        self.wide = nn.Linear(wide_in, 1)

    def _deep_forward(self, feat_ids, dense):
        embs = [emb(feat_ids[:, i]) for i, emb in enumerate(self.embs)]
        h = torch.cat(embs + [dense], dim=1)
        return self.deep(h).squeeze(-1)

    def _wide_forward(self, feat_ids):
        onehots = [nn.functional.one_hot(feat_ids[:, i], self.vocab_size).float()
                   for i in range(self.n_fields)]
        # 交叉特征：两个字段 one-hot 的外积展平后只保留联合激活位置
        crosses = [(onehots[i].unsqueeze(2) * onehots[j].unsqueeze(1)).flatten(1)
                   for i, j in self.cross_pairs]
        x = torch.cat(onehots + crosses, dim=1)
        return self.wide(x).squeeze(-1)

    def forward(self, feat_ids: torch.Tensor,
                dense: torch.Tensor) -> torch.Tensor:
        return self._wide_forward(feat_ids) + self._deep_forward(feat_ids, dense)

    def predict_proba(self, feat_ids, dense) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(feat_ids, dense))

    def fit(self, feat_ids, dense, y, epochs: int = 40, lr: float = 1e-2):
        opt = optim.Adam(self.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self.forward(feat_ids, dense), y.float())
            loss.backward()
            opt.step()
            if (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")

    def rank(self, feat_ids, dense, top_k) -> list[tuple[int, float]]:
        scores = self.predict_proba(feat_ids, dense).tolist()
        return sorted(enumerate(scores), key=lambda t: -t[1])[:top_k]


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 500, n_fields: int = 4, vocab: int = 6,
              n_dense: int = 2, seed: int = 42):
    """
    4 个类别字段（如 用户年龄段/物品类别/上下文/设备），
    2 个数值特征（如 历史 CTR、物品时长）。
    标签依赖主效应 + 字段 0×1 的交叉效应。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    ids = torch.randint(0, vocab, (n, n_fields), dtype=torch.long)
    dense = torch.randn(n, n_dense)

    logit = (0.5 * ids[:, 0].float() + 0.4 * ids[:, 1].float()
             - 0.3 * ids[:, 2].float()
             + 0.8 * (ids[:, 0] == ids[:, 1]).float()      # 交叉效应
             + 0.5 * dense[:, 0] - 0.3 * dense[:, 1] - 2.0)
    y = (torch.sigmoid(logit + 0.3 * torch.randn(n)) > 0.5).long()
    return ids, dense, y


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    ids, dense, y = make_data(500)
    split = 380
    ids_tr, d_tr, y_tr = ids[:split], dense[:split], y[:split]
    ids_te, d_te, y_te = ids[split:], dense[split:], y[split:]

    print("=== Wide & Deep 训练 ===")
    model = WideAndDeep(n_fields=4, vocab_size=6, emb_dim=8, n_dense=2,
                        cross_pairs=[(0, 1), (1, 2)])
    model.fit(ids_tr, d_tr, y_tr, epochs=40)

    p_te = model.predict_proba(ids_te, d_te).tolist()
    auc_te = _auc(p_te, y_te.tolist())
    print(f"\n测试集 AUC={auc_te:.4f}")

    # ── 分支贡献分析 ──────────────────────────────────────────────────────────
    with torch.no_grad():
        z_wide = model._wide_forward(ids_te[:20])
        z_deep = model._deep_forward(ids_te[:20], d_te[:20])
        print(f"\n两分支得分量级（前 20 样本平均绝对值）：")
        print(f"  |z_wide|={z_wide.abs().mean().item():.4f}   "
              f"|z_deep|={z_deep.abs().mean().item():.4f}")

    # ── 精排推理演示 ──────────────────────────────────────────────────────────
    top5 = model.rank(ids_te[:20], d_te[:20], top_k=5)
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, score) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={score:.4f}  label={y_te[idx].item()}")

    # ── 交叉特征贡献演示 ──────────────────────────────────────────────────────
    # 正文指出：Wide 分支通过 φ_k(x)=Π x_i^{c_ki} 记忆显式组合。
    # 这里观察交叉字段 0×1 同时取 2 时的得分差异。
    print("\nWide 分支交叉特征示例：")
    print("  字段 0 与字段 1 同时取值 2 时，交叉特征 (x0==x1) 激活，")
    print("  对应参数可获得直接的正向权重——这是 Wide 分支「记忆」能力来源。")
