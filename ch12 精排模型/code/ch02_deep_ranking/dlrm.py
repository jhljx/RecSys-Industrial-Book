"""
DLRM 精排模型（第12章 §12.4.1.5）

Deep Learning Recommendation Model（Meta, 2019）：

    e_dense  = bottom_mlp(x_dense)          # 稠密特征 → 同维嵌入
    e_i      = Embedding(feat_i)             # 类别特征 → 嵌入
    s_{ij}   = e_i · e_j                     # 两两内积（显式二阶交互）
    z        = [e_dense; s_{01},…,s_{m-1,m}]
    ŷ        = σ(top_mlp(z))

特征交互模块仅计算内积，不构造高阶交互（区别于 CIN/xDeepFM）。
顶部 MLP 在二阶交互结果和稠密嵌入的基础上学习高阶非线性。

本书依据：《第十二章-精排模型.pdf》§12.4.1.5，PDF 第 30–36 页。
依赖：PyTorch。运行：python3 dlrm.py
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


def _make_mlp(dims: list[int], act=nn.ReLU) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


# ── DLRM 模型 ─────────────────────────────────────────────────────────────────

class DLRM(nn.Module):
    """
    DLRM：底部 MLP + 类别嵌入 → 两两内积交互 → 顶部 MLP。

    正文（PDF 第 30–36 页）关键设计：
      - bottom_mlp 把稠密数值特征映射到与嵌入相同的 emb_dim 维度；
      - 内积交互层计算所有 C(m+1,2) 对特征向量的内积，输出为一维向量 s；
      - top_mlp 的输入为 [e_dense; s]（保留稠密特征表示 + 交互特征）；
      - 与 DeepFM 将 FM 和 DNN 直接融合不同，DLRM 将交互结果作为 top_mlp 输入。
    """

    def __init__(self, n_sparse: int, vocab_size: int, emb_dim: int = 8,
                 dense_dim: int = 4,
                 bottom_dims: list[int] | None = None,
                 top_dims: list[int] | None = None):
        """
        参数：
          n_sparse   : 类别（稀疏）字段数
          vocab_size : 每个字段的取值数（统一 vocab，演示简化）
          emb_dim    : 嵌入维度（bottom_mlp 输出维度也是此值）
          dense_dim  : 稠密数值特征维度
          bottom_dims: bottom_mlp 隐藏层维度，最后一层必须 == emb_dim
          top_dims   : top_mlp 各层维度，最后为 1（输出 logit）
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.n_sparse = n_sparse

        # 类别特征嵌入（每个字段独立嵌入表，真实系统中 vocab 各不相同）
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for _ in range(n_sparse)])

        # 底部 MLP：稠密特征 → emb_dim
        b_dims = bottom_dims or [dense_dim, emb_dim * 2, emb_dim]
        assert b_dims[-1] == emb_dim, "bottom_mlp 最后一层维度必须等于 emb_dim"
        self.bottom_mlp = _make_mlp(b_dims, nn.ReLU)

        # 两两内积后的维度：C(n_sparse+1, 2) 个标量
        n_vecs = n_sparse + 1                       # +1 for e_dense
        n_interact = n_vecs * (n_vecs - 1) // 2

        # 顶部 MLP：[e_dense(emb_dim); s(n_interact)] → 1
        t_dims = top_dims or [emb_dim + n_interact, 32, 1]
        self.top_mlp = _make_mlp(t_dims, nn.ReLU)

        for emb in self.embs:
            nn.init.normal_(emb.weight, std=0.01)

    def _interact(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        vecs : (B, m+1, emb_dim)
        返回  : (B, C(m+1,2))  两两内积
        """
        n = vecs.size(1)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        dots = [
            (vecs[:, i, :] * vecs[:, j, :]).sum(dim=1, keepdim=True)
            for i, j in pairs
        ]
        return torch.cat(dots, dim=1)                # (B, n_interact)

    def forward(self, feat_ids: torch.Tensor,
                dense: torch.Tensor) -> torch.Tensor:
        """
        feat_ids : (B, n_sparse) — 类别特征索引
        dense    : (B, dense_dim) — 稠密数值特征
        """
        # 稠密特征 → 嵌入空间
        e_dense = self.bottom_mlp(dense)             # (B, emb_dim)

        # 类别特征嵌入
        e_sparse = [emb(feat_ids[:, i])
                    for i, emb in enumerate(self.embs)]  # n_sparse × (B, emb_dim)

        # 拼成向量组: (B, m+1, emb_dim)
        vecs = torch.stack([e_dense] + e_sparse, dim=1)

        # 两两内积
        s = self._interact(vecs)                     # (B, n_interact)

        # top_mlp 输入：保留稠密嵌入 + 交互向量
        z = torch.cat([e_dense, s], dim=1)
        return self.top_mlp(z).squeeze(-1)

    def predict_proba(self, feat_ids, dense) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(feat_ids, dense))

    def fit(self, feat_ids, dense, y, epochs: int = 40, lr: float = 1e-2):
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for ep in range(epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                self.forward(feat_ids, dense), y.float())
            loss.backward()
            opt.step()
            if (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")
        return loss.item()

    def rank(self, feat_ids, dense, top_k):
        scores = self.predict_proba(feat_ids, dense).tolist()
        return sorted(enumerate(scores), key=lambda t: -t[1])[:top_k]


# ── 合成数据 ──────────────────────────────────────────────────────────────────

def make_data(n: int = 500, n_sparse: int = 4, vocab: int = 8,
              dense_dim: int = 4, seed: int = 42):
    """
    n_sparse 个类别字段（如 用户ID桶 / 物品类别 / 设备 / 地域），
    dense_dim 个数值字段（如 历史CTR / 时长统计 / 价格 / 评分）。
    标签依赖类别字段的二阶交互 + 数值主效应，这正是 DLRM 内积层的
    建模目标：e_i · e_j 捕捉类别对之间的二阶关系。
    """
    torch.manual_seed(seed)
    random.seed(seed)
    ids = torch.randint(0, vocab, (n, n_sparse), dtype=torch.long)
    dense = torch.randn(n, dense_dim)

    local = ids.float()
    # 类别主效应 + 0×1 二阶交互 + 数值主效应
    logit = (0.3 * local[:, 0] + 0.4 * local[:, 1]
             + 0.9 * (local[:, 0] == local[:, 1]).float()
             + 0.7 * (local[:, 2] == local[:, 3]).float()
             + 0.5 * dense[:, 0] - 0.4 * dense[:, 1] - 2.5)
    y = (torch.sigmoid(logit + 0.3 * torch.randn(n)) > 0.5).long()
    return ids, dense, y


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    N_SPARSE, VOCAB, DENSE_DIM = 4, 8, 4
    ids, dense, y = make_data(500, N_SPARSE, VOCAB, DENSE_DIM)
    split = 380
    ids_tr, d_tr, y_tr = ids[:split], dense[:split], y[:split]
    ids_te, d_te, y_te = ids[split:], dense[split:], y[split:]

    print("=== DLRM 训练（底部 MLP + 类别嵌入 + 两两内积 + 顶部 MLP）===")
    model = DLRM(n_sparse=N_SPARSE, vocab_size=VOCAB, emb_dim=8,
                 dense_dim=DENSE_DIM,
                 bottom_dims=[DENSE_DIM, 16, 8],
                 top_dims=[8 + 10, 32, 1])   # 10 = C(5,2) 对内积
    model.fit(ids_tr, d_tr, y_tr, epochs=40)

    p_te = model.predict_proba(ids_te, d_te).tolist()
    auc_te = _auc(p_te, y_te.tolist())
    print(f"\n测试集 AUC={auc_te:.4f}")

    # ── 交互维度验证 ──────────────────────────────────────────────────────────
    n_vecs = N_SPARSE + 1
    n_interact = n_vecs * (n_vecs - 1) // 2
    print(f"\n特征向量数 m+1={n_vecs}  两两内积维度 C({n_vecs},2)={n_interact}")
    print(f"顶部 MLP 输入维度: emb_dim({model.emb_dim}) + n_interact({n_interact})"
          f" = {model.emb_dim + n_interact}")
    print("  正文（PDF 第 32 页）：顶部 MLP 保留稠密特征表示 e_dense，")
    print("  避免预测过程完全依赖内积交互结果。")

    # ── 与 DeepFM 架构对比 ────────────────────────────────────────────────────
    print(f"\n与 DeepFM 的关键区别（正文 PDF 第 35 页）：")
    print("  DeepFM: FM 二阶项 + DNN 并联，共享嵌入，直接融合两分支得分")
    print("  DLRM : 内积 → 拼接为 top_mlp 输入，而非与 DNN 直接相加")
    print("  DLRM 更侧重控制交互粒度（每对特征压缩为 1 个标量），适合工业大规模部署。")

    # ── 精排推理演示 ──────────────────────────────────────────────────────────
    top5 = model.rank(ids_te[:20], d_te[:20], top_k=5)
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={s:.4f}  label={y_te[idx].item()}")
