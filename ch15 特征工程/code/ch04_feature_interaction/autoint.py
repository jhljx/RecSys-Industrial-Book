"""
AutoInt：基于 Self-Attention 的自动特征交互（第15章 §15.5.3）

把每个 field 的 Embedding 当成一个 token：
    Q = E W_Q, K = E W_K, V = E W_V
    A = softmax(Q K^T / sqrt(d_k)),  H = A V
堆叠多头注意力，最后把各 field 的表示展平送入 MLP。相比 DCN 的固定交叉，
Attention 的权重随样本变化，是"动态特征交互"的代表。

运行：python3 ch04_feature_interaction/autoint.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _train_utils import (  # noqa: E402
    N_FIELDS, MLP, BaselineMLP, count_params, embedding_table, print_comparison,
    run_experiment, stack_embeddings,
)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_fields: int, dim: int, n_heads: int = 2):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B, F, d]
        B, F, d = E.shape
        def split(t):
            return t.view(B, F, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split(self.wq(E)), split(self.wk(E)), split(self.wv(E))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d_head), dim=-1)
        h = (att @ v).transpose(1, 2).reshape(B, F, d)
        return self.wo(h)


class AutoInt(nn.Module):
    def __init__(self, dim: int = 4, n_blocks: int = 1, n_heads: int = 2, hidden=(32,)):
        super().__init__()
        self.emb = embedding_table(dim)
        self.blocks = nn.ModuleList([MultiHeadSelfAttention(N_FIELDS, dim, n_heads)
                                     for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_blocks)])
        self.mlp = MLP(dim * N_FIELDS, hidden, out_dim=1, dropout=0.2)

    def forward(self, x):
        E = stack_embeddings(self.emb, x)
        for block, norm in zip(self.blocks, self.norms):
            E = norm(E + block(E))   # 残差 + LayerNorm
        return self.mlp(E.flatten(1)).squeeze(-1)

    def attention_map(self, x):
        """返回第一个 block 的平均注意力矩阵，用于观察'谁和谁交互'。"""
        E = stack_embeddings(self.emb, x)
        block = self.blocks[0]
        B, F, d = E.shape
        q = block.wq(E).view(B, F, block.n_heads, block.d_head).transpose(1, 2)
        k = block.wk(E).view(B, F, block.n_heads, block.d_head).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(block.d_head), dim=-1)
        return att.mean(1).mean(0)


def main() -> None:
    print("AutoInt：A = softmax(QK^T/√d_k)，H = AV（§15.5.3）\n")
    model, auc, _ = run_experiment(lambda: AutoInt(), "AutoInt", steps=600)
    _, base_auc, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    # 观察某一个 batch 的注意力图
    from _train_utils import FIELD_NAMES, make_dataset
    X, _ = make_dataset(64, seed=3)
    with torch.no_grad():
        att = model.attention_map(torch.from_numpy(X).long())
    print("\nAutoInt 第一个 block 的平均注意力（行=query field，列=key field）：")
    print("        " + "  ".join(f"{n[:4]:>4}" for n in FIELD_NAMES))
    for i, name in enumerate(FIELD_NAMES):
        row = "  ".join(f"{att[i, j].item():.2f}" for j in range(N_FIELDS))
        print(f"{name[:8]:>8}  " + row)
    print("\n  对角线较大表示 field 保留自身信息；非对角元越大，说明该 field 越依赖其他 field。")
    print("  权重随样本变化正是 Attention 与固定 Cross 结构的关键区别。")

    print_comparison([
        ("Baseline MLP", base_auc, count_params(BaselineMLP())),
        ("AutoInt（1 block, 2 heads）", auc, count_params(AutoInt())),
    ])


if __name__ == "__main__":
    main()
