"""
DCN-V2（第15章 §15.5.2）

把 DCN 的向量参数化改成矩阵：
    x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l
参数量从 O(Ld) 升到 O(Ld²)，表达能力更强；工业上常用低秩分解
    W_l ≈ U_l V_l^T，U_l, V_l ∈ R^{d×r}
在效果与显存之间折中。

运行：python3 ch04_feature_interaction/dcn_v2.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _train_utils import (  # noqa: E402
    N_FIELDS, MLP, BaselineMLP, count_params, embedding_table, print_comparison,
    run_experiment, stack_embeddings,
)


class MatrixCrossLayer(nn.Module):
    """
    三种参数化方式：
        rank=None  : 满矩阵 W ∈ R^{d×d}
        rank=r>0   : 低秩 W = U V，U∈R^{d×r}, V∈R^{r×d}
        rank=0     : 退化为 DCN 的向量形式 x0 ⊙ (w ⊙ xl)，即 W = w 的对角矩阵
    rank=0 的分支单独实现，避免创建 0 维 Linear 触发 PyTorch 的空张量警告。
    """

    def __init__(self, dim: int, rank: int | None = None):
        super().__init__()
        self.rank = rank
        if rank is None:
            self.w = nn.Linear(dim, dim, bias=True)
        elif rank == 0:
            self.vec = nn.Parameter(torch.empty(dim))
            nn.init.normal_(self.vec, std=1e-3)
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.u = nn.Linear(dim, rank, bias=False)
            self.v = nn.Linear(rank, dim, bias=True)

    def forward(self, x0, xl):
        if self.rank is None:
            wx = self.w(xl)
        elif self.rank == 0:
            wx = (xl * self.vec).sum(1, keepdim=True).expand_as(xl) + self.bias
        else:
            wx = self.v(self.u(xl))
        return x0 * wx + xl


class DCNV2(nn.Module):
    def __init__(self, dim: int = 4, n_cross: int = 2, rank: int | None = None, hidden=(32,)):
        super().__init__()
        self.emb = embedding_table(dim)
        d = dim * N_FIELDS
        self.cross = nn.ModuleList([MatrixCrossLayer(d, rank) for _ in range(n_cross)])
        self.deep = MLP(d, hidden, out_dim=None, dropout=0.2)
        self.head = nn.Linear(d + hidden[-1], 1)

    def forward(self, x):
        x0 = stack_embeddings(self.emb, x).flatten(1)
        xl = x0
        for layer in self.cross:
            xl = layer(x0, xl)
        h = self.deep(x0)
        return self.head(torch.cat([xl, h], dim=1)).squeeze(-1)


def main() -> None:
    d = 4 * N_FIELDS
    print(f"输入维度 d = {d}（10 个 field × 4 维）\n")
    print("Cross 层参数量随参数化方式的变化（2 层）：")
    print(f"  DCN 向量版      : {2 * 2 * d:,}")
    print(f"  DCN-V2 满矩阵   : {2 * d * d + 2 * d:,}（{d}x{d} 权重 + bias）")
    print(f"  DCN-V2 低秩 r=8 : {2 * (d * 8 + 8 * d + d):,}")
    print("  → 满矩阵是向量版的 d 倍；d=80 时还能承受，d=1024 时就会 OOM。\n")

    _, a_vec, _ = run_experiment(lambda: DCNV2(rank=0), "DCN 向量版(r=0)", steps=200)
    _, a_full, _ = run_experiment(lambda: DCNV2(rank=None), "DCN-V2-full", steps=200)
    _, a_base, _ = run_experiment(BaselineMLP, "Baseline", steps=200)
    _, a_lr, _ = run_experiment(lambda: DCNV2(rank=8), "DCN-V2-lowrank(r=8)", steps=200)

    print_comparison([
        ("Baseline MLP", a_base, count_params(BaselineMLP())),
        ("DCN 向量版（r=0 特例）", a_vec, count_params(DCNV2(rank=0))),
        ("DCN-V2 低秩 r=8", a_lr, count_params(DCNV2(rank=8))),
        ("DCN-V2 满矩阵", a_full, count_params(DCNV2(rank=None))),
    ])
    print("\n读法：满矩阵参数量最多，但不一定显著更好——正文强调 DCN-V2 的工业价值"
          "在于'把表达能力与成本一起考虑'（§15.5.2）。")


if __name__ == "__main__":
    main()
