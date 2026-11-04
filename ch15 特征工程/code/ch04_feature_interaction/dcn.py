"""
DCN：Deep & Cross Network（第15章 §15.5.1）

Cross Layer（正文式 15.x）：
    x_{l+1} = x_0 x_l^T w_l + b_l + x_l
其中 x_0 是 Embedding 拼接后的输入，w_l、b_l 是 d 维向量。每层只需 O(d) 参数，
堆 L 层可显式构造到 L+1 阶交互；Deep 分支用 MLP 学隐式交互，最后拼接输出。

运行：python3 ch04_feature_interaction/dcn.py
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


class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        # 小初始化，避免深层交叉数值爆炸
        nn.init.normal_(self.w, std=1e-3)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0 [B,d], xl [B,d] -> [B,1] 再广播
        scalar = (xl * self.w).sum(dim=1, keepdim=True)  # x_l^T w
        return x0 * scalar + self.b + xl


class DCN(nn.Module):
    def __init__(self, dim: int = 4, n_cross: int = 2, hidden=(32,)):
        super().__init__()
        self.emb = embedding_table(dim)
        d = dim * N_FIELDS
        self.cross = nn.ModuleList([CrossLayer(d) for _ in range(n_cross)])
        self.deep = MLP(d, hidden, out_dim=None, dropout=0.2)
        self.head = nn.Linear(d + hidden[-1], 1)

    def forward(self, x):
        x0 = stack_embeddings(self.emb, x).flatten(1)
        xl = x0
        for layer in self.cross:
            xl = layer(x0, xl)
        h_deep = self.deep(x0)
        return self.head(torch.cat([xl, h_deep], dim=1)).squeeze(-1)


def main() -> None:
    print("DCN：x_{l+1} = x_0 (x_l^T w_l) + b_l + x_l\n")
    _, auc, _ = run_experiment(lambda: DCN(), "DCN", steps=600)
    # 参数量拆解
    d = 4 * N_FIELDS
    print(f"\nCross 部分参数量：{2 * 2 * d}（2 层 × (w {d} + b {d})），"
          f"与输入维度 d={d} 线性相关；"
          "这正是正文'参数量 O(Ld)、适合工业级'的由来（§15.5.1）。")

    # 与同维度纯 MLP 对照
    _, base_auc, _ = run_experiment(BaselineMLP, "Baseline", steps=600)
    print_comparison([
        ("Baseline MLP", base_auc, count_params(BaselineMLP())),
        ("DCN（2 层 Cross）", auc, count_params(DCN())),
    ])


if __name__ == "__main__":
    main()
