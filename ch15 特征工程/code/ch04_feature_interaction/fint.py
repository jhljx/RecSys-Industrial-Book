"""
FINT：Field-aware Interaction Network（第15章 §15.5.8）

与把多 Field 压成一个向量的做法不同，FINT 为每个 Field 单独维护交互向量：
    v_i^l = Σ_j w_{i,j}^l (v_i^{l-1} ⊙ v_j^0) + u_i^l v_i^{l-1}
- j 遍历所有 Field，与原始 Embedding v_j^0 做 Hadamard 积
- w_{i,j}^l 是 Field 级权重（M×M，每层一份）
- u_i^l 是残差标量
堆叠 K 层后，把 M 个 Field 的最终表示拼接送 DNN。

每多一层，就多引入一次与 v^0 的 Hadamard 积，因此第 l 层天然包含 l+1 阶交互。

运行：python3 ch04_feature_interaction/fint.py
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


class FieldAwareInteractionLayer(nn.Module):
    def __init__(self, n_fields: int, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_fields, n_fields))
        nn.init.xavier_uniform_(self.w)
        self.u = nn.Parameter(torch.zeros(n_fields))

    def forward(self, v_prev: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        # v_prev / v0: [B, M, d]
        had = v_prev.unsqueeze(2) * v0.unsqueeze(1)        # [B, M(i), M(j), d]
        agg = torch.einsum("ij,bijd->bid", self.w, had)    # Σ_j w_ij (v_i ⊙ v_j^0)
        return agg + self.u.view(1, -1, 1) * v_prev


class FINT(nn.Module):
    def __init__(self, dim: int = 4, n_layers: int = 2, hidden=(32,)):
        super().__init__()
        self.emb = embedding_table(dim)
        self.layers = nn.ModuleList(
            [FieldAwareInteractionLayer(N_FIELDS, dim) for _ in range(n_layers)])
        self.mlp = MLP(N_FIELDS * dim, hidden, out_dim=1, dropout=0.2)

    def forward(self, x):
        v0 = stack_embeddings(self.emb, x)
        v = v0
        for layer in self.layers:
            v = layer(v, v0)
        return self.mlp(v.flatten(1)).squeeze(-1)


def main() -> None:
    print("FINT：v_i^l = Σ_j w_ij (v_i^{l-1} ⊙ v_j^0) + u_i v_i^{l-1}（§15.5.8）\n")
    print(f"Field 级权重矩阵：M×M = {N_FIELDS}×{N_FIELDS}，每层一份；"
          f"2 层共 {2 * N_FIELDS * N_FIELDS} 个权重 + {2 * N_FIELDS} 个残差标量。")
    print("注意参数量与 Field 数平方相关，而不是与交叉组合数指数相关——这是它能")
    print("在保留 Field 边界的同时堆深的前提。\n")

    _, a1, _ = run_experiment(lambda: FINT(n_layers=1), "FINT-1层", steps=600)
    _, a2, _ = run_experiment(lambda: FINT(n_layers=2), "FINT-2层", steps=600)
    _, a3, _ = run_experiment(lambda: FINT(n_layers=3), "FINT-3层", steps=600)
    _, a_base, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    print_comparison([
        ("Baseline MLP", a_base, count_params(BaselineMLP())),
        ("FINT（1 层）", a1, count_params(FINT(n_layers=1))),
        ("FINT（2 层）", a2, count_params(FINT(n_layers=2))),
        ("FINT（3 层）", a3, count_params(FINT(n_layers=3))),
    ])
    print("\n层数增加对应交互阶数增加（1 层≈二阶，2 层≈三阶，3 层≈四阶），")
    print("但增益会随数据量收敛；参数量只随层数线性增长，这是 Field-aware 设计的好处。")


if __name__ == "__main__":
    main()
