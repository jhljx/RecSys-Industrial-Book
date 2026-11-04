"""
FiBiNet：SENET 特征重加权 + Bilinear 二阶交互（第15章 §15.5.4）

SENET：
    z_i = (1/k) Σ_t e_i^(t)                 # squeeze：每个 field 压成一个标量
    A = σ(W_2 σ(W_1 Z))                    # excitation：field 权重
    V = [a_i e_i]                          # re-weight
Bilinear（三种共享方式）：
    p_ij = (e_i^T W) ⊙ e_j
    - Field-All：所有 pair 共享 W
    - Field-Each：每个 field 一个 W_i
    - Field-Interaction：每个 pair 一个 W_ij（参数量最大）
两条通路 E→P、E→V→Q 拼接后过 DNN。

运行：python3 ch04_feature_interaction/fibinet.py
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _train_utils import (  # noqa: E402
    FIELD_NAMES, N_FIELDS, MLP, BaselineMLP, count_params, embedding_table, print_comparison,
    run_experiment, stack_embeddings,
)


class SENET(nn.Module):
    def __init__(self, n_fields: int, reduction: int = 2):
        super().__init__()
        self.w1 = nn.Linear(n_fields, max(1, n_fields // reduction))
        self.w2 = nn.Linear(max(1, n_fields // reduction), n_fields)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B, F, d]
        z = E.mean(dim=2)                       # squeeze -> [B, F]
        a = torch.sigmoid(self.w2(torch.relu(self.w1(z))))  # excitation
        return E * a.unsqueeze(-1)              # re-weight


class BilinearInteraction(nn.Module):
    """mode: all / each / interaction"""

    def __init__(self, n_fields: int, dim: int, mode: str = "all"):
        super().__init__()
        self.mode = mode
        if mode == "all":
            self.w = nn.Parameter(torch.empty(dim, dim))
            nn.init.xavier_uniform_(self.w)
        elif mode == "each":
            self.w = nn.Parameter(torch.empty(n_fields, dim, dim))
            nn.init.xavier_uniform_(self.w)
        elif mode == "interaction":
            self.w = nn.Parameter(torch.empty(n_fields, n_fields, dim, dim))
            nn.init.xavier_uniform_(self.w)
        else:
            raise ValueError(mode)
        self.register_buffer("pairs", torch.tensor(
            list(itertools.combinations(range(n_fields), 2)), dtype=torch.long))

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B, F, d] -> 所有 i<j 的 (e_i^T W) ⊙ e_j
        B, F, d = E.shape
        pi, pj = self.pairs[:, 0], self.pairs[:, 1]
        ei, ej = E[:, pi], E[:, pj]              # [B, P, d]
        if self.mode == "all":
            x = ei @ self.w                     # [B, P, d]
        elif self.mode == "each":
            # 每对使用 e_i 自己的 W_i
            x = torch.einsum("bpd,pde->bpe", ei, self.w[pi])
        else:
            x = torch.einsum("bpd,pqde->bpe", ei, self.w[pi, pj])
        return x * ej                           # Hadamard


class FiBiNet(nn.Module):
    def __init__(self, dim: int = 4, mode: str = "all", hidden=(32,)):
        super().__init__()
        self.emb = embedding_table(dim)
        self.senet = SENET(N_FIELDS)
        self.bi = BilinearInteraction(N_FIELDS, dim, mode)   # 两条通路共享交互参数
        n_pairs = N_FIELDS * (N_FIELDS - 1) // 2
        self.mlp = MLP(2 * n_pairs * dim, hidden, out_dim=1, dropout=0.2)

    def forward(self, x):
        E = stack_embeddings(self.emb, x)     # [B, F, d]
        V = self.senet(E)
        P = self.bi(E).flatten(1)             # 原始通路
        Q = self.bi(V).flatten(1)             # 重加权通路
        return self.mlp(torch.cat([P, Q], dim=1)).squeeze(-1)


def main() -> None:
    print("FiBiNet：SENET 重加权 + Bilinear 二阶交互（§15.5.4）\n")
    n_pairs = N_FIELDS * (N_FIELDS - 1) // 2
    d = 4
    print(f"Bilinear 参数量（F={N_FIELDS}, d={d}, pair 数={n_pairs}）：")
    print(f"  Field-All         : {d * d:,}")
    print(f"  Field-Each        : {N_FIELDS * d * d:,}")
    print(f"  Field-Interaction : {N_FIELDS * N_FIELDS * d * d:,}（是 All 的 {N_FIELDS ** 2} 倍）\n")

    _, a_all, _ = run_experiment(lambda: FiBiNet(mode="all"), "Field-All", steps=600)
    _, a_each, _ = run_experiment(lambda: FiBiNet(mode="each"), "Field-Each", steps=600)
    _, a_base, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    print_comparison([
        ("Baseline MLP", a_base, count_params(BaselineMLP())),
        ("FiBiNet Field-All", a_all, count_params(FiBiNet(mode="all"))),
        ("FiBiNet Field-Each", a_each, count_params(FiBiNet(mode="each"))),
    ])
    print("\n参数量从 All 到 Each 涨了 F 倍，AUC 却不一定同步上涨——")
    print("这就是正文'表达能力与成本权衡'的具体表现（§15.5.4）。")


if __name__ == "__main__":
    main()
