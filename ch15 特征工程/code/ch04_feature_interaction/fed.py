"""
FED：Dimension Relation Module + Field-wise + Element-wise（第15章 §15.5.6）

DRM（维度级）：
    U = V^T
    S = softmax((U W_θ)(U W_φ)^T)
    E = (W_δ V) S + V
Field-wise（Field 级）：
    H = softmax((E W_θ^1)(E W_φ^1)^T)
    F = H (E W_δ^1) + E
Element-wise：把 E 展平送 DNN。
最终 x = [x_dim; x_field; x_element]，Logistic 输出。

运行：python3 ch04_feature_interaction/fed.py
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


class DRM(nn.Module):
    """
    注意维度的对齐：正文写作 E = (W_δ V) S + V，其中 V∈R^{m×d}、S∈R^{d×d}。
    直接相乘维度不匹配（m×d 与 d×d），实现上把 V 的 d 维向量先做一次线性变换，
    再让 d×d 的注意力矩阵 S 在维度轴上做混合：
        E = [S (V W_δ)ᵀ]ᵀ + V
    这与"在 Embedding 维度空间里做一次自注意力聚合"是同一件事，只是把 W_δ 放到
    转置后的一侧，保证 S 作用在 d 这个轴上。Field-wise 模块则让同形的 H 作用在
    Field 轴上，两者分别对应正文里的"维度级"与"Field 级"关系建模。
    """

    def __init__(self, n_fields: int, dim: int):
        super().__init__()
        self.w_theta = nn.Linear(n_fields, n_fields, bias=False)
        self.w_phi = nn.Linear(n_fields, n_fields, bias=False)
        self.w_delta = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        # V: [B, m, d]；U: [B, d, m]
        U = V.transpose(1, 2)
        S = torch.softmax((self.w_theta(U) @ self.w_phi(U).transpose(1, 2)), dim=-1)  # [B, d, d]
        # 对每个 field 的 d 维向量做 W_δ 变换，再让 S 在 d 轴上混合
        Vd = self.w_delta(V)                          # [B, m, d]
        E = (S @ Vd.transpose(1, 2)).transpose(1, 2) + V   # [B, d, m] -> [B, m, d]
        return self.norm(E)


class FieldWise(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w_theta = nn.Linear(dim, dim, bias=False)
        self.w_phi = nn.Linear(dim, dim, bias=False)
        self.w_delta = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        H = torch.softmax(self.w_theta(E) @ self.w_phi(E).transpose(1, 2), dim=-1)
        F = torch.bmm(H, self.w_delta(E)) + E   # [B,m,m] @ [B,m,d]
        return self.norm(F)


class FED(nn.Module):
    def __init__(self, dim: int = 4, hidden=(32,)):
        super().__init__()
        self.emb = embedding_table(dim)
        self.drm = DRM(N_FIELDS, dim)
        self.field = FieldWise(dim)
        self.element_mlp = MLP(N_FIELDS * dim, hidden, out_dim=1, dropout=0.2)
        # x_dim + x_field + x_element
        self.head = nn.Linear(2 * N_FIELDS * dim + 1, 1)

    def forward(self, x):
        V = stack_embeddings(self.emb, x)       # [B, m, d]
        E = self.drm(V)
        F = self.field(E)
        x_dim = E.flatten(1)
        x_field = F.flatten(1)
        x_elem = self.element_mlp(E.flatten(1))
        return self.head(torch.cat([x_dim, x_field, x_elem], dim=1)).squeeze(-1)


def main() -> None:
    print("FED：DRM(维度级) + Field-wise + Element-wise（§15.5.6）\n")
    _, auc, _ = run_experiment(lambda: FED(), "FED", steps=600)
    _, base_auc, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    d, m = 4, N_FIELDS
    print(f"\n三个子模块各自的参数量（m={m}, d={d}）：")
    print(f"  DRM        : 2 × m × m + d × d = {2 * m * m + d * d}"
          "（W_θ, W_φ 作用在维度轴上；W_δ 为 d×d）")
    print(f"  Field-wise : 3 × d × d = {3 * d * d}（W_θ^1, W_φ^1, W_δ^1，都是 d×d）")
    print(f"  Element    : MLP({m * d} → 32)")
    print("  DRM 的注意力作用在 Embedding 维度上，Field-wise 作用在原始 Field 上；")
    print("  前者挖掘的是隐式维度关系，后者挖掘的是显式特征域关系（§15.5.6）。")

    print_comparison([
        ("Baseline MLP", base_auc, count_params(BaselineMLP())),
        ("FED", auc, count_params(FED())),
    ])


if __name__ == "__main__":
    main()
