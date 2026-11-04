"""
IPA：特征交互的统一框架（第15章 §15.5.10）

IPA 把一个交互模型拆成三个可替换的组件：
    1) Interaction Function  f(t_i, t_j) ∈ R^K
         naive     : t_i ⊙ t_j                       （≈ FM）
         weighted  : w_ij · (t_i ⊙ t_j)              （≈ FwFM，每对一个标量）
         diagonal  : w_ij ⊙ (t_i ⊙ t_j)              （≈ FvFM，每对一个向量）
         projected : (t_i W_ij) ⊙ t_j                （≈ FmFM / DCN-V2）
    2) Layer Pooling         如何从上一层构造高一阶
         field-wise : 每个 Field 保留自己的交互表示（DCN-V2 的 Field Pooling）
         global-wise: 所有 Field 聚成一条表示（xDeepFM 的 CIN 更接近这种）
    3) Layer Aggregator      如何融合不同阶
         direct  : 等权拼接
         layer   : 每层一个标量 α_l
         term    : 每层每个 term 一个标量 α_{l,n}
         element : 每个元素一个 α_{l,n,k}

本脚本把上面的组合真正实例化并逐组对比，输出的是"同一份数据下不同组合"
的 AUC，说明组件选择本身就是设计空间。

运行：python3 ch04_feature_interaction/ipa.py
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _train_utils import (  # noqa: E402
    N_FIELDS, MLP, BaselineMLP, count_params, embedding_table, print_comparison,
    run_experiment, stack_embeddings,
)

DIM = 4


class InteractionFunction(nn.Module):
    """四种 Interaction Function，统一返回 t_i ⊙ (变换后的 t_j) ∈ R^K。"""

    def __init__(self, kind: str, dim: int, n_fields: int):
        super().__init__()
        assert kind in {"naive", "weighted", "diagonal", "projected"}
        self.kind = kind
        self.dim = dim
        n_pairs = n_fields * (n_fields - 1) // 2
        if kind == "weighted":
            self.w = nn.Parameter(torch.ones(n_pairs))
        elif kind == "diagonal":
            self.w = nn.Parameter(torch.ones(n_pairs, dim))
        elif kind == "projected":
            self.w = nn.Parameter(torch.empty(n_pairs, dim, dim))
            nn.init.xavier_uniform_(self.w)
        self.register_buffer(
            "pairs", torch.tensor(list(itertools.combinations(range(n_fields), 2))))

    def forward(self, ei: torch.Tensor, ej: torch.Tensor) -> torch.Tensor:
        # ei, ej: [B, P, d]
        if self.kind == "naive":
            return ei * ej
        if self.kind == "weighted":
            return (ei * ej) * self.w.view(1, -1, 1)
        if self.kind == "diagonal":
            return (ei * ej) * self.w.view(1, self.w.shape[0], self.dim)
        # projected：(e_i W) ⊙ e_j
        return torch.einsum("bpd,pde->bpe", ei, self.w) * ej


class IPA(nn.Module):
    def __init__(self, if_kind: str = "projected", pooling: str = "field",
                 aggregator: str = "layer", dim: int = DIM, n_layers: int = 3,
                 hidden=(32,)):
        super().__init__()
        assert pooling in {"field", "global"}
        assert aggregator in {"direct", "layer", "term", "element"}
        self.if_kind, self.pooling, self.aggregator = if_kind, pooling, aggregator
        self.dim, self.n_layers = dim, n_layers
        self.pairs = list(itertools.combinations(range(N_FIELDS), 2))
        self.n_pairs = len(self.pairs)
        # 用 buffer 存 pair 下标，避免每次 forward 重新建 tensor
        self.register_buffer("pi", torch.tensor([p[0] for p in self.pairs]))
        self.register_buffer("pj", torch.tensor([p[1] for p in self.pairs]))
        # 每一层都用自己的 Interaction Function（参数逐层独立）
        self.fns = nn.ModuleList(
            [InteractionFunction(if_kind, dim, N_FIELDS) for _ in range(n_layers)])
        self.emb = embedding_table(dim)

        # 每层的 term 数：field-wise = Field 数（每个 Field 一个表示）
        #                  global-wise = 1（全部 pair 聚成一条）
        n_terms = N_FIELDS if pooling == "field" else 1
        self.n_terms = n_terms
        if aggregator == "layer":
            self.alpha = nn.Parameter(torch.ones(n_layers))
        elif aggregator == "term":
            self.alpha = nn.Parameter(torch.ones(n_layers, n_terms))
        elif aggregator == "element":
            self.alpha = nn.Parameter(torch.ones(n_layers, n_terms, dim))

        if aggregator == "direct":
            agg_dim = n_layers * n_terms * dim
        else:
            agg_dim = n_terms * dim          # 加权后逐元素相加，维度不变
        self.mlp = MLP(agg_dim, hidden, out_dim=1, dropout=0.2)

    def layer_terms(self, v: torch.Tensor, v0: torch.Tensor, layer_idx: int):
        """
        给定上一层表示 v（[B, M, d]）与原始 Embedding v0，构造一层的 terms。
        做法：先用 Interaction Function 计算所有 pair，再按 pooling 归并。
        """
        fn = self.fns[layer_idx]
        pi = self.pi
        pj = self.pj
        # global pooling 的上一层只剩 1 条表示，需要广播回 M 个位置参与下一轮 pair 计算
        v_left = v if v.shape[1] == N_FIELDS else v.expand(-1, N_FIELDS, -1)
        inter = fn(v_left[:, pi], v0[:, pj])             # [B, P, d]
        if self.pooling == "field":
            # 每个 Field i 聚合所有以 i 为左端的 pair
            out = torch.zeros(v.shape[0], N_FIELDS, self.dim, dtype=inter.dtype)
            out.index_add_(1, pi, inter)
            return out
        return inter.sum(1, keepdim=True)                # [B, 1, d]

    def forward(self, x):
        v0 = stack_embeddings(self.emb, x)
        v = v0
        terms = []
        for l in range(self.n_layers):
            v = self.layer_terms(v, v0, l)
            terms.append(v)
        stacked = torch.stack(terms, dim=1)              # [B, L, n_terms, d]

        if self.aggregator == "direct":
            out = stacked.flatten(1)
        elif self.aggregator == "layer":
            out = (stacked * self.alpha.view(1, -1, 1, 1)).sum(1).flatten(1)
        elif self.aggregator == "term":
            out = (stacked * self.alpha.view(1, self.n_layers, self.n_terms, 1)
                   ).sum(1).flatten(1)
        else:
            out = (stacked * self.alpha.view(1, self.n_layers, self.n_terms, self.dim)
                   ).sum(1).flatten(1)
        return self.mlp(out).squeeze(-1)


def main() -> None:
    print("IPA：Interaction Function × Layer Pooling × Layer Aggregator（§15.5.10）\n")
    print(f"Field 数 M={N_FIELDS}，pair 数={N_FIELDS * (N_FIELDS - 1) // 2}，"
          f"d={DIM}，L=3 层\n")

    print("① Interaction Function 对比（固定 field pooling + layer aggregator）")
    rows = []
    for kind in ("naive", "weighted", "diagonal", "projected"):
        _, auc, _ = run_experiment(lambda k=kind: IPA(if_kind=k), kind, steps=600)
        rows.append((f"IPA-{kind}", auc, count_params(IPA(if_kind=kind))))

    print("\n② Layer Pooling 对比（固定 projected + layer aggregator）")
    for pooling in ("field", "global"):
        _, auc, _ = run_experiment(
            lambda p=pooling: IPA(pooling=p), f"pooling={pooling}", steps=600)
        rows.append((f"IPA-projected / {pooling}", auc,
                     count_params(IPA(pooling=pooling))))

    print("\n③ Layer Aggregator 对比（固定 projected + field pooling）")
    for agg in ("direct", "layer", "term", "element"):
        _, auc, _ = run_experiment(
            lambda a=agg: IPA(aggregator=a), f"aggregator={agg}", steps=600)
        rows.append((f"IPA-projected / field / {agg}", auc,
                     count_params(IPA(aggregator=agg))))

    _, base_auc, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    print_comparison([("Baseline MLP", base_auc, count_params(BaselineMLP()))] + rows)

    n = 3
    print(f"\nAggregator 参数量（L={n}, M={N_FIELDS}, K={DIM}）：")
    print(f"  direct  : 0")
    print(f"  layer   : L        = {n}")
    print(f"  term    : L×M      = {n * N_FIELDS}")
    print(f"  element : L×M×K    = {n * N_FIELDS * DIM}")
    print("\nIPA 的贡献不是又一个新算子，而是把'如何交互 / 如何构造高阶 / 如何聚合'")
    print("拆成三个正交维度：同一个框架下能拼出 FM、DCN-V2、xDeepFM 等已有模型，")
    print("上面三组对比就是在'设计空间'里逐维扫了一遍（§15.5.10）。")


if __name__ == "__main__":
    main()
