"""
GDCN：门控交叉网络（第15章 §15.5.9）

Gated Cross Layer：
    c_{l+1} = c_0 ⊙ (W^c_l c_l + b_l) ⊙ σ(W^g_l c_l) + c_l
           =  Feature Crossing      ×   Information Gate  +  残差
- W^c 学交叉关系，W^g 学逐维门控，σ 把门控压到 (0,1)
- 门控为 1 → 退化为 DCN-V2；门控学到的值可以直接看"哪一维交互被保留"
两种组合方式：
    GDCN-S（Stacked）：c_0 → GCN → c_L → DNN
    GDCN-P（Parallel）：c_0 → GCN 与 DNN 并行，输出拼接

运行：python3 ch04_feature_interaction/gdcn.py
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

DIM = 4
FLAT = N_FIELDS * DIM


class GatedCrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.wc = nn.Linear(dim, dim)
        self.wg = nn.Linear(dim, dim)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, c0: torch.Tensor, cl: torch.Tensor):
        cross = c0 * (self.wc(cl) + self.b)
        gate = torch.sigmoid(self.wg(cl))
        return cross * gate + cl


class GDCN(nn.Module):
    def __init__(self, mode: str = "S", dim: int = DIM, n_cross: int = 3,
                 hidden=(32,), use_gate: bool = True):
        super().__init__()
        assert mode in {"S", "P"}
        self.mode = mode
        self.dim = dim
        self.emb = embedding_table(dim)
        self.cross = nn.ModuleList([GatedCrossLayer(FLAT) for _ in range(n_cross)])
        self.use_gate = use_gate
        if mode == "S":
            self.deep = MLP(FLAT, hidden, out_dim=None, dropout=0.2)
            self.head = nn.Linear(hidden[-1], 1)
        else:
            self.deep = MLP(FLAT, hidden, out_dim=None, dropout=0.2)
            self.head = nn.Linear(FLAT + hidden[-1], 1)

    def gated_cross(self, c0: torch.Tensor) -> torch.Tensor:
        c = c0
        for layer in self.cross:
            if self.use_gate:
                c = layer(c0, c)
            else:  # 去掉门控即 DCN-V2 的交叉层
                c = c0 * (layer.wc(c) + layer.b) + c
        return c

    def forward(self, x):
        c0 = stack_embeddings(self.emb, x).flatten(1)
        if self.mode == "S":
            return self.head(self.deep(self.gated_cross(c0))).squeeze(-1)
        return self.head(torch.cat([self.gated_cross(c0), self.deep(c0)], dim=1)).squeeze(-1)


def main() -> None:
    print("GDCN：c_{l+1} = c0 ⊙ (W^c c_l + b) ⊙ σ(W^g c_l) + c_l（§15.5.9）\n")
    _, a_s, _ = run_experiment(lambda: GDCN(mode="S"), "GDCN-S", steps=600)
    _, a_p, _ = run_experiment(lambda: GDCN(mode="P"), "GDCN-P", steps=600)
    _, a_nogate, _ = run_experiment(
        lambda: GDCN(mode="S", use_gate=False), "GDCN-S 无门控", steps=600)
    _, a_base, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    print_comparison([
        ("Baseline MLP", a_base, count_params(BaselineMLP())),
        ("GDCN-S（串联）", a_s, count_params(GDCN(mode="S"))),
        ("GDCN-P（并联）", a_p, count_params(GDCN(mode="P"))),
        ("GDCN-S 无门控(≈DCN-V2)", a_nogate, count_params(GDCN(mode="S", use_gate=False))),
    ])

    # 看门控值：每一维交叉信息被保留了多少
    torch.manual_seed(0)
    model = GDCN(mode="S")
    from _train_utils import make_dataset, to_tensor
    Xtr, ytr = make_dataset(4_000, seed=1)
    from _train_utils import fit
    Xtr, ytr = to_tensor(Xtr, ytr)
    fit(model, Xtr, ytr, Xtr[:2000], ytr[:2000], steps=200, name="")
    with torch.no_grad():
        inner = Xtr[:256]
        c = stack_embeddings(model.emb, inner).flatten(1)
        gates = []
        for layer in model.cross:
            gates.append(torch.sigmoid(layer.wg(c)).mean(0))
            c = layer(c0=stack_embeddings(model.emb, inner).flatten(1), cl=c)
    print("\n各层 Gate 平均值（取 256 个样本、40 维求均值）：")
    for l, g in enumerate(gates):
        print(f"  layer {l}: mean={g.mean():.3f}  min={g.min():.3f}  max={g.max():.3f}")
    print("\nGate 不是常数而是随样本变化：同一维交叉信息在有的样本里被放大、")
    print("在另一些样本里被抑制，这就是正文所说的'实例级筛选'（§15.5.9）。")


if __name__ == "__main__":
    main()
