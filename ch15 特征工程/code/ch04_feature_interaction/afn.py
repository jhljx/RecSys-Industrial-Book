"""
AFN：自适应阶数的对数神经元（第15章 §15.5.5）

对数神经元：
    y = exp(Σ_i w_i ln x_i) = Π_i x_i^{w_i}
对第 j 个神经元、第 i 个 field：
    z_j = exp(Σ_i w_ij ln(e_i)) = Π_i e_i^{w_ij}
    O_j = Σ_i |w_ij|            # 该神经元学到的有效交互阶数

AFN 的关键是"阶数本身是参数"，不需要预先指定 2 阶还是 3 阶。Embedding 可能
为负，工程上先取绝对值并加 eps（正文提到的"数值处理"），并在前后加 BN。

运行：python3 ch04_feature_interaction/afn.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _train_utils import (  # noqa: E402
    FIELD_NAMES, N_FIELDS, MLP, BaselineMLP, count_params, embedding_table, print_comparison,
    run_experiment, stack_embeddings,
)


class LogTransformationLayer(nn.Module):
    """J 个对数神经元，每个向量维度各自做一次 Π e_i^{w_ij}。"""

    def __init__(self, n_fields: int, dim: int, n_neurons: int = 12):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_fields, n_neurons))
        # 初始化到 0 附近：起步接近"不交互"，让模型自己长阶数
        nn.init.normal_(self.w, std=0.1)
        self.bn_in = nn.BatchNorm1d(n_fields * dim)
        self.bn_out = nn.BatchNorm1d(n_neurons * dim)
        self.dim = dim
        self.n_neurons = n_neurons

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: [B, F, d]
        B, F, d = E.shape
        e = self.bn_in(E.flatten(1)).view(B, F, d)
        log_e = torch.log(e.abs() + 1e-3)              # [B, F, d]
        # Σ_i w_ij ln(e_i)  -> [B, J, d]
        z = torch.einsum("bfd,fj->bjd", log_e, self.w)
        out = torch.exp(torch.clamp(z, -6, 6))         # 防溢出
        return self.bn_out(out.flatten(1)).view(B, self.n_neurons, d)

    def orders(self) -> list[float]:
        return [float(v) for v in self.w.abs().sum(0)]


class AFN(nn.Module):
    def __init__(self, dim: int = 4, n_neurons: int = 12, hidden=(32,)):
        super().__init__()
        self.emb = embedding_table(dim)
        self.log_layer = LogTransformationLayer(N_FIELDS, dim, n_neurons)
        self.mlp = MLP(n_neurons * dim, hidden, out_dim=1, dropout=0.2)

    def forward(self, x):
        z = self.log_layer(stack_embeddings(self.emb, x))
        return self.mlp(z.flatten(1)).squeeze(-1)


def main() -> None:
    print("AFN：z_j = exp(Σ_i w_ij ln e_i)，阶数 O_j = Σ_i |w_ij|（§15.5.5）\n")
    model, auc, _ = run_experiment(lambda: AFN(), "AFN", steps=600)
    _, base_auc, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    with torch.no_grad():
        orders = model.log_layer.orders()
    print("\n各对数神经元学到的有效交互阶数 O_j：")
    for j, o in enumerate(orders):
        bar = "█" * max(1, int(o * 8))
        print(f"  neuron {j:2d}: O={o:5.2f}  {bar}")
    print(f"\n  阶数分布范围 {min(orders):.2f} ~ {max(orders):.2f}："
          "模型自己选择了从近 0 阶（几乎不交互）到多阶的混合，")
    print("  这就是'自适应阶数'——不需要人工指定最大交互阶数。")

    print_comparison([
        ("Baseline MLP", base_auc, count_params(BaselineMLP())),
        ("AFN（12 神经元）", auc, count_params(AFN())),
    ])
    print("\n注：本实现按正文对 Embedding 取绝对值加 eps 再取对数（§15.5.5 的'数值处理'）。")


if __name__ == "__main__":
    main()
