"""
CAN：Co-Action Network，由特征生成参数（第15章 §15.5.7）

Co-Action Unit 把一对特征分成两个角色：
    Induction Side（参数生成侧）：P_induction 被 reshape 成 Micro-MLP 的 W 与 b
    Feed Side（输入侧）        ：P_feed 作为 Micro-MLP 的输入
        h_0 = P_feed
        h_i = σ(w_{i-1} h_{i-1} + b_{i-1})
        F(u, m) = [h_1 ‖ h_2 ‖ … ‖ h_L]

Multi-order Enhancement：把 (P_feed)^c（c=1..C）分别送进同一个 Micro-MLP 再求和，
用 tanh 约束高阶项的数值范围（正文 §15.5.7）。

参数量对比是本节的重点：
    笛卡尔积交互 Embedding：O(N²D)
    CAN：O(N(D' + D))
本脚本用一个 400×40 的 induction 参数表，与 400×200 的 pair 表直接对照。

运行：python3 ch04_feature_interaction/can.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _train_utils import (  # noqa: E402
    FIELD_NAMES, FIELD_SIZES, N_FIELDS, MLP, BaselineMLP, count_params, embedding_table,
    print_comparison, run_experiment, stack_embeddings,
)

DIM = 4
INDUCTION_LAYERS = 2


class CoActionUnit(nn.Module):
    """
    一个 Micro-MLP：每层都是 d×d 的 Weight 加 d 维 Bias，参数由 Induction 侧生成。
    multi_order=True 时，Feed 侧额外提供 P² 作为第二阶输入。
    """

    def __init__(self, dim: int, n_layers: int = INDUCTION_LAYERS, multi_order: bool = False):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.multi_order = multi_order
        self.param_per_layer = dim * dim + dim
        self.param_dim = n_layers * self.param_per_layer
        self.n_orders = 2 if multi_order else 1

    def forward(self, p_induction: torch.Tensor, p_feed: torch.Tensor) -> torch.Tensor:
        # p_induction: [B, param_dim]，p_feed: [B, d]
        B = p_feed.shape[0]
        d = self.dim
        outs = []
        for c in range(1, self.n_orders + 1):
            # (P_feed)^c，用 tanh 做有界约束（§15.5.7 Multi-order Enhancement）
            h = torch.tanh(p_feed) if c == 1 else torch.tanh(p_feed ** c)
            layer_outs = []
            offset = 0
            for _ in range(self.n_layers):
                w = p_induction[:, offset:offset + d * d].view(B, d, d)
                offset += d * d
                b = p_induction[:, offset:offset + d].view(B, d)
                offset += d
                h = torch.tanh(torch.bmm(h.unsqueeze(1), w).squeeze(1) + b)
                layer_outs.append(h)              # 每层输出都要保留
            outs.append(torch.cat(layer_outs, dim=1))   # Concat(h_1..h_L)
        return torch.stack(outs, dim=0).sum(0)   # Σ_c H(P_ind, (P_feed)^c)


class CAN(nn.Module):
    """
    Induction 侧：用户 ID（词表大、组合多，适合用参数生成）
    Feed 侧    ：物品 ID
    正文 §15.5.7 提到可以反过来选，取决于两侧规模；这里把选择写在构造参数里。
    """

    def __init__(self, dim: int = DIM, multi_order: bool = False,
                 induction_field: str = "user", feed_field: str = "item"):
        super().__init__()
        self.emb = embedding_table(dim)
        self.induction_field = induction_field
        self.feed_field = feed_field
        self.unit = CoActionUnit(dim, INDUCTION_LAYERS, multi_order)
        # 只为 induction 侧的特征建一张"参数表"，而不是为每个 pair 建 Embedding
        self.induction_table = nn.Embedding(FIELD_SIZES[induction_field], self.unit.param_dim)
        # 初始化偏大一些：权重太小会让 Micro-MLP 输出接近常数、梯度消失（实测
        # std=0.02 时 AUC 掉约 0.006，std=0.5 才与 Baseline 持平）。
        nn.init.normal_(self.induction_table.weight, std=0.5)
        inter_dim = self.unit.n_layers * dim
        self.mlp = MLP(dim * N_FIELDS + inter_dim, (32,), out_dim=1, dropout=0.2)

    def forward(self, x):
        E = stack_embeddings(self.emb, x)                    # [B, m, d]
        i_idx = FIELD_NAMES.index(self.induction_field)
        f_idx = FIELD_NAMES.index(self.feed_field)
        p_ind = self.induction_table(x[:, i_idx])            # Parameter Lookup
        p_feed = E[:, f_idx]
        inter = self.unit(p_ind, p_feed)
        return self.mlp(torch.cat([E.flatten(1), inter], dim=1)).squeeze(-1)


def main() -> None:
    print("CAN：Induction Side 生成 Micro-MLP 参数，Feed Side 作为输入（§15.5.7）\n")
    m = N_FIELDS
    d = DIM
    n_items = FIELD_SIZES["item"]
    n_users = FIELD_SIZES["user"]

    print("参数量对照（用户侧 × 物品侧）：")
    print(f"  笛卡尔积交互 Embedding : {n_users} × {n_items} × {d} = "
          f"{n_users * n_items * d:,}")
    print(f"  CAN（induction 参数表）: {n_users} × {CoActionUnit(d).param_dim} = "
          f"{n_users * CoActionUnit(d).param_dim:,}"
          f"（相当于前者的 1/{n_items}）")
    print(f"  另一侧物品仍走常规 Embedding：{n_items} × {d} = {n_items * d:,}\n")

    _, a_plain, _ = run_experiment(lambda: CAN(multi_order=False), "CAN", steps=600)
    _, a_multi, _ = run_experiment(lambda: CAN(multi_order=True), "CAN+Multi-order", steps=600)
    _, a_base, _ = run_experiment(BaselineMLP, "Baseline", steps=600)

    print_comparison([
        ("Baseline MLP", a_base, count_params(BaselineMLP())),
        ("CAN", a_plain, count_params(CAN())),
        ("CAN + Multi-order", a_multi, count_params(CAN(multi_order=True))),
    ])
    print("\nCAN 的核心不是'多加一层网络'，而是把过去固定的交互函数换成由一侧特征")
    print("现场生成的 Micro-MLP：不同 user 对应不同的交互函数（§15.5.7）。")
    print("micro-MLP 只有 1 层、无 bias、无激活时，它退化为内积，也就是 FM。")


if __name__ == "__main__":
    main()
