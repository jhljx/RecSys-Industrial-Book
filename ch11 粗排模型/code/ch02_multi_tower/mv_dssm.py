"""
MV-DSSM 多视角多塔粗排模型（第11章 §11.2.1 MV-DSSM模型）

论文出处：Elkahky et al., "A Multi-View Deep Learning Approach for Cross Domain
User Modeling in Recommendation Systems", WWW 2015（微软）。

核心思想：
原始 DSSM 只能建模「一类用户 vs 一类物品」的匹配关系。MV-DSSM 提出
「单用户塔、多物品塔」的多视角（Multi-View）结构：
    e_u          = f_u(x_u)                （所有业务域共享的用户塔）
    e_i^(m)      = f_i^(m)(x_i^(m))        （第 m 个业务域独立的物品塔）
    s_m(u, i)    = cos(e_u, e_i^(m))
训练时每条样本只激活其所属域的物品塔，但所有域的监督信号都会共同优化共享
用户塔，从而实现跨域知识迁移（Knowledge Transfer）：数据量大的域帮助数据量
小的新域学到更鲁棒的用户兴趣表示。

本实现：PyTorch autograd，三个业务域（新闻 / App / 视频），每域各自的物品
Embedding 空间与物品塔维度可以不同（支持异构输入特征）。
一个共享用户塔 + 多个域独立 item 塔（nn.ModuleDict 存各域 item tower）。
演示同一 u_emb 在三个域各出 Top-3。
"""

import math
import random
import collections
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TowerMLP(nn.Module):
    """通用 MLP 塔：多层线性 + ReLU，最后一层无激活。"""

    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class ItemView(nn.Module):
    """单个业务域（视角）：独立的物品 Embedding 表 + 独立的物品塔。"""

    def __init__(
        self,
        name: str,
        n_items: int,
        emb_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ):
        super().__init__()
        self.name = name
        self.n_items = n_items
        # 各域物品塔参数完全独立（输入维度可不同 → 支持异构特征）
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.tower = TowerMLP([emb_dim] + hidden_dims + [output_dim])
        scale = math.sqrt(1.0 / emb_dim)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

    def encode(self, iid: torch.Tensor) -> torch.Tensor:
        return self.tower(self.item_emb(iid))


class MVDSSM(nn.Module):
    """
    MV-DSSM 模型：一个共享用户塔 + 多个业务域独立物品塔（nn.ModuleDict）。

    所有域的物品塔输出维度必须与用户塔输出维度一致，才能做余弦匹配。
    """

    def __init__(
        self,
        n_users: int,
        view_specs: List[Tuple[str, int, int]],
        emb_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 16,
        temperature: float = 0.1,
    ):
        """view_specs: [(域名, 物品数, 该域物品 Embedding 维度), ...]"""
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        self.output_dim = output_dim
        self.temperature = temperature

        self.user_emb = nn.Embedding(n_users, emb_dim)
        # 共享用户塔：所有域共用一套参数
        self.user_tower = TowerMLP([emb_dim] + hidden_dims + [output_dim])
        # nn.ModuleDict 存各域 item tower
        self.views = nn.ModuleDict({
            name: ItemView(name, n_items, v_emb, hidden_dims, output_dim)
            for name, n_items, v_emb in view_specs
        })

        scale = math.sqrt(1.0 / emb_dim)
        nn.init.uniform_(self.user_emb.weight, -scale, scale)

    def encode_user(self, uid: torch.Tensor) -> torch.Tensor:
        """e_u = f_u(x_u)，跨域共享，一次计算可服务所有域。"""
        return self.user_tower(self.user_emb(uid))

    def forward(
        self,
        uid: torch.Tensor,
        view_name: str,
        pos_iid: torch.Tensor,
        neg_iids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sampled Softmax：只激活当前域的物品塔，
        但共享用户塔会被所有域的样本共同更新 → 跨域知识迁移。
        pos_iid: [B], neg_iids: [B, n_neg]
        """
        view: ItemView = self.views[view_name]
        e_u = self.encode_user(uid)                         # [B, D]
        e_pos = view.encode(pos_iid)                        # [B, D]
        e_neg = view.encode(neg_iids.view(-1)).view(         # [B, n_neg, D]
            neg_iids.size(0), neg_iids.size(1), -1
        )
        # logits: pos + negs → [B, 1+n_neg]
        s_pos = F.cosine_similarity(e_u, e_pos, dim=1) / self.temperature   # [B]
        s_neg = (
            F.cosine_similarity(
                e_u.unsqueeze(1).expand_as(e_neg), e_neg, dim=2
            ) / self.temperature
        )  # [B, n_neg]
        logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)  # [B, 1+n_neg]
        labels = torch.zeros(logits.size(0), dtype=torch.long)  # 第 0 列为正样本
        return F.cross_entropy(logits, labels)

    def pre_rank(
        self,
        uid: int,
        candidates: List[int],
        top_k: int = 20,
        view: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """粗排：指定业务域对候选集打分并返回 Top-K。默认取第一个域。"""
        view = view or next(iter(self.views))
        v: ItemView = self.views[view]
        with torch.no_grad():
            u = torch.tensor([uid])
            e_u = self.encode_user(u)                      # [1, D]
            cand_t = torch.tensor(candidates)
            e_i = v.encode(cand_t)                         # [N, D]
            sims = F.cosine_similarity(e_u, e_i, dim=1)   # [N]
            scores = [(candidates[j], sims[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def train(
    model: MVDSSM,
    samples: List[Tuple[int, str, int]],
    epochs: int = 6,
    n_neg: int = 4,
    lr: float = 0.05,
) -> None:
    """samples: [(uid, 域名, 正样本 item_id), ...]；多域样本交替进行 SGD。"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = list(samples)

    # 按域分组，方便快速采样
    view_items: Dict[str, int] = {
        name: v.n_items for name, v in model.views.items()
    }

    for epoch in range(epochs):
        random.shuffle(data)
        per_view_loss: Dict[str, List[float]] = collections.defaultdict(list)

        # 按域 mini-batch 处理（每个样本单独一个 step，保持与原逻辑一致）
        for uid, view_name, pos_iid in data:
            v_size = view_items[view_name]
            # 采 n_neg 个随机负样本（不严格去重，保持效率）
            neg_ids = [random.randrange(v_size) for _ in range(n_neg)]

            uid_t = torch.tensor([uid])
            pos_t = torch.tensor([pos_iid])
            neg_t = torch.tensor([neg_ids])  # [1, n_neg]

            optimizer.zero_grad()
            loss = model(uid_t, view_name, pos_t, neg_t)
            loss.backward()
            optimizer.step()
            per_view_loss[view_name].append(loss.item())

        total = sum(sum(v) for v in per_view_loss.values()) / max(len(data), 1)
        detail = "  ".join(
            f"{name}={sum(ls) / len(ls):.4f}(n={len(ls)})"
            for name, ls in sorted(per_view_loss.items())
        )
        print(f"  epoch {epoch:2d}  avg_loss={total:.4f}  |  {detail}")


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users = 8
    # 三个业务域：新闻样本多，视频域为数据稀疏的新业务 → 依赖跨域迁移
    view_specs = [("news", 24, 16), ("app", 18, 12), ("video", 20, 20)]
    sample_cnt = {"news": 48, "app": 24, "video": 10}

    samples: List[Tuple[int, str, int]] = []
    for name, n_items, _ in view_specs:
        for _ in range(sample_cnt[name]):
            samples.append((random.randrange(n_users), name, random.randrange(n_items)))

    model = MVDSSM(n_users, view_specs, emb_dim=16, hidden_dims=[32, 16],
                   output_dim=16, temperature=0.1)
    print("MV-DSSM 多视角训练（共享用户塔 + 3 个独立物品塔）：")
    train(model, samples, epochs=6, n_neg=4, lr=0.05)

    uid = 0
    print(f"\n同一用户表示 e_u 在多个业务域下分别粗排（用户 {uid}）：")
    for name, n_items, _ in view_specs:
        res = model.pre_rank(uid, list(range(n_items)), top_k=3, view=name)
        pretty = ", ".join(f"item {i}:{s:.4f}" for i, s in res)
        print(f"  [{name:5s}] Top-3 → {pretty}")

    with torch.no_grad():
        e_u = model.encode_user(torch.tensor([uid]))
        norm = e_u.norm().item()
    print(f"\n共享用户 Embedding 模长 = {norm:.4f}"
          f"（一次计算，复用于全部 {len(view_specs)} 个域的粗排）")
