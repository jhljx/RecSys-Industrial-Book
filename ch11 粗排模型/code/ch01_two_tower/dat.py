"""
DAT 双塔增强粗排模型（第11章 §11.1.4 DAT模型）

论文出处：Yu et al., "A Dual Augmented Two-tower Model for Online Large-scale
Recommendation", DLP-KDD / CIKM 2021（美团）。

核心思想：
传统双塔模型中用户塔看不到物品信息、物品塔看不到用户信息，两塔之间缺乏交互。
DAT 为每个用户和每个物品各配一个可学习的「增强向量」（Augmented Vector）
a_u / a_v，拼接进本侧塔的输入，让单塔能隐式地「看到」对侧的历史交互语义：
    z_u = [e_u, a_u],  z_v = [e_v, a_v]
    p_u = L2Norm(MLP_u(z_u)),  p_v = L2Norm(MLP_v(z_v))
    s(u, v) = <p_u, p_v>

三层训练目标（§11.1.4 的 L = L_p + λ1·L_u + λ2·L_v + λ3·L_CA）：
1) L_p  主损失：二分类交叉熵，监督匹配准确性；
2) L_u / L_v  自适应模仿机制（Adaptive-Mimic Mechanism, AMM）：
       L_u = (1/T)·Σ y·‖a_u − sg(p_v)‖²,   L_v = (1/T)·Σ y·‖a_v − sg(p_u)‖²
   只在正样本（y=1）上拟合，sg(·) 为 Stop Gradient，PyTorch autograd 用 .detach() 实现；
3) L_CA 类别对齐损失（Category Alignment Loss）：对齐不同类别 item embedding 的
   二阶统计量（均值差 + 协方差差），缓解类别数据分布不均衡：
       L_CA = Σ_i ‖C(S_major) − C(S_i)‖_F²

本实现：PyTorch autograd，三项损失联合训练（超参 α, β 固定为合理默认值）。
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
    """双塔 MLP：多层线性 + ReLU，最后一层无激活。"""

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


class DAT(nn.Module):
    """
    DAT（Dual Augmented Two-tower）粗排模型。

    结构：user_emb + a_u → 用户塔 → L2Norm → p_u
         item_emb + a_v → 物品塔 → L2Norm → p_v
         s(u, v) = <p_u, p_v>
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        item_category: List[int],
        emb_dim: int = 16,
        aug_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 16,
        lambda_u: float = 0.3,
        lambda_v: float = 0.3,
        lambda_ca: float = 0.05,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        # 增强向量维度必须等于对侧塔输出维度，模仿损失才能逐元素做差
        assert aug_dim == output_dim, "增强向量维度需与塔输出维度一致（模仿损失要求）"
        self.n_items = n_items
        self.output_dim = output_dim
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_ca = lambda_ca

        # 注册 item_category 为 buffer，方便随模型一起保存/移动设备
        self.register_buffer("item_category", torch.tensor(item_category, dtype=torch.long))

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # (a) 双塔增强结构：每个用户/物品各一个可学习增强向量
        self.aug_u = nn.Embedding(n_users, aug_dim)
        self.aug_v = nn.Embedding(n_items, aug_dim)

        dims = [emb_dim + aug_dim] + hidden_dims + [output_dim]
        self.user_tower = TowerMLP(dims)
        self.item_tower = TowerMLP(dims)

        # 初始化 Embedding
        scale = math.sqrt(1.0 / emb_dim)
        nn.init.uniform_(self.user_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)
        nn.init.zeros_(self.aug_u.weight)
        nn.init.zeros_(self.aug_v.weight)

    def encode_user(self, uid: torch.Tensor) -> torch.Tensor:
        """z_u = [e_u, a_u] → MLP → L2Norm  →  p_u  [B, D]"""
        z = torch.cat([self.user_emb(uid), self.aug_u(uid)], dim=-1)
        h = self.user_tower(z)
        return F.normalize(h, p=2, dim=-1)

    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        """z_v = [e_v, a_v] → MLP → L2Norm  →  p_v  [B, D]"""
        z = torch.cat([self.item_emb(iid), self.aug_v(iid)], dim=-1)
        h = self.item_tower(z)
        return F.normalize(h, p=2, dim=-1)

    # ── (b) 自适应模仿损失 ────────────────────────────────────────────────────
    def _mimic_loss(
        self,
        uid: torch.Tensor,
        iid: torch.Tensor,
        y: torch.Tensor,
        p_u: torch.Tensor,
        p_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_u = Σ y·‖a_u − sg(p_v)‖²,  L_v = Σ y·‖a_v − sg(p_u)‖²
        只在正样本上拟合；监督目标走 stop_gradient（.detach()），
        梯度只更新增强向量 aug_u / aug_v 本身。
        """
        mask = (y > 0.5).float()  # [B]
        z_v = p_v.detach()        # stop_gradient
        z_u = p_u.detach()        # stop_gradient
        a_u = self.aug_u(uid)
        a_v = self.aug_v(iid)
        loss_u = ((a_u - z_v).pow(2).sum(dim=-1) * mask).sum()
        loss_v = ((a_v - z_u).pow(2).sum(dim=-1) * mask).sum()
        n = mask.sum().clamp(min=1.0)
        return (loss_u + loss_v) / n

    # ── (c) 类别对齐损失 ──────────────────────────────────────────────────────
    def _category_alignment_loss(
        self,
        pos_iids: torch.Tensor,
        p_v_pos: torch.Tensor,
        cov_dim: int = 6,
    ) -> torch.Tensor:
        """
        L_CA = Σ_{i≥2} ‖C(S_major) − C(S_i)‖_F²
        C(S) = (1/|S|)·Σ (p − μ)(p − μ)^T 为类别内表示的协方差矩阵。
        对齐不同类别 item embedding 的二阶统计量（均值差 + 协方差差）。
        """
        cats = self.item_category[pos_iids]  # [M]
        unique_cats = cats.unique()
        reprs: Dict[int, torch.Tensor] = {}
        for c in unique_cats.tolist():
            mask = cats == c
            if mask.sum() >= 2:
                reprs[c] = p_v_pos[mask, :cov_dim]  # [k, cov_dim]

        if len(reprs) < 2:
            return torch.tensor(0.0)

        major = max(reprs, key=lambda c: reprs[c].size(0))

        def cov(r: torch.Tensor) -> torch.Tensor:
            mu = r.mean(dim=0, keepdim=True)
            diff = r - mu
            return diff.t().mm(diff) / r.size(0)  # [cov_dim, cov_dim]

        C_major = cov(reprs[major])
        total = torch.tensor(0.0)
        for c, r in reprs.items():
            if c == major:
                continue
            total = total + (C_major - cov(r)).pow(2).sum()
        return total

    # ── (d) 总训练目标 ────────────────────────────────────────────────────────
    def forward(
        self,
        uid: torch.Tensor,
        iid: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        L = L_p + λ1·L_u + λ2·L_v + λ3·L_CA
        Returns: (total_loss, L_p, L_mimic, L_CA)
        """
        p_u = self.encode_user(uid)   # [B, D]
        p_v = self.encode_item(iid)   # [B, D]

        # L_p：二分类交叉熵
        logits = (p_u * p_v).sum(dim=-1)  # [B]
        loss_p = F.binary_cross_entropy_with_logits(logits, y)

        # L_mimic：AMM 损失（b）
        loss_mimic = self._mimic_loss(uid, iid, y, p_u, p_v)

        # L_CA：类别对齐损失（c）—— 只在正样本物品上计算
        pos_mask = y > 0.5
        if pos_mask.any():
            loss_ca = self._category_alignment_loss(iid[pos_mask], p_v[pos_mask])
        else:
            loss_ca = torch.tensor(0.0)

        total = loss_p + (self.lambda_u + self.lambda_v) * loss_mimic + self.lambda_ca * loss_ca
        return total, loss_p, loss_mimic, loss_ca

    def pre_rank(
        self, uid: int, candidates: List[int], top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """粗排：p_v 可离线预计算，线上仅算 p_u 与内积。"""
        with torch.no_grad():
            u = torch.tensor([uid])
            p_u = self.encode_user(u)  # [1, D]
            cand_t = torch.tensor(candidates)
            p_v = self.encode_item(cand_t)  # [N, D]
            scores_t = (p_u * p_v).sum(dim=-1)  # [N]
            scores = [(candidates[j], scores_t[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def train(
    model: DAT,
    samples: List[Tuple[int, int, float]],
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 0.05,
) -> None:
    """L = L_p + λ1·L_u + λ2·L_v + λ3·L_CA"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = list(samples)
    for epoch in range(epochs):
        random.shuffle(data)
        acc_p = acc_m = acc_ca = acc_total = 0.0
        n_batch = 0
        for start in range(0, len(data), batch_size):
            batch = data[start: start + batch_size]
            if not batch:
                continue
            uids = torch.tensor([s[0] for s in batch])
            iids = torch.tensor([s[1] for s in batch])
            ys = torch.tensor([s[2] for s in batch], dtype=torch.float)

            optimizer.zero_grad()
            total, lp, lm, lca = model(uids, iids, ys)
            total.backward()
            optimizer.step()

            acc_total += total.item()
            acc_p += lp.item()
            acc_m += lm.item()
            acc_ca += lca.item() if isinstance(lca, torch.Tensor) else lca
            n_batch += 1

        nb = max(n_batch, 1)
        print(
            f"  epoch {epoch:2d}  L_total={acc_total / nb:.4f}  L_p={acc_p / nb:.4f}  "
            f"L_mimic={acc_m / nb:.4f}  L_CA={acc_ca / nb:.4f}"
        )


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items = 8, 24
    # 类别分布故意不均衡：类别 0 为主类别（长尾类别样本很少）→ 触发 CAL
    item_category = [0 if i < 14 else (1 if i < 20 else 2) for i in range(n_items)]

    samples: List[Tuple[int, int, float]] = []
    for uid in range(n_users):
        pos = {(uid * 3 + k) % n_items for k in range(4)}
        for iid in pos:
            samples.append((uid, iid, 1.0))
        for _ in range(4):
            neg = random.randint(0, n_items - 1)
            if neg not in pos:
                samples.append((uid, neg, 0.0))

    model = DAT(
        n_users, n_items, item_category,
        emb_dim=16, aug_dim=16, hidden_dims=[32, 16], output_dim=16,
        lambda_u=0.3, lambda_v=0.3, lambda_ca=0.05,
    )
    print("DAT 模型训练（主损失 + 模仿损失 + 类别对齐损失）：")
    train(model, samples, epochs=6, batch_size=16, lr=0.05)

    uid = 0
    seen = {iid for u, iid, y in samples if u == uid and y > 0}
    candidates = [i for i in range(n_items) if i not in seen]
    print(f"\n用户 {uid} 粗排 Top-5：")
    for iid, s in model.pre_rank(uid, candidates, top_k=5):
        print(f"  item {iid:2d} (类别 {item_category[iid]}): {s:.4f}")

    # 演示 L_mimic 效果：增强向量 a_u 的模长反映了 AMM 拟合程度
    with torch.no_grad():
        a_u_norm = model.aug_u(torch.tensor([uid])).norm().item()
    print(
        f"\n用户 {uid} 增强向量 a_u 模长 = {a_u_norm:.4f}"
        f"（由 AMM 拟合正样本物品表示 sg(p_v) 得到）"
    )
