"""
DSSM + FM 粗排模型（第11章 §11.2.2 DSSM + FM模型）

核心思想：
传统双塔（DSSM）用户塔与物品塔完全独立，只能依赖 Embedding 空间的隐式语义
相似性完成匹配，缺乏显式的跨侧特征交互。DSSM+FM 在双塔主网络之外增加一个
FM 辅助分支，对少量重要离散 ID 特征（用户ID、设备ID、物品ID、作者ID 等）
做显式二阶交叉：
    s_DSSM = <e_u, e_i>                          （双塔语义匹配分）
    s_FM   = w0 + Σ_i w_i·x_i + Σ_i Σ_{j>i} <v_i, v_j>·x_i·x_j
    s      = s_DSSM + s_FM
其中二阶项用「和平方 − 平方和」技巧把复杂度从 O(k·n²) 降到 O(k·n)：
    Σ_i Σ_{j>i} <v_i,v_j> x_i x_j
        = 0.5 · Σ_f [ (Σ_i v_{i,f}·x_i)² − Σ_i (v_{i,f}·x_i)² ]

工程意义：物品塔 Embedding 仍可离线预计算，双塔的可预计算效率被完整保留；
FM 分支只涉及少量 ID 的 O(k·n) 计算，几乎不增加在线时延，却补回了跨侧特征
交叉能力。工业实践中 FM 隐向量常与 DSSM 主网络 Embedding 共享参数。

本实现：PyTorch autograd，FM 二阶交叉用 (sum_of_embeddings)^2 - sum_of_squared_embeddings
实现 O(k·n)；FM 隐向量与双塔 embedding 共享同一 nn.Embedding 对象。
保留 with_fm vs without_fm 对比演示。
"""

import math
import random
from typing import List, Tuple, Optional

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


class FMLayer(nn.Module):
    """
    FM 二阶交叉层：对若干 field 的 Embedding（隐向量 v_i）做二阶交叉。

    采用「和平方 − 平方和」O(k·n) 实现：
        s_2nd = 0.5 · Σ_f [ (Σ_i v_{i,f})² − Σ_i v_{i,f}² ]
    这里离散 ID 特征取值 x_i = 1，因此 v_i·x_i 退化为 v_i。
    """

    def __init__(self, n_fields: int, k: int):
        super().__init__()
        self.n_fields = n_fields
        self.k = k
        self.w0 = nn.Parameter(torch.zeros(1))                 # 全局偏置
        self.w = nn.Parameter(torch.zeros(n_fields))            # 一阶权重 w_i

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """
        输入：fields = [n_fields, k] 的隐向量堆叠（FM 隐向量与主网络共享）。
        返回：s_FM（标量）。
        """
        # 一阶项：Σ w_i · Σ_f v_{i,f}（x_i = 1，用隐向量元素和代表激活强度）
        first = self.w0 + (self.w * fields.sum(dim=1)).sum()
        # 二阶项：和平方 − 平方和
        sum_vec = fields.sum(dim=0)          # [k]
        square_of_sum = sum_vec.pow(2).sum()
        sum_of_square = fields.pow(2).sum()
        second = 0.5 * (square_of_sum - sum_of_square)
        return first + second


class DSSMFM(nn.Module):
    """
    DSSM + FM 粗排模型。

    双塔主网络：user_emb → 用户塔 → e_u；[item_emb, author_emb] → 物品塔 → e_i
    FM 辅助网络：对 4 个重要 ID field（用户ID / 设备ID / 物品ID / 作者ID）
                 的 Embedding 做二阶交叉，隐向量与主网络 Embedding 参数共享。
    最终打分：s = s_DSSM + s_FM
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_authors: int,
        n_devices: int,
        emb_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 16,
        fm_weight: float = 1.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        self.output_dim = output_dim
        self.fm_weight = fm_weight

        # 这 4 张 Embedding 表同时服务 DSSM 主网络与 FM 分支（参数共享）
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.author_emb = nn.Embedding(n_authors, emb_dim)
        self.device_emb = nn.Embedding(n_devices, emb_dim)

        self.user_tower = TowerMLP([emb_dim * 2] + hidden_dims + [output_dim])
        self.item_tower = TowerMLP([emb_dim * 2] + hidden_dims + [output_dim])
        self.fm = FMLayer(n_fields=4, k=emb_dim)

        scale = math.sqrt(1.0 / emb_dim)
        for emb in (self.user_emb, self.item_emb, self.author_emb, self.device_emb):
            nn.init.uniform_(emb.weight, -scale, scale)

    # ── 双塔主网络 ──────────────────────────────────────────────────────────
    def encode_user(self, uid: torch.Tensor, did: torch.Tensor) -> torch.Tensor:
        """e_u = f_u([用户ID Emb, 设备ID Emb])"""
        x = torch.cat([self.user_emb(uid), self.device_emb(did)], dim=-1)
        return self.user_tower(x)

    def encode_item(self, iid: torch.Tensor, aid: torch.Tensor) -> torch.Tensor:
        """e_i = f_i([物品ID Emb, 作者ID Emb])，可离线预计算。"""
        x = torch.cat([self.item_emb(iid), self.author_emb(aid)], dim=-1)
        return self.item_tower(x)

    # ── FM 辅助网络 ────────────────────────────────────────────────────────
    def _fm_fields(self, uid, did, iid, aid) -> torch.Tensor:
        """取出 4 个重要 ID field 的隐向量堆叠 [n_fields, k]（与主网络共享 Embedding）。"""
        return torch.stack([
            self.user_emb(uid).squeeze(),
            self.device_emb(did).squeeze(),
            self.item_emb(iid).squeeze(),
            self.author_emb(aid).squeeze(),
        ], dim=0)

    def forward(self, uid, did, iid, aid, y) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        交叉熵损失 + 分量统计。
        Returns: (loss, s_dssm_abs_mean, s_fm_abs_mean)
        """
        e_u = self.encode_user(uid, did)   # [B, D]
        e_i = self.encode_item(iid, aid)    # [B, D]
        s_dssm = (e_u * e_i).sum(dim=-1)   # [B] 双塔内积

        # FM 分支：逐样本计算（fields 为 [n_fields, k]）
        s_fm_list = []
        for b in range(uid.size(0)):
            fields = self._fm_fields(uid[b:b+1], did[b:b+1], iid[b:b+1], aid[b:b+1])
            s_fm_list.append(self.fm(fields))
        s_fm = torch.stack(s_fm_list).squeeze(-1)  # [B]

        s = s_dssm + self.fm_weight * s_fm
        loss = F.binary_cross_entropy_with_logits(s, y)
        return loss, s_dssm.abs().mean(), (self.fm_weight * s_fm).abs().mean()

    def pre_rank(
        self,
        uid: int,
        candidates: List[int],
        top_k: int = 20,
        device_id: int = 0,
        item_author: Optional[List[int]] = None,
        with_fm: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        粗排：用户塔在线算一次，物品塔 Embedding 可离线预计算；
        FM 分支只有 O(k·n) 的少量计算，可按需开关。
        """
        item_author = item_author or [0] * (max(candidates) + 1)
        with torch.no_grad():
            u = torch.tensor([uid])
            d = torch.tensor([device_id])
            e_u = self.encode_user(u, d)  # [1, D]
            cand_t = torch.tensor(candidates)
            aid_t = torch.tensor([item_author[i] for i in candidates])
            e_i = self.encode_item(cand_t, aid_t)  # [N, D]
            s = (e_u * e_i).sum(dim=-1)            # [N]
            if with_fm:
                fm_scores = []
                for idx, iid in enumerate(candidates):
                    fields = self._fm_fields(
                        u, d, cand_t[idx:idx+1], aid_t[idx:idx+1],
                    )
                    fm_scores.append(self.fm(fields))
                s = s + self.fm_weight * torch.stack(fm_scores).squeeze(-1)
            scores = [(candidates[j], s[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def train(
    model: DSSMFM,
    samples: List[Tuple[int, int, int, int, float]],
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 0.05,
) -> None:
    """samples: [(uid, did, iid, aid, y), ...]，采用交叉熵损失。"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = list(samples)
    for epoch in range(epochs):
        random.shuffle(data)
        acc_loss = acc_dssm = acc_fm = 0.0
        n = 0
        for start in range(0, len(data), batch_size):
            batch = data[start: start + batch_size]
            uids = torch.tensor([s[0] for s in batch])
            dids = torch.tensor([s[1] for s in batch])
            iids = torch.tensor([s[2] for s in batch])
            aids = torch.tensor([s[3] for s in batch])
            ys = torch.tensor([s[4] for s in batch], dtype=torch.float)

            optimizer.zero_grad()
            loss, dssm_m, fm_m = model(uids, dids, iids, aids, ys)
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()
            acc_dssm += dssm_m.item()
            acc_fm += fm_m.item()
            n += 1
        nn_ = max(n, 1)
        print(f"  epoch {epoch:2d}  loss={acc_loss / nn_:.4f}  "
              f"|s_DSSM|={acc_dssm / nn_:.4f}  |s_FM|={acc_fm / nn_:.4f}")


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items, n_authors, n_devices = 8, 24, 6, 3
    item_author = [i % n_authors for i in range(n_items)]
    user_device = [u % n_devices for u in range(n_users)]

    # 合成偏好：用户只偏好「与自己 ID 同余」的作者 → 典型的跨侧交叉模式
    samples: List[Tuple[int, int, int, int, float]] = []
    for uid in range(n_users):
        fav_author = uid % n_authors
        for iid in range(n_items):
            aid = item_author[iid]
            y = 1.0 if aid == fav_author else 0.0
            if y > 0 or random.random() < 0.25:
                samples.append((uid, user_device[uid], iid, aid, y))

    model = DSSMFM(n_users, n_items, n_authors, n_devices, emb_dim=16,
                   hidden_dims=[32, 16], output_dim=16, fm_weight=1.0)
    print(f"DSSM+FM 粗排模型训练（样本数 {len(samples)}）：")
    train(model, samples, epochs=6, batch_size=16, lr=0.05)

    uid = 0
    cands = list(range(n_items))
    print(f"\n用户 {uid}（偏好作者 {uid % n_authors}）粗排 Top-5（双塔 + FM）：")
    for iid, s in model.pre_rank(uid, cands, top_k=5, device_id=user_device[uid],
                                 item_author=item_author, with_fm=True):
        print(f"  item {iid:2d} (作者 {item_author[iid]}): {s:.4f}")

    print("\n对照：仅用双塔分数（物品 Embedding 全部可离线预计算）Top-5：")
    for iid, s in model.pre_rank(uid, cands, top_k=5, device_id=user_device[uid],
                                 item_author=item_author, with_fm=False):
        print(f"  item {iid:2d} (作者 {item_author[iid]}): {s:.4f}")

    print("\n结论：FM 二阶交叉补回了跨侧特征交互能力（O(k·n) 计算量），"
          "\n      而物品塔 Embedding 仍可离线预计算，双塔的在线效率没有损失。")
