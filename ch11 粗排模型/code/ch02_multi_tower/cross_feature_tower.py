"""
带交叉特征的多塔粗排模型（第11章 §11.2.4 带交叉特征的多塔模型）

核心思想：
传统双塔（DSSM）用户塔与物品塔完全独立，只能依赖 Embedding 空间的隐式语义
相似度匹配，无法表达「用户对该物品类目的历史点击率」这类具有明确业务含义的
用户-物品组合特征。因此工业界在双塔之外再加一个轻量的交叉特征塔
（Cross Interaction Tower），构成「用户塔 + 物品塔 + 交叉塔」三塔结构：

    e_user  = f_user(x_user)                 （深网络，在线算一次）
    e_item  = f_item(x_item)                 （深网络，可离线预计算）
    h_cross = f_cross(x_user, x_item)        （1~2 层轻量网络，只能在线实时算）
    s       = <e_user, e_item> + b_cross,   b_cross = w^T·h_cross

交叉塔只吃交叉特征，输出一个偏置分 b_cross 加到双塔余弦分上。

工程折中（本文件的重点）：
交叉特征同时依赖用户侧和物品侧，无法离线预计算，每个 (user, item) Pair 都要在线
前向一次。候选集规模为 N 时在线开销是 O(N) 次交叉塔前向，粗排动辄上万候选难以
承受。因此工业实践采用两阶段打分（Two-stage Scoring）：
    阶段一：用纯双塔内积给全部 N 个候选打分（物品 Embedding 已预计算，近乎免费），
            截断出 Top-M（M << N）；
    阶段二：只对这 M 个候选计算交叉塔，得到最终融合分数并重排。
这样交叉塔的在线开销从 O(N) 降到 O(M)，既补回了显式交互能力，又守住了粗排的
毫秒级时延预算。

本实现：PyTorch autograd，三个塔端到端联合优化。
"""

import math
import random
import collections
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 交叉特征库 ────────────────────────────────────────────────────────────────

class CrossFeatureStore:
    """
    用户-物品交叉特征库：模拟线上实时拼接的统计类交叉特征。

    本实现提供 4 维交叉特征（工业实践中通常是经过筛选的几十到数百维）：
      x0: 用户对该物品类目的历史点击率（CTR by category）
      x1: 用户对该物品作者的历史点击次数（归一化）
      x2: 用户地域与物品地域是否匹配（0/1）
      x3: 用户活跃度 × 物品热度的匹配统计分
    这些特征依赖 (user, item) 两侧，因此无法离线预计算。
    """

    CROSS_DIM = 4

    def __init__(self, n_users: int, n_items: int, item_category: List[int],
                 item_author: List[int], n_categories: int):
        self.item_category = item_category
        self.item_author = item_author
        # 用户 × 类目 的历史点击率统计表（离线可算，但与物品 join 后才成为交叉特征）
        self.user_cate_ctr = [[random.uniform(0.0, 1.0) for _ in range(n_categories)]
                              for _ in range(n_users)]
        self.user_author_cnt: dict = collections.defaultdict(float)
        self.user_region = [random.randint(0, 2) for _ in range(n_users)]
        self.item_region = [random.randint(0, 2) for _ in range(n_items)]
        self.user_active = [random.uniform(0.2, 1.0) for _ in range(n_users)]
        self.item_hot = [random.uniform(0.2, 1.0) for _ in range(n_items)]

    def lookup(self, uid: int, iid: int) -> List[float]:
        """在线实时拼接 (uid, iid) 的交叉特征向量，维度固定为 CROSS_DIM。"""
        cate, author = self.item_category[iid], self.item_author[iid]
        x = [
            self.user_cate_ctr[uid][cate],
            min(1.0, self.user_author_cnt[(uid, author)] / 5.0),
            1.0 if self.user_region[uid] == self.item_region[iid] else 0.0,
            self.user_active[uid] * self.item_hot[iid],
        ]
        assert len(x) == self.CROSS_DIM
        return x

    def lookup_batch(self, uids: List[int], iids: List[int]) -> torch.Tensor:
        """批量查表，返回 [B, CROSS_DIM] 张量。"""
        rows = [self.lookup(u, i) for u, i in zip(uids, iids)]
        return torch.tensor(rows, dtype=torch.float)

    def observe_click(self, uid: int, iid: int) -> None:
        """累积用户对作者的历史点击次数（线上由实时统计服务维护）。"""
        self.user_author_cnt[(uid, self.item_author[iid])] += 1.0


# ── 模型 ──────────────────────────────────────────────────────────────────────

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


class CrossFeatureTowerModel(nn.Module):
    """
    「双塔 + 轻量交叉塔」三塔粗排模型。

    s(u, i) = cos(e_user, e_item) + b_cross(u, i)
    其中 b_cross = f_cross(x_cross)，f_cross 为 1 层隐藏层的轻量网络（输出 1 维）。
    用户塔 / 物品塔较深（可预计算物品侧），交叉塔刻意做轻（只能在线算）。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        store: CrossFeatureStore,
        emb_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 16,
        cross_hidden: int = 8,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        self.store = store
        self.output_dim = output_dim
        self.last_cross_calls = 0  # 统计上一次 pre_rank 的交叉塔在线调用次数

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        dims = [emb_dim] + hidden_dims + [output_dim]
        self.user_tower = TowerMLP(dims)   # 深塔：每次请求在线算一次
        self.item_tower = TowerMLP(dims)   # 深塔：可离线预计算并缓存
        # 轻量交叉塔：CROSS_DIM → cross_hidden → 1（直接输出偏置分）
        self.cross_tower = TowerMLP([store.CROSS_DIM, cross_hidden, 1])

        scale = math.sqrt(1.0 / emb_dim)
        nn.init.uniform_(self.user_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

    def encode_user(self, uid: torch.Tensor) -> torch.Tensor:
        return self.user_tower(self.user_emb(uid))

    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        return self.item_tower(self.item_emb(iid))

    def cross_bias(self, cross_feat: torch.Tensor) -> torch.Tensor:
        """b_cross = f_cross(x_cross)，只能在线实时计算。[B, 1] → [B]"""
        return self.cross_tower(cross_feat).squeeze(-1)

    def forward(
        self,
        uid: torch.Tensor,
        iid: torch.Tensor,
        cross_feat: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        交叉熵损失，三个塔端到端联合优化。
        Returns: (loss, s_tower_abs_mean, b_cross_abs_mean)
        """
        e_u = self.encode_user(uid)   # [B, D]
        e_i = self.encode_item(iid)   # [B, D]
        s_tower = F.cosine_similarity(e_u, e_i, dim=1)  # [B]
        b_cross = self.cross_bias(cross_feat)            # [B]
        s = s_tower + b_cross
        loss = F.binary_cross_entropy_with_logits(s, y)
        return loss, s_tower.abs().mean(), b_cross.abs().mean()

    # ── 两阶段打分：工程折中的核心 ─────────────────────────────────────────
    def pre_rank(
        self,
        uid: int,
        candidates: List[int],
        top_k: int = 20,
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        粗排（两阶段打分）：
          阶段一：双塔余弦给全部 N 个候选打分 → 截断 Top-N（top_n 个）；
          阶段二：只对 Top-N 调用交叉塔补上偏置分，再重排取 Top-K。
        交叉塔在线调用次数由 O(N) 降为 O(top_n)，top_n=None 时退化为全量计算。
        """
        with torch.no_grad():
            u = torch.tensor([uid])
            e_u = self.encode_user(u)  # [1, D]
            cand_t = torch.tensor(candidates)
            e_i = self.encode_item(cand_t)  # [N, D]
            s1 = F.cosine_similarity(e_u, e_i, dim=1)  # [N]
            stage1 = sorted(
                [(candidates[j], s1[j].item()) for j in range(len(candidates))],
                key=lambda x: x[1], reverse=True,
            )
            m = len(stage1) if top_n is None else min(top_n, len(stage1))
            # 阶段二：对 Top-M 用交叉塔重新打分
            top_m_iids = [iid for iid, _ in stage1[:m]]
            cross_feats = self.store.lookup_batch([uid] * m, top_m_iids)  # [m, CROSS_DIM]
            b = self.cross_bias(cross_feats)  # [m]
            rescored = [
                (top_m_iids[j], stage1[j][1] + b[j].item())
                for j in range(m)
            ]
            rescored.sort(key=lambda x: x[1], reverse=True)
            # 未进入阶段二的候选保留纯双塔分，排在已重排候选之后
            self.last_cross_calls = m
        return (rescored + stage1[m:])[:top_k]


def train(
    model: CrossFeatureTowerModel,
    samples: List[Tuple[int, int, float]],
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 0.05,
) -> None:
    """交叉熵损失，三个塔端到端联合优化。"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = list(samples)
    for epoch in range(epochs):
        random.shuffle(data)
        acc_loss = acc_tw = acc_cr = 0.0
        n = 0
        for start in range(0, len(data), batch_size):
            batch = data[start: start + batch_size]
            uids_list = [s[0] for s in batch]
            iids_list = [s[1] for s in batch]
            uids = torch.tensor(uids_list)
            iids = torch.tensor(iids_list)
            ys = torch.tensor([s[2] for s in batch], dtype=torch.float)
            cross_feat = model.store.lookup_batch(uids_list, iids_list)

            optimizer.zero_grad()
            loss, tw, cr = model(uids, iids, cross_feat, ys)
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()
            acc_tw += tw.item()
            acc_cr += cr.item()
            n += 1
        nn_ = max(n, 1)
        print(f"  epoch {epoch:2d}  loss={acc_loss / nn_:.4f}  "
              f"|s_tower|={acc_tw / nn_:.4f}  |b_cross|={acc_cr / nn_:.4f}")


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items, n_cates, n_authors = 8, 30, 4, 6
    item_category = [i % n_cates for i in range(n_items)]
    item_author = [i % n_authors for i in range(n_items)]
    store = CrossFeatureStore(n_users, n_items, item_category, item_author, n_cates)

    # 合成偏好：标签主要由「用户对该类目的历史 CTR」这一交叉特征决定
    samples: List[Tuple[int, int, float]] = []
    for uid in range(n_users):
        for iid in range(n_items):
            if random.random() > 0.4:
                continue
            ctr = store.user_cate_ctr[uid][item_category[iid]]
            y = 1.0 if random.random() < ctr else 0.0
            samples.append((uid, iid, y))
            if y > 0:
                store.observe_click(uid, iid)

    model = CrossFeatureTowerModel(
        n_users, n_items, store,
        emb_dim=16, hidden_dims=[32, 16], output_dim=16, cross_hidden=8,
    )
    print(f"三塔粗排模型训练（用户塔 + 物品塔 + 轻量交叉塔，样本 {len(samples)} 条）：")
    train(model, samples, epochs=6, batch_size=16, lr=0.05)

    uid = 0
    cands = list(range(n_items))
    print(f"\n在线两阶段打分（候选 N={len(cands)}）：")
    res_full = model.pre_rank(uid, cands, top_k=5, top_n=None)
    full_set = {i for i, _ in res_full}
    for top_n in (5, 10, 20):
        res = model.pre_rank(uid, cands, top_k=5, top_n=top_n)
        ov = len({i for i, _ in res} & full_set)
        print(f"  Top-N={top_n:2d}：交叉塔调用 {model.last_cross_calls:2d} 次"
              f"（省去 {len(cands) - model.last_cross_calls:2d} 次 Pair 级前向），"
              f"与全量 Top-5 重合 {ov}/5")

    TOP_N = 10
    print(f"\n两阶段（Top-N={TOP_N}）最终粗排 Top-5：")
    for iid, s in model.pre_rank(uid, cands, top_k=5, top_n=TOP_N):
        ctr = store.user_cate_ctr[uid][item_category[iid]]
        print(f"  item {iid:2d} (类目 {item_category[iid]}, 用户该类目 CTR={ctr:.3f}): {s:.4f}")

    print(f"\n对照 A：全量计算交叉塔（交叉塔调用 {len(cands)} 次，线上开销 O(N)）：")
    print("  " + ", ".join(f"item {i}:{s:.4f}" for i, s in res_full))

    with torch.no_grad():
        e_u = model.encode_user(torch.tensor([uid]))
        cand_t = torch.tensor(cands)
        e_i = model.encode_item(cand_t)
        s_tower = F.cosine_similarity(e_u, e_i, dim=1)
    res_tower = sorted(
        [(cands[j], s_tower[j].item()) for j in range(len(cands))],
        key=lambda x: x[1], reverse=True,
    )[:5]
    print("\n对照 B：纯双塔（交叉塔调用 0 次，物品 Embedding 全部可离线预计算）：")
    print("  " + ", ".join(f"item {i}:{s:.4f}" for i, s in res_tower))

    print("\n工程结论：交叉特征依赖用户与物品两侧，无法离线预计算，全量打分需 O(N) 次"
          "\nPair 级前向。两阶段打分把交叉塔调用压到 O(M)，Top-N 越大越逼近全量结果，"
          "\n是一个可按时延预算调节的「精度 / 开销」折中旋钮。")
