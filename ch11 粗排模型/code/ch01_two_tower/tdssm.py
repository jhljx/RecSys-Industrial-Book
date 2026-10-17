"""
TDSSM 时序双塔粗排模型（第11章 §11.1.2）

核心扩展（相对 DSSM）：
- 用户塔分为静态兴趣（MLP）和动态兴趣（GRU 序列编码）两部分
- 两者融合后与物品 Embedding 做余弦相似度
- 体现长期兴趣 + 短期兴趣解耦建模的思想

本实现：静态 MLP（长期兴趣）+ nn.GRU（短期兴趣），PyTorch autograd。
"""

import math
import random
import collections
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TDSSM(nn.Module):
    """
    时序 DSSM（TDSSM）。

    用户塔 = 静态 MLP（长期兴趣）+ GRU（短期兴趣）→ element-wise 相加融合
    物品塔 = MLP
    融合方式：element-wise 相加（可换成 concat 或加权乘）
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        static_hidden: int = 32,
        lstm_hidden: int = 32,
        output_dim: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.output_dim = output_dim
        self.temperature = temperature

        self.user_static_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # 静态用户塔（长期兴趣 MLP）
        self.static_mlp = nn.Sequential(
            nn.Linear(emb_dim, static_hidden),
            nn.ReLU(),
            nn.Linear(static_hidden, output_dim),
        )

        # 短期兴趣：GRU 编码行为序列
        self.gru = nn.GRU(emb_dim, lstm_hidden, batch_first=True)
        self.gru_proj = nn.Linear(lstm_hidden, output_dim)

        # 物品塔 MLP
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, output_dim),
        )

        # 初始化 Embedding
        scale = math.sqrt(1.0 / emb_dim)
        nn.init.uniform_(self.user_static_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

    def encode_user(self, uid: torch.Tensor, seq_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            uid:     [B] 用户 ID（用于静态特征）
            seq_ids: [B, T] 用户近期行为序列（物品 ID），不足 T 的补 0
            seq_len: 实际最大序列长度（用于 packed / mean pooling）
        Returns:
            user_emb: [B, output_dim]
        """
        # 静态兴趣
        e_static = self.static_mlp(self.user_static_emb(uid))  # [B, D]

        # 短期兴趣（GRU）
        if seq_len > 0:
            seq_emb = self.item_emb(seq_ids)          # [B, T, emb_dim]
            _, h_n = self.gru(seq_emb)                 # h_n: [1, B, lstm_hidden]
            e_temp = self.gru_proj(h_n.squeeze(0))     # [B, output_dim]
        else:
            e_temp = torch.zeros(uid.size(0), self.output_dim)

        # 融合（element-wise 相加）
        return e_static + e_temp

    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        return self.item_mlp(self.item_emb(iid))  # [B, output_dim] or [N, output_dim]

    def forward(
        self,
        uid: torch.Tensor,
        seq_ids: torch.Tensor,
        pos_iid: torch.Tensor,
        neg_iid: torch.Tensor,
    ) -> torch.Tensor:
        """BPR loss: − log σ(s_pos − s_neg)"""
        ue = self.encode_user(uid, seq_ids, seq_ids.size(1))  # [B, D]
        pe = self.encode_item(pos_iid)                         # [B, D]
        ne = self.encode_item(neg_iid)                         # [B, D]
        s_pos = F.cosine_similarity(ue, pe, dim=1)
        s_neg = F.cosine_similarity(ue, ne, dim=1)
        loss = -F.logsigmoid(s_pos - s_neg).mean()
        return loss

    def pre_rank(
        self, uid: int, seq: List[int], candidates: List[int], top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """保留接口：pre_rank(uid, seq, candidates, top_k) -> List[Tuple[int, float]]"""
        with torch.no_grad():
            u = torch.tensor([uid])
            T = len(seq)
            if T > 0:
                seq_t = torch.tensor([seq]).long()  # [1, T]
            else:
                seq_t = torch.zeros(1, 1, dtype=torch.long)
            ue = self.encode_user(u, seq_t, T)  # [1, D]
            cand_t = torch.tensor(candidates)
            ie = self.encode_item(cand_t)         # [N, D]
            sims = F.cosine_similarity(ue, ie, dim=1)
            scores = [(candidates[j], sims[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _make_batch(
    samples: List[Tuple[int, List[int], int]],
    n_items: int,
    user_pos: Dict[int, set],
    max_seq_len: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    uids, seqs, pos_ids, neg_ids = [], [], [], []
    for uid, context, pos_id in samples:
        neg_id = random.randint(0, n_items - 1)
        while neg_id in user_pos[uid]:
            neg_id = random.randint(0, n_items - 1)
        uids.append(uid)
        pos_ids.append(pos_id)
        neg_ids.append(neg_id)
        # pad / truncate sequence
        ctx = context[-max_seq_len:]
        pad = [0] * (max_seq_len - len(ctx))
        seqs.append(pad + ctx)
    return (
        torch.tensor(uids),
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(pos_ids),
        torch.tensor(neg_ids),
    )


def train(
    model: TDSSM,
    user_seqs: Dict[int, List[int]],
    epochs: int = 10,
    lr: float = 0.01,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    user_pos: Dict[int, set] = collections.defaultdict(set)
    for uid, seq in user_seqs.items():
        for i in seq:
            user_pos[uid].add(i)

    samples = []
    for uid, seq in user_seqs.items():
        if len(seq) < 2:
            continue
        for t in range(1, len(seq)):
            samples.append((uid, seq[:t], seq[t]))

    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        uids_t, seqs_t, pos_t, neg_t = _make_batch(
            samples, model.n_items, user_pos
        )
        optimizer.zero_grad()
        loss = model(uids_t, seqs_t, pos_t, neg_t)
        loss.backward()
        optimizer.step()
        total_loss = loss.item()
        if epoch % 2 == 0:
            print(f"  epoch {epoch:3d}  avg_loss={total_loss:.4f}")


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items = 8, 20
    user_seqs = {u: random.sample(range(n_items), k=random.randint(5, 10)) for u in range(n_users)}

    model = TDSSM(n_users, n_items, emb_dim=16, static_hidden=16, lstm_hidden=16, output_dim=16)
    print("TDSSM 训练：")
    train(model, user_seqs, epochs=6, lr=0.05)

    uid = 0
    seq = user_seqs[uid]
    candidates = [i for i in range(n_items) if i not in set(seq)]
    result = model.pre_rank(uid, seq[-5:], candidates, top_k=5)
    print(f"\n用户 {uid} 粗排 Top-5（最近 5 个行为作为短期序列）：")
    for iid, s in result:
        print(f"  item {iid}: {s:.4f}")
