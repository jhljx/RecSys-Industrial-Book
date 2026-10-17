"""
DSSM + Transformer 粗排模型（第11章 §11.1.3）

核心思想：
- 用 Transformer 编码器替换 TDSSM 中的 LSTM，对用户行为序列建模
- 位置编码 + 多头自注意力 → Mean Pooling → 输出用户动态兴趣向量
- 与静态用户 MLP 融合后和物品塔做余弦相似度

本实现：nn.TransformerEncoderLayer + mean pooling，PyTorch autograd。
"""

import math
import random
import collections
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DSSMTransformer(nn.Module):
    """
    DSSM + Transformer 粗排模型。
    用户塔 = 静态 MLP + Transformer 序列编码（mean pooling），element-wise 相加。
    物品塔 = MLP。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        output_dim: int = 32,
        ffn_dim: int = 64,
        nhead: int = 1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.n_items = n_items
        self.output_dim = output_dim
        self.temperature = temperature

        self.user_static_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # 固定式正弦位置编码（最多 50 个位置）
        max_len = 50
        pe = torch.zeros(max_len, emb_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        if emb_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, emb_dim]

        # Transformer Encoder（单层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 静态 MLP: emb_dim → output_dim
        self.static_mlp = nn.Linear(emb_dim, output_dim)

        # 动态投影: emb_dim → output_dim
        self.dyn_proj = nn.Linear(emb_dim, output_dim)

        # 物品 MLP: emb_dim → output_dim
        self.item_mlp = nn.Linear(emb_dim, output_dim)

        # 初始化 Embedding
        scale = math.sqrt(1.0 / emb_dim)
        nn.init.uniform_(self.user_static_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

    def encode_user(self, uid: torch.Tensor, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uid:     [B] 用户 ID
            seq_ids: [B, T] 行为序列，0 表示 padding
        Returns:
            user_emb: [B, output_dim]
        """
        # 静态兴趣
        e_static = self.static_mlp(self.user_static_emb(uid))  # [B, D]

        # 动态兴趣（Transformer + mean pooling）
        T = seq_ids.size(1)
        if T > 0:
            seq_emb = self.item_emb(seq_ids)  # [B, T, emb_dim]
            # 加上固定位置编码
            seq_emb = seq_emb + self.pe[:T].unsqueeze(0)  # [B, T, emb_dim]
            # padding mask（0 ID 视为 pad）
            pad_mask = (seq_ids == 0)  # [B, T]，True = padding
            encoded = self.transformer(seq_emb, src_key_padding_mask=pad_mask)  # [B, T, emb_dim]
            # mean pooling（忽略 padding 位置）
            valid = (~pad_mask).float().unsqueeze(-1)  # [B, T, 1]
            e_temp = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # [B, emb_dim]
            e_temp = self.dyn_proj(e_temp)  # [B, output_dim]
        else:
            e_temp = torch.zeros(uid.size(0), self.output_dim)

        return e_static + e_temp

    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        return self.item_mlp(self.item_emb(iid))

    def forward(
        self,
        uid: torch.Tensor,
        seq_ids: torch.Tensor,
        pos_iid: torch.Tensor,
        neg_iid: torch.Tensor,
    ) -> torch.Tensor:
        """BPR loss: − log σ(s_pos − s_neg)"""
        ue = self.encode_user(uid, seq_ids)
        pe = self.encode_item(pos_iid)
        ne = self.encode_item(neg_iid)
        s_pos = F.cosine_similarity(ue, pe, dim=1)
        s_neg = F.cosine_similarity(ue, ne, dim=1)
        loss = -F.logsigmoid(s_pos - s_neg).mean()
        return loss

    def pre_rank(
        self, uid: int, seq: List[int], candidates: List[int], top_k: int = 20
    ) -> List[Tuple[int, float]]:
        with torch.no_grad():
            u = torch.tensor([uid])
            seq_trunc = seq[-50:]
            seq_t = torch.tensor([seq_trunc], dtype=torch.long) if seq_trunc else torch.zeros(1, 1, dtype=torch.long)
            ue = self.encode_user(u, seq_t)  # [1, D]
            cand_t = torch.tensor(candidates)
            ie = self.encode_item(cand_t)    # [N, D]
            sims = F.cosine_similarity(ue, ie, dim=1)
            scores = [(candidates[j], sims[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _make_batch(
    samples: List[Tuple[int, List[int], int]],
    n_items: int,
    user_pos: Dict[int, set],
    max_seq_len: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    uids, seqs, pos_ids, neg_ids = [], [], [], []
    for uid, context, pos_id in samples:
        neg_id = random.randint(0, n_items - 1)
        while neg_id in user_pos[uid]:
            neg_id = random.randint(0, n_items - 1)
        uids.append(uid)
        pos_ids.append(pos_id)
        neg_ids.append(neg_id)
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
    model: DSSMTransformer,
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
        uids_t, seqs_t, pos_t, neg_t = _make_batch(samples, model.n_items, user_pos)
        optimizer.zero_grad()
        loss = model(uids_t, seqs_t, pos_t, neg_t)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"  epoch {epoch:3d}  avg_loss={loss.item():.4f}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items = 8, 20
    user_seqs = {u: random.sample(range(n_items), k=random.randint(5, 10)) for u in range(n_users)}
    model = DSSMTransformer(n_users, n_items, emb_dim=16, output_dim=16, ffn_dim=32)
    print("DSSM+Transformer 训练：")
    train(model, user_seqs, epochs=6, lr=0.05)
    uid, seq = 0, user_seqs[0]
    candidates = [i for i in range(n_items) if i not in set(seq)]
    for iid, s in model.pre_rank(uid, seq[-5:], candidates, top_k=5):
        print(f"  item {iid}: {s:.4f}")
