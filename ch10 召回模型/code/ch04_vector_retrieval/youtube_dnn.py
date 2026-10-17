"""
YoutubeDNN 向量召回（第10章 §10.3.1）

论文：Covington et al., "Deep Neural Networks for YouTube Recommendations", RecSys 2016

核心思想：
- 用户塔：将用户历史观看 ID 的平均 Embedding + 其他上下文特征输入多层 MLP，
  输出用户 Embedding
- 训练时把问题转换为超大规模多分类（对全量物品做 Softmax），
  线上用 item Embedding 做 ANN 检索
- 本实现简化为：Sampled Softmax + In-Batch 负采样，使用 PyTorch autograd
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 模型 ─────────────────────────────────────────────────────────────────────

class YoutubeDNN(nn.Module):
    """
    简化版 YoutubeDNN 召回模型。

    用户塔：avg_pool(历史 item emb) → MLP → user_emb
    物品塔：item_emb（可直接查表）
    训练：对每个正样本采 neg_k 个负样本，做 Sampled Softmax

    线上召回时：计算 user_emb，再在 item_emb 矩阵上做点积排序。
    """

    def __init__(
        self,
        n_items: int,
        item_emb_dim: int = 32,
        hidden_dims: list = None,
        output_dim: int = 32,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        self.n_items = n_items
        self.item_emb_dim = item_emb_dim
        self.output_dim = output_dim

        # 物品查找表（用于用户塔输入池化）
        self.item_emb = nn.Embedding(n_items, item_emb_dim)
        # 输出侧物品 Embedding（用于 Softmax 内积）
        self.item_output_emb = nn.Embedding(n_items, output_dim)

        # MLP 层：item_emb_dim → hidden_dims → output_dim
        layer_dims = [item_emb_dim] + hidden_dims + [output_dim]
        layers = []
        for in_d, out_d in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        # 移除最后一个 ReLU（输出层不加激活）
        layers = layers[:-1]
        self.mlp = nn.Sequential(*layers)

    def encode_user(self, history: list) -> torch.Tensor:
        """返回用户 Embedding（用于线上召回）。"""
        if not history:
            return torch.zeros(self.output_dim)
        with torch.no_grad():
            hist_tensor = torch.tensor(history, dtype=torch.long)
            h_embs = self.item_emb(hist_tensor)          # (len, item_emb_dim)
            pooled = torch.mean(h_embs, dim=0)           # (item_emb_dim,)
            return self.mlp(pooled)

    def forward(self, history_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        前向：对历史做 mean pooling → MLP → user_emb，
        与候选物品 output_emb 做内积，返回 logits。
        history_ids: (hist_len,)  candidate_ids: (num_cands,)
        """
        h_embs = self.item_emb(history_ids)              # (hist_len, item_emb_dim)
        pooled = torch.mean(h_embs, dim=0)               # (item_emb_dim,)
        user_emb = self.mlp(pooled)                      # (output_dim,)
        cand_embs = self.item_output_emb(candidate_ids)  # (num_cands, output_dim)
        logits = cand_embs @ user_emb                    # (num_cands,)
        return logits

    def train_model(
        self,
        user_history: dict,
        pos_samples: list,
        epochs: int = 10,
        neg_k: int = 10,
        lr: float = 0.01,
    ) -> None:
        """
        Args:
            user_history: {uid: [item_id, ...]} 历史行为（不含正样本）
            pos_samples:  [(uid, pos_item_id), ...]
            neg_k: 每个正样本的负采样数
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        user_pos = defaultdict(set)
        for uid, iid in pos_samples:
            user_pos[uid].add(iid)

        for epoch in range(epochs):
            random.shuffle(pos_samples)
            total_loss = 0.0
            self.train()
            for uid, pos_id in pos_samples:
                history = [i for i in user_history.get(uid, []) if i != pos_id]
                if not history:
                    history = [pos_id]
                neg_ids = []
                while len(neg_ids) < neg_k:
                    neg = random.randint(0, self.n_items - 1)
                    if neg not in user_pos[uid]:
                        neg_ids.append(neg)

                # 候选集 = {pos} ∪ {neg_ids}，正样本 label=0（排在最前）
                candidate_ids = torch.tensor([pos_id] + neg_ids, dtype=torch.long)
                history_tensor = torch.tensor(history, dtype=torch.long)

                optimizer.zero_grad()
                logits = self.forward(history_tensor, candidate_ids)  # (1+neg_k,)
                # F.cross_entropy: label=0 表示第一个候选（正样本）是目标
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 2 == 0:
                print(f"  epoch {epoch:3d}  avg_loss={total_loss / len(pos_samples):.4f}")

    def recommend(self, uid: int, history: list, top_k: int = 10, exclude: set = None) -> list:
        exclude = exclude or set()
        self.eval()
        with torch.no_grad():
            if not history:
                history = [0]
            history_tensor = torch.tensor(history, dtype=torch.long)
            h_embs = self.item_emb(history_tensor)
            pooled = torch.mean(h_embs, dim=0)
            user_emb = self.mlp(pooled)
            all_item_ids = [iid for iid in range(self.n_items) if iid not in exclude]
            item_tensor = torch.tensor(all_item_ids, dtype=torch.long)
            item_embs = self.item_output_emb(item_tensor)  # (n, output_dim)
            scores = (item_embs @ user_emb).tolist()
            pairs = list(zip(all_item_ids, scores))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items = 10, 20

    user_history = {
        u: random.sample(range(n_items), k=random.randint(3, 8))
        for u in range(n_users)
    }
    pos_samples = []
    for u, hist in user_history.items():
        for i in hist:
            pos_samples.append((u, i))

    model = YoutubeDNN(n_items, item_emb_dim=16, hidden_dims=[32, 16], output_dim=16)
    print("YoutubeDNN 训练：")
    model.train_model(user_history, pos_samples, epochs=6, neg_k=5, lr=0.05)

    uid = 0
    hist = user_history[uid]
    seen = set(hist)
    recs = model.recommend(uid, hist, top_k=5, exclude=seen)
    print(f"\n用户 {uid} 推荐结果（历史：{hist}）：")
    for iid, s in recs:
        print(f"  item {iid}: {s:.4f}")
