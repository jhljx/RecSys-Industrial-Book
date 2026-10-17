"""
MIND 多兴趣召回（第10章 §10.5.1）

论文：Li et al., "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall", CIKM 2019

核心思想：
- 用胶囊网络（Capsule Network）的动态路由机制，从用户行为序列中抽取多个兴趣向量
- 每个兴趣胶囊对应用户某一方面的偏好
- 训练时用 Label-aware Attention 机制，根据目标物品选择最相关的兴趣向量

本实现：PyTorch 实现，动态路由 + Label-aware Attention + Sampled Softmax loss。
"""

import math
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _squash(v: torch.Tensor) -> torch.Tensor:
    """Capsule squash 函数：将向量压缩到单位球内，保持方向。v: [..., d]"""
    norm_sq = (v * v).sum(dim=-1, keepdim=True)
    norm = torch.sqrt(norm_sq + 1e-9)
    scale = norm_sq / (1.0 + norm_sq) / norm
    return v * scale


class MIND(nn.Module):
    """
    简化版 MIND 多兴趣模型。

    组件：
      item_emb:   物品 Embedding 表
      动态路由：  3 次迭代，b_ij += dot(interest_k, item_emb)，softmax 得路由权重
      label_aware_attention: w_k = softmax(pow(dot(interest_k, target), p))
      Sampled Softmax loss 训练
    """

    def __init__(
        self,
        n_items: int,
        emb_dim: int = 32,
        n_interests: int = 4,
        routing_iters: int = 3,
        lr: float = 0.01,
        reg: float = 1e-4,
    ):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_interests = n_interests
        self.routing_iters = routing_iters
        self.reg = reg

        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.uniform_(self.item_emb.weight, -math.sqrt(1.0 / emb_dim), math.sqrt(1.0 / emb_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def _dynamic_routing(self, seq_embs: torch.Tensor) -> torch.Tensor:
        """
        EM 式动态路由，输入行为序列 Embedding，输出 n_interests 个兴趣向量。

        Args:
            seq_embs: [T, d] 行为序列 Embedding

        Returns:
            interests: [K, d]
        """
        T, d = seq_embs.shape
        K = self.n_interests

        b = torch.zeros(T, K, device=seq_embs.device)  # 路由 logit

        interests = torch.zeros(K, d, device=seq_embs.device)
        for _ in range(self.routing_iters):
            c = F.softmax(b, dim=1)  # [T, K]
            # 加权聚合：interests[k] = sum_t c[t][k] * seq_embs[t]
            interests = torch.einsum('tk,td->kd', c, seq_embs)  # [K, d]
            interests = _squash(interests)  # 胶囊 squash
            # 更新路由 logit
            b = b + torch.einsum('td,kd->tk', seq_embs, interests)  # [T, K]

        return interests

    def encode_user(self, seq: list) -> torch.Tensor:
        """给定用户行为序列（id 列表），返回 K 个兴趣向量 [K, d]。"""
        if not seq:
            return torch.zeros(self.n_interests, self.emb_dim)
        seq_t = torch.tensor(seq, dtype=torch.long)
        with torch.no_grad():
            seq_embs = self.item_emb(seq_t)  # [T, d]
        return self._dynamic_routing(seq_embs)

    def _label_aware_attention(
        self, interests: torch.Tensor, target_emb: torch.Tensor, pow_p: float = 2.0
    ) -> torch.Tensor:
        """
        Label-aware Attention：对多个兴趣向量加权求和，权重由目标物品相似度决定。

        attention_weight[k] = softmax(pow(<interest_k, target_emb>, pow_p))
        p=1 时退化为普通 softmax。
        """
        scores = (interests * target_emb.unsqueeze(0)).sum(dim=-1)  # [K]
        weights = F.softmax(scores.pow(pow_p), dim=0)  # [K]
        fused = (weights.unsqueeze(-1) * interests).sum(dim=0)  # [d]
        return fused

    def forward(self, seq_ids: torch.Tensor, pos_id: torch.Tensor, neg_id: torch.Tensor):
        """BPR 训练步骤：用 label-aware attention 融合兴趣后打分，返回 loss。"""
        seq_embs = self.item_emb(seq_ids)    # [T, d]
        pos_emb = self.item_emb(pos_id)      # [d]
        neg_emb = self.item_emb(neg_id)      # [d]

        interests = self._dynamic_routing(seq_embs)  # [K, d]
        user_emb = self._label_aware_attention(interests, pos_emb)  # [d]

        s_pos = F.cosine_similarity(user_emb.unsqueeze(0), pos_emb.unsqueeze(0))
        s_neg = F.cosine_similarity(user_emb.unsqueeze(0), neg_emb.unsqueeze(0))
        loss = -F.logsigmoid(s_pos - s_neg).mean()
        return loss

    def train_model(
        self,
        user_seqs: dict,
        epochs: int = 10,
        neg_k: int = 1,
    ) -> None:
        user_pos = collections.defaultdict(set)
        for uid, seq in user_seqs.items():
            user_pos[uid].update(seq)

        samples = []
        for uid, seq in user_seqs.items():
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                samples.append((uid, seq[:t], seq[t]))

        self.train()
        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss = 0.0
            for uid, context, pos_id in samples:
                for _ in range(neg_k):
                    neg_id = random.randint(0, self.n_items - 1)
                    while neg_id in user_pos[uid]:
                        neg_id = random.randint(0, self.n_items - 1)

                    seq_t = torch.tensor(context, dtype=torch.long)
                    pos_t = torch.tensor(pos_id, dtype=torch.long)
                    neg_t = torch.tensor(neg_id, dtype=torch.long)

                    self.optimizer.zero_grad()
                    loss = self(seq_t, pos_t, neg_t)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            if epoch % 2 == 0:
                print(f"  epoch {epoch:3d}  avg_loss={total_loss / max(len(samples), 1):.4f}")

    def recommend(
        self, seq: list, top_k: int = 10, exclude: set = None
    ) -> list:
        """
        线上推荐：对每个兴趣向量做 ANN 检索，取各兴趣的 Top-K 物品合并后重排。
        """
        exclude = exclude or set(seq)
        self.eval()
        with torch.no_grad():
            interests = self.encode_user(seq)  # [K, d]
            all_embs = self.item_emb.weight    # [n_items, d]
            # 对每个兴趣计算余弦相似度
            interests_norm = F.normalize(interests, dim=-1)   # [K, d]
            all_norm = F.normalize(all_embs, dim=-1)          # [n_items, d]
            sims = torch.mm(interests_norm, all_norm.t())      # [K, n_items]
            max_sims, _ = sims.max(dim=0)                      # [n_items]

        score_dict = {}
        for iid in range(self.n_items):
            if iid not in exclude:
                score_dict[iid] = max_sims[iid].item()

        ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_items = 20
    user_seqs = {
        u: random.sample(range(n_items), k=random.randint(5, 10))
        for u in range(8)
    }

    model = MIND(n_items, emb_dim=16, n_interests=3, routing_iters=3, lr=0.05)
    print("MIND 多兴趣模型训练：")
    model.train_model(user_seqs, epochs=6)

    seq = user_seqs[0]
    recs = model.recommend(seq, top_k=5)
    print(f"\n用户序列 {seq} 的 Top-5 推荐：")
    for iid, s in recs:
        print(f"  item {iid}: {s:.4f}")

    # 可视化多兴趣向量（查看是否分散）
    interests = model.encode_user(seq)
    print(f"\n{len(interests)} 个兴趣向量与 item 的相似度分布：")
    model.eval()
    with torch.no_grad():
        all_embs = model.item_emb.weight
        interests_norm = F.normalize(interests, dim=-1)
        all_norm = F.normalize(all_embs, dim=-1)
        sims = torch.mm(interests_norm, all_norm.t())  # [K, n_items]
    for k in range(len(interests)):
        norm_val = interests[k].norm().item()
        max_sim = sims[k].max().item()
        mean_sim = sims[k].mean().item()
        print(f"  interest {k}: norm={norm_val:.4f}  max_sim={max_sim:.4f}  mean_sim={mean_sim:.4f}")
