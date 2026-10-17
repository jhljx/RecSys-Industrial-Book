"""
ComiRec 可控多兴趣召回（第10章 §10.5.2）

论文：Cen et al., "Controllable Multi-Interest Framework for Recommendation", KDD 2020

核心思想：
- 用 K 个兴趣向量 V_u = [v_u^1, ..., v_u^K] 表示一个用户，而非单一 Embedding
- 两种多兴趣提取方式：ComiRec-DR（胶囊网络动态路由，与 MIND 一致）与
  ComiRec-SA（自注意力  A = Softmax(W2^T tanh(W1 H))^T ,  V_u = H A）
- 训练时用兴趣选择器只更新与目标物品最相关的兴趣：k* = argmax_k (v_u^k)^T e_i
- 最大创新：可控的召回聚合（Controllable Aggregation），用贪心推理求解
    Q(u, S) = Σ_{i∈S} f(u,i) + λ Σ_{i,j∈S} g(i,j)
  其中 f(u,i) = max_k (e_i^T v_u^k) 为相关性，g(i,j) 为物品间多样性度量。
  λ=0 时退化为普通 Top-N 召回；λ 越大，召回结果类目覆盖越广。

本实现：PyTorch 实现，DR/SA 两个变体 + 可控多样性聚合。
"""

import math
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _squash(v: torch.Tensor) -> torch.Tensor:
    """Capsule squash 函数：压缩向量长度到 (0,1)，保持方向不变。v: [..., d]"""
    norm_sq = (v * v).sum(dim=-1, keepdim=True)
    norm = torch.sqrt(norm_sq + 1e-9)
    scale = norm_sq / (1.0 + norm_sq) / norm
    return v * scale


class ComiRec(nn.Module):
    """
    简化版 ComiRec 多兴趣召回模型。

    mode="SA" 为自注意力分支，mode="DR" 为动态路由分支；item_cate 供聚合模块的 g(i,j) 使用。
    参数：item_emb、W1 [d_a×d]、W2 [d_a×K]（SA 分支的注意力投影权重）。
    """

    def __init__(
        self, n_items: int, item_cate: list, emb_dim: int = 16, n_interests: int = 3,
        att_dim: int = 8, routing_iters: int = 3, mode: str = "SA",
        lr: float = 0.05, reg: float = 1e-4,
    ):
        super().__init__()
        assert mode in ("SA", "DR"), "mode 只支持 SA / DR"
        self.n_items = n_items
        self.item_cate = item_cate
        self.emb_dim = emb_dim
        self.K = n_interests
        self.att_dim = att_dim
        self.routing_iters = routing_iters
        self.mode = mode
        self.reg = reg

        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.uniform_(self.item_emb.weight, -math.sqrt(1.0 / emb_dim), math.sqrt(1.0 / emb_dim))

        # SA 分支专用参数：A = Softmax(W2^T tanh(W1 H))^T
        if mode == "SA":
            self.W1 = nn.Parameter(torch.empty(att_dim, emb_dim))
            self.W2 = nn.Parameter(torch.empty(att_dim, n_interests))
            nn.init.normal_(self.W1, std=math.sqrt(1.0 / emb_dim))
            nn.init.normal_(self.W2, std=math.sqrt(1.0 / emb_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    # ── 多兴趣提取：ComiRec-SA ────────────────────────────────────────────────

    def _self_attentive(self, H: torch.Tensor) -> tuple:
        """
        ComiRec-SA 多兴趣提取。

        H ∈ R^{n×d}（行为序列 Embedding）。
          U[t]    = tanh(W1 h_t)              ∈ R^{d_a}
          Z[k][t] = W2[:,k]^T U[t]            ∈ R^{K×n}
          A[t][k] = Softmax_t(Z[k][:])        ∈ R^{n×K}  每个兴趣在行为上的注意力分布
          V_u[k]  = Σ_t A[t][k] * h_t         ∈ R^{d×K}  即 V_u = H A

        Returns:
            (interests [K×d], A [n×K])
        """
        U = torch.tanh(H @ self.W1.t())          # [n, d_a]
        Z = U @ self.W2                           # [n, K]
        A = F.softmax(Z, dim=0)                   # [n, K]，沿行为维做 softmax
        interests = (H.t() @ A).t()              # [K, d]，V_u = H A
        return interests, A

    # ── 多兴趣提取：ComiRec-DR ────────────────────────────────────────────────

    def _dynamic_routing(self, H: torch.Tensor) -> tuple:
        """
        ComiRec-DR 多兴趣提取（胶囊网络动态路由，与 MIND 一致）。

        迭代过程：
          c[t][k] = Softmax_k(b[t][:])          行为 t 分配到兴趣 k 的概率
          v_k     = squash(Σ_t c[t][k] * h_t)   兴趣胶囊
          b[t][k] += <h_t, v_k>                 用一致性更新路由 logit
        """
        T, d = H.shape
        K = self.K
        b = torch.zeros(T, K, device=H.device)

        interests = torch.zeros(K, d, device=H.device)
        A = torch.ones(T, K, device=H.device) / K
        for _ in range(self.routing_iters):
            A = F.softmax(b, dim=1)                             # [T, K]
            interests = _squash(torch.einsum('tk,td->kd', A, H))  # [K, d]
            b = b + torch.einsum('td,kd->tk', H, interests)    # 路由一致性更新
        return interests, A

    def encode_user(self, seq: list) -> torch.Tensor:
        """给定行为序列，输出 K 个兴趣向量 [K, d]。"""
        if not seq:
            return torch.zeros(self.K, self.emb_dim)
        seq_t = torch.tensor(seq, dtype=torch.long)
        with torch.no_grad():
            H = self.item_emb(seq_t)  # [n, d]
        if self.mode == "SA":
            return self._self_attentive(H)[0]
        return self._dynamic_routing(H)[0]

    # ── 训练 ────────────────────────────────────────────────────────────────

    def forward(self, seq_ids: torch.Tensor, pos_id: torch.Tensor, neg_ids: torch.Tensor):
        """
        Sampled Softmax 训练步，兴趣选择器只激活与目标物品最相关的兴趣 k*。
        neg_ids: [n_negs]
        """
        H = self.item_emb(seq_ids)        # [T, d]
        pos_emb = self.item_emb(pos_id)   # [d]
        neg_embs = self.item_emb(neg_ids) # [n_negs, d]

        if self.mode == "SA":
            interests, _ = self._self_attentive(H)
        else:
            interests, _ = self._dynamic_routing(H)
        # [K, d]

        # 兴趣选择器：k* = argmax_k (v_u^k)^T e_i
        scores_k = (interests * pos_emb.unsqueeze(0)).sum(dim=-1)  # [K]
        kstar = scores_k.argmax().item()
        v_u = interests[kstar]  # [d]

        # Sampled Softmax（温度 0.2）
        cand_embs = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)  # [1+n_negs, d]
        v_u_norm = F.normalize(v_u.unsqueeze(0), dim=-1)
        cand_norm = F.normalize(cand_embs, dim=-1)
        logits = (v_u_norm @ cand_norm.t()).squeeze(0) / 0.2  # [1+n_negs]
        loss = F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=logits.device))
        return loss

    def train_model(self, user_seqs: dict, epochs: int = 6, n_negs: int = 4) -> None:
        """用「序列前缀 → 下一个物品」构造样本，Sampled Softmax 训练。"""
        user_pos = collections.defaultdict(set)
        samples = []
        for uid, seq in user_seqs.items():
            user_pos[uid].update(seq)
            for t in range(1, len(seq)):
                samples.append((uid, seq[:t], seq[t]))

        self.train()
        for epoch in range(epochs):
            random.shuffle(samples)
            total = 0.0
            for uid, ctx, pos in samples:
                negs = []
                while len(negs) < n_negs:
                    c = random.randrange(self.n_items)
                    if c not in user_pos[uid]:
                        negs.append(c)

                seq_t = torch.tensor(ctx, dtype=torch.long)
                pos_t = torch.tensor(pos, dtype=torch.long)
                neg_t = torch.tensor(negs, dtype=torch.long)

                self.optimizer.zero_grad()
                loss = self(seq_t, pos_t, neg_t)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
            print(f"  [{self.mode}] epoch {epoch}  avg_loss={total / max(len(samples), 1):.4f}")

    # ── 可控聚合 ────────────────────────────────────────────────────────────

    def _diversity(self, i: int, j: int) -> float:
        """多样性度量 g(i,j)：不同类目记 1 分，同类目记 0 分。"""
        return 1.0 if self.item_cate[i] != self.item_cate[j] else 0.0

    def _candidate_pool(self, seq: list, per_interest: int, exclude: set) -> dict:
        """每个兴趣向量各自检索 Top-N，合并成候选集 M，并记录 f(u,i)=max_k 相似度。"""
        self.eval()
        with torch.no_grad():
            interests = self.encode_user(seq)       # [K, d]
            all_embs = self.item_emb.weight          # [n_items, d]
            int_norm = F.normalize(interests, dim=-1)
            all_norm = F.normalize(all_embs, dim=-1)
            sims = torch.mm(int_norm, all_norm.t())  # [K, n_items]

        pool = {}
        for k in range(self.K):
            sim_k = sims[k]
            for i in range(self.n_items):
                if i not in exclude:
                    s = sim_k[i].item()
                    if s > pool.get(i, -2.0):
                        pool[i] = s
            # keep only top per_interest per interest
            ranked = sorted(pool.items(), key=lambda x: x[1], reverse=True)[:per_interest * (k + 1)]
            pool = dict(ranked)
        return pool

    def recommend(
        self, seq: list, top_k: int = 5, lam: float = 0.0,
        per_interest: int = 8, exclude: set = None,
    ) -> list:
        """
        可控召回聚合（贪心推理）。

        每一步选择使 f(u,i) + λ Σ_{j∈S} g(i,j) 最大的候选加入结果集 S。
        λ=0 等价于按 f(u,i) 排序的普通 Top-N 召回；λ 越大结果类目覆盖越广。
        """
        exclude = set(seq) if exclude is None else exclude
        pool = self._candidate_pool(seq, per_interest, exclude)
        selected = []
        chosen = set()
        while len(selected) < top_k and len(chosen) < len(pool):
            best_i, best_gain = -1, -1e18
            for i, f in pool.items():
                if i in chosen:
                    continue
                gain = f + lam * sum(self._diversity(i, j) for j, _ in selected)
                if gain > best_gain:
                    best_i, best_gain = i, gain
            selected.append((best_i, pool[best_i]))
            chosen.add(best_i)
        return selected


def _norm_val(v: torch.Tensor) -> float:
    return v.norm().item()


def _report(tag: str, model: ComiRec, seq: list, lams, n_cate: int, top_k: int = 6) -> None:
    """打印不同 λ 下的召回结果，对比相关性与类目多样性的权衡。"""
    for lam in lams:
        recs = model.recommend(seq, top_k=top_k, lam=lam, per_interest=10)
        cov = len({model.item_cate[i] for i, _ in recs})
        rel = sum(s for _, s in recs) / len(recs)
        detail = "  ".join(f"i{i}(c{model.item_cate[i]})" for i, _ in recs)
        print(f"  [{tag}] λ={lam:<4} 类目覆盖={cov}/{n_cate} 平均相关性={rel:+.3f}  {detail}")


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_items, n_users, n_cate = 16, 8, 8
    item_cate = [i % n_cate for i in range(n_items)]  # 16 个物品划分到 8 个类目

    # 合成数据：每个用户偏好 3 个类目，每个类目各采样 1~2 个物品
    user_seqs = {}
    user_favs = {}
    for u in range(n_users):
        favs = random.sample(range(n_cate), 3)
        seq = []
        for c in favs:
            seq += random.sample([i for i in range(n_items) if item_cate[i] == c],
                                 k=random.randint(1, 2))
        random.shuffle(seq)
        user_seqs[u], user_favs[u] = seq, favs

    print("物品类目分布：",
          {c: [i for i in range(n_items) if item_cate[i] == c] for c in range(n_cate)})

    print("\n=== ComiRec-SA 训练（自注意力多兴趣） ===")
    sa = ComiRec(n_items, item_cate, emb_dim=16, n_interests=3, att_dim=8, mode="SA", lr=0.05)
    sa.train_model(user_seqs, epochs=6, n_negs=4)

    print("\n=== ComiRec-DR 训练（动态路由多兴趣） ===")
    dr = ComiRec(n_items, item_cate, emb_dim=16, n_interests=3, mode="DR", lr=0.05)
    dr.train_model(user_seqs, epochs=6, n_negs=4)

    seq = user_seqs[0]
    print(f"\n目标用户行为序列：{seq}  "
          f"（类目 {[item_cate[i] for i in seq]}，偏好类目 {user_favs[0]}）")

    print("\n--- 可控聚合模块：λ 对召回准确率/多样性的权衡 ---")
    _report("SA", sa, seq, (0.0, 0.2, 0.5, 1.0), n_cate)
    _report("DR", dr, seq, (0.0, 1.0), n_cate)

    sa.eval()
    with torch.no_grad():
        interests = sa.encode_user(seq)
    norms = "  ".join(f"k={k}:{_norm_val(interests[k]):.4f}" for k in range(len(interests)))
    print(f"\n--- SA 兴趣向量范数（检查兴趣是否坍缩） --- \n  {norms}")
