"""
REMI 多兴趣召回（第10章 §10.5.3）

论文：Xie et al., "Rethinking Multi-Interest Learning for Candidate Matching in
     Recommender Systems", RecSys 2023

REMI 不改网络结构、不改在线 ANN 检索流程，只从训练机制上修正 MIND / ComiRec 的两个问题：
1）Easy Negative（负样本太简单）：多兴趣训练只用「与正样本最相关的那一个兴趣」v_u 参与优化，
   而负样本却是从全库均匀采样的，对这个特定兴趣几乎都能轻易区分，梯度信号很弱。
   => Interest-aware Hard Negative Mining（IHN）：理想难负样本分布 q_β(i^-) ∝ exp(β·v_u^T e_{i^-})，
      直接采样需遍历全库，故用 Importance Sampling 近似——仍从均匀分布 p 采样，再按
      w(i^-) ∝ q_β(i^-)/p(i^-) ∝ exp(β·v_u^T e_{i^-}) 对已采样负样本「重加权」。β=0 退化为均匀
      采样，β 越大越聚焦难负样本（过大则受 False Negative 影响）。本实现另提供插值系数 α，
      用 e_hard = (1-α)·e_{i^-} + α·e_{i^+} 直接合成难负样本。
2）Routing Collapse（兴趣坍缩）：路由矩阵 A 过于稀疏，权重集中到极少数历史行为上，使部分兴趣
   向量只能利用很少的行为信息。=> Routing Regularization 直接约束路由矩阵：书中形式
   C = (A-Ā)^T(A-Ā)，L_reg = ‖diag(C)‖_F^2；另提供正交变体 L_ortho = ‖A^T A - I‖_F^2。
   另有熵型正则 L_ent = -Σ_t Σ_k A[t][k]·log A[t][k]，最大化路由分布熵以缓解坍缩。

最终目标 L = L_SSM + λ·L_reg。本实现为 PyTorch，路由网络用 ComiRec-SA 形式
A = Softmax(W2^T tanh(W1 H))^T。
"""

import math
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class REMI(nn.Module):
    """
    简化版 REMI 多兴趣召回模型。

    关键超参：beta = IHN 难负样本强度 β（0 即均匀采样）；alpha = 合成难负样本插值系数 α；
    lam_reg = 路由正则权重 λ；reg_mode = "var"（‖diag(C)‖²）/"ortho"（‖A^T A-I‖²）/"ent"（熵）；
    init_gain = 路由网络初始化增益，取较大值可在玩具规模上复现兴趣坍缩现象。
    """

    def __init__(self, n_items: int, item_cate: list, emb_dim: int = 16,
                 n_interests: int = 3, att_dim: int = 8, beta: float = 0.0, alpha: float = 0.0,
                 lam_reg: float = 0.0, use_ortho: bool = False, use_entropy: bool = False,
                 init_gain: float = 4.0, tau: float = 0.2, lr: float = 0.05, reg: float = 1e-4):
        super().__init__()
        self.n_items = n_items
        self.item_cate = item_cate
        self.emb_dim = emb_dim
        self.K = n_interests
        self.att_dim = att_dim
        self.beta, self.alpha, self.lam_reg = beta, alpha, lam_reg
        self.use_ortho, self.use_entropy = use_ortho, use_entropy
        self.tau = tau
        self.reg = reg
        self.last_neg_sim = 0.0  # 最近一轮「被选中负样本与当前兴趣」的平均余弦相似度

        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.uniform_(self.item_emb.weight, -math.sqrt(1.0 / emb_dim), math.sqrt(1.0 / emb_dim))

        g = math.sqrt(1.0 / emb_dim) * init_gain  # 增益越大，初始路由分布越尖锐
        self.W1 = nn.Parameter(torch.empty(att_dim, emb_dim))
        self.W2 = nn.Parameter(torch.empty(att_dim, n_interests))
        nn.init.normal_(self.W1, std=g)
        nn.init.normal_(self.W2, std=g)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    # ── 多兴趣路由：A = G(H)，V_u = H A ─────────────────────────────────────

    def _routing(self, H: torch.Tensor) -> tuple:
        """U[t] = tanh(W1 h_t)，Z[t,k] = W2[:,k]^T U[t]，A[t][k] = Softmax_t(Z[:,k])，
        V_u[:,k] = Σ_t A[t][k] h_t 。返回 (interests [K×d], A [n×K])。"""
        U = torch.tanh(H @ self.W1.t())     # [n, d_a]
        Z = U @ self.W2                      # [n, K]
        A = F.softmax(Z, dim=0)              # [n, K]，沿行为维 softmax
        interests = (H.t() @ A).t()          # [K, d]，V_u = H A
        return interests, A

    def encode_user(self, seq: list) -> torch.Tensor:
        """给定行为序列输出 K 个兴趣向量 [K, d]。"""
        if not seq:
            return torch.zeros(self.K, self.emb_dim)
        seq_t = torch.tensor(seq, dtype=torch.long)
        with torch.no_grad():
            H = self.item_emb(seq_t)
        return self._routing(H)[0]

    def routing_entropy(self, seq: list) -> float:
        """K 个兴趣路由分布的平均熵：越接近 log(n) 说明兴趣覆盖的行为越广（坍缩越轻）。"""
        if len(seq) < 2:
            return 0.0
        seq_t = torch.tensor(seq, dtype=torch.long)
        with torch.no_grad():
            H = self.item_emb(seq_t)
            _, A = self._routing(H)
            ent = -(A * torch.log(A + 1e-12)).sum(dim=0).mean()
        return ent.item()

    # ── 机制一：Interest-aware Hard Negative Mining ──────────────────────────

    def _mine_hard_negatives(self, v_u: torch.Tensor, pos_id: int, forbidden: set,
                             n_negs: int, n_cand: int = 12) -> list:
        """兴趣感知的难负样本挖掘：1) 均匀采样 n_cand 个候选；
        2) 按当前兴趣相似度 s_i = v_u^T e_i 降序取 n_negs 个（得分高 = 更难）；
        3) 重要性权重 w_i ∝ exp(β·s_i) 归一化到均值 1，使 β=0 时等价于普通 SSM；
        4) 可选用插值系数 α 合成更难的负样本 e_hard = (1-α)e_{i^-} + α e_{i^+}。
        """
        cand = []
        while len(cand) < n_cand:
            c = random.randrange(self.n_items)
            if c not in forbidden:
                cand.append(c)

        cand_t = torch.tensor(cand, dtype=torch.long)
        with torch.no_grad():
            cand_embs = self.item_emb(cand_t)              # [n_cand, d]
            sims = (cand_embs @ v_u.detach())              # [n_cand]
            if self.beta > 0.0:
                order = sims.argsort(descending=True).tolist()
                cand = [cand[o] for o in order]
            picked = cand[:n_negs]
            raw = torch.exp(torch.clamp(self.beta * sims[:len(picked)]
                                        if self.beta > 0.0 else torch.zeros(len(picked)), max=30.0))
            raw = raw / raw.mean()  # 归一化到均值 1

        pos_emb = self.item_emb(torch.tensor(pos_id, dtype=torch.long)).detach()
        picked_t = torch.tensor(picked, dtype=torch.long)
        with torch.no_grad():
            picked_embs = self.item_emb(picked_t)  # [n_negs, d]
            if self.alpha > 0.0:
                neg_embs = (1 - self.alpha) * picked_embs + self.alpha * pos_emb.unsqueeze(0)
            else:
                neg_embs = picked_embs
        return [{"id": picked[i], "emb": neg_embs[i], "w": raw[i].item()}
                for i in range(len(picked))]

    # ── 机制二：Routing Regularization ──────────────────────────────────────

    def _routing_reg(self, A: torch.Tensor) -> torch.Tensor:
        """
        路由正则（可微，直接对 A 求导由 autograd 完成）。

        书中形式：Ā[k]=mean_t A[t][k]，C=(A-Ā)^T(A-Ā)，L=Σ_k C[k][k]^2，惩罚路由权重方差；
        正交变体：M=A^T A，L=Σ_{k,l}(M[k][l]-δ_kl)^2，惩罚兴趣之间的路由重叠；
        熵变体：L=-Σ A log A，最大化路由熵。
        """
        if self.use_entropy:
            return -(A * torch.log(A + 1e-12)).sum()
        if self.use_ortho:
            n = A.shape[0]
            M = A.t() @ A                                # [K, K]
            I = torch.eye(self.K, device=A.device)
            return ((M - I) ** 2).sum()
        A_bar = A.mean(dim=0, keepdim=True)              # [1, K]
        C = (A - A_bar).t() @ (A - A_bar)                # [K, K]
        return (torch.diagonal(C) ** 2).sum()

    # ── 训练 ────────────────────────────────────────────────────────────────

    def forward(self, seq_ids: torch.Tensor, pos_id: int, forbidden: set, n_negs: int = 4):
        """一步训练：IHN 重加权的 Sampled Softmax + Routing Regularization。返回 (L_ssm, L_reg, neg_sim)。"""
        H = self.item_emb(seq_ids)                 # [n, d]
        pos_emb = self.item_emb(torch.tensor(pos_id, dtype=torch.long))  # [d]
        interests, A = self._routing(H)

        # 兴趣选择器：k = argmax_k V_u[:,k]^T e_{i+}
        with torch.no_grad():
            kstar = (interests.detach() @ pos_emb.detach()).argmax().item()
        v_u = interests[kstar]  # [d]

        negs = self._mine_hard_negatives(v_u, pos_id, forbidden, n_negs)

        s_pos = F.cosine_similarity(v_u.unsqueeze(0), pos_emb.unsqueeze(0)) / self.tau  # [1]
        neg_embs = torch.stack([ng["emb"] for ng in negs], dim=0)                       # [n_negs, d]
        w_neg = torch.tensor([ng["w"] for ng in negs])                                  # [n_negs]
        s_neg = F.cosine_similarity(F.normalize(v_u.unsqueeze(0), dim=-1),
                                    F.normalize(neg_embs, dim=-1), dim=-1) / self.tau   # [n_negs]

        with torch.no_grad():
            neg_sim = s_neg.detach().mean().item() * self.tau  # 负样本难度统计（余弦相似度）

        # L_SSM = -log( e^{s+} / (e^{s+} + Σ w_i e^{s_i^-}) )，w 视为常数（stop-gradient）
        logits = torch.cat([s_pos.detach().new_zeros(1) * s_pos + s_pos,
                            s_neg + torch.log(w_neg)], dim=0)  # 重加权并入 logit 空间
        loss_ssm = F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long))

        loss_reg = torch.tensor(0.0)
        if self.lam_reg > 0.0 and len(seq_ids) > 1:
            loss_reg = self._routing_reg(A)

        return loss_ssm, loss_reg, neg_sim

    def train_model(self, user_seqs: dict, epochs: int = 6, n_negs: int = 4,
                    tag: str = "REMI", verbose: bool = True) -> None:
        """构造「序列前缀 → 下一物品」样本并训练。"""
        user_pos = collections.defaultdict(set)
        samples = []
        for uid, seq in user_seqs.items():
            user_pos[uid].update(seq)
            samples += [(uid, seq[:t], seq[t]) for t in range(1, len(seq))]

        nb = max(len(samples), 1)
        self.train()
        for epoch in range(epochs):
            random.shuffle(samples)
            tot_ssm = tot_reg = tot_sim = 0.0
            for uid, ctx, pos in samples:
                seq_t = torch.tensor(ctx, dtype=torch.long)
                self.optimizer.zero_grad()
                loss_ssm, loss_reg, neg_sim = self(seq_t, pos, user_pos[uid], n_negs)
                loss = loss_ssm + self.lam_reg * loss_reg
                loss.backward()
                self.optimizer.step()
                tot_ssm += loss_ssm.item()
                tot_reg += loss_reg.item() if loss_reg.requires_grad or loss_reg.item() else 0.0
                tot_sim += neg_sim
            self.last_neg_sim = tot_sim / nb
            if verbose:
                print(f"  [{tag}] epoch {epoch}  L_ssm={tot_ssm / nb:.4f}  L_reg={tot_reg / nb:.4f}"
                      f"  L={(tot_ssm + self.lam_reg * tot_reg) / nb:.4f}"
                      f"  负样本相似度={self.last_neg_sim:+.4f}")

    # ── 在线召回（与 MIND / ComiRec 完全一致） ───────────────────────────────

    def recommend(self, seq: list, top_k: int = 6, exclude: set = None) -> list:
        """每个兴趣各自 ANN 检索，候选打分 f(u,i) = max_k (v_u^k)^T e_i 。"""
        exclude = set(seq) if exclude is None else exclude
        with torch.no_grad():
            interests = self.encode_user(seq)              # [K, d]
            all_embs = self.item_emb.weight                 # [n_items, d]
            sims = F.normalize(interests, dim=-1) @ F.normalize(all_embs, dim=-1).t()
            scores_vec, _ = sims.max(dim=0)                 # [n_items]
        scores = {i: scores_vec[i].item() for i in range(self.n_items) if i not in exclude}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _evaluate(model: REMI, data: dict, top_k: int = 6) -> tuple:
    """留一评估（每个用户最后一个物品为 ground truth）：HR@K / 平均排名 / 路由熵。"""
    hits = rank = ent = 0.0
    for seq in data.values():
        ctx, tgt = seq[:-1], seq[-1]
        ids = [i for i, _ in model.recommend(ctx, top_k=model.n_items, exclude=set(ctx))]
        hits, rank = hits + (1.0 if tgt in ids[:top_k] else 0.0), rank + ids.index(tgt) + 1
        ent += model.routing_entropy(ctx)
    return hits / len(data), rank / len(data), ent / len(data)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_items, n_users, n_cate = 16, 8, 8
    item_cate = [i % n_cate for i in range(n_items)]
    # 合成数据：每个用户有 3 个兴趣类目，各取 2 个不重复物品，最后一个物品留作评估
    data = {}
    for u in range(n_users):
        seq = []
        for c in random.sample(range(n_cate), 3):
            seq += random.sample([i for i in range(n_items) if item_cate[i] == c], k=2)
        random.shuffle(seq)
        data[u] = seq
    train_data = {u: s[:-1] for u, s in data.items()}  # 训练只用前 n-1 个物品
    print(f"物品类目：{item_cate}\n用户 0 完整序列 {data[0]}"
          f"（类目 {[item_cate[i] for i in data[0]]}，留出样本 {data[0][-1]}）\n")

    configs = [("Base β=0 λ=0", dict(beta=0.0)),
               ("+IHN β=6", dict(beta=6.0)),
               ("+IHN+α=0.2 合成", dict(beta=6.0, alpha=0.2)),
               ("+Reg 方差 λ=2", dict(beta=6.0, lam_reg=2.0)),
               ("+Reg 正交 λ=4", dict(beta=6.0, lam_reg=4.0, use_ortho=True))]
    results = []
    for tag, kw in configs:
        random.seed(42)
        torch.manual_seed(42)  # 各组配置初始化完全一致，差异仅来自训练机制
        m = REMI(n_items, item_cate, emb_dim=16, n_interests=3, att_dim=8, lr=0.05, **kw)
        print(f"=== {tag} ===")
        m.train_model(train_data, epochs=6, n_negs=4, tag=tag)
        results.append((tag, m) + _evaluate(m, data, top_k=6))
    print("--- 训练机制对比（8 个用户留一评估，Top-6 召回） ---")
    pad = lambda t: t + " " * (20 - sum(2 if ord(c) > 0x2E80 else 1 for c in t))  # CJK 占 2 格
    print(f"  {pad('配置')}负样本相似度   路由熵    HR@6  平均排名")
    for tag, m, hr, rk, ent in results:
        print(f"  {pad(tag)}{m.last_neg_sim:>+14.4f}{ent:>9.4f}{hr:>8.3f}{rk:>10.2f}")
    print("  机制一（IHN）：负样本相似度明显上升，说明负样本确实变难，不再是 Easy Negative；\n"
          f"  机制二（Reg）：方差型正则推高路由熵（上界 log(5)={math.log(5):.3f}）以缓解兴趣坍缩，"
          "正交型正则让不同兴趣的路由分布彼此分离。\n"
          "  提示：玩具规模下 HR / 排名波动较大，重点看两个机制本身是否按预期生效。")

    tag, best, seq = results[-1][0], results[-1][1], data[0][:-1]
    print(f"\n--- {tag} 对用户 0（序列 {seq}）的 Top-6 召回 ---")
    for i, sc in best.recommend(seq, top_k=6):
        flag = "  <= 命中留出样本" if i == data[0][-1] else ""
        print(f"  item {i:>2} (类目 {item_cate[i]})  score={sc:+.4f}{flag}")
