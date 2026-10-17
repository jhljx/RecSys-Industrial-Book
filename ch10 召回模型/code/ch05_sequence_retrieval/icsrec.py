"""
ICSRec 意图对比学习序列召回（第10章 §10.4.5「对比学习序列召回 - ICSRec召回」）

论文：Chen et al., "Intent Contrastive Learning for Sequential Recommendation",
      WWW 2022
核心思想：
- 序列推荐中用户行为受多种潜在意图（Intent）驱动，单纯的实例级对比（CL4SRec）
  忽略了这种意图结构；ICSRec 在对比学习中引入意图聚类，让同意图的序列相互吸引，
  不同意图的序列相互排斥。
- K-Means 聚类（余弦距离，Python 实现，不参与反向传播）：
      每隔若干 epoch 在当前序列 Embedding 上运行一次 K-Means，
      将 N 条序列划分为 K 个意图簇，得到聚类中心 μ_k 和分配 c_i。
- 两级对比损失：
    * CICL（Cluster-level Instance CL）——意图级对比：
          每条序列与 K 个意图中心做余弦 softmax，正例为自身所属意图中心：
          L_cicl = F.cross_entropy(h @ prototypes.T / τ, cluster_labels)
    * FICL（Fine-grained Instance CL）——实例级对比（带 False Negative Mask）：
          同一序列的两个增强视图互为正例；同一 Mini-Batch 内其它用户视图为负例，
          但同属一个意图簇的序列可能也是正例（False Negative），需从负例中屏蔽：
          mask[i, j] = 1  if i ≠ j and cluster[i] == cluster[j]
          L_ficl = F.cross_entropy(sim_matrix_masked / τ, arange(B))
- 联合训练：L_total = L_rec + λ_cicl · L_cicl + λ_ficl · L_ficl
  其中 L_rec 为 BCE 形式的下一物品预测损失（与 CL4SRec 相同）。

本实现：K-Means 用 PyTorch 张量操作（无 autograd）；对比损失和主任务损失用
PyTorch autograd 完整反向传播。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── K-Means 聚类（余弦距离，Python 实现，不接入 autograd）────────────────────

def kmeans(embeddings: torch.Tensor, K: int,
           n_iter: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """
    余弦 K-Means（在 detached 张量上运行，结果不参与梯度计算）。

    参数：
      embeddings: [N, dim]  序列 Embedding（已归一化或未归一化均可，内部会归一化）
      K:          聚类数
      n_iter:     最大迭代次数

    返回：
      assignments: [N]     每条序列的聚类标签（0 … K-1）
      centroids:   [K, dim] 归一化后的聚类中心
    """
    N, dim = embeddings.shape
    K = min(K, N)
    emb = F.normalize(embeddings.detach(), dim=-1)  # [N, dim]，不接入 autograd

    # 随机初始化：从数据点中随机挑 K 个作为初始中心
    indices = torch.randperm(N)[:K]
    centroids = emb[indices].clone()                 # [K, dim]

    assignments = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iter):
        # E 步：将每个点分配给余弦相似度最高的中心
        sim = emb @ centroids.T                      # [N, K]
        new_assign = sim.argmax(dim=-1)              # [N]

        if (new_assign == assignments).all():
            break
        assignments = new_assign

        # M 步：用各簇成员均值重新计算中心，并重新归一化
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            members = (assignments == k).nonzero(as_tuple=True)[0]
            if members.numel() > 0:
                new_centroids[k] = emb[members].mean(dim=0)
        centroids = F.normalize(new_centroids, dim=-1)

    return assignments, centroids


# ── 序列数据增强算子（与 CL4SRec 相同）──────────────────────────────────────

class SeqAugment:
    """三种序列数据增强算子：Crop / Mask / Reorder。"""

    def __init__(self, mask_token: int, eta: float = 0.6,
                 gamma: float = 0.3, beta: float = 0.5):
        self.mask_token = mask_token
        self.eta = eta
        self.gamma = gamma
        self.beta = beta

    def item_crop(self, seq: list[int]) -> list[int]:
        T = len(seq)
        sub_len = max(1, int(self.eta * T))
        start = random.randint(0, T - sub_len)
        return seq[start:start + sub_len]

    def item_mask(self, seq: list[int]) -> list[int]:
        out = list(seq)
        n_mask = max(1, int(self.gamma * len(seq)))
        for idx in random.sample(range(len(seq)), n_mask):
            out[idx] = self.mask_token
        return out

    def item_reorder(self, seq: list[int]) -> list[int]:
        out = list(seq)
        T = len(seq)
        sub_len = max(2, int(self.beta * T)) if T >= 2 else T
        sub_len = min(sub_len, T)
        start = random.randint(0, T - sub_len)
        chunk = out[start:start + sub_len]
        random.shuffle(chunk)
        out[start:start + sub_len] = chunk
        return out

    def random_augment(self, seq: list[int]) -> list[int]:
        op = random.choice([self.item_crop, self.item_mask, self.item_reorder])
        return op(seq)


# ── ICSRec 模型 ───────────────────────────────────────────────────────────────

class ICSRec(nn.Module):
    """
    ICSRec = SASRec 编码器 + K-Means 意图聚类 + CICL + FICL 对比学习。

    组件：
      item_emb : nn.Embedding [(n_items + 1) × dim]  最后一行为 [MASK] Embedding
      pos_emb  : nn.Embedding [max_len × dim]
      transformer_layer: 单层因果 TransformerEncoderLayer（与 SASRec / CL4SRec 相同）
      prototypes : [K, dim]  K-Means 聚类中心（非 Parameter，每隔 proto_update_freq
                             epoch 更新一次，不接入主任务 autograd）
    """

    def __init__(self, n_items: int, dim: int = 16, max_len: int = 10,
                 K: int = 4, tau: float = 0.5,
                 lam_cicl: float = 0.2, lam_ficl: float = 0.2,
                 lr: float = 0.05, reg: float = 1e-4,
                 n_heads: int = 1, dropout: float = 0.1,
                 proto_update_freq: int = 2):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        self.max_len = max_len
        self.K = K
        self.tau = tau
        self.lam_cicl = lam_cicl
        self.lam_ficl = lam_ficl
        self.lr = lr
        self.reg = reg
        self.mask_token = n_items
        self.proto_update_freq = proto_update_freq

        scale = math.sqrt(1.0 / dim)
        self.item_emb = nn.Embedding(n_items + 1, dim)   # +1 for [MASK]
        self.pos_emb = nn.Embedding(max_len, dim)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)
        nn.init.uniform_(self.pos_emb.weight, -scale, scale)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.aug = SeqAugment(self.mask_token)

        # 聚类中心和分配（非 Parameter，不参与主任务 autograd）
        self.register_buffer("prototypes",
                             torch.randn(K, dim))      # [K, dim]
        self.assignments: dict[int, int] = {}          # uid -> cluster_id

    def _encode_batch(self, seqs: list[list[int]],
                      pool: str = "last") -> torch.Tensor:
        """左填充后经单层因果 Transformer 编码，返回 [B, dim]。"""
        T = min(max(len(s) for s in seqs), self.max_len)
        B = len(seqs)
        padded = torch.zeros(B, T, dtype=torch.long)
        for i, seq in enumerate(seqs):
            s = seq[-T:]
            padded[i, T - len(s):] = torch.tensor(s, dtype=torch.long)

        positions = torch.arange(T).unsqueeze(0)
        x = self.item_emb(padded) + self.pos_emb(positions)         # [B, T, dim]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device)
        out = self.transformer_layer(x, src_mask=causal_mask,
                                     is_causal=True)                  # [B, T, dim]

        if pool == "mean":
            lengths = torch.tensor([min(len(s), T) for s in seqs],
                                   dtype=torch.float)
            mask = torch.zeros(B, T)
            for i, seq in enumerate(seqs):
                l = min(len(seq), T)
                mask[i, T - l:] = 1.0
            return (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        return out[:, -1, :]

    # ── 原型更新（不参与 autograd）────────────────────────────────────────
    def update_prototypes(self, user_seqs: dict[int, list[int]]) -> None:
        """
        在当前模型权重下计算全量用户 Embedding，然后运行余弦 K-Means，
        更新 self.prototypes 和 self.assignments。
        """
        uids = sorted(user_seqs.keys())
        with torch.no_grad():
            all_seqs = [user_seqs[u] for u in uids]
            # 分批编码，避免超长 batch
            batch = 32
            parts = []
            for i in range(0, len(all_seqs), batch):
                parts.append(
                    self._encode_batch(all_seqs[i:i + batch], pool="mean"))
            emb_all = torch.cat(parts, dim=0)             # [N, dim]

        assign, centroids = kmeans(emb_all, self.K)
        self.prototypes = centroids.to(emb_all.device)    # [K, dim]
        for i, uid in enumerate(uids):
            self.assignments[uid] = int(assign[i].item())

    # ── CICL 损失（意图级对比）────────────────────────────────────────────
    def _cicl_loss(self, seqs: list[list[int]],
                   cluster_labels: list[int]) -> torch.Tensor:
        """
        每条序列的 mean-pool Embedding 与 K 个意图原型做 InfoNCE：
        正例为自身所属簇的中心；L_cicl = cross_entropy(h @ μ.T / τ, c)
        """
        h = F.normalize(self._encode_batch(seqs, pool="mean"), dim=-1)  # [B, dim]
        proto = F.normalize(self.prototypes, dim=-1)                     # [K, dim]
        sim = (h @ proto.T) / self.tau                                   # [B, K]
        labels = torch.tensor(cluster_labels, dtype=torch.long,
                              device=sim.device)
        return F.cross_entropy(sim, labels)

    # ── FICL 损失（实例级对比 + False Negative Mask）─────────────────────
    def _ficl_loss(self, seqs_a: list[list[int]],
                   seqs_b: list[list[int]],
                   cluster_labels: list[int]) -> torch.Tensor:
        """
        两视图的 InfoNCE，但将同簇的非自身对从负例中屏蔽（False Negative Mask）：
        mask[i, j] = True  ⟺  i ≠ j  且  cluster[i] == cluster[j]
        屏蔽位置的 logit 设为 -∞，不参与分母归一化。
        """
        ha = F.normalize(self._encode_batch(seqs_a, pool="mean"), dim=-1)  # [B, dim]
        hb = F.normalize(self._encode_batch(seqs_b, pool="mean"), dim=-1)  # [B, dim]
        B = ha.size(0)

        # sim_matrix[i, j] = cosine(ha[i], hb[j]) / τ
        sim = (ha @ hb.T) / self.tau                                       # [B, B]

        # False Negative Mask：同簇且非自身对 → 屏蔽
        c = torch.tensor(cluster_labels, dtype=torch.long)                 # [B]
        same_cluster = (c.unsqueeze(0) == c.unsqueeze(1))                  # [B, B]
        diagonal = torch.eye(B, dtype=torch.bool)
        fn_mask = same_cluster & ~diagonal                                  # [B, B]
        sim = sim.masked_fill(fn_mask, float("-inf"))

        labels = torch.arange(B, dtype=torch.long, device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0

    # ── 主任务：下一物品预测（BCE）────────────────────────────────────────
    def _rec_loss(self, contexts: list[list[int]],
                  pos_ids: list[int],
                  neg_ids: list[int]) -> torch.Tensor:
        h = self._encode_batch(contexts, pool="last")                      # [B, dim]
        v_pos = self.item_emb(torch.tensor(pos_ids, dtype=torch.long))
        v_neg = self.item_emb(torch.tensor(neg_ids, dtype=torch.long))
        s_pos = (h * v_pos).sum(-1)
        s_neg = (h * v_neg).sum(-1)
        return (F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos))
                + F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg)))

    # ── 联合训练 ──────────────────────────────────────────────────────────
    def train_model(self, user_seqs: dict[int, list[int]],
                    epochs: int = 8,
                    batch_size: int = 4) -> None:
        """
        多任务训练：L_total = L_rec + λ_cicl·L_cicl + λ_ficl·L_ficl。
        每 proto_update_freq 个 epoch 更新一次聚类中心。
        """
        uids = list(user_seqs.keys())
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        # 首次聚类
        self.update_prototypes(user_seqs)

        for epoch in range(epochs):
            # 按频率更新原型（epoch 0 已在训练前初始化）
            if epoch > 0 and epoch % self.proto_update_freq == 0:
                self.update_prototypes(user_seqs)

            random.shuffle(uids)
            rec_sum, cicl_sum, ficl_sum, n_batch = 0.0, 0.0, 0.0, 0

            for b_start in range(0, len(uids), batch_size):
                batch_uids = uids[b_start:b_start + batch_size]
                contexts_rec, pos_ids, neg_ids = [], [], []
                seqs_a, seqs_b, cl_labels = [], [], []

                for uid in batch_uids:
                    seq = user_seqs[uid]
                    # 主任务：全前缀预测下一物品
                    for t in range(1, len(seq)):
                        contexts_rec.append(seq[:t])
                        pos_ids.append(seq[t])
                        neg = random.randrange(self.n_items)
                        while neg == seq[t]:
                            neg = random.randrange(self.n_items)
                        neg_ids.append(neg)
                    # 对比任务：两次随机增强 + 当前簇标签
                    seqs_a.append(self.aug.random_augment(seq[:-1]))
                    seqs_b.append(self.aug.random_augment(seq[:-1]))
                    cl_labels.append(self.assignments.get(uid, 0))

                optimizer.zero_grad()
                rec_loss  = self._rec_loss(contexts_rec, pos_ids, neg_ids)
                cicl_loss = self._cicl_loss(seqs_a, cl_labels)
                ficl_loss = self._ficl_loss(seqs_a, seqs_b, cl_labels)
                loss = rec_loss + self.lam_cicl * cicl_loss + self.lam_ficl * ficl_loss
                loss.backward()
                optimizer.step()

                rec_sum  += rec_loss.item()
                cicl_sum += cicl_loss.item()
                ficl_sum += ficl_loss.item()
                n_batch  += 1

            nb = max(n_batch, 1)
            print(f"  epoch {epoch:2d}  L_rec={rec_sum/nb:.4f}  "
                  f"L_cicl={cicl_sum/nb:.4f}  L_ficl={ficl_sum/nb:.4f}  "
                  f"L_total={rec_sum/nb + self.lam_cicl*cicl_sum/nb + self.lam_ficl*ficl_sum/nb:.4f}")

    def recommend(self, seq: list[int], top_k: int = 5,
                  exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """序列 Embedding 对全量物品打分（线上等价于 ANN 检索）。"""
        exclude = exclude if exclude is not None else set(seq)
        with torch.no_grad():
            h = self._encode_batch([seq], pool="last")[0]          # [dim]
            scores = self.item_emb.weight[:self.n_items] @ h       # [n_items]
            for iid in exclude:
                scores[iid] = -1e9
            top_vals, top_idx = scores.topk(top_k)
        return [(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_items, n_users, K = 20, 8, 3
    # 合成 4 个兴趣簇的行为序列
    user_seqs: dict[int, list[int]] = {}
    for u in range(n_users):
        cluster_start = (u % 4) * 5
        user_seqs[u] = [(cluster_start + k) % n_items for k in range(8)]

    # 演示 K-Means
    print(f"=== K-Means 聚类演示（K={K}，余弦距离）===")
    dummy_emb = torch.randn(n_users, 16)
    assign, centroids = kmeans(dummy_emb, K=K, n_iter=20)
    for u in range(n_users):
        print(f"  user {u}: cluster={int(assign[u])}")

    model = ICSRec(n_items, dim=16, max_len=10, K=K,
                   tau=0.5, lam_cicl=0.2, lam_ficl=0.2,
                   lr=0.05, proto_update_freq=2)
    print(f"\nICSRec 训练（{n_users} 用户，K={K} 个意图簇）：")
    model.train_model(user_seqs, epochs=8, batch_size=4)

    recs = model.recommend(user_seqs[0], top_k=5)
    print(f"\n用户 0 序列 {user_seqs[0]} 的 Top-5 召回：")
    for iid, sc in recs:
        print(f"  item {iid:2d}  score={sc:.4f}")

    print("\n最终聚类分配：")
    for uid, cid in sorted(model.assignments.items()):
        print(f"  user {uid}: cluster {cid}  seq={user_seqs[uid]}")

    # False Negative Mask 效果：同簇序列的实例级余弦相似度
    with torch.no_grad():
        reps = {u: F.normalize(model._encode_batch([s], pool="mean")[0], dim=-1)
                for u, s in user_seqs.items()}
    same, diff = [], []
    for u in user_seqs:
        for v in user_seqs:
            if u >= v:
                continue
            sim_val = (reps[u] @ reps[v]).item()
            if model.assignments.get(u, -1) == model.assignments.get(v, -2):
                same.append(sim_val)
            else:
                diff.append(sim_val)
    print(f"\n同意图簇序列平均余弦相似度={sum(same)/max(len(same),1):.4f}  "
          f"不同意图簇平均余弦相似度={sum(diff)/max(len(diff),1):.4f}")
