"""
双塔召回负采样偏差处理（第10章 §10.3.2）

核心思想：
- In-Batch 负采样会让热门物品以更高概率成为别人的负样本，导致热门物品被"过度惩罚"，
  Batch Softmax 与全局 Softmax 之间产生采样偏差（Sampling Bias）。
- LogQ 矫正：s_c(x_i, y_j) = s(x_i, y_j) - log p_j，用采样概率的对数抵消高频采样偏差。
- 流式频率估计：不直接计数，而是用哈希数组记录物品两次被采样之间的 Step Gap，用 EMA 在线
  更新平均间隔 δ，则 p_j ≈ 1/δ；借鉴 Count-Min Sketch 的多重哈希思想缓解哈希冲突。
- 训练技巧：Embedding L2 归一化（学余弦相似度）+ 温度系数 τ + 混合负采样（Batch 内负样本
  + 全局均匀负样本）。

论文出处：Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
Recommendations, RecSys 2019.（PyTorch autograd 实现）
"""

import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class StreamingFrequencyEstimator:
    """
    流式频率估计器：在线估计物品进入 Batch 的采样概率 p_j ≈ 1/δ。

    多重哈希时取各哈希桶 δ 的最大值（等价于取 p 的最小值），以抑制哈希冲突导致的
    频率高估，这与 Count-Min Sketch 取 min 的思想一致。
    """

    def __init__(self, n_hash: int = 2, n_bucket: int = 64, alpha: float = 0.05, init_delta: float = 100.0):
        self.n_bucket, self.alpha = n_bucket, alpha
        self.A = [[0] * n_bucket for _ in range(n_hash)]
        self.B = [[init_delta] * n_bucket for _ in range(n_hash)]
        self.primes = [(73, 19), (151, 37), (257, 53)][:n_hash]

    def _buckets(self, item: int) -> list:
        return [(item * p + q) % self.n_bucket for p, q in self.primes]

    def update(self, item: int, step: int) -> None:
        for h, idx in enumerate(self._buckets(item)):
            gap = float(step - self.A[h][idx]) if self.A[h][idx] > 0 else self.B[h][idx]
            self.B[h][idx] = (1.0 - self.alpha) * self.B[h][idx] + self.alpha * max(gap, 1.0)
            self.A[h][idx] = step

    def prob(self, item: int) -> float:
        delta = max(self.B[h][idx] for h, idx in enumerate(self._buckets(item)))
        return min(1.0, 1.0 / max(delta, 1.0))


class Tower(nn.Module):
    """单塔 MLP：emb_dim → hidden → output_dim（ReLU 激活，最后一层线性输出）。"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DebiasedTwoTower(nn.Module):
    """
    带采样偏差修正的双塔召回模型。

    logits[c] = <u_hat, v_hat_c> / τ - log p_c      （LogQ 矫正）
    L_B = -1/B Σ_i log softmax(logits)[i]            （矫正后的 Batch Softmax）
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 16, hidden: int = 16,
                 out_dim: int = 16, temperature: float = 0.1,
                 logq_correction: bool = True, n_global_neg: int = 4):
        super().__init__()
        self.n_items = n_items
        self.tau = temperature
        self.logq = logq_correction
        self.n_global_neg = n_global_neg

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_tower = Tower(emb_dim, hidden, out_dim)
        self.item_tower = Tower(emb_dim, hidden, out_dim)
        self.freq = StreamingFrequencyEstimator()
        self.step = 0

    def _encode_user(self, uid_tensor: torch.Tensor) -> torch.Tensor:
        x = self.user_emb(uid_tensor)
        out = self.user_tower(x)
        return F.normalize(out, dim=-1)

    def _encode_item(self, iid_tensor: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(iid_tensor)
        out = self.item_tower(x)
        return F.normalize(out, dim=-1)

    def forward(self, uids: torch.Tensor, iids: torch.Tensor,
                log_p: torch.Tensor) -> torch.Tensor:
        """
        uids: (B,)  iids: (C,) 候选物品（正样本 + 全局负样本）
        log_p: (C,) 各候选物品的 log 采样概率
        正样本 label = 对角线：label[i] = i
        """
        B = uids.shape[0]
        u_hat = self._encode_user(uids)    # (B, d)
        v_hat = self._encode_item(iids)    # (C, d)

        # 相似度矩阵 (B, C)
        sim = u_hat @ v_hat.T / self.tau
        if self.logq:
            sim = sim - log_p.unsqueeze(0)  # (B, C) - (1, C) = (B, C)

        # 正样本为对角线（前 B 个候选）
        labels = torch.arange(B)
        return F.cross_entropy(sim, labels)

    def train_model(self, interactions: list, epochs: int = 8,
                    batch_size: int = 8, lr: float = 0.08) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        data = list(interactions)
        for ep in range(epochs):
            random.shuffle(data)
            tot, nb = 0.0, 0
            self.train()
            for st in range(0, len(data), batch_size):
                bt = data[st: st + batch_size]
                if len(bt) < 2:
                    continue
                self.step += 1
                B = len(bt)

                # 混合负采样：Batch 内正样本物品 + 全局均匀负样本
                batch_iids = [iid for _, iid in bt]
                global_neg = [random.randrange(self.n_items) for _ in range(self.n_global_neg)]
                cand_iids = batch_iids + global_neg

                for iid in cand_iids:
                    self.freq.update(iid, self.step)

                # log 采样概率：Batch 内用流式估计，全局负样本用均匀概率
                log_p_list = [math.log(max(self.freq.prob(iid), 1e-6)) for iid in batch_iids] + \
                             [math.log(1.0 / self.n_items)] * len(global_neg)

                uids_t = torch.tensor([u for u, _ in bt], dtype=torch.long)
                iids_t = torch.tensor(cand_iids, dtype=torch.long)
                log_p_t = torch.tensor(log_p_list, dtype=torch.float)

                optimizer.zero_grad()
                loss = self.forward(uids_t, iids_t, log_p_t)
                loss.backward()
                optimizer.step()
                tot += loss.item()
                nb += 1
            print(f"    epoch {ep:2d}  avg_loss={tot / max(nb, 1):.4f}")

    def warmup_frequency(self, interactions: list, batch_size: int = 8) -> None:
        data = list(interactions)
        for _ in range(6):
            random.shuffle(data)
            for st in range(0, len(data), batch_size):
                self.step += 1
                for _, i in data[st: st + batch_size]:
                    self.freq.update(i, self.step)

    def recommend(self, uid: int, top_k: int = 10) -> list:
        self.eval()
        with torch.no_grad():
            u_hat = self._encode_user(torch.tensor([uid]))   # (1, d)
            all_ids = torch.arange(self.n_items)
            v_hat = self._encode_item(all_ids)               # (n, d)
            scores = (u_hat @ v_hat.T).squeeze(0).tolist()
            pairs = list(zip(range(self.n_items), scores))
            return sorted(pairs, key=lambda x: -x[1])[:top_k]


def _hot_item_ratio(m: DebiasedTwoTower, n_users: int, hot: set, top_k: int = 5) -> float:
    """统计召回结果 Top-K 中热门物品的占比。"""
    return sum(1 for u in range(n_users) for i, _ in m.recommend(u, top_k) if i in hot) / (n_users * top_k)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, batch_size = 12, 30, 8

    # 物品流行度服从长尾分布（Zipf）：物品 0、1、2 ... 依次变冷
    pop = [1.0 / (rank + 1) ** 1.3 for rank in range(n_items)]
    cum = [sum(pop[:i + 1]) for i in range(n_items)]
    interactions = []
    for uid in range(n_users):
        for _ in range(24):
            r = random.random() * cum[-1]
            interactions.append((uid, next(i for i in range(n_items) if cum[i] >= r)))

    emp, total = Counter(iid for _, iid in interactions), len(interactions)
    rank_by_pop = sorted(range(n_items), key=lambda k: -emp[k])
    hot_items = set(rank_by_pop[:5])
    print(f"合成数据：{n_users} 用户 / {n_items} 物品 / {total} 条交互（长尾 Zipf 分布）")
    print(f"真实曝光最高的 5 个热门物品：{sorted(hot_items)}（理想 Top-5 占比 = 100%）")

    models = {}
    for name, use_logq in [("A 未矫正（原始 In-Batch 负采样）", False),
                           ("B LogQ 矫正 + 流式频率估计 + 混合负采样", True)]:
        print(f"\n【模型 {name}】")
        random.seed(42)
        torch.manual_seed(42)
        m = DebiasedTwoTower(n_users, n_items, logq_correction=use_logq)
        m.warmup_frequency(interactions, batch_size)
        m.train_model(interactions, epochs=8, batch_size=batch_size, lr=0.08)
        models[use_logq] = m
    print("  （注：A 与 B 优化的是不同的 logits，loss 数值不可直接横向比较）")

    print("\n流式频率估计效果（p_j ≈ 1/δ，对照「物品出现在一个 Batch 中的真实概率」）：")
    seen = [i for i in rank_by_pop if emp[i] > 0]
    for iid in seen[:3] + seen[-2:]:
        f = emp[iid] / total
        p_true = 1.0 - (1.0 - f) ** batch_size
        tag = "热门" if iid in hot_items else "长尾"
        print(f"  item {iid:2d} [{tag}]  估计 p={models[True].freq.prob(iid):.4f}  "
              f"真实 p={p_true:.4f}  样本频率={f:.4f}")
    print("  （长尾物品估计值偏高源于哈希冲突，增大桶数或哈希个数可进一步降低误差）")

    ra = _hot_item_ratio(models[False], n_users, hot_items, top_k=5)
    rb = _hot_item_ratio(models[True], n_users, hot_items, top_k=5)
    print("\n召回结果 Top-5 中热门物品的占比（理想值 100%）：")
    print(f"  模型 A（未矫正）  ：{ra:.2%}\n  模型 B（LogQ 矫正）：{rb:.2%}")
    print("  结论：未矫正时热门物品被反复当作负样本而遭过度惩罚、排序被压低；LogQ 矫正把 logit")
    print(f"        减去 log p_j 抵消该偏差，占比变化 {rb - ra:+.2%}，Batch Softmax 更接近全局 Softmax。")
