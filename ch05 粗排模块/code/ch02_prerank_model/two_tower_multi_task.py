"""
轻量双塔多任务粗排模型（第 5 章 §5.2 粗排模型）

对应正文概念
------------
正文 §5.2 指出：早期工业界广泛采用双塔模型（图 5.2），由用户塔与物品塔分别编码，
「最终通过向量相似度计算（比如向量点积）获得用户与物品之间的匹配分数」；其优势是
「用户向量和物品向量可以分别离线计算，因此在线阶段只需要进行向量检索和点积运算」。
随后粗排从单目标演进为多目标，「通常会采用多任务学习架构，通过多个任务塔同时输出
不同目标对应的 pxtr 预估值，为后续的多目标融合提供基础输入」。

本实现做了什么
--------------
用 torch 实现一个可训练的轻量双塔 + 多任务塔粗排模型，并从**模块层**验证正文的三个
论断：
1. 「解耦」：物品塔输出可全量离线预计算；本文件对比「在线重算物品塔」与
   「离线缓存物品向量 + 在线仅做点积」两种打分方式的耗时与数值一致性。
2. 「多任务」：共享底座 + 3 个任务头（ctr / cvr / ltr），一次前向出 3 个 pxtr。
3. 「粗排不追求看得最准，而是淘汰得足够准」：用 Recall@K（下游关心的物品是否被保留）
   而非 AUC 作为粗排评价指标，并给出不同 K 下的保留率。

关键公式
--------
    e_u = f_user(x_u),  e_i = f_item(x_i)
    s_t = sigmoid(<e_u ⊙ h_t, e_i>)        （t ∈ {ctr, cvr, ltr}，任务特定门控）
    L   = Σ_t λ_t · BCE(s_t, y_t)
    Recall@K = |Relevant ∩ TopK(score)| / |Relevant|      （§5.7 式）

[MOCK] 组件
-----------
`SyntheticLiveData` 合成直播场景数据（用户/物品特征与 3 个标签），`ItemVectorCache`
模拟线上的物品向量缓存（真实系统是 ANN 索引 / KV 存储）。
"""

import random
import time

import torch
import torch.nn as nn


USER_DIM, ITEM_DIM, EMB_DIM = 16, 12, 24
TASKS = ("ctr", "cvr", "ltr")
# 多任务 loss 权重 λ_t：转化稀疏、点赞次之，给 cvr 更大权重
TASK_LAMBDA = {"ctr": 1.0, "cvr": 1.5, "ltr": 0.8}


# ----------------------------------------------------------------------------
# [MOCK] 合成数据
# ----------------------------------------------------------------------------
class SyntheticLiveData:
    """[MOCK] 直播推荐合成数据集。

    真实标签由一组隐藏的「教师」线性-交互函数生成，保证三个任务既相关又不相同：
        logit_ctr = <u, W_c i> + b_c
        logit_cvr = 0.6*logit_ctr + 0.4*<u, W_v i>   （与 ctr 正相关但不同序）
        logit_ltr = <u, W_l i>                        （与 ctr 近乎独立）
    """

    def __init__(self, n_user: int, n_item: int, gen: torch.Generator):
        self.user_feat = torch.randn(n_user, USER_DIM, generator=gen)
        self.item_feat = torch.randn(n_item, ITEM_DIM, generator=gen)
        self.Wc = torch.randn(USER_DIM, ITEM_DIM, generator=gen) * 0.25
        self.Wv = torch.randn(USER_DIM, ITEM_DIM, generator=gen) * 0.25
        self.Wl = torch.randn(USER_DIM, ITEM_DIM, generator=gen) * 0.25

    def true_logits(self, uid: torch.Tensor, iid: torch.Tensor) -> dict[str, torch.Tensor]:
        u, i = self.user_feat[uid], self.item_feat[iid]
        lc = (u @ self.Wc * i).sum(-1) - 2.0        # 基准 ctr ≈ 12%
        lv = 0.6 * lc + 0.4 * ((u @ self.Wv * i).sum(-1)) - 2.0
        ll = (u @ self.Wl * i).sum(-1) - 1.2
        return {"ctr": lc, "cvr": lv, "ltr": ll}

    def sample(self, n: int, gen: torch.Generator):
        uid = torch.randint(0, self.user_feat.size(0), (n,), generator=gen)
        iid = torch.randint(0, self.item_feat.size(0), (n,), generator=gen)
        logits = self.true_logits(uid, iid)
        labels = {
            t: torch.bernoulli(torch.sigmoid(v), generator=gen)
            for t, v in logits.items()
        }
        return uid, iid, labels


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------
class Tower(nn.Module):
    """轻量塔：2 层 MLP。粗排塔刻意做得很窄（§5.2「模型深度以及网络参数规模均远低于精排」）。"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 归一化，使点积落在 [-1, 1]，便于多任务门控共享同一个尺度
        e = self.net(x)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-8)


class TwoTowerMultiTask(nn.Module):
    """双塔 + 多任务门控。

    任务差异只体现在用户侧的逐元素门控 h_t（每个任务一组可学习的维度权重）与
    任务偏置上，物品塔对所有任务共享 —— 这正是「物品向量可以离线全量预计算」的
    前提：物品侧表达不依赖任务也不依赖用户。
    """

    def __init__(self):
        super().__init__()
        self.user_tower = Tower(USER_DIM, 32, EMB_DIM)
        self.item_tower = Tower(ITEM_DIM, 32, EMB_DIM)
        self.gate = nn.Parameter(torch.ones(len(TASKS), EMB_DIM))
        self.scale = nn.Parameter(torch.full((len(TASKS),), 4.0))
        self.bias = nn.Parameter(torch.tensor([-2.0, -2.4, -1.2]))

    def item_embed(self, item_feat: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_feat)

    def score_from_item_emb(self, user_feat, item_emb) -> dict[str, torch.Tensor]:
        eu = self.user_tower(user_feat)                      # [B, D]
        out = {}
        for k, t in enumerate(TASKS):
            s = (eu * self.gate[k] * item_emb).sum(-1)
            out[t] = torch.sigmoid(self.scale[k] * s + self.bias[k])
        return out

    def forward(self, user_feat, item_feat) -> dict[str, torch.Tensor]:
        return self.score_from_item_emb(user_feat, self.item_embed(item_feat))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# [MOCK] 物品向量缓存
# ----------------------------------------------------------------------------
class ItemVectorCache:
    """[MOCK] 线上物品向量缓存。真实系统中由离线任务天级/小时级刷新到 KV 或 ANN 索引。"""

    def __init__(self, model: TwoTowerMultiTask, item_feat: torch.Tensor):
        with torch.no_grad():
            self.table = model.item_embed(item_feat).clone()
        self.n_item, self.dim = self.table.shape

    def lookup(self, iid: torch.Tensor) -> torch.Tensor:
        return self.table[iid]

    def bytes_fp32(self) -> int:
        return self.n_item * self.dim * 4


# ----------------------------------------------------------------------------
# 训练与评估
# ----------------------------------------------------------------------------
def train(model, data, gen, steps=400, batch=512, lr=5e-3) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    curve = []
    for step in range(steps):
        uid, iid, labels = data.sample(batch, gen)
        pred = model(data.user_feat[uid], data.item_feat[iid])
        loss = sum(TASK_LAMBDA[t] * bce(pred[t].clamp(1e-6, 1 - 1e-6), labels[t])
                   for t in TASKS)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == steps - 1:
            curve.append((step, float(loss)))
    return curve


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """按秩的 AUC（Mann-Whitney U）。"""
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_k(model_score: torch.Tensor, teacher_score: torch.Tensor,
                k: int, n_relevant: int) -> float:
    """§5.7 Recall@K：下游（教师）关心的 Top-n_relevant 物品，有多少被粗排 TopK 保留。"""
    relevant = set(teacher_score.topk(n_relevant).indices.tolist())
    kept = set(model_score.topk(k).indices.tolist())
    return len(relevant & kept) / max(1, n_relevant)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    data = SyntheticLiveData(n_user=2000, n_item=5000, gen=gen)
    model = TwoTowerMultiTask()
    print("=" * 96)
    print("轻量双塔多任务粗排模型（§5.2）")
    print("=" * 96)
    print(f"参数量 = {model.n_params()} （粗排刻意轻量；精排模型通常是其 10^2~10^3 倍）")

    t0 = time.perf_counter()
    curve = train(model, data, gen)
    print(f"训练 400 step 耗时 {time.perf_counter() - t0:.2f}s")
    print("\n[多任务 loss 曲线] step -> L = Σ_t λ_t·BCE")
    for step, loss in curve:
        print(f"  step {step:>4}: {loss:.4f}")

    # ---- 离线评估：三个任务的 AUC ----
    model.eval()
    with torch.no_grad():
        uid, iid, labels = data.sample(20000, gen)
        pred = model(data.user_feat[uid], data.item_feat[iid])
    print("\n[各任务离线 AUC]")
    print(f"{'task':<8}{'λ':>6}{'AUC':>10}{'正样本率':>12}{'平均pxtr':>12}")
    for t in TASKS:
        print(f"{t:<8}{TASK_LAMBDA[t]:>6.1f}{auc(pred[t], labels[t]):>10.4f}"
              f"{float(labels[t].mean()):>12.4f}{float(pred[t].mean()):>12.4f}")

    # ---- 在线打分路径：离线缓存 vs 在线重算 ----
    print("\n" + "=" * 96)
    print("打分路径对比：物品塔离线预计算 + 在线点积  vs  在线重算物品塔（§5.2 解耦优势）")
    print("=" * 96)
    cache = ItemVectorCache(model, data.item_feat)
    cand = torch.randint(0, data.item_feat.size(0), (3000,), generator=gen)
    one_user = data.user_feat[0:1].expand(cand.numel(), -1)

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(10):
            s_online = model(one_user, data.item_feat[cand])
        t_online = (time.perf_counter() - t0) / 10 * 1000
        t0 = time.perf_counter()
        for _ in range(10):
            s_cached = model.score_from_item_emb(one_user, cache.lookup(cand))
        t_cached = (time.perf_counter() - t0) / 10 * 1000

    max_diff = max(float((s_online[t] - s_cached[t]).abs().max()) for t in TASKS)
    print(f"候选量 = {cand.numel()}")
    print(f"在线重算物品塔     : {t_online:7.3f} ms/请求")
    print(f"离线缓存 + 在线点积 : {t_cached:7.3f} ms/请求  （加速 {t_online / t_cached:.2f}x）")
    print(f"两路打分最大数值差 : {max_diff:.2e}  （物品塔与任务/用户无关 => 数值完全一致）")
    print(f"物品向量表大小     : {cache.n_item} x {cache.dim} fp32 "
          f"= {cache.bytes_fp32() / 1024:.0f} KB")

    # ---- 粗排真正的指标：Recall@K ----
    print("\n" + "=" * 96)
    print("粗排评价：不是「看得最准」而是「淘汰得足够准」—— Recall@K（§5.2 / §5.7）")
    print("=" * 96)
    N_CAND, N_RELEVANT = 5000, 50
    with torch.no_grad():
        all_items = torch.arange(N_CAND) % data.item_feat.size(0)
        uf = data.user_feat[7:8].expand(N_CAND, -1)
        p = model(uf, data.item_feat[all_items])
        # 粗排融合分（线性，§5.4.1）
        prerank_score = 0.5 * p["ctr"] + 0.3 * p["cvr"] + 0.2 * p["ltr"]
        # 教师 = 数据生成过程的真实期望收益（模拟下游精排的偏好）
        tl = data.true_logits(torch.zeros(N_CAND, dtype=torch.long) + 7, all_items)
        teacher = (0.5 * torch.sigmoid(tl["ctr"]) + 0.3 * torch.sigmoid(tl["cvr"])
                   + 0.2 * torch.sigmoid(tl["ltr"]))
    print(f"候选集 N={N_CAND}, 下游真正偏好的 Relevant = Top{N_RELEVANT}")
    print(f"{'K':>7}{'压缩比':>10}{'Recall@K':>12}")
    for k in (100, 300, 500, 1000, 2000):
        r = recall_at_k(prerank_score, teacher, k, N_RELEVANT)
        print(f"{k:>7}{k / N_CAND:>10.1%}{r:>12.4f}")
    print("\n结论：K=300（正文 §5.6.3 的典型截断值）即把候选压到 6%，"
          f"仍保留了下游偏好内容的 {recall_at_k(prerank_score, teacher, 300, N_RELEVANT):.0%}。"
          "\n这就是「淘汰得足够准」的含义：粗排允许 Top1 排错，但不允许把优质内容整体丢掉。")
