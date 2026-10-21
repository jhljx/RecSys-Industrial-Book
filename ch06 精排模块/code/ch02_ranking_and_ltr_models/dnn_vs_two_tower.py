"""
精排 DNN 主模型 vs 粗排双塔模型（第 6 章 §6.2 精排模型与 LTR 模型）

对应正文概念：
正文指出，粗排用双塔（用户塔/物品塔分别编码，最后内积），精排则把
「用户侧特征 + 物品侧特征 + 上下文特征统一编码并拼接（Concatenate）后
输入 DNN」，因而能在网络内部自动学习高阶特征交叉；精排主模型采用
多任务学习（MTL）框架联合预估 click / effective_view / like / comment /
follow / share 等多个行为目标。正文同时强调：两者差异源于「效果、算力、
时延之间的工程权衡」，而非建模能力的本质差异。

本实现做了什么：
1. 用 torch 构造一份带「用户年龄 × 内容类目」交叉效应的合成数据——这是正文
   举的例子，双塔的后期内积结构很难拟合，而拼接式 DNN 可以。
2. 同时训练双塔模型与拼接式 DNN 多任务模型，对比 AUC 与参数量/推理 FLOPs，
   量化「精排为什么更准、为什么更贵」。
3. DNN 侧用共享底座 + 每任务独立塔头的 MTL 结构，输出 6 个 pxtr。

关键公式：
    双塔：    score = <f_user(x_u), f_item(x_i)>
    精排DNN： h = MLP([x_u ; x_i ; x_c]),  p_k = sigmoid(w_k^T h + b_k)
    MTL 损失：L = sum_k lambda_k * BCE(p_k, y_k)
"""

import math
import random

import torch
import torch.nn as nn

NUM_AGE_BUCKET = 32  # 用户侧离散特征（年龄分桶 × 人群标签的组合桶）
NUM_CATEGORY = 40    # 物品侧离散特征（一级类目 × 二级类目）
NUM_CTX = 4          # 上下文：时段
INTERACT_DIM = 8     # 双塔内积维度 = 粗排的在线算力预算，受工程约束
TASKS = ["click", "effective_view", "like", "comment", "follow", "share"]
# 正文 §6.3.2 给出的稀疏度量级：CTR 3%~10%，LTR <1%，CMTR <1‰，FTR <1‱
TASK_BASE_RATE = [0.08, 0.40, 0.009, 0.0009, 0.0004, 0.002]


# --------------------------------------------------------------------------
# 合成数据：核心是「用户年龄 × 内容类目」的高阶交叉
# --------------------------------------------------------------------------


def _calibrate_bias(s: torch.Tensor, target: float) -> float:
    """二分搜索偏置，使 mean(sigmoid(bias + s)) ≈ target（对齐正文给出的稀疏度）。"""
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if torch.sigmoid(mid + s).mean().item() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_dataset(n: int, gen: torch.Generator):
    age = torch.randint(0, NUM_AGE_BUCKET, (n,), generator=gen)
    cat = torch.randint(0, NUM_CATEGORY, (n,), generator=gen)
    ctx = torch.randint(0, NUM_CTX, (n,), generator=gen)
    u_dense = torch.rand(n, 4, generator=gen)
    i_dense = torch.rand(n, 4, generator=gen)

    # (1) 满秩的 (年龄段 × 类目) 交叉矩阵。双塔最终只能用 emb 维内积逼近它，
    #     即只能表达秩 <= EMB_DIM 的双线性形式；当交叉矩阵秩远高于 EMB_DIM 时，
    #     双塔结构性地欠拟合。
    g2 = torch.Generator().manual_seed(7)
    cross = torch.randn(NUM_AGE_BUCKET, NUM_CATEGORY, generator=g2) * 1.5
    age_main = torch.randn(NUM_AGE_BUCKET, generator=g2) * 0.5   # 可加主效应
    cat_main = torch.randn(NUM_CATEGORY, generator=g2) * 0.5

    # (2) 稠密特征上的非双线性交互：L1 距离 |u - i| 无法写成任何有限维的
    #     <f(u), g(i)> 形式，因此双塔的「后期内积」结构天然学不到。
    dense_cross = -2.2 * (u_dense - i_dense).abs().sum(1)

    logit_core = (
        cross[age, cat] + age_main[age] + cat_main[cat]
        + dense_cross + 0.25 * ctx.float()
    )
    labels = torch.zeros(n, len(TASKS))
    for k, base in enumerate(TASK_BASE_RATE):
        s = (0.9 + 0.15 * k) * logit_core
        p = torch.sigmoid(_calibrate_bias(s, base) + s)
        labels[:, k] = torch.bernoulli(p, generator=gen)
    # oracle 不参与训练，只用于报告「理论 AUC 上限」
    return {"age": age, "cat": cat, "ctx": ctx, "oracle": logit_core,
            "u_dense": u_dense, "i_dense": i_dense, "y": labels}


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------


class TwoTower(nn.Module):
    """粗排双塔：两侧独立编码，最后仅靠 INTERACT_DIM 维内积交互。

    注意内积维度 INTERACT_DIM 是粗排的工程预算（决定向量检索/打分开销），
    不能随意放大。而 score 对 (age, cat) 的依赖只能写成
        s(age, cat) = <f(age, ...), g(cat, ...)>
    这是一个秩 <= INTERACT_DIM 的双线性型；当真值交叉矩阵秩更高时，
    双塔就是结构性欠拟合，加宽隐层也救不回来。
    """

    def __init__(self, emb=16, hid=128, n_tasks=len(TASKS)):
        super().__init__()
        self.age_emb = nn.Embedding(NUM_AGE_BUCKET, emb)
        self.cat_emb = nn.Embedding(NUM_CATEGORY, emb)
        self.ctx_emb = nn.Embedding(NUM_CTX, emb)
        self.user_tower = nn.Sequential(
            nn.Linear(emb * 2 + 4, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, INTERACT_DIM))
        self.item_tower = nn.Sequential(
            nn.Linear(emb + 4, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, INTERACT_DIM))
        # 每个任务一组缩放/偏置，让双塔也有多任务输出能力
        self.scale = nn.Parameter(torch.ones(n_tasks))
        self.bias = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, b):
        u = self.user_tower(torch.cat(
            [self.age_emb(b["age"]), self.ctx_emb(b["ctx"]), b["u_dense"]], 1))
        i = self.item_tower(torch.cat([self.cat_emb(b["cat"]), b["i_dense"]], 1))
        dot = (u * i).sum(1, keepdim=True)
        return dot * self.scale + self.bias


class RankingDNN(nn.Module):
    """精排 DNN：全部特征拼接 → 共享底座 → 每任务独立塔头（MTL）。"""

    def __init__(self, emb=16, hid=(128, 64), n_tasks=len(TASKS)):
        super().__init__()
        self.age_emb = nn.Embedding(NUM_AGE_BUCKET, emb)
        self.cat_emb = nn.Embedding(NUM_CATEGORY, emb)
        self.ctx_emb = nn.Embedding(NUM_CTX, emb)
        in_dim = emb * 3 + 8
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hid[0]), nn.ReLU(),
            nn.Linear(hid[0], hid[1]), nn.ReLU())
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hid[1], 32), nn.ReLU(), nn.Linear(32, 1))
             for _ in range(n_tasks)])

    def forward(self, b):
        x = torch.cat([self.age_emb(b["age"]), self.cat_emb(b["cat"]),
                       self.ctx_emb(b["ctx"]), b["u_dense"], b["i_dense"]], 1)
        h = self.shared(x)
        return torch.cat([head(h) for head in self.heads], 1)


# --------------------------------------------------------------------------
# 训练 / 评估
# --------------------------------------------------------------------------


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """按秩计算 AUC；全正或全负时返回 nan。"""
    pos = labels.sum().item()
    neg = labels.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    return ((ranks[labels == 1].sum().item() - pos * (pos + 1) / 2)
            / (pos * neg))


def train(model, data, epochs=12, bs=512, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n = data["y"].shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            batch = {k: v[idx] for k, v in data.items()}
            logits = model(batch)
            loss = sum(lossf(logits[:, k], batch["y"][:, k])
                       for k in range(len(TASKS)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    train_data = make_dataset(40000, gen)
    test_data = make_dataset(10000, gen)
    print("=" * 72)
    print("精排 DNN vs 粗排双塔：拟合「用户年龄 × 内容类目」高阶交叉")
    print("=" * 72)
    print(f"训练集 {train_data['y'].shape[0]} 条，测试集 {test_data['y'].shape[0]} 条")
    print("各任务正样本率（测试集）：")
    for k, t in enumerate(TASKS):
        r = test_data["y"][:, k].mean().item()
        print(f"  {t:<16}{r * 100:7.3f}%   （设定基准 {TASK_BASE_RATE[k] * 100:.3f}%）")

    results = {}
    for name, model in (("TwoTower(双塔)", TwoTower()),
                        ("RankingDNN(精排)", RankingDNN())):
        train(model, train_data)
        model.eval()
        with torch.no_grad():
            logits = model(test_data)
        results[name] = [auc(logits[:, k], test_data["y"][:, k])
                         for k in range(len(TASKS))]
        print(f"\n{name}  参数量 = {n_params(model):,}")

    print("\n" + "=" * 78)
    print("测试集 AUC 对比（Oracle = 用真值 logit 排序的理论上限）")
    print("=" * 78)
    names = list(results)
    print(f"{'任务':<16}" + "".join(f"{k:>18}" for k in names)
          + f"{'ΔAUC':>9}{'Oracle':>9}")
    for k, t in enumerate(TASKS):
        a, b = results[names[0]][k], results[names[1]][k]
        o = auc(test_data["oracle"], test_data["y"][:, k])
        print(f"{t:<16}{a:>18.4f}{b:>18.4f}{b - a:>+9.4f}{o:>9.4f}")
    ma = sum(results[names[0]]) / len(TASKS)
    mb = sum(results[names[1]]) / len(TASKS)
    mo = sum(auc(test_data["oracle"], test_data["y"][:, k])
             for k in range(len(TASKS))) / len(TASKS)
    print(f"{'平均':<16}{ma:>18.4f}{mb:>18.4f}{mb - ma:>+9.4f}{mo:>9.4f}")
    print(f"\n距 Oracle 上限的差距：双塔 {mo - ma:.4f}，精排 DNN {mo - mb:.4f}"
          f"（精排收回了 {(mb - ma) / (mo - ma) * 100:.1f}% 的可弥补差距）")

    # 在线单候选算力：双塔的物品塔可离线预计算、用户塔每次请求只算一次，
    # 因此每多一个候选只需 INTERACT_DIM 次乘加；精排 DNN 每个候选都要跑全网络。
    dnn = RankingDNN()
    dnn_flops = sum(m.in_features * m.out_features
                    for m in dnn.modules() if isinstance(m, nn.Linear))
    print(f"\n在线单候选打分乘加次数：双塔 ≈ {INTERACT_DIM}（物品向量离线预算好，"
          f"只剩内积）\n                        精排 DNN ≈ {dnn_flops:,}"
          f"（每个候选跑一遍全网络，约 {dnn_flops // INTERACT_DIM} 倍）")

    print("\n结论：双塔把两侧特征压成 INTERACT_DIM="
          f"{INTERACT_DIM} 维后才内积，只能表达秩 ≤ {INTERACT_DIM} 的双线性"
          "交叉，\n      且对 |u-i| 这类非双线性关系无能为力；拼接式 DNN 在输入层"
          "就看到两侧特征，\n      AUC 更高，代价是每候选在线算力高出三个数量级——"
          "这正是正文所说的\n      「效果 / 算力 / 时延」工程权衡：粗排候选上千只能用双塔，"
          "精排候选几百才用得起 DNN。")
