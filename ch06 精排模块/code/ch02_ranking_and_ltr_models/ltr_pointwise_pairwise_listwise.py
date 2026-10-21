"""
LTR 排序学习：Pointwise / Pairwise / Listwise 最小对比
（第 6 章 §6.2 精排模型与 LTR 模型）

对应正文概念：
正文指出 LTR 模型「不直接预测用户行为标签，而是基于业务经验构造排序
Reward 或排序监督信号（用户综合满意度、长周期价值、高价值行为组合），
通过排序学习的方式直接优化最终排序效果」。本脚本就围绕这个「排序
Reward」标签，把 LTR 的三大范式放在同一份数据、同一个打分网络上对比。

注意：正文 §6.2 只提出了「LTR 直接优化排序效果」这一思想，没有点名具体
算法。本脚本选用学界最经典的三条基线（RankNet / BPR / ListNet /
LambdaRank），论文出处见 README。

本实现做了什么：
用同一个 3 层 MLP 打分器 f(x)，分别用 5 种损失训练，在同一测试集上比较
NDCG@10 / NDCG@5 / MAP：
    1. Pointwise MSE      ：把 Reward 当回归值拟合
    2. Pointwise BCE      ：把「高 Reward」当二分类
    3. Pairwise RankNet   ：文档对的概率排序损失
    4. Pairwise BPR       ：正负样本对的 BPR 损失
    5. Listwise ListNet   ：Top-1 概率分布的交叉熵
    6. Pairwise LambdaRank：RankNet 梯度 × |ΔNDCG|，直接对齐排序指标

关键公式：
    RankNet:    P_ij = sigmoid(s_i - s_j),  L = -log P_ij  (当 y_i > y_j)
    BPR:        L = -log sigmoid(s_pos - s_neg)
    ListNet:    P(i) = exp(s_i)/sum_j exp(s_j),  L = -sum_i P_y(i) log P_s(i)
    LambdaRank: lambda_ij = |ΔNDCG_ij| * (-sigmoid(-(s_i - s_j)))
    NDCG@k:     DCG@k = sum_{i<=k} (2^{rel_i} - 1)/log2(i+1),  NDCG = DCG/IDCG
"""

import math
import random

import torch
import torch.nn as nn

FEAT_DIM = 12
LIST_LEN = 20          # 每个请求（query）下的候选物品数
N_TRAIN_LIST = 600
N_TEST_LIST = 200
MAX_REL = 4            # 相关度等级 0..4，由业务 Reward 分档而来


# --------------------------------------------------------------------------
# 合成数据：每个 query 一组候选，标签是「业务排序 Reward」在 query 内分档
#
# 关键设定（真实 LTR 场景）：不同请求的候选池质量差异很大（m_q 代表这一批
# 候选的整体水位），而排序标签是「query 内相对好坏」的档位。也就是说
# 「绝对特征值 -> 档位」的映射依赖同 query 其它候选，模型单看一条样本是看
# 不出来的。Pairwise/Listwise 只需学到 query 内的相对顺序，对 m_q 这种
# query 级平移天然免疫；Pointwise 要去拟合绝对档位值，就会被 m_q 混淆。
# --------------------------------------------------------------------------


def make_lists(n_list: int, gen: torch.Generator):
    """返回 (X, rel)，X:[n_list, LIST_LEN, FEAT_DIM]，rel:[n_list, LIST_LEN]。"""
    # m_q：query 级候选池水位（有的请求候选整体优质，有的整体平庸）
    m_q = torch.randn(n_list, 1, 1, generator=gen) * 1.5
    X = torch.randn(n_list, LIST_LEN, FEAT_DIM, generator=gen) + m_q
    g2 = torch.Generator().manual_seed(11)
    w_lin = torch.randn(FEAT_DIM, generator=g2)
    w_sq = torch.randn(FEAT_DIM, generator=g2) * 0.5
    # 业务 Reward：时长项 + 互动项 + 非线性交叉项（模拟综合满意度）
    reward = (X @ w_lin
              + (X ** 2) @ w_sq
              + 1.2 * X[..., 0] * X[..., 1]
              - 0.8 * X[..., 2].abs())
    reward = reward + 0.6 * torch.randn_like(reward)   # 行为噪声
    # 按 query 内分位数分档到 0..MAX_REL（工业上常把 Reward 离散成档位做 LTR）
    rel = torch.zeros_like(reward)
    order = reward.argsort(dim=1, descending=True)
    # 档位配比 5% / 10% / 15% / 25% / 45%，头部稀缺
    cuts = [int(LIST_LEN * c) for c in (0.05, 0.15, 0.30, 0.55)]
    for li in range(n_list):
        for pos, idx in enumerate(order[li]):
            if pos < cuts[0]:
                rel[li, idx] = 4
            elif pos < cuts[1]:
                rel[li, idx] = 3
            elif pos < cuts[2]:
                rel[li, idx] = 2
            elif pos < cuts[3]:
                rel[li, idx] = 1
    return X, rel


# --------------------------------------------------------------------------
# 评估指标
# --------------------------------------------------------------------------


def dcg(rels: torch.Tensor, k: int) -> float:
    return sum((2.0 ** r.item() - 1) / math.log2(i + 2)
               for i, r in enumerate(rels[:k]))


def ndcg_at_k(scores: torch.Tensor, rel: torch.Tensor, k: int) -> float:
    ideal = dcg(rel.sort(descending=True).values, k)
    if ideal == 0:
        return 0.0
    return dcg(rel[scores.argsort(descending=True)], k) / ideal


def map_score(scores: torch.Tensor, rel: torch.Tensor) -> float:
    """以 rel >= 3 为「相关」计算 Average Precision。"""
    order = scores.argsort(descending=True)
    hits, ap = 0, 0.0
    binrel = (rel >= 3).float()
    total = binrel.sum().item()
    if total == 0:
        return 0.0
    for i, idx in enumerate(order):
        if binrel[idx] > 0:
            hits += 1
            ap += hits / (i + 1)
    return ap / total


def evaluate(model, X, rel):
    model.eval()
    with torch.no_grad():
        s = model(X).squeeze(-1)           # [n_list, LIST_LEN]
    n = X.shape[0]
    n10 = sum(ndcg_at_k(s[i], rel[i], 10) for i in range(n)) / n
    n5 = sum(ndcg_at_k(s[i], rel[i], 5) for i in range(n)) / n
    mp = sum(map_score(s[i], rel[i]) for i in range(n)) / n
    model.train()
    return n10, n5, mp


# --------------------------------------------------------------------------
# 打分器（五种损失共用同一结构，保证对比公平）
# --------------------------------------------------------------------------


def make_scorer() -> nn.Module:
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(FEAT_DIM, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1))


# --------------------------------------------------------------------------
# 五种 LTR 损失
# --------------------------------------------------------------------------


def loss_pointwise_mse(s, rel):
    return ((s - rel) ** 2).mean()


def loss_pointwise_bce(s, rel):
    y = (rel >= 3).float()      # 「高 Reward」作为正样本
    return nn.functional.binary_cross_entropy_with_logits(s, y)


def _pair_masks(rel):
    """返回 [B, L, L] 的 mask，(i, j) 为 True 表示 rel_i > rel_j。"""
    return (rel.unsqueeze(2) > rel.unsqueeze(1))


def loss_ranknet(s, rel):
    diff = s.unsqueeze(2) - s.unsqueeze(1)        # s_i - s_j
    mask = _pair_masks(rel)
    if mask.sum() == 0:
        return s.sum() * 0.0
    # L = -log sigmoid(s_i - s_j) = softplus(-(s_i - s_j))
    return nn.functional.softplus(-diff)[mask].mean()


def loss_bpr(s, rel):
    """BPR：每个 list 内随机采一对 (高档, 低档) 做对比。"""
    B, L = s.shape
    pos_mask = rel >= 3
    neg_mask = rel == 0
    losses = []
    for b in range(B):
        pi = pos_mask[b].nonzero().flatten()
        ni = neg_mask[b].nonzero().flatten()
        if len(pi) == 0 or len(ni) == 0:
            continue
        p = pi[torch.randint(len(pi), (len(pi),))]
        n = ni[torch.randint(len(ni), (len(pi),))]
        losses.append(nn.functional.softplus(-(s[b, p] - s[b, n])))
    if not losses:
        return s.sum() * 0.0
    return torch.cat(losses).mean()


def loss_listnet(s, rel):
    """ListNet Top-1 概率分布交叉熵。"""
    p_true = torch.softmax(rel, dim=1)
    log_p_pred = torch.log_softmax(s, dim=1)
    return -(p_true * log_p_pred).sum(1).mean()


def loss_lambdarank(s, rel):
    """LambdaRank：RankNet pairwise 损失按 |ΔNDCG| 加权。"""
    B, L = s.shape
    gains = 2.0 ** rel - 1.0
    # 当前排名（按预估分）
    order = s.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(L, device=s.device).expand(B, L)
    ranks.scatter_(1, order, ar)
    disc = 1.0 / torch.log2(ranks.float() + 2.0)        # [B, L]
    idcg = torch.stack([
        torch.tensor(dcg(rel[b].sort(descending=True).values, L))
        for b in range(B)]).clamp(min=1e-6).unsqueeze(1).unsqueeze(2)

    # 交换 i, j 两个位置带来的 NDCG 变化量
    dg = gains.unsqueeze(2) - gains.unsqueeze(1)        # g_i - g_j
    dd = disc.unsqueeze(2) - disc.unsqueeze(1)          # d_i - d_j
    delta_ndcg = (dg * dd).abs() / idcg.squeeze(2).unsqueeze(2)

    diff = s.unsqueeze(2) - s.unsqueeze(1)
    mask = _pair_masks(rel)
    if mask.sum() == 0:
        return s.sum() * 0.0
    weighted = nn.functional.softplus(-diff) * delta_ndcg.detach()
    return weighted[mask].mean() * 10.0     # 放大量级便于与其它损失同 lr 训练


METHODS = [
    ("Pointwise-MSE", loss_pointwise_mse, "回归拟合 Reward 数值"),
    ("Pointwise-BCE", loss_pointwise_bce, "二分类拟合「高 Reward」"),
    ("Pairwise-RankNet", loss_ranknet, "全量文档对的概率排序损失"),
    ("Pairwise-BPR", loss_bpr, "正负样本对的 BPR 损失"),
    ("Listwise-ListNet", loss_listnet, "Top-1 概率分布交叉熵"),
    ("Pairwise-LambdaRank", loss_lambdarank, "RankNet 梯度 × |ΔNDCG|"),
]


def train_one(loss_fn, Xtr, reltr, epochs=40, bs=64, lr=2e-3):
    model = make_scorer()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for st in range(0, n, bs):
            idx = perm[st:st + bs]
            s = model(Xtr[idx]).squeeze(-1)
            loss = loss_fn(s, reltr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    Xtr, reltr = make_lists(N_TRAIN_LIST, gen)
    Xte, relte = make_lists(N_TEST_LIST, gen)

    print("=" * 84)
    print("LTR 三大范式对比（同一打分网络，仅损失函数不同）")
    print("=" * 84)
    print(f"训练 {N_TRAIN_LIST} 个 query × {LIST_LEN} 候选，"
          f"测试 {N_TEST_LIST} 个 query × {LIST_LEN} 候选")
    dist = [int((reltr == r).sum().item()) for r in range(MAX_REL + 1)]
    print("训练集相关度档位分布 rel=0..4：" + ", ".join(
        f"{r}:{c}({c / reltr.numel():.0%})" for r, c in enumerate(dist)))

    # 随机打分基线
    torch.manual_seed(0)
    rnd = torch.randn(N_TEST_LIST, LIST_LEN)
    base10 = sum(ndcg_at_k(rnd[i], relte[i], 10)
                 for i in range(N_TEST_LIST)) / N_TEST_LIST
    base5 = sum(ndcg_at_k(rnd[i], relte[i], 5)
                for i in range(N_TEST_LIST)) / N_TEST_LIST
    basemap = sum(map_score(rnd[i], relte[i])
                  for i in range(N_TEST_LIST)) / N_TEST_LIST

    print("\n" + "=" * 84)
    print(f"{'方法':<22}{'NDCG@10':>10}{'NDCG@5':>10}{'MAP':>10}   说明")
    print("=" * 84)
    print(f"{'Random(基线)':<22}{base10:>10.4f}{base5:>10.4f}{basemap:>10.4f}"
          f"   随机打分")

    rows = []
    for name, fn, desc in METHODS:
        torch.manual_seed(42)
        model = train_one(fn, Xtr, reltr)
        n10, n5, mp = evaluate(model, Xte, relte)
        rows.append((name, n10, n5, mp))
        print(f"{name:<22}{n10:>10.4f}{n5:>10.4f}{mp:>10.4f}   {desc}")

    best = max(rows, key=lambda r: r[1])
    print("=" * 84)
    print(f"最佳：{best[0]}，NDCG@10 = {best[1]:.4f}"
          f"（较随机基线 +{best[1] - base10:.4f}）")
    print("\n观察：Pointwise 只拟合绝对值、不关心 query 内相对顺序；Pairwise 显式"
          "优化两两序关系；\n      Listwise/LambdaRank 把整个列表（乃至 NDCG 指标"
          "本身）纳入损失，因此在排序指标上\n      通常更优——这正是正文所说的"
          "「LTR 直接优化最终排序效果」。")
