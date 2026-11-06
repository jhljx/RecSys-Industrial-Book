"""
逐点损失函数精排实验（第12章 §12.5.1）

本书依据：《第十二章-精排模型.pdf》§12.5.1，PDF 第 70–82 页。

逐点（Pointwise）学习将每条样本独立视为分类/回归问题：

  BCE（二元交叉熵）：L = -Σ [y·log p + (1-y)·log(1-p)]
  MSE：             L = Σ (y - p)²          （回归，p 未过 sigmoid）
  Huber：           L = MSE if |r|<δ else δ(|r|-δ/2)
  Focal（何恺明）：  L = -Σ αₜ(1-pₜ)^γ·log pₜ
                      γ 降低易分样本权重；α 平衡正负样本
  PCOC 对齐：       L_pcoc = (mean_sigmoid - target_ctr)²  （保证预测均值 ≈ 真实 CTR）

  联合目标：        L = L_BCE + λ_pcoc·L_pcoc

本文件同时运行书中的 5 种损失对比（含 Pairwise / ListNet），
数据集协议与 §12.5 正文完全一致：
  - 1800 训练请求 × 16 候选（线性评分模型，从零初始化）
  - 600 验证请求，验证集 CTR ≈ 0.23
  - 22 个 epoch，批大小 = 全量
  - 指标：AUC / NDCG@5 / 请求级 Pairwise 准确率 / LogLoss / 均值 sigmoid / PCOC

依赖：PyTorch。运行：python3 pointwise.py
"""

import math
import random
import torch
import torch.nn.functional as F


# ── 合成数据集（与 pairwise.py / listwise.py 共用同一生成协议）────────────────

def make_dataset(n_queries: int, list_size: int = 16, seed: int = 0):
    """
    构造「请求 → 候选列表」精排数据集。

    特征工程（8 维）：
      user_context: (Q, 1, 2)  请求特征
      item:         (Q, L, 4)  候选物品特征
      交互项：[ctx0*item0, ctx1*item1]     2 维
      x = [user_ctx(扩展), item, 交互]     合计 8 维

    真实权重（对模型隐藏）：
      w = [0.65, -0.45, 0.90, 0.58, -0.42, 0.25, 0.62, -0.38]
    非线性：0.90*(item[:,2]²-1) + 0.65*max(item[:,3],0)

    验证集 CTR 目标 ≈ 0.23（与正文一致）。
    """
    torch.manual_seed(seed)
    random.seed(seed)

    Q, L = n_queries, list_size
    ctx = torch.randn(Q, 1, 2)
    item = torch.randn(Q, L, 4)
    inter = torch.stack([ctx[:, :, 0] * item[:, :, 0],
                         ctx[:, :, 1] * item[:, :, 1]], dim=-1)
    x = torch.cat([ctx.expand(-1, L, -1), item, inter], dim=-1)  # (Q, L, 8)

    w = torch.tensor([0.65, -0.45, 0.90, 0.58, -0.42, 0.25, 0.62, -0.38])
    logits = (-2.05
              + torch.einsum("qld,d->ql", x, w)
              + 0.90 * (item[:, :, 2] ** 2 - 1.0)
              + 0.65 * torch.clamp(item[:, :, 3], min=0.0))

    probs = torch.sigmoid(logits)
    y = torch.bernoulli(probs)

    return x, y, logits           # (Q,L,8), (Q,L), (Q,L)


# ── 线性精排模型 ──────────────────────────────────────────────────────────────

class LinearScorer:
    """s(x) = wᵀx + b，从零初始化，用于隔离损失函数的影响。"""

    def __init__(self, n_feat: int = 8):
        self.w = torch.zeros(n_feat, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def score(self, x):
        """x: (..., n_feat) → logit (...,)"""
        return (x @ self.w) + self.b

    def params(self):
        return [self.w, self.b]


# ── 指标 ─────────────────────────────────────────────────────────────────────

def ndcg_at_k(scores, labels, k=5):
    """请求级 NDCG@k，取均值（向量化）。

    正文口径：只对至少包含一个正样本的请求计算——全负请求没有可排序的正例，
    ideal DCG 为 0，若一并计入会把均值无意义地拉低。
    """
    s, y = scores, labels                                 # (Q,L) 张量
    order = s.argsort(dim=1, descending=True)[:, :k]       # (Q,k)
    discounts = 1.0 / torch.log2(torch.arange(k) + 2.0)    # (k,)
    dcg = (y.gather(1, order) * discounts).sum(dim=1)      # (Q,)
    ideal = (y.sort(dim=1, descending=True).values[:, :k]
             * discounts).sum(dim=1)                       # (Q,)
    valid = (y.sum(dim=1) > 0).float()                     # (Q,)
    ndcg = dcg / ideal.clamp(min=1e-6)
    return (ndcg * valid).sum() / valid.sum().clamp(min=1)


def pairwise_acc(scores, labels):
    """请求级 Pairwise 准确率（正样本分数 > 负样本分数的比例，向量化）。"""
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)       # (Q,L,L): diff[i,j]=s_i-s_j
    mask = (labels.unsqueeze(2) > labels.unsqueeze(1)).float()  # (Q,L,L): y_i>y_j → 1
    correct = ((diff > 0).float() * mask).sum()
    # 平局按 0.5 计
    tie = ((diff == 0).float() * mask).sum()
    total = mask.sum()
    return ((correct + 0.5 * tie) / total.clamp(min=1)).item()


def compute_metrics(scorer, x_val, y_val):
    """x_val: (Q,L,8), y_val: (Q,L)"""
    with torch.no_grad():
        s = scorer.score(x_val)                            # (Q, L)
        y_hat = torch.sigmoid(s)

    # AUC：拍平到样本级（按排序计算，避免 O(P·N) 双重循环）
    # 注意 rank 必须从「最小分数」开始编号（升序），否则算出来的是 1-AUC
    flat_s = s.reshape(-1)
    flat_y = y_val.reshape(-1)
    order = flat_s.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, flat_s.numel() + 1, dtype=torch.float)
    n_pos = flat_y.sum().item()
    n_neg = flat_y.numel() - n_pos
    if n_pos > 0 and n_neg > 0:
        sum_ranks_pos = ranks[flat_y > 0].sum().item()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auc = 0.5

    logloss = F.binary_cross_entropy_with_logits(
        s, y_val).item()
    mean_sig = y_hat.mean().item()
    true_ctr = y_val.float().mean().item()
    pcoc = mean_sig / true_ctr if true_ctr > 0 else float("nan")
    ndcg = ndcg_at_k(s, y_val, k=5)
    pair = pairwise_acc(s, y_val)

    return dict(auc=auc, ndcg5=ndcg, pair=pair,
                logloss=logloss, mean_sig=mean_sig, pcoc=pcoc)


# ── 各损失函数 ────────────────────────────────────────────────────────────────

def loss_bce(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y)


def loss_focal(logit, y, gamma=2.0, alpha=0.25):
    """Focal Loss：α(1-pₜ)^γ · BCE，自动处理正负样本不均衡。"""
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0, p, 1.0 - p)
    alpha_t = torch.where(y > 0, torch.full_like(y, alpha),
                          torch.full_like(y, 1.0 - alpha))
    return (alpha_t * (1.0 - pt) ** gamma * bce).mean()


def loss_pairwise(logit_q, y_q):
    """
    RankNet Pairwise 损失（单请求）：
      ℓ = -log σ(s_i - s_j)  对所有 (i,j) 满足 y_i > y_j
    正文指出其「平移不变性」：给所有分数加同一个常数不改变 loss。
    """
    diff = logit_q.unsqueeze(1) - logit_q.unsqueeze(0)    # (L,L): diff[i,j]=s_j-s_i
    mask = (y_q.unsqueeze(0) < y_q.unsqueeze(1)).float()  # (L,L): yi<yj → 1
    return (-F.logsigmoid(diff) * mask).sum() / mask.sum().clamp(min=1)


def loss_pairwise_batch(logit, y):
    """对 batch 内每个请求计算 Pairwise 损失，取均值（向量化）。"""
    diff = logit.unsqueeze(2) - logit.unsqueeze(1)         # (Q,L,L)
    mask = (y.unsqueeze(1) < y.unsqueeze(2)).float()       # (Q,L,L)
    per_q = (-F.logsigmoid(diff) * mask).sum(dim=(1, 2))
    return (per_q / mask.sum(dim=(1, 2)).clamp(min=1)).mean()


def loss_listnet(logit_q, y_q, tau_y=1.0, tau_s=1.0):
    """
    ListNet（§12.5.3）：
      q_i = softmax(y/τ_y)   目标分布（基于真实分数/标签）
      p_i = softmax(s/τ_s)   预测分布
      L = -Σ q_i log p_i     （KL 散度 + 常数）
    跳过全为负样本的请求。
    """
    if y_q.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    q = F.softmax(y_q / tau_y, dim=0)
    p = F.log_softmax(logit_q / tau_s, dim=0)
    return -(q * p).sum()


def loss_listnet_batch(logit, y, tau_y=1.0, tau_s=1.0):
    """向量化 ListNet；跳过全负请求（用掩码把其损失置零）。"""
    has_pos = (y.sum(dim=1) > 0).float()                  # (Q,)
    q = F.softmax(y / tau_y, dim=1)
    logp = F.log_softmax(logit / tau_s, dim=1)
    per_q = -(q * logp).sum(dim=1)                        # (Q,)
    return (per_q * has_pos).sum() / has_pos.sum().clamp(min=1)


# ── 五损失综合对比实验 ────────────────────────────────────────────────────────

def run_experiment():
    """
    复现正文 §12.5 的五损失对比实验：
      BCE / Pairwise / BCE+0.20·Pair / Focal(γ=2,α=0.25) / ListNet

    训练协议按正文（代码 12.10）：
      - 逐点目标（BCE / Focal）把「请求 × 候选」展平，按 64×16 条曝光一批；
      - 排序类目标（Pairwise / BCE+Pair、ListNet）保留请求维度，
        按 64 个请求一批，以便在批内构造物品对与列表级损失；
      - 22 个 epoch，SGD，学习率 5e-2，全零初始化。
    """
    TRAIN_Q, VAL_Q = 1800, 600
    LIST_SIZE, EPOCHS = 16, 22
    BATCH_Q, BATCH_FLAT = 64, 64 * 16
    LR = 5e-2

    x_tr, y_tr, _ = make_dataset(TRAIN_Q, LIST_SIZE, seed=0)
    x_va, y_va, _ = make_dataset(VAL_Q,   LIST_SIZE, seed=1)
    flat_x = x_tr.reshape(-1, x_tr.size(-1))
    flat_y = y_tr.reshape(-1)

    experiments = [
        # (名称, 损失函数, 是否逐点展平, 是否施加平移不变约束)
        ("BCE",              lambda l, y: loss_bce(l, y),
                             True,  False),
        ("Pairwise",         lambda l, y: loss_pairwise_batch(l, y),
                             False, True),
        ("BCE+0.20·Pair",    lambda l, y: loss_bce(l, y) + 0.20 * loss_pairwise_batch(l, y),
                             False, False),
        ("Focal(γ=2,α=0.25)",lambda l, y: loss_focal(l, y, gamma=2.0, alpha=0.25),
                             True,  False),
        ("ListNet",          lambda l, y: loss_listnet_batch(l, y),
                             False, True),
    ]

    print(f"{'损失函数':<22}  {'AUC':>6}  {'NDCG@5':>7}  "
          f"{'Pair准确率':>9}  {'LogLoss':>8}  {'μ(sigmoid)':>10}  {'PCOC':>6}")
    print("-" * 78)

    results = []
    for name, loss_fn, flat_mode, zero_linear in experiments:
        torch.manual_seed(42)
        scorer = LinearScorer(n_feat=8)
        opt = torch.optim.SGD(scorer.params(), lr=LR)
        g = torch.Generator().manual_seed(0)

        for ep in range(EPOCHS):
            if flat_mode:
                for idx in torch.randperm(len(flat_y), generator=g).split(BATCH_FLAT):
                    logit = scorer.score(flat_x[idx])       # (b,)
                    loss = loss_fn(logit, flat_y[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            else:
                for idx in torch.randperm(TRAIN_Q, generator=g).split(BATCH_Q):
                    logit = scorer.score(x_tr[idx])         # (b, L)
                    loss = loss_fn(logit, y_tr[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if zero_linear:
                        # 正文口径：纯 Pairwise / ListNet 的梯度只作用于分数差分，
                        # 平移不变性使偏置与线性项前两维不受约束，每步后清零。
                        with torch.no_grad():
                            scorer.w[:2].zero_()
                            scorer.b.zero_()

        m = compute_metrics(scorer, x_va, y_va)
        results.append((name, m))
        if name == "ListNet":
            # 正文说明：ListNet 输出的是请求内归一化的相对分布，
            # 不能解释为单个物品的点击概率，因此不报告 LogLoss / μ(sigmoid) / PCOC。
            print(f"{name:<22}  {m['auc']:>6.4f}  {m['ndcg5']:>7.4f}  "
                  f"{m['pair']:>9.4f}  {'–':>8}  {'–':>10}  {'–':>6}")
        else:
            print(f"{name:<22}  {m['auc']:>6.4f}  {m['ndcg5']:>7.4f}  "
                  f"{m['pair']:>9.4f}  {m['logloss']:>8.4f}  "
                  f"{m['mean_sig']:>10.4f}  {m['pcoc']:>6.3f}")

    return results


# ── PCOC 示意：BCE vs 无校准 ─────────────────────────────────────────────────

def demo_pcoc():
    """
    用 BCE + λ_pcoc·L_pcoc 对比纯 BCE，演示 PCOC 对齐的效果。
    """
    TRAIN_Q, EPOCHS, LR = 1800, 22, 5e-2
    x_tr, y_tr, _ = make_dataset(TRAIN_Q, seed=0)
    x_va, y_va, _ = make_dataset(600, seed=1)
    flat_x, flat_y = x_tr.reshape(-1, 8), y_tr.reshape(-1)
    TRUE_CTR = y_va.mean().item()

    print(f"\n  验证集真实 CTR = {TRUE_CTR:.4f}")
    for lam in (0.0, 5.0):
        torch.manual_seed(42)
        scorer = LinearScorer()
        opt = torch.optim.SGD(scorer.params(), lr=LR)
        g = torch.Generator().manual_seed(0)
        for _ in range(EPOCHS):
            for idx in torch.randperm(len(flat_y), generator=g).split(64 * 16):
                logit = scorer.score(flat_x[idx])
                loss = loss_bce(logit, flat_y[idx])
                if lam > 0:
                    pred_ctr = torch.sigmoid(logit).mean()
                    loss = loss + lam * (pred_ctr - TRUE_CTR) ** 2
                opt.zero_grad()
                loss.backward()
                opt.step()
        m = compute_metrics(scorer, x_va, y_va)
        tag = "BCE" if lam == 0 else f"BCE + {lam}·L_pcoc"
        print(f"  {tag:<22}  PCOC={m['pcoc']:.4f}  AUC={m['auc']:.4f}")


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 78)
    print("五损失函数对比实验（正文 §12.5，线性评分模型，零初始化，22 epochs）")
    print("=" * 78)
    run_experiment()

    print("\n" + "=" * 78)
    print("PCOC 对齐演示（正文 §12.5.1：平均预测值 / 真实 CTR）")
    print("=" * 78)
    demo_pcoc()

    print("\n要点（正文 §12.5.1）：")
    print("  BCE  ：直接优化预测准确性，PCOC 接近 1.0，可用于竞价出价；")
    print("  Focal：下调易分样本权重，对长尾正样本更友好，γ 越大正则越强；")
    print("  Huber：对异常值鲁棒的回归目标，适合预测点击次数而非概率；")
    print("  PCOC ：联合 BCE + 均值正则，防止系统性偏高/偏低，保证校准。")
