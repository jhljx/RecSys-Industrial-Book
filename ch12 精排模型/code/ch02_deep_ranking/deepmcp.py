"""
DeepMCP 精排模型（第12章 §12.4.1.6）

Representation Learning-Assisted Click-Through Rate Prediction（阿里，IJCAI 2019）。

三个子网络共享底层特征嵌入矩阵：

  1. 预测子网络（Prediction Subnet）
        m = Concat(e_1,…,e_n)
        ŷ = σ(wᵀ MLP(m) + b)
        L_p = BCE(y, ŷ)

  2. 匹配子网络（Matching Subnet）
        v_u = tanh(MLP_u([e_user; e_query]))
        v_a = tanh(MLP_a(e_ad))
        s(u,a) = σ(v_uᵀ v_a)
        L_m = BCE(y, s(u,a))          # 点式匹配，复用点击标签

  3. 相关性子网络（Correlation Subnet，Skip-gram 式）
        p⁺(a_j|a_i) = σ(h_{a_j}ᵀ h_{a_i})
        ℓ_c = -log σ(h_{a_j}ᵀh_{a_i}) - Σ_q log σ(-h_{a_q⁻}ᵀh_{a_i})
        L_c = mean over 正样本对

联合训练：L = L_p + α·L_m + β·L_c

关键工程性质（正文 PDF 第 40 页附近）：匹配与相关性子网络只在离线训练阶段
参与前向与梯度更新，线上推理只运行预测子网络，因此不增加线上算力消耗。

实现说明：
  - 广告标题为多值特征，先分别查嵌入再做 Sum Pooling，对应正文的多值特征处理；
  - 匹配子网络最后一层用 tanh（而非 ReLU），保留表示的正负方向，使内积可作为匹配信号；
  - α、β 为辅助任务权重，演示中取 0.5 / 0.05（相关性子网络权重较小，
    避免其梯度主导共享嵌入更新）。

本书依据：《第十二章-精排模型.pdf》§12.4.1.6。
依赖：PyTorch。运行：python3 deepmcp.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _auc(scores, labels) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


def _mlp(dims, last_act=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    if last_act is not None:
        layers.append(last_act())
    return nn.Sequential(*layers)


# ── DeepMCP 模型 ─────────────────────────────────────────────────────────────

class DeepMCP(nn.Module):
    """
    三子网络 + 共享嵌入。

    共享的嵌入表是三个任务传递监督信号的关键：预测任务改善的是「特征→点击」
    的映射，匹配任务要求用户表示与广告表示在同一空间可比，相关性任务要求
    广告表示能反映共现关系，三种梯度都会回流到同一组嵌入参数上。
    """

    def __init__(self, n_users: int, n_queries: int, n_ads: int,
                 n_ctx: int, n_tokens: int, emb_dim: int = 8,
                 title_len: int = 3):
        super().__init__()
        self.emb_dim = emb_dim
        self.title_len = title_len

        # 共享嵌入表
        self.emb_user = nn.Embedding(n_users, emb_dim)
        self.emb_query = nn.Embedding(n_queries, emb_dim)
        self.emb_ad = nn.Embedding(n_ads, emb_dim)
        self.emb_ctx = nn.Embedding(n_ctx, emb_dim)
        self.emb_token = nn.Embedding(n_tokens, emb_dim)   # 广告标题多值特征

        # 预测子网络：四组特征（含标题 Sum Pooling）+ 后续 MLP
        # 输入维度: user + query + ad + ctx + title_pooled
        pred_in = emb_dim * 5
        self.pred_mlp = _mlp([pred_in, 24, 12, 1])

        # 匹配子网络
        self.match_user = _mlp([emb_dim * 2, 16, emb_dim], last_act=nn.Tanh)
        self.match_ad = _mlp([emb_dim, 16, emb_dim], last_act=nn.Tanh)

        # 相关性子网络：广告 → 高层表示（用于 Skip-gram 内积打分）
        self.corr_mlp = _mlp([emb_dim, 16, emb_dim], last_act=nn.Tanh)

        for emb in (self.emb_user, self.emb_query, self.emb_ad,
                    self.emb_ctx, self.emb_token):
            nn.init.normal_(emb.weight, std=0.05)

    # ── 预测子网络 ───────────────────────────────────────────────────────────
    def predict_logit(self, user, query, ad, ctx, title_ids):
        e_u = self.emb_user(user)
        e_q = self.emb_query(query)
        e_a = self.emb_ad(ad)
        e_c = self.emb_ctx(ctx)
        # 多值特征：Sum Pooling
        e_t = self.emb_token(title_ids).sum(dim=1)       # (B, emb_dim)
        m = torch.cat([e_u, e_q, e_a, e_c, e_t], dim=1)
        return self.pred_mlp(m).squeeze(-1)

    # ── 匹配子网络 ───────────────────────────────────────────────────────────
    def match_score(self, user, query, ad):
        v_u = self.match_user(
            torch.cat([self.emb_user(user), self.emb_query(query)], dim=1))
        v_a = self.match_ad(self.emb_ad(ad))
        return (v_u * v_a).sum(dim=1)                    # 内积（未过 sigmoid）

    # ── 相关性子网络 ─────────────────────────────────────────────────────────
    def ad_repr(self, ad_ids):
        return self.corr_mlp(self.emb_ad(ad_ids))

    def correlation_loss(self, center, pos_ctx, neg_ctx):
        """
        center/pos_ctx: (B,) 中心广告与真实上下文广告
        neg_ctx:        (B, Q) 负采样广告
        """
        h_c = self.ad_repr(center)                       # (B, D)
        h_p = self.ad_repr(pos_ctx)                      # (B, D)
        h_n = self.ad_repr(neg_ctx)                      # (B, Q, D)

        pos_score = (h_c * h_p).sum(dim=1)               # (B,)
        neg_score = torch.einsum("bd,bqd->bq", h_c, h_n)  # (B, Q)

        loss = -F.logsigmoid(pos_score) - F.logsigmoid(-neg_score).sum(dim=1)
        return loss.mean()

    # ── 线上推理只需预测子网络 ───────────────────────────────────────────────
    def ctr_score(self, user, query, ad, ctx, title_ids):
        with torch.no_grad():
            return torch.sigmoid(
                self.predict_logit(user, query, ad, ctx, title_ids))


# ── 合成数据 ─────────────────────────────────────────────────────────────────

def make_data(n: int = 600, n_ads: int = 10, embed_cat: int = 3,
              seed: int = 42):
    """
    构造带隐类目结构的数据：
      - 每个广告属于 embed_cat 个隐类目之一（代码里不暴露给模型）；
      - 用户对某个类目的偏好强度不同，点击概率由「用户偏好 × 广告类目」决定；
      - 广告标题 token 与类目弱相关；
      - 用户点击序列按「偏好类目」生成，供相关性子网络学习广告共现。

    这种结构让匹配任务与相关性任务都能提供与 CTR 目标一致的监督信号，
    否则辅助任务只会带来噪声——正文也特别强调辅助任务需与主任务相关。
    """
    torch.manual_seed(seed)
    random.seed(seed)

    ad_cat = torch.tensor([a % embed_cat for a in range(n_ads)])
    # 每个用户对不同类目的偏好（固定可复现）
    user_pref = torch.randn(6, embed_cat)
    n_ctx = 4

    user = torch.randint(0, 6, (n,))
    query = torch.randint(0, 4, (n,))
    ad = torch.randint(0, n_ads, (n,))
    ctx = torch.randint(0, n_ctx, (n,))
    title = torch.randint(0, 30, (n, 3))

    pref = user_pref[user, ad_cat[ad]]                   # (n,)
    logit = (1.6 * pref + 0.5 * torch.sin(
        2.0 * torch.tensor([float(q) for q in query])) - 0.3)
    y = (torch.sigmoid(logit + 0.4 * torch.randn(n)) > 0.5).long()

    # 相关性训练数据：每个用户生成一条按偏好类目采样的点击序列
    seqs = []
    for u in range(6):
        best = int(user_pref[u].argmax())
        pool = [a for a in range(n_ads) if ad_cat[a].item() == best]
        if len(pool) < 2:
            pool = list(range(n_ads))
        seq = [random.choice(pool) if random.random() < 0.75
               else random.randrange(n_ads) for _ in range(12)]
        seqs.append(seq)

    return dict(user=user, query=query, ad=ad, ctx=ctx, title=title,
                y=y, seqs=seqs, n_ctx=n_ctx, n_users=6)


def make_pairs(seqs, n_ads: int, window: int = 2, n_neg: int = 4,
               seed: int = 7):
    """从点击序列中构造 (中心, 正上下文, 负样本) 三元组。"""
    rng = random.Random(seed)
    centers, poss, negs = [], [], []
    for seq in seqs:
        for i in range(len(seq)):
            for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                if i == j:
                    continue
                centers.append(seq[i])
                poss.append(seq[j])
                negs.append([rng.randrange(n_ads) for _ in range(n_neg)])
    return (torch.tensor(centers), torch.tensor(poss),
            torch.tensor(negs))


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    # 训练样本刻意稀疏：CTR 标注成本高，而点击序列可提供大量共现监督，
    # 这正是正文强调的「用单条 CTR 样本之外的信号改善表示」的适用场景。
    data = make_data(670, seed=37)
    N = len(data["y"])
    split = 350
    tr = {k: v[:split] for k, v in data.items()
          if torch.is_tensor(v) and v.size(0) == N}
    te = {k: v[split:] for k, v in data.items()
          if torch.is_tensor(v) and v.size(0) == N}
    te_y = data["y"][split:]

    cen, pos, neg = make_pairs(data["seqs"], n_ads=10)

    def build():
        torch.manual_seed(42)
        return DeepMCP(n_users=6, n_queries=4, n_ads=10, n_ctx=data["n_ctx"],
                       n_tokens=30, emb_dim=8)

    def run(model, use_aux: bool, epochs: int = 200, alpha=0.5, beta=0.05):
        opt = optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
        for ep in range(epochs):
            opt.zero_grad()
            logit = model.predict_logit(tr["user"], tr["query"], tr["ad"],
                                        tr["ctx"], tr["title"])
            loss = F.binary_cross_entropy_with_logits(logit, tr["y"].float())
            l_p = loss.item()
            if use_aux:
                m = model.match_score(tr["user"], tr["query"], tr["ad"])
                l_m = F.binary_cross_entropy_with_logits(m, tr["y"].float())
                l_c = model.correlation_loss(cen, pos, neg)
                loss = loss + alpha * l_m + beta * l_c
            loss.backward()
            opt.step()
            if (ep + 1) % 100 == 0:
                tag = f"  L_p={l_p:.4f}"
                if use_aux:
                    tag += f"  L_m={l_m.item():.4f}  L_c={l_c.item():.4f}"
                print(f"  epoch {ep+1:3d}{tag}")
        return model

    def evaluate(model):
        s = model.ctr_score(te["user"], te["query"], te["ad"],
                            te["ctx"], te["title"]).tolist()
        return _auc(s, te_y.tolist())

    print("=== 对照组：只用预测子网络（L_p）===")
    m_main = run(build(), use_aux=False)
    auc_main = evaluate(m_main)

    print("\n=== DeepMCP：L_p + 0.5·L_m + 0.05·L_c 联合训练 ===")
    m_full = run(build(), use_aux=True)
    auc_full = evaluate(m_full)

    print(f"\n预测子网络 AUC：仅 L_p={auc_main:.4f}  "
          f"联合训练={auc_full:.4f}  (Δ={auc_full - auc_main:+.4f})")

    # ── 匹配子网络的独立诊断 ────────────────────────────────────────────────
    with torch.no_grad():
        s_match = torch.sigmoid(
            m_full.match_score(te["user"], te["query"], te["ad"])).tolist()
    print(f"匹配子网络 AUC={_auc(s_match, te_y.tolist()):.4f}"
          "（点式匹配，复用点击标签，不额外构造样本对）")

    # ── 相关性子网络：正负样本得分 ──────────────────────────────────────────
    with torch.no_grad():
        h_c = m_full.ad_repr(cen[:200])
        h_p = m_full.ad_repr(pos[:200])
        h_n = m_full.ad_repr(neg[:200])
        s_pos = torch.sigmoid((h_c * h_p).sum(1)).mean().item()
        s_neg = torch.sigmoid(
            torch.einsum("bd,bqd->bq", h_c, h_n)).mean().item()
    print(f"相关性得分：真实上下文={s_pos:.4f}  负样本={s_neg:.4f}")

    # ── 训练 / 推理解耦 ──────────────────────────────────────────────────────
    n_all = sum(p.numel() for p in m_full.parameters())
    n_pred = sum(p.numel() for p in m_full.pred_mlp.parameters())
    n_pred += sum(p.numel() for p in m_full.emb_user.parameters())
    n_pred += sum(p.numel() for p in m_full.emb_query.parameters())
    n_pred += sum(p.numel() for p in m_full.emb_ad.parameters())
    n_pred += sum(p.numel() for p in m_full.emb_ctx.parameters())
    n_pred += sum(p.numel() for p in m_full.emb_token.parameters())
    print(f"\n参数分布：总参数量={n_all}  线上推理路径（嵌入 + 预测 MLP）={n_pred}")
    print("  正文：匹配/相关性子网络只参与离线训练，线上只跑预测子网络，")
    print("  因此辅助任务不增加高并发精排链路的推理开销。")

    # ── 精排推理演示 ─────────────────────────────────────────────────────────
    s = m_full.ctr_score(te["user"][:20], te["query"][:20], te["ad"][:20],
                         te["ctx"][:20], te["title"][:20]).tolist()
    top5 = sorted(enumerate(s), key=lambda t: -t[1])[:5]
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, score) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  p_click={score:.4f}  "
              f"label={te_y[idx].item()}")
