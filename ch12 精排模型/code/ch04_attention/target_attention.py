"""
Target Attention（目标注意力）精排示例（第12章 §12.4.3.2）

本书依据：《第十二章-精排模型.pdf》§12.4.3.2，代码 12.3。

推荐精排中的 Target Attention 用「当前候选物品」对用户行为序列做加权汇聚，
从而提取与候选相关的兴趣表示。它与缩放点积 Cross-Attention 的主要区别在于：
打分函数是一个可学习的非线性 MLP，而非简单的向量内积。

    缩放点积 Cross-Attention：α_i = softmax( (e_a · e_i) / √d )
    Target Attention（DIN 风格）：α_i = MLP([e_a; e_i; e_a−e_i; e_a⊙e_i])

后者能将候选、历史行为及两者之间的差异与交互作为打分的输入，
因此当「差分」或「元素乘积」携带有效判别信息时，MLP 打分函数的表达能力更强。

正文指出：完整的 DIN 模型和 BST 模型将在第20章《用户兴趣与序列建模》详细介绍；
本节只展示 Target Attention 模块在精排特征交互框架中的作用。

本文件的对照实验：
  - 基线：平均池化（忽略候选物品，静态汇聚序列兴趣）
  - 缩放点积 Cross-Attention（内积打分，不使用差分/交互项）
  - Target Attention（MLP 打分，使用 [e_a; e_i; e_a−e_i; e_a⊙e_i]）

依赖：PyTorch。运行：python3 target_attention.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


def _auc(scores, labels) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


# ── Target Attention（代码 12.3，正文 §12.4.3.2）────────────────────────────

class TargetAttention(nn.Module):
    """
    DIN 风格目标注意力（正文代码 12.3）。

    打分网络输入 [target; history; target−history; target⊙history]，
    包含四类关系：原始向量、差分（方向）、逐元素乘积（交互）。
    与仅使用内积相比，能捕获非线性的相关性模式。

    与缩放点积 Cross-Attention 的关键区别：
      - 缩放内积：α_i = softmax(e_a · e_i / √d)  — 线性相似度
      - 本模块：  α_i = MLP([e_a; e_i; e_a−e_i; e_a⊙e_i]) — 可学习非线性打分
    正文强调，推荐系统不必局限于向量内积，可以使用更丰富的匹配特征。
    """

    def __init__(self, d_model: int, hidden_dim: int = 16):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, target: torch.Tensor, history: torch.Tensor,
                valid_mask: torch.Tensor | None = None):
        """
        target     : (B, D)   当前候选物品嵌入
        history    : (B, L, D) 用户历史行为嵌入序列
        valid_mask : (B, L) bool，True 表示有效位置（可选）
        返回: pooled (B, D)，weights (B, L)
        """
        t = target.unsqueeze(1).expand_as(history)             # (B, L, D)
        feat = torch.cat([t, history, t - history, t * history], dim=-1)
        scores = self.score_net(feat).squeeze(-1)               # (B, L)

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask,
                                        torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)                 # (B, L)
        if valid_mask is not None:
            weights = weights * valid_mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        pooled = (weights.unsqueeze(-1) * history).sum(dim=1)  # (B, D)
        return pooled, weights


class TargetAttentionUnnormalized(TargetAttention):
    """
    DIN 原文的「局部激活单元」：直接用原始相关性打分做加权求和，不做 softmax 归一化。

    正文代码 12.3 为演示简洁采用了 softmax；而 DIN 论文强调不做归一化，
    以便在候选与某条历史行为高度相关时，让该行为的表示被**放大**，
    而不是仅仅在若干行为之间重新分配 1 的总权重。
    当序列中只有少数行为与候选真正相关时，归一化会把注意力「摊薄」，
    未归一化则保留了兴趣强度信息。
    """

    def forward(self, target: torch.Tensor, history: torch.Tensor,
                valid_mask: torch.Tensor | None = None):
        t = target.unsqueeze(1).expand_as(history)
        feat = torch.cat([t, history, t - history, t * history], dim=-1)
        weights = self.score_net(feat).squeeze(-1)             # (B, L) 原始打分
        if valid_mask is not None:
            weights = weights.masked_fill(~valid_mask, 0.0)
        pooled = (weights.unsqueeze(-1) * history).sum(dim=1)
        return pooled, weights


# ── 缩放点积 Cross-Attention（对照组）────────────────────────────────────────

class ScaledDotCrossAttention(nn.Module):
    """缩放点积 Cross-Attention：α_i = softmax((W_q e_a)·(W_k e_i) / √d)。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)

    def forward(self, target: torch.Tensor, history: torch.Tensor,
                valid_mask: torch.Tensor | None = None):
        q = self.q(target).unsqueeze(1)                        # (B,1,D)
        k = self.k(history)                                    # (B,L,D)
        scores = (q * k).sum(-1) / self.scale                  # (B,L)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask,
                                        torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * history).sum(dim=1)
        return pooled, weights


# ── CTR 预测器（三种汇聚方式共用相同 MLP 头）────────────────────────────────

class CTRPredictor(nn.Module):
    """
    嵌入 → 行为序列汇聚 → MLP → CTR。

    pool_mode 取值：
      "avg"         平均池化（基线）
      "dot"         缩放点积 Cross-Attention
      "target_attn" Target Attention（MLP 打分 + softmax，正文代码 12.3）
      "target_unnorm" Target Attention（MLP 打分 + 不归一化，DIN 原文做法）
    """

    def __init__(self, vocab: int, n_fields: int, emb_dim: int = 16,
                 hidden: int = 32, pool_mode: str = "target_attn"):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim)
        self.pool_mode = pool_mode
        self.n_fields = n_fields

        if pool_mode == "dot":
            self.attn = ScaledDotCrossAttention(emb_dim)
        elif pool_mode == "target_attn":
            self.attn = TargetAttention(emb_dim, hidden_dim=16)
        elif pool_mode == "target_unnorm":
            self.attn = TargetAttentionUnnormalized(emb_dim, hidden_dim=16)
        # avg 不需要额外模块

        # 输入：汇聚兴趣 + 候选物品 + 场景字段
        mlp_in = emb_dim * (n_fields + 2)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, seq_ids: torch.Tensor, target_id: torch.Tensor,
                feat_ids: torch.Tensor) -> torch.Tensor:
        seq_emb = self.emb(seq_ids)                            # (B, T, D)
        target_emb = self.emb(target_id)                       # (B, D)

        if self.pool_mode == "avg":
            v = seq_emb.mean(dim=1)
        else:
            v, _ = self.attn(target_emb, seq_emb)

        fields = [self.emb(feat_ids[:, i]) for i in range(self.n_fields)]
        z = torch.cat([v, target_emb] + fields, dim=1)
        return self.mlp(z).squeeze(-1)

    def fit(self, seq_ids, target_id, feat_ids, y,
            epochs: int = 40, lr: float = 5e-3):
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                self.forward(seq_ids, target_id, feat_ids), y.float())
            loss.backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")
        return loss.item()

    def predict_proba(self, seq_ids, target_id, feat_ids) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(seq_ids, target_id, feat_ids))


# ── 合成数据：差分信息比内积更能刻画「相关」────────────────────────────────────

def make_data(n: int = 3000, vocab: int = 16, n_cat: int = 4,
              seq_len: int = 6, n_fields: int = 3, seed: int = 42):
    """
    用户偏好与物品类目绑定：候选物品与偏好同类目时点击率高；
    行为序列中约 60% 是偏好类目物品，其余为其他类目的随机浏览。

    这种构造要求模型从序列里挑出「与候选同类目」的那部分行为。
      - 平均池化与候选无关，只能给出静态摘要，无法区分候选；
      - 内积打分只能利用线性相似度；
      - MLP 打分函数可以利用 [e_a; e_i; e_a−e_i; e_a⊙e_i] 中的差分与交互项，
        表达更复杂的相关性判断。
    """
    torch.manual_seed(seed)
    random.seed(seed)

    item_cat = torch.randint(0, n_cat, (vocab,))
    user_pref = torch.randint(0, n_cat, (n,))
    target = torch.randint(0, vocab, (n,))
    feat = torch.randint(0, vocab, (n, n_fields))

    match = (item_cat[target] == user_pref).float()
    logit = 2.4 * match - 1.0
    y = (torch.sigmoid(logit + 0.5 * torch.randn(n)) > 0.5).long()

    seqs = []
    for p in user_pref.tolist():
        s = []
        for _ in range(seq_len):
            if random.random() < 0.6:
                pool = (item_cat == p).nonzero(as_tuple=True)[0].tolist()
            else:
                pool = (item_cat != p).nonzero(as_tuple=True)[0].tolist()
            s.append(random.choice(pool) if pool else random.randrange(vocab))
        seqs.append(s)
    seq = torch.tensor(seqs)

    return dict(seq=seq, target=target, feat=feat, y=y)


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    VOCAB, N_FIELDS, SEQ_LEN = 16, 3, 6
    data = make_data(3000, vocab=VOCAB, seq_len=SEQ_LEN, n_fields=N_FIELDS)
    split = 2400
    tr = {k: v[:split] for k, v in data.items()}
    te = {k: v[split:] for k, v in data.items()}

    def build(mode):
        torch.manual_seed(42)
        return CTRPredictor(VOCAB, N_FIELDS, emb_dim=8, hidden=32,
                            pool_mode=mode)

    def evaluate(model):
        p = model.predict_proba(te["seq"], te["target"], te["feat"])
        return _auc(p.tolist(), te["y"].tolist())

    results, models = {}, {}
    for mode, label in [("avg", "平均池化（基线）"),
                        ("dot", "缩放点积 Cross-Attention"),
                        ("target_attn", "Target Attention（MLP + softmax，代码 12.3）"),
                        ("target_unnorm", "Target Attention（MLP，不归一化，DIN 原文）")]:
        print(f"\n=== {label} ===")
        m = build(mode)
        m.fit(tr["seq"], tr["target"], tr["feat"], tr["y"], epochs=40)
        models[mode] = m
        results[mode] = evaluate(m)
        print(f"  AUC = {results[mode]:.4f}")

    print(f"\n对比（正文 §12.4.3.2）：")
    print(f"  平均池化                    AUC={results['avg']:.4f}")
    print(f"  缩放点积 CrossAttn          AUC={results['dot']:.4f}"
          f"  (Δ={results['dot'] - results['avg']:+.4f})")
    print(f"  Target Attn + softmax       AUC={results['target_attn']:.4f}"
          f"  (Δ={results['target_attn'] - results['avg']:+.4f})")
    print(f"  Target Attn（不归一化）     AUC={results['target_unnorm']:.4f}"
          f"  (Δ={results['target_unnorm'] - results['avg']:+.4f})")
    print("  正文要点：Target Attention 的打分函数可以是可学习的非线性网络，")
    print("  将目标、历史行为及二者之间的差分与逐元素乘积作为输入，")
    print("  从而表达比向量内积更丰富的关系；它按候选动态汇聚兴趣，")
    print("  而平均池化只能给出与候选无关的静态兴趣摘要。")
    print("  另外，归一化与不归一化的差别很关键：softmax 把总权重固定为 1，")
    print("  序列中只有少数行为真正相关时会摊薄注意力；DIN 原文不做归一化，")
    print("  让高度相关的行为被放大，保留兴趣强度。")

    # ── 关键性质：同一序列、不同候选，汇聚表示应变化 ────────────────────────
    print("\n同一序列在不同候选下的汇聚表示变化（L2 距离，16 个候选的平均）：")
    with torch.no_grad():
        for mode in ("avg", "target_attn"):
            m = models[mode]
            se = m.emb(te["seq"][:1])
            if mode == "avg":
                delta = 0.0
            else:
                base = se.mean(dim=1)
                vs = torch.cat([m.attn(m.emb(torch.tensor([c])), se)[0]
                                for c in range(VOCAB)])
                delta = (vs - base).norm(dim=1).mean().item()
            print(f"  {mode:<12} 汇聚表示随候选的变化 = {delta:.4f}")

    # ── 注意力权重分析：同一序列、不同候选，权重应不同 ──────────────────────
    print("\nTarget Attention 权重（前 3 个测试样本，对比同一序列对不同候选的关注）：")
    m_ta = models["target_attn"]
    with torch.no_grad():
        for i in range(3):
            seq_i = te["seq"][i:i + 1]
            seq_emb = m_ta.emb(seq_i)
            cand_ids = [te["target"][i].item(), (te["target"][i].item() + 5) % VOCAB]
            for c in cand_ids:
                c_emb = m_ta.emb(torch.tensor([c]))
                _, w = m_ta.attn(c_emb, seq_emb)
                print(f"  样本{i} 候选={c:2d} 权重="
                      f"{[f'{v:.2f}' for v in w[0].tolist()]}")

    # ── 精排推理演示 ─────────────────────────────────────────────────────────
    p = m_ta.predict_proba(te["seq"][:20], te["target"][:20],
                           te["feat"][:20]).tolist()
    top5 = sorted(enumerate(p), key=lambda t: -t[1])[:5]
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  p_click={s:.4f}  "
              f"label={te['y'][idx].item()}")
