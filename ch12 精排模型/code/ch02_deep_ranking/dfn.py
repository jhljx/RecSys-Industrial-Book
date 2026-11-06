"""
DFN 精排模型（第12章 §12.4.1.7）

Deep Feedback Network for Recommendation（腾讯，IJCAI 2020）。

核心思路：将用户历史行为划分为三类，通过两层反馈交互提炼偏好表示，
再交给 Wide+FM+Deep 组件做 CTR 预估。

1. 行为分类（正文 PDF 第 43–44 页）
   C：隐式正反馈——用户点击序列
   D：显式负反馈——用户主动不喜欢序列
   U：隐式负反馈——曝光未点击序列（语义不确定，需进一步处理）

2. 内部反馈交互（IntraFeedback）
   将目标物品 t 与对应序列组合：B_x = [e_t+p_0, e_x1+p_1, …]
   → 多头自注意力 → Average Pooling → f_c / f_d / f_u

3. 外部反馈交互（InterFeedback）
   利用 f_d（可靠负向）对 U 做注意力 → f_ud（潜在负兴趣）
   利用 f_c（可靠正向）对 U 做注意力 → f_uc（潜在正兴趣）
   匹配函数：g(a,b) = MLP([a; b; a-b; a⊙b])

4. 特征交互（Wide + FM + Deep）
   将原始特征 + 目标物品 + 5 种反馈表示组合送入三组件

5. 联合损失
   L = -(λ_c·Σ log p(x∈C) + λ_u·Σ log(1-p(x∈U)) + λ_d·Σ log(1-p(x∈D)))

演示数据的构造刻意贴合上述动机：
  - C / D 序列短（长度 3）且噪声大（同类目占比 60% / 40%），单独信息量弱；
  - U 序列长（长度 8），同类目与跨类目物品各占一半、位置随机打乱；
    此时 f_u 会把正负兴趣平均掉，只有外部交互能把两类信号分离开。
  因此本文件中的对照实验直接检验「外部反馈交互是否真的有必要」。

本书依据：《第十二章-精排模型.pdf》§12.4.1.7，PDF 第 43–51 页。
依赖：PyTorch。运行：python3 dfn.py
"""

import math
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


# ── 多头自注意力（内部反馈交互的核心算子）──────────────────────────────────────

class MultiHeadSelfAttn(nn.Module):
    """
    轻量多头自注意力，用于学习目标物品与行为序列之间的关联。

    输入：(B, L, D)，L = 目标物品（位置0）+ 序列长度
    输出：(B, L, D)，再做 Average Pooling → (B, D)
    """

    def __init__(self, d: int, n_heads: int = 2):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.scale = math.sqrt(self.d_head)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        def proj(m, t):
            return m(t).view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)

        q, k, v = proj(self.q, x), proj(self.k, x), proj(self.v, x)
        att = (q @ k.transpose(-2, -1)) / self.scale       # (B,H,L,L)
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


# ── 匹配函数 g(a,b) ────────────────────────────────────────────────────────────

class MatchFunc(nn.Module):
    """
    外部反馈交互的匹配函数：
      g(a,b) = MLP([a; b; a-b; a⊙b])
    输入 4D → 标量得分（未过 sigmoid）
    """

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d * 4, d * 2), nn.ReLU(),
            nn.Linear(d * 2, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (B, D),  b: (B, L, D)  → (B, L)
        a_exp = a.unsqueeze(1).expand_as(b)
        feat = torch.cat([a_exp, b, a_exp - b, a_exp * b], dim=-1)
        return self.mlp(feat).squeeze(-1)                  # (B, L)


# ── DFN 模型 ─────────────────────────────────────────────────────────────────

class DFN(nn.Module):
    """
    DFN 完整实现：内部反馈交互 + 外部反馈交互 + Wide/FM/Deep 特征交互。
    """

    def __init__(self, vocab: int, n_fields: int, emb_dim: int = 16,
                 n_heads: int = 2, dnn_hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim)
        self.pos_emb = nn.Embedding(32, emb_dim)            # 位置嵌入，最长 32
        self.n_fields = n_fields
        self.emb_dim = emb_dim

        # 内部反馈交互（三组序列各自独立 MHA）
        self.attn_c = MultiHeadSelfAttn(emb_dim, n_heads)
        self.attn_d = MultiHeadSelfAttn(emb_dim, n_heads)
        self.attn_u = MultiHeadSelfAttn(emb_dim, n_heads)

        # 外部反馈交互的匹配函数
        self.match_neg = MatchFunc(emb_dim)   # f_d 引导 U → f_ud
        self.match_pos = MatchFunc(emb_dim)   # f_c 引导 U → f_uc

        # Wide 组件（n_fields 原始字段 + 目标物品 + 5 种反馈表示）
        self.wide = nn.ModuleList([
            nn.Linear(emb_dim, 1) for _ in range(n_fields + 6)])

        # Deep 组件：输入为原始字段嵌入 + 目标物品 + 5 种反馈表示
        deep_in = emb_dim * (n_fields + 6)
        self.deep = nn.Sequential(
            nn.Linear(deep_in, dnn_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(dnn_hidden, dnn_hidden // 2), nn.ReLU(),
        )

        # 预测层
        wide_out = n_fields + 6
        fm_out = emb_dim
        deep_out = dnn_hidden // 2
        self.pred = nn.Linear(wide_out + fm_out + deep_out, 1)

    # ── 内部反馈交互 ─────────────────────────────────────────────────────────
    def _intra(self, attn_mod, target_emb, seq_ids):
        """
        target_emb: (B, D)
        seq_ids:    (B, Ls) — 序列物品 id
        返回: (B, D)  Average Pooling 后的反馈表示
        """
        B, D = target_emb.shape
        Ls = seq_ids.size(1)
        e_seq = self.emb(seq_ids)                          # (B, Ls, D)
        pos_ids = torch.arange(Ls + 1, device=seq_ids.device)
        p = self.pos_emb(pos_ids)                          # (Ls+1, D)
        e_target = (target_emb + p[0]).unsqueeze(1)        # (B, 1, D)
        e_seq = e_seq + p[1:].unsqueeze(0)                 # (B, Ls, D)
        x = torch.cat([e_target, e_seq], dim=1)            # (B, Ls+1, D)
        out = attn_mod(x)                                   # (B, Ls+1, D)
        return out.mean(dim=1)                             # (B, D)

    # ── 外部反馈交互 ─────────────────────────────────────────────────────────
    def _inter(self, guide, u_reps, match_func):
        """
        guide:   (B, D)  — f_c 或 f_d 作为引导
        u_reps:  (B, Lu, D) — 未点击序列各位置表示
        返回: (B, D) 加权汇聚后的偏好表示
        """
        scores = match_func(guide, u_reps)                 # (B, Lu)
        weights = F.softmax(scores, dim=1)
        return (weights.unsqueeze(-1) * u_reps).sum(dim=1)  # (B, D)

    # ── Wide + FM + Deep 特征交互 ────────────────────────────────────────────
    def _feature_interact(self, field_embs):
        """
        field_embs: list of (B, D) — 原始字段嵌入 + 5 种反馈表示
        返回: logit (B,)
        """
        # Wide
        y_wide = torch.cat([w(e) for w, e in
                             zip(self.wide, field_embs)], dim=1)  # (B, F)

        # FM（Bi-interaction：sum² - sum of squares）
        stack = torch.stack(field_embs, dim=1)             # (B, F, D)
        s = stack.sum(dim=1)                               # (B, D)
        s2 = (stack ** 2).sum(dim=1)
        y_fm = 0.5 * (s * s - s2)                         # (B, D)

        # Deep
        deep_in = stack.reshape(stack.size(0), -1)
        y_deep = self.deep(deep_in)                        # (B, D/2)

        y = torch.cat([y_wide, y_fm, y_deep], dim=1)
        return self.pred(y).squeeze(-1)

    def forward(self, feat_ids, target_id, seq_c, seq_d, seq_u):
        """
        feat_ids : (B, n_fields) 原始字段嵌入 id（用户/物品/场景特征）
        target_id: (B,)          目标物品 id
        seq_c    : (B, Lc)       点击序列 id
        seq_d    : (B, Ld)       不喜欢序列 id
        seq_u    : (B, Lu)       曝光未点击序列 id
        """
        # 字段嵌入
        field_embs = [self.emb(feat_ids[:, i])
                      for i in range(self.n_fields)]       # n_fields × (B,D)
        target_emb = self.emb(target_id)                   # (B, D)

        # 内部反馈交互
        f_c = self._intra(self.attn_c, target_emb, seq_c)
        f_d = self._intra(self.attn_d, target_emb, seq_d)
        f_u = self._intra(self.attn_u, target_emb, seq_u)

        # 外部反馈交互：获取 U 序列每个位置的表示
        u_reps = self.emb(seq_u)                           # (B, Lu, D)
        f_ud = self._inter(f_d, u_reps, self.match_neg)
        f_uc = self._inter(f_c, u_reps, self.match_pos)

        # 目标物品 + 五种反馈表示 + 原始字段 → 特征交互
        all_embs = field_embs + [target_emb, f_c, f_d, f_u, f_uc, f_ud]
        return self._feature_interact(all_embs)

    def predict_proba(self, *args):
        with torch.no_grad():
            return torch.sigmoid(self.forward(*args))


# ── 合成数据 ─────────────────────────────────────────────────────────────────

def make_data(n: int = 3000, vocab: int = 64, n_fields: int = 4,
              seq_len: int = 3, seq_len_u: int = 8, seed: int = 42):
    """
    合成三类反馈数据：
      - 每个样本有一个隐藏的用户偏好类目（不直接作为特征给出）；
      - feat 全部为噪声字段，不含用户 id——模型无法从原始特征推断偏好；
      - 目标物品类目与用户偏好一致时 CTR 较高；
      - 点击序列（C）短且噪声大（60% 同类目），单独不足以确定偏好；
      - 不喜欢序列（D）同样短且噪声大（40% 同类目）；
      - 曝光未点击序列（U）较长，且**同类目与跨类目物品各占一半并随机打乱位置**。

    这正是正文 §12.4.1.7 所描述的场景：
      - U 是曝光未点击，语义模糊——一半可能是「潜在正兴趣」（看了没点但同偏好），
        一半是「潜在负兴趣」（跨类目）；
      - 内部反馈交互只做平均池化，f_u 会把正负兴趣混在一起，信号互相抵消；
      - 外部反馈交互用 f_c 引导 U 找出正向部分（f_uc）、用 f_d 引导 U 找出
        负向部分（f_ud），把混杂的 U 拆成两个可用表示。

    因此外部交互的价值不是「补充信息」，而是「分离 U 内部互相抵消的两类信号」。
    """
    torch.manual_seed(seed)
    random.seed(seed)

    n_cat = 4
    item_cat = torch.randint(0, n_cat, (vocab,))           # 每个 item 的隐类目
    user_pref = torch.randint(0, n_cat, (n,))              # 每个样本的用户偏好

    feat = torch.randint(0, vocab, (n, n_fields))          # 全部为噪声字段
    target = torch.randint(0, vocab, (n,))

    match = (item_cat[target] == user_pref).float()
    logit = 2.2 * match - 0.9
    y = (torch.sigmoid(logit + 0.6 * torch.randn(n)) > 0.5).long()

    def make_seq(prefs, like, length, purity=None):
        """purity: 采到「同偏好类目」物品的概率；None 表示 like 决定（0.75/0.25）"""
        if purity is None:
            purity = 0.75 if like else 0.25
        seqs = []
        for p in prefs.tolist():
            s = []
            for _ in range(length):
                pool = ((item_cat == p) if random.random() < purity
                        else (item_cat != p)).nonzero(as_tuple=True)[0].tolist()
                s.append(random.choice(pool) if pool else random.randrange(vocab))
            seqs.append(s)
        return torch.tensor(seqs)

    # C / D 序列短且噪声大：单独看只能提供很弱的偏好线索
    seq_c = make_seq(user_pref, like=True, length=seq_len, purity=0.60)
    seq_d = make_seq(user_pref, like=False, length=seq_len, purity=0.40)

    # U 序列：同类目与跨类目各占一半，位置随机打乱（位置不泄露信息）
    seq_u = make_seq(user_pref, like=True, length=seq_len_u)
    mask = torch.rand(n, seq_len_u) < 0.5
    seq_u = torch.where(mask, make_seq(user_pref, like=False, length=seq_len_u),
                        seq_u)
    perm = torch.argsort(torch.rand(n, seq_len_u), dim=1)
    seq_u = torch.gather(seq_u, 1, perm)

    return dict(feat=feat, target=target, seq_c=seq_c, seq_d=seq_d,
                seq_u=seq_u, y=y, vocab=vocab, n_fields=n_fields)


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    N_FIELDS, VOCAB = 4, 16
    data = make_data(4000, vocab=VOCAB, n_fields=N_FIELDS)
    split = 3200

    keys = ("feat", "target", "seq_c", "seq_d", "seq_u", "y")
    tr = {k: data[k][:split] for k in keys}
    te = {k: data[k][split:] for k in keys}
    te_y = data["y"][split:]

    def build():
        torch.manual_seed(42)
        return DFN(vocab=VOCAB, n_fields=N_FIELDS, emb_dim=8, n_heads=2,
                   dnn_hidden=24)

    # ── 对照组：无外部交互（只用 f_c, f_d, f_u，跳过 f_uc / f_ud）──────────
    class DFN_NoInter(DFN):
        def forward(self, feat_ids, target_id, seq_c, seq_d, seq_u):
            field_embs = [self.emb(feat_ids[:, i])
                          for i in range(self.n_fields)]
            target_emb = self.emb(target_id)
            f_c = self._intra(self.attn_c, target_emb, seq_c)
            f_d = self._intra(self.attn_d, target_emb, seq_d)
            f_u = self._intra(self.attn_u, target_emb, seq_u)
            # 无外部交互：用 f_u 填充 f_uc 和 f_ud（无信息补充）
            all_embs = field_embs + [target_emb, f_c, f_d, f_u, f_u, f_u]
            return self._feature_interact(all_embs)

    def run(model, epochs=80, lc=1.0, lu=0.5, ld=0.5):
        """
        DFN 联合损失：
          L = -(λ_c·Σlog p(click) + λ_u·Σlog(1-p(unclick)) + λ_d·Σlog(1-p(hate)))
        正文强调：三类样本可靠性不同，λ_c > λ_d > λ_u 反映这种权重差异。
        """
        opt = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        args = (tr["feat"], tr["target"], tr["seq_c"],
                tr["seq_d"], tr["seq_u"])
        y = tr["y"].float()

        # 按标签类型拆分：点击(y=1) / 不喜欢(y=0 but in D) / 未点击(y=0)
        # 演示中统一用点击标签简化：1→点击样本，0→非点击样本（含未点击和不喜欢）
        for ep in range(epochs):
            opt.zero_grad()
            logit = model(*args)
            # 点击样本按 λ_c 权重，非点击样本按 (λ_u+λ_d)/2 权重
            w = torch.where(y == 1, torch.tensor(lc), torch.tensor((lu + ld) / 2))
            loss = F.binary_cross_entropy_with_logits(logit, y, weight=w)
            loss.backward()
            opt.step()
            if (ep + 1) % 40 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")
        return model

    def evaluate(model):
        with torch.no_grad():
            p = torch.sigmoid(model(te["feat"], te["target"],
                                    te["seq_c"], te["seq_d"], te["seq_u"]))
        return _auc(p.tolist(), te_y.tolist())

    print("=== 对照组：无外部反馈交互（仅内部 Attention）===")
    m_base = run(DFN_NoInter(VOCAB, N_FIELDS, emb_dim=8, n_heads=2,
                             dnn_hidden=24))
    auc_base = evaluate(m_base)

    print("\n=== DFN：内部 + 外部反馈交互 ===")
    m_dfn = run(build())
    auc_dfn = evaluate(m_dfn)

    print(f"\n对比（正文 §12.4.1.7）：")
    print(f"  无外部交互 AUC={auc_base:.4f}")
    print(f"  DFN 完整   AUC={auc_dfn:.4f}  (Δ={auc_dfn - auc_base:+.4f})")
    print("  外部反馈交互利用 f_c / f_d 分别引导未点击行为，")
    print("  提取其中的潜在正向（f_uc）与负向（f_ud）兴趣，减少语义混淆。")

    # ── 注意力权重分析：外部交互对 U 序列的差异化关注 ──────────────────────
    print("\n外部反馈交互注意力权重（前 4 个测试样本）：")
    with torch.no_grad():
        sample_feat = te["feat"][:4]
        sample_tgt  = te["target"][:4]
        sc, sd, su  = te["seq_c"][:4], te["seq_d"][:4], te["seq_u"][:4]

        tgt_emb = m_dfn.emb(sample_tgt)
        f_c4 = m_dfn._intra(m_dfn.attn_c, tgt_emb, sc)
        f_d4 = m_dfn._intra(m_dfn.attn_d, tgt_emb, sd)
        u_reps = m_dfn.emb(su)

        w_neg = F.softmax(m_dfn.match_neg(f_d4, u_reps), dim=1)
        w_pos = F.softmax(m_dfn.match_pos(f_c4, u_reps), dim=1)

    for i in range(4):
        print(f"  样本{i} pos引导权重={[f'{v:.3f}' for v in w_pos[i].tolist()]} "
              f"neg引导权重={[f'{v:.3f}' for v in w_neg[i].tolist()]}")
    print("  两组权重差异表明模型对同一未点击序列赋予了不同的正/负向注意力。")

    # ── 精排推理演示 ─────────────────────────────────────────────────────────
    scores = m_dfn.predict_proba(
        te["feat"][:20], te["target"][:20],
        te["seq_c"][:20], te["seq_d"][:20], te["seq_u"][:20]).tolist()
    top5 = sorted(enumerate(scores), key=lambda t: -t[1])[:5]
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  p_click={s:.4f}  "
              f"label={te_y[idx].item()}")
