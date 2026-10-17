"""
SDM 长短期兴趣序列召回（第10章 §10.4.4「SDM召回」）

论文：Lv et al., "SDM: Sequential Deep Matching Model for Online Large-scale
      Recommender System", CIKM 2019（阿里巴巴）
核心思想（长短期兴趣分离 + 门控融合）：
- 短期兴趣：最近 Session 行为序列 → LSTM → 多头自注意力 → 以用户画像 Embedding
  e_u 为 query 的 User-Attention Pooling，得到短期兴趣向量 s；
- 长期兴趣：长期行为按多个特征维度（item / 类目 / 店铺 / 品牌）分别 Attention
  Pooling 得到 z_f，拼接后经全连接融合得到长期兴趣向量 p；
- 门控融合（Gated Fusion，SDM 的核心）：
      G = sigmoid(W1·e_u + W2·s + W3·p + b);  o = (1 - G) ⊙ p + G ⊙ s
- 训练用 Sampled Softmax（下一次点击为正样本 + 负采样），线上用 o 做 ANN 检索。

本实现：PyTorch autograd，用 nn.LSTM / nn.MultiheadAttention / nn.Linear 实现各模块；
完整反向传播由 autograd 自动完成。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── SDM 模型 ──────────────────────────────────────────────────────────────────

class SDM(nn.Module):
    """
    简化版 SDM：LSTM + 多头自注意力 + User-Attention（短期兴趣），多特征维度
    Attention Pooling + FC（长期兴趣），门控融合输出最终用户 Embedding。

    参数：item/cate/shop/brand_emb（物品与 side info）、user/age/gender_emb
    （用户画像）、LSTM、MultiheadAttention、W_p、W_g1/g2/g3/b_g。
    """

    LONG_FIELDS = ("item", "cate", "shop", "brand")

    def __init__(self, n_items: int, n_cates: int, n_shops: int, n_brands: int,
                 n_users: int, n_ages: int = 3, n_genders: int = 2,
                 dim: int = 16, n_heads: int = 2, lr: float = 0.05, reg: float = 1e-4):
        super().__init__()
        assert dim % n_heads == 0, "dim 必须能被 n_heads 整除"
        self.n_items = n_items
        self.dim = dim
        self.n_heads = n_heads
        self.lr = lr
        self.reg = reg

        # 物品与 side info Embedding（长期兴趣的多个特征维度）
        self.item_emb = nn.Embedding(n_items, dim)
        self.cate_emb = nn.Embedding(n_cates, dim)
        self.shop_emb = nn.Embedding(n_shops, dim)
        self.brand_emb = nn.Embedding(n_brands, dim)

        # 用户画像 Embedding：uid + 年龄段 + 性别，加和后经 W_u 投影为 e_u
        self.user_emb = nn.Embedding(n_users, dim)
        self.age_emb = nn.Embedding(n_ages, dim)
        self.gender_emb = nn.Embedding(n_genders, dim)
        self.W_u = nn.Linear(dim, dim)   # 投影（原文拼接三者，简化为加和后投影）

        # 短期兴趣：LSTM + 多头自注意力
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                               batch_first=True)

        # 长期兴趣融合：p = tanh(W_p · Concat(z_item, z_cate, z_shop, z_brand) + b_p)
        self.W_p = nn.Linear(len(self.LONG_FIELDS) * dim, dim)

        # 门控融合参数：W_g1 作用在 e_u，W_g2 作用在短期 s，W_g3 作用在长期 p
        self.W_g1 = nn.Linear(dim, dim, bias=False)
        self.W_g2 = nn.Linear(dim, dim, bias=False)
        self.W_g3 = nn.Linear(dim, dim)   # 带 bias（b_g）

        scale = math.sqrt(1.0 / dim)
        for emb in (self.item_emb, self.cate_emb, self.shop_emb, self.brand_emb,
                    self.user_emb, self.age_emb, self.gender_emb):
            nn.init.uniform_(emb.weight, -scale, scale)

    def _emb_table(self, field: str) -> nn.Embedding:
        return {"item": self.item_emb, "cate": self.cate_emb,
                "shop": self.shop_emb, "brand": self.brand_emb}[field]

    def user_profile(self, uid: torch.Tensor, age: torch.Tensor,
                     gender: torch.Tensor) -> torch.Tensor:
        """e_u = LayerNorm(W_u(uid_emb + age_emb + gender_emb))，shape [B, dim]。"""
        cat = self.user_emb(uid) + self.age_emb(age) + self.gender_emb(gender)
        return F.layer_norm(self.W_u(cat), [self.dim])

    # ── 短期兴趣：LSTM → 多头自注意力 → User-Attention ────────────────────
    def short_term(self, session_ids: torch.Tensor,
                   e_u: torch.Tensor) -> torch.Tensor:
        """
        session_ids: [B, T]
        e_u:         [B, dim]
        返回短期兴趣向量 s，shape [B, dim]
        """
        embs = self.item_emb(session_ids)              # [B, T, dim]
        lstm_out, _ = self.lstm(embs)                   # [B, T, dim]

        # 多头自注意力（不加因果 mask，SDM 短期序列较短，双向看也合理）
        H_hat, _ = self.self_attn(lstm_out, lstm_out, lstm_out)  # [B, T, dim]
        H_hat = F.layer_norm(H_hat + lstm_out, [self.dim])       # 残差 + LN

        # User-Attention Pooling：以 e_u 为 query，对 H_hat 做软注意力加权求和
        # scores: [B, T] = bmm([B,1,dim], [B,dim,T])
        scores = torch.bmm(e_u.unsqueeze(1), H_hat.transpose(1, 2)).squeeze(1)
        attn_weights = scores.softmax(dim=-1)           # [B, T]
        s = (attn_weights.unsqueeze(-1) * H_hat).sum(dim=1)  # [B, dim]
        return s

    # ── 长期兴趣：多特征维度 Attention Pooling + FC ───────────────────────
    def long_term(self, long_feats: dict[str, torch.Tensor],
                  e_u: torch.Tensor) -> torch.Tensor:
        """
        long_feats: {field: [B, K_f]}  各特征维度的历史 id 序列
        e_u:        [B, dim]
        返回长期兴趣向量 p，shape [B, dim]
        """
        z_all = []
        for f in self.LONG_FIELDS:
            ids = long_feats[f]                              # [B, K_f]
            g = self._emb_table(f)(ids)                      # [B, K_f, dim]
            # Attention Pooling：scores = g · e_u，α = softmax(scores)
            scores = torch.bmm(g, e_u.unsqueeze(-1)).squeeze(-1)  # [B, K_f]
            alpha = scores.softmax(dim=-1)                         # [B, K_f]
            z_f = (alpha.unsqueeze(-1) * g).sum(dim=1)            # [B, dim]
            z_all.append(z_f)
        concat = torch.cat(z_all, dim=-1)                    # [B, 4*dim]
        p = torch.tanh(self.W_p(concat))                     # [B, dim]
        return p

    # ── 门控融合（SDM 核心）───────────────────────────────────────────────
    def gated_fusion(self, e_u: torch.Tensor, s: torch.Tensor,
                     p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        G = σ(W1·e_u + W2·s + W3·p + b)；o = (1-G)⊙p + G⊙s
        返回 (o, G)，shape 均为 [B, dim]
        """
        G = torch.sigmoid(self.W_g1(e_u) + self.W_g2(s) + self.W_g3(p))
        o = (1.0 - G) * p + G * s
        return o, G

    def user_vector(self, sample: dict) -> tuple[torch.Tensor, dict]:
        """前向：画像 → 短期 / 长期 → 门控融合，返回最终用户 Embedding o 和缓存。"""
        p = sample["profile"]
        uid = torch.tensor([p["uid"]], dtype=torch.long)
        age = torch.tensor([p["age"]], dtype=torch.long)
        gender = torch.tensor([p["gender"]], dtype=torch.long)
        e_u = self.user_profile(uid, age, gender)            # [1, dim]

        session = torch.tensor([sample["session"]], dtype=torch.long)
        s = self.short_term(session, e_u)                    # [1, dim]

        long = {f: torch.tensor([sample["long"][f] or [0]], dtype=torch.long)
                for f in self.LONG_FIELDS}
        p_vec = self.long_term(long, e_u)                    # [1, dim]

        o, G = self.gated_fusion(e_u, s, p_vec)             # [1, dim]
        return o.squeeze(0), {"e_u": e_u, "s": s, "p": p_vec, "G": G}

    # ── 训练（Sampled Softmax）─────────────────────────────────────────────
    def train_model(self, samples: list[dict], epochs: int = 6, neg_k: int = 4) -> None:
        """samples: 每条包含 profile / session / long / target 四个字段。"""
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        for epoch in range(epochs):
            random.shuffle(samples)
            total = 0.0
            for smp in samples:
                pos = smp["target"]
                neg_candidates = []
                while len(neg_candidates) < neg_k:
                    c = random.randrange(self.n_items)
                    if c != pos:
                        neg_candidates.append(c)
                cands = [pos] + neg_candidates                  # [1 + neg_k]
                cand_t = torch.tensor(cands, dtype=torch.long)

                optimizer.zero_grad()
                o, _ = self.user_vector(smp)                    # [dim]
                logits = self.item_emb(cand_t) @ o              # [1+neg_k]
                label = torch.zeros(1, dtype=torch.long)        # pos is index 0
                loss = F.cross_entropy(logits.unsqueeze(0), label)
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"  epoch {epoch:2d}  avg_loss={total / max(len(samples), 1):.4f}")

    def recommend(self, sample: dict, top_k: int = 5) -> tuple[list[tuple[int, float]], dict]:
        """在线推理：实时算出用户 Embedding o，再对全量物品打分（等价 ANN 检索）。"""
        with torch.no_grad():
            o, cache = self.user_vector(sample)
            scores = self.item_emb.weight @ o                   # [n_items]
            exclude = set(sample["session"]) | set(sample["long"]["item"])
            for iid in exclude:
                scores[iid] = -1e9
            top_vals, top_idx = scores.topk(top_k)
        return ([(int(i), float(s)) for i, s in zip(top_idx.tolist(), top_vals.tolist())],
                cache)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, n_cates, n_shops, n_brands = 8, 20, 5, 4, 6
    item_cate = {i: i % n_cates for i in range(n_items)}
    item_shop = {i: i % n_shops for i in range(n_items)}
    item_brand = {i: i % n_brands for i in range(n_items)}
    samples = []
    for uid in range(n_users):
        profile = {"uid": uid, "age": uid % 3, "gender": uid % 2}
        hist = random.sample(range(n_items), k=12)
        long_items, session = hist[:6], hist[6:]
        long_feats = {
            "item": long_items,
            "cate": sorted({item_cate[i] for i in long_items}),
            "shop": sorted({item_shop[i] for i in long_items}),
            "brand": sorted({item_brand[i] for i in long_items}),
        }
        for t in range(3, len(session)):
            samples.append({"profile": profile, "long": long_feats,
                            "session": session[:t], "target": session[t]})

    model = SDM(n_items, n_cates, n_shops, n_brands, n_users, dim=16, n_heads=2, lr=0.05)
    print(f"SDM 训练（{len(samples)} 条样本，Sampled Softmax，负样本数=4）：")
    model.train_model(samples, epochs=6, neg_k=4)

    demo = samples[0]
    recs, cache = model.recommend(demo, top_k=5)
    print(f"\n用户 {demo['profile']['uid']} 最近 Session={demo['session']}"
          f"  长期行为 item={demo['long']['item']}  cate={demo['long']['cate']}")
    print(f"门控向量 G 均值={cache['G'].mean().item():.4f}（越接近 1 越偏向短期兴趣）")
    print("Top-5 召回结果：")
    for iid, sc in recs:
        print(f"  item {iid:2d}  score={sc:.4f}  cate={item_cate[iid]}")

    with torch.no_grad():
        g_avg = sum(
            model.user_vector(smp)[1]["G"].mean().item() for smp in samples
        ) / len(samples)
    print(f"\n全样本平均门控权重 G={g_avg:.4f}")
