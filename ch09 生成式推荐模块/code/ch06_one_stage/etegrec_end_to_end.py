r"""对应正文第 9 章 §9.6.2「ETEGRec 算法」: 双 Encoder-Decoder 的端到端可学习 Tokenizer。

正文公式(本实现逐条复刻):
  Tokenizer(RQ 概率式量化)
    P(k|v_l) = exp(-||v_l - e_k^l||²) / Σ_j exp(-||v_l - e_j^l||²)
    c_l = argmax_k P(k|v_l);  v_{l+1} = v_l - e_{c_l}^l;  r~ = Σ_l e_{c_l}^l
    L_SQ = L_RECON + L_RQ
    L_RECON = ||z - z~||²
    L_RQ = Σ_l ( ||sg[v_l] - e_{c_l}^l||² + β||v_l - sg[e_{c_l}^l]||² )
  生成式推荐模型(Transformer Encoder-Decoder)
    L_REC = -Σ_j log P(Y_j | X, Y_<j)
  推荐导向对齐
    SIA: z_E = MLP(MeanPool(H_E));  L_SIA = Σ_l ( KL(P_z^l || P_{z_E}^l) + KL(P_{z_E}^l || P_z^l) )
    PSA: L_PSA = -log exp(s(z~,h_D)/τ)/Σ_{h^∈B} exp(s(z~,h^)/τ)
                 -log exp(s(h_D,z~)/τ)/Σ_{z^∈B} exp(s(h_D,z^)/τ)
  交替优化
    优化 Tokenizer:  L_IT = L_SQ + μ L_SIA + λ L_PSA   (冻结推荐模型)
    优化推荐模型:    L_GR = L_REC + μ L_SIA + λ L_PSA  (冻结 Tokenizer)

本实现用教学规模(L=3, K=8, d_model=32)的双塔跑交替优化, 并对比
「两阶段(Tokenizer 先训好再冻结)」与「ETEGRec 交替优化」的下一物品 SID 命中率、
码本利用率, 以及 L_SIA / L_PSA 的下降轨迹。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

L_LAYER = 3
K_CODE = 8
D_LAT = 16
D_SEM = 24           # 预训练语义/协同 Embedding 维度 d_s
D_MODEL = 32
N_ITEM = 150
SEQ_LEN = 6
BOS = L_LAYER * K_CODE          # Decoder 起始 Token
VOCAB = L_LAYER * K_CODE + 1


class ItemTokenizer(nn.Module):
    """RQ 概率式 Tokenizer: Encoder_T -> 逐层残差量化 -> Decoder_T 重建。"""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D_SEM, 32), nn.GELU(), nn.Linear(32, D_LAT))
        self.dec = nn.Sequential(nn.Linear(D_LAT, 32), nn.GELU(), nn.Linear(32, D_SEM))
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(K_CODE, D_LAT) * 0.3) for _ in range(L_LAYER)])

    def layer_probs(self, r):
        """返回每层 P(k|v_l)(正文公式), 以及硬 Token、量化和。"""
        v = r
        probs, toks, quant = [], [], 0.0
        rq = 0.0
        for l in range(L_LAYER):
            d2 = torch.cdist(v, self.codebooks[l]) ** 2         # ||v_l - e_k||²
            p = F.softmax(-d2, dim=-1)
            c = torch.argmax(p, dim=-1)
            e = self.codebooks[l][c]
            # L_RQ: codebook 项 + commitment 项(β=0.25)
            rq = rq + ((v.detach() - e) ** 2).sum(-1).mean() \
                + 0.25 * ((v - e.detach()) ** 2).sum(-1).mean()
            probs.append(p)
            toks.append(c)
            quant = quant + e
            v = v - e                                            # v_{l+1}
        return probs, torch.stack(toks, 1), quant, rq

    def forward(self, z):
        r = self.enc(z)
        probs, toks, quant, rq = self.layer_probs(r)
        # STE: 让梯度从量化结果回传到 Encoder
        quant_st = r + (quant - r).detach()
        z_rec = self.dec(quant_st)
        recon = ((z - z_rec) ** 2).sum(-1).mean()
        return dict(tok=toks, probs=probs, z_rec=z_rec, quant=quant_st,
                    l_sq=recon + rq, l_recon=float(recon), l_rq=float(rq))

    def probs_of(self, z):
        """给任意语义向量(如 z_E)算每层 Token 分布, 用于 SIA。"""
        return self.layer_probs(self.enc(z))[0]


class GenRecommender(nn.Module):
    """Transformer Encoder-Decoder: H_E = Encoder_R(X), H_D = Decoder_R(H_E, Y~)。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_e = nn.Embedding(SEQ_LEN * L_LAYER, D_MODEL)
        self.pos_d = nn.Embedding(L_LAYER + 1, D_MODEL)
        # 显式搭 Encoder/Decoder(而非 nn.Transformer): norm_first=True 时
        # nested-tensor 快速路径不可用, 只有在构造时传 enable_nested_tensor=False
        # 才不会打印 UserWarning
        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, 4, 4 * D_MODEL, dropout=0.0,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, 2, norm=nn.LayerNorm(D_MODEL),
            enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(
            D_MODEL, 4, 4 * D_MODEL, dropout=0.0,
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            dec_layer, 2, norm=nn.LayerNorm(D_MODEL))
        self.out = nn.Linear(D_MODEL, K_CODE * L_LAYER)
        self.mlp_e = nn.Sequential(nn.Linear(D_MODEL, 32), nn.GELU(),
                                   nn.Linear(32, D_SEM))     # SIA 的 MLP
        self.mlp_h = nn.Sequential(nn.Linear(D_MODEL, 32), nn.GELU(),
                                   nn.Linear(32, D_LAT))     # PSA 的 MLP
        self.mlp_z = nn.Sequential(nn.Linear(D_LAT, 32), nn.GELU(),
                                   nn.Linear(32, D_LAT))

    def encode(self, x):
        t = x.shape[1]
        h = self.emb(x) + self.pos_e(torch.arange(t))[None]
        return self.encoder(h)

    def decode(self, mem, y_in):
        t = y_in.shape[1]
        h = self.emb(y_in) + self.pos_d(torch.arange(t))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(t)
        return self.decoder(h, mem, tgt_mask=mask)


def flat(layer, code):
    return layer * K_CODE + code


def rec_loss_and_hit(model, x, tgt_tok):
    """L_REC = -Σ_j log P(Y_j|X,Y_<j); 同时返回 h_D(BOS 位置隐状态)与全命中率。"""
    mem = model.encode(x)
    y_in = torch.cat([torch.full((x.shape[0], 1), BOS, dtype=torch.long),
                      torch.stack([flat(l, tgt_tok[:, l])
                                   for l in range(L_LAYER)], 1)], 1)[:, :L_LAYER]
    hd = model.decode(mem, y_in)
    logits = model.out(hd)                            # (B, L, K*L)
    loss, hit = 0.0, torch.ones(x.shape[0], dtype=torch.bool)
    for l in range(L_LAYER):
        lo = logits[:, l, l * K_CODE:(l + 1) * K_CODE]
        loss = loss + F.cross_entropy(lo, tgt_tok[:, l])
        hit &= (lo.argmax(-1) == tgt_tok[:, l])
    return loss, mem, hd[:, 0], hit.float().mean()


def l_sia(p_z, p_ze):
    """对称 KL: Σ_l ( KL(P_z||P_zE) + KL(P_zE||P_z) )。"""
    tot = 0.0
    for a, b in zip(p_z, p_ze):
        a = a.clamp_min(1e-9)
        b = b.clamp_min(1e-9)
        tot = tot + (a * (a / b).log()).sum(-1).mean() \
            + (b * (b / a).log()).sum(-1).mean()
    return tot


def l_psa(z_tilde, h_d, tau=0.1):
    """双向 InfoNCE(余弦相似度, batch 内负样本)。"""
    a = F.normalize(z_tilde, dim=-1)
    b = F.normalize(h_d, dim=-1)
    s = a @ b.t() / tau
    lab = torch.arange(a.shape[0])
    return F.cross_entropy(s, lab) + F.cross_entropy(s.t(), lab)


def perplexity(tok):
    out = []
    for l in range(L_LAYER):
        cnt = torch.bincount(tok[:, l], minlength=K_CODE).float()
        p = (cnt / cnt.sum()).clamp_min(1e-12)
        out.append(float((-(p * p.log()).sum()).exp()))
    return out


def build_data(gen):
    centers = torch.randn(8, D_SEM, generator=gen) * 2.0
    grp = torch.randint(0, 8, (N_ITEM,), generator=gen)
    z = centers[grp] + torch.randn(N_ITEM, D_SEM, generator=gen) * 0.5
    seqs, tgt = [], []
    for _ in range(320):
        g = int(torch.randint(0, 8, (1,), generator=gen))
        pool = (grp == g).nonzero().flatten()
        pick = pool[torch.randint(0, len(pool), (SEQ_LEN + 1,), generator=gen)]
        seqs.append(pick[:SEQ_LEN])
        tgt.append(int(pick[SEQ_LEN]))
    return z, torch.stack(seqs), torch.tensor(tgt)


def seq_tokens(tok_all, seqs):
    """把历史物品序列展开成 Token 序列 X = [c_1,...,c_n]。"""
    t = tok_all[seqs]                       # (B, T, L)
    off = (torch.arange(L_LAYER) * K_CODE)[None, None]
    return (t + off).reshape(t.shape[0], -1)


def train(mode, z, seqs, tgt, rounds=5, ep_tok=3, ep_rec=6, mu=0.05, lam=0.05):
    """mode = 'two_stage' | 'etegrec'(交替优化)。"""
    torch.manual_seed(42)
    tk, gr = ItemTokenizer(), GenRecommender()
    opt_t = torch.optim.Adam(tk.parameters(), lr=0.01)
    opt_r = torch.optim.Adam(gr.parameters(), lr=0.01)
    traj = []

    if mode == "two_stage":                 # 先把 Tokenizer 训到收敛再冻结
        for _ in range(rounds * ep_tok):
            out = tk(z)
            opt_t.zero_grad()
            out["l_sq"].backward()
            opt_t.step()

    for rd in range(rounds):
        if mode == "etegrec":               # ---- 阶段 1: 只更新 Tokenizer ----
            for _ in range(ep_tok):
                out = tk(z)
                tok_all = out["tok"]
                x = seq_tokens(tok_all, seqs)
                with torch.no_grad():
                    mem = gr.encode(x)
                    hd_raw = gr.decode(
                        mem, torch.cat([torch.full((x.shape[0], 1), BOS,
                                                   dtype=torch.long),
                                        (tok_all[tgt] +
                                         (torch.arange(L_LAYER) * K_CODE)[None]
                                         )[:, :L_LAYER - 1]], 1))[:, 0]
                    z_e = gr.mlp_e(mem.mean(1))
                p_z = [p[tgt] for p in out["probs"]]
                p_ze = tk.probs_of(z_e)
                sia = l_sia(p_z, p_ze)
                psa = l_psa(gr.mlp_z(out["quant"][tgt]), gr.mlp_h(hd_raw))
                loss = out["l_sq"] + mu * sia + lam * psa      # L_IT
                opt_t.zero_grad()
                loss.backward()
                opt_t.step()

        # ---- 阶段 2: 冻结 Tokenizer, 只更新推荐模型 ----
        with torch.no_grad():
            out = tk(z)
            tok_all = out["tok"]
            quant = out["quant"]
        x = seq_tokens(tok_all, seqs)
        for _ in range(ep_rec):
            rec, mem, hd, hit = rec_loss_and_hit(gr, x, tok_all[tgt])
            z_e = gr.mlp_e(mem.mean(1))
            p_z = [p[tgt].detach() for p in out["probs"]]
            p_ze = tk.probs_of(z_e)
            sia = l_sia(p_z, p_ze)
            psa = l_psa(gr.mlp_z(quant[tgt]), gr.mlp_h(hd))
            loss = rec + mu * sia + lam * psa                  # L_GR
            opt_r.zero_grad()
            loss.backward()
            opt_r.step()
        traj.append((rd + 1, float(rec), float(sia), float(psa),
                     float(out["l_sq"]), float(hit)))

    with torch.no_grad():
        out = tk(z)
        tok_all = out["tok"]
        x = seq_tokens(tok_all, seqs)
        rec, _, _, hit = rec_loss_and_hit(gr, x, tok_all[tgt])
    uniq = len({tuple(t.tolist()) for t in tok_all})
    return traj, float(rec), float(hit), perplexity(tok_all), uniq, \
        out["l_recon"], out["l_rq"]


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)
    z, seqs, tgt = build_data(gen)

    print("=" * 88)
    print(f"配置: 物品数={N_ITEM}, 序列数={len(seqs)}, L={L_LAYER}, K={K_CODE}, "
          f"d_s={D_SEM}, d_model={D_MODEL}")

    tj_e, rec_e, hit_e, ppl_e, uq_e, rc_e, rq_e = train("etegrec", z, seqs, tgt)
    tj_t, rec_t, hit_t, ppl_t, uq_t, rc_t, rq_t = train("two_stage", z, seqs, tgt)

    print()
    print("表 1  ETEGRec 交替优化轮次轨迹(每轮: 先 L_IT 更新 Tokenizer, 再 L_GR 更新推荐模型)")
    print("-" * 88)
    print(f"{'轮次':>5}{'L_REC':>10}{'L_SIA':>10}{'L_PSA':>10}{'L_SQ':>10}"
          f"{'训练 SID 全命中':>18}")
    for rd, rec, sia, psa, sq, hit in tj_e:
        print(f"{rd:>5}{rec:>10.4f}{sia:>10.4f}{psa:>10.4f}{sq:>10.4f}{hit:>18.4f}")

    print()
    print("表 2  两阶段(Tokenizer 冻结) vs. ETEGRec(交替优化)")
    print("-" * 88)
    print(f"{'方案':<28}{'L_REC':>10}{'SID 全命中率':>14}"
          f"{'perplexity(L1/L2/L3)':>26}{'不同 SID 数':>12}")
    for name, rec, hit, ppl, uq in (
            ("两阶段: RQ-VAE 后冻结", rec_t, hit_t, ppl_t, uq_t),
            ("ETEGRec: 交替优化+双对齐", rec_e, hit_e, ppl_e, uq_e)):
        print(f"{name:<28}{rec:>10.4f}{hit:>14.4f}"
              f"{'/'.join(f'{x:.2f}' for x in ppl):>26}{uq:>12}")
    print("-" * 88)
    print(f"Tokenizer 重建/量化损失: 两阶段 L_RECON={rc_t:.4f} L_RQ={rq_t:.4f};  "
          f"ETEGRec L_RECON={rc_e:.4f} L_RQ={rq_e:.4f}")
    print("正文指出: 两阶段方案 Tokenizer 只关心语义重建, 故重建损失更低, 但其 Token")
    print("组织方式未受推荐目标约束; ETEGRec 用 L_SIA/L_PSA 把推荐信号回传到 Tokenizer,")
    print("牺牲一点重建精度换取更适合生成任务的语义 ID。")

    # --------- 对齐损失的消融 ---------
    print()
    print("表 3  对齐机制消融(ETEGRec 框架下, 5 轮交替优化)")
    print("-" * 88)
    print(f"{'配置':<30}{'L_REC':>10}{'SID 全命中率':>14}{'L_SIA(末轮)':>14}"
          f"{'L_PSA(末轮)':>14}")
    for name, mu, lam in (("仅 L_SQ+L_REC(mu=lam=0)", 0.0, 0.0),
                          ("+ SIA(mu=0.05)", 0.05, 0.0),
                          ("+ PSA(lam=0.05)", 0.0, 0.05),
                          ("+ SIA & PSA", 0.05, 0.05)):
        tj, rec, hit, *_ = train("etegrec", z, seqs, tgt, mu=mu, lam=lam)
        print(f"{name:<30}{rec:>10.4f}{hit:>14.4f}{tj[-1][2]:>14.4f}"
              f"{tj[-1][3]:>14.4f}")
    print("-" * 88)
    print("注: 教学规模模型 + 合成数据, 绝对数值不代表论文结论, 仅展示各损失的耦合关系。")


if __name__ == "__main__":
    main()
