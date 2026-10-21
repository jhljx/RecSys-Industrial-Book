r"""对应正文第 9 章 §9.6.1「DIGER 算法」的 DRIL(可微语义 ID + 探索式学习)范式。

正文公式(本实现逐条复刻):
  * logits:        l_{v,j,i} = sim(r_{v,j}, e_i)
  * Gumbel 噪声:   g = -log(-log u), u ~ U(0,1)      (逆变换采样)
  * Gumbel-Softmax: y~_{v,j,i} = exp((l+g)/τ) / Σ_k exp((l+g)/τ)
  * 前向硬选择:    c_{v,j} = argmax_i (l_{v,j,i} + g_{v,j,i})
  * 反向软码字:    ē_{v,j} = Σ_i y~_{v,j,i} · e_i      ("Forward Hard, Backward Soft")
  * SDUD:          L_σ = L_gen / (2(σ+λ)²) + log(σ+λ),  σ* = max(0, sqrt(L_gen) - λ)
  * FrqUD:         f_i^(e) = β f_i^(e-1) + (1-β) f^_i^(e);  γ = r/K;
                   高频码字(f_i > γ)保留 Gumbel 噪声, 低频码字用确定性 Softmax

本实现:
  1. 用一个小的「物品表示 -> 可微语义 ID -> Next-SID 预测」联合模型做端到端训练;
  2. 对比三种离散化方式: 纯 STE(argmax 最近邻) / DRIL(Gumbel, 固定 τ) / DRIL+FrqUD;
  3. 打印 Gumbel-Max Trick 的数值验证表、σ* 随 L_gen 衰减的轨迹表、
     以及三种方式的码本利用率(perplexity)与生成准确率对比表。
  4. 生成式推荐模型为教学规模的单层 GRU + 线性头, 不是正文中的大模型。
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

L_LAYER = 3          # 语义 ID 层数 m
K_CODE = 8           # 每层码字数 K
D_LAT = 16           # 码字维度
N_ITEM = 120


# ---------------------------------------------------------------- DRIL
def gumbel_noise(shape, gen, sigma=1.0):
    """g = -log(-log u), u ~ Uniform(0,1); sigma 为正文 SDUD 中的噪声尺度。"""
    u = torch.rand(shape, generator=gen).clamp_(1e-9, 1 - 1e-9)
    return sigma * (-torch.log(-torch.log(u)))


def dril_quantize(r, codebook, tau, gen, sigma=1.0, noise_mask=None):
    """返回 (硬 Token, 可微软码字 ē, 软概率 y~)。

    sim 用负欧氏距离(正文仅写 sim(·,·), 此处与 RQ-VAE 保持一致取 -||r-e||²)。
    noise_mask: FrqUD 用, shape (K,), 1 表示该码字加 Gumbel 噪声, 0 表示关闭。
    """
    logits = -torch.cdist(r, codebook) ** 2                 # (B, K)
    g = gumbel_noise(logits.shape, gen, sigma)
    if noise_mask is not None:
        g = g * noise_mask[None, :]
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    hard = torch.argmax(logits + g, dim=-1)                 # 正文: 硬 argmax
    e_bar = y_soft @ codebook                               # 正文: 软码字
    # Forward Hard / Backward Soft: 前向用硬码字数值, 梯度走软码字
    e_hard = codebook[hard]
    e_out = e_hard + (e_bar - e_bar.detach())
    return hard, e_out, y_soft


def ste_quantize(r, codebook):
    """对照组: 普通 VQ-VAE 的 STE(最近邻 + 直通梯度), 无探索。"""
    d = torch.cdist(r, codebook) ** 2
    hard = torch.argmax(-d, dim=-1)
    e = codebook[hard]
    return hard, r + (e - r).detach(), F.softmax(-d, dim=-1)


class DigerModel(nn.Module):
    """物品塔(Tokenizer) + 生成式推荐塔, 两者通过可微语义 ID 相连。"""

    def __init__(self, d_in=24, mode="dril"):
        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(nn.Linear(d_in, 32), nn.GELU(),
                                 nn.Linear(32, L_LAYER * D_LAT))
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(K_CODE, D_LAT) * 0.5) for _ in range(L_LAYER)])
        self.gru = nn.GRU(L_LAYER * D_LAT, 48, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(48, K_CODE) for _ in range(L_LAYER)])

    def tokenize(self, feat, tau, gen, sigma=1.0, masks=None):
        """feat: (N, d_in) -> tokens (N, L), 可微 item 表示 (N, L*D)。

        正文中 DIGER 的连续表示直接是 r_v=(r_{v,1},...,r_{v,m}), 每个位置 j 独立地
        与第 j 个码本做相似度比较, 因此这里按层并行量化(不做残差累减)。
        """
        r = self.enc(feat).view(-1, L_LAYER, D_LAT)
        toks, embs, probs = [], [], []
        for l in range(L_LAYER):
            cur = r[:, l]
            if self.mode == "ste":
                t, e, p = ste_quantize(cur, self.codebooks[l])
            else:
                m = None if masks is None else masks[l]
                t, e, p = dril_quantize(cur, self.codebooks[l], tau, gen, sigma, m)
            toks.append(t)
            embs.append(e)
            probs.append(p)
        return torch.stack(toks, 1), torch.cat(embs, -1), probs

    def gen_loss(self, seq_emb, target_tok):
        """Next-SID Prediction: L_gen = -Σ_l log P(c_l | 历史序列)。"""
        h, _ = self.gru(seq_emb)
        last = h[:, -1]
        loss = 0.0
        correct = torch.ones(last.shape[0], dtype=torch.bool)
        for l in range(L_LAYER):
            logit = self.heads[l](last)
            loss = loss + F.cross_entropy(logit, target_tok[:, l])
            correct &= (logit.argmax(-1) == target_tok[:, l])
        return loss, correct.float().mean()


def perplexity(tokens):
    """码本 perplexity = exp(H(p)); 越接近 K 说明利用率越高。"""
    out = []
    for l in range(L_LAYER):
        cnt = torch.bincount(tokens[:, l], minlength=K_CODE).float()
        p = cnt / cnt.sum()
        h = -(p[p > 0] * p[p > 0].log()).sum()
        out.append(float(h.exp()))
    return out


def run(mode, feat, seqs, tgt, gen, epochs=18, tau=1.0, frqud=False,
        sdud=True, lam=0.3, beta=0.9, r_hot=1.5):
    torch.manual_seed(42)
    model = DigerModel(feat.shape[1], mode="ste" if mode == "ste" else "dril")
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    freq = [torch.full((K_CODE,), 1.0 / K_CODE) for _ in range(L_LAYER)]
    sigma, traj = 1.0, []
    for ep in range(epochs):
        masks = None
        if frqud:
            gamma = r_hot / K_CODE
            masks = [(f > gamma).float() for f in freq]
        toks, emb, _ = model.tokenize(feat, tau, gen, sigma, masks)
        seq_emb = emb[seqs]                        # (B, T, L*D)
        tgt_tok = toks[tgt]
        loss, acc = model.gen_loss(seq_emb, tgt_tok)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lg = float(loss)
        if sdud and mode != "ste":                 # SDUD: σ* = max(0, sqrt(L_gen)-λ)
            sigma = max(0.0, math.sqrt(max(lg, 0.0)) - lam)
        if frqud:                                  # 频率 EMA
            for l in range(L_LAYER):
                cnt = torch.bincount(toks[:, l], minlength=K_CODE).float()
                freq[l] = beta * freq[l] + (1 - beta) * cnt / cnt.sum()
        traj.append((ep + 1, lg, sigma, float(acc)))
    with torch.no_grad():
        toks, emb, _ = model.tokenize(feat, 1e-6, gen, 0.0)   # 推理: 确定性 argmax
        seq_emb = emb[seqs]
        loss, acc = model.gen_loss(seq_emb, toks[tgt])
    uniq = len({tuple(t.tolist()) for t in toks})
    return traj, perplexity(toks), float(loss), float(acc), uniq, freq


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    # ---------- 1. Gumbel-Max Trick 数值验证 ----------
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    N = 200000
    g = gumbel_noise((N, 4), gen)
    emp = torch.bincount((logits + g).argmax(-1), minlength=4).float() / N
    ref = F.softmax(logits, -1)
    print("=" * 84)
    print("表 1  Gumbel-Max Trick 验证: argmax_i(l_i + g_i) ~ Softmax(l)")
    print(f"      采样 {N} 次, logits = {logits.tolist()}")
    print("-" * 84)
    print(f"{'类别 i':>8}{'logit':>10}{'Softmax(l)':>14}{'argmax 经验频率':>18}"
          f"{'绝对误差':>12}")
    for i in range(4):
        print(f"{i:>8}{logits[i]:>10.1f}{ref[i]:>14.4f}{emp[i]:>18.4f}"
              f"{abs(ref[i] - emp[i]):>12.4f}")

    # ---------- 2. 合成数据 ----------
    D_IN, T, B = 24, 6, 256
    centers = torch.randn(6, D_IN, generator=gen) * 2.0
    item_grp = torch.randint(0, 6, (N_ITEM,), generator=gen)
    feat = centers[item_grp] + torch.randn(N_ITEM, D_IN, generator=gen) * 0.4
    seqs, tgt = [], []
    for _ in range(B):
        grp = int(torch.randint(0, 6, (1,), generator=gen))
        pool = (item_grp == grp).nonzero().flatten()
        pick = pool[torch.randint(0, len(pool), (T + 1,), generator=gen)]
        seqs.append(pick[:T])
        tgt.append(int(pick[T]))
    seqs = torch.stack(seqs)
    tgt = torch.tensor(tgt)

    # ---------- 3. SDUD 轨迹 ----------
    traj_d, ppl_d, loss_d, acc_d, uq_d, _ = run("dril", feat, seqs, tgt, gen)
    print()
    print("表 2  SDUD 噪声尺度自适应衰减(λ=0.3): σ* = max(0, sqrt(L_gen) - λ)")
    print("-" * 84)
    print(f"{'epoch':>7}{'L_gen':>12}{'sqrt(L_gen)':>14}{'σ*':>10}{'训练 SID 准确率':>18}")
    for ep, lg, sg, ac in traj_d:
        if ep % 3 == 1 or ep == len(traj_d):
            print(f"{ep:>7}{lg:>12.4f}{math.sqrt(max(lg, 0)):>14.4f}{sg:>10.4f}"
                  f"{ac:>18.4f}")
    print("-" * 84)
    print("可见生成损失下降 -> 最优噪声尺度自动减小 -> 从探索平滑过渡到利用。")

    # ---------- 4. 三种离散化方式对比 ----------
    traj_s, ppl_s, loss_s, acc_s, uq_s, _ = run("ste", feat, seqs, tgt, gen)
    traj_f, ppl_f, loss_f, acc_f, uq_f, _ = run("dril", feat, seqs, tgt, gen,
                                                frqud=True)
    print()
    print("表 3  离散化方式对比(推理时统一关闭噪声, 走确定性 argmax)")
    print("-" * 84)
    print(f"{'方式':<22}{'L_gen':>10}{'SID 全命中率':>14}"
          f"{'码本 perplexity(L1/L2/L3)':>28}{'不同 SID 数':>12}")
    for name, lo, ac, pp, uq in (
            ("STE(最近邻直通)", loss_s, acc_s, ppl_s, uq_s),
            ("DRIL(Gumbel+SDUD)", loss_d, acc_d, ppl_d, uq_d),
            ("DRIL+FrqUD", loss_f, acc_f, ppl_f, uq_f)):
        pps = "/".join(f"{x:.2f}" for x in pp)
        print(f"{name:<22}{lo:>10.4f}{ac:>14.4f}{pps:>28}{uq:>12}")
    print("-" * 84)
    print(f"注: 每层码本上限 perplexity = K = {K_CODE}; Gumbel 噪声的随机探索提高了")
    print("    码字访问覆盖, 对应正文所述'缓解码本坍塌、提高语义空间利用率'。")

    # ---------- 5. FrqUD 的高/低频划分 ----------
    print()
    print("表 4  FrqUD 码字划分示例(β=0.9, r=1.5, γ=r/K={:.4f})".format(1.5 / K_CODE))
    print("-" * 84)
    *_, freq_f = run("dril", feat, seqs, tgt, gen, frqud=True)
    gamma = r_hot = 1.5 / K_CODE
    for l in range(L_LAYER):
        f = freq_f[l]
        hi = (f > gamma).nonzero().flatten().tolist()
        lo = [i for i in range(K_CODE) if i not in hi]
        print(f"  第 {l + 1} 层 EMA 频率 = [" + ", ".join(f"{x:.3f}" for x in f) + "]")
        print(f"         I_high(保留 Gumbel 探索) = {hi};  I_low(确定性 Softmax) = {lo}")
    print("-" * 84)
    print("正文: 对过度使用的热门码字保留探索, 对低频码字采用稳定的确定性分配。")


if __name__ == "__main__":
    main()
