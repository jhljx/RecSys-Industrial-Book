r"""对应正文第 9 章 §9.4.2「生成式推荐训练目标」中 NTP 之外的两种目标。

正文在 NTP(Next Token Prediction)之后提到:
  * 多 Token 预测(Multi-Token Prediction, MTP): 在同一位置同时预测未来多个 Token,
    对应损失 L_MTP = -sum_t sum_{i=1}^{n} log P(s_{t+i} | s_<t), 正文指出它能提供
    更强的监督信号并天然适配投机解码(DeepSeek-V3 / Better&Faster LLM 的做法);
  * 扩散式(Diffusion / Masked LM 式)目标: 随机掩码一部分 Token 并一次性预测,
    L_mask = -sum_{t in M} log P(s_t | s_{\M}), 对应正文提到的大语言扩散模型路线。

本脚本在同一份合成语义 Token 序列、同一个小 Transformer backbone 上对比三种目标:
  (A) NTP          : 一个输出头, 预测 s_{t+1}
  (B) MTP (n=2)    : 两个输出头, 分别预测 s_{t+1}, s_{t+2}
  (C) Masked 预测  : 随机掩掉 15% 的 Token 一次性还原(双向注意力)
输出三者在**同一验证集、同一道题**(预测末尾两个 Token)上的准确率, 以及解出这两个
Token 各自所需的前向次数。为保证可比: NTP 走真实两步自回归(第 2 步喂自己的预测,
不用 teacher forcing), MTP 一次前向由两个头给出, Masked 一次前向还原两个 [MASK]。
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_CODE, L_LAYER = 16, 3
VOCAB = K_CODE * L_LAYER + 2          # +BOS +MASK


class Backbone(nn.Module):
    """共享的小 Transformer; causal=True 为 Decoder-only, False 为双向。"""

    def __init__(self, d=40, n_head=4, n_block=2, max_len=32, causal=True):
        super().__init__()
        self.causal = causal
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d, n_head, 4 * d, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_block, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.d = d

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        mask = (torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
                if self.causal else None)
        return self.ln(self.enc(x, mask=mask))


def make_data(n_seq: int, n_item: int, seq_len: int, gen):
    """合成: 每个序列对应一个小兴趣集合, 在其中游走, 使下一个物品可由上下文推断。"""
    sid = torch.stack([torch.randint(0, K_CODE, (n_item,), generator=gen)
                       for _ in range(L_LAYER)], 1)
    rows = []
    for _ in range(n_seq):
        fav = int(torch.randint(0, K_CODE, (1,), generator=gen))
        pool = [i for i in range(n_item) if int(sid[i, 0]) == fav] or list(range(n_item))
        k = min(4, len(pool))
        interest = [pool[int(torch.randint(0, len(pool), (1,), generator=gen))]
                    for _ in range(k)]
        row = [VOCAB - 2]            # BOS
        for _ in range(seq_len):
            it = interest[int(torch.randint(0, k, (1,), generator=gen))]
            row += [l * K_CODE + int(sid[it, l]) for l in range(L_LAYER)]
        rows.append(row)
    return torch.tensor(rows)


def train_ntp(data, gen, epochs=6):
    bb = Backbone(causal=True)
    head = nn.Linear(bb.d, VOCAB, bias=False)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(head.parameters()), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(data), generator=gen)
        for i in range(0, len(data), 64):
            b = data[perm[i:i + 64]]
            logits = head(bb(b[:, :-1]))
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return bb, [head]


def train_mtp(data, gen, n_future=2, epochs=6):
    """MTP: 共享 backbone + n 个独立输出头, 第 i 个头预测 s_{t+i}。"""
    bb = Backbone(causal=True)
    heads = nn.ModuleList([nn.Linear(bb.d, VOCAB, bias=False) for _ in range(n_future)])
    opt = torch.optim.AdamW(list(bb.parameters()) + list(heads.parameters()), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(data), generator=gen)
        for i in range(0, len(data), 64):
            b = data[perm[i:i + 64]]
            T = b.shape[1] - n_future
            h = bb(b[:, :T])
            loss = 0.0
            for j, hd in enumerate(heads):
                tgt = b[:, 1 + j: 1 + j + T]
                loss = loss + F.cross_entropy(
                    hd(h).reshape(-1, VOCAB), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return bb, list(heads)


def train_masked(data, gen, p_mask=0.15, epochs=6):
    """掩码式(扩散/MLM 风格): 双向注意力, 一次性还原被掩位置。"""
    bb = Backbone(causal=False)
    head = nn.Linear(bb.d, VOCAB, bias=False)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(head.parameters()), lr=3e-3)
    MASK = VOCAB - 1
    for _ in range(epochs):
        perm = torch.randperm(len(data), generator=gen)
        for i in range(0, len(data), 64):
            b = data[perm[i:i + 64]].clone()
            m = torch.rand(b.shape, generator=gen) < p_mask
            m[:, 0] = False
            if not m.any():
                continue
            inp = b.clone()
            inp[m] = MASK
            logits = head(bb(inp))
            loss = F.cross_entropy(logits[m], b[m])
            opt.zero_grad(); loss.backward(); opt.step()
    return bb, [head]


@torch.no_grad()
def eval_last_two(bb, heads, valid, mode: str):
    """统一口径: 三种目标做**同一道题**——给定前缀 s_<T, 预测末尾两个 Token。

    三者只在"怎么把这两个 Token 解出来"上不同, 评测集合/指标完全一致:
      mode='ntp'    : 真实两步自回归 —— 先前向一次解出 s_T, 把**自己的预测**
                      拼回输入再前向一次解出 s_{T+1}(无 teacher forcing), 2 次前向;
      mode='mtp'    : 一次前向, 第 1/2 个头分别给出 s_T / s_{T+1}, 1 次前向;
      mode='masked' : 把末尾两位置置 [MASK], 双向注意力一次前向同时还原, 1 次前向。

    返回 (第 1 个 Token acc, 两个 Token 联合 acc, 前向次数)。
    """
    bb.eval()
    T = valid.shape[1] - 2               # 前缀长度; 目标为 valid[:, T], valid[:, T+1]
    tgt1, tgt2 = valid[:, T], valid[:, T + 1]

    if mode == "ntp":
        p1 = heads[0](bb(valid[:, :T]))[:, -1].argmax(-1)
        step2_in = torch.cat([valid[:, :T], p1[:, None]], 1)   # 拼接自己的预测
        p2 = heads[0](bb(step2_in))[:, -1].argmax(-1)
        n_fwd = 2
    elif mode == "mtp":
        h = bb(valid[:, :T])[:, -1]
        p1 = heads[0](h).argmax(-1)
        p2 = heads[1](h).argmax(-1)
        n_fwd = 1
    elif mode == "masked":
        MASK = VOCAB - 1
        inp = valid.clone()
        inp[:, T] = MASK
        inp[:, T + 1] = MASK
        logits = heads[0](bb(inp))
        p1 = logits[:, T].argmax(-1)
        p2 = logits[:, T + 1].argmax(-1)
        n_fwd = 1
    else:
        raise ValueError(mode)

    a1 = (p1 == tgt1).float().mean().item()
    a2 = ((p1 == tgt1) & (p2 == tgt2)).float().mean().item()
    return a1, a2, n_fwd


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    data = make_data(900, 240, seq_len=8, gen=gen)
    n_tr = int(0.85 * len(data))
    train, valid = data[:n_tr], data[n_tr:]

    print("=" * 84)
    print(f"训练目标对比 (序列数={len(data)}, 长度={data.shape[1]}, 词表={VOCAB})")
    print("-" * 84)

    torch.manual_seed(42)
    bb_a, h_a = train_ntp(train, torch.Generator().manual_seed(42))
    torch.manual_seed(42)
    bb_b, h_b = train_mtp(train, torch.Generator().manual_seed(42))
    torch.manual_seed(42)
    bb_c, h_c = train_masked(train, torch.Generator().manual_seed(42))

    a1, a2, fa = eval_last_two(bb_a, h_a, valid, mode="ntp")
    b1, b2, fb = eval_last_two(bb_b, h_b, valid, mode="mtp")
    c1, c2, fc = eval_last_two(bb_c, h_c, valid, mode="masked")

    print("表 1  三种训练目标在**同一道题**上的验证集表现")
    print("     (题面统一: 给定同一前缀, 预测末尾两个 Token; 指标口径完全一致)")
    print(f"{'目标':<26}{'注意力':<12}{'解出方式':<22}"
          f"{'第1个 Token acc':>17}{'两个 Token 联合':>17}")
    print(f"{'A NTP':<26}{'causal':<12}{'两步自回归':<22}{a1:>17.1%}{a2:>17.1%}")
    print(f"{'B MTP (n=2)':<26}{'causal':<12}{'一次前向 双头':<22}{b1:>17.1%}{b2:>17.1%}")
    print(f"{'C Masked (p=15%)':<26}{'双向':<12}{'一次前向 还原 2 个 [MASK]':<22}"
          f"{c1:>17.1%}{c2:>17.1%}")
    print("-" * 84)
    print("说明: NTP 第 2 步用的是**自己的预测**而非真值(无 teacher forcing), 因此")
    print("      三行的联合准确率可直接比较; 差异来自目标函数而非评测口径。")
    best = max([("A NTP", a2), ("B MTP (n=2)", b2), ("C Masked", c2)],
               key=lambda kv: kv[1])
    print(f"      本次实跑联合准确率最高者: {best[0]} ({best[1]:.1%});"
          f" A/B/C 第 1 个 Token acc = {a1:.1%}/{b1:.1%}/{c1:.1%}。")
    print("      注: 玩具规模(小词表 + 合成序列)下的绝对数值不代表真实系统结论,")
    print("      正文 §9.4.2 的论点是 MTP 提供更密集的监督信号并天然适配投机解码。")

    print()
    print("表 2  产出同样这 2 个 Token 所需的前向次数(由上面的评测实测返回)")
    print("-" * 84)
    print(f"{'目标':<26}{'输出头数':>9}{'解出 2 个 Token 的前向次数':>28}"
          f"{'外推: 生成 L=3 层':>20}")
    print(f"{'A NTP':<26}{1:>9}{fa:>28}{L_LAYER:>20}")
    print(f"{'B MTP (n=2)':<26}{2:>9}{fb:>28}{math.ceil(L_LAYER / 2):>20}")
    print(f"{'C Masked 一次性还原':<26}{1:>9}{fc:>28}{1:>20}")
    print("-" * 84)
    print(f"注: 同一道题 NTP 需 {fa} 次前向, MTP/Masked 各需 {fb}/{fc} 次——这正是正文所说")
    print("    MTP 适配投机解码、掩码/扩散式目标可并行产出整组 Token 的来源;")
    print("    但正文同时指出扩散式路线在推荐场景仍属探索方向。")


if __name__ == "__main__":
    main()
