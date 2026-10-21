"""对应正文第 9 章 §9.4.1「Transformer 模型」与 §9.4.2「生成式推荐训练目标」。

正文要点与本实现的对应关系:
  * 自注意力: Attention(Q,K,V) = Softmax(Q K^T / sqrt(d_k)) V, 生成式推荐用因果
    掩码(Causal Mask)保证位置 t 只能看到 s_{<t};
  * 多头: MultiHead = Concat(head_1,...,head_h) W_O;
  * FFN(x) = W_2 sigma(W_1 x + b_1) + b_2, sigma 取 GELU;
  * Pre-LN 残差结构(正文: GPT-3/LLaMA 等主流 LLM 采用) y = x + f(LayerNorm(x));
  * LayerNorm(y) = gamma (y - mu)/sqrt(sigma^2 + eps) + beta;
  * 训练目标 NTP: P(s_t|s_<t) = Softmax(z_{t,s_t}); max_theta sum_t log P(s_t|s_<t),
    即交叉熵损失 L = -1/T sum_t log P(s_t|s_<t)。

本脚本从零实现一个 Decoder-only Transformer(手写 scaled dot-product attention +
causal mask + 可学习位置编码), 把每个物品表示为 L=3 个语义 Token, 用户行为序列
展开成语义 Token 序列后做 Next Token Prediction, 打印逐 epoch 的损失/困惑度/
逐层 Token 准确率, 以及整条语义 ID 的 top-k 命中率。
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L_LAYER = 3          # 每个物品的语义 ID 层数
K_CODE = 16          # 每层码本大小


class CausalSelfAttention(nn.Module):
    """手写多头因果自注意力(不使用 nn.MultiheadAttention)。"""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_k = n_head, d_model // n_head
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.W_qkv(x).split(D, dim=2)
        shape = (B, T, self.n_head, self.d_k)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # Softmax(QK^T / sqrt(d_k)) V, 上三角置 -inf 实现因果掩码
        att = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.W_O(out)


class Block(nn.Module):
    """Pre-LN 结构: y = x + f(LayerNorm(x))。"""

    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class SemanticIDDecoder(nn.Module):
    """Decoder-only Transformer, 词表 = L*K 个语义 Token + 1 个 BOS。"""

    def __init__(self, d_model=48, n_head=4, n_layer=2, max_len=40):
        super().__init__()
        self.vocab = L_LAYER * K_CODE + 1
        self.bos = self.vocab - 1
        self.tok_emb = nn.Embedding(self.vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)   # 可学习位置编码
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, 4 * d_model) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def flat_token(layer: int, code: int) -> int:
    """把 (层, 码字) 映射为全局 token id, 使不同层的 token 不冲突。"""
    return layer * K_CODE + code


def build_corpus(n_item: int, n_user: int, seq_item: int, gen: torch.Generator):
    """合成: 每个物品一个语义 ID; 每个用户有一个小兴趣集合, 行为序列在其中游走。

    这样"下一个物品"的语义 ID 可由上下文推断(模型需要学会关注历史 Token),
    符合正文中生成式推荐"用历史语义 Token 序列预测下一个物品语义 ID"的设定。
    """
    sid = torch.stack([torch.randint(0, K_CODE, (n_item,), generator=gen)
                       for _ in range(L_LAYER)], 1)
    seqs = []
    for _ in range(n_user):
        fav = int(torch.randint(0, K_CODE, (1,), generator=gen))   # 偏好大类
        pool = [i for i in range(n_item) if int(sid[i, 0]) == fav] or list(range(n_item))
        k = min(4, len(pool))
        interest = [pool[int(torch.randint(0, len(pool), (1,), generator=gen))]
                    for _ in range(k)]
        pick = [interest[int(torch.randint(0, k, (1,), generator=gen))]
                for _ in range(seq_item)]
        seqs.append(pick)
    return sid, seqs


def to_tokens(seqs, sid, bos: int) -> torch.Tensor:
    rows = []
    for s in seqs:
        row = [bos]
        for it in s:
            row += [flat_token(l, int(sid[it, l])) for l in range(L_LAYER)]
        rows.append(row)
    return torch.tensor(rows)


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    N_ITEM, N_USER, SEQ_ITEM = 300, 1200, 8
    sid, seqs = build_corpus(N_ITEM, N_USER, SEQ_ITEM, gen)
    model = SemanticIDDecoder()
    data = to_tokens(seqs, sid, model.bos)
    n_tr = int(0.85 * len(data))
    train, valid = data[:n_tr], data[n_tr:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    print("=" * 84)
    print(f"Decoder-only Transformer (d_model=48, 4 heads, 2 blocks), "
          f"词表={model.vocab}")
    print(f"物品数={N_ITEM}, 用户序列数={N_USER}, 每序列 {SEQ_ITEM} 物品 x "
          f"{L_LAYER} 层 = {SEQ_ITEM * L_LAYER} tokens")
    print("-" * 84)
    print(f"{'epoch':>6}{'train CE':>11}{'valid CE':>11}{'valid PPL':>12}"
          f"{'层1 acc':>10}{'层2 acc':>10}{'层3 acc':>10}")

    for ep in range(1, 9):
        model.train()
        perm = torch.randperm(len(train), generator=gen)
        tot, nb = 0.0, 0
        for i in range(0, len(train), 64):
            b = train[perm[i:i + 64]]
            logits = model(b[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   b[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            lv = model(valid[:, :-1])
            tgt = valid[:, 1:]
            vce = F.cross_entropy(lv.reshape(-1, model.vocab), tgt.reshape(-1)).item()
            pred = lv.argmax(-1)
            accs = []
            for l in range(L_LAYER):
                # 目标位置 t 对应物品内层号 = t % L_LAYER
                pos = torch.arange(tgt.shape[1]) % L_LAYER == l
                accs.append((pred[:, pos] == tgt[:, pos]).float().mean().item())
        print(f"{ep:>6}{tot / nb:>11.4f}{vce:>11.4f}{math.exp(vce):>12.2f}"
              + "".join(f"{a:>10.1%}" for a in accs))

    # ---------------- 整条语义 ID 的 top-k 命中 ----------------
    print()
    print("表 2  下一物品完整语义 ID 的 top-k 命中率(验证集, 贪心展开每层 top-k)")
    print("-" * 84)
    sid_set = {tuple(r.tolist()) for r in sid}
    model.eval()
    hits = {1: 0, 5: 0, 10: 0}
    legal = 0
    n_eval = min(120, len(valid))
    with torch.no_grad():
        for i in range(n_eval):
            row = valid[i]
            prefix = row[: 1 + (SEQ_ITEM - 1) * L_LAYER].unsqueeze(0)
            truth = tuple(int(row[1 + (SEQ_ITEM - 1) * L_LAYER + l]) % K_CODE
                          for l in range(L_LAYER))
            # 逐层取 top-10 候选, 用 beam=10 的近似展开
            beams = [([], 0.0)]
            for l in range(L_LAYER):
                nxt = []
                for toks, sc in beams:
                    inp = torch.cat(
                        [prefix, torch.tensor([[flat_token(j, c)
                                                for j, c in enumerate(toks)]],
                                              dtype=torch.long)], 1) \
                        if toks else prefix
                    lp = model(inp)[0, -1].log_softmax(-1)
                    span = lp[l * K_CODE:(l + 1) * K_CODE]
                    top = span.topk(min(10, K_CODE))
                    for s, c in zip(top.values.tolist(), top.indices.tolist()):
                        nxt.append((toks + [c], sc + s))
                beams = sorted(nxt, key=lambda t: -t[1])[:10]
            ranked = [tuple(t) for t, _ in beams]
            if ranked and ranked[0] in sid_set:
                legal += 1
            for k in hits:
                if truth in ranked[:k]:
                    hits[k] += 1
    print(f"{'k':>6}{'HR@k':>12}{'命中数':>10}")
    for k in sorted(hits):
        print(f"{k:>6}{hits[k] / n_eval:>12.1%}{hits[k]:>10}")
    print(f"\n评估样本数 = {n_eval}; top-1 路径落在合法语义 ID 表内的比例 = "
          f"{legal / n_eval:.1%}")
    print(f"随机猜中概率 HR@1 = 1/K^L = {1 / K_CODE ** L_LAYER:.5f}")
    print("-" * 84)
    print("注: 本实现严格采用正文 §9.4.2 的 NTP 目标(交叉熵), Pre-LN + 因果掩码;")
    print("    不同层的语义 Token 使用互不重叠的 id 区间, 解码时只在当前层区间内取 top-k。")


if __name__ == "__main__":
    main()
