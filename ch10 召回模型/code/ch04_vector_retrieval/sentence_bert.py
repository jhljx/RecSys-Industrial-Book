"""
Sentence-BERT（SBERT）模型（第10章 §10.3.3 基于预训练语义模型的向量召回）

核心思想：
- 原生 BERT 做句子相似度要用 Cross-Encoder：把两句拼成 [CLS] A [SEP] B [SEP] 联合编码。
  精度高，但一个 Query 要和 N 条候选各做一次前向，大规模检索复杂度 O(N) 次 Transformer
  推理，两两匹配更是 O(N^2)，完全无法满足在线召回的延迟要求。
- SBERT 借鉴孪生网络（Siamese Network）改成 Bi-Encoder：两句分别过**共享参数**的同一个
  编码器，各自 Pooling 得到定长句向量，再用余弦相似度比较：
      sim(u, v) = u^T v / (||u|| * ||v||)
  于是候选文本可以离线编码建 ANN 索引，在线只需编码 Query 一次 → O(1) 次推理 + 一次检索。
- Pooling 策略：CLS / Mean / Max，论文实验表明 Mean Pooling 最稳定，本实现采用 Mean Pooling。
- 训练目标：回归目标 MSE(cos(u,v), gold) 与检索常用的 Triplet Loss
      L = max(d(a, p) - d(a, n) + margin, 0)
  本文件两者都实现。

论文出处：Reimers & Gurevych, Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks, EMNLP 2019.（PyTorch autograd 实现）
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CLS = "[CLS]"


class SentenceBERT(nn.Module):
    """
    孪生（Siamese）句向量编码器：两句共享同一套参数。

    句向量： e = W_proj * MeanPool( TransformerEncoder( TokenEmb + PosEmb ) )
    使用 PyTorch autograd 完成端到端训练。
    """

    def __init__(self, vocab: list, d_model: int = 16, n_heads: int = 2,
                 d_ff: int = 32, max_len: int = 8, lr: float = 0.05):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.lr = lr
        self.tok2id = {w: i for i, w in enumerate(vocab)}

        self.tok_emb = nn.Embedding(len(vocab), d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(d_model, d_model)

    def _ids(self, tokens: list) -> list:
        return [self.tok2id[w] for w in ([CLS] + tokens)[: self.max_len]]

    def encode(self, tokens: list) -> torch.Tensor:
        """返回 Mean Pooling 后的句向量 (d_model,)。"""
        ids = self._ids(tokens)
        T = len(ids)
        ids_t = torch.tensor([ids], dtype=torch.long)                 # (1, T)
        pos_t = torch.arange(T, dtype=torch.long).unsqueeze(0)        # (1, T)
        x = self.tok_emb(ids_t) + self.pos_emb(pos_t)                 # (1, T, d)
        h = self.encoder(x)                                           # (1, T, d)
        pooled = h[0].mean(dim=0)                                     # (d,)
        return self.proj(pooled)                                      # (d,)

    def embed(self, tokens: list) -> list:
        """对外接口：只拿句向量（离线批量编码候选文本时用）。"""
        with torch.no_grad():
            return self.encode(tokens).tolist()

    def train_regression(self, pairs: list, epochs: int = 20) -> None:
        """回归目标：L = ( cos(u, v) - gold )^2，让句向量余弦逼近人工标注的相似度。"""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        data = list(pairs)
        for ep in range(epochs):
            random.shuffle(data)
            tot = 0.0
            self.train()
            for sa, sb, gold in data:
                u = self.encode(sa)
                v = self.encode(sb)
                cos = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0))
                gold_t = torch.tensor([[gold]], dtype=torch.float)
                loss = F.mse_loss(cos, gold_t.squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot += loss.item()
            if ep % 4 == 0 or ep == epochs - 1:
                print(f"    epoch {ep:2d}  mse_loss={tot / len(data):.4f}")

    def triplet_loss(self, a: list, p: list, n: list, margin: float = 0.2) -> float:
        """Triplet Loss：L = max( d(a,p) - d(a,n) + margin, 0 )，d 取余弦距离 1-cos。"""
        with torch.no_grad():
            ea = torch.tensor(self.embed(a))
            ep_ = torch.tensor(self.embed(p))
            en = torch.tensor(self.embed(n))
            d_pos = 1 - F.cosine_similarity(ea.unsqueeze(0), ep_.unsqueeze(0)).item()
            d_neg = 1 - F.cosine_similarity(ea.unsqueeze(0), en.unsqueeze(0)).item()
            return max(d_pos - d_neg + margin, 0.0)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    # 合成语料：6 个商品主题簇，同簇内句子语义相近（gold=1），跨簇不相关（gold=0）
    clusters = [
        [["蓝牙", "耳机", "降噪"], ["无线", "耳机", "续航"], ["耳机", "降噪", "续航"]],
        [["电竞", "显示器", "高刷"], ["游戏", "显示器", "面板"], ["高刷", "面板", "显示器"]],
        [["男士", "纯棉", "短袖"], ["男士", "透气", "T恤"], ["纯棉", "透气", "短袖"]],
        [["智能", "手表", "运动"], ["运动", "手表", "跑步"], ["智能", "跑步", "手表"]],
        [["笔记本", "轻薄", "商务"], ["轻薄", "便携", "办公"], ["笔记本", "商务", "办公"]],
        [["保温杯", "不锈钢", "大容量"], ["保温杯", "便携", "大容量"], ["不锈钢", "大容量", "保温杯"]],
    ]
    vocab = [CLS] + sorted({w for cl in clusters for s in cl for w in s})
    model = SentenceBERT(vocab, d_model=16, n_heads=2, d_ff=32, max_len=8, lr=0.05)

    pairs = []
    for ci, cl in enumerate(clusters):
        for a in range(len(cl)):
            pairs.append((cl[a], cl[(a + 1) % len(cl)], 1.0))
            for _ in range(2):
                cj = random.choice([j for j in range(len(clusters)) if j != ci])
                pairs.append((cl[a], random.choice(clusters[cj]), 0.0))

    probe_pos = (["蓝牙", "耳机", "续航"], ["无线", "耳机", "降噪"])
    probe_neg = (["蓝牙", "耳机", "续航"], ["保温杯", "不锈钢", "大容量"])

    def cos_sim(a, b):
        av = torch.tensor(model.embed(a))
        bv = torch.tensor(model.embed(b))
        return F.cosine_similarity(av.unsqueeze(0), bv.unsqueeze(0)).item()

    before_p = cos_sim(*probe_pos)
    before_n = cos_sim(*probe_neg)

    print(f"孪生编码器：vocab={len(vocab)}  d_model={model.d_model}  "
          f"头数=2  Pooling=Mean  训练对={len(pairs)}")
    print(f"\n微调前：同义句相似度={before_p:.4f}   不相关句相似度={before_n:.4f}")
    print("\n回归目标（MSE）微调：")
    model.train_regression(pairs, epochs=24)

    after_p = cos_sim(*probe_pos)
    after_n = cos_sim(*probe_neg)
    print(f"\n微调后：同义句相似度={after_p:.4f}（{after_p - before_p:+.4f}）   "
          f"不相关句相似度={after_n:.4f}（{after_n - before_n:+.4f}）")

    tl = model.triplet_loss(
        ["蓝牙", "耳机", "续航"], ["无线", "耳机", "降噪"], ["保温杯", "不锈钢", "大容量"]
    )
    print(f"Triplet Loss（anchor=蓝牙耳机续航, pos=无线耳机降噪, neg=保温杯）= {tl:.4f}")

    corpus = [["蓝牙", "无线", "耳机"], ["电竞", "游戏", "高刷"], ["男士", "纯棉", "T恤"],
              ["智能", "手表", "跑步"], ["笔记本", "轻薄", "办公"], ["保温杯", "不锈钢", "便携"]]
    index = [(c, torch.tensor(model.embed(c))) for c in corpus]
    query = ["耳机", "降噪", "无线"]
    q_vec = torch.tensor(model.embed(query))
    hits = sorted(
        [(F.cosine_similarity(q_vec.unsqueeze(0), v.unsqueeze(0)).item(), c) for c, v in index],
        reverse=True
    )
    print(f"\n【语义召回】Query = {' '.join(query)}  （离线编码 {len(index)} 条，在线仅 1 次前向）")
    for s, c in hits[:3]:
        print(f"  {s:.4f}  {' '.join(c)}")
    print(f"\n对比：Cross-Encoder 需要 {len(corpus)} 次 Transformer 前向才能给出同样的排序；")
    print("      Bi-Encoder 把编码与匹配解耦，这正是工业界 Embedding 召回的标准做法。")
