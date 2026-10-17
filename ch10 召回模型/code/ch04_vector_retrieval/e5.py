"""
E5 模型（第10章 §10.3.3 基于预训练语义模型的向量召回）

核心思想：
- E5（EmbEddings from bidirEctional Encoder rEpresentations）用大规模**弱监督**文本对
  （CCPairs：标题-正文、问题-回答、Query-点击文档等天然配对数据，无需人工标注）做对比预训练，
  得到通用的文本向量模型。
- 关键设计是**非对称的指令前缀**：查询侧统一加 "query: "，文档侧统一加 "passage: "。
  检索任务中 Query（短、口语化）与 Passage（长、书面化）形态差异很大，加前缀让**同一个共享
  编码器**知道自己当前在编码哪一侧，从而把两类文本对齐到同一语义空间，而不必维护两个塔。
      e_q = Encoder("query: "   + q)
      e_p = Encoder("passage: " + p)
- 训练目标为带温度系数的 InfoNCE，负样本 = Batch 内其他文档（In-Batch Negatives）
  + 显式挖掘的**难负样本**（Hard Negatives，与 Query 字面重叠但语义不匹配）：
      L = -log[ exp(sim(e_q, e_p+)/τ) / ( exp(sim(e_q, e_p+)/τ) + Σ_{p-} exp(sim(e_q, e_p-)/τ) ) ]
  难负样本迫使模型学会语义级区分，而不是停留在词面匹配。

论文出处：Wang et al., Text Embeddings by Weakly-Supervised Contrastive Pre-training,
arXiv 2022.（PyTorch autograd 实现）
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Q_PREFIX, P_PREFIX = "query:", "passage:"


class E5Encoder(nn.Module):
    """
    E5 文本向量编码器：查询与文档共享同一套参数，仅靠指令前缀区分两侧。

    向量： e = W_proj * MeanPool( TransformerEncoder( TokenEmb + PosEmb ) )，输入首位是指令前缀 Token。
    """

    def __init__(self, vocab: list, d_model: int = 16, n_heads: int = 2, d_ff: int = 32,
                 max_len: int = 8, tau: float = 0.08, lr: float = 0.05):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.tau = tau
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

    def _encode(self, tokens: list, prefix: str) -> torch.Tensor:
        """按指定前缀编码文本，返回句向量 (d_model,)。"""
        ids = [self.tok2id[w] for w in ([prefix] + tokens)[: self.max_len]]
        T = len(ids)
        ids_t = torch.tensor([ids], dtype=torch.long)
        pos_t = torch.arange(T, dtype=torch.long).unsqueeze(0)
        x = self.tok_emb(ids_t) + self.pos_emb(pos_t)   # (1, T, d)
        h = self.encoder(x)                              # (1, T, d)
        pooled = h[0].mean(dim=0)                        # (d,)
        return self.proj(pooled)

    def embed_query(self, tokens: list) -> torch.Tensor:
        """查询侧：加 "query: " 前缀。"""
        with torch.no_grad():
            return self._encode(tokens, Q_PREFIX)

    def embed_passage(self, tokens: list) -> torch.Tensor:
        """文档侧：加 "passage: " 前缀（可离线批量预计算，灌入 ANN 索引）。"""
        with torch.no_grad():
            return self._encode(tokens, P_PREFIX)

    def train_model(self, data: list, epochs: int = 40, batch_size: int = 4) -> None:
        """
        data 为弱监督三元组 (query, 正文档, 难负文档)。
        候选集 = 本 Batch 的全部正文档（彼此互为 In-Batch 负样本）
        + 全部难负样本；第 i 条查询的正确答案是第 i 个正文档。
        InfoNCE loss 同 SimCSE。
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        ds = list(data)
        for ep in range(epochs):
            random.shuffle(ds)
            tot, nb = 0.0, 0
            for st in range(0, len(ds), batch_size):
                bt = ds[st: st + batch_size]
                if len(bt) < 2:
                    continue
                self.train()
                B = len(bt)
                # 查询向量 (B, d)
                EQ = torch.stack([self._encode(q, Q_PREFIX) for q, _, _ in bt])
                # 正文档 (B, d) + 难负样本 (B, d) → 候选 (2B, d)
                EP = torch.stack([self._encode(p, P_PREFIX) for _, p, _ in bt])
                EN = torch.stack([self._encode(n, P_PREFIX) for _, _, n in bt])
                EC = torch.cat([EP, EN], dim=0)    # (2B, d)

                # InfoNCE：第 i 个查询的正例是第 i 个正文档（EC 中下标 i）
                EQ_n = F.normalize(EQ, dim=-1)
                EC_n = F.normalize(EC, dim=-1)
                sim = EQ_n @ EC_n.T / self.tau     # (B, 2B)
                labels = torch.arange(B)           # 正例在前 B 列的对角线
                loss = F.cross_entropy(sim, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot += loss.item()
                nb += 1
            if ep % 8 == 0 or ep == epochs - 1:
                print(f"    epoch {ep:2d}  infonce_loss={tot / max(nb, 1):.4f}")

    def search(self, query: list, index: list, top_k: int = 3,
               prefix: str = Q_PREFIX) -> list:
        """用查询向量在离线建好的文档索引中检索 Top-K。"""
        with torch.no_grad():
            qv = self._encode(query, prefix)
            qv_n = F.normalize(qv.unsqueeze(0), dim=-1)
            results = []
            for doc, dv in index:
                dv_n = F.normalize(dv.unsqueeze(0), dim=-1)
                score = (qv_n @ dv_n.T).item()
                results.append((score, doc))
            return sorted(results, key=lambda x: -x[0])[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    # 弱监督数据：每个商品主题一条正文档 + 一条难负文档
    topics = [
        (["蓝牙", "耳机", "主动", "降噪", "续航"], ["蓝牙", "音箱", "低音", "户外"],
         [["降噪", "耳机"], ["蓝牙", "耳机", "续航"]]),
        (["电竞", "显示器", "高刷", "面板", "游戏"], ["显示器", "支架", "升降", "底座"],
         [["高刷", "显示器"], ["电竞", "显示器", "游戏"]]),
        (["男士", "纯棉", "短袖", "透气", "T恤"], ["男士", "牛仔", "长裤", "宽松"],
         [["纯棉", "短袖"], ["男士", "透气", "短袖"]]),
        (["笔记本", "轻薄", "商务", "办公", "便携"], ["笔记本", "内胆包", "防摔", "收纳"],
         [["轻薄", "笔记本"], ["商务", "办公", "笔记本"]]),
    ]
    triples = [(q, pos, neg) for pos, neg, qs in topics for q in qs]
    words = sorted({w for q, p, n in triples for s in (q, p, n) for w in s})
    vocab = [Q_PREFIX, P_PREFIX] + words
    model = E5Encoder(vocab, d_model=16, n_heads=2, d_ff=32, max_len=8, tau=0.08, lr=0.05)
    print(f"E5 配置：vocab={len(vocab)}（含 \"{Q_PREFIX} \" / \"{P_PREFIX} \" 两个指令前缀）"
          f"  d_model={model.d_model}  τ={model.tau}")
    print(f"弱监督三元组 {len(triples)} 条（query, 正文档, 难负文档），共享编码器单塔")

    corpus = [doc for pos, neg, _ in topics for doc in (pos, neg)]
    gold = {tuple(q): tuple(p) for q, p, _ in triples}
    hard = {tuple(q): tuple(n) for q, _, n in triples}

    def top1_acc(prefix: str) -> float:
        """用指定前缀编码查询，统计 Top-1 命中正文档的比例。"""
        idx = [(doc, model.embed_passage(doc)) for doc in corpus]
        hit = sum(1 for q, _, _ in triples
                  if tuple(model.search(q, idx, 1, prefix)[0][1]) == gold[tuple(q)])
        return hit / len(triples)

    print(f"\n训练前 Top-1 准确率：{top1_acc(Q_PREFIX):.2%}（文档库 {len(corpus)} 条）")
    print("\n弱监督对比预训练（InfoNCE = In-Batch 负样本 + 难负样本）：")
    model.train_model(triples, epochs=40, batch_size=4)

    index = [(doc, model.embed_passage(doc)) for doc in corpus]
    acc = top1_acc(Q_PREFIX)
    print(f"\n训练后 Top-1 准确率：{acc:.2%}")
    print("\n【非对称检索：query 侧加 \"query: \"，文档侧加 \"passage: \"】")
    for q in [["降噪", "耳机"], ["高刷", "显示器"], ["纯棉", "短袖"], ["轻薄", "笔记本"]]:
        res = model.search(q, index, len(corpus))
        s1, d1 = res[0]
        tag = "✓" if tuple(d1) == gold[tuple(q)] else "✗"
        rk = next(i for i, (_, d) in enumerate(res, 1) if tuple(d) == hard[tuple(q)])
        print(f"  Query={' '.join(q):<11} Top-1 {tag} {' '.join(d1):<19}{s1:+.3f}"
              f"  |  难负样本「{' '.join(hard[tuple(q)])}」排名 {rk}/{len(corpus)}  {res[rk - 1][0]:+.3f}")

    print(f"\n【前缀消融】把查询也错误地加上 \"{P_PREFIX} \" 前缀："
          f"Top-1 准确率 {acc:.2%} → {top1_acc(P_PREFIX):.2%}")
    print("结论：指令前缀让一个共享编码器同时胜任查询侧与文档侧的编码，")
    print("      难负样本把「字面重叠但品类不同」的文档压到正例之下，")
    print("      文档向量可离线预计算入库，在线只需编码 Query 一次即可完成向量召回。")
