"""
SimCSE 模型（第10章 §10.3.3 基于预训练语义模型的向量召回）

核心思想：
- 原生 BERT 句向量存在「各向异性 / 表示退化」：所有句向量挤在向量空间的一个狭窄锥形区域里，
  任意两句的余弦相似度都接近 1，几乎无法区分，直接拿来做向量召回效果很差。
- 无监督 SimCSE 的做法极其简洁：把**同一个句子**送入编码器两次，由于 Dropout 每次随机
  置零的神经元不同，得到两个略有差异的向量 (h_i, h_i^+)，它们构成**正样本对**；
  同一 Batch 内其他句子的表示则是负样本。Dropout 本身就是最小的数据增强。
- 训练目标为 InfoNCE（带温度系数 τ 的对比损失）：
      L_i = -log[ exp(sim(h_i, h_i^+)/τ) / Σ_{j=1..N} exp(sim(h_i, h_j^+)/τ) ]
  其中 sim 为余弦相似度。该损失同时优化两个性质：
      Alignment（对齐性）：正样本对要靠近；
      Uniformity（均匀性）：不同句子在超球面上要分散开 —— 正是这一项缓解了各向异性。
- 因此训练后的典型现象是：同义句相似度依然很高，而不相关句相似度显著下降，两者**拉开差距**。

论文出处：Gao, Yao & Chen, SimCSE: Simple Contrastive Learning of Sentence Embeddings,
EMNLP 2021.（PyTorch autograd 实现）
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CLS = "[CLS]"


class SimCSE(nn.Module):
    """
    无监督 SimCSE 句向量编码器。

    句向量： h = W_proj * MeanPool( TransformerEncoder( Dropout(TokenEmb + PosEmb) ) )
    训练时 Dropout 开启（两次前向得到正样本对），推理时关闭（结果确定）。
    """

    def __init__(self, vocab: list, d_model: int = 16, n_heads: int = 2, d_ff: int = 32,
                 max_len: int = 8, p_drop: float = 0.15, tau: float = 0.1, lr: float = 0.05):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.p_drop = p_drop
        self.tau = tau
        self.lr = lr
        self.tok2id = {w: i for i, w in enumerate(vocab)}

        self.tok_emb = nn.Embedding(len(vocab), d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(p_drop)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=p_drop, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(d_model, d_model)

    def _encode_tokens(self, tokens: list) -> torch.Tensor:
        """tokens → 句向量 (d_model,)，当 self.training=True 时 Dropout 生效。"""
        ids = [self.tok2id[w] for w in ([CLS] + tokens)[: self.max_len]]
        T = len(ids)
        ids_t = torch.tensor([ids], dtype=torch.long)
        pos_t = torch.arange(T, dtype=torch.long).unsqueeze(0)
        x = self.emb_drop(self.tok_emb(ids_t) + self.pos_emb(pos_t))  # (1, T, d)
        h = self.encoder(x)                                            # (1, T, d)
        pooled = h[0].mean(dim=0)                                      # (d,)
        return self.proj(pooled)

    def encode_batch(self, batch: list) -> torch.Tensor:
        """batch: list of token lists → (N, d_model)，用于 InfoNCE。"""
        return torch.stack([self._encode_tokens(s) for s in batch])

    def embed(self, tokens: list) -> list:
        """推理接口：关闭 Dropout，输出确定的句向量。"""
        self.eval()
        with torch.no_grad():
            return self._encode_tokens(tokens).tolist()

    def train_model(self, corpus: list, epochs: int = 30,
                    batch_size: int = 6, lr: float = None) -> None:
        lr = lr or self.lr
        optimizer = optim.Adam(self.parameters(), lr=lr)
        data = list(corpus)
        for ep in range(epochs):
            random.shuffle(data)
            tot, nb = 0.0, 0
            for st in range(0, len(data), batch_size):
                bt = data[st: st + batch_size]
                if len(bt) < 2:
                    continue
                self.train()
                # 同一句子两次前向，Dropout 不同 → 正样本对
                H = self.encode_batch(bt)   # (B, d) — 第一次 Dropout
                P = self.encode_batch(bt)   # (B, d) — 第二次 Dropout（掩码不同）

                # InfoNCE: sim_matrix[i, j] = cos(H[i], P[j]) / τ
                H_norm = F.normalize(H, dim=-1)
                P_norm = F.normalize(P, dim=-1)
                sim_matrix = H_norm @ P_norm.T / self.tau  # (B, B)
                labels = torch.arange(len(bt))
                loss = F.cross_entropy(sim_matrix, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot += loss.item()
                nb += 1
            if ep % 5 == 0 or ep == epochs - 1:
                print(f"    epoch {ep:2d}  infonce_loss={tot / max(nb, 1):.4f}")

    def mean_pairwise_sim(self, corpus: list, group: int = 3) -> tuple:
        """
        统计语料内两两句向量的平均余弦：返回 (同主题平均, 跨主题平均)。
        corpus 按每 group 句一个主题排列。同主题平均高、跨主题平均低才说明向量空间可用；
        若两者都接近 1，就是各向异性（表示退化）。
        """
        vs = [torch.tensor(self.embed(s)) for s in corpus]
        intra, inter = [], []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                cos = F.cosine_similarity(vs[i].unsqueeze(0), vs[j].unsqueeze(0)).item()
                (intra if i // group == j // group else inter).append(cos)
        return sum(intra) / len(intra), sum(inter) / len(inter)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    corpus = [["蓝牙", "耳机", "降噪"], ["无线", "耳机", "续航"], ["耳机", "降噪", "续航"],
              ["电竞", "显示器", "高刷"], ["游戏", "显示器", "面板"], ["高刷", "面板", "显示器"],
              ["男士", "纯棉", "短袖"], ["男士", "透气", "T恤"], ["纯棉", "透气", "短袖"],
              ["笔记本", "轻薄", "商务"], ["轻薄", "便携", "办公"], ["笔记本", "商务", "办公"]]
    vocab = [CLS] + sorted({w for s in corpus for w in s})
    model = SimCSE(vocab, d_model=16, n_heads=2, d_ff=32, max_len=8,
                   p_drop=0.15, tau=0.1, lr=0.05)
    print(f"SimCSE 配置：vocab={len(vocab)}  d_model={model.d_model}  "
          f"dropout={model.p_drop}  τ={model.tau}  无标注语料 {len(corpus)} 句")

    # 演示正样本对：同一句话两次带 Dropout 的前向，cos < 1 说明掩码确实不同
    probe = ["蓝牙", "耳机", "降噪"]
    model.train()
    with torch.no_grad():
        h1 = model._encode_tokens(probe)
        h2 = model._encode_tokens(probe)
    cos_before = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
    print(f"\n【正样本对的构造】同一句话两次带 Dropout 的前向："
          f"cos(h, h+) = {cos_before:.4f}（<1 说明两次掩码确实不同）")

    syn = (corpus[0], corpus[1])
    dis = (corpus[0], corpus[9])
    b_syn = F.cosine_similarity(
        torch.tensor(model.embed(syn[0])).unsqueeze(0),
        torch.tensor(model.embed(syn[1])).unsqueeze(0)
    ).item()
    b_dis = F.cosine_similarity(
        torch.tensor(model.embed(dis[0])).unsqueeze(0),
        torch.tensor(model.embed(dis[1])).unsqueeze(0)
    ).item()
    b_in, b_out = model.mean_pairwise_sim(corpus)
    print(f"\n训练前：同主题平均 cos={b_in:.4f}   跨主题平均 cos={b_out:.4f}   "
          f"差距={b_in - b_out:.4f}")
    print(f"        两者都偏高且区分度低 → 典型的各向异性（表示退化）")

    print("\n无监督对比学习（InfoNCE，Batch 内负采样）：")
    model.train_model(corpus, epochs=30, batch_size=6)

    a_syn = F.cosine_similarity(
        torch.tensor(model.embed(syn[0])).unsqueeze(0),
        torch.tensor(model.embed(syn[1])).unsqueeze(0)
    ).item()
    a_dis = F.cosine_similarity(
        torch.tensor(model.embed(dis[0])).unsqueeze(0),
        torch.tensor(model.embed(dis[1])).unsqueeze(0)
    ).item()
    a_in, a_out = model.mean_pairwise_sim(corpus)
    print(f"\n训练后：同主题平均 cos={a_in:.4f}（{a_in - b_in:+.4f}）   "
          f"跨主题平均 cos={a_out:.4f}（{a_out - b_out:+.4f}）   差距={a_in - a_out:.4f}")
    print(f"\n具体例子：")
    print(f"  相似  {' '.join(syn[0])} ↔ {' '.join(syn[1])}：{b_syn:.4f} → {a_syn:.4f}")
    print(f"  不相似 {' '.join(dis[0])} ↔ {' '.join(dis[1])}：{b_dis:.4f} → {a_dis:.4f}")
    print(f"\n结论：同主题与跨主题的相似度差距由 {b_in - b_out:.4f} 扩大到 {a_in - a_out:.4f}；")
    print("      Alignment 让 Dropout 正例对齐（同主题相似度保持高位），")
    print("      Uniformity 把不同句子在超球面上推开（跨主题相似度大幅下降），")
    print("      整个过程不需要任何人工标注，这正是 SimCSE 在向量召回中被广泛使用的原因。")
