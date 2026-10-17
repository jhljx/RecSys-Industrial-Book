"""
FastText 模型（第10章 §10.3.3 基于预训练语义模型的向量召回）

核心思想：
- Word2Vec 为每个词学习一个独立 Embedding，无法处理未登录词（OOV），也无法共享词形信息。
- FastText 把词拆成字符 n-gram（subword），为每个 n-gram 学习 Embedding，并用哈希技巧把
  海量 n-gram 映射到固定数量的桶中以控制内存：
      e_w = ( z_<w> + Σ_{g ∈ G(w)} z_g ) / (1 + |G(w)|)
  其中 G(w) 为词 w 的字符 n-gram 集合（两端补 '<' '>' 边界符）。
- 打分函数：s(w, c) = e_w^T v_c，用 Skip-Gram / CBOW + 负采样训练：
      L = -log σ(s(w, c)) - Σ_{n ∈ Neg} log σ(-s(w, n))
- 关键优势：即使词从未在训练语料出现过，只要能切出字符 n-gram 就能组合出合理向量，
  因此在推荐系统中能缓解新词、长尾商品标题的冷启动问题。

论文出处：Bojanowski et al., Enriching Word Vectors with Subword Information, TACL 2017.
（PyTorch autograd 实现）
"""

import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _char_ngrams(word: str, n_min: int = 3, n_max: int = 4) -> list:
    """
    抽取字符 n-gram：两端补边界符 '<' '>'，再滑窗切分。
    例如 where → <wh, whe, her, ere, re>, <whe, wher, here, ere> ...
    """
    marked = "<" + word + ">"
    grams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(marked) - n + 1):
            grams.append(marked[i: i + n])
    return grams


def _hash_bucket(gram: str, n_buckets: int) -> int:
    """FNV-1a 哈希，把任意 n-gram 字符串映射到固定数量的桶（控制内存开销）。"""
    h = 2166136261
    for ch in gram:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h % n_buckets


class FastText(nn.Module):
    """
    FastText 词向量模型（Skip-Gram / CBOW + 负采样）。

    参数：
    - word_in  ：词本身的 Embedding（"<word>"整词作为一个特殊 subword）
    - bucket   ：字符 n-gram 哈希桶的 Embedding，不同词之间通过公共 n-gram 实现参数共享
    - ctx      ：上下文词（输出层）的 Embedding
    """

    def __init__(self, dim: int = 16, n_buckets: int = 200, window: int = 2,
                 n_neg: int = 5, mode: str = "skipgram"):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.window = window
        self.n_neg = n_neg
        self.mode = mode

        self.vocab: dict = {}
        self.idx2word: list = []
        self.neg_table: list = []
        self._gram_cache: dict = {}

        # 字符 n-gram 哈希桶 Embedding
        self.bucket = nn.Embedding(n_buckets, dim)
        nn.init.uniform_(self.bucket.weight, -0.5 / dim, 0.5 / dim)

        # 词本身的 Embedding（word_in）和上下文 Embedding（ctx）在 build_vocab 后初始化
        self.word_in: nn.Embedding = None
        self.ctx: nn.Embedding = None

    def build_vocab(self, corpus: list) -> None:
        """构建词表，并按 unigram^0.75 分布构造负采样表。"""
        freq: dict = defaultdict(int)
        for sent in corpus:
            for w in sent:
                freq[w] += 1
        for w in sorted(freq):
            self.vocab[w] = len(self.idx2word)
            self.idx2word.append(w)
        V = len(self.idx2word)
        self.word_in = nn.Embedding(V, self.dim)
        self.ctx = nn.Embedding(V, self.dim)
        nn.init.uniform_(self.word_in.weight, -0.5 / self.dim, 0.5 / self.dim)
        nn.init.zeros_(self.ctx.weight)
        for w, c in freq.items():
            self.neg_table.extend([self.vocab[w]] * max(1, int(c ** 0.75 * 4)))

    def _buckets_of(self, word: str) -> list:
        if word not in self._gram_cache:
            self._gram_cache[word] = [_hash_bucket(g, self.n_buckets) for g in _char_ngrams(word)]
        return self._gram_cache[word]

    def get_word_vector(self, word: str) -> torch.Tensor:
        """
        词向量 = 词本身 Embedding + 所有字符 n-gram Embedding，再求平均。
        OOV 词没有 word_in，只靠 n-gram 组合出向量 —— 这正是 FastText 的关键优势。
        """
        with torch.no_grad():
            parts = self._buckets_of(word)
            bucket_ids = torch.tensor(parts, dtype=torch.long)
            vec = self.bucket(bucket_ids).sum(dim=0)  # sum pooling
            cnt = len(parts)
            if word in self.vocab:
                wid = torch.tensor([self.vocab[word]], dtype=torch.long)
                vec = vec + self.word_in(wid).squeeze(0)
                cnt += 1
            return vec / max(cnt, 1)

    def _get_word_vector_train(self, word: str) -> torch.Tensor:
        """训练时调用（保留梯度图）。"""
        parts = self._buckets_of(word)
        bucket_ids = torch.tensor(parts, dtype=torch.long)
        vec = self.bucket(bucket_ids).sum(dim=0)
        cnt = len(parts)
        if word in self.vocab:
            wid = torch.tensor([self.vocab[word]], dtype=torch.long)
            vec = vec + self.word_in(wid).squeeze(0)
            cnt += 1
        return vec / max(cnt, 1)

    def _update_pair(self, optimizer: optim.Optimizer, center: str, target_idx: int) -> float:
        """一对 (中心词, 上下文词) 的负采样更新，返回 loss。"""
        samples = [(target_idx, 1.0)]
        while len(samples) < self.n_neg + 1:
            neg = random.choice(self.neg_table)
            if neg != target_idx:
                samples.append((neg, 0.0))

        optimizer.zero_grad()
        e_w = self._get_word_vector_train(center)
        loss = torch.tensor(0.0)
        for idx, label in samples:
            ctx_id = torch.tensor([idx], dtype=torch.long)
            v_c = self.ctx(ctx_id).squeeze(0)
            score = torch.dot(e_w, v_c)
            lbl = torch.tensor(label)
            loss = loss + F.binary_cross_entropy_with_logits(score.unsqueeze(0), lbl.unsqueeze(0))
        loss = loss / len(samples)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, corpus: list, epochs: int = 8, lr: float = 0.05) -> None:
        """Skip-Gram：中心词预测上下文；CBOW：上下文（逐个）预测中心词。"""
        if not self.vocab:
            self.build_vocab(corpus)
        optimizer = optim.SGD(self.parameters(), lr=lr)
        data = list(corpus)
        for ep in range(epochs):
            random.shuffle(data)
            tot, n = 0.0, 0
            for sent in data:
                for i, w in enumerate(sent):
                    lo, hi = max(0, i - self.window), min(len(sent), i + self.window + 1)
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        if self.mode == "skipgram":
                            tot += self._update_pair(optimizer, w, self.vocab[sent[j]])
                        else:
                            tot += self._update_pair(optimizer, sent[j], self.vocab[w])
                        n += 1
            if ep % 2 == 0 or ep == epochs - 1:
                print(f"    epoch {ep:2d}  avg_loss={tot / max(n, 1):.4f}")

    def most_similar(self, word: str, top_k: int = 3) -> list:
        """
        在词表中检索与给定词（可以是 OOV）语义最接近的词。

        注意：Word2Vec/FastText 这类静态词向量存在各向异性（Anisotropy），所有向量挤在
        向量空间的少数方向上，裸余弦相似度普遍接近 1 而缺乏区分度。这里先减去词表均值
        向量（centering）再算余弦 —— 这一表示退化问题正是后文 SimCSE 要解决的核心问题。
        """
        with torch.no_grad():
            vecs = {w: self.get_word_vector(w) for w in self.idx2word}
            mu = torch.stack(list(vecs.values())).mean(dim=0)
            q = self.get_word_vector(word) - mu
            sims = []
            for w in self.idx2word:
                if w == word:
                    continue
                v = vecs[w] - mu
                cos = F.cosine_similarity(q.unsqueeze(0), v.unsqueeze(0)).item()
                sims.append((w, cos))
            return sorted(sims, key=lambda x: -x[1])[:top_k]

    def word2vec_style_vector(self, word: str):
        """Word2Vec 基线：只查词表，OOV 直接返回 None（无法给出任何向量）。"""
        if word not in self.vocab:
            return None
        with torch.no_grad():
            wid = torch.tensor([self.vocab[word]], dtype=torch.long)
            return self.word_in(wid).squeeze(0)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    # 合成语料：刻意构造「同词根不同词形」的词族，便于观察 subword 参数共享
    corpus = [s.split() for s in [
        "the player likes playing football",
        "he played football daily",
        "the players play football well",
        "she loves running fast",
        "the runner runs daily",
        "he is running fast daily",
        "the reader likes reading novel",
        "she reads novel daily",
        "he read novel well",
        "the athlete likes jumping well",
        "he jumped fast daily",
        "she jumps fast well",
        "the player reads novel",
        "the runner likes football",
        "she is reading novel fast",
        "he plays football well",
    ]]
    model = FastText(dim=16, n_buckets=200, window=2, n_neg=5, mode="skipgram")
    model.build_vocab(corpus)
    print(f"语料：{len(corpus)} 句，词表大小 {len(model.vocab)}，n-gram 哈希桶 {model.n_buckets}")
    print(f"示例 n-gram：playing → {_char_ngrams('playing')[:6]} ...")
    print("\nSkip-Gram + 负采样训练：")
    model.train(corpus, epochs=20, lr=0.08)

    print("\n【登录词相似检索】")
    for w in ["played", "reads", "runner"]:
        res = "  ".join(f"{x}({s:.3f})" for x, s in model.most_similar(w, 3))
        print(f"  {w:<9} → {res}")

    print("\n【OOV 未登录词：FastText 靠字符 n-gram 仍能给出合理向量】")
    for oov in ["playful", "jumper", "runners"]:
        assert oov not in model.vocab, oov
        w2v = model.word2vec_style_vector(oov)
        res = "  ".join(f"{x}({s:.3f})" for x, s in model.most_similar(oov, 3))
        print(f"  {oov:<9} Word2Vec 向量={'None（无法表示）' if w2v is None else '有'}"
              f" | FastText 最近邻 → {res}")
    print("\n结论：OOV 词在 Word2Vec 中完全没有向量，而 FastText 通过共享的字符 n-gram")
    print("      （如 <pl/pla/lay、jum/ump、run/unn/ner）组合出向量，并检索到同词根的词。")
