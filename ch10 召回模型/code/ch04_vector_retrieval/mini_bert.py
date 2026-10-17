"""
mini-BERT 模型（第10章 §10.3.3 基于预训练语义模型的向量召回）

核心思想：
- BERT 采用 Transformer Encoder 堆叠结构，通过双向自注意力让每个 Token 同时看到左右上下文，
  从而学到比 Word2Vec 更丰富的上下文语义表示。
- 输入表示：Token Embedding + Position Embedding（本文件省略 Segment Embedding）。
- 每个 Encoder Block：多头自注意力 → Add & Norm → FFN → Add & Norm。
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
- 预训练任务 MLM（Masked Language Model）：随机遮盖 15% 的 Token（80% 换成 [MASK]、
  10% 换成随机词、10% 保持原词），再用上下文预测被遮盖位置的原始 Token。
- 规模刻意取极小（vocab≈30，d_model=16，层数 2，头数 4，seq_len=8），保证可快速运行。
  使用 PyTorch autograd 完成端到端训练。

论文出处：Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding, NAACL 2019.（PyTorch autograd 实现）
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PAD, CLS, MASK = "[PAD]", "[CLS]", "[MASK]"


class MiniBERT(nn.Module):
    """
    极简 BERT：Token/Position Embedding + N 层 TransformerEncoder + MLM 预测头。

    MLM 训练目标： L = -1/|M| Σ_{t ∈ M} log softmax(W_mlm h_t)[y_t]
    """

    def __init__(self, vocab: list, d_model: int = 16, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 32, max_len: int = 8, lr: float = 0.05):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.lr = lr
        self.tok2id = {w: i for i, w in enumerate(vocab)}

        self.tok_emb = nn.Embedding(len(vocab), d_model, padding_idx=self.tok2id[PAD])
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # MLM 输出层
        self.mlm_head = nn.Linear(d_model, len(vocab))

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) → (B, T, d_model)"""
        T = ids.size(1)
        pos = torch.arange(T, dtype=torch.long).unsqueeze(0)  # (1, T)
        return self.tok_emb(ids) + self.pos_emb(pos)

    def encode(self, ids: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        ids: (B, T) → 隐状态 (B, T, d_model)
        pad_mask: (B, T) bool，True 表示 PAD 位置（被忽略）
        """
        x = self._embed(ids)                  # (B, T, d_model)
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def forward(self, ids: torch.Tensor, mask_positions: list,
                labels: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        ids: (B, T) 含 [MASK] 的输入
        mask_positions: list of lists，每句被遮盖的位置
        labels: (B, T) 原始 token id，非 mask 位置填 -100（被 cross_entropy 忽略）
        """
        h = self.encode(ids, pad_mask)                      # (B, T, d_model)
        logits = self.mlm_head(h)                           # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), labels.view(B * T), ignore_index=-100)
        return loss

    def _mask_tokens(self, ids: list) -> tuple:
        """随机遮盖 15% 的 Token：80% → [MASK]，10% → 随机词，10% 保持原词。"""
        mask_id = self.tok2id[MASK]
        out = list(ids)
        targets = []
        for t in range(1, len(ids)):          # 位置 0 是 [CLS]
            if random.random() < 0.15:
                targets.append((t, ids[t]))
                r = random.random()
                if r < 0.8:
                    out[t] = mask_id
                elif r < 0.9:
                    out[t] = random.randrange(len(self.vocab))
        if not targets:                       # 至少遮一个
            t = random.randrange(1, len(ids))
            targets.append((t, ids[t]))
            out[t] = mask_id
        return out, targets

    def train_mlm(self, sentences: list, epochs: int = 8) -> None:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        for ep in range(epochs):
            random.shuffle(sentences)
            tot, n = 0.0, 0
            self.train()
            for ids in sentences:
                masked, targets = self._mask_tokens(ids)

                # 构造 label 张量：非 mask 位置填 -100
                label_ids = [-100] * len(masked)
                for t, y in targets:
                    label_ids[t] = y

                ids_t = torch.tensor([masked], dtype=torch.long)         # (1, T)
                label_t = torch.tensor([label_ids], dtype=torch.long)    # (1, T)

                optimizer.zero_grad()
                loss = self.forward(ids_t, [p for p, _ in targets], label_t)
                loss.backward()
                optimizer.step()
                tot += loss.item()
                n += 1
            if ep % 25 == 0 or ep == epochs - 1:
                print(f"    epoch {ep:3d}  mlm_loss={tot / max(n, 1):.4f}")

    def predict_masked(self, tokens: list, top_k: int = 3) -> list:
        """给定含 [MASK] 的句子，预测 [MASK] 位置最可能的原始 Token。"""
        self.eval()
        with torch.no_grad():
            ids = torch.tensor([[self.tok2id[w] for w in tokens]], dtype=torch.long)
            pos = tokens.index(MASK)
            h = self.encode(ids)                             # (1, T, d_model)
            logits = self.mlm_head(h[0, pos])               # (V,)
            probs = F.softmax(logits, dim=-1)
            top = torch.topk(probs, top_k)
            return [(self.vocab[i.item()], top.values[k].item()) for k, i in enumerate(top.indices)]

    def cls_embedding(self, tokens: list) -> list:
        """[CLS] 位置的输出向量，常被当作整句语义表示（用于向量召回）。"""
        self.eval()
        with torch.no_grad():
            ids = torch.tensor([[self.tok2id[w] for w in tokens]], dtype=torch.long)
            h = self.encode(ids)
            return h[0, 0].tolist()


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    words = ["耳机", "蓝牙", "降噪", "续航", "显示器", "高刷", "电竞", "面板",
             "短袖", "纯棉", "透气", "男士", "推荐", "购买", "喜欢", "好评",
             "价格", "便宜", "质量", "不错", "物流", "很快", "商品", "标题",
             "的", "很", "非常"]
    vocab = [PAD, CLS, MASK] + words
    model = MiniBERT(vocab, d_model=16, n_layers=2, n_heads=4, d_ff=32, max_len=8, lr=0.05)
    print(f"mini-BERT 配置：vocab={len(vocab)}  d_model={model.d_model}  "
          f"层数=2  头数=4  seq_len={model.max_len}")

    raw = [
        ["蓝牙", "耳机", "降噪", "很", "不错"], ["耳机", "续航", "非常", "好评"],
        ["电竞", "显示器", "高刷", "面板"], ["显示器", "面板", "质量", "不错"],
        ["男士", "纯棉", "短袖", "透气"], ["短袖", "透气", "非常", "喜欢"],
        ["商品", "价格", "便宜", "好评"], ["物流", "很快", "质量", "不错"],
        ["推荐", "购买", "蓝牙", "耳机"], ["喜欢", "降噪", "耳机", "续航"],
        ["电竞", "面板", "高刷", "好评"], ["纯棉", "男士", "短袖", "喜欢"],
    ]
    sentences = [[model.tok2id[CLS]] + [model.tok2id[w] for w in s] for s in raw]
    print(f"\n合成语料 {len(sentences)} 句，MLM 预训练（随机遮盖 15% Token）：")
    model.train_mlm(sentences, epochs=150)

    print("\n【MLM 完形填空：用双向上下文预测被遮盖的 Token】")
    for probe in [[CLS, "蓝牙", MASK, "降噪", "很", "不错"],
                  [CLS, "男士", "纯棉", MASK, "透气"],
                  [CLS, "电竞", MASK, "高刷", "面板"]]:
        pred = "  ".join(f"{w}({p:.3f})" for w, p in model.predict_masked(probe, 3))
        print(f"  {' '.join(probe[1:])}  →  {pred}")

    cls_vec = model.cls_embedding([CLS, "蓝牙", "耳机", "降噪"])
    print(f"\n[CLS] 句向量（维度 {len(cls_vec)}）前 6 维：{[round(v, 3) for v in cls_vec[:6]]}")
    print("提示：原生 BERT 用 [CLS] 做句向量，两句相似度需要把句子对拼接后联合编码，")
    print("      大规模检索要 O(n) 次前向推理；下一节 Sentence-BERT 正是为解决这个问题。")
