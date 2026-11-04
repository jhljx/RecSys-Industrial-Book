"""
多模态特征的编码与融合（第15章 §15.1.5）

演示正文 §15.1.5 的流程：
    e_text = f_text(x_text), e_vision = f_vision(x_vision), e_audio = f_audio(x_audio)
    e_i = f(e_text, e_vision, e_audio)

其中文本/视觉/音频编码器用 [MOCK] 线性投影代替（真实系统是 BERT/ViT/AST 等）。
重点比较三种融合方式：Concat、加权求和、Attention，并说明多模态特征在冷启动
场景下如何提供 ID 特征无法提供的泛化能力。

运行：python3 ch01_feature_system/multimodal_fusion.py
"""

from __future__ import annotations

import math
import random

import numpy as np

D_MODAL = 32
D_ITEM = 32


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


class MockEncoder:
    """[MOCK] 模态编码器：固定随机投影 + ReLU，模拟预训练模型输出的 Dense 向量。"""

    def __init__(self, in_dim: int, out_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 1 / math.sqrt(in_dim), size=(in_dim, out_dim))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.w, 0)
        return _normalize(h)


def build_modalities(rng: random.Random, n: int) -> dict[str, np.ndarray]:
    """为 n 个 item 生成三个模态的原始特征（模拟文本/图像/音频输入）。"""
    return {
        "text": np.array([[rng.gauss(0, 1) for _ in range(24)] for _ in range(n)]),
        "vision": np.array([[rng.gauss(0, 1) for _ in range(64)] for _ in range(n)]),
        "audio": np.array([[rng.gauss(0, 1) for _ in range(16)] for _ in range(n)]),
    }


def concat_fusion(e_text, e_vision, e_audio, proj: np.ndarray) -> np.ndarray:
    """拼接后用线性层压回 d 维：e = [e_t; e_v; e_a] W。"""
    cat = np.concatenate([e_text, e_vision, e_audio], axis=-1)
    return _normalize(cat @ proj)


def weighted_fusion(e_text, e_vision, e_audio, w) -> np.ndarray:
    """固定权重加权：e = w_t e_t + w_v e_v + w_a e_a。"""
    fused = w[0] * e_text + w[1] * e_vision + w[2] * e_audio
    return _normalize(fused)


def attention_fusion(e_text, e_vision, e_audio, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """样本级 Attention：用用户兴趣 query 动态决定每个模态的权重。"""
    stack = np.stack([e_text, e_vision, e_audio], axis=0)  # [3, d]
    scores = stack @ query / math.sqrt(stack.shape[-1])
    scores = scores - scores.max()
    a = np.exp(scores)
    a = a / a.sum()
    return _normalize(a @ stack), a


def main() -> None:
    rng = random.Random(42)
    n_items = 6
    raw = build_modalities(rng, n_items)
    enc_t = MockEncoder(24, D_MODAL, 1)
    enc_v = MockEncoder(64, D_MODAL, 2)
    enc_a = MockEncoder(16, D_MODAL, 3)

    e_t = np.stack([enc_t(raw["text"][i]) for i in range(n_items)])
    e_v = np.stack([enc_v(raw["vision"][i]) for i in range(n_items)])
    e_a = np.stack([enc_a(raw["audio"][i]) for i in range(n_items)])

    rng_np = np.random.default_rng(7)
    proj = rng_np.normal(0, 1 / math.sqrt(3 * D_MODAL), size=(3 * D_MODAL, D_ITEM))
    w = np.array([0.4, 0.4, 0.2])

    # 用户兴趣 query：假设该用户最近在消费与 item0 同类的（体育）内容，
    # 因此 query 取 item0 文本向量加少量噪声，模拟"用户偏好某类文本语义"。
    user_query = _normalize(e_t[0] + rng_np.normal(0, 0.3, size=D_MODAL))

    print("各模态 Dense 向量（前 4 维）与模长：")
    for name, e in (("text", e_t), ("vision", e_v), ("audio", e_a)):
        print(f"  {name:7s} item0={np.round(e[0][:4], 3)}  |e|={np.linalg.norm(e[0]):.4f}")

    print("\n三种融合方式的 item-item 余弦相似度矩阵（上三角）：")
    fusions = {
        "concat": np.stack([concat_fusion(e_t[i], e_v[i], e_a[i], proj) for i in range(n_items)]),
        "weighted": np.stack([weighted_fusion(e_t[i], e_v[i], e_a[i], w) for i in range(n_items)]),
    }
    attn_vecs, attn_weights = [], []
    for i in range(n_items):
        v, a = attention_fusion(e_t[i], e_v[i], e_a[i], user_query)
        attn_vecs.append(v)
        attn_weights.append(a)
    fusions["attention"] = np.stack(attn_vecs)

    for name, vecs in fusions.items():
        sim = vecs @ vecs.T
        vals = [f"{sim[i, j]:+.3f}" for i in range(n_items) for j in range(i + 1, n_items)]
        print(f"  {name:10s} " + " ".join(vals[:6]) + " ...")

    print("\nAttention 融合的模态权重（同一用户 query，不同 item）：")
    for i in range(n_items):
        a = attn_weights[i]
        print(f"  item{i}: text={a[0]:.3f} vision={a[1]:.3f} audio={a[2]:.3f}")

    # ---- 冷启动：新 item 没有 ID，但有内容 ----
    print("\n冷启动对比（新 item 无历史行为）：")
    new_raw = {k: v[:1] + rng_np.normal(size=(1, v.shape[1])) for k, v in raw.items()}
    new_t = enc_t(new_raw["text"][0])
    new_v = enc_v(new_raw["vision"][0])
    new_a = enc_a(new_raw["audio"][0])
    for name, fn in (
        ("concat", lambda: concat_fusion(new_t, new_v, new_a, proj)),
        ("weighted", lambda: weighted_fusion(new_t, new_v, new_a, w)),
        ("attention", lambda: attention_fusion(new_t, new_v, new_a, user_query)[0]),
    ):
        e = fn()
        sims = fusions[name] @ e
        print(f"  {name:10s} 与库内 item 的最大相似度 = {sims.max():.4f}，"
              f"可据此召回内容相近的老 item（ID 特征此时完全缺失）")

    print("\n结论：ID 特征只能记忆，多模态特征才能泛化到新物品；"
          "融合方式决定'看内容'还是'看当前用户的兴趣'（§15.1.5）。")


if __name__ == "__main__":
    main()
