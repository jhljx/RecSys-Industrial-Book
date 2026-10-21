"""对应正文第 9 章 §9.3.4「其他离散 Tokenizer 方法」。

正文把离散 Tokenizer 形式化为: 学习一个 Codebook 矩阵 C in R^{m x d}, 把 Embedding
矩阵 Z in R^{n x d} 中的每个向量映射到最近的码字:
    q: z -> argmin_{k=1..m} D(z, c_k)
并给出了三类范式:
  (1) Level-wise Quantization(层次化): RQ-VAE / RQ-KMeans, 逐层量化残差;
  (2) Group-wise Quantization(分组): Product Quantization(PQ), 把 d 维向量切成 S 段
      x = concat(x_1, ..., x_S), 每段独立量化, C(x) = (C_1(x_1), ..., C_S(x_S));
  (3) Lookup-Free Quantization(无查找表): FSQ 与 LFQ。
      FSQ: 先把表征映射到很少的维度, 再逐维四舍五入到有限标量集合, 隐式构成码本,
           z_hat_i = round(L_i/2 * tanh(z_i)) 这类形式, 隐式码本大小 = prod_i L_i;
      LFQ: 把码本维度直接降到 1, 每维只取 {-1, +1}, q(z)_i = sign(z_i), 隐式码本
           大小 2^d; 并引入熵正则提升码本利用率:
           L_entropy = E_x[H(q(z))] - H(E_x[q(z)])
           其中第一项降低单样本不确定性, 第二项鼓励样本在码本空间均匀分布。

本脚本在**同一份合成 item embedding**上对比 RQ(层次化)、PQ(分组)、FSQ、LFQ
四种 Tokenizer 的重构误差、隐式/显式码本大小、码本利用率与语义 ID 冲突率。
"""

from __future__ import annotations

import math
import random

import numpy as np


def make_items(n: int, d: int, n_cluster: int, rng: np.random.Generator) -> np.ndarray:
    centers = rng.normal(0, 2.0, size=(n_cluster, d))
    idx = rng.integers(0, n_cluster, size=n)
    return centers[idx] + 0.3 * rng.normal(size=(n, d))


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 12):
    mu = x[rng.choice(len(x), size=min(k, len(x)), replace=False)].copy()
    assign = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        d2 = (x ** 2).sum(1)[:, None] - 2 * x @ mu.T + (mu ** 2).sum(1)[None, :]
        assign = d2.argmin(1)
        for j in range(len(mu)):
            m = assign == j
            if m.any():
                mu[j] = x[m].mean(0)
    return mu, assign


# ---------------------------------------------------------------- (1) 层次化 RQ
def rq_tokenize(x: np.ndarray, L: int, K: int, rng: np.random.Generator):
    residual, codes, cbs = x.copy(), [], []
    for _ in range(L):
        mu, a = _kmeans(residual, K, rng)
        cbs.append(mu)
        codes.append(a)
        residual = residual - mu[a]
    codes = np.stack(codes, 1)
    recon = sum(cb[codes[:, i]] for i, cb in enumerate(cbs))
    used = sum(int((np.bincount(codes[:, i], minlength=K) > 0).sum()) for i in range(L))
    return codes, recon, L * K, used, K ** L


# ---------------------------------------------------------------- (2) 分组 PQ
def pq_tokenize(x: np.ndarray, S: int, K: int, rng: np.random.Generator):
    """把 d 维切成 S 段, 每段独立 k-means 量化。"""
    d = x.shape[1]
    assert d % S == 0
    seg = d // S
    codes, recon = [], np.zeros_like(x)
    used = 0
    for s in range(S):
        sub = x[:, s * seg:(s + 1) * seg]
        mu, a = _kmeans(sub, K, rng)
        codes.append(a)
        recon[:, s * seg:(s + 1) * seg] = mu[a]
        used += int((np.bincount(a, minlength=K) > 0).sum())
    return np.stack(codes, 1), recon, S * K, used, K ** S


# ---------------------------------------------------------------- (3) FSQ
def fsq_tokenize(x: np.ndarray, levels: list[int], rng: np.random.Generator):
    """FSQ: 先线性降维到 len(levels) 维, 再逐维 bounded + round。

    每维 i 取 L_i 个标量取值: z_hat_i = round((L_i-1)/2 * tanh(z_i)),
    隐式码本大小 = prod_i L_i, 无需查找表。
    """
    d, m = x.shape[1], len(levels)
    W = rng.normal(0, 1.0 / math.sqrt(d), size=(d, m))   # 降维映射 f
    Wi = np.linalg.pinv(W)                               # 解码(最小二乘逆)
    z_raw = x @ W
    scale = z_raw.std(0, keepdims=True) + 1e-8           # 标准化使 tanh 落在有效区
    z = z_raw / scale
    half = np.array([(l - 1) / 2 for l in levels])
    q = np.round(half * np.tanh(z))                      # 逐维四舍五入到有限标量集合
    q = np.clip(q, -half, half)
    # 解码: 由离散标量反推 tanh 值 -> z -> 原空间
    z_hat = np.arctanh(np.clip(q / half, -0.999, 0.999)) * scale
    recon = z_hat @ Wi
    idx = np.rint(q + half).astype(np.int64)             # 每维编号 0..L_i-1
    codes = np.zeros(len(x), dtype=np.int64)
    radix = 1
    for i, l in enumerate(levels):
        codes = codes + idx[:, i] * radix
        radix *= l
    implicit = int(np.prod(levels))
    used = len(np.unique(codes))
    return idx, recon, 0, used, implicit


# ---------------------------------------------------------------- (4) LFQ
def lfq_tokenize(x: np.ndarray, m: int, rng: np.random.Generator,
                 tau: float = 1.0):
    """LFQ: 码本维度降为 1, 每维取值 {-1,+1}, q(z)_i = sign(z_i), 隐式码本 2^m。

    同时按正文公式计算熵正则项:
        L_entropy = E_x[H(q(z))] - H(E_x[q(z)])
        H(q(z)) = sum_i H(q_hat(c_i | z_i)), q_hat 为 softmax(-tau*|c_i - z_i|) 的伯努利分布
    """
    d = x.shape[1]
    W = rng.normal(0, 1.0 / math.sqrt(d), size=(d, m))
    Wi = np.linalg.pinv(W)
    z_raw = x @ W
    scale = z_raw.std(0, keepdims=True) + 1e-8
    z = z_raw / scale
    q = np.where(z >= 0, 1.0, -1.0)
    recon = (q * scale) @ Wi
    # 伯努利概率: p_i = exp(-tau*|1 - z_i|) / (exp(-tau*|1-z_i|) + exp(-tau*|-1-z_i|))
    a = np.exp(-tau * np.abs(1.0 - z))
    b = np.exp(-tau * np.abs(-1.0 - z))
    p = a / (a + b)

    def H(pp):
        pp = np.clip(pp, 1e-9, 1 - 1e-9)
        return -(pp * np.log(pp) + (1 - pp) * np.log1p(-pp))

    per_sample_H = H(p).sum(1).mean()          # E_x[H(q(z))]
    marginal_H = H(p.mean(0)).sum()            # H(E_x[q(z)])
    l_entropy = per_sample_H - marginal_H
    bits = ((q > 0).astype(np.int64))
    codes = np.zeros(len(x), dtype=np.int64)
    for i in range(m):
        codes = codes | (bits[:, i] << i)
    return bits, recon, 0, len(np.unique(codes)), 2 ** m, l_entropy, per_sample_H, marginal_H


def rel_mse(x: np.ndarray, recon: np.ndarray) -> float:
    return float(((x - recon) ** 2).mean() / x.var())


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    N, D = 2000, 32
    x = make_items(N, D, n_cluster=50, rng=rng)

    rq_codes, rq_rec, rq_tbl, rq_used, rq_space = rq_tokenize(
        x, L=3, K=32, rng=np.random.default_rng(42))
    pq_codes, pq_rec, pq_tbl, pq_used, pq_space = pq_tokenize(
        x, S=4, K=32, rng=np.random.default_rng(42))
    fsq_idx, fsq_rec, _, fsq_used, fsq_space = fsq_tokenize(
        x, levels=[8, 8, 5, 5], rng=np.random.default_rng(42))
    (lfq_bits, lfq_rec, _, lfq_used, lfq_space,
     l_ent, h_sample, h_marg) = lfq_tokenize(x, m=12, rng=np.random.default_rng(42))

    print("=" * 94)
    print(f"同一合成 item embedding 上的离散 Tokenizer 对比 (N={N}, dim={D})")
    print("-" * 94)
    print(f"{'方法':<26}{'范式':<18}{'显式码字数':>11}{'隐式ID空间':>13}"
          f"{'实际用到ID':>12}{'相对MSE':>10}")
    rows = [
        ("RQ (L=3,K=32)", "Level-wise", rq_tbl, rq_space, len({tuple(r) for r in rq_codes}),
         rel_mse(x, rq_rec)),
        ("PQ (S=4,K=32)", "Group-wise", pq_tbl, pq_space, len({tuple(r) for r in pq_codes}),
         rel_mse(x, pq_rec)),
        ("FSQ (levels=8,8,5,5)", "Lookup-Free", 0, fsq_space, fsq_used,
         rel_mse(x, fsq_rec)),
        ("LFQ (m=12 bits)", "Lookup-Free", 0, lfq_space, lfq_used,
         rel_mse(x, lfq_rec)),
    ]
    for name, para, tbl, space, used, mse in rows:
        tbl_s = "-(无查找表)" if tbl == 0 else f"{tbl}"
        print(f"{name:<26}{para:<18}{tbl_s:>11}{space:>13,}{used:>12}{mse:>10.3f}")

    print()
    print("表 2  码本/ID 空间利用率")
    print("-" * 94)
    print(f"{'方法':<26}{'实际ID数':>11}{'ID空间':>13}{'空间利用率':>13}{'样本覆盖度':>13}")
    for name, _, _, space, used, _ in rows:
        print(f"{name:<26}{used:>11}{space:>13,}{used / space:>13.2%}{used / N:>13.1%}")

    print()
    print("表 3  LFQ 熵正则项各分量 (L_entropy = E_x[H(q(z))] - H(E_x[q(z)]))")
    print("-" * 94)
    print(f"{'分量':<40}{'数值(nats)':>14}{'说明':<30}")
    print(f"{'E_x[H(q(z))] 单样本平均熵':<40}{h_sample:>14.4f}  越小越确定")
    print(f"{'H(E_x[q(z)]) 边缘分布熵':<40}{h_marg:>14.4f}  越大越均匀(上限 m*ln2="
          f"{12 * math.log(2):.3f})")
    print(f"{'L_entropy':<40}{l_ent:>14.4f}  训练中最小化该项")

    print()
    print("表 4  各方法逐层/逐组的 ID 分布(前 3 个物品的离散编码)")
    print("-" * 94)
    for i in range(3):
        print(f"  item_{i}: RQ={tuple(rq_codes[i])}  PQ={tuple(pq_codes[i])}  "
              f"FSQ={tuple(int(v) for v in fsq_idx[i])}  "
              f"LFQ={''.join(str(int(b)) for b in lfq_bits[i])}")
    print("-" * 94)
    print("注: 相对 MSE = MSE / Var(x)。RQ/PQ 依赖显式码本查找, 重构最准;")
    print("    FSQ/LFQ 无需查找表、天然不会出现'整块码字死亡', 但受降维瓶颈限制,")
    print("    重构误差更高——与正文'无查找表量化以表达能力换工程简洁'的结论一致。")


if __name__ == "__main__":
    main()
