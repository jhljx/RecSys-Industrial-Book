"""对应正文第 9 章 §9.3.3「RQ-KMeans 算法」(快手 OneRec 使用的语义 ID 方案)。

正文给出的算法流程(逐层残差聚类):
    c_1 = argmin_j ||x_0 - mu_j||^2
    r_1 = x_0 - mu_{c_1}
    c_2 = argmin_j ||r_1 - mu_j^{(2)}||^2
    ...  最终得到语义 ID (c_1, c_2, ..., c_L), 且 x ~= sum_{l=1}^{L} mu_{c_l}^{(l)}
与 RQ-VAE 相比不需要训练 Encoder/Decoder, 也不存在 STE 带来的梯度近似问题。

正文还提到改进版 RQ-KMeans++: 继承 KMeans++ 思想, 每层残差聚类时新的聚类中心
不再随机初始化, 而是按样本到已有中心距离平方成比例的概率采样:
    P(选中 x) ∝ min_{mu in 已选} ||x - mu||^2
使码本更均匀覆盖向量空间, 降低陷入局部最优的风险。

本脚本用 numpy 实现 RQ-KMeans 与 RQ-KMeans++, 在同一合成 embedding 上对比
逐层残差能量、码本使用率、语义 ID 冲突率。
"""

from __future__ import annotations

import random

import numpy as np


def make_items(n: int, d: int, n_cluster: int, rng: np.random.Generator) -> np.ndarray:
    centers = rng.normal(0, 2.0, size=(n_cluster, d))
    idx = rng.integers(0, n_cluster, size=n)
    return centers[idx] + 0.3 * rng.normal(size=(n, d))


def _assign(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    d2 = (x ** 2).sum(1)[:, None] - 2 * x @ mu.T + (mu ** 2).sum(1)[None, :]
    return d2.argmin(1)


def init_random(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """KMeans 的随机初始化: 直接随机抽 k 个样本。"""
    return x[rng.choice(len(x), size=k, replace=False)].copy()


def init_pp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """KMeans++ 初始化: 按到已有中心的距离平方成比例采样。"""
    centers = [x[rng.integers(len(x))]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d2 / max(d2.sum(), 1e-12)
        centers.append(x[rng.choice(len(x), p=p)])
        nd = ((x - centers[-1]) ** 2).sum(1)
        d2 = np.minimum(d2, nd)
    return np.stack(centers)


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, *,
           plus_plus: bool, iters: int = 15) -> tuple[np.ndarray, np.ndarray]:
    mu = init_pp(x, k, rng) if plus_plus else init_random(x, k, rng)
    assign = _assign(x, mu)
    for _ in range(iters):
        for j in range(k):
            m = assign == j
            if m.any():
                mu[j] = x[m].mean(0)
        new = _assign(x, mu)
        if (new == assign).all():
            assign = new
            break
        assign = new
    return mu, assign


def rq_kmeans(x: np.ndarray, n_layer: int, k: int, rng: np.random.Generator,
              plus_plus: bool) -> dict:
    residual = x.copy()
    codebooks, codes, layer_energy = [], [], []
    base = (x ** 2).sum(1).mean()
    for _ in range(n_layer):
        mu, assign = kmeans(residual, k, rng, plus_plus=plus_plus)
        codebooks.append(mu)
        codes.append(assign)
        residual = residual - mu[assign]
        layer_energy.append(float((residual ** 2).sum(1).mean()))
    codes = np.stack(codes, 1)
    recon = sum(cb[codes[:, l]] for l, cb in enumerate(codebooks))
    usage = [float((np.bincount(codes[:, l], minlength=k) > 0).mean())
             for l in range(n_layer)]
    ppl = []
    for l in range(n_layer):
        p = np.bincount(codes[:, l], minlength=k) / len(x)
        nz = p[p > 0]
        ppl.append(float(np.exp(-(nz * np.log(nz)).sum())))
    uniq = len({tuple(r) for r in codes})
    return {"base": float(base), "layer_energy": layer_energy,
            "mse": float(((x - recon) ** 2).mean()),
            "usage": usage, "ppl": ppl, "uniq": uniq,
            "codes": codes}


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    N, D, L, K = 3000, 32, 3, 64
    rng = np.random.default_rng(42)
    x = make_items(N, D, n_cluster=50, rng=rng)

    res_rand = rq_kmeans(x, L, K, np.random.default_rng(42), plus_plus=False)
    res_pp = rq_kmeans(x, L, K, np.random.default_rng(42), plus_plus=True)

    print("=" * 80)
    print(f"RQ-KMeans / RQ-KMeans++ (N={N}, dim={D}, L={L} 层, K={K}/层, "
          f"ID 空间 K^L={K ** L:,})")
    print("-" * 80)
    print("表 1  逐层残差能量 E||r_l||^2 (x ~= sum_l mu_{c_l}^{(l)})")
    print(f"{'层 l':>6}{'RQ-KMeans':>16}{'占初始':>10}{'RQ-KMeans++':>16}{'占初始':>10}")
    print(f"{0:>6}{res_rand['base']:>16.4f}{1.0:>10.1%}"
          f"{res_pp['base']:>16.4f}{1.0:>10.1%}")
    for l in range(L):
        a, b = res_rand["layer_energy"][l], res_pp["layer_energy"][l]
        print(f"{l + 1:>6}{a:>16.4f}{a / res_rand['base']:>10.1%}"
              f"{b:>16.4f}{b / res_pp['base']:>10.1%}")

    print()
    print("表 2  每层码本使用率与 Perplexity (上限 K=%d)" % K)
    print("-" * 80)
    print(f"{'层 l':>6}{'随机初始化使用率':>18}{'PPL':>9}{'KMeans++使用率':>18}{'PPL':>9}")
    for l in range(L):
        print(f"{l + 1:>6}{res_rand['usage'][l]:>17.1%}{res_rand['ppl'][l]:>9.1f}"
              f"{res_pp['usage'][l]:>17.1%}{res_pp['ppl'][l]:>9.1f}")

    print()
    print("表 3  整体对比")
    print("-" * 80)
    print(f"{'指标':<28}{'RQ-KMeans(随机)':>20}{'RQ-KMeans++':>18}")
    print(f"{'重构 MSE':<28}{res_rand['mse']:>20.5f}{res_pp['mse']:>18.5f}")
    print(f"{'相对残差(末层/初始)':<28}"
          f"{res_rand['layer_energy'][-1] / res_rand['base']:>19.2%}"
          f"{res_pp['layer_energy'][-1] / res_pp['base']:>17.2%}")
    print(f"{'唯一语义 ID 数':<28}{res_rand['uniq']:>20}{res_pp['uniq']:>18}")
    print(f"{'语义 ID 冲突率':<28}{1 - res_rand['uniq'] / N:>19.2%}"
          f"{1 - res_pp['uniq'] / N:>17.2%}")
    print(f"{'平均 Perplexity':<28}"
          f"{sum(res_rand['ppl']) / L:>20.2f}{sum(res_pp['ppl']) / L:>18.2f}")
    print("-" * 80)
    print("注: RQ-KMeans 无需神经网络与 STE, 第一层即吸收大部分能量, 后续层做细化;")
    print("    KMeans++ 初始化让码本更均匀覆盖残差空间(Perplexity 更高)。")

    print()
    print("表 4  前 5 个物品的语义 ID(RQ-KMeans++)")
    print("-" * 80)
    for i in range(5):
        print(f"  item_{i:<4} SID = ({', '.join(str(c) for c in res_pp['codes'][i])})")


if __name__ == "__main__":
    main()
