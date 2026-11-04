"""
ANN 的成本直觉：为什么要做近似搜索（第18章 §18.1）

正文给的算例：物品库 N=10^9、向量维度 d=128，一次精确检索约需
    N × d ≈ 10^11 次浮点运算
即使在高性能服务器上也难以满足毫秒级响应。

本脚本做四件事：
1) 复现这个算例，并把它换算成"单核每秒 10 亿次浮点运算"下的耗时与吞吐；
2) 在本机真实测量 NumPy 暴力检索的吞吐，用实测值再算一遍（数字更具体）；
3) 计算存储成本：float32 / int8 / PQ 三种编码下 10^9 个向量各占多少内存；
4) 打印"只扫描 x% 的库、召回率还能保持多少"的权衡表——这是所有 ANN 算法
   共同的 trade-off，后面的 KD 树、PQ、HNSW 只是用不同方式实现这个扫描比例。

[MOCK]：机器算力（10 GFLOPs）、内存带宽、内存价格按典型云主机量级取整。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    knn_exact, make_queries, make_vectors, recall_at_k, sq_dists, table, timing,
    topk_from_sq,
)

N_DB = 10 ** 9          # 物品库规模
DIM = 128               # Embedding 维度
TOPK = 100
GFLOP_PER_SEC = 10.0    # [MOCK] 单核算力，GFLOP/s

# 本机实测用的小库
N_MEASURED = 20_000
N_QUERIES = 200


def main() -> None:
    print("ANN 的成本直觉（§18.1）\n")

    # ---------------- 1) 正文算例 ----------------
    flops = N_DB * DIM
    print("① 精确检索的算力账（正文 N=10^9, d=128）：")
    print(f"  单次查询距离计算量 : N × d = {N_DB:,} × {DIM} = {flops:.1e} FLOPs")
    print(f"  平方距离约为 3 次乘加 ≈ {3 * flops:.1e} FLOPs（|x|²+|y|²-2xᵀy 展开）")
    print(f"  按 {GFLOP_PER_SEC:.0f} GFLOP/s 单核估算单次查询耗时 : "
          f"{flops / (GFLOP_PER_SEC * 1e9):.1f} s")
    print(f"  一个 32 核的检索节点、单次查询仍要 "
          f"{flops / (GFLOP_PER_SEC * 1e9 * 32) * 1000:.0f} ms")
    print("  → 毫秒级 SLA 下只能接受访问 10^4–10^5 个候选，即全库的 0.001%–0.01%。\n")

    # ---------------- 2) 本机实测 ----------------
    X, _ = make_vectors(n=N_MEASURED, dim=DIM, n_clusters=50, spread=0.4)
    Q = make_queries(X, n_queries=N_QUERIES)
    t_brute, d = timing(sq_dists, X, Q, repeat=3)
    qps = N_QUERIES / t_brute
    print("② 本机 NumPy 暴力检索实测（BLAS 单机多线程）：")
    print(f"  库规模 {N_MEASURED:,} × {DIM} 维，{N_QUERIES} 个查询，耗时 {t_brute * 1000:.1f} ms")
    print(f"  吞吐 ≈ {qps:,.0f} QPS，单查询 ≈ {t_brute / N_QUERIES * 1e6:.0f} µs")
    print(f"  按这个吞吐外推到 10^9 条：单查询 ≈ "
          f"{t_brute / N_QUERIES * (N_DB / N_MEASURED) * 1000:.0f} ms")
    print("  注意：实测值包含了 BLAS 的多线程与缓存命中，不代表「真正的单核算力」；")
    print("  外推值只能说明数量级——即便乐观估计，也与毫秒级 SLA 相差 3 个数量级以上。\n")

    # ---------------- 3) 存储 ----------------
    print("③ 存储账（10^9 条向量）：")
    rows = []
    encodings = [
        ("float32", 4 * DIM, "原始精度，Faiss IndexFlat / HNSWFlat"),
        ("float16", 2 * DIM, "半精度，召回略降，HNSW 常用"),
        ("int8 (SQ)", DIM, "标量量化，每维 8 bit"),
        ("PQ m=8,K=256", 8, "8 个子空间 × 8 bit，§18.4 详述"),
        ("PQ m=16,K=256", 16, "16 个子空间 × 8 bit，精度更高"),
        ("二值码 128bit", 16, "1 bit/维，汉明距离检索"),
    ]
    for name, nbytes, note in encodings:
        total_gb = N_DB * nbytes / 1e9
        rows.append([name, f"{nbytes} B/条", f"{total_gb:,.1f} GB", note])
    table(["编码", "单条大小", "总量", "典型用途"], rows)
    print("\n  再算一笔账：10^9 条 float32 正好 512 GB，一台 512 GB 内存的机器放完向量就")
    print("  没有余量给索引结构、操作系统与副本了。所以十亿级要么压缩（PQ/二值码），")
    print("  要么分片（§18.7 的分布式 ANN），二者通常一起用。\n")

    # ---------------- 4) 扫描比例 vs 召回 ----------------
    # 用"随机采样候选 + 精确重排"模拟：只访问 p 比例的库，能保住多少 Recall@100
    truth, _ = knn_exact(X, Q, TOPK)
    rng = np.random.default_rng(0)
    print("④ 扫描比例 vs 召回（随机采样 p 比例的库向量，再精确重排）：")
    rows = []
    for p in (1.0, 0.5, 0.1, 0.01, 0.001):
        m = max(TOPK, int(N_MEASURED * p))
        idx = np.stack([rng.choice(N_MEASURED, m, replace=False) for _ in Q])
        sub = np.asarray([[d[i, j] for j in row] for i, row in enumerate(idx)])
        pred, _ = topk_from_sq(sub, TOPK)
        pred = np.take_along_axis(idx, pred, axis=1)
        r = recall_at_k(pred, truth)
        rows.append([f"{p:.1%}", f"{m:,}", f"{r:.3f}",
                     "暴力" if p == 1.0 else "无结构采样，只是下界"])
    table(["扫描比例", "访问候选数", "Recall@100", "说明"], rows)
    print("\n  这张表说明：随机丢候选的 Recall 是线性掉的，1% 的扫描只剩约 1% 召回。")
    print("  ANN 的全部价值在于——用结构把「该扫的那 1%」找出来，而不是均匀地扫 1%。")
    print("  后面的 KD 树(§18.2)、LSH(§18.3)、PQ(§18.4)、HNSW(§18.5) 都是在做这件事。\n")

    # ---------------- 5) 复杂度对照 ----------------
    print("⑤ 各算法一次查询的复杂度（N=库大小，d=维度）：")
    table(
        ["算法", "查询复杂度", "依赖的结构", "主要误差来源"],
        [
            ["精确检索", "O(N·d)", "无", "无"],
            ["KD 树 / 球树", "低维 O(log N)，高维退化 O(N)", "超矩形 / 超球", "剪枝失效（维度灾难）"],
            ["随机投影树", "O(log N)（内在维度低时）", "随机超平面", "随机划分的方差"],
            ["LSH", "O(N·(1-ε)) 亚线性", "哈希桶", "碰撞概率 / 桶内还需精确算"],
            ["PQ + IVF", "O(K·d + m·N/nprobe)", "码本 + 倒排表", "量化误差"],
            ["HNSW", "O(efSearch · log N)", "多层近邻图", "图不连通 / 局部最优"],
        ],
    )


if __name__ == "__main__":
    main()
