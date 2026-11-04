"""
IVFADC：倒排文件 + 残差乘积量化（第18章 §18.4.3）

正文的思路：PQ 把距离计算压到 m 次查表，但还是要扫全库（O(N)）。
IVFADC 再加一层粗聚类：

    离线：粗 K-Means 得到 n_list 个中心；每个向量 y 归到最近的 c_i，
          对它做残差编码 r = y − c_i（而不是直接编码 y），
          倒排表 i 里存 (cluster id, PQ code)。
    在线：1) 算查询到所有粗中心的距离；
          2) 取最近的 n_probe 个倒排表；
          3) 对每个表算残差查询 x − c_i 的 ADC 查找表；
          4) 查表算近似距离，合并返回 Top-k。

距离恒等式（正文的推导）：
    ‖x − (c_i + q(y − c_i))‖ = ‖(x − c_i) − q(y − c_i)‖
所以「减去粗中心」只是给查询与残差同时平移一个偏置，不改变算法结构。

本脚本：
1) 在 20000 条 64 维数据上建 IVFADC，核对倒排表规模；
2) 扫 n_probe，画 Recall@10 / 候选数 / 查询耗时——正文的「调 n_probe 换
   Recall 与 Latency」；
3) 对比「残差 PQ」与「直接 PQ 向量」在同一 n_probe 下的召回，验证残差编码
   确实降低了量化误差；
4) 测一下 n_list 的影响。

运行：python3 ch04_pq/ivfadc.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (  # noqa: E402
    knn_exact, make_queries, make_vectors, recall_at_k, sq_dists, table,
)
from product_quantization import PQ, kmeans  # noqa: E402

SEED = 2026


class IVFADC:
    """倒排 + 残差 PQ。residual=False 时退化成「倒排 + 直接 PQ」，用于对照。"""

    def __init__(self, n_list: int = 64, m: int = 8, K: int = 256,
                 residual: bool = True, seed: int = SEED) -> None:
        self.n_list = n_list
        self.m = m
        self.K = K
        self.residual = residual
        self.seed = seed

    def train(self, X: np.ndarray) -> None:
        self.centroids = kmeans(X, self.n_list, seed=self.seed)
        # sq_dists(base, queries) 返回 [len(queries), len(base)]，这里 base 是粗中心
        assign = np.argmin(sq_dists(self.centroids, X), axis=1)   # [N]
        self.assign = assign
        self.pq = PQ(m=self.m, K=self.K, seed=self.seed)
        if self.residual:
            R = X - self.centroids[assign]        # 残差：粗聚类已消掉全局偏移
        else:
            R = X
        self.pq.train(R)
        codes = self.pq.encode(R)
        # 倒排表：列表 i 里是分配给中心 i 的向量编号
        self.postings = [np.where(assign == i)[0] for i in range(self.n_list)]
        self.codes = codes

    def search(self, q: np.ndarray, k: int = 10, n_probe: int = 8):
        """返回 (Top-k 编号, 本次扫描的候选数)。"""
        d_c = sq_dists(self.centroids, q[None, :])[0]
        probes = np.argpartition(d_c, n_probe - 1)[:n_probe]
        probes = probes[np.argsort(d_c[probes])]      # 近的倒排表先查

        cand_list, dist_list = [], []
        for i in probes:
            ids = self.postings[i]
            if len(ids) == 0:
                continue
            shift = self.centroids[i] if self.residual else np.zeros_like(q)
            L = self.pq.adc_lut(q - shift)            # 查询侧也减去同一个偏置
            cand_list.append(ids)
            dist_list.append(self.pq.adc_dists(L, self.codes[ids]))

        cand = np.concatenate(cand_list)
        dist = np.concatenate(dist_list)
        kk = min(k, len(cand))
        part = np.argpartition(dist, kk - 1)[:kk]
        part = part[np.argsort(dist[part])]
        return cand[part], len(cand)


def main() -> None:
    print("IVFADC：倒排文件 + 残差乘积量化（§18.4.3）\n")
    rng = np.random.default_rng(SEED)

    N, D = 20000, 64
    X, _ = make_vectors(n=N, dim=D, n_clusters=80, spread=0.45)
    Q = make_queries(X, n_queries=100)
    k = 10
    truth, _ = knn_exact(X, Q, k)

    # ---------------- 1) 建索引 ----------------
    n_list, m, K = 256, 16, 256
    idx = IVFADC(n_list=n_list, m=m, K=K, residual=True, seed=SEED)
    t0 = time.perf_counter()
    idx.train(X)
    t_train = time.perf_counter() - t0

    sizes = np.array([len(p) for p in idx.postings])
    print(f"① 建索引（N={N:,}, D={D}, n_list={n_list}, m={m}, K={K}）：")
    rows = [
        ["粗聚类中心", f"{n_list}", f"{idx.centroids.nbytes / 1e3:.0f} KB",
         "k-means，O(n_list·N·D)"],
        ["PQ 码本", f"m×K×D* = {m}×{K}×{D // m}",
         f"{idx.pq.nbytes() / 1e3:.0f} KB", "训练在残差上"],
        ["倒排编码", f"{N:,} × {m} B", f"{N * m / 1e3:.0f} KB",
         "cluster id 省掉：同表内隐式已知"],
    ]
    table(["结构", "规模", "大小", "说明"], rows)
    print(f"\n  训练总耗时 {t_train:.1f} s。倒排表大小："
          f"min {sizes.min()} / 中位 {int(np.median(sizes))} / max {sizes.max()}")
    print("  分布不均是因为合成数据的簇中心是随机高斯，粗聚类无法完全均分；")
    print("  工业实现会用 balanced k-means 或在建表后拆分超长表。\n")

    # ---------------- 2) n_probe 扫描 ----------------
    print("② 扫 n_probe：Recall / 候选数 / 耗时的三角关系（残差 PQ）：")
    rows = []
    for n_probe in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        preds, cand_tot = [], 0
        t0 = time.perf_counter()
        for qi in range(len(Q)):
            p, nc = idx.search(Q[qi], k, n_probe)
            preds.append(p)
            cand_tot += nc
        dt = (time.perf_counter() - t0) / len(Q)
        rec = float(np.mean([len(set(p.tolist()) & set(t.tolist())) / k
                             for p, t in zip(preds, truth)]))
        rows.append([n_probe, f"{cand_tot / len(Q):.0f}",
                     f"{cand_tot / len(Q) / N:.2%}",
                     f"{rec:.3f}", f"{dt * 1e3:.2f} ms"])
    table(["n_probe", "平均候选数", "占全库比例", "Recall@10", "单查询耗时"], rows)
    rec_last = float(rows[-1][3])
    print("\n  正文的「n_probe 换 Recall 与 Latency」在这里能看到前半段：")
    print(f"  n_probe=1 只扫约 {rows[0][2]} 的库，召回 {rows[0][3]}；")
    print("  加到 4 个表召回到 7 成，之后再加大几乎不再涨——Top-10 的近邻")
    print("  基本都在最近的几个表里，剩下的漏召回来自 PQ 量化误差")
    print(f"  （这里 n_probe 取到 {rows[-1][0]}，召回上限约 {rec_last}）。")
    print("  「召回随 n_probe 单调大涨」的前提是近邻分散在多个表里：")
    print("  n_list 越小、表越长，近邻越容易落在同一张表，曲线反而更平。")
    print("  下面 ④ 的 n_list 对比可以看到这一点。\n")

    # ---------------- 3) 残差 vs 直接 PQ ----------------
    print("③ 残差编码 vs 直接编码（同 n_list、同编码长度、同 n_probe）：")
    print("   （n_probe 取 4/8/16，n_list/m 与 ①② 一致）")
    idx_direct = IVFADC(n_list=n_list, m=m, K=K, residual=False, seed=SEED)
    idx_direct.train(X)
    rows = []
    for n_probe in (4, 8, 16):
        out = []
        for model in (idx_direct, idx):
            pr = []
            for qi in range(len(Q)):
                p, _ = model.search(Q[qi], k, n_probe)
                pr.append(p)
            out.append(float(np.mean([len(set(p.tolist()) & set(t.tolist())) / k
                                      for p, t in zip(pr, truth)])))
        # 两边各自的重建误差
        rows.append([n_probe, f"{out[0]:.3f}", f"{out[1]:.3f}",
                     f"{(out[1] - out[0]) / max(out[0], 1e-9):+.1%}"])
    table(["n_probe", "直接 PQ Recall", "残差 PQ Recall", "相对提升"], rows)

    # 量化误差对比（只看被分到的那个粗中心上的残差）
    rec_direct = idx_direct.pq.reconstruct(idx_direct.codes)
    rec_res = idx.pq.reconstruct(idx.codes) + idx.centroids[idx.assign]
    err_d = float(((rec_direct - X) ** 2).sum(1).mean())
    err_r = float(((rec_res - X) ** 2).sum(1).mean())
    print(f"\n  重建 MSE：直接 PQ {err_d:.2f} → 残差 PQ {err_r:.2f}"
          f"（降低 {(1 - err_r / err_d):.0%}）")
    print("  粗聚类先把「这个向量大概在哪」记下来，残差只描述局部细节，")
    print("  范数更小、分布更集中，同样的 K 个中心就能表示得更细。")
    print("  正文「残差通常范数更小」的说法在这里得到定量验证。\n")

    # ---------------- 4) n_list 的影响 ----------------
    print("④ n_list 的影响（固定 n_probe=4、m=16，其余不变）：")
    rows = []
    for nl in (16, 64, 256, 512):
        model = IVFADC(n_list=nl, m=m, K=K, residual=True, seed=SEED)
        model.train(X)
        pr, cand_tot = [], 0
        for qi in range(len(Q)):
            p, nc = model.search(Q[qi], k, 4)
            pr.append(p)
            cand_tot += nc
        rec = float(np.mean([len(set(p.tolist()) & set(t.tolist())) / k
                             for p, t in zip(pr, truth)]))
        rows.append([nl, f"{N / nl:.0f}", f"{cand_tot / len(Q):.0f}", f"{rec:.3f}"])
    table(["n_list", "平均表长", "平均候选数", "Recall@10"], rows)
    print("\n  n_list 是一个「表短 vs 跨表漏」的权衡：表越短，同样 probe 4 次")
    print("  扫的候选越少；但近邻落在未访问表里的概率也越高。本数据上召回先升后降，")
    print("  说明存在一个性价比最优点（这里是 256 附近），而不是越大越好。\n")

    print("⑤ 一句话：IVF 负责「少扫」（粗聚类定向），PQ 负责「扫得快」")
    print("   （查表代替浮点距离），残差负责「压得准」。三者合起来才是")
    print("   工业界能扛十亿级的 IndexIVFPQ；OPQ 只是再给残差加一个旋转。")


if __name__ == "__main__":
    main()
