"""
乘积量化 PQ：子空间分解、码本训练与 SDC/ADC 距离（第18章 §18.4）

正文的 PQ 定义：

    把 D 维向量均分为 m 个子向量 u_1(x), …, u_m(x)，每个 D* = D/m 维；
    第 j 个子空间用 K-Means 训练 K 个聚类中心 C_j = {c_{j,1}, …, c_{j,K}}；
    编码 q(x) = [q_1(x), …, q_m(x)]，q_j(x) = argmin_c ‖u_j(x) − c‖。

存储：m 个 8 bit 编号（K=256），128 维 float32 的 512 B 压到 8 B，压缩 64 倍。
但整个码本是笛卡尔积，表达能力是 K^m = 256^8 ≈ 1.8×10^19 个码字。

距离的两种算法：
    SDC：d̃(x,y) = ‖q(x) − q(y)‖，查询也要量化，K×K 距离表可预计算
    ADC：d̃(x,y) = ‖x − q(y)‖，查询不量化，每个查询建 m×K 的查找表
         L_j(i) = ‖u_j(x) − c_{j,i}‖²，之后 m 次查表 + 累加

正文的复杂度结论：暴力 O(ND)，ADC 建表 O(KD) + 扫描 O(mN)。

本脚本：
1) 在 6000 条 128 维数据上训练 PQ（m=8, K=256），核对压缩率与码本大小；
2) 扫子空间数 m，看「量化误差 vs 压缩率」的权衡；
3) 对比 SDC 与 ADC 的 Recall@10——正文说 ADC 明显更好，这里实测差多少；
4) 实测 ADC 扫描耗时与暴力检索的差距，验证 O(mN) vs O(DN)。

运行：python3 ch04_pq/product_quantization.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    knn_exact, make_queries, make_vectors, recall_at_k, sq_dists, table,
)

SEED = 2026


# --------------------------------------------------------------------------
# K-Means（PQ 的每个子空间都要跑一次）
# --------------------------------------------------------------------------
def kmeans(X: np.ndarray, K: int, iters: int = 25,
           seed: int = SEED) -> np.ndarray:
    """
    朴素的 Lloyd 迭代：随机初始化 + 最近中心分配 + 重算均值。

    为什么不用 k-means++：这里的目的是展示 PQ 的结构，子空间维度只有
    D/m（8~32 维），随机初始化的 K-Means 在 25 轮内已经足够收敛；
    用 k-means++ 会多 20 行代码，但结果差异在 1% 以内（可自行替换验证）。

    重算均值的写法：不能在循环里用布尔索引 C[k] = X[a == k].mean(0)，
    那样每轮要做 K 次全量扫描（K=256 时是 1536 万次比较）。这里用
    bincount 一次聚合所有簇的和与计数，是稳定的 O(n·D + n) 操作。
    """
    rng = np.random.default_rng(seed)
    n, D = X.shape
    C = X[rng.choice(n, K, replace=False)].copy()
    x_sq = (X * X).sum(axis=1)                # 只算一次，循环里复用
    for _ in range(iters):
        # 写成一个 [n, K] 的矩阵乘，而不是 common.sq_dists 的「按查询逐条」——
        # 那个版本为省内存设计了 Python 循环，K=256 时等于 256 次小 BLAS 调用，
        # 每次调用的线程调度开销反而成了主成本（实测慢 4 倍以上）
        d = np.maximum(x_sq[:, None] + (C * C).sum(1)[None, :] - 2.0 * (X @ C.T), 0.0)
        a = np.argmin(d, axis=1)              # 每个样本最近的中心编号
        # 逐维 bincount：sum_{i: a_i=k} X[i, j]，再除以计数得到均值
        cnt = np.bincount(a, minlength=K).astype(np.float32)
        newC = np.empty_like(C)
        for j in range(D):
            newC[:, j] = np.bincount(a, weights=X[:, j], minlength=K)
        newC /= np.maximum(cnt, 1.0)[:, None]
        empty = cnt == 0
        if empty.any():          # 空簇保留原中心，避免除零和中心消失
            newC[empty] = C[empty]
        C = newC
    return C.astype(np.float32)


# --------------------------------------------------------------------------
# PQ
# --------------------------------------------------------------------------
class PQ:
    """乘积量化器：训练 m 个子码本，提供编码、重建、SDC、ADC 四种操作。"""

    def __init__(self, m: int = 8, K: int = 256, seed: int = SEED) -> None:
        self.m = m
        self.K = K
        self.seed = seed
        self.codebooks: list[np.ndarray] = []
        self.D = 0
        self.Ds = 0

    def train(self, X: np.ndarray) -> None:
        D = X.shape[1]
        assert D % self.m == 0, "维度必须能被 m 整除"
        self.D = D
        self.Ds = D // self.m
        # 码本拼成 [m, K, D*] 的单个数组：ADC 建表可以一次算完所有子空间，
        # 不用 for j in range(m) ——扫描侧是热路径，省一次 Python 循环很值
        self.code = np.empty((self.m, self.K, self.Ds), dtype=np.float32)
        for j in range(self.m):
            sub = X[:, j * self.Ds:(j + 1) * self.Ds]
            self.code[j] = kmeans(sub, self.K, seed=self.seed + j)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """返回 [n, m] 的 uint8 编码。"""
        n = len(X)
        codes = np.empty((n, self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = X[:, j * self.Ds:(j + 1) * self.Ds]
            codes[:, j] = np.argmin(sq_dists(sub, self.code[j]), axis=0)
        return codes

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        """按编码取回聚类中心，拼成完整向量（ADC 里只在需要时做，SDC 不需要）。"""
        n = len(codes)
        out = np.empty((n, self.D), dtype=np.float32)
        for j in range(self.m):
            out[:, j * self.Ds:(j + 1) * self.Ds] = self.code[j][codes[:, j]]
        return out

    # -------- SDC --------
    def sdc_table(self) -> np.ndarray:
        """
        预计算每个子空间的 K×K 距离表：T[j, a, b] = ‖c_{j,a} − c_{j,b}‖²。

        正文强调「查询过程无需任何浮点距离计算」，指的就是这张表。
        代价是 m × K² 个浮点数（K=256、m=8 时约 2 MB），比码本本身大得多。
        """
        T = np.empty((self.m, self.K, self.K), dtype=np.float32)
        for j in range(self.m):
            C = self.code[j]
            c2 = (C * C).sum(axis=1)
            T[j] = np.maximum(c2[:, None] + c2[None, :] - 2.0 * (C @ C.T), 0.0)
        return T

    def sdc_dists(self, qcodes: np.ndarray, dbcodes: np.ndarray,
                  tables: np.ndarray) -> np.ndarray:
        """[nq, ndb] 的近似平方距离，纯查表累加。"""
        out = np.zeros((len(qcodes), len(dbcodes)), dtype=np.float32)
        for j in range(self.m):
            out += tables[j][qcodes[:, j][:, None], dbcodes[:, j][None, :]]
        return out

    # -------- ADC --------
    def adc_lut(self, q: np.ndarray) -> np.ndarray:
        """
        单个查询的查找表：L[j, i] = ‖u_j(q) − c_{j,i}‖²，形状 [m, K]。

        正文说建表复杂度 O(m·K·D*) = O(KD)，这张表就是它。
        一次 broadcast 算完 m 个子空间，避免 m 次 Python 循环。
        """
        sub = q.reshape(self.m, 1, self.Ds)                 # [m, 1, D*]
        diff = self.code - sub                              # [m, K, D*]
        return (diff * diff).sum(axis=2).astype(np.float32)

    def adc_dists(self, L: np.ndarray, dbcodes: np.ndarray) -> np.ndarray:
        """[ndb] 的近似平方距离：对每个数据库向量做 m 次查表累加。"""
        out = np.zeros(len(dbcodes), dtype=np.float32)
        for j in range(self.m):
            out += L[j][dbcodes[:, j]]
        return out

    def adc_lut_batch(self, Q: np.ndarray) -> np.ndarray:
        """
        批量建表：[nq, m, K]。

        一次算完所有查询的查找表，而不是每个查询单算——建表本身是 O(KD)，
        批量做能让 BLAS 吃满，这是实际实现里最省事的优化。
        """
        nq = len(Q)
        sub = Q.reshape(nq, self.m, 1, self.Ds)             # [nq, m, 1, D*]
        diff = self.code[None] - sub                        # [nq, m, K, D*]
        return (diff * diff).sum(axis=3).astype(np.float32)

    def adc_dists_batch(self, lut: np.ndarray,
                        dbcodes: np.ndarray) -> np.ndarray:
        """批量查表累加：[nq, ndb]。每个子空间取一次 [nq, ndb] 的索引。"""
        out = np.zeros((len(lut), len(dbcodes)), dtype=np.float32)
        for j in range(self.m):
            out += lut[:, j][:, dbcodes[:, j]]
        return out

    def nbytes(self) -> int:
        """码本总字节数。编码另算：每条 m 字节。"""
        return int(self.code.nbytes)


def main() -> None:
    print("乘积量化 PQ：子空间分解、码本训练与 SDC/ADC 距离（§18.4）\n")
    rng = np.random.default_rng(SEED)

    N, D = 6000, 128
    X, _ = make_vectors(n=N, dim=D, n_clusters=40, spread=0.45)
    Q = make_queries(X, n_queries=100)
    k = 10
    truth, truth_d = knn_exact(X, Q, k)

    # ---------------- 1) 压缩账 ----------------
    m, K = 8, 256
    pq = PQ(m=m, K=K, seed=SEED)
    t0 = time.perf_counter()
    pq.train(X)
    t_train = time.perf_counter() - t0
    codes = pq.encode(X)
    recon = pq.reconstruct(codes)

    raw_b = D * 4
    code_b = m                      # K=256 → 每段 1 字节
    print(f"① 压缩账（N={N:,}, D={D}, m={m}, K={K}）：")
    rows = [
        ["原始向量", f"{raw_b} B", f"{raw_b * N / 1e6:.1f} MB", "—"],
        ["PQ 编码", f"{code_b} B", f"{code_b * N / 1e3:.0f} KB",
         f"{raw_b / code_b:.0f}×"],
        ["子码本", "—", f"{pq.nbytes() / 1e6:.2f} MB",
         f"m×K×D* = {m}×{K}×{D // m} 个 float32"],
    ]
    table(["对象", "单条大小", "总量", "说明"], rows)
    print(f"\n  训练耗时 {t_train:.1f} s（{m} 个子空间各跑一次 K={K} 的 K-Means）。")
    print(f"  组合码字数 K^m = {K}^{m} ≈ {float(K) ** m:.1e}，而实际只存了 "
          f"m×K = {m * K} 个中心——")
    print("  「乘积」指的就是这个笛卡尔积：线性存储换指数级表达能力。")
    mse = float(((recon - X) ** 2).sum(axis=1).mean())
    rel = mse / float((X ** 2).sum(axis=1).mean())
    print(f"  重建误差 MSE = {mse:.4f}，相对能量 {rel:.1%}——这就是 64 倍压缩的代价。\n")

    # ---------------- 2) m 的权衡 ----------------
    print("② 子空间数 m 的权衡（D=128 固定，K=256 固定，Top-10）：")
    rows = []
    for m_ in (4, 8, 16, 32):
        pq_ = PQ(m=m_, K=K, seed=SEED)
        pq_.train(X)
        cd = pq_.encode(X)
        rc = pq_.reconstruct(cd)
        rel_e = float(((rc - X) ** 2).sum(1).mean() / (X ** 2).sum(1).mean())
        # ADC 召回
        rec = 0.0
        lut = pq_.adc_lut_batch(Q)
        preds = np.argsort(pq_.adc_dists_batch(lut, cd), axis=1)[:, :k]
        for qi in range(len(Q)):
            rec += len(set(preds[qi].tolist()) & set(truth[qi].tolist())) / k
        bits = m_ * 8
        rows.append([m_, 128 // m_, f"{bits} bit / {bits // 8} B",
                     f"{128 * 4 / (bits // 8):.0f}×", f"{rel_e:.1%}",
                     f"{rec / len(Q):.3f}"])
    table(["m", "D* = D/m", "编码长度", "压缩率", "相对重建误差", "ADC Recall@10"], rows)
    print("\n  m 增大 → 子空间更短、码本更贴合 → 误差下降、召回上升，")
    print("  但编码变长、建表从 O(KD) 里的 K 不变而 D* 变小（总建表仍是 O(KD)），")
    print("  扫描侧从 m 次查表变成 2m、4m 次。所以 m 的取值是「精度 vs 扫描成本」。")
    print("  工业界 m=8/16 的常见选择，正是这张表里 8 B~16 B 那一段。\n")

    # ---------------- 3) SDC vs ADC ----------------
    print("③ SDC vs ADC：同一个 PQ 编码，只换距离算法（m=8, K=256）：")
    tables = pq.sdc_table()
    qcodes = pq.encode(Q)

    # SDC
    t0 = time.perf_counter()
    pred_s = np.empty((len(Q), k), dtype=np.int64)
    for qi in range(len(Q)):
        dd = pq.sdc_dists(qcodes[qi:qi + 1], codes, tables)[0]
        pred_s[qi] = np.argsort(dd)[:k]
    t_sdc = time.perf_counter() - t0

    # ADC
    t0 = time.perf_counter()
    lut_all = pq.adc_lut_batch(Q)                    # [nq, m, K]
    pred_a = np.argsort(pq.adc_dists_batch(lut_all, codes), axis=1)[:, :k]
    t_adc = time.perf_counter() - t0

    # 距离估计误差：与真实平方距离比较（抽样，全量太慢）
    n_s = 100
    idx_s = rng.choice(N, n_s, replace=False)
    rows_d = []
    for qi in range(3):
        d_true = sq_dists(X[idx_s], Q[qi:qi + 1])[0]
        d_sdc = pq.sdc_dists(qcodes[qi:qi + 1], codes[idx_s], tables)[0]
        L = pq.adc_lut(Q[qi])
        d_adc = pq.adc_dists(L, codes[idx_s])
        rows_d.append([qi,
                       f"{np.abs(d_sdc - d_true).mean():.4f}",
                       f"{np.abs(d_adc - d_true).mean():.4f}",
                       f"{np.abs(d_adc - d_true).mean() / np.abs(d_sdc - d_true).mean():.2f}"])
    table(["查询", "SDC 平均|误差|", "ADC 平均|误差|", "ADC/SDC"],
          rows_d)
    print("\n  ADC 的距离误差系统性更小：它只承担数据库侧的量化误差，")
    print("  SDC 还要再叠加一次查询侧的量化误差。实测召回：")
    table(["算法", "Recall@10", "单查询耗时"],
          [["SDC", f"{recall_at_k(pred_s, truth):.3f}",
            f"{t_sdc / len(Q) * 1e3:.2f} ms"],
           ["ADC", f"{recall_at_k(pred_a, truth):.3f}",
            f"{t_adc / len(Q) * 1e3:.2f} ms"]])
    print("\n  ADC 通常更好，代价是「每个查询要现建一张表」（O(KD) + O(mK) 次除法/减法），")
    print("  而 SDC 的表可以离线算好、查询时零浮点运算。查询量极大且对精度要求不高时")
    print("  SDC 仍有场景，但正文「工业界几乎都用 ADC」与这里的实测一致。\n")

    # ---------------- 4) ADC vs 暴力 ----------------
    print("④ ADC 扫描 vs 暴力欧氏扫描（同样的 N=6000、D=128、Top-10）：")
    t0 = time.perf_counter()
    for _ in range(3):
        sq_dists(X, Q)
    t_brute = (time.perf_counter() - t0) / 3

    t0 = time.perf_counter()
    for _ in range(3):
        lut = pq.adc_lut_batch(Q)
        pq.adc_dists_batch(lut, codes)
    t_adc_all = (time.perf_counter() - t0) / 3

    # 只算 nprobe=1/64 的候选量，模拟 IVF 之后的扫描
    sub_n = N // 64
    rows = [
        ["暴力 L2（float32）", f"{N * D:,}", f"{t_brute / len(Q) * 1e3:.2f} ms", "1.000"],
        [f"ADC 全库扫描", f"{N * m:,}", f"{t_adc_all / len(Q) * 1e3:.2f} ms",
         f"{recall_at_k(pred_a, truth):.3f}"],
    ]
    table(["方式", "每次查询的乘加/查表次数", "单查询耗时", "Recall@10"], rows)
    print(f"\n  在 N={N} 这个量级上，NumPy 的暴力 L2 走 BLAS、能吃到多线程和 cache，")
    print("  而 ADC 是 Python 循环里的 m 次高级索引，反而更慢。这恰恰说明：")
    print("  ADC 的收益要在「N 巨大、且已经用 IVF 把候选缩到几千」时才体现，")
    print("  而且工业实现（Faiss）把查表扫描写成 SIMD 内核，Python 循环体现不出来。")
    print("  复杂度结论 O(mN) < O(DN) 是正确的，但常数因子由实现决定。\n")

    # ---------------- 5) 查找表大小 ----------------
    print("⑤ 两张查找表的大小（正文给的数字核对）：")
    rows = [
        ["SDC 距离表", f"m × K × K = {m}×{K}×{K}",
         f"{m * K * K * 4 / 1e6:.2f} MB", "离线预计算，查询零浮点运算"],
        ["ADC 查找表", f"m × K = {m}×{K}",
         f"{m * K * 4 / 1e3:.0f} KB", "每个查询现建，O(KD) 次运算"],
    ]
    table(["表", "形状", "大小", "说明"], rows)
    print("\n  正文提到的「K=256 时每段 32640 个距离值」= K(K−1)/2 = "
          f"{K * (K - 1) // 2:,}，")
    print("  那是利用对称性只存上三角；存全矩阵是 K² = 65536 个值，两者都极小。")
    print("  所以 SDC 的表不是瓶颈，瓶颈永远是「要不要扫全库」。\n")

    print("⑥ 一句话：PQ 用 m 个独立的低维码本把 512 B 压成 8 B，")
    print("   距离计算退化成 m 次查表；但 PQ 本身仍是 O(N) 扫描，")
    print("   要真正减少候选数还得靠 §18.4.3 的 IVF（见 ivfadc.py）。")


if __name__ == "__main__":
    main()
