"""
SimHash：随机超平面、谷歌 64 位指纹与分块倒排检索（第18章 §18.3.2）

正文的三层内容，本脚本逐层验证：

1) 随机超平面 LSH：h_r(x) = sign(xᵀr)，r ~ N(0, I)，夹角 θ 的两条向量满足
       P[h(x) = h(y)] = 1 − θ/π
   长度 L 的签名里，汉明距离 H 满足 H/L ≈ θ/π，于是 θ ≈ π·H/L。
2) 谷歌 SimHash：每个特征先做普通哈希得到 L 位编码 b_ij，转成
       r_ij = 2b_ij − 1 ∈ {−1, +1}
   再按权重累加 s = Σ w_i r_i，取 sign(s)。正文证明了它与随机超平面
   LSH 完全等价——第 j 位就是 r_jᵀx，只是 r_j 由哈希"隐式"生成，
   不需要存储 L×d 的投影矩阵。
3) 谷歌的检索方案：64 位指纹切成 m = k+1 块建倒排索引。若两条指纹的
   汉明距离 ≤ k，鸽笼原理保证至少有一块完全相同，所以只查 m 条倒排链
   就能召回全部近重复，不用扫描全库。

本脚本：
1) 实测 P(same) 与 1−θ/π、H/L 与 θ/π 两条等式（随机超平面）；
2) 实现谷歌 SimHash（splitmix64 + 带权累加），与显式随机投影逐位比对，
   并验证"汉明距离反推夹角"的精度；
3) 20 万条 64 位指纹上对比暴力扫描与 4×16 bit 分块倒排：候选数、
   召回率、耗时，并演示 m < k+1 时鸽笼原理失效、必然漏召回；
4) 说明 Faiss IndexLSH 为什么只是"压缩"而不是子线性检索。

运行：python3 ch03_lsh/simhash.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import table  # noqa: E402

SEED = 2026
BITS = 64          # 谷歌 SimHash 的指纹长度
N_DB = 200_000     # 检索演示的库规模


# --------------------------------------------------------------------------
# 一、64 位哈希（splitmix64）：模拟工业界的"普通哈希函数"
# --------------------------------------------------------------------------
GOLD = np.uint64(0x9E3779B97F4A7C15)
MUL1 = np.uint64(0xBF58476D1CE4E5B9)
MUL2 = np.uint64(0x94D049BB133111EB)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """
    splitmix64：把整数打散成 64 位伪随机数。

    选它的原因：一行位运算就能在 numpy 里向量化，且雪崩性质好，
    可以当作"普通哈希函数"的替身。正文强调谷歌方案里每个特征的编码
    来自普通哈希而非随机数生成器，这里正是按这个思路实现的。
    """
    with np.errstate(over="ignore"):          # uint64 溢出是预期行为
        z = x + GOLD
        z = (z ^ (z >> np.uint64(30))) * MUL1
        z = (z ^ (z >> np.uint64(27))) * MUL2
        return z ^ (z >> np.uint64(31))


def hash_bits(fids: np.ndarray, bits: int = BITS) -> np.ndarray:
    """特征编号 → [m, bits] 的 0/1 编码 b_ij。"""
    h = splitmix64(fids.astype(np.uint64))
    shifts = np.arange(bits, dtype=np.uint64)
    return ((h[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.int8)


def google_simhash(fids: np.ndarray, weights: np.ndarray,
                   bits: int = BITS) -> tuple[int, np.ndarray]:
    """
    正文的谷歌 SimHash：

        b_ij ∈ {0,1}  →  r_ij = 2b_ij − 1
        s_j = Σ_i w_i·r_ij
        指纹第 j 位 = 1 if s_j ≥ 0 else 0

    返回 (指纹整数, 64 维累计权重 s)。
    """
    r = 2 * hash_bits(fids, bits).astype(np.float64) - 1.0     # [m, L]
    s = weights @ r                                            # [L]
    bits_arr = (s >= 0).astype(np.uint8)
    fp = int(sum(int(b) << j for j, b in enumerate(bits_arr)))
    return fp, s


def popcount(x: np.ndarray) -> np.ndarray:
    """对 uint64 数组做 popcount（numpy 没有内置位计数）。"""
    u8 = x.view(np.uint8).reshape(-1, 8)
    return np.unpackbits(u8, axis=1).sum(axis=1)


def hamming(fp: np.ndarray, q: int) -> np.ndarray:
    return popcount(fp ^ np.uint64(q))


def build_block_tables(db_fp: np.ndarray, m_blocks: int) -> list[dict]:
    """
    谷歌方案的倒排表：64 位指纹切成 m 块，每块作为 Key 建一张表。

    真实系统里这四张表是离线建好的；这里抽成函数，方便对比不同 m。
    """
    width = BITS // m_blocks
    mask = (1 << width) - 1
    tables: list[dict] = [{} for _ in range(m_blocks)]
    for doc, fpv in enumerate(db_fp):
        v = int(fpv)
        for b in range(m_blocks):
            tables[b].setdefault((v >> (b * width)) & mask, []).append(doc)
    return tables


def query_block_tables(q_int: int, tables: list[dict],
                       m_blocks: int) -> set:
    """把查询指纹按同样的切法查 m 条倒排链，取并集。"""
    width = BITS // m_blocks
    mask = (1 << width) - 1
    cand: set = set()
    for b in range(m_blocks):
        cand.update(tables[b].get((q_int >> (b * width)) & mask, ()))
    return cand


# --------------------------------------------------------------------------
def batch_simhash(fids: np.ndarray, weights: np.ndarray,
                  bits: int = BITS, chunk: int = 2000) -> np.ndarray:
    """
    批量版谷歌 SimHash：fids / weights 形状 [n, m]，返回 [n] 的 uint64 指纹。

    逐条调用会做 n 次 Python 循环，20 万条太慢；这里分块向量化：
    [chunk, m, L] 的符号矩阵与权重做 einsum，再按位打包成整数。
    分块是为了控制中间张量的内存（2000×80×64 的 float32 约 41 MB）。
    """
    n = len(fids)
    out = np.empty(n, dtype=np.uint64)
    powers = (np.uint64(1) << np.arange(bits, dtype=np.uint64))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        B = hash_bits(fids[s:e].ravel(), bits).reshape(e - s, fids.shape[1], bits)
        R = 2.0 * B.astype(np.float32) - 1.0
        S = np.einsum("nml,nm->nl", R, weights[s:e].astype(np.float32))
        bits_mat = (S >= 0).astype(np.uint64)
        out[s:e] = (bits_mat * powers[None, :]).sum(axis=1)
    return out


def rotate(x: np.ndarray, theta: float, u: np.ndarray) -> np.ndarray:
    """把单位向量 x 在 span{x,u} 平面内旋转 θ 弧度。"""
    return np.cos(theta) * x + np.sin(theta) * u


def main() -> None:
    print("SimHash：随机超平面、谷歌 64 位指纹与分块倒排检索（§18.3.2）\n")
    rng = np.random.default_rng(SEED)

    # ---------------- 1) 随机超平面的两条等式 ----------------
    print("① 随机超平面：P(同侧) = 1 − θ/π，H/L ≈ θ/π（D=64，2 万个随机方向）")
    D = 64
    x = rng.normal(size=D)
    x /= np.linalg.norm(x)
    u = rng.normal(size=D)
    u -= (u @ x) * x                 # 与 x 正交
    u /= np.linalg.norm(u)

    n_proj = 20_000
    R = rng.normal(size=(n_proj, D))
    rows = []
    for frac in (0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0):
        theta = np.pi * frac
        y = rotate(x, theta, u)
        hx = (R @ x) >= 0
        hy = (R @ y) >= 0
        emp = float((hx == hy).mean())
        rows.append([f"{frac:.2f}π",
                     f"{emp:.4f}", f"{1 - frac:.4f}",
                     f"{1 - emp:.4f}", f"{frac:.4f}"])
    table(["夹角 θ", "实测 P(同侧)", "1−θ/π", "实测 H/L", "θ/π"], rows)
    print("\n  两列比值几乎逐行相等：每一位就是一个 Bernoulli(1−θ/π)，")
    print("  L 位独立，汉明距离 H 的期望是 L·θ/π，std 是 sqrt(L·θ/π·(1−θ/π))。")
    print("  注意 θ=0 与 θ=π 两行是退化情形：两个向量共线，符号永远相同或永远相反。\n")

    # ---------------- 2) 谷歌 SimHash 与显式随机投影 ----------------
    print("② 谷歌 SimHash 与显式随机超平面是否逐位一致：")
    m_feat = 200
    fids = np.arange(m_feat)
    weights = rng.uniform(0.2, 3.0, size=m_feat)

    fp, s = google_simhash(fids, weights)
    bits_google = np.array([(fp >> j) & 1 for j in range(BITS)], dtype=np.uint8)

    # 显式版本：哈希位矩阵转置后，第 j 列就是随机投影向量 r_j，
    # 那么 s_j = r_jᵀx（x 为特征权重向量）——与谷歌版应逐位相同
    Bm = hash_bits(fids, BITS)                       # [m, L]
    Rj = (2.0 * Bm - 1.0).T                          # [L, m]
    s_explicit = Rj @ weights
    bits_explicit = (s_explicit >= 0).astype(np.uint8)

    agree = float((bits_google == bits_explicit).mean())
    print(f"  特征数 m = {m_feat}，指纹长度 L = {BITS}，随机权重 w_i ~ U(0.2, 3.0)")
    print(f"  两种实现的逐位一致率 = {agree:.4f}")
    print(f"  最大绝对差 max|s_google − s_explicit| = {np.abs(s - s_explicit).max():.2e}")
    print("  正文的等价性推导在这里是恒等式而非近似：谷歌版算的就是 r_jᵀx，")
    print("  只是 r_j 的第 i 个分量 r_ij 由「特征 i 的哈希第 j 位」给出。")
    print("  工程收益是存储：不必保存 L×m 的投影矩阵，只需一个 64 位哈希函数。\n")

    # 夹角估计精度
    print("  用汉明距离反推夹角（两条 200 维文档向量，人为旋转到不同夹角）：")
    rows = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5):
        theta = np.pi * frac
        # 构造 y：在特征权重空间里把 x 旋转 θ，再用同一套哈希指纹比较
        v = rng.normal(size=m_feat)
        v -= (v @ weights) / (weights @ weights) * weights
        v *= np.linalg.norm(weights) / np.linalg.norm(v)
        wy = np.cos(theta) * weights + np.sin(theta) * v
        fp_x, _ = google_simhash(fids, weights)
        fp_y, _ = google_simhash(fids, wy)
        h = float(popcount(np.array([np.uint64(fp_x ^ fp_y)]))[0])
        rows.append([f"{frac:.1f}π", f"{h:.0f} / {BITS}",
                     f"{np.pi * h / BITS:.3f}", f"{theta:.3f}"])
    table(["真实 θ", "汉明距离 H", "π·H/L 估计", "真实 θ"], rows)
    print("\n  L=64 时每位的标准差约 0.5/√64 ≈ 0.06（弧度约 0.2），")
    print("  所以角度估计有可见误差——这也是工业上 SimHash 只用来做粗筛，")
    print("  候选还要回到原始向量或原始文档做精确比对的原因。\n")

    # ---------------- 3) 分块倒排检索 ----------------
    print("③ 谷歌的分块倒排检索：暴力扫指纹 vs m = k+1 块查表")
    print(f"   （{N_DB:,} 条 64 位指纹，用随机文档生成）")

    # 库：随机稀疏文档的 SimHash 指纹；再构造"近重复查询"
    n_docs = N_DB
    vocab = 5000
    kmax = 120
    n_feat = rng.integers(40, kmax + 1, size=n_docs)
    # 用矩形 [n, kmax] 存稀疏特征：不足 kmax 的位置填 fid=0、w=0，
    # 对结果没有影响（权重为 0），但能把整批计算放进 numpy
    feat_ids = np.zeros((n_docs, kmax), dtype=np.int64)
    feat_w = np.zeros((n_docs, kmax), dtype=np.float32)
    for i in range(n_docs):
        k = int(n_feat[i])
        feat_ids[i, :k] = rng.choice(vocab, k, replace=False)
        feat_w[i, :k] = rng.uniform(0.2, 3.0, size=k)
    db_fp = batch_simhash(feat_ids, feat_w)

    # 查询：以第 0 条为基准，翻转 k 个随机 bit，保证真实汉明距离正好是 k
    q_fp = int(db_fp[0])
    n_q = 20

    def make_query(k_flip: int, t: int, m_blocks: int | None = None) -> int:
        """
        构造与库中第 0 条指纹汉明距离恰为 k_flip 的查询。

        m_blocks 给定时，若 m_blocks ≤ k_flip，就强制「每个块至少被翻一位」——
        这正是鸽笼原理失效的最坏情形。不做这个强制的话，随机翻位有
        ~1 − m!/m^k 的概率把两位翻进同一块，恰好留下一个完整块，
        于是 m = k 也会偶然召回，看不出问题（第一版就踩了这个坑）。
        """
        rngq = np.random.default_rng(SEED + 100 * k_flip + t)
        bits_q = [(q_fp >> j) & 1 for j in range(BITS)]
        if k_flip:
            if m_blocks is not None and m_blocks <= k_flip:
                width = BITS // m_blocks
                chosen = [int(b * width + rngq.integers(width))
                          for b in range(m_blocks)]
                rest = k_flip - m_blocks
                if rest > 0:
                    pool = [j for j in range(BITS) if j not in chosen]
                    chosen += rngq.choice(pool, rest, replace=False).tolist()
            else:
                chosen = rngq.choice(BITS, k_flip, replace=False).tolist()
            for j in chosen:
                bits_q[j] ^= 1
        return sum(b << j for j, b in enumerate(bits_q))

    rows = []
    for k_flip in (0, 1, 2, 3, 4):
        m_blocks = k_flip + 1
        tables = build_block_tables(db_fp, m_blocks)      # 离线建表，不计入查询耗时
        hit = 0
        cand_tot = 0
        t_brute = 0.0
        t_ivf = 0.0
        for t in range(n_q):
            q_int = make_query(k_flip, t)
            t0 = time.perf_counter()
            h_all = hamming(db_fp, q_int)
            truth = np.where(h_all <= k_flip)[0]
            t_brute += time.perf_counter() - t0

            t0 = time.perf_counter()
            cand = query_block_tables(q_int, tables, m_blocks)
            # 候选上再算精确汉明距离（正文的最后一步）
            cand_arr = np.fromiter(cand, dtype=np.int64, count=len(cand))
            _ = hamming(db_fp[cand_arr], q_int)
            t_ivf += time.perf_counter() - t0

            hit += len(set(truth.tolist()) & cand) / max(len(truth), 1)
            cand_tot += len(cand)
        rows.append([k_flip, m_blocks, f"{hit / n_q:.4f}",
                     f"{cand_tot / n_q:,.0f}",
                     f"{t_brute / n_q * 1e3:.2f} ms",
                     f"{t_ivf / n_q * 1e3:.2f} ms"])
    table(["k", "块数 m", "近重复召回率", "平均候选数",
           "暴力扫描", "查表+重排"], rows)
    print("\n  建表是离线的，这里只计查询：m 次字典查找 + 候选上的精确汉明距离。")
    print("  候选数远小于 20 万——16 bit 的块有 6.5 万个可能取值，随机指纹几乎")
    print("  不会撞桶，所以查询常只带回 1 条真实近重复；k 增大时块变短，桶才变挤。")
    print("  这就是正文说的「只访问 4 条倒排链」，计算量降低数万倍。")
    print("  召回率始终保持 1.000，因为 m = k+1 时鸽笼原理保证至少一块相同。\n")

    print("  反例：块数与 k 不匹配时，鸽笼原理失效：")
    rows = []
    for k_flip in (1, 2, 3, 4):
        for m_blocks in (1, 2, 4, 5):
            tables = build_block_tables(db_fp, m_blocks)
            hit = 0
            for t in range(n_q):
                q_int = make_query(k_flip, t, m_blocks)
                truth = set(np.where(hamming(db_fp, q_int) <= k_flip)[0].tolist())
                cand = query_block_tables(q_int, tables, m_blocks)
                hit += len(truth & cand) / max(len(truth), 1)
            rows.append([k_flip, m_blocks, "✓" if m_blocks > k_flip else "✗",
                         f"{hit / n_q:.3f}"])
    table(["k", "块数 m", "m > k?", "近重复召回率"], rows)
    print("\n  只有 m ≥ k+1 的行召回是 1.000；m ≤ k 时每个块都可能被改到，")
    print("  必然漏掉近重复（这里对 m ≤ k 强制每块各翻一位，即最坏情形）。")
    print("  代价是块越短桶越挤：m 增大 → 候选数上升，这正是 §18.3.3 Banding")
    print("  要用 b、r 两个参数平衡的东西。\n")

    # ---------------- 4) IndexLSH：压缩，不是子线性 ----------------
    print("④ Faiss IndexLSH 的本质：全量汉明比较的压缩检索")
    print("   （2000 条 32 维簇状向量，组内夹角小；随机高维向量两两近正交，")
    print("    SimHash 在这种数据上本来就分不开，不适合演示压缩的代价）")
    N = 2000
    dim = 32
    centers = rng.normal(size=(20, dim)) * 3.0
    labels = rng.integers(0, 20, size=N)
    Xv = (centers[labels] + rng.normal(0, 0.35, (N, dim))).astype(np.float32)
    Xv /= np.linalg.norm(Xv, axis=1, keepdims=True)
    pick = rng.choice(N, 100, replace=False)
    Qv = Xv[pick] + rng.normal(0, 0.01, (100, dim)).astype(np.float32)
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)

    k = 10
    rows = []
    for L in (32, 64, 128):
        proj = rng.normal(size=(L, dim)).astype(np.float32)
        code = (Xv @ proj.T >= 0)
        qcode = (Qv @ proj.T >= 0)
        rec = 0.0
        for i in range(len(Qv)):
            # IndexLSH 的默认查询：对全部编码算汉明距离，取最近的 k 个候选
            hd = (code != qcode[i]).sum(axis=1)
            cand = np.argsort(hd)[:k]
            true = np.argsort(-(Xv @ Qv[i]))[:k]
            rec += len(set(cand.tolist()) & set(true.tolist())) / k
        rows.append([L, f"{rec / len(Qv):.3f}",
                     f"{N * L / 8 / 1024:.1f} KB", "全量 2000 次汉明比较"])
    table(["签名长度 L", "Recall@10", "索引大小", "查询方式"], rows)
    print("\n  召回随 L 上升，但查询始终是 O(N)——索引只是把 32 维 float32")
    print("  （128 B）压成 L 位（L/8 B），省的是内存和带宽，不是扫描量。")
    print("  真正子线性的是 IndexBinaryMultiHash / 谷歌分块倒排：多张哈希表，")
    print("  只访问命中的桶。二者的区别正是正文「压缩编码 vs 子线性检索」那句话。\n")

    print("⑤ 一句话：SimHash 用 sign(xᵀr) 把夹角编码成 bit，H/L ≈ θ/π；")
    print("   谷歌版用普通哈希隐式生成 r，再用 k+1 块倒排把「找汉明距离 ≤ k」")
    print("   变成几次字典查找。前者是编码，后者才是索引。")


if __name__ == "__main__":
    main()
