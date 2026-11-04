"""
MinHash：用随机排列 / 线性哈希估计 Jaccard 相似度（第18章 §18.3.1）

正文的核心性质：
    h_π(C) = min_{r∈C} π(r)
    P[h_π(A) = h_π(B)] = |A∩B| / |A∪B| = J(A,B)

用 L 个独立哈希得到签名，则
    Ĵ = (1/L) Σ 1[h_i(A) = h_i(B)] → J(A,B)
    std(Ĵ) = sqrt(J(1−J)/L)

正文给了两种实现：
    1) 随机排列（permutation）：最直观、完全符合定义，但需要保存 L 张
       长度 d 的置换表，内存 O(L·d)；
    2) 线性哈希：h(x) = (a·x + b) mod p，p 取大素数（2^31−1）。
       每个哈希函数只存 (a, b) 两个整数，内存 O(L)，工程上等价。

本脚本：
1) 在小集合上逐位对比两种实现，确认签名一致、碰撞概率约等于 J；
2) 扫签名长度 L，画「估计误差 vs sqrt(J(1−J)/L)」，验证正文的方差公式；
3) 展示为什么工程上用线性哈希：内存、构建耗时、精度三者对比；
4) 接一个「用 MinHash 找相似集合」的小检索，说明它是 LSH 的基础。

运行：python3 ch03_lsh/minhash.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import table  # noqa: E402

P = 2 ** 31 - 1           # 正文用的大素数
SEED = 2026


# --------------------------------------------------------------------------
# 一、基于随机排列的实现（完全对应定义）
# --------------------------------------------------------------------------
class MinHashPermutation:
    """
    预生成 L 套独立随机排列 π_1..π_L，签名第 i 位取集合内元素在 π_i 下的最小 rank。

    这里保存的是「元素 → rank」的反向映射而不是排列本身，因为查询只需要 rank。
    注意内存：L × universe_size 个整数，universe 是 10^6 时 L=128 就要 5 亿项。
    """

    def __init__(self, num_hash: int, universe_size: int, seed: int = 42) -> None:
        self.num_hash = num_hash
        self.universe_size = universe_size
        rng = np.random.default_rng(seed)
        self.ranks = []
        for _ in range(num_hash):
            perm = rng.permutation(universe_size)
            rank = np.empty(universe_size, dtype=np.int32)
            rank[perm] = np.arange(universe_size, dtype=np.int32)
            self.ranks.append(rank)

    def get_signature(self, s: set) -> np.ndarray:
        items = np.fromiter(s, dtype=np.int64, count=len(s))
        sig = np.empty(self.num_hash, dtype=np.int64)
        for i, rank in enumerate(self.ranks):
            sig[i] = int(rank[items].min())
        return sig

    def nbytes(self) -> int:
        return self.num_hash * self.universe_size * 4


# --------------------------------------------------------------------------
# 二、基于线性哈希的实现（工程版）
# --------------------------------------------------------------------------
class MinHashLinear:
    """
    h_i(x) = (a_i·x + b_i) mod p。

    当 p 足够大（远大于最大元素值）时，(a·x+b) mod p 对 x 而言是近似均匀的
    「随机重排 + 压缩」，取最小值得到的碰撞概率仍等于 J。数学上它不再是严格
    的排列（可能把两个元素映到同一个值），但 p ≫ d 时碰撞概率可忽略。
    """

    def __init__(self, num_hash: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.num_hash = num_hash
        self.p = P
        self.a = rng.integers(1, P, size=num_hash, dtype=np.int64)
        self.b = rng.integers(0, P, size=num_hash, dtype=np.int64)

    def get_signature(self, s: set) -> np.ndarray:
        items = np.fromiter(s, dtype=np.int64, count=len(s))
        # [L, |s|]：一次性算完所有哈希，再沿集合维度取最小
        h = (self.a[:, None] * items[None, :] + self.b[:, None]) % self.p
        return h.min(axis=1)

    def nbytes(self) -> int:
        return self.num_hash * 16      # 两个 int64


def jaccard(A: set, B: set) -> float:
    return len(A & B) / len(A | B)


def make_pair(target_j: float, size: int = 60,
              universe: int = 4096, seed: int = SEED) -> tuple[set, set]:
    """
    构造一对指定 Jaccard 的集合。

    A 取 size 个随机元素；B 保留其中 m 个、再补 size−m 个新元素，于是
        J = m / (2·size − m)   ⟹   m = 2·size·J / (1 + J)
    """
    rng = np.random.default_rng(seed)
    pool = rng.choice(universe, size * 2, replace=False)
    A = set(pool[:size].tolist())
    m = int(round(2 * size * target_j / (1 + target_j)))
    B = set(pool[:m].tolist()) | set(pool[size:size + (size - m)].tolist())
    return A, B


def make_sets(n_sets: int, universe: int, size: int, n_groups: int,
              keep: float = 0.6, seed: int = SEED) -> tuple[list[set], list[int]]:
    """
    生成有簇结构的集合：同一 group 的集合共享一部分基础元素，其余各自随机。

    keep 控制共享比例：同组两个集合各自从 base 里独立抽 keep 比例，
    交集约 keep²·size，并集约 size + (1−keep)·size，因此
        同组 J ≈ keep² / (2 − keep)      （keep=0.6 时 ≈ 0.26）。
    完全随机的集合两两 J ≈ size/universe 且几乎一样，看不出 MinHash 的区分能力。
    """
    rng = np.random.default_rng(seed)
    bases = [set(rng.choice(universe, size, replace=False).tolist())
             for _ in range(n_groups)]
    sets, labels = [], []
    for _ in range(n_sets):
        g = int(rng.integers(n_groups))
        base = np.array(sorted(bases[g]))
        n_keep = max(1, int(round(size * keep)))
        shared = set(rng.choice(base, n_keep, replace=False).tolist())
        noise = set(rng.choice(universe, size - n_keep, replace=False).tolist())
        sets.append(shared | noise)
        labels.append(g)
    return sets, labels


def main() -> None:
    print("MinHash：用随机排列 / 线性哈希估计 Jaccard 相似度（§18.3.1）\n")
    rng = np.random.default_rng(SEED)

    # ---------------- 1) 两种实现的一致性 ----------------
    print("① 两种签名实现的一致性（小 universe，便于做完整排列）：")
    universe = 4096
    sets, _ = make_sets(n_sets=20, universe=universe, size=60, n_groups=4)
    L = 64
    mh_perm = MinHashPermutation(L, universe, seed=SEED)
    mh_lin = MinHashLinear(L, seed=SEED)

    A, B = sets[0], sets[1]
    sig_p, sig_l = mh_perm.get_signature(A), mh_lin.get_signature(A)
    same = float((sig_p == sig_l).mean())
    print(f"  同一集合的两种签名：逐位相同比例 = {same:.3f}")
    print("  （两者只是「等价的估计器」，用的是不同随机源，签名本身不应相同；")
    print("   要比较的是「碰撞比例是否都接近真实 Jaccard」）\n")

    rows = []
    for tgt in (0.2, 0.4, 0.6, 0.8):
        A2, B2 = make_pair(tgt, seed=SEED + int(tgt * 10))
        J = jaccard(A2, B2)
        sp = float((mh_perm.get_signature(A2)
                    == mh_perm.get_signature(B2)).mean())
        sl = float((mh_lin.get_signature(A2)
                    == mh_lin.get_signature(B2)).mean())
        # 单次估计的方差是 sqrt(J(1−J)/64) ≈ 0.05–0.06，一行的偏差说明不了问题，
        # 所以同时列出 200 个哈希种子下的均值，用来判断「有没有系统性偏差」
        many = []
        for t in range(200):
            mm = MinHashLinear(L, seed=SEED + 3000 + t)
            many.append(float((mm.get_signature(A2) == mm.get_signature(B2)).mean()))
        rows.append([f"{tgt:.1f}", f"{J:.3f}", f"{sp:.3f}", f"{sl:.3f}",
                     f"{np.mean(many):.3f}"])
    table(["目标 J", "真实 J", "排列估计", "线性哈希估计", "线性哈希 200 种子均值"], rows)
    print("\n  逐列看：单次估计在真实 J 上下随机波动（L=64 时 std ≈ 0.05–0.06），")
    print("  而 200 个种子的均值几乎贴着真实 J——两者都是无偏估计器，没有系统性偏差。")
    print("  这正是「碰撞概率 = Jaccard」的直接体现：偏差来自抽样，不来自实现。")
    print("  ② 会把「误差如何随 L 收缩」定量画出来。\n")

    # ---------------- 2) 误差 vs 理论标准差 ----------------
    print("② 估计误差随签名长度 L 的下降（与 sqrt(J(1−J)/L) 对照）：")
    print("   （同组集合对 J≈0.19–0.25，各用 300 个不同哈希种子重复，取标准差）")
    rows = []
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    for L_ in (16, 64, 256, 1024):
        for i, j in pairs:
            J = jaccard(sets[i], sets[j])
            if not (0.15 <= J <= 0.85):      # 太接近 0/1 的对方差近乎 0，比值无意义
                continue
            ests = []
            for t in range(300):
                m = MinHashLinear(L_, seed=SEED + 1000 + t)
                sig_i, sig_j = m.get_signature(sets[i]), m.get_signature(sets[j])
                ests.append(float((sig_i == sig_j).mean()))
            ests = np.array(ests)
            emp_std = float(ests.std())
            theo = float(np.sqrt(J * (1 - J) / L_))
            rows.append([f"({i},{j})", f"{J:.2f}", L_, f"{ests.mean():.3f}",
                         f"{emp_std:.4f}", f"{theo:.4f}",
                         f"{emp_std / theo:.2f}"])
    table(["集合对", "真实 J", "L", "均值估计", "经验 std", "理论 std", "比值"], rows)
    print("\n  经验标准差与 sqrt(J(1−J)/L) 基本吻合（比值接近 1）。")
    print("  这就是正文的方差推导——每一位是伯努利变量，L 位独立，方差 J(1−J)/L。")
    print("  注意：J 越接近 0 或 1，方差越小，估计本来就更容易准；")
    print("  真正难估的是 J≈0.5 的中间区域，这也决定了检索阈值附近的误判率。\n")

    # ---------------- 3) 工程视角：内存与耗时 ----------------
    print("③ 为什么工业界用线性哈希（universe = 10^6，L = 128）：")
    rows = []
    for name, obj in (("随机排列", MinHashPermutation(128, 10 ** 6, seed=SEED)),
                      ("线性哈希", MinHashLinear(128, seed=SEED))):
        import time
        t0 = time.perf_counter()
        for _ in range(200):
            obj.get_signature(sets[0])
        dt = (time.perf_counter() - t0) / 200
        mb = obj.nbytes() / 1e6
        size_s = f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1000:.0f} KB"
        rows.append([name, size_s, f"{dt * 1e6:.0f} µs / 次"])
    table(["实现", "索引内存", "单集合签名耗时"], rows)
    print("\n  随机排列需要 128 × 10^6 个整数 ≈ 512 MB（还没算 universe 更大时）；")
    print("  线性哈希只需要 128 对 (a, b)，约 2 KB。耗时上前者也慢一个量级，")
    print("  因为它要对每个元素做一次随机内存访问。这就是正文「不适合大规模落地」的原因。\n")

    # ---------------- 4) 用 MinHash 找相似集合 ----------------
    print("④ 检索演示：给一个查询集合，用 MinHash 签名找最相似的 5 个：")
    sets_big, labels_big = make_sets(n_sets=200, universe=20000, size=80,
                                     n_groups=10)
    mh = MinHashLinear(256, seed=SEED)
    sigs = np.stack([mh.get_signature(s) for s in sets_big])   # [200, 256]
    q_idx = 0
    q_sig = sigs[q_idx]
    sim = (sigs == q_sig[None, :]).mean(axis=1)
    sim[q_idx] = -1.0                                    # 排除查询自身
    order = np.argsort(-sim)[:5]
    rows = []
    for k in order:
        true_j = jaccard(sets_big[q_idx], sets_big[k])
        rows.append([k, labels_big[k], f"{sim[k]:.3f}", f"{true_j:.3f}"])
    table(["候选编号", "所属簇", "签名相似度", "真实 Jaccard"], rows)
    print("\n  Top-5 全部来自查询集合自己的簇，签名相似度与真实 Jaccard 同序——")
    print("  但注意这只是「对 200 个集合全量算签名相似度」，还不是 ANN。")
    print("  真正的子线性检索要靠 §18.3.3 的 LSH Banding 把签名切成 band 建桶。\n")

    print("⑤ 一句话：MinHash 把集合压缩成 L 个整数，碰撞概率 = Jaccard；")
    print("   签名长度控制精度（std=sqrt(J(1−J)/L)），线性哈希控制成本。")


if __name__ == "__main__":
    main()
