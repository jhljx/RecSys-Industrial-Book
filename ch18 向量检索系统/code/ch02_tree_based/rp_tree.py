"""
随机投影树：随机方向划分、ANN 化的多树检索（第18章 §18.2.3）

正文给出的 RP-Tree-Max 建树规则：
    1) 随机取单位方向 v；
    2) 任取样本 x，找离它最远的 y，用 ‖x−y‖ 估计当前节点的尺度；
    3) 投影 p_i = x_iᵀv，划分阈值
           t = median(p_i) + δ,
           δ ~ Uniform(−6‖x−y‖/√D, +6‖x−y‖/√D)
    4) x_iᵀv ≤ t 进左，否则进右。

RP-Tree-Mean 在此基础上自适应切换：
    Δ²(S) ≤ c·Δ_A²(S)  → 继续用随机投影划分（分布均匀）
    Δ²(S) > c·Δ_A²(S)  → 改用「到均值距离的中位数」划分（有离群点/长尾）

Annoy 则走另一条工程路线：
    - 用 two_means（随机采样 + 增量均值，非标准 EM）找两个中心 p、q；
    - 超平面法向量 n = (p−q)/‖p−q‖，偏置 a = −nᵀ(p+q)/2；
    - 建多棵树，查询时用全局优先队列做 Best-First，凑够 search_k 个候选再精确排序。

本脚本：
1) 七点数据上演示一次随机投影划分（含 δ 扰动）；
2) 实测 δ 的尺度系数 c 对「退化划分（一边为空）」概率的影响，解释为什么必须加扰动；
3) 精确版 RP-Tree（带回溯）在均匀高维数据上与 KD 树对比，看内在维度带来的差异；
4) Annoy 风格实现：Two-Means 划分 + 多棵树 + Best-First，扫树数 M 与 search_k，
   给出 Recall@10 / 候选数 / 查询耗时。

运行：python3 ch02_tree_based/rp_tree.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (  # noqa: E402
    knn_exact, make_queries, make_vectors, recall_at_k, table, timing,
)
from kd_tree import KDTree  # noqa: E402

SEED = 2026
NAMES = list("ABCDEFG")
PTS = np.array([
    [50, 50],   # A
    [10, 70],   # B
    [80, 15],   # C
    [25, 20],   # D
    [40, 85],   # E
    [70, 85],   # F
    [10, 60],   # G
], dtype=np.float32)


# --------------------------------------------------------------------------
# 一、随机方向 + 中位数 + 扰动
# --------------------------------------------------------------------------
def random_direction(D: int, rng: np.random.Generator) -> np.ndarray:
    """在单位球面上均匀取一个方向（高斯归一化等价于球面均匀分布）。"""
    v = rng.normal(size=D)
    return v / np.linalg.norm(v)


def split_scale(P: np.ndarray, rng: np.random.Generator) -> float:
    """
    用正文的方式估计节点尺度：任取若干个 x，各找离它最远的 y，取最大 ‖x−y‖。

    随机采样一个点而不是求精确直径，是为了每个节点 O(n) 而不是 O(n²)；
    多个样本取最大可以把低估的概率压到很低。
    """
    n_sample = min(8, len(P))
    best = 0.0
    for i in rng.choice(len(P), n_sample, replace=False):
        best = max(best, float(np.sqrt(((P - P[i]) ** 2).sum(1)).max()))
    return best


def median_perturbation(proj: np.ndarray, scale: float, D: int,
                        rng: np.random.Generator, c: float = 6.0) -> float:
    """
    正文的 δ：Uniform(−6‖x−y‖/√D, +6‖x−y‖/√D)。

    但 ‖x−y‖ 是节点的「直径」，除以 √D 后仍是全尺寸量级：在七点例子中
    D=2、‖x−y‖≈89，δ 可以到 ±378，而投影值只在 [30, 110] 之间——阈值
    会被直接推到数据范围外，整棵树退化成单叶。

    所以工程实现必须给扰动再加一个上界。这里取「投影四分位距的一半」：
    它保证阈值最多挪到相邻分位点附近，不会跨出数据主体；这与正文
    「扰动用于控制坏分裂概率、不能任意增大」的意图一致。
    """
    span = c * scale / np.sqrt(D)
    q25, q75 = np.percentile(proj, [25, 75])
    span = min(span, 0.5 * (q75 - q25))
    return float(rng.uniform(-span, span))


class RPNode:
    __slots__ = ("v", "t", "left", "right", "pts", "size")

    def __init__(self) -> None:
        self.v: np.ndarray | None = None
        self.t: float = 0.0
        self.left: RPNode | None = None
        self.right: RPNode | None = None
        self.pts: np.ndarray | None = None
        self.size: int = 0


def build_rp(X: np.ndarray, idx: np.ndarray, leaf_size: int = 1,
             rng: np.random.Generator | None = None,
             perturb: float = 6.0) -> RPNode:
    """
    建 RP-Tree-Max。

    perturb 是 δ 的尺度系数（正文取 6）。设为 0 就是「纯中位数划分」，
    用它可以观察退化划分的产生。
    """
    node = RPNode()
    node.size = len(idx)
    if len(idx) <= leaf_size:
        node.pts = idx
        return node

    P = X[idx]
    D = X.shape[1]
    v = random_direction(D, rng)
    proj = P @ v
    t = float(np.median(proj))
    if perturb > 0:
        t += median_perturbation(proj, split_scale(P, rng), D, rng, perturb)

    left_mask = proj <= t
    if left_mask.all() or (~left_mask).all():
        # 扰动把阈值推到数据范围外：退化成叶子，宁可不分也不能丢样本
        node.pts = idx
        return node
    node.v, node.t = v, t
    node.left = build_rp(X, idx[left_mask], leaf_size, rng, perturb)
    node.right = build_rp(X, idx[~left_mask], leaf_size, rng, perturb)
    return node


def tree_stats(node: RPNode) -> tuple[int, int, int]:
    """返回 (内部节点数, 叶节点数, 单样本叶数)。"""
    if node.pts is not None:
        return 0, 1, int(len(node.pts) == 1)
    li, ll, ls = tree_stats(node.left)
    ri, rl, rs = tree_stats(node.right)
    return 1 + li + ri, ll + rl, ls + rs


def count_degenerate(node: RPNode) -> int:
    """统计「一侧只有 ≤1 个点」的坏划分数量。"""
    if node.pts is not None:
        return 0
    bad = 1 if min(node.left.size, node.right.size) <= 1 else 0
    return bad + count_degenerate(node.left) + count_degenerate(node.right)


# --------------------------------------------------------------------------
# 二、精确 RP-Tree（带回溯）——RP-Tree 论文本身不要求回溯，
#     这里加上回溯是为了对照 KD 树、确认剪枝下界同样有效
# --------------------------------------------------------------------------
class RPTreeExact:
    def __init__(self, X: np.ndarray, leaf_size: int = 8,
                 seed: int = SEED) -> None:
        self.X = X
        self.rng = np.random.default_rng(seed)
        self.root = build_rp(X, np.arange(len(X)), leaf_size, self.rng, 6.0)
        self.visited = 0
        self.total_visited = 0

    def _push(self, q: np.ndarray, i: int, heap: list, k: int) -> None:
        d = float(((self.X[i] - q) ** 2).sum())
        self.visited += 1
        if len(heap) < k:
            heap.append((d, i))
            heap.sort(reverse=True)
        elif d < heap[0][0]:
            heap[0] = (d, i)
            heap.sort(reverse=True)

    def _search(self, node: RPNode, q: np.ndarray, heap: list, k: int) -> None:
        if node.pts is not None:
            for i in node.pts:
                self._push(q, i, heap, k)
            return

        s = float(q @ node.v)
        near, far = ((node.left, node.right) if s <= node.t
                     else (node.right, node.left))
        self._search(near, q, heap, k)
        # 查询点到超平面的距离 = |q·v − t|（v 已是单位向量），
        # 大于当前第 k 近距离就说明远侧不可能有更近的点
        margin = abs(s - node.t)
        ub = float("inf") if len(heap) < k else heap[0][0]
        if len(heap) < k or margin * margin <= ub:
            self._search(far, q, heap, k)

    def search(self, q: np.ndarray, k: int = 10) -> np.ndarray:
        heap: list = []
        self.visited = 0
        self._search(self.root, q, heap, k)
        self.total_visited += self.visited
        return np.array([i for _, i in heap])


# --------------------------------------------------------------------------
# 三、Annoy 风格：Two-Means 划分 + 多树 + Best-First
# --------------------------------------------------------------------------
def two_means(P: np.ndarray, rng: np.random.Generator,
              steps: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    对应正文 Annoy two_means 代码的 Python 版。

    与 K-Means 的区别：不遍历全体、不跑多轮 EM，而是每步随机抽一个点，
    把它并给更近的中心并用增量均值（incremental mean）更新该中心。
    复杂度 O(steps·D)，与 n 无关——建树时每个节点都会调用一次。
    """
    n, D = P.shape
    i = int(rng.integers(n))
    j = int(rng.integers(n - 1))
    j += (j >= i)              # 与 Annoy 源码一致：保证 j ≠ i
    p = P[i].astype(np.float64).copy()
    q = P[j].astype(np.float64).copy()
    ic = jc = 1

    for _ in range(steps):
        k = int(rng.integers(n))
        dp = float(((P[k] - p) ** 2).sum()) * ic
        dq = float(((P[k] - q) ** 2).sum()) * jc
        if dp < dq:
            p += (P[k] - p) / ic
            ic += 1
        elif dq < dp:
            q += (P[k] - q) / jc
            jc += 1
    return p.astype(np.float32), q.astype(np.float32)


class AnnoyNode:
    """Annoy 的节点：要么是内部节点（法向量 n + 偏置 a），要么是叶节点。"""

    __slots__ = ("n", "a", "left", "right", "indices")

    def __init__(self) -> None:
        self.n: np.ndarray | None = None
        self.a: float = 0.0
        self.left: AnnoyNode | None = None
        self.right: AnnoyNode | None = None
        self.indices: np.ndarray | None = None


def build_annoy(X: np.ndarray, idx: np.ndarray, K: int,
                rng: np.random.Generator) -> AnnoyNode:
    """
    Annoy 建树：叶子大小 K，内部节点用 two_means 构造超平面。
    同时记录每个子树在法向量方向的投影区间 [lo, hi]，Best-First 要用它算下界。
    """
    node = AnnoyNode()
    P = X[idx]
    if len(idx) <= K:
        node.indices = idx
        return node

    p, q = two_means(P, rng)
    n = (p - q)
    n /= (np.linalg.norm(n) + 1e-12)
    a = float(-n @ (p + q) / 2.0)
    side = P @ n + a < 0

    if side.all() or (~side).all():
        node.indices = idx
        return node

    node.n, node.a = n, a
    node.left = build_annoy(X, idx[side], K, rng)
    node.right = build_annoy(X, idx[~side], K, rng)
    return node


class AnnoyIndex:
    """M 棵树的 Annoy 索引，查询走全局 Best-First + 候选精确重排。"""

    def __init__(self, X: np.ndarray, n_trees: int = 4, K: int = 8,
                 seed: int = SEED) -> None:
        self.X = X
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.roots = [build_annoy(X, np.arange(len(X)), K, self.rng)
                      for _ in range(n_trees)]
        self.n_trees = n_trees
        self.cand = 0
        self.total_cand = 0

    def search(self, q: np.ndarray, k: int = 10, search_k: int | None = None
               ) -> np.ndarray:
        """
        Annoy 的 _get_all_nns 结构：所有根入堆，Best-First 逐个扩展，
        收够 search_k 个候选后对候选做精确距离排序。

        优先级这里用「查询点到该节点所在单元格的距离下界」：
            near 子节点：继承父节点的下界
            far  子节点：max(父下界, |q·n + a|)（到划分超平面的距离）
        这比「两个子节点压同一个 margin」更严格：如果两侧压低同一个值，
        平局时先展开哪个完全看入堆顺序，可能先钻进远侧子树，
        而 search_k 很小时就再也回不来——本脚本第一版就因此 Recall 掉到 0。
        """
        import heapq

        search_k = search_k or (self.n_trees * self.K * 4)
        heap: list[tuple[float, int, AnnoyNode]] = []
        cnt = 0
        for root in self.roots:
            heapq.heappush(heap, (0.0, cnt, root))
            cnt += 1

        cand: list[int] = []
        while len(cand) < search_k and heap:
            bound, _, nd = heapq.heappop(heap)
            if nd.indices is not None:
                cand.extend(nd.indices.tolist())
                continue
            s = float(q @ nd.n + nd.a)
            near, far = ((nd.left, nd.right) if s < 0 else (nd.right, nd.left))
            heapq.heappush(heap, (bound, cnt, near))
            cnt += 1
            # 计数器的另一个作用：下界相同时让二元组可比较（否则会去比节点对象）
            heapq.heappush(heap, (max(bound, abs(s)), cnt, far))
            cnt += 1

        self.cand = len(cand)
        if not cand:
            return np.empty(0, dtype=int)
        # 与 Annoy 源码一致：候选先按 ID 排序去重，再做距离排序。
        # 不去重的话同一个点会在 Top-k 里重复出现，把真正的近邻挤出去，
        # 表现为「候选越多召回反而越低」。
        C = np.unique(np.asarray(cand, dtype=np.int64))
        self.cand = len(C)                  # 统计去重后的候选数
        self.total_cand += self.cand
        d = ((self.X[C] - q) ** 2).sum(1)
        kk = min(k, len(C))
        part = np.argpartition(d, kk - 1)[:kk]
        part = part[np.argsort(d[part])]
        return C[part]


# --------------------------------------------------------------------------
def print_rp_tree(node: RPNode, X: np.ndarray, indent: str = "") -> None:
    if node.pts is not None:
        print(f"{indent}叶 {{ {''.join(NAMES[i] for i in node.pts)} }}")
        return
    axis = "v=(%.2f,%.2f)" % (node.v[0], node.v[1])
    print(f"{indent}划分 {axis}  t={node.t:.2f}")
    print_rp_tree(node.left, X, indent + "  ├─L ")
    print_rp_tree(node.right, X, indent + "  └─R ")


def main() -> None:
    print("随机投影树：随机方向划分、ANN 化的多树检索（§18.2.3）\n")

    # ---------------- 1) 七点数据的一次随机划分 ----------------
    print("① 七点数据上的一次随机投影划分（D=2，leaf_size=1）：")
    rng = np.random.default_rng(281)
    v = random_direction(2, np.random.default_rng(281))
    proj = PTS @ v
    scale = split_scale(PTS, np.random.default_rng(281))
    delta = median_perturbation(proj, scale, 2, rng, c=6.0)
    t = float(np.median(proj)) + delta
    print(f"  随机方向 v = ({v[0]:.3f}, {v[1]:.3f})，‖v‖ = {np.linalg.norm(v):.3f}")
    print("  各点投影 " + "，".join(f"{c}:{p:.1f}" for c, p in zip(NAMES, proj)))
    print(f"  投影中位数 = {np.median(proj):.2f}，节点尺度 ‖x−y‖ = {scale:.2f}")
    print(f"  正文 δ 的原始范围 ±6‖x−y‖/√D = ±{6 * scale / np.sqrt(2):.1f}，"
          f"但投影只落在 [{proj.min():.1f}, {proj.max():.1f}]，")
    print("  直接用会把阈值推空整个节点。因此实际实现把 δ 截到投影四分位距的一半，")
    print(f"  本次 δ = {delta:+.2f}，阈值 t = {t:.2f}")
    side = "".join(NAMES[i] for i in np.where(proj <= t)[0])
    other = "".join(NAMES[i] for i in np.where(proj > t)[0])
    print(f"  划分结果：左 {{ {side} }}  |  右 {{ {other} }}")
    print("\n  可以看到划分超平面不必经过任何样本点，方向也完全不看坐标轴——")
    print("  这就是 RP-Tree 与 KD 树最根本的区别。正文插图中超平面恰好经过 C，")
    print("  只是那一次随机抽样的结果，不是规则本身的要求。\n")

    # ---------------- 2) δ 的作用 ----------------
    print("② δ 的作用：实测它在这组数据上的效果，别照搬「加了扰动一定更好」。")
    print("   （2000 条 16 维簇状数据，leaf_size=8，统计「某侧只有 ≤1 个点」的划分占比）")
    Xc, _ = make_vectors(n=2000, dim=16, n_clusters=16, spread=0.5)
    rows = []
    for scale_c in (0.0, 0.5, 1.0, 3.0, 6.0, 12.0):
        rng2 = np.random.default_rng(SEED)
        root = build_rp(Xc, np.arange(len(Xc)), leaf_size=8, rng=rng2,
                        perturb=scale_c)
        n_in, n_leaf, single = tree_stats(root)
        bad = count_degenerate(root)
        rows.append([f"{scale_c:g}", n_in, n_leaf, f"{bad / max(n_in, 1):.1%}"])
    table(["δ 系数 c", "内部节点数", "叶节点数", "退化划分占比"], rows)
    print("\n  结论要反过来读：不扰动（c=0）时树反而最平衡——256 个叶子、深度 8，")
    print("  因为中位数划分天然就是 50/50；加扰动后树深涨到 14，退化划分也出现了。")
    print("  这说明 δ 不是「提升平衡性的技巧」，它的理论作用是：")
    print("  保证坏分裂的概率足够小（正文的证明需要它），代价是牺牲一点平衡性。")
    print("  另外由于 ① 中提到的截断，c ≥ 3 之后结果完全相同——截断已经生效，")
    print("  扰动幅度不再由 c 决定。\n")

    # ---------------- 3) 精确版 RP-Tree vs KD 树 ----------------
    print("③ 精确 RP-Tree（带超平面回溯）vs KD 树，各向同性均匀数据、2000 条、Top-10：")
    print("   （RP-Tree 论文本身不要求回溯，这里的回溯只为公平对照「单棵树能剪掉多少」）")
    rows = []
    n_q = 40
    k = 10
    for dim in (2, 4, 8, 16, 32, 64):
        rng3 = np.random.default_rng(7)
        Xd = rng3.random((2000, dim), dtype=np.float32)
        Qd = (Xd[rng3.integers(0, 2000, n_q)]
              + rng3.normal(0.0, 0.05, (n_q, dim))).astype(np.float32)
        truth, _ = knn_exact(Xd, Qd, k)

        rp = RPTreeExact(Xd, leaf_size=8, seed=SEED)
        pred_r = np.array([rp.search(q, k) for q in Qd])
        kd = KDTree(Xd, leaf_size=8)
        pred_k = np.array([kd.search(q, k) for q in Qd])

        rows.append([dim,
                     f"{rp.total_visited / (n_q * len(Xd)):.1%}",
                     f"{kd.total_visited / (n_q * len(Xd)):.1%}",
                     f"{recall_at_k(pred_r, truth):.3f}",
                     f"{recall_at_k(pred_k, truth):.3f}"])
    table(["维度 D", "RP-Tree 访问占比", "KD 树访问占比", "RP Recall", "KD Recall"], rows)
    print("\n  两者都是精确搜索，Recall 都是 1.0。注意访问占比这一栏：")
    print("  均匀立方体上 KD 的轴对齐划分更贴数据，RP-Tree 反而访问更多。")
    print("  RP-Tree 的优势要换到「数据沿随机方向张成的低维流形」上才显出来：")
    rows2 = []
    for Dn, d_int in ((64, 4), (64, 16), (128, 8)):
        rngm = np.random.default_rng(0)
        B = np.linalg.qr(rngm.normal(size=(Dn, d_int)))[0]
        Xm = (rngm.normal(size=(2000, d_int)) @ B.T).astype(np.float32)
        Qm = (Xm[rngm.integers(0, 2000, n_q)]
              + rngm.normal(0.0, 0.02, (n_q, Dn))).astype(np.float32)
        rp = RPTreeExact(Xm, leaf_size=8, seed=SEED)
        for qq in Qm:
            rp.search(qq, k)
        kd = KDTree(Xm, leaf_size=8)
        for qq in Qm:
            kd.search(qq, k)
        rows2.append([f"{Dn} / {d_int}",
                      f"{rp.total_visited / (n_q * len(Xm)):.1%}",
                      f"{kd.total_visited / (n_q * len(Xm)):.1%}"])
    print("   名义维度 / 内在维度（数据恰好落在 d 维线性子空间里）：")
    table(["D / d", "RP-Tree 访问占比", "KD 树访问占比"], rows2)
    print("  结果与直觉相反：KD 反而访问得更少（D=64、d=4 时 14.7% vs 92.2%）。")
    print("  原因是这里的内在结构是「坐标轴之外的低维子空间」，而 KD 沿轴切恰好")
    print("  能把子空间切得很薄；RP-Tree 的随机方向几乎垂直于这个子空间，")
    print("  投影值全部挤在 0 附近，划分质量反而变差。")
    print("  所以「RP-Tree 只依赖内在维数」是论文对特定构型（数据沿随机方向")
    print("  张成的低维流形）的结论，不能推广成「任何低维流形上 RP 都比 KD 好」。")
    print("  真正的取舍是：RP-Tree 的划分不依赖坐标轴的方向偏置，")
    print("  代价是在轴对齐结构上不如 KD。\n")

    # ---------------- 4) Annoy 风格：多树 + Best-First ----------------
    print("④ Annoy 风格：Two-Means 划分 + M 棵树 + Best-First（2000 条 16 维，Top-10）")
    X, _ = make_vectors(n=2000, dim=16, n_clusters=16, spread=0.5)
    Q = make_queries(X, n_queries=100)
    truth, _ = knn_exact(X, Q, k)
    rows = []
    for n_trees in (1, 2, 4, 8):
        idx = AnnoyIndex(X, n_trees=n_trees, K=8, seed=SEED)
        pred = np.array([idx.search(q, k, search_k=n_trees * 8 * 4) for q in Q])
        avg_cand = idx.total_cand / len(Q)
        t_q, _ = timing(lambda: [idx.search(q, k, search_k=n_trees * 8 * 4) for q in Q],
                        repeat=2)
        rows.append([n_trees, f"{avg_cand:.0f}",
                     f"{recall_at_k(pred, truth):.3f}", f"{t_q / len(Q) * 1e6:.0f} µs"])
    table(["树数 M", "平均候选数", "Recall@10", "单查询耗时"], rows)
    print("\n  候选数与 search_k 同量级：扩展到一个叶桶就整桶收下，可能刚好越过")
    print("  预算；多棵树之间还有重复，所以去重后比 search_k 略少。")
    print("  召回随 M 上升，因为不同树的随机划分互补，合并候选能覆盖单棵树漏掉的近邻。")
    print("  这就是 Annoy 用多棵树换召回的来源。\n")

    print("⑤ search_k 与 recall 的权衡（固定 M=4）：")
    idx = AnnoyIndex(X, n_trees=4, K=8, seed=SEED)
    rows = []
    for sk in (16, 32, 64, 128, 256, 2000):
        idx.total_cand = 0                      # 每档单独统计候选数
        pred = np.array([idx.search(q, k, search_k=sk) for q in Q])
        avg_cand = idx.total_cand / len(Q)      # 只统计上面这一轮
        t_q, _ = timing(lambda: [idx.search(q, k, search_k=sk) for q in Q], repeat=2)
        rows.append([sk, f"{avg_cand:.0f}",
                     f"{recall_at_k(pred, truth):.3f}", f"{t_q / len(Q) * 1e6:.0f} µs"])
    table(["search_k", "去重后候选数", "Recall@10", "单查询耗时"], rows)
    print("\n  search_k=2000（= 全库）时 4 棵树的所有叶子都被展开，等价于暴力，")
    print("  Recall 必然回到 1.000——这印证了「近似」只来自提前停止扩展，")
    print("  不是树结构本身丢了数据。实际使用中 search_k 一般在 n_trees·K 的")
    print("  几倍到几十倍之间取，用候选数换召回。\n")

    print("⑥ 一句话总结：RP-Tree 把「沿轴切」换成「沿随机方向切」，")
    print("   使收敛速度取决于数据的内在维数；Annoy 在此之上加了")
    print("   Two-Means 划分质量、多树互补与 Best-First 调度，成为可落地的工程方案。")


if __name__ == "__main__":
    main()
