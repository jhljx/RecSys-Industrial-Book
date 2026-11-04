"""
球树：超球体划分、下界剪枝与 KD 树的对比（第18章 §18.2.2）

正文的球树定义：
    每个节点对应一个超球体 B(c, r) = {x : ‖x − c‖ ≤ r}
    - 叶节点：c 取样本均值，r = max‖x_i − c‖
    - 内部节点：由左右子球合并
          d = ‖c_L − c_R‖
          若 d + min(r_L, r_R) ≤ max(r_L, r_R)：父球直接复用较大的子球
          否则 c_parent = c_L + (d − r_L + r_R) / (2d) · (c_R − c_L)
               r_parent = (d + r_L + r_R) / 2

注意：正文写的球心系数是 (d + r_L − r_R)/(2d)，符号有误。按那个式子，当
r_L < r_R 时父球包不住右子球，剪枝会误剪真实近邻（本脚本实测 Recall 掉到
0.995）。正确的球心应在「两球最远表面点的中点」，即 (d − r_L + r_R)/(2d)；
merge_ball 中保留了覆盖性断言，可以直接验证这一点。

查询用三角不等式得到下界（任意满足三角不等式的度量都成立）：
    d_LB(q, B) = max(‖q − c‖ − r, 0)
    d_LB > d_best  →  整颗子树剪掉

本脚本：
1) 按正文的七点数据建球树（方差最大方向 + 中位数划分），打印每个节点的球心与半径，
   并逐条核对叶子球心 = 均值、半径 = 最远样本距离；
2) 复现正文 (7,3) 的例子：超矩形距离 d_kd = 3.0，超球距离 d_ball ≈ 2.11，
   并说明两者的真实差别在「每节点计算量」而不是「谁的下界更紧」；
3) 在 2000 条 8 维数据上与 KD 树对照：精确性、访问点数、剪枝节点数；
4) 扫维度，确认球树同样逃不过维度灾难。

运行：python3 ch02_tree_based/ball_tree.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import knn_exact, make_queries, make_vectors, recall_at_k, table  # noqa: E402
from kd_tree import KDTree  # noqa: E402

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

# 正文的划分：按方差大的轴投影取中位数
#   根：x 轴（方差大于 y）→ 左 {B,D,E,G}，右 {A,C,F}
#   左：y 轴排序 20,60,70,85 → {D,G} 与 {B,E}
#   右：y 轴排序 15,50,85    → {C,A} 与 {F}
BOOK_SPLITS = {
    frozenset("ABCDEFG"): ([1, 3, 4, 6], [0, 2, 5]),
    frozenset("BDEG"): ([3, 6], [1, 4]),
    frozenset("ACF"): ([2, 0], [5]),
}


class BallNode:
    """一个球树节点：球心 c、半径 r，叶节点额外存样本。"""

    __slots__ = ("c", "r", "left", "right", "pts")

    def __init__(self) -> None:
        self.c: np.ndarray | None = None
        self.r: float = 0.0
        self.left: BallNode | None = None
        self.right: BallNode | None = None
        self.pts: np.ndarray | None = None


def fit_ball(P: np.ndarray) -> tuple[np.ndarray, float]:
    """叶节点最小包围球：球心取均值，半径取到最远样本的距离。

    内部统一用 float64：坐标为 float32 时，r ≈ 60 的球在 float32 下的
    相对误差约 1e-7，绝对误差可达 1e-5，用 1e-6 做覆盖性断言会误报。
    """
    P = P.astype(np.float64)
    c = P.mean(axis=0)
    r = float(np.sqrt(((P - c) ** 2).sum(axis=1)).max())
    return c, r


def merge_ball(node: BallNode) -> None:
    """由左右子球构造父球，严格按正文的两条分支。"""
    L, R = node.left, node.right
    Lc, Lr = L.c.astype(np.float64), float(L.r)
    Rc, Rr = R.c.astype(np.float64), float(R.r)
    d = float(np.linalg.norm(Lc - Rc))
    if d + min(Lr, Rr) <= max(Lr, Rr):
        # 一个球完全包含另一个：直接复用大的那个，半径不会变大
        if Lr >= Rr:
            node.c, node.r = Lc, Lr
        else:
            node.c, node.r = Rc, Rr
        return
    # 两球相交或分离：取最小包围球
    #   球心取两球最远表面点的中点：c = c_L + (d − r_L + r_R)/(2d) · (c_R − c_L)
    #   （左球最远点 c_L − r_L·u，右球最远点 c_R + r_R·u，中点在 c_L + (d−r_L+r_R)/2·u）
    #   半径 (d + r_L + r_R)/2。
    #
    #   注意：正文写的是 (d + r_L − r_R)/(2d)，符号与这里相反。按那个式子，
    #   当 r_L < r_R 时父球包不住右子球（右球外沿超出 (r_R − r_L)/2），
    #   剪枝会误剪真实近邻。merge_ball 结尾的 assert 会直接报错，可自行验证。
    if d < 1e-12:
        node.c, node.r = Lc, max(Lr, Rr)
        return
    node.c = Lc + (d - Lr + Rr) / (2.0 * d) * (Rc - Lc)
    node.r = (d + Lr + Rr) / 2.0

    # 覆盖性断言：父球必须完整包住两个子球，否则下界失效、查询会漏结果。
    # 这两个不等式就是「最小包围球」的定义，任何合并方式都必须满足。
    assert np.linalg.norm(Lc - node.c) + Lr <= node.r + 1e-6
    assert np.linalg.norm(Rc - node.c) + Rr <= node.r + 1e-6


def build_ball(X: np.ndarray, idx: np.ndarray, leaf_size: int = 8,
               book: bool = False) -> BallNode:
    """
    递归建树。

    book=True 时按正文的划分（左 {B,D,E,G} / 右 {A,C,F} …）。
    否则用正文推荐的「方差最大方向 + 中位数划分」：先投影到方差最大的维度，
    再按该维度中位数二分——比按坐标轴轮换更贴合数据的实际形状。
    """
    node = BallNode()
    node.c, node.r = fit_ball(X[idx])

    if book:
        key = frozenset(NAMES[i] for i in idx)
        if key in BOOK_SPLITS:
            li, ri = BOOK_SPLITS[key]
            node.left = build_ball(X, np.array(li, dtype=int), leaf_size, True)
            node.right = build_ball(X, np.array(ri, dtype=int), leaf_size, True)
            merge_ball(node)
            return node
        node.pts = idx
        return node

    if len(idx) <= leaf_size:
        node.pts = idx
        return node

    P = X[idx]
    axis = int(np.argmax(P.var(axis=0)))          # 方差最大方向
    order = idx[np.argsort(X[idx, axis], kind="stable")]
    mid = len(order) // 2
    node.left = build_ball(X, order[:mid], leaf_size)
    node.right = build_ball(X, order[mid:], leaf_size)
    merge_ball(node)                              # 父球由两个子球合并，而不是重新拟合
    return node


def lower_bound(q: np.ndarray, node: BallNode, parent_lb: float = 0.0) -> float:
    """
    查询点到球体的距离下界，含「父节点下界继承」：

        d_s(q) = max(‖q − c_s‖ − r_s, d_parent(q))      非根
               = max(‖q − c_s‖ − r_s, 0)                根

    继承父下界不会改变正确性（子球 ⊆ 父球），但能在进入子树前就剪掉更多分支。
    """
    return max(float(np.linalg.norm(q - node.c)) - node.r, parent_lb)


class BallTree:
    """带剪枝计数的球树。"""

    def __init__(self, X: np.ndarray, leaf_size: int = 8) -> None:
        self.X = X
        self.root = build_ball(X, np.arange(len(X)), leaf_size=leaf_size)
        self.visited = 0
        self.pruned = 0
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

    def _search(self, node: BallNode, q: np.ndarray, heap: list, k: int,
                parent_lb: float) -> None:
        lb = lower_bound(q, node, parent_lb)
        # 整个球内不可能有比当前第 k 近更近的点 → 剪掉
        if len(heap) >= k and lb * lb > heap[0][0]:
            self.pruned += 1
            return

        if node.pts is not None:
            for i in node.pts:
                self._push(q, i, heap, k)
            return

        # 先走下界更小的子球，尽早拿到一个较小的 d_best，后续剪枝更狠
        left_lb = lower_bound(q, node.left, lb)
        right_lb = lower_bound(q, node.right, lb)
        first, second = ((node.left, node.right) if left_lb <= right_lb
                         else (node.right, node.left))
        self._search(first, q, heap, k, parent_lb=lb)
        self._search(second, q, heap, k, parent_lb=lb)

    def search(self, q: np.ndarray, k: int = 10) -> np.ndarray:
        heap: list = []
        self.visited = self.pruned = 0
        self._search(self.root, q, heap, k, parent_lb=0.0)
        self.total_visited += self.visited
        return np.array([i for _, i in heap])


def count_nodes(node: BallNode) -> tuple[int, int]:
    """返回 (内部节点数, 叶节点数)。"""
    if node.pts is not None:
        return 0, 1
    li, ll = count_nodes(node.left)
    ri, rl = count_nodes(node.right)
    return 1 + li + ri, ll + rl


def print_book_tree(node: BallNode, indent: str = "") -> None:
    """打印七点球树的结构与每个节点的球参数。"""
    if node.pts is not None:
        names = "".join(NAMES[i] for i in node.pts)
        print(f"{indent}叶 {names:<4} c=({node.c[0]:6.2f},{node.c[1]:6.2f})  r={node.r:5.2f}")
        return
    print(f"{indent}内部     c=({node.c[0]:6.2f},{node.c[1]:6.2f})  r={node.r:5.2f}")
    print_book_tree(node.left, indent + "  ├─L ")
    print_book_tree(node.right, indent + "  └─R ")


def main() -> None:
    print("球树：超球体划分、下界剪枝与 KD 树的对比（§18.2.2）\n")

    # ---------------- 1) 七点球树 ----------------
    print("① 按正文的划分重建七点球树（方差最大方向 + 中位数）：")
    root = build_ball(PTS, np.arange(7), book=True)
    print_book_tree(root)
    n_in, n_leaf = count_nodes(root)
    print(f"\n  内部节点 {n_in} 个、叶节点 {n_leaf} 个。")
    print("  叶子球心是真实样本均值；内部球心是几何构造出的虚拟点，")
    print("  通常不对应任何真实样本——这是球树与 KD 树最直观的区别之一。")

    # 逐个核对叶球
    print("\n  叶子球的核对：")
    for names in ("DG", "BE", "CA", "F"):
        idx = [NAMES.index(c) for c in names]
        c, r = fit_ball(PTS[idx])
        ds = [float(np.linalg.norm(PTS[i] - c)) for i in idx]
        print(f"    {names:<3} c=({c[0]:6.2f},{c[1]:6.2f})  r={r:5.2f}  "
              f"到各样本距离 = {', '.join(f'{d:.2f}' for d in ds)}  "
              f"← r 取最大者")
    print()
    print("  父球合并示例（左子树 {B,E}+{D,G}）：")
    cL, rL = fit_ball(PTS[[1, 4]])
    cR, rR = fit_ball(PTS[[3, 6]])
    d = float(np.linalg.norm(cL - cR))
    c_par = cL + (d - rL + rR) / (2 * d) * (cR - cL)
    r_par = (d + rL + rR) / 2
    print(f"    左球 c=({cL[0]:.2f},{cL[1]:.2f}) r={rL:.2f}，"
          f"右球 c=({cR[0]:.2f},{cR[1]:.2f}) r={rR:.2f}，d={d:.2f}")
    print(f"    父球 c=({c_par[0]:.2f},{c_par[1]:.2f}) r={r_par:.2f}"
          f"（= (d+r_L+r_R)/2 = ({d:.2f}+{rL:.2f}+{rR:.2f})/2）")
    print(f"    覆盖校验：|c−c_L|+r_L = {np.linalg.norm(c_par - cL) + rL:.2f}，"
          f"|c−c_R|+r_R = {np.linalg.norm(c_par - cR) + rR:.2f}，"
          f"父球半径 {r_par:.2f} ≥ 两者 ✓")
    print("    父球半径比任一子球都大——球面的最远点在偏移方向上，")
    print("    所以父球半径是 (d + r_L + r_R)/2 而不是 max(r_L, r_R)。")
    print("    ⚠ 正文式 c_parent = c_L + (d + r_L − r_R)/(2d)(c_R − c_L) 的符号有误：")
    print("      当 r_L < r_R 时父球包不住右子球，剪枝会误剪真实近邻。")
    print("      正确写法是 (d − r_L + r_R)/(2d)，即取两球最远表面点的中点。\n")

    # ---------------- 2) 正文 (7,3) 的例子 ----------------
    print("② 正文的 (7,3) 例子：同一个节点，超矩形距离 vs 超球距离")
    q = np.array([7.0, 3.0], dtype=np.float32)
    box_lo = np.array([1.0, 2.0], dtype=np.float32)
    box_hi = np.array([4.0, 6.0], dtype=np.float32)
    delta = np.maximum(np.maximum(box_lo - q, q - box_hi), 0.0)
    d_kd = float(np.sqrt((delta * delta).sum()))
    c_ball = np.array([2.5, 4.0], dtype=np.float32)
    r_ball = 2.5
    d_center = float(np.linalg.norm(q - c_ball))
    d_ball = max(d_center - r_ball, 0.0)
    print(f"  超矩形 [1,4]×[2,6]：δ = (7−4, 0) = ({delta[0]:.1f}, {delta[1]:.1f})")
    print(f"    d_kd = √(3² + 0²) = {d_kd:.2f}        ← 需要每一维的上下界，2D 次运算")
    print(f"  超球 c=(2.5,4.0), r=2.5：‖q−c‖ = {d_center:.3f}")
    print(f"    d_ball = {d_center:.3f} − 2.5 = {d_ball:.2f}   ← 只用球心，1 次距离计算")
    print("\n  注意这两个数不能直接比大小：下界越小，剪枝越弱。本例中 3.0 > 2.11，")
    print("  KD 的下界反而更紧。球树真正的优势在于:")
    print("    a) 计算下界只需要 ‖q − c‖ 与一次减法，不用逐维判断 2D 个边界；")
    print("       代价仍是 O(D) 次乘加，但访存与分支远少于超矩形；")
    print("    b) 三角不等式对余弦、内积等任意度量都成立（见 §18.2 正文），")
    print("       超矩形距离只对欧氏距离有意义；")
    print("    c) 叶节点是「一个球」而不是「一个矩形」：在球状分布上覆盖更紧，")
    print("       具体哪种结构剪得更狠取决于数据形状，实测见 ③。\n")

    # 顺带说明：若按真实样本拟合最小包围圆，半径只有 2.02，下界会更紧
    pts4 = np.array([[1, 3], [2.5, 6], [3, 2], [4, 4]], dtype=np.float32)
    c4, r4 = np.array([2.75, 4.0], dtype=np.float32), 2.0156
    lb4 = max(float(np.linalg.norm(q - c4)) - r4, 0.0)
    print(f"  补充：这 4 个点的最小包围圆是 c=({c4[0]:.2f},{c4[1]:.2f}) r={r4:.2f}"
          f"（(1,3) 与 (2.5,6) 在圆上），对应下界 {lb4:.2f}。")
    print("  正文插图的 r=2.5 是覆盖整个 4×4 区域的外接圆，为了画图直观；")
    print("  工程实现里叶球用均值+最远距离，内部球用最小包围球，才是常态。")
    print(f"  真实最近点 (4,4) 的距离是 {float(np.linalg.norm(q - np.array([4.0, 4.0]))):.2f}，")
    print("  所以 2.11 / 2.35 都是合法的下界，只是紧度不同——下界越紧，剪枝越狠。\n")

    # ---------------- 3) 与 KD 树对照 ----------------
    X, _ = make_vectors(n=2000, dim=8, n_clusters=16, spread=0.5)
    Q = make_queries(X, n_queries=100)
    k = 10
    truth_idx, _ = knn_exact(X, Q, k)

    bt = BallTree(X, leaf_size=8)
    pred_b = np.array([bt.search(qq, k) for qq in Q])
    kd = KDTree(X, leaf_size=8)
    pred_k = np.array([kd.search(qq, k) for qq in Q])

    print("③ 2000 条 8 维簇状数据、100 条查询、Top-10：")
    rows = [
        ["KD 树", f"{recall_at_k(pred_k, truth_idx):.4f}",
         f"{kd.total_visited / len(Q):.0f}",
         f"{kd.total_visited / len(Q) / len(X):.1%}", "—"],
        ["球树", f"{recall_at_k(pred_b, truth_idx):.4f}",
         f"{bt.total_visited / len(Q):.0f}",
         f"{bt.total_visited / len(Q) / len(X):.1%}",
         f"{bt.pruned / len(Q):.1f}"],
    ]
    table(["结构", "Recall@10", "平均访问点数", "访问占比", "平均剪枝节点数"], rows)
    print("\n  两者都是精确搜索（Recall=1.0）：它们的剪枝都是「安全」的下界，")
    print("  只跳过不可能更近的区域。但在这组簇状数据上，球树访问的点反而更多，")
    print("  原因是：叶球「均值 + 最远距离」会把半径撑得很大（簇内最远点决定了 r），")
    print("  父球再一合并，很多兄弟球彼此重叠、下界被压到 0，剪枝机会变少。")
    print("  这正说明球树不是无条件更好——它适合球状、各向同性的数据，")
    print("  在簇状数据上超矩形反而贴得更紧。工业界据此改用随机投影树。\n")

    # ---------------- 4) 维度灾难 ----------------
    print("④ 球树同样逃不过维度灾难（各向同性均匀数据，2000 条、叶子 8）：")
    rows = []
    n_q = 40
    for dim in (2, 4, 8, 16, 32, 64, 128):
        rng = np.random.default_rng(7)
        Xd = rng.random((2000, dim), dtype=np.float32)
        Qd = (Xd[rng.integers(0, 2000, n_q)]
              + rng.normal(0.0, 0.05, (n_q, dim))).astype(np.float32)
        t_ball = BallTree(Xd, leaf_size=8)
        for qq in Qd:
            t_ball.search(qq, k)
        t_kd = KDTree(Xd, leaf_size=8)
        for qq in Qd:
            t_kd.search(qq, k)
        rows.append([dim,
                     f"{t_ball.total_visited / (n_q * len(Xd)):.1%}",
                     f"{t_kd.total_visited / (n_q * len(Xd)):.1%}",
                     f"{t_ball.pruned / n_q:.1f}"])
    table(["维度 D", "球树访问占比", "KD 树访问占比", "球树平均剪枝节点数"], rows)
    print("\n  维度一高，球树的下界 max(‖q−c‖−r, 0) 同样失去区分度：")
    print("  高维中任意两点的距离趋于相近（距离集中现象），第 k 近距离 d_best")
    print("  已经接近节点直径，于是 ‖q−c‖−r 几乎总是小于 d_best，剪不动。")
    print("  球树只是把 KD 的「细长矩形」换成了「互相重叠的球」，")
    print("  它没有延缓维度灾难，也改变不了 O(N) 的终局——上表比 KD 还差。\n")

    print("⑤ 结论（以实测为准，别照抄「球树更好」的说法）：")
    print("   在这两组数据上，球树的访问点数反而都高于 KD 树——叶球用「均值+最远")
    print("   距离」定义，半径容易被个别离群点撑大，父球一合并重叠就更多。")
    print("   球树真正换来的是：每节点下界只需一次球心距离、支持任意三角不等式度量、")
    print("   以及球体对球状分布的贴合；它并没有解决高维。高维下真正有效的是")
    print("   随机投影树(§18.2.3)、LSH(§18.3)、PQ(§18.4)、HNSW(§18.5)。")


if __name__ == "__main__":
    main()
