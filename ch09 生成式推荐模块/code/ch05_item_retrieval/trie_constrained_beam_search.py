"""对应正文第 9 章 §9.5「物品ID检索」中的 Trie 前缀约束 Beam Search。

正文要点:
  * 生成式模型按自回归方式逐层生成语义 ID: P(SID) = prod_{i=1}^{L} P(s_i | s_<i, u),
    实际用对数概率累加作为路径得分 Score(SID) = sum_{i=1}^{L} log P(s_i | s_<i, u),
    并据此排序剪枝; 为减轻长度偏置常使用长度归一化后的得分;
  * 若不加约束, 自回归可能生成"无效语义 ID(Invalid SID)", 即在映射表中找不到物品;
  * 对层次化语义 ID 可构建 Trie 树进行前缀检索, 在每一步只允许 Trie 中存在的子节点,
    从而保证生成结果必然落在合法物品空间内。

本实现:
  1. 用合成的物品语义 ID 表构建 Trie(前缀树);
  2. 用一个 [MOCK] 打分器(固定随机权重的小型自回归打分函数, 不训练)提供 logits;
  3. 对比「无约束 Beam Search」与「Trie 前缀约束 Beam Search」的非法率/有效候选数;
  4. 打印 Trie 结构统计与两种方式的 top-5 路径得分表。
"""

from __future__ import annotations

import math
import random

import numpy as np


# ------------------------------------------------------------------ Trie
class TrieNode:
    __slots__ = ("children", "items")

    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.items: list[int] = []        # 落在该完整路径上的物品 ID(可能多个)


class SidTrie:
    """语义 ID 前缀树: 支持 allowed(prefix) 查询下一层合法码字集合。"""

    def __init__(self, sid_table: dict[int, tuple[int, ...]]):
        self.root = TrieNode()
        self.n_node = 1
        for item, sid in sid_table.items():
            node = self.root
            for c in sid:
                if c not in node.children:
                    node.children[c] = TrieNode()
                    self.n_node += 1
                node = node.children[c]
            node.items.append(item)

    def allowed(self, prefix: tuple[int, ...]) -> list[int]:
        node = self.root
        for c in prefix:
            node = node.children.get(c)
            if node is None:
                return []
        return sorted(node.children)

    def lookup(self, sid: tuple[int, ...]) -> list[int]:
        node = self.root
        for c in sid:
            node = node.children.get(c)
            if node is None:
                return []
        return node.items

    def depth_fanout(self, L: int) -> list[tuple[int, int, float]]:
        """返回每层 (节点数, 最大分支, 平均分支)。"""
        stats, frontier = [], [self.root]
        for _ in range(L):
            fan = [len(n.children) for n in frontier]
            nxt = [c for n in frontier for c in n.children.values()]
            stats.append((len(frontier), max(fan) if fan else 0,
                          sum(fan) / max(len(fan), 1)))
            frontier = nxt
        return stats


# ------------------------------------------------- [MOCK] 自回归打分器
class MockScorer:
    """[MOCK] 教学用打分器: 不训练, 用固定随机矩阵模拟 Transformer 的下一层 logits。

    每一步 logits = W[layer] @ h, 其中 h 由用户向量与已生成前缀 embedding 累加得到,
    仅用于提供"稳定、可复现、带偏好结构"的概率分布, 以便对比两种 Beam Search。
    """

    def __init__(self, L: int, K: int, dim: int, rng: np.random.Generator):
        self.L, self.K, self.dim = L, K, dim
        self.W = [rng.normal(0, 1.0, size=(K, dim)) for _ in range(L)]
        self.tok = [rng.normal(0, 0.5, size=(K, dim)) for _ in range(L)]

    def log_probs(self, user: np.ndarray, prefix: tuple[int, ...]) -> np.ndarray:
        h = user.copy()
        for l, c in enumerate(prefix):
            h = h + self.tok[l][c]
        z = self.W[len(prefix)] @ h / math.sqrt(self.dim)
        z = z - z.max()
        p = np.exp(z)
        return np.log(p / p.sum())


# ------------------------------------------------------------ Beam Search
def beam_search(scorer: MockScorer, user: np.ndarray, beam: int,
                trie: SidTrie | None, length_norm: bool = False):
    """Score(SID) = sum_i log P(s_i|s_<i,u); trie 非 None 时做前缀约束。"""
    beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for step in range(scorer.L):
        cand = []
        for prefix, score in beams:
            lp = scorer.log_probs(user, prefix)
            allow = range(scorer.K) if trie is None else trie.allowed(prefix)
            if not allow:
                continue
            for c in allow:
                cand.append((prefix + (c,), score + float(lp[c])))
        if not cand:
            return []
        key = (lambda t: -t[1] / (len(t[0]) if length_norm else 1))
        beams = sorted(cand, key=key)[:beam]
    return beams


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    L, K, DIM = 3, 8, 16
    N_ITEM = 220
    # 合成语义 ID 表: 故意让部分物品共享同一语义 ID(正文的"多对一映射"问题)
    sid_table = {}
    for i in range(N_ITEM):
        sid_table[i] = tuple(int(c) for c in rng.integers(0, K, size=L))
    trie = SidTrie(sid_table)

    sid2items: dict[tuple[int, ...], list[int]] = {}
    for it, s in sid_table.items():
        sid2items.setdefault(s, []).append(it)

    print("=" * 86)
    print(f"语义 ID 空间 K^L = {K}^{L} = {K ** L}; 物品数 = {N_ITEM}; "
          f"互不相同的语义 ID = {len(sid2items)}")
    print(f"合法路径占全空间比例 = {len(sid2items) / K ** L:.1%}; "
          f"Trie 节点数 = {trie.n_node}")
    print()
    print("表 1  Trie 各层结构统计")
    print("-" * 86)
    print(f"{'层 l':>6}{'该层节点数':>12}{'最大分支':>10}{'平均分支':>10}")
    for l, (n, mx, avg) in enumerate(trie.depth_fanout(L), 1):
        print(f"{l:>6}{n:>12}{mx:>10}{avg:>10.2f}")

    scorer = MockScorer(L, K, DIM, rng)
    users = rng.normal(0, 1.0, size=(200, DIM))

    print()
    print("表 2  无约束 Beam Search vs. Trie 前缀约束 Beam Search(200 个用户)")
    print("-" * 86)
    print(f"{'beam':>6}{'无约束非法率':>14}{'无约束有效候选':>16}"
          f"{'约束非法率':>14}{'约束有效候选':>16}")
    for beam in (5, 10, 20, 50):
        f_illegal = f_valid = c_illegal = c_valid = 0
        for u in users:
            free = beam_search(scorer, u, beam, None)
            bad = sum(1 for s, _ in free if s not in sid2items)
            f_illegal += bad
            f_valid += len(free) - bad
            con = beam_search(scorer, u, beam, trie)
            cbad = sum(1 for s, _ in con if s not in sid2items)
            c_illegal += cbad
            c_valid += len(con) - cbad
        tot = beam * len(users)
        print(f"{beam:>6}{f_illegal / tot:>14.1%}{f_valid / len(users):>16.2f}"
              f"{c_illegal / tot:>14.1%}{c_valid / len(users):>16.2f}")
    print("-" * 86)
    print("结论: 无约束 Beam Search 的大部分路径是 Invalid SID(正文所述问题),")
    print("      Trie 前缀约束把搜索空间限制在合法前缀上, 非法率恒为 0。")

    # ----------------- 单用户 top-5 对比 -----------------
    u = users[0]
    print()
    print("表 3  单个用户的 top-5 路径(beam=10)")
    print("-" * 86)
    free = beam_search(scorer, u, 10, None)[:5]
    con = beam_search(scorer, u, 10, trie)[:5]
    print(f"{'rank':>5}  {'无约束 SID':<16}{'score':>9}{'合法?':>7}   "
          f"{'约束 SID':<16}{'score':>9}{'命中物品数':>10}")
    for r in range(5):
        fs, fv = free[r]
        cs, cv = con[r]
        print(f"{r + 1:>5}  {str(fs):<16}{fv:>9.3f}"
              f"{('是' if fs in sid2items else '否'):>7}   "
              f"{str(cs):<16}{cv:>9.3f}{len(sid2items[cs]):>10}")

    # ------------- 长度归一化对得分的影响 -------------
    print()
    print("表 4  长度归一化(正文: 减轻长度偏置)对 top-1 路径的影响, beam=10")
    print("-" * 86)
    print(f"{'用户':>5}{'原始得分 top1':>22}{'归一化得分 top1':>24}{'是否同一路径':>14}")
    for i in range(5):
        a = beam_search(scorer, users[i], 10, trie)[0]
        b = beam_search(scorer, users[i], 10, trie, length_norm=True)[0]
        print(f"{i:>5}{str(a[0]) + f' ({a[1]:.3f})':>22}"
              f"{str(b[0]) + f' ({b[1] / L:.3f})':>24}"
              f"{('是' if a[0] == b[0] else '否'):>14}")
    print("-" * 86)
    print("注: L 固定时长度归一化只是整体缩放, 不改变排序; 正文强调它在变长生成")
    print("    (如同时生成多个物品的 SID 串)场景才真正影响排序。")


if __name__ == "__main__":
    main()
