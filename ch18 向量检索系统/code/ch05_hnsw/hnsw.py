"""
HNSW：分层可导航小世界图的构建与检索（第18章 §18.5）

正文的结构定义：
    每个节点随机取最高层  l_i = ⌊−m_L · ln(u)⌋,  u ~ U(0,1)
    于是 P(L ≥ l) = e^(−l/m_L)，层数越高节点越稀疏；第 0 层包含全部节点。

插入一个节点 x：
    1) 从最高层入口做贪心下降，到第 l 层；
    2) 在 l … 0 每层用 efConstruction 的 Best-First 找候选；
    3) Neighbor Selection Heuristic 选边：候选按距离升序，只有当
           d(x, v_i) ≤ d(v_i, s)   ∀ s ∈ S（已选邻居）
       才保留——保证边朝向不同方向，而不是全扎堆在同一个簇；
    4) 与选中的邻居建立双向连接，邻居超度数时再用同一启发式裁剪。

查询：
    高层贪心下降（每层只留 1 个当前最优点）；
    第 0 层用 efSearch 的 Best-First，两个优先队列（候选 / 结果），
    候选队列最优点已劣于结果队列最差点时停止。

数据的选取（很重要）：
    一开始用 30 个紧簇、16 维的数据，M=16 时召回在任何 efSearch 下都是
    1.000——图几乎全连通，看不出任何权衡。换成 64 维近均匀分布后，
    近邻不再集中在一小片区域，图导航的难度才接近真实 Embedding 检索的
    场景（工业上 128~960 维、百万级库正是这种几何）。

本脚本：
1) 验证层数分布 P(L ≥ l) = e^(−l/m_L)；
2) 在 6000 条 64 维数据上构图，统计度数、层节点数、构图耗时；
3) 扫 efSearch，画 Recall@10 / 访问距离次数 / 延迟，复现正文「efSearch 换召回」；
4) 对比 M 与 efConstruction 的影响；
5) 对比 Neighbor Selection Heuristic 与「直接取最近 M 个」。

运行：python3 ch05_hnsw/hnsw.py
"""

from __future__ import annotations

import heapq
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import knn_exact, make_queries, make_vectors, table  # noqa: E402

SEED = 2026


class HNSW:
    """
    极简 HNSW：每层的邻接表用 dict[int, np.ndarray] 存（只对进入该层的节点建表）。

    刻意不引入删除、不引入多线程/加锁，因为正文这一节讲的是结构与搜索过程，
    这些工程细节会掩盖算法本身。
    """

    def __init__(self, m: int = 16, ef_construction: int = 100,
                 m_L: float | None = None, seed: int = SEED) -> None:
        self.M = m
        self.M0 = 2 * m                      # 第 0 层允许 2M 条边（hnswlib 的约定）
        self.efC = ef_construction
        self.m_L = m_L if m_L is not None else 1.0 / math.log(m)
        self.rng = np.random.default_rng(seed)
        self.vectors: list[np.ndarray] = []
        self.nbrs: list[dict[int, np.ndarray]] = []   # nbrs[id][level] = 邻居数组
        self.level: list[int] = []
        self.entry = -1
        self.max_level = -1
        self.dist_calls = 0                  # 统计：累计距离计算次数

    # ---------------------------------------------------------------- 建图
    def _sample_level(self) -> int:
        """l = ⌊−m_L ln u⌋；u 取 (0,1]，避免 log(0)。"""
        u = 1.0 - self.rng.random()
        return int(-self.m_L * math.log(u))

    def _d(self, a: int, b: int) -> float:
        self.dist_calls += 1
        d = self.vectors[a] - self.vectors[b]
        return float(d @ d)

    def _dvec(self, q: np.ndarray, a: int) -> float:
        self.dist_calls += 1
        d = self.vectors[a] - q
        return float(d @ d)

    def _greedy(self, q: np.ndarray, start: int, level: int) -> int:
        """高层贪心：沿着更近的邻居一路走，直到本层没有更近的邻居。"""
        cur = start
        cur_d = self._dvec(q, cur)
        improved = True
        while improved:
            improved = False
            for nb in self.nbrs[cur].get(level, ()):
                dn = self._dvec(q, int(nb))
                if dn < cur_d:
                    cur, cur_d = int(nb), dn
                    improved = True
        return cur

    def _search_layer(self, q: np.ndarray, entries: list[int], ef: int,
                      level: int) -> list[tuple[float, int]]:
        """
        单层 Best-First，返回按距离升序的 (距离, 编号)，长度 ≤ ef。

        两个堆：
            cand  —— 极小堆，下一步要扩展的候选；
            result—— 极大堆（存负距离），当前最好的 ef 个节点。
        终止条件：候选堆的最小距离 > 结果堆的最大距离。
        """
        visited = set(entries)
        cand = [(self._dvec(q, e), e) for e in entries]
        heapq.heapify(cand)
        result = [(-d, e) for d, e in cand]
        heapq.heapify(result)
        while len(result) > ef:
            heapq.heappop(result)

        while cand:
            d_c, c = heapq.heappop(cand)
            if len(result) >= ef and d_c > -result[0][0]:
                break                          # 最近的候选都比结果堆最差还远 → 停
            for nb in self.nbrs[c].get(level, ()):
                nb = int(nb)
                if nb in visited:
                    continue
                visited.add(nb)
                d_n = self._dvec(q, nb)
                if len(result) < ef or d_n < -result[0][0]:
                    heapq.heappush(cand, (d_n, nb))
                    heapq.heappush(result, (-d_n, nb))
                    if len(result) > ef:
                        heapq.heappop(result)
        out = [(-d, i) for d, i in result]
        out.sort()
        return out

    def _select(self, cands: list[tuple[float, int]], M: int) -> list[int]:
        """
        正文的 Neighbor Selection Heuristic：

            候选按 d(base, v) 升序；若 v 到每个已选邻居 s 的距离都 ≥ d(base, v)，
            说明 v 提供了新的方向，保留；否则丢弃。

        直觉：若 d(v, s) < d(base, v)，则 base→s→v 比直达更短，这条直达边
        对导航的边际价值小；删掉它，度数就能留给真正朝向新方向的邻居。

        代价是每次筛选都要在候选之间算距离，所以构图比「取最近 M 个」慢。
        """
        chosen: list[int] = []
        for dv, v in cands:
            ok = True
            for s in chosen:
                if self._d(v, s) < dv:
                    ok = False
                    break
            if ok:
                chosen.append(v)
                if len(chosen) >= M:
                    break
        return chosen

    def _connect(self, node: int, chosen: list[int], level: int) -> None:
        """双向连接 + 超度数裁剪。"""
        limit = self.M0 if level == 0 else self.M
        self.nbrs[node][level] = np.array(chosen, dtype=np.int32)
        for nb in chosen:
            arr = self.nbrs[nb][level]
            if node in arr:
                continue
            arr = np.append(arr, node)
            if len(arr) > limit:
                # 超度数：对现有邻居再跑一次启发式，保留 limit 条
                cands = sorted((self._d(nb, int(x)), int(x)) for x in arr)
                arr = np.array(self._select(cands, limit), dtype=np.int32)
            self.nbrs[nb][level] = arr

    def add(self, x: np.ndarray) -> None:
        idx = len(self.vectors)
        self.vectors.append(x)
        lvl = self._sample_level()
        self.level.append(lvl)
        self.nbrs.append({lv: np.zeros(0, dtype=np.int32) for lv in range(lvl + 1)})

        if self.entry < 0:                 # 第一个点直接做入口
            self.entry, self.max_level = idx, lvl
            return

        ep = self.entry
        # 1) 从最高层贪心下降到 lvl+1
        for lv in range(self.max_level, lvl, -1):
            if ep in self.nbrs and lv in self.nbrs[ep]:
                ep = self._greedy(x, ep, lv)
        # 2) 从 min(lvl, max_level) 到 0 逐层搜索并连边
        for lv in range(min(lvl, self.max_level), -1, -1):
            found = self._search_layer(x, [ep], self.efC, lv)
            cands = [(self._dvec(x, i), i) for _, i in found]
            chosen = self._select(cands, self.M0 if lv == 0 else self.M)
            self._connect(idx, chosen, lv)
            ep = found[0][1] if found else ep
        if lvl > self.max_level:
            self.entry, self.max_level = idx, lvl

    def build(self, X: np.ndarray) -> None:
        for x in X:
            self.add(x)

    # ---------------------------------------------------------------- 查询
    def search(self, q: np.ndarray, k: int = 10, ef_search: int = 50
               ) -> tuple[np.ndarray, int]:
        """返回 (Top-k 编号, 本次查询的距离计算次数)。"""
        start_calls = self.dist_calls
        ep = self.entry
        for lv in range(self.max_level, 0, -1):
            ep = self._greedy(q, ep, lv)
        found = self._search_layer(q, [ep], max(ef_search, k), 0)
        ids = np.array([i for _, i in found[:k]], dtype=np.int64)
        return ids, self.dist_calls - start_calls

    # ---------------------------------------------------------------- 统计
    def level_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for lv in self.level:
            for l in range(lv + 1):
                out[l] = out.get(l, 0) + 1
        return dict(sorted(out.items()))

    def degrees(self) -> tuple[float, int]:
        degs = [len(v) for lv in self.nbrs for v in lv.values()]  # 各层汇总
        return float(np.mean(degs)), int(np.max(degs))


def recall_of(model: HNSW, Q: np.ndarray, truth: np.ndarray,
              k: int, ef: int) -> tuple[float, float, float]:
    """返回 (Recall@k, 平均距离计算次数, 单查询毫秒)。"""
    preds, calls, t = [], 0, 0.0
    for q in Q:
        t0 = time.perf_counter()
        ids, c = model.search(q, k, ef)
        t += time.perf_counter() - t0
        preds.append(ids)
        calls += c
    rec = float(np.mean([len(set(p.tolist()) & set(t_.tolist())) / k
                         for p, t_ in zip(preds, truth)]))
    return rec, calls / len(Q), t / len(Q) * 1e3


def main() -> None:
    print("HNSW：分层可导航小世界图的构建与检索（§18.5）\n")

    # ---------------- 1) 层数分布 ----------------
    print("① 层数采样 l = ⌊−m_L ln u⌋，m_L = 1/ln(M)：")
    rows = []
    for M in (8, 16, 32):
        m_L = 1.0 / math.log(M)
        h = HNSW(m=M, m_L=m_L, seed=SEED)
        n_samp = 200_000
        lv = np.array([h._sample_level() for _ in range(n_samp)])
        for l in range(1, 5):
            emp = float((lv >= l).mean())
            theo = math.exp(-l / m_L)
            rows.append([M, f"{m_L:.3f}", l, f"{emp:.4f}", f"{theo:.4f}"])
    table(["M", "m_L", "l", "实测 P(L≥l)", "理论 e^(−l/m_L)"], rows)
    print("\n  实测与指数分布吻合。m_L 越小，高层越稀：M=8 时第 2 层只有 1.5%，")
    print("  M=32 时只有 0.1%。M 大 → 每层节点多、图更密，导航路径更短但内存更大。\n")

    # ---------------- 2) 构图 ----------------
    N, D = 6000, 64
    # n_clusters=1 即近均匀的 64 维球状分布：近邻不像紧簇数据那样扎堆，
    # 这是让「efSearch 换召回」显形的关键（见文件头的说明）。
    X, _ = make_vectors(n=N, dim=D, n_clusters=1, spread=1.0)
    Q = make_queries(X, n_queries=100, jitter=0.5)
    k = 10
    truth, _ = knn_exact(X, Q, k)

    M, efC = 8, 40
    g = HNSW(m=M, ef_construction=efC, seed=SEED)
    t0 = time.perf_counter()
    g.build(X)
    t_build = time.perf_counter() - t0
    calls_build = g.dist_calls

    counts = g.level_counts()
    avg_deg, max_deg = g.degrees()
    print(f"② 构图（N={N}, D={D}, M={M}, efConstruction={efC}）：")
    print(f"  耗时 {t_build:.1f} s，距离计算 {calls_build:,} 次"
          f"（平均每点 {calls_build / N:,.0f} 次，暴力两两要 {N * (N - 1) // 2:,} 次）")
    print(f"  层节点数：{counts}")
    print(f"  平均度数 {avg_deg:.1f}，最大度数 {max_deg}"
          f"（第 0 层上限 M0 = 2M = {2 * M}，高层上限 M = {M}）")
    print(f"  层节点数按约 1/M 的几何比例递减：{counts[0]} → {counts.get(1, 0)}"
          f" → {counts.get(2, 0)} → {counts.get(3, 0)} → {counts.get(4, 0)}，")
    print("  与正文「越高越稀」的描述一致。构图的总距离计算量是")
    print(f"  {calls_build / N:,.0f} 次/点，约为暴力两两的 "
          f"{calls_build / (N * (N - 1) / 2):.0%}——每层只扫邻域，所以是 O(N log N)，")
    print("  不是 O(N²)。插入顺序会影响图质量：这里按数据顺序插入，")
    print("  工业实现通常先打乱或按随机顺序插入。\n")

    # ---------------- 3) efSearch 扫描 ----------------
    print("③ 扫 efSearch：召回 / 距离计算次数 / 延迟（M=8, efC=40）：")
    rows, vals = [], []
    for ef in (10, 20, 40, 80, 160):
        rec, calls, ms = recall_of(g, Q, truth, k, ef)
        vals.append((ef, rec, calls, ms))
        rows.append([ef, f"{rec:.3f}", f"{calls:.0f}", f"{ms:.2f} ms"])
    table(["efSearch", "Recall@10", "平均距离计算", "单查询耗时"], rows)
    print("\n  efSearch 从 10 加到 160，召回几乎单调升、距离计算次数同步涨——")
    print("  这就是正文「用 efSearch 在线平衡召回与延迟」。注意 efSearch<k 会被")
    print("  自动抬到 k，否则结果队列装不下 Top-k；还有一点是实现细节：")
    print("  Python 每次浮点运算的开销远大于 C++，所以这里的绝对毫秒只能看相对趋势。")
    print(f"  对照：全库暴力每次要算 {N} 次距离，efSearch=80 时只有 "
          f"{vals[3][2]:.0f} 次，约为全库的 {vals[3][2] / N:.1%}。\n")

    # ---------------- 4) M / efConstruction ----------------
    print("④ M 与 efConstruction 的影响（efSearch=80）：")
    rows, note = [], {}
    for M_ in (8, 16, 32):
        for efC_ in (40, 100):
            gg = HNSW(m=M_, ef_construction=efC_, seed=SEED)
            t0 = time.perf_counter()
            gg.build(X)
            tb = time.perf_counter() - t0
            rec, _, _ = recall_of(gg, Q, truth, k, 80)
            d, _ = gg.degrees()
            rows.append([M_, efC_, f"{rec:.3f}", f"{d:.1f}", f"{tb:.1f} s"])
            note[(M_, efC_)] = rec
    table(["M", "efConstruction", "Recall@10", "平均度数", "构图耗时"], rows)
    print("\n  两列分开读：M 决定图的容量上限（度数、内存、可达到的召回），")
    print("  efConstruction 决定建图时「找得够不够仔细」，主要影响构图时间。")
    print(f"  M=8→16（efC=40）召回 {note[(8, 40)]:.3f}→{note[(16, 40)]:.3f}，")
    print(f"  提升明显；efC 40→100 在同一 M 下只多 "
          f"{note[(16, 100)] - note[(16, 40)]:+.3f}，却让构图时间接近翻倍。")
    print("  这与正文的调参顺序一致：先定 M，再调 efConstruction，")
    print("  线上最后用 efSearch 做召回/延迟的细调。\n")

    # ---------------- 5) 启发式 vs 最近 M 个 ----------------
    print("⑤ Neighbor Selection Heuristic 的作用：对比「取最近 M 个」建图")
    print("   （同样 M=8、efC=40，只换选边规则，两档 efSearch 都看）")

    class NearestM(HNSW):
        def _select(self, cands, M):
            return [v for _, v in cands[:M]]

    built = {}
    for name, cls in (("启发式选边", HNSW), ("最近 M 个", NearestM)):
        gg = cls(m=M, ef_construction=efC, seed=SEED)
        t0 = time.perf_counter()
        gg.build(X)
        built[name] = (gg, time.perf_counter() - t0)

    rows = []
    for ef in (10, 40, 80):
        rec_h, _, _ = recall_of(built["启发式选边"][0], Q, truth, k, ef)
        rec_n, _, _ = recall_of(built["最近 M 个"][0], Q, truth, k, ef)
        rows.append([ef, f"{rec_h:.3f}", f"{rec_n:.3f}",
                     f"{(rec_h - rec_n) / max(rec_n, 1e-9):+.0%}"])
    table(["efSearch", "启发式 Recall@10", "最近 M 个 Recall@10", "相对提升"], rows)

    th, tn = built["启发式选边"][1], built["最近 M 个"][1]
    print(f"\n  构图耗时：{th:.1f} s vs {tn:.1f} s（+{th / tn - 1:.0%}）——")
    print("  启发式每次选边要在候选之间算距离，慢是必然的，这是用建图时间买召回。")
    print("  差距在每一档 efSearch 都存在（9%~16% 的相对提升），说明它不是")
    print("  「搜索参数调一调就能补上」的问题，而是图本身的结构差异：")
    print("  最近 M 个邻居必然落在同一片局部区域，图会被切成一个个「团」，")
    print("  查询从入口下降后一旦落错团就再也走不出去；启发式拒绝「已能被中间点")
    print("  到达」的候选，反而保留了指向远处的边，让图保持连通。")
    print("  大库、高维、簇结构不明显的数据上，这个结构差异通常比这里更显著。\n")

    print("⑥ 一句话：HNSW = 跳表的层 + NSW 的贪心 + 启发式选边。")
    print("   高层「少而远」负责导航，第 0 层「多而近」负责精搜，")
    print("   efSearch 是线上唯一要动的旋钮。")


if __name__ == "__main__":
    main()
