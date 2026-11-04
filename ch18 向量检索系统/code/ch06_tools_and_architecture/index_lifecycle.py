"""
索引生命周期与双缓冲一致性：把「召回率下降」拆回工程问题（第18章 §18.7）

正文的两个论断，本脚本都做成可跑的模拟：

1) 一篇候选集合只有约 100 个 Item 的业务，用全量 ANN 召回（k=100）
   理应返回 100 条，但线上只返回 80 多条。ANN 召回没有任何过滤逻辑，
   所以这不是算法漏召回，而是部分 Item 没进到底层索引。

2) 库有几千万条时，同样比例的缺失对 Top-K 和 A/B 指标几乎无感，
   Bug 会长期潜伏。

模拟里刻意只注入两类真实故障：
    A. 全量构建时有批次异常退出（部分 Item 未写入新索引）；
    B. 构建期间新 Item 写入旧索引，而增量回放（Replay）只覆盖了一部分。
不引入 ANN 误差——用精确暴力索引，让「缺失」完全来自工程流程。

验证手段按正文列的四种来做：
    索引条数校验 / Embedding 表与索引集合差集 / 切换后抽样自检索 / 覆盖率监控

运行：python3 ch06_tools_and_architecture/index_lifecycle.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import make_queries, make_vectors, table  # noqa: E402

SEED = 2026


class FlatIndex:
    """暴力索引（IndexFlat 的等价物）：只为排除 ANN 误差的干扰。"""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.vecs: list[np.ndarray] = []

    def add(self, ids, vecs) -> None:
        self.ids.extend(int(i) for i in ids)
        self.vecs.extend(np.asarray(vecs))

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        X = np.stack(self.vecs)
        d = ((X - q) ** 2).sum(axis=1)
        k = min(k, len(d))
        return np.array(self.ids)[np.argsort(d)[:k]]


def build_new_index(X: np.ndarray, ids: np.ndarray, drop_rate: float,
                    rng: np.random.Generator, per_batch: bool = True) -> FlatIndex:
    """
    模拟离线全量构建：按批（每批 512 条）写入。

    per_batch=True  ——「整批异常退出」：任务重试没做好的常见形态，
                       缺失集中在整批；批次开得大时，一个异常就是几百上千条。
    per_batch=False ——「单条写入失败」：逐条丢，缺失更分散。

    两种故障的差别只影响缺失的聚集程度，不影响后面的结论。
    """
    idx = FlatIndex()
    batch = 512
    for s in range(0, len(X), batch):
        e = min(s + batch, len(X))
        if per_batch:
            if rng.random() < drop_rate:
                continue                  # 该批写入失败，异常被上层吞掉
            idx.add(ids[s:e], X[s:e])
        else:                             # 逐条判定
            for j in range(s, e):
                if rng.random() >= drop_rate:
                    idx.add(ids[j:j + 1], X[j:j + 1])
    return idx


def coverage(index: FlatIndex, probe_ids: np.ndarray,
             probe_vecs: np.ndarray) -> float:
    """自检索覆盖率：用 Item 自己的向量查询，Top-1 是不是它自己。"""
    have = set(index.ids)
    hit = 0
    for i, v in zip(probe_ids, probe_vecs):
        if int(i) not in have:
            continue                      # 索引里根本没有 → 必然召回不到
        top = index.search(v, 1)
        hit += int(len(top) > 0 and top[0] == i)
    return hit / len(probe_ids)


def main() -> None:
    print("索引生命周期与双缓冲一致性（§18.7）\n")
    rng = np.random.default_rng(SEED)

    # ---------------- 1) 小候选集：100 条少了几条才看得见 ----------------
    print("① 候选集只有 100 条：全量 ANN（k=100）本应返回 100 条")
    Xs, _ = make_vectors(n=100, dim=16, n_clusters=5, spread=0.3)
    ids_s = np.arange(100)
    # 这里用「逐条丢失」：100 条只够一个批次，按批丢会变成 0 或全丢
    small = build_new_index(Xs, ids_s, drop_rate=0.10, rng=rng, per_batch=False)
    Qs = make_queries(Xs, n_queries=100, jitter=0.0)
    returned = np.array([len(small.search(q, 100)) for q in Qs])
    rows = [
        ["应返回", "100", "100%", "业务侧认为候选池是满的"],
        ["实际返回", f"{returned.mean():.1f}", f"{returned.mean():.1f}%",
         "索引里只有 %d 条" % len(small)],
        ["最少的一次", f"{returned.min()}", f"{returned.min():.0f}%",
         "线上告警通常从这里开始"],
    ]
    table(["口径", "条数", "占应返回", "说明"], rows)
    print("\n  这就是正文描述的现象：召回没有任何过滤，返回数却不够。")
    print("  在这个量级上，缺失会立刻被业务发现，所以问题一般当天就能定位。\n")

    # ---------------- 2) 大库：同样的缺失率，指标几乎无感 ----------------
    print("② 候选集 20000 条：2% 的 Item 没进索引，Top-10 召回几乎看不出异常")
    N, D = 20000, 16
    X, _ = make_vectors(n=N, dim=D, n_clusters=100, spread=0.4)
    ids = np.arange(N)
    Q = make_queries(X, n_queries=200, jitter=0.1)

    def top10_recall(index: FlatIndex) -> float:
        rec = 0.0
        for q in Q:
            got = index.search(q, 10).tolist()
            # 真值用完整索引（不丢失）算
            truth = full.search(q, 10).tolist()
            rec += len(set(got) & set(truth)) / 10
        return rec / len(Q)

    full = FlatIndex()
    full.add(ids, X)
    # 逐条丢失 2%：比按批丢更稳定（按批丢时 39 个批次可能一个都没命中，
    # 结论就成了「缺失 0 条」，什么也说明不了）
    broken = build_new_index(X, ids, drop_rate=0.02, rng=rng, per_batch=False)
    assert len(broken) < len(full), "随机数没命中任何丢失，换种子或调高丢失率"
    rec_b = top10_recall(broken)
    cover_b = coverage(broken, ids[:2000], X[:2000])
    rows = [
        ["完整索引", f"{len(full):,}", "0", "1.000", "基准"],
        ["有缺失的索引", f"{len(broken):,}", f"{len(full) - len(broken):,}",
         f"{rec_b:.3f}", "缺失率 "
         f"{(1 - len(broken) / len(full)):.1%}"],
    ]
    table(["索引", "条数", "缺失", "Top-10 召回", "备注"], rows)
    print(f"\n  缺失 {len(full) - len(broken):,} 条（{(1 - len(broken) / len(full)):.1%}），")
    print(f"  Top-10 召回只从 1.000 掉到 {rec_b:.3f}——每条查询有 10 个名额，")
    print("  少数目标缺失时，第 11 名的候选会顺位补上，指标几乎不动。")
    print("  这正是正文说的「大规模系统里 Bug 更隐蔽」。\n")

    # ---------------- 3) 增量回放缺口 ----------------
    print("③ 构建期间新增 Item 的回放缺口（双缓冲最常用的失败点）")
    n_inc = 500
    X_inc, _ = make_vectors(n=n_inc, dim=D, n_clusters=20, spread=0.4,
                            seed=SEED + 7)
    ids_inc = np.arange(N, N + n_inc)
    replay = 0.7                                   # 只回放了 70% 的增量
    n_replayed = int(n_inc * replay)
    broken.add(ids_inc[:n_replayed], X_inc[:n_replayed])
    all_ids = np.concatenate([ids, ids_inc])
    all_vecs = np.concatenate([X, X_inc])
    present = set(broken.ids)
    missing_inc = [int(i) for i in ids_inc if int(i) not in present]
    print(f"  构建期间新增 {n_inc} 条，回放覆盖 {replay:.0%} → "
          f"新索引里缺 {len(missing_inc)} 条新增 Item")
    print(f"  旧索引里这些 Item 是有的（它们写进了旧索引），切换后直接消失。")
    print("  这类缺失在切换前不会报错：新旧索引各自都是「合法」的。\n")

    # ---------------- 4) 四种校验能发现什么 ----------------
    print("④ 切换前的校验手段，各能发现什么：")
    n_missing_total = len(all_ids) - len(present)
    checks = []

    # (1) 条数校验
    cnt_ok = len(broken) == len(all_ids)
    checks.append(["索引条数 vs Embedding 表条数",
                   f"{len(broken):,} vs {len(all_ids):,}",
                   "✓ 发现" if not cnt_ok else "✗ 通过",
                   f"缺 {n_missing_total} 条"])

    # (2) 集合差集（id 级别）
    diff = set(int(i) for i in all_ids) - present
    checks.append(["Embedding 表与索引 ID 集合差集", f"|{len(all_ids):,} − {len(present):,}|",
                   "✓ 发现" if diff else "✗ 通过",
                   f"{len(diff)} 条不在索引里"])

    # (3) 切换后抽样自检索
    sample_ids = rng.choice(all_ids, 300, replace=False)
    sample_vecs = all_vecs[sample_ids]
    cov = coverage(broken, sample_ids, sample_vecs)
    checks.append(["切换后随机抽样自检索（300 条）", f"覆盖率 {cov:.3f}",
                   "✓ 发现" if cov < 0.995 else "✗ 通过",
                   "抽样也能看出缺口"])

    # (4) 线上召回覆盖率监控：只在候选池内部统计「该出现的有没有出现」
    pool = set(range(100))
    pool_in_index = pool & set(small.ids)
    pool_cov = len(pool_in_index) / len(pool)
    checks.append(["线上召回覆盖率监控（100 条小候选池）",
                   f"池内 {len(pool_in_index)}/{len(pool)} = {pool_cov:.2f}",
                   "✓ 发现" if pool_cov < 0.99 else "✗ 通过",
                   "按候选池统计，最贴近业务体感"])
    table(["校验手段", "本次读数", "结论", "备注"], checks)

    # ---------------- 5) 修好之后 ----------------
    print("\n⑤ 修复：批次失败重试 + 切换前阻塞式校验 + 增量全量回放")
    fixed = FlatIndex()
    for s in range(0, len(X), 512):
        e = min(s + 512, len(X))
        fixed.add(ids[s:e], X[s:e])          # 失败重试，全部写入
    fixed.add(ids_inc, X_inc)                # 增量全量回放
    cov_fix = coverage(fixed, all_ids[:2000], all_vecs[:2000])
    rows = [
        ["条数一致", f"{len(fixed):,} / {len(all_ids):,}",
         "✓" if len(fixed) == len(all_ids) else "✗"],
        ["ID 集合一致", f"缺 {len(set(map(int, all_ids)) - set(fixed.ids))} 条",
         "✓" if not (set(map(int, all_ids)) - set(fixed.ids)) else "✗"],
        ["自检索覆盖率", f"{cov_fix:.4f}", "✓" if cov_fix > 0.999 else "✗"],
    ]
    table(["检查项", "读数", "结论"], rows)
    print("\n  校验必须在切换之前、阻塞执行；切换后只做兜底监控。")
    print("  顺序反了就变成「先故障、再发现」——正文说的「先查索引，再查算法」，")
    print("  前提正是这套校验在切换前就已经跑过。\n")

    print("⑥ 一句话：把「ANN 召回率下降」当成告警现象，而不是结论。")
    print("   先看索引条数、ID 差集、增量回放，再谈算法——本脚本里所有缺失")
    print("   都来自构建流程，检索算法一行都没错。")


if __name__ == "__main__":
    main()
