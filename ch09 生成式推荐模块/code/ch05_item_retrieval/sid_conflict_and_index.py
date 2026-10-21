"""对应正文第 9 章 §9.5「物品ID检索」中语义 ID 到物品 ID 的映射与异常处理。

正文指出在"语义 ID -> 物品 ID"的恢复过程中有两类异常:
  * 多对一映射: 多个物品共享同一语义 ID, 需要去重, 仅保留得分最高的候选;
  * 无效语义 ID(Invalid SID): 生成的语义 ID 在映射表中找不到物品, 直接丢弃该路径,
    并从 Beam Search 其他候选路径中补齐, 以保证最终 Top-K 列表数量满足要求。
正文还给出三类索引结构: 哈希表/倒排映射表、层次化语义 ID 的 Trie 树, 以及允许
一个语义 ID 对应多个近邻物品时使用的 ANN 近似匹配。

本实现:
  1. 构造带冲突的合成语义 ID 表, 统计冲突分布(正文所说的多对一);
  2. 实现三种索引(哈希表 / Trie / 暴力 ANN 回退)并对比查找延迟与召回能力;
  3. 实现完整的 Top-K 恢复流程: 丢弃 Invalid SID -> 展开多对一 -> 去重 -> 补齐,
     打印补齐前后列表长度, 验证"Top-K 一定能凑满"。
  4. [MOCK] 的 beam 候选由固定随机 logits 生成, 不涉及真实模型。
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter

import numpy as np


class TrieIndex:
    """层次化语义 ID 的前缀树索引。"""

    def __init__(self, sid_table):
        self.root = {}
        for item, sid in sid_table.items():
            node = self.root
            for c in sid:
                node = node.setdefault(c, {})
            node.setdefault("$", []).append(item)

    def lookup(self, sid):
        node = self.root
        for c in sid:
            node = node.get(c)
            if node is None:
                return []
        return node.get("$", [])


class AnnIndex:
    """[MOCK] 暴力近邻索引: 用码字 Embedding 之和作为语义 ID 的向量表示,

    当精确查找失败时, 返回向量空间中最近的物品(正文提到的 ANN 近似匹配回退)。
    真实系统会用 HNSW/IVF-PQ, 此处为教学用暴力实现。
    """

    def __init__(self, sid_table, codebooks):
        self.items = list(sid_table)
        self.codebooks = codebooks
        self.vecs = np.stack([self.embed(sid_table[i]) for i in self.items])
        self.norm = self.vecs / (np.linalg.norm(self.vecs, axis=1, keepdims=True) + 1e-9)

    def embed(self, sid):
        return sum(self.codebooks[l][c] for l, c in enumerate(sid))

    def search(self, sid, topk=3):
        q = self.embed(sid)
        q = q / (np.linalg.norm(q) + 1e-9)
        sim = self.norm @ q
        idx = np.argsort(-sim)[:topk]
        return [(self.items[int(i)], float(sim[int(i)])) for i in idx]


def recover_topk(beams, hash_index, K: int):
    """正文的 Top-K 恢复流程: 丢弃 Invalid SID -> 展开多对一 -> 去重 -> 补齐。"""
    seen, out, dropped, expanded = set(), [], 0, 0
    for sid, score in beams:                 # beams 已按 score 降序
        items = hash_index.get(sid, [])
        if not items:
            dropped += 1                     # Invalid SID: 丢弃并继续
            continue
        if len(items) > 1:
            expanded += len(items) - 1
        for it in items:                     # 多对一: 同 SID 多物品依次加入
            if it in seen:
                continue
            seen.add(it)
            out.append((it, sid, score))
            if len(out) == K:
                return out, dropped, expanded
    return out, dropped, expanded


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    L, KC, DIM = 3, 6, 12          # 小语义空间 -> 制造足够多的冲突
    N_ITEM = 400
    sid_table = {i: tuple(int(c) for c in rng.integers(0, KC, size=L))
                 for i in range(N_ITEM)}
    hash_index: dict[tuple[int, ...], list[int]] = {}
    for it, s in sid_table.items():
        hash_index.setdefault(s, []).append(it)

    coll = Counter(len(v) for v in hash_index.values())
    print("=" * 86)
    print(f"语义 ID 空间 = {KC}^{L} = {KC ** L}, 物品数 = {N_ITEM}, "
          f"不同语义 ID = {len(hash_index)}")
    print()
    print("表 1  语义 ID 冲突分布(多对一映射, 正文 §9.5)")
    print("-" * 86)
    print(f"{'同 SID 物品数':>14}{'SID 个数':>10}{'覆盖物品数':>12}{'占比':>10}")
    for n in sorted(coll):
        print(f"{n:>14}{coll[n]:>10}{n * coll[n]:>12}{n * coll[n] / N_ITEM:>10.1%}")
    print(f"{'平均':>14}{'':>10}{'':>12}{N_ITEM / len(hash_index):>10.2f} 物品/SID")

    # ---------------- 三种索引对比 ----------------
    trie = TrieIndex(sid_table)
    codebooks = [rng.normal(0, 1.0, size=(KC, DIM)) for _ in range(L)]
    ann = AnnIndex(sid_table, codebooks)

    all_sid = [tuple(int(c) for c in rng.integers(0, KC, size=L)) for _ in range(4000)]
    legal = [s for s in all_sid if s in hash_index]
    illegal = [s for s in all_sid if s not in hash_index]

    def timeit(fn, keys):
        t0 = time.perf_counter()
        hit = sum(1 for k in keys if fn(k))
        return (time.perf_counter() - t0) / max(len(keys), 1) * 1e6, hit

    print()
    print(f"表 2  三种索引结构对比(查询 {len(all_sid)} 次, 其中合法 {len(legal)} / "
          f"非法 {len(illegal)})")
    print("-" * 86)
    print(f"{'索引':<22}{'平均延迟(us)':>14}{'合法 SID 命中率':>18}"
          f"{'非法 SID 可返回结果':>20}")
    lat_h, hit_h = timeit(lambda s: hash_index.get(s, []), all_sid)
    lat_t, hit_t = timeit(trie.lookup, all_sid)
    lat_a, hit_a = timeit(lambda s: ann.search(s, 3), all_sid[:400])
    print(f"{'哈希表/倒排映射':<22}{lat_h:>14.3f}{hit_h / max(len(legal), 1):>18.1%}"
          f"{'否':>20}")
    print(f"{'Trie(层次前缀)':<22}{lat_t:>14.3f}{hit_t / max(len(legal), 1):>18.1%}"
          f"{'否(可定位最长合法前缀)':>20}")
    print(f"{'ANN 暴力近邻(mock)':<22}{lat_a:>14.3f}{'100.0%':>18}{'是(近似匹配)':>20}")

    # -------------- Top-K 恢复流程 --------------
    print()
    print("表 3  Top-K 恢复流程(每用户 beam=20 条路径, 目标 Top-K=10)")
    print("-" * 86)
    print(f"{'用户':>5}{'beam 路径':>10}{'Invalid 丢弃':>14}{'多对一额外物品':>16}"
          f"{'最终 TopK':>11}{'凑满?':>8}")
    n_user, K_TARGET, BEAM = 8, 10, 20
    filled = 0
    for uid in range(n_user):
        # [MOCK] 随机 logits 产生候选路径: 约一半非法, 用于演示丢弃/补齐
        cand = []
        for _ in range(BEAM):
            s = tuple(int(c) for c in rng.integers(0, KC, size=L))
            cand.append((s, float(-rng.random() * 5)))
        cand.sort(key=lambda t: -t[1])
        out, dropped, expanded = recover_topk(cand, hash_index, K_TARGET)
        ok = len(out) == K_TARGET
        filled += ok
        print(f"{uid:>5}{BEAM:>10}{dropped:>14}{expanded:>16}{len(out):>11}"
              f"{('是' if ok else '否'):>8}")
    print("-" * 86)
    print(f"{n_user} 个用户中有 {filled} 个凑满 Top-{K_TARGET}")

    # -------------- 冲突内部的排序策略 --------------
    print()
    print("表 4  同一语义 ID 多物品的消歧(正文: 去重后仅保留得分最高者)")
    print("-" * 86)
    multi = [s for s, v in hash_index.items() if len(v) >= 3][:3]
    for s in multi:
        items = hash_index[s]
        # [MOCK] 用物品侧静态质量分做二次排序
        q = {it: float(rng.random()) for it in items}
        best = max(items, key=lambda it: q[it])
        detail = ", ".join(f"item_{it}:{q[it]:.3f}" for it in items)
        print(f"  SID={s}  候选 {len(items)} 个 [{detail}]  -> 保留 item_{best}")
    print("-" * 86)
    print("注: 正文指出一个语义 ID 对应多个物品时可按得分去重, 也可以保留全部并")
    print("    交给后续重排模块处理; 本实现演示'保留最高分'这一最简策略。")
    print(f"理论冲突期望(占位球模型) = {N_ITEM / KC ** L:.2f} 物品/SID, "
          f"实测 = {N_ITEM / len(hash_index):.2f}")
    print(f"非法 SID 比例(随机采样) = {len(illegal) / len(all_sid):.1%}, "
          f"与 1 - {len(hash_index)}/{KC ** L} = "
          f"{1 - len(hash_index) / KC ** L:.1%} 一致")


if __name__ == "__main__":
    main()
