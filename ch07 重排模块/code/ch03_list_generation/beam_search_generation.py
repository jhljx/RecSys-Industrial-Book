"""对应正文第 7 章 §7.3「基于束搜索(beam search)的序列生成方法」。

本实现做了什么
----------------
1. 复刻正文 code_beam_search_core 的思路:在 beam search 的每一步对每条路径的权重向量
   再做一次高斯扰动采样,用新权重贪心排序产出新序列,按序列分保留 top-beam_size 条;
2. 另外实现正文正文更强调的「逐位置扩展 + 业务规则剪枝」版 beam search:
   状态 = 已选前缀,每步枚举所有未选候选,嵌入正文给出的硬约束
   「相同垂类内容不得连续出现」,对非法路径提前剪枝,保留 top-K 前缀;
3. 对比三者:单次贪心、权重扰动式 beam search、逐位置 beam search 的
   序列分与「相邻同垂类对数」,并统计逐位置 beam search 剪掉了多少条非法路径;
4. 打印穷举规模 P(N,T) 与 beam search 实际扩展的节点数,说明剪枝带来的复杂度收益。

关键公式
--------
- 路径分:score(pi) = sum_{t=1..T} w_t_decay * <w, pxtr_{i_t}>,这里位置衰减 w_t_decay = 1/t
  (正文 §7.4 的位置偏置权重 w_i = 1/i);
- beam search:每步把 B 条路径各扩展 |剩余候选| 个子节点,按 score 截断回 B 条,
  复杂度 O(T * B * N) 远小于穷举 P(N, T) = N!/(N-T)!;
- 权重扰动:w' ~ N(w, Sigma),Sigma = diag(0.01)。

注:正文代码用 numpy + pandas,本脚本只用 numpy。
"""

import math
import random

import numpy as np

TARGETS = ["p_evr", "p_lvr", "p_ltr", "p_cmr", "p_ftr"]
CATEGORIES = ["影视", "美食", "萌宠", "知识", "游戏"]

BOOK_TABLE = [
    [0.374540, 0.020584, 0.611853, 0.607545, 0.122038],
    [0.950714, 0.969910, 0.139494, 0.170524, 0.495177],
    [0.731994, 0.832443, 0.292145, 0.065052, 0.034389],
    [0.598658, 0.212339, 0.366362, 0.948886, 0.909320],
    [0.156019, 0.181825, 0.456070, 0.965632, 0.258780],
]


def build_items(num_items=10):
    rows = [list(r) for r in BOOK_TABLE]
    while len(rows) < num_items:
        rows.append([round(random.random(), 6) for _ in TARGETS])
    items = []
    for i in range(num_items):
        # 垂类人为做成头部效应:影视占一半,便于观察「同垂类不得连续」的剪枝效果
        cat = "影视" if i % 2 == 0 else CATEGORIES[1 + (i // 2) % 4]
        items.append({"idx": i, "item_id": "item_%d" % i, "pxtr": rows[i], "cat": cat})
    return items


def point_score(item, w):
    return float(np.dot(w, item["pxtr"]))


def seq_score(seq, w):
    """位置衰减加权的序列分,w_t = 1/t。"""
    return sum(point_score(it, w) / (t + 1) for t, it in enumerate(seq))


def greedy(items, w, top_t):
    ranked = sorted(items, key=lambda it: -point_score(it, w))
    return ranked[:top_t]


def beam_search_weight_perturb(items, mean, cov_scale, top_t, beam_size=3, num_steps=5,
                               rng=None):
    """正文 code_beam_search_core 的口径:beam 中保存(序列, 分数, 权重),
    每步对每条路径的权重再采样 beam_size 组,用贪心重新生成整条序列。"""
    cov = np.diag([cov_scale] * len(mean))
    beam = []
    for w in rng.multivariate_normal(mean, cov, beam_size):
        seq = greedy(items, w, top_t)
        beam.append((seq, seq_score(seq, w), w))
    expanded = beam_size
    for _ in range(num_steps):
        cands = []
        for seq, score, w in beam:
            for new_w in rng.multivariate_normal(w, cov, beam_size):
                new_seq = greedy(items, new_w, top_t)
                cands.append((new_seq, seq_score(new_seq, new_w), new_w))
                expanded += 1
        cands.sort(key=lambda x: -x[1])
        beam = cands[:beam_size]
    return beam, expanded


def beam_search_positionwise(items, w, top_t, beam_size=4, no_same_cat_adjacent=True):
    """逐位置扩展的 beam search,带「相邻同垂类禁止」硬约束剪枝。

    返回 (beam, expanded, pruned):beam 为 [(序列, 分数)],按分数降序。
    """
    beam = [([], 0.0)]
    expanded = pruned = 0
    for t in range(top_t):
        cands = []
        for seq, score in beam:
            used = {it["idx"] for it in seq}
            for it in items:
                if it["idx"] in used:
                    continue
                expanded += 1
                if no_same_cat_adjacent and seq and seq[-1]["cat"] == it["cat"]:
                    pruned += 1       # 中间步骤提前剔除非法路径
                    continue
                cands.append((seq + [it], score + point_score(it, w) / (t + 1)))
        cands.sort(key=lambda x: -x[1])
        beam = cands[:beam_size]
    return beam, expanded, pruned


def adj_same_cat(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if a["cat"] == b["cat"])


def fmt(seq):
    return " > ".join("%s(%s)" % (it["item_id"].replace("item_", "i"), it["cat"])
                      for it in seq)


def main():
    random.seed(42)
    rng = np.random.default_rng(42)

    items = build_items(10)
    top_t, n = 6, 10
    mean = np.array([0.2, 0.3, 0.2, 0.2, 0.1])

    print("=" * 84)
    print("§7.3 基于 Beam Search 的序列生成(含业务规则剪枝)")
    print("=" * 84)
    print("\n[0] 候选集:%d 个物品,目标序列长度 T=%d" % (n, top_t))
    print("    垂类分布:%s" % ", ".join(
        "%s=%d" % (c, sum(1 for it in items if it["cat"] == c)) for c in CATEGORIES))
    print("    穷举排列数 P(%d,%d) = %d" % (n, top_t, math.perm(n, top_t)))

    # ---------- A. 单次贪心 ----------
    g_seq = greedy(items, mean, top_t)
    print("\n[1] 基线:单次贪心(基准权重)")
    print("    %s" % fmt(g_seq))
    print("    序列分(1/i 衰减)=%.4f,相邻同垂类对数=%d" % (seq_score(g_seq, mean),
                                                        adj_same_cat(g_seq)))

    # ---------- B. 权重扰动式 beam search ----------
    beam, expanded = beam_search_weight_perturb(items, mean, 0.01, top_t,
                                                beam_size=3, num_steps=5, rng=rng)
    print("\n[2] 权重扰动式 Beam Search(beam_size=3, steps=5,正文代码口径)")
    print("    共扩展 %d 个候选序列,最终保留 3 条:" % expanded)
    for i, (seq, score, w) in enumerate(beam):
        print("    候选%d 分数=%.4f 权重=%s" % (i + 1, score, np.round(w, 3).tolist()))
        print("          %s  相邻同垂类=%d" % (fmt(seq), adj_same_cat(seq)))
    print("    注:该口径每条路径仍由「贪心全量重排」产出,权重是唯一的搜索变量,")
    print("        因此无法保证满足「相邻不同垂类」这类序列级硬约束。")

    # ---------- C. 逐位置 beam search + 剪枝 ----------
    for bs in (1, 4, 16):
        pb, exp2, pruned = beam_search_positionwise(items, mean, top_t, beam_size=bs)
        best, bscore = pb[0]
        print("\n[3] 逐位置 Beam Search(beam_size=%d,相邻同垂类剪枝)" % bs)
        print("    扩展节点 %d 个,其中 %d 个被约束剪掉(%.1f%%)"
              % (exp2, pruned, 100.0 * pruned / exp2))
        print("    最优:%s" % fmt(best))
        print("    序列分=%.4f,相邻同垂类对数=%d" % (bscore, adj_same_cat(best)))

    # ---------- D. 与穷举最优对比(T=6, N=10 可枚举) ----------
    print("\n[4] 与穷举最优对比(合法序列中,位置衰减序列分最大者)")
    import itertools
    best_exh, best_v = None, -1e18
    legal = 0
    for perm in itertools.permutations(items, top_t):
        if adj_same_cat(perm):
            continue
        legal += 1
        v = seq_score(perm, mean)
        if v > best_v:
            best_exh, best_v = perm, v
    print("    合法排列 %d / %d 条" % (legal, math.perm(n, top_t)))
    print("    穷举最优:%s" % fmt(list(best_exh)))
    print("    穷举最优序列分=%.4f" % best_v)
    pb16, _, _ = beam_search_positionwise(items, mean, top_t, beam_size=16)
    print("    beam_size=16 的最优分=%.4f,与穷举最优差距=%.6f"
          % (pb16[0][1], best_v - pb16[0][1]))

    print("\n[结论] beam search 把搜索从 P(N,T)=%d 条降到 O(T*B*N)=%d 次扩展;"
          % (math.perm(n, top_t), top_t * 16 * n))
    print("       同时「相邻同垂类禁止」等业务约束能在中间步骤剪枝,是贪心生成做不到的。")


if __name__ == "__main__":
    main()
