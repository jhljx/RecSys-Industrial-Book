"""对应正文第 7 章 §7.1「为什么需要重排」。

本实现做了什么
----------------
1. 构造一个合成的短视频候选集，给每个物品一个精排口径的单点分 f(x_i)（预估观看时长）；
2. 用正文 §7.4 的「是否继续下滑」模型定义序列价值（曝光概率依赖于前面物品，
   因此序列价值无法分解为单点分之和）；
3. 枚举全部 P(N, T) 个排列，对比
   - 精排式 pointwise 贪心排序（按 f(x_i) 降序取前 T 个）
   - 序列级最优排列
   给出两者的价值差距与贪心解在全部排列中的排名；
4. 复现正文关于搜索空间规模的论述：500^8 / 100^8 / K!。

关键公式（§7.1 + §7.4）
------------------------
    p(item_1) = p0
    p(item_i) = p0 * prod_{j=1}^{i-1} p_slide(x_j)        (i > 1)
    value(pi) = sum_{i=1}^{T} p(item_i) * f(x_i)

其中 p_slide(x_j) 依赖上下文：当第 j 条与第 j-1 条同垂类时用户更易疲劳、下滑概率下降。
正是这种物品间的相互影响，使得「每个位置取当前最高分」的贪心策略只能得到局部最优，
从而需要在精排之后引入序列级优化的重排模块。
"""

import itertools
import math
import random

CATEGORIES = ("影视", "美食", "萌宠")

# 相邻同垂类时的下滑概率衰减系数（内容疲劳）
SAME_CATEGORY_FATIGUE = 0.60


def make_candidates(n):
    """生成候选集：每个物品带一个质量分 q 与一个内容垂类。"""
    items = []
    for i in range(n):
        items.append(
            {
                "id": "item_%d" % i,
                "q": round(random.uniform(0.20, 0.95), 3),
                "cat": random.choice(CATEGORIES),
            }
        )
    return items


def pointwise_score(item):
    """精排输出的单点分 f(x_i)：此处取预估观看时长（秒）。"""
    return 10.0 + 40.0 * item["q"]


def p_slide(item, prev_item):
    """看完 item 后继续下滑的概率 p_slide(x_j)，与前一条内容相关。"""
    p = 0.55 + 0.40 * item["q"]
    if prev_item is not None and prev_item["cat"] == item["cat"]:
        p *= SAME_CATEGORY_FATIGUE
    return min(p, 0.99)


def sequence_value(seq, p0=1.0):
    """序列整体价值 value = sum_i p(item_i) * f(x_i)。"""
    value = 0.0
    prob = p0
    prev = None
    for item in seq:
        value += prob * pointwise_score(item)
        prob *= p_slide(item, prev)
        prev = item
    return value


def naive_sum_value(seq):
    """对照口径：单点分简单加和（与排列顺序无关）。"""
    return sum(pointwise_score(it) for it in seq)


def fmt_seq(seq):
    return " -> ".join("%s(%s,%.1fs)" % (it["id"], it["cat"], pointwise_score(it)) for it in seq)


def permutation_count(n, t):
    return math.factorial(n) // math.factorial(n - t)


def main():
    random.seed(42)

    n_candidates, list_len = 6, 4
    items = make_candidates(n_candidates)

    print("=" * 78)
    print("§7.1 精排 pointwise 贪心 vs 重排序列级最优")
    print("=" * 78)
    print("候选集（N=%d），精排单点分 f(x_i) = 预估观看时长：" % n_candidates)
    print("  %-8s %-6s %-6s %-10s" % ("物品", "垂类", "质量q", "f(x_i)/秒"))
    for it in sorted(items, key=pointwise_score, reverse=True):
        print("  %-8s %-6s %-6.3f %-10.2f" % (it["id"], it["cat"], it["q"], pointwise_score(it)))

    # ---- 1. 精排式 pointwise 贪心：按单点分降序截断 ----
    greedy_seq = sorted(items, key=pointwise_score, reverse=True)[:list_len]
    greedy_value = sequence_value(greedy_seq)

    # ---- 2. 序列级枚举：P(N, T) 个排列全算一遍 ----
    all_perms = list(itertools.permutations(items, list_len))
    scored = sorted(((sequence_value(p), p) for p in all_perms), key=lambda x: -x[0])
    best_value, best_seq = scored[0]
    worst_value, worst_seq = scored[-1]

    greedy_rank = 1 + sum(1 for v, _ in scored if v > greedy_value + 1e-12)

    print("\n[搜索空间] T=%d 时共 P(%d,%d)=%d 个排列，全部枚举完毕。"
          % (list_len, n_candidates, list_len, len(all_perms)))

    print("\n--- 精排 pointwise 贪心序列（按 f 降序）---")
    print("  " + fmt_seq(greedy_seq))
    print("  单点分加和            = %.2f" % naive_sum_value(greedy_seq))
    print("  序列价值 value(pi)    = %.4f" % greedy_value)
    print("  在 %d 个排列中排名    = 第 %d 名" % (len(all_perms), greedy_rank))

    print("\n--- 序列级最优排列 ---")
    print("  " + fmt_seq(list(best_seq)))
    print("  单点分加和            = %.2f" % naive_sum_value(best_seq))
    print("  序列价值 value(pi)    = %.4f" % best_value)

    print("\n--- 序列级最差排列（对照）---")
    print("  " + fmt_seq(list(worst_seq)))
    print("  序列价值 value(pi)    = %.4f" % worst_value)

    gain = (best_value - greedy_value) / greedy_value * 100.0
    spread = (best_value - worst_value) / worst_value * 100.0
    print("\n[结论] 序列级最优相对 pointwise 贪心提升 %.2f%%；"
          "最优与最差排列之间差距 %.2f%%。" % (gain, spread))

    same_cat_greedy = sum(
        1 for a, b in zip(greedy_seq, greedy_seq[1:]) if a["cat"] == b["cat"]
    )
    same_cat_best = sum(1 for a, b in zip(best_seq, best_seq[1:]) if a["cat"] == b["cat"])
    print("       相邻同垂类对数：贪心 %d 对，序列最优 %d 对"
          % (same_cat_greedy, same_cat_best))

    # ---- 3. Top-K 集合选择差异：序列级最优未必取单点分 Top-K ----
    greedy_set = {it["id"] for it in greedy_seq}
    best_set = {it["id"] for it in best_seq}
    print("\n[集合层面] 贪心选中 %s" % sorted(greedy_set))
    print("           序列最优选中 %s" % sorted(best_set))
    print("           两者集合%s" % ("相同（差异只在顺序）" if greedy_set == best_set else "不同"))

    # ---- 4. 正文 §7.1 的搜索空间规模 ----
    print("\n" + "=" * 78)
    print("§7.1 搜索空间规模（正文口径）")
    print("=" * 78)
    print("  精排候选 500、展示 8 条：近似 500^8 = %.3e" % (500.0 ** 8))
    print("  截断 top-100 后重排    ：近似 100^8 = %.3e" % (100.0 ** 8))
    print("  精确排列数 P(100, 8)   = %.3e" % permutation_count(100, 8))
    for k in (5, 8, 10):
        print("  截断到 top-%-2d 再重排   ：%d! = %d" % (k, k, math.factorial(k)))
    print("\n  => 重排把组合爆炸从 N^T 压到 K!（K<10），才有可能在毫秒级延迟内做序列级优化。")


if __name__ == "__main__":
    main()
