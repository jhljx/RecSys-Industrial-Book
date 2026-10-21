"""对应正文第 7 章 §7.1「位置偏置（position bias）」与 §7.4 的位置衰减权重。

本实现做了什么
----------------
1. [MOCK] 合成上下滑短视频会话日志：用户从第 1 条开始看，每看完一条以概率 p_slide 继续下滑，
   同一条视频在越靠后的位置观看时长越短（注意力衰减）；
2. 从日志中统计「每个位置的平均观看时长」与「每个位置的曝光占比」，复现正文现象：
   随着会话位置后移，单条视频观看时长递减；
3. 把经验衰减曲线与正文给出的两种位置权重做对比：
       w_i = 1 / i        （倒数衰减）
       w_i = e^{-(i-1)}   （指数衰减）
4. 给出一个重要结论的数值验证：若序列价值只有位置权重而没有物品间交互
   （value = sum_i w_i f(x_i)，w 单调递减），那么按 f 降序排列就是最优解（重排列不等式），
   此时重排没有收益——序列级优化的价值必须来自物品之间的相互影响（见 pointwise_vs_listwise.py）。

关键公式（§7.4）
-----------------
    value = sum_{i=1}^{N} w_i * f(x_i)
"""

import itertools
import math
import random

N_SESSIONS = 20000
MAX_POS = 10

# 位置注意力真实衰减率（合成数据的 ground truth，线上需要用日志估计）
TRUE_ATTENTION_DECAY = 0.18


def simulate_sessions(n_sessions, max_pos):
    """[MOCK] 模拟上下滑会话：返回 (每个位置的观看时长列表, 每个位置的曝光次数)。"""
    watch_by_pos = [[] for _ in range(max_pos)]
    exposure = [0] * max_pos
    for _ in range(n_sessions):
        for pos in range(1, max_pos + 1):
            exposure[pos - 1] += 1
            base = random.uniform(15.0, 45.0)              # 视频本身的吸引力
            attention = math.exp(-TRUE_ATTENTION_DECAY * (pos - 1))
            noise = math.exp(random.gauss(0.0, 0.25))      # 对数正态噪声
            watch_by_pos[pos - 1].append(base * attention * noise)
            # 是否继续下滑
            if random.random() > 0.90 - 0.015 * (pos - 1):
                break
    return watch_by_pos, exposure


def main():
    random.seed(42)

    print("=" * 84)
    print("§7.1 位置偏置：合成会话日志统计（%d 个会话）" % N_SESSIONS)
    print("=" * 84)

    watch_by_pos, exposure = simulate_sessions(N_SESSIONS, MAX_POS)
    avg_watch = [sum(v) / len(v) if v else 0.0 for v in watch_by_pos]
    base_watch = avg_watch[0]

    print("%-6s %-10s %-12s %-14s %-12s %-12s"
          % ("位置i", "曝光数", "曝光占比", "平均观看时长", "经验衰减", "1/i"))
    for i in range(MAX_POS):
        print("%-6d %-10d %-12.4f %-14.3f %-12.4f %-12.4f"
              % (i + 1, exposure[i], exposure[i] / float(N_SESSIONS),
                 avg_watch[i], avg_watch[i] / base_watch, 1.0 / (i + 1)))

    # 用对数线性回归估计注意力衰减率： log(avg_i) = log(avg_1) - k*(i-1)
    xs = [float(i) for i in range(MAX_POS)]
    ys = [math.log(avg_watch[i]) for i in range(MAX_POS)]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    k_hat = -cov / var
    print("\n[估计] 注意力衰减率 k_hat = %.4f（合成数据真值 %.4f）" % (k_hat, TRUE_ATTENTION_DECAY))

    print("\n" + "=" * 84)
    print("§7.4 两种位置权重的量级对比")
    print("=" * 84)
    print("%-6s %-14s %-18s %-16s" % ("位置i", "w=1/i", "w=exp(-(i-1))", "经验衰减 exp(-k_hat(i-1))"))
    for i in range(1, MAX_POS + 1):
        print("%-6d %-14.6f %-18.6f %-16.6f"
              % (i, 1.0 / i, math.exp(-(i - 1)), math.exp(-k_hat * (i - 1))))
    print("注：e^{-(i-1)} 衰减极快，第 5 位权重已不足 2%，等价于只优化头部 2~3 个位置；")
    print("    1/i 衰减更缓，更接近本例经验衰减曲线的中段。实际系统应以日志估计的衰减为准。")

    # ---- 位置权重单独不足以产生重排收益 ----
    print("\n" + "=" * 84)
    print("只有位置权重时，按 f 降序即最优（枚举验证）")
    print("=" * 84)
    scores = [round(random.uniform(5.0, 50.0), 2) for _ in range(6)]
    print("  6 个候选的单点分 f = %s" % scores)
    for name, weight_fn in (("w=1/i", lambda i: 1.0 / i),
                            ("w=exp(-(i-1))", lambda i: math.exp(-(i - 1)))):
        best = max(
            itertools.permutations(scores, 4),
            key=lambda p: sum(weight_fn(i + 1) * s for i, s in enumerate(p)),
        )
        desc = tuple(sorted(scores, reverse=True)[:4])
        best_v = sum(weight_fn(i + 1) * s for i, s in enumerate(best))
        desc_v = sum(weight_fn(i + 1) * s for i, s in enumerate(desc))
        print("  %-16s 枚举最优序列 %s (value=%.4f)" % (name, best, best_v))
        print("  %-16s 按 f 降序排列 %s (value=%.4f) -> %s"
              % ("", desc, desc_v, "一致" if best == desc else "不一致"))
    print("\n[结论] 位置偏置本身只解释了「为什么要加权」，并不构成重排的收益来源；")
    print("       重排的收益来自物品间交互（疲劳、相似度、垂类分布）带来的非可分解项。")


if __name__ == "__main__":
    main()
