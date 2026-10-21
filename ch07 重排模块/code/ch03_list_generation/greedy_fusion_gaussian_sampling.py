"""对应正文第 7 章 §7.3「基于多目标融合排序的贪心序列生成方法」。

本实现做了什么
----------------
1. 复刻正文表 7.1 的 5 个物品 pxtr 预估值(pevr/plvr/pltr/pcmr/pftr),并把候选集扩到 10 个;
2. 用正文给出的线性融合公式打分并贪心排序,得到 1 条序列:
       final_score = w1*p_evr + w2*p_lvr + w3*p_ltr + w4*p_cmr + w5*p_ftr
3. 以人工经验权重 mean = [0.2, 0.3, 0.2, 0.2, 0.1] 为均值、cov = diag(0.01) 为协方差,
   做多元高斯采样得到多组权重 w,每组权重贪心排序产出 1 条候选序列,
   即正文「序列维度的召回」:用权重扰动换取序列空间的覆盖;
4. 统计采样得到的不同序列条数、各位置的物品分布,说明协方差越大探索越广。

关键公式
--------
- 融合分:final_score(i) = <w, pxtr_i>, w ~ N(mu, Sigma)
- 序列生成:pi(w) = argsort_desc_i final_score(i),取前 T 个
- 高斯采样概率密度 p(w) ∝ exp(-0.5 (w-mu)^T Sigma^{-1} (w-mu)),
  因此离 mu 越远的权重被采到的概率越低(正文图 7.3 的直观含义)。

说明:正文代码用 numpy + pandas,本脚本只用 numpy(pandas 非本仓库允许依赖),
      打印时用手写的定宽表格代替 DataFrame 输出。
"""

import random

import numpy as np

TARGETS = ["p_evr", "p_lvr", "p_ltr", "p_cmr", "p_ftr"]

# 正文表 7.1 给出的前 5 个物品的 pxtr 预估值(np.random.seed(42) 下 rand 的取值)
BOOK_TABLE = [
    [0.374540, 0.020584, 0.611853, 0.607545, 0.122038],
    [0.950714, 0.969910, 0.139494, 0.170524, 0.495177],
    [0.731994, 0.832443, 0.292145, 0.065052, 0.034389],
    [0.598658, 0.212339, 0.366362, 0.948886, 0.909320],
    [0.156019, 0.181825, 0.456070, 0.965632, 0.258780],
]


def build_items(num_items=10):
    """前 5 个物品沿用正文表 7.1,后 5 个随机补齐,凑成 10 个候选。"""
    rows = [list(r) for r in BOOK_TABLE]
    while len(rows) < num_items:
        rows.append([round(random.random(), 6) for _ in TARGETS])
    return [{"item_id": "item_%d" % i, "pxtr": rows[i]} for i in range(num_items)]


def fuse_and_rank(items, w, top_t):
    """线性融合 + 贪心(按融合分降序)生成序列。"""
    scored = [(float(np.dot(w, it["pxtr"])), it["item_id"]) for it in items]
    scored.sort(key=lambda x: -x[0])
    seq = [item_id for _, item_id in scored[:top_t]]
    total = sum(s for s, _ in scored[:top_t])
    return seq, total


def print_item_table(items):
    print("    %-8s %s" % ("item_id", " ".join("%9s" % t for t in TARGETS)))
    for it in items:
        print("    %-8s %s" % (it["item_id"], " ".join("%9.6f" % v for v in it["pxtr"])))


def sample_weights(mean, cov_scale, n, rng):
    cov = np.diag([cov_scale] * len(mean))
    return rng.multivariate_normal(mean, cov, n)


def main():
    random.seed(42)
    rng = np.random.default_rng(42)

    items = build_items(10)
    top_t = 6
    mean = np.array([0.2, 0.3, 0.2, 0.2, 0.1])

    print("=" * 78)
    print("§7.3 基于多目标融合排序的贪心序列生成(高斯采样权重)")
    print("=" * 78)
    print("\n[1] 候选物品 pxtr 预估值(前 5 行与正文表 7.1 完全一致):")
    print_item_table(items)

    print("\n[2] 基准权重(人工经验)w0 = %s,贪心排序结果:" % np.round(mean, 3).tolist())
    base_seq, base_score = fuse_and_rank(items, mean, top_t)
    print("    序列 = %s" % " > ".join(base_seq))
    print("    前 %d 位融合分之和 = %.4f" % (top_t, base_score))

    print("\n[3] 从 N(w0, 0.01 I) 采样 8 组权重,每组贪心生成 1 条候选序列:")
    weights = sample_weights(mean, 0.01, 8, rng)
    seqs = []
    for i, w in enumerate(weights):
        seq, score = fuse_and_rank(items, w, top_t)
        seqs.append(tuple(seq))
        print("    #%d w=%s" % (i, np.round(w, 3).tolist()))
        print("       seq=%s  sum=%.4f  %s"
              % (" > ".join(s.replace("item_", "i") for s in seq), score,
                 "(= 基准序列)" if tuple(seq) == tuple(base_seq) else ""))

    uniq = []
    for s in seqs:
        if s not in uniq:
            uniq.append(s)
    print("\n    8 次采样共产出 %d 条互不相同的序列(重复率 %.1f%%)"
          % (len(uniq), 100.0 * (1 - len(uniq) / len(seqs))))

    print("\n[4] 协方差控制探索范围:同样采 200 组权重,统计不同序列条数")
    print("    %-12s %-16s %s" % ("cov 对角值", "不同序列条数", "首位物品分布"))
    for cov_scale in (0.0001, 0.001, 0.01, 0.1, 1.0):
        rng2 = np.random.default_rng(7)
        ws = sample_weights(mean, cov_scale, 200, rng2)
        all_seqs, heads = set(), {}
        for w in ws:
            seq, _ = fuse_and_rank(items, w, top_t)
            all_seqs.add(tuple(seq))
            heads[seq[0]] = heads.get(seq[0], 0) + 1
        head_str = ", ".join("%s:%d" % (k.replace("item_", "i"), v)
                             for k, v in sorted(heads.items(), key=lambda x: -x[1])[:4])
        print("    %-12s %-16d %s" % (cov_scale, len(all_seqs), head_str))

    print("\n[结论] 贪心生成每一步只取当前融合分最高者,单次采样必然是局部最优;")
    print("       靠「权重高斯采样 + 多次贪心」把序列空间覆盖面做大,是工业界最常用的")
    print("       序列生成方案(正文:序列维度的召回)。协方差越大探索越广,但偏离")
    print("       人工经验权重越远的序列,被序列评估选中的概率也越低。")


if __name__ == "__main__":
    main()
