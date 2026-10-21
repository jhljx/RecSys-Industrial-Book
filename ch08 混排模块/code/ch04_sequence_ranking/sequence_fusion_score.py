"""序列融合分与 Top-1 序列选择。

对应正文第 8 章 §8.4「序列排序」中的序列融合分计算与 argmax 选择。

本实现做了什么：
1. 按正文公式实现三层递进的序列打分口径：
   模型分 + 业务分融合 → 多业务目标加权 → 引入位置衰减因子；
2. 用同一批候选序列对比三种口径选出的 Top-1，展示「位置偏差建模比排序算法本身更重要」；
3. 扫描 alpha/beta 与衰减系数 lambda，打印 Top-1 归属随权重变化的对照表。

关键公式（全部取自正文）：
    两项融合      Score_seq = alpha * Score_model + beta * Score_business
    多目标扩展    Score_seq = sum_{k=1..N} w_k * V_k
    位置衰减      Score_seq = sum_{i=1..L} gamma_i * Score_i
    指数衰减      gamma_i = exp(-lambda * (i - 1))
                  lambda = 0.2 时 gamma = [1.00, 0.82, 0.67, 0.55]
    最终选择      Seq* = argmax_Seq Score_seq

[MOCK] 序列评估模型的输出 Score_model 由合成函数给出（不训练真实模型），
       业务价值 V_k 也由合成数据生成，仅用于演示打分口径对 Top-1 的影响。
"""

import math
import random

# 业务目标权重 w_k：正文的 V_k 例子是「广告收入 / 观看时长 / 用户留存 / 直播成交额」
BIZ_OBJ_WEIGHT = {"ad_revenue": 1.00, "watch_time": 1.20, "retention": 2.00, "live_gmv": 0.60}

LAMBDA_DECAY = 0.2   # 正文给的指数衰减系数示例
ALPHA = 0.6          # Score_model 权重
BETA = 0.4           # Score_business 权重


def gamma(i, lam=LAMBDA_DECAY):
    """位置衰减因子 gamma_i = exp(-lambda * (i - 1))，i 从 1 开始。"""
    return math.exp(-lam * (i - 1))


def business_score(item):
    """Score_business = sum_k w_k * V_k：各业务目标价值的加权组合。"""
    return sum(BIZ_OBJ_WEIGHT[k] * item["obj"].get(k, 0.0) for k in BIZ_OBJ_WEIGHT)


def model_score(item):
    """[MOCK] 序列评估模型对单 Item 的收益预测（这里用合成的 item['model'] 代替）。"""
    return item["model"]


def score_fusion_flat(seq):
    """口径 A：两项融合但不做位置衰减（正文说的「简单求和往往不合理」）。"""
    return sum(ALPHA * model_score(it) + BETA * business_score(it) for it in seq)


def score_multi_obj_flat(seq):
    """口径 B：多目标加权求和 sum_k w_k V_k，仍不做位置衰减。"""
    return sum(business_score(it) for it in seq)


def score_with_position(seq, lam=LAMBDA_DECAY, alpha=ALPHA, beta=BETA):
    """口径 C：引入位置衰减 Score_seq = sum_i gamma_i * Score_i（正文推荐口径）。"""
    total = 0.0
    for i, it in enumerate(seq, start=1):
        total += gamma(i, lam) * (alpha * model_score(it) + beta * business_score(it))
    return total


def pick_top1(seqs, scorer):
    """Seq* = argmax_Seq Score_seq。"""
    best_i, best_s = 0, -1e18
    for i, s in enumerate(seqs):
        v = scorer(s)
        if v > best_s:
            best_i, best_s = i, v
    return best_i, best_s


# -------------------------------------------------------------- 合成候选序列

def make_item(idx, biz):
    """[MOCK] 合成一个 Item：含模型分与四个业务目标价值。"""
    obj = {k: 0.0 for k in BIZ_OBJ_WEIGHT}
    if biz == "video":
        obj["watch_time"] = random.uniform(0.8, 3.0)
        obj["retention"] = random.uniform(0.02, 0.10)
    elif biz == "ad":
        obj["ad_revenue"] = random.uniform(0.6, 2.4)
        obj["retention"] = random.uniform(-0.04, 0.0)      # 广告对留存是负贡献
    else:  # live
        obj["live_gmv"] = random.uniform(0.5, 4.0)
        obj["watch_time"] = random.uniform(0.3, 1.5)
        obj["retention"] = random.uniform(0.0, 0.05)
    return {"id": f"{biz[0].upper()}{idx}", "biz": biz,
            "model": random.uniform(0.3, 1.1), "obj": obj}


def make_candidate_sequences(n_seq=6, length=6):
    """[MOCK] 合成若干等长候选序列（长度不等的情况见 length_normalization.py）。"""
    pool = [make_item(i, b) for i, b in enumerate(
        ["video"] * 8 + ["ad"] * 3 + ["live"] * 3)]
    seqs = []
    for _ in range(n_seq):
        seqs.append(random.sample(pool, length))
    return seqs


def fmt_seq(seq):
    return "|".join(it["id"] for it in seq)


# ------------------------------------------------------------------ 演示

def demo_gamma_table():
    print("=" * 78)
    print("【1】位置衰减因子 gamma_i = exp(-lambda*(i-1))（正文 lambda=0.2 的示例）")
    print("=" * 78)
    print(f"{'位置 i':>8}" + "".join(f"{i:>9}" for i in range(1, 7)))
    for lam in (0.0, 0.1, 0.2, 0.4):
        row = "".join(f"{gamma(i, lam):>9.3f}" for i in range(1, 7))
        tag = f"lambda={lam:.1f}"
        print(f"{tag:>8}{row}")
    print("-" * 78)
    g = [gamma(i, 0.2) for i in range(1, 5)]
    print("正文给出 lambda=0.2 时 gamma_1..4 = 1.0, 0.82, 0.67, 0.55；"
          f"本实现算得 {', '.join(f'{x:.2f}' for x in g)} → 一致。")


def demo_three_scorers(seqs):
    print()
    print("=" * 78)
    print("【2】三种打分口径下的候选序列得分（6 条等长候选，L=6）")
    print("=" * 78)
    print(f"{'候选序列':<26}{'A:模型+业务':>12}{'B:多目标和':>12}{'C:+位置衰减':>13}")
    print("-" * 78)
    for s in seqs:
        print(f"{fmt_seq(s):<26}{score_fusion_flat(s):>12.4f}"
              f"{score_multi_obj_flat(s):>12.4f}{score_with_position(s):>13.4f}")
    print("-" * 78)
    for name, fn in (("A 模型+业务融合", score_fusion_flat),
                     ("B 多目标加权和", score_multi_obj_flat),
                     ("C 融合+位置衰减", score_with_position)):
        i, v = pick_top1(seqs, fn)
        print(f"{name:<16} Top-1 = 候选#{i}  {fmt_seq(seqs[i]):<24} score={v:.4f}")
    print("不同口径选出的 Top-1 并不相同 → 正文强调「核心挑战来自价值函数设计，而非排序算法」。")


def demo_alpha_beta_sweep(seqs):
    print()
    print("=" * 78)
    print("【3】alpha/beta 扫描：模型分与业务分的权重如何改变 Top-1（lambda=0.2 固定）")
    print("=" * 78)
    print(f"{'alpha':>7}{'beta':>7}{'Top-1':>10}{'最高分':>10}{'次高分':>10}{'领先幅度':>11}")
    print("-" * 78)
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        b = 1.0 - a
        scored = sorted(((score_with_position(s, LAMBDA_DECAY, a, b), i)
                         for i, s in enumerate(seqs)), reverse=True)
        lead = (scored[0][0] - scored[1][0]) / max(scored[1][0], 1e-9)
        print(f"{a:>7.2f}{b:>7.2f}{'#' + str(scored[0][1]):>10}"
              f"{scored[0][0]:>10.4f}{scored[1][0]:>10.4f}{lead:>10.2%}")
    print("-" * 78)
    print("alpha 偏向序列评估模型、beta 偏向业务价值分；两者权衡会直接改写最终曝光序列。")


def demo_lambda_sweep(seqs):
    print()
    print("=" * 78)
    print("【4】lambda 扫描：位置偏差建模强度对 Top-1 的影响（alpha=0.6, beta=0.4）")
    print("=" * 78)
    print(f"{'lambda':>8}{'Top-1':>8}{'得分':>10}{'首位Item':>11}{'首位业务':>10}"
          f"{'前2位分数占比':>14}")
    print("-" * 78)
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        i, v = pick_top1(seqs, lambda s, l=lam: score_with_position(s, l))
        best = seqs[i]
        head = sum(gamma(j, lam) * (ALPHA * model_score(it) + BETA * business_score(it))
                   for j, it in enumerate(best[:2], start=1))
        print(f"{lam:>8.2f}{'#' + str(i):>8}{v:>10.4f}{best[0]['id']:>11}"
              f"{best[0]['biz']:>10}{head / v:>13.1%}")
    print("-" * 78)
    print("lambda 越大，首屏权重越集中（前 2 位贡献占比从约 1/3 升到过半），"
          "排序越倾向把高价值内容顶到首位。")


if __name__ == "__main__":
    random.seed(42)
    seqs = make_candidate_sequences()
    demo_gamma_table()
    demo_three_scorers(seqs)
    demo_alpha_beta_sweep(seqs)
    demo_lambda_sweep(seqs)
