r"""对应正文第 9 章 §9.7「生成式推荐落地问题」中的曝光偏差/数据循环与候选聚集度。

正文要点:
  * 曝光偏差(Exposure Bias)-> 数据循环(Data Feedback Loop): 模型倾向生成历史上
    表现好的内容, 这类内容更易曝光, 消费反馈又进入用户序列, 下一轮继续强化偏好,
    长期迭代后用户陷入信息茧房、模型丧失探索能力;
  * 对策: 不能把高反馈物品直接当正样本, 要区分内容垂类/营销号属性/内容质量,
    并平衡 Exploration 与 Exploitation;
  * 候选集聚集度过高: 生成式模型端到端直出候选, 缺少显式多样性控制;
    对策是在 Decoder 生成与 Beam Search 中引入多样性约束 —— 增大束宽扩大搜索空间,
    在路径评分中加入类别覆盖率/新颖性/多样性约束项并配合剪枝。

本实现:
  1. [MOCK] 一个闭环仿真器: 物品有"垂类 + 质量分 + 诱导点击系数(营销号)",
     用户点击率由真实质量与诱导系数共同决定; 模型用 Token 频次统计近似训练,
     每轮用上轮曝光日志重训, 迭代 12 轮, 对比
        (a) 朴素: 高反馈物品直接当正样本;
        (b) 质量加权 + 垂类降权(正文的细粒度样本构建);
        (c) 质量加权 + epsilon 探索;
     打印基尼系数、垂类熵、营销号曝光占比、真实质量均值的逐轮表。
  2. 多样性约束 Beam Search: 得分 = log P - alpha * 类别重复惩罚 + beta * 新颖性,
     对比不同束宽/不同约束强度下的类别覆盖数与平均 log P 损失。
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

N_ITEM = 300
N_CAT = 8
TOPK = 10


def build_items(rng):
    """每个物品: 垂类 cat, 真实质量 qual, 诱导点击系数 bait(营销号 > 1)。"""
    cat = rng.integers(0, N_CAT, size=N_ITEM)
    qual = rng.random(N_ITEM)
    bait = np.ones(N_ITEM)
    # 2 号与 5 号垂类里有一半是"营销号": 诱导点击强但真实质量不高
    mk = ((cat == 2) | (cat == 5)) & (rng.random(N_ITEM) < 0.5)
    bait[mk] = 2.6
    return cat, qual, bait


def click_prob(qual, bait):
    """用户点击概率: 诱导系数放大短期点击, 与真实质量弱相关(曝光偏差来源)。"""
    p = 0.08 + 0.25 * qual
    return np.clip(p * bait, 0.0, 0.95)


def gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def entropy(counts):
    c = np.asarray(counts, dtype=float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())


def simulate(strategy, cat, qual, bait, rng, rounds=12, n_user=400, eps=0.15):
    """[MOCK] 生成式推荐闭环: 模型得分 = 训练样本中该物品的加权频次(平滑)。"""
    score = np.ones(N_ITEM)                       # 初始均匀(冷启)
    cp = click_prob(qual, bait)
    rows = []
    for rd in range(rounds):
        p = score / score.sum()
        expo = np.zeros(N_ITEM)
        pos_w = np.zeros(N_ITEM)                  # 下一轮训练样本权重
        for _ in range(n_user):
            if strategy == "quality+explore" and rng.random() < eps:
                pick = rng.choice(N_ITEM, size=TOPK, replace=False)
            else:
                pick = rng.choice(N_ITEM, size=TOPK, replace=False, p=p)
            expo[pick] += 1
            clicked = pick[rng.random(TOPK) < cp[pick]]
            if strategy == "naive":               # 高反馈直接当正样本
                pos_w[clicked] += 1.0
            else:                                  # 质量加权 + 营销号垂类降权
                w = qual[clicked] / np.maximum(bait[clicked], 1e-9)
                pos_w[clicked] += w
        score = 0.3 * score + 0.7 * (pos_w + 0.5)  # 平滑重训
        shown = expo > 0
        rows.append((rd + 1,
                     gini(expo),
                     entropy([expo[cat == c].sum() for c in range(N_CAT)]),
                     float((expo * (bait > 1)).sum() / expo.sum()),
                     float((expo * qual).sum() / expo.sum()),
                     int(shown.sum())))
    return rows


def diversity_beam(logp_fn, cat_of, beam, alpha, beta, pop, n_step=TOPK):
    """多样性约束 Beam Search: Score = Σ log P - alpha*类别重复 + beta*新颖性。

    每步从候选物品中扩展一个, 生成长度为 n_step 的推荐列表。
    """
    beams = [([], 0.0, 0.0)]                     # (已选列表, 带约束得分, 纯 logP)
    for _ in range(n_step):
        cand = []
        for seq, sc, raw in beams:
            lp = logp_fn(seq)
            order = np.argsort(-lp)[:beam * 3]   # 剪枝: 只看概率最高的若干物品
            for it in order:
                it = int(it)
                if it in seq:
                    continue
                rep = sum(1 for s in seq if cat_of[s] == cat_of[it])
                nov = -math.log(pop[it] + 1e-6)
                cand.append((seq + [it],
                             sc + float(lp[it]) - alpha * rep + beta * nov,
                             raw + float(lp[it])))
        beams = sorted(cand, key=lambda t: -t[1])[:beam]
    return beams[0]


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    cat, qual, bait = build_items(rng)

    print("=" * 94)
    print(f"物品 {N_ITEM} 个, 垂类 {N_CAT} 个, 营销号(诱导系数 2.6)占比 "
          f"{(bait > 1).mean():.1%}; 每用户曝光 Top-{TOPK}")
    print(f"全库真实质量均值 = {qual.mean():.4f}; "
          f"营销号真实质量均值 = {qual[bait > 1].mean():.4f}")

    print()
    print("表 1  数据循环仿真: 三种样本构建策略(12 轮闭环迭代)")
    print("-" * 94)
    print(f"{'轮次':>5}" + "".join(f"{n:>29}" for n in
                                  ("朴素(高反馈=正样本)", "质量加权+垂类降权",
                                   "质量加权+ε探索(ε=0.15)")))
    print(f"{'':>5}" + "".join(f"{'基尼/垂类熵/营销占比/质量':>29}" for _ in range(3)))
    res = {s: simulate(s, cat, qual, bait, np.random.default_rng(7))
           for s in ("naive", "quality", "quality+explore")}
    for i in range(12):
        line = f"{i + 1:>5}"
        for s in ("naive", "quality", "quality+explore"):
            _, g, e, b, q, _ = res[s][i]
            line += f"{f'{g:.3f}/{e:.3f}/{b:.3f}/{q:.3f}':>29}"
        print(line)
    print("-" * 94)
    for s, name in (("naive", "朴素"), ("quality", "质量加权+垂类降权"),
                    ("quality+explore", "质量加权+ε探索")):
        _, g, e, b, q, cov = res[s][-1]
        print(f"  末轮 {name:<18} 曝光基尼={g:.3f} 垂类熵={e:.3f}"
              f"(上限 {math.log(N_CAT):.3f}) 营销号曝光占比={b:.1%} "
              f"真实质量={q:.4f} 覆盖物品={cov}")
    print("正文: 朴素策略在闭环中持续强化高诱导内容 -> 曝光集中(基尼升高)、垂类熵下降;")
    print("      细粒度样本构建与探索机制可同时改善生态指标与真实质量。")

    # ---------------- 多样性约束 Beam Search ----------------
    base = rng.random(N_ITEM) * 2.0
    # [MOCK] 让模型偏好集中在 3 个垂类上, 制造聚集度过高的候选集
    logits = base + 2.5 * np.isin(cat, [0, 2, 5])
    pop_norm = np.exp(logits) / np.exp(logits).sum()

    def logp_fn(seq):
        # 已选物品置 -1e9 后直接算 log-softmax(而非先 exp 再取 log),
        # 避免 exp 下溢到 0 时 log(0) 触发 divide-by-zero
        z = logits.copy()
        z[list(seq)] = -1e9
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    print()
    print("表 2  Beam Search 多样性约束(生成 Top-10 列表)")
    print("-" * 94)
    print(f"{'beam':>6}{'alpha(类别重复罚)':>18}{'beta(新颖性)':>14}"
          f"{'覆盖垂类数':>12}{'最大同类占比':>14}{'Σlog P':>12}{'相对最优损失':>14}")
    best_raw = None
    for beam, alpha, beta in ((1, 0.0, 0.0), (5, 0.0, 0.0), (20, 0.0, 0.0),
                              (5, 0.5, 0.0), (5, 1.5, 0.0), (5, 1.5, 0.3),
                              (20, 1.5, 0.3)):
        seq, sc, raw = diversity_beam(logp_fn, cat, beam, alpha, beta, pop_norm)
        cc = Counter(int(cat[i]) for i in seq)
        if best_raw is None:
            best_raw = raw
        best_raw = max(best_raw, raw)
        print(f"{beam:>6}{alpha:>18.1f}{beta:>14.1f}{len(cc):>12}"
              f"{max(cc.values()) / TOPK:>14.1%}{raw:>12.3f}"
              f"{(best_raw - raw):>14.3f}")
    print("-" * 94)
    print("可见: 单纯增大束宽并不会提升多样性(贪心解已是概率最优), 必须在路径评分里")
    print("      显式加入类别重复惩罚与新颖性奖励, 才能用可控的 log P 损失换取覆盖度。")


if __name__ == "__main__":
    main()
