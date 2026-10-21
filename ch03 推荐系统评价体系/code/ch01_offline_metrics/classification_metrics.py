"""对应正文第 3 章 §3.1.1「分类指标」。

本实现做了什么
---------------
1. 从混淆矩阵 (TP / FP / FN / TN) 出发，实现正文给出的四个分类指标；
2. 复刻正文的论断「短视频场景 CTR 仅 1%，把所有样本全部预测为未点击，
   Accuracy 仍能达到 99%」——用合成的极度不均衡数据跑出这个数字，
   并展示此时 Precision / Recall / F1 全部退化为 0（或无定义）；
3. 做一次阈值扫描，说明分类指标依赖阈值，而 §3.1.2 的 AUC 不依赖阈值。

关键公式（正文式 3.1~3.4）
--------------------------
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)

F1 是 Precision 与 Recall 的调和平均：1/F1 = (1/P + 1/R) / 2。
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Confusion:
    """二分类混淆矩阵。tp/fp/fn/tn 对应真正例/假正例/假负例/真负例。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def accuracy(self) -> float:
        if self.total == 0:
            return float("nan")
        return (self.tp + self.tn) / self.total

    def precision(self) -> float:
        # 模型没有预测出任何正样本时 Precision 无定义，工程上通常记 0
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def report(self) -> str:
        return (
            f"TP={self.tp:<6d} FP={self.fp:<6d} FN={self.fn:<6d} TN={self.tn:<6d} | "
            f"Acc={self.accuracy():.4f} P={self.precision():.4f} "
            f"R={self.recall():.4f} F1={self.f1():.4f}"
        )


def confusion_matrix(labels: list[int], scores: list[float], threshold: float) -> Confusion:
    """按阈值把打分二值化后统计混淆矩阵。"""
    cm = Confusion()
    for y, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if y == 1 and pred == 1:
            cm.tp += 1
        elif y == 0 and pred == 1:
            cm.fp += 1
        elif y == 1 and pred == 0:
            cm.fn += 1
        else:
            cm.tn += 1
    return cm


def make_ctr_samples(n: int, pos_rate: float, signal: float) -> tuple[list[int], list[float]]:
    """合成 CTR 数据集。

    label ~ Bernoulli(pos_rate)；打分 = 基础概率 + signal * label + 噪声，
    signal 越大模型区分能力越强。返回 (labels, scores)，scores 已裁剪到 [0, 1]。
    """
    labels: list[int] = []
    scores: list[float] = []
    for _ in range(n):
        y = 1 if random.random() < pos_rate else 0
        base = random.gauss(pos_rate, 0.02)
        s = base + signal * y + random.gauss(0.0, 0.01)
        labels.append(y)
        scores.append(min(1.0, max(0.0, s)))
    return labels, scores


def all_negative_baseline(labels: list[int]) -> Confusion:
    """「把所有样本都预测为未点击」的平凡模型。"""
    pos = sum(labels)
    return Confusion(tp=0, fp=0, fn=pos, tn=len(labels) - pos)


def main() -> None:
    random.seed(42)

    print("=" * 78)
    print("§3.1.1 分类指标：Accuracy / Precision / Recall / F1")
    print("=" * 78)

    # ---------- (1) 均衡场景：分类指标正常工作 ----------
    labels, scores = make_ctr_samples(n=4000, pos_rate=0.50, signal=0.30)
    print(f"\n[均衡场景] 样本 {len(labels)} 条，正样本率 {sum(labels)/len(labels):.4f}")
    cm = confusion_matrix(labels, scores, threshold=0.50)
    print("  阈值 0.50 ->", cm.report())

    # ---------- (2) 极度不均衡场景：Accuracy 失效 ----------
    labels, scores = make_ctr_samples(n=100000, pos_rate=0.01, signal=0.30)
    pos = sum(labels)
    print(f"\n[不均衡场景] 样本 {len(labels)} 条，正样本 {pos} 条"
          f"（正样本率 {pos/len(labels):.4f}，模拟短视频 1% CTR）")

    trivial = all_negative_baseline(labels)
    print("  全预测未点击 ->", trivial.report())
    print(f"  >>> Accuracy 高达 {trivial.accuracy():.4f}，但 Recall={trivial.recall():.4f}、"
          f"F1={trivial.f1():.4f}，模型没有任何推荐能力。")
    print("  >>> 正文结论：高 Accuracy 并不意味着模型具备良好的推荐能力，")
    print("      Accuracy 在推荐场景只能作为辅助观察指标。")

    # 一个真正有区分能力的模型，在同样数据上按合适阈值评估
    print("\n  同一批数据上，有区分能力的模型按不同阈值评估：")
    print(f"  {'阈值':>6} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}")
    print("  " + "-" * 54)
    for th in (0.02, 0.05, 0.10, 0.20, 0.31, 0.50):
        c = confusion_matrix(labels, scores, threshold=th)
        print(f"  {th:>6.2f} | {c.accuracy():>8.4f} | {c.precision():>9.4f} | "
              f"{c.recall():>8.4f} | {c.f1():>8.4f}")
    print("  >>> 四个分类指标全部依赖阈值；阈值一变结论就变，")
    print("      这也是工业界更偏好与阈值无关的 AUC/GAUC 的原因（见 §3.1.2）。")

    # ---------- (3) Precision 与 Recall 的冲突，F1 给出均衡评价 ----------
    print("\n[P/R 冲突] 两个模型，一个保守一个激进：")
    conservative = Confusion(tp=30, fp=5, fn=170, tn=9795)   # 少推但很准
    aggressive = Confusion(tp=170, fp=1800, fn=30, tn=8000)  # 多推但很脏
    for name, c in (("保守模型", conservative), ("激进模型", aggressive)):
        print(f"  {name}:", c.report())
    print(f"  >>> 保守模型 Precision 更高（{conservative.precision():.4f} vs "
          f"{aggressive.precision():.4f}），")
    print(f"      激进模型 Recall 更高（{aggressive.recall():.4f} vs "
          f"{conservative.recall():.4f}）；")
    print(f"      F1 给出均衡结论：{conservative.f1():.4f} vs {aggressive.f1():.4f}。")
    print("  >>> 正文口径：召回阶段重点看 Recall（Recall 过低会让潜在感兴趣内容")
    print("      在排序前就被过滤掉）；F1 在召回模型评估、冷启动、样本极度不均衡时有参考价值。")


if __name__ == "__main__":
    main()
