---
id: harmonrank-rank-consistency
title: HarmonRank 排序目标一致性
chapter: 25
prerequisites: [ltr-fusion-supervision, dynamic-fusion-constraints]
---
## 定义
HarmonRank 用可微排序近似多目标 AUC，融合网络并行建模目标间关系、目标独立贡献和线性路径。
## 核心判断
分类损失与最终排序指标可能不一致；关系感知注意力和关系无关门控分别保留共享及独有信息，线性路径提供基础表达。
## 工程检查
在固定请求候选和多目标标签上验证排序投影的梯度及约束；对照 AUC、分目标收益、三分支消融、计算成本和线上 Top-K 行为。
## 本书依据
《第二十五章-多任务融合.pdf》，PDF 第 24–29 页（25.6.4 节）。
## 扩展说明
可微代理目标与真实线上效用仍有差距，需结合实验护栏判定收益。
