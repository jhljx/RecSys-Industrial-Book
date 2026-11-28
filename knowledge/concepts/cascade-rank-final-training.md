---
id: cascade-rank-final-training
title: 粗排 Rank/Final 双目标训练
chapter: 22
prerequisites: [cascade-selection-bias, pre-ranking-chain-consistency]
---
## 定义
Rank 以精排排序 Top-K 为正例，Final 以最终重排 Top-K 为正例；两目标共享级联候选但允许同一物品标签不同。
## 核心判断
过高训练 AUC 可能只说明负样本太容易；Hard Negative、共享层数、Focal Loss 和正例位置权重应协同设计。
## 工程检查
按请求统计 Top/Bottom/中间段及未进精排负样本；比较双目标梯度、分来源 AUC、下游保留率和线上收益，检查位置权重只作用于损失。
## 本书依据
《第二十二章-链路一致性建模.pdf》，PDF 第 30–35 页（22.2 节）；第 37 页（22.4 节）。
## 扩展说明
一致性是提高候选传递效率的中间目标；持续加大权重可能压缩粗排兴趣发现能力。
