---
id: bert4rec-masked-items
title: BERT4Rec双向掩码建模
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
BERT4Rec通过掩码物品预测利用行为前后文，学习双向序列表示。

## 核心判断
预训练时可看后文，线上预测时只能看已发生行为，必须明确训练任务与服务输入差异。

## 工程检查
检查掩码样本构造、线上可用特征边界和与单向序列模型的增量。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 90–92 页（10.4.3 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
