---
id: lightgcn-propagation
title: LightGCN邻居传播
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
LightGCN简化图卷积中的特征变换和非线性激活，通过多层邻居传播汇集协同信号。

## 核心判断
层数越深越可能把不同兴趣混在一起，应检查各层贡献而非盲目堆叠。

## 工程检查
对比传播层数、长尾节点收益和图快照时效，记录离线传播与在线向量版本。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 44–46 页（10.2.2 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
