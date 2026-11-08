---
id: sasrec-causal-attention
title: SASRec因果序列注意力
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
SASRec以因果自注意力从历史行为中选择和下一次预测相关的兴趣。

## 核心判断
兴趣变化快且历史混杂时注意力有益；序列变长也会提高计算和噪声成本。

## 工程检查
核对时间顺序与因果mask，对照不同历史长度下的下一物品召回和推理时延。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 89–90 页（10.4.3 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
