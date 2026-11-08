---
id: eta-simhash-retrieval
title: ETA SimHash 行为检索
chapter: 20
prerequisites: [sequence-hash-aggregation]
---

## 定义
ETA用目标与行为的哈希距离筛选历史，再做精确注意力。

## 核心判断
ETA用目标与行为的哈希距离筛选历史，再做精确注意力；近似检索的漏检应计入整体排序指标。

## 工程检查
对照精确相似度检索，测哈希位数、候选 Top-K、召回率、P99 和冷热数据切换。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 59–62 页（20.5.2 节）。

## 扩展说明
近似检索的漏检应计入整体排序指标。
