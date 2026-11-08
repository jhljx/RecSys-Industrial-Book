---
id: minhash-jaccard
title: MinHash集合相似检索
chapter: 18
prerequisites: [lsh-retrieval]
---

## 定义
MinHash用随机排列下的最小哈希值近似集合的Jaccard相似度。

## 核心判断
集合交集是否是业务真正的相似性定义应先确认；把连续语义向量直接套入集合哈希会改变问题。

## 工程检查
在固定签名长度下测Jaccard估计误差与候选数，比较集合稀疏度分桶。

## 本书依据
《第十八章-向量检索系统.pdf》，PDF 第 19–24 页（18.3.1 节）。

## 扩展说明
签名长度决定方差和存储，不存在通用的最优参数。
