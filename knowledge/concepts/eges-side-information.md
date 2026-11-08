---
id: eges-side-information
title: EGES侧信息融合
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
EGES将物品ID向量与侧信息向量按可学习权重融合，缓解仅靠交互边的稀疏性。

## 核心判断
新物品缺少交互时侧信息有价值；侧信息质量差时可能将不同商品错误拉近。

## 工程检查
分新老物品比较召回，核对侧信息缺失率、类别污染和向量版本。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 30–36 页（10.2.1 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
