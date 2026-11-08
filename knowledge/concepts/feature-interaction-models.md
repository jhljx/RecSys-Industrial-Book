---
id: feature-interaction-models
title: 特征交互模型选型
chapter: 15
prerequisites: [feature-engineering]
---

## 定义
特征交互模型用显式交叉、域间注意力、双线性与协同作用等不同结构表达用户和物品的组合关系。

## 核心判断
DCN、AutoInt、FiBiNet、AFN 等方法解决的交互形态不同；应结合域数、稀疏度和在线预算比较。

## 工程检查
固定特征集合，逐项消融交互结构并记录长尾样本收益、查表开销和 P99。

## 本书依据
《第十五章-特征工程.pdf》，PDF 第 13–29 页（15.5 节）。

## 扩展说明
论文节点中的专属概念卡进一步说明每种交互算子的机制。
