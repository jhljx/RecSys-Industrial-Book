---
id: neural-cf-interaction
title: Neural CF非线性交互
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
Neural CF用可学习的用户与物品向量交互替代固定内积，NeuMF组合线性与非线性交互。

## 核心判断
若简单矩阵分解已经充分拟合稀疏交互，增加MLP未必改善冷启动与尾部召回。

## 工程检查
固定训练样本和候选预算，对照MF与NeuMF在稀疏用户及尾部物品上的收益。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 18–21 页（10.1.6 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
