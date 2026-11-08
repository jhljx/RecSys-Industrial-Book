---
id: sequence-contrast-augmentation
title: 序列对比视图增强
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
CL4SRec对同一序列构造两种增强视图，使序列表征对合理扰动保持接近。

## 核心判断
裁剪或重排若改变了原始意图，正样本就会误导模型，增强强度应服从行为语义。

## 工程检查
分短序列与稀疏兴趣用户检查增强有效率，比较无对比损失的召回质量。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 96–97 页（10.4.5 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
