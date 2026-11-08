---
id: sequence-contrast-overview
title: 对比学习与反馈序列建模
chapter: 20
prerequisites: [sequential-recall]
---

## 定义
对比学习通过序列视图和行为反馈构造正负样本，辅助学习可区分的用户兴趣表示。

## 核心判断
InfoNCE、SimCLR、MoCo、SimSiam 在负样本来源与梯度路径上不同；反馈增强须保持用户真实意图。

## 工程检查
核对正负反馈含义、视图增强、辅助损失梯度范围及与主任务的权重。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 1–12 页（20.1 节）。

## 扩展说明
论文节点下保留各方法专属卡；此卡只作方法族导航。
