---
id: semantic-id
title: 语义 ID
chapter: 9
prerequisites: [generative-recsys]
---

## 核心判断
语义 ID 将物品表示为离散 Token 序列，使生成模型能够直接生成候选物品标识，同时把量化误差和 ID 合法性带入链路。

## 工程检查
评估碰撞、非法 ID、新物品编码、增量更新、码本版本和生成到物品映射的一致性。

## 本书依据
《第九章-生成式推荐模块.pdf》，PDF 第 4–9 页。

## 扩展说明
选型必须与同预算双塔加 ANN 基线比较，而不是只比较模型结构。
