---
id: sdim-collision-aggregation
title: SDIM 碰撞聚合
chapter: 20
prerequisites: [eta-simhash-retrieval]
---

## 定义
SDIM用多轮哈希碰撞近似目标关注权重，并可把行为编码移到独立服务。

## 核心判断
SDIM用多轮哈希碰撞近似目标关注权重，并可把行为编码移到独立服务；不再对命中行为逐一运行标准 Target Attention。

## 工程检查
测桶内偏差、哈希次数、编码服务时效、跨请求复用和异常降级。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 64–66 页（20.5.3 节）。

## 扩展说明
不再对命中行为逐一运行标准 Target Attention。
