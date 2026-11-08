---
id: cofars-context-prototype-search
title: CoFARS 上下文原型筛选
chapter: 20
prerequisites: [context-temporal-prototypes]
---

## 定义
CoFARS 根据上下文偏好分布学习原型，以时序图聚合转移关系，并按目标上下文筛选历史子序列。

## 核心判断
CoFARS按目标场景匹配偏好原型，再筛选候选无关的历史子序列，适合多个候选共享上下文的请求。

## 工程检查
按场景比较原型匹配质量、上下文冷启、图更新与候选规模变化时的 P99。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 81–84 页（20.7.2 节）。

## 扩展说明
候选无关筛选可摊薄多候选请求成本，但场景切换和新上下文会降低原型命中质量。
