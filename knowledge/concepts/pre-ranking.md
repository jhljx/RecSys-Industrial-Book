---
id: pre-ranking
title: 粗排
chapter: 5
prerequisites: [recsys-funnel]
---

## 核心判断
粗排在大候选量和有限延迟下逼近精排偏好，目标是保留优质候选而非独立追求最终排序最优。

## 工程检查
同时评估 Top-K 保真、吞吐、P99 和与精排的一致性。

## 本书依据
《第五章-粗排模块.pdf》，PDF 第2页。

## 扩展说明
模型复杂度应由候选规模和预算反推。
