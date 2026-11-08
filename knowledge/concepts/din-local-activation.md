---
id: din-local-activation
title: DIN 目标局部激活
chapter: 20
prerequisites: [target-attention-ranking]
---

## 定义
DIN依据目标物品动态加权历史行为，区别于将所有行为做相同 Pooling。

## 核心判断
DIN依据目标物品动态加权历史行为，区别于将所有行为做相同 Pooling；它仍需为每个候选读取相关历史。

## 工程检查
同一候选和历史窗口比较均值池化与目标激活，报告多兴趣人群收益及候选数放大后的 P99。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 13–15 页（20.2.1 节）。

## 扩展说明
它仍需为每个候选读取相关历史。
