---
id: disenctr-dynamic-interest-graph
title: DisenCTR 动态兴趣图
chapter: 20
prerequisites: [context-temporal-prototypes]
---

## 定义
DisenCTR 在动态用户物品图上解耦多个兴趣分量，并结合时间影响预测交互。

## 核心判断
动态用户物品图中的高阶协同与多兴趣解耦可以补充单用户序列，时间影响需与图传播共同评估。

## 工程检查
核查图快照时间、邻居采样、兴趣分量独立性、Hawkes 时间项与增量图维护。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 78–81 页（20.7.1 节）。

## 扩展说明
高阶图关系可补充个人序列，但旧图快照与时间泄漏会扭曲动态兴趣。
