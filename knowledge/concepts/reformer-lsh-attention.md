---
id: reformer-lsh-attention
title: Reformer LSH 注意力
chapter: 20
prerequisites: [sequence-hash-aggregation]
---

## 定义
LSH把全量两两注意力限制在相似哈希桶内，并配合可逆结构节省内存。

## 核心判断
LSH把全量两两注意力限制在相似哈希桶内，并配合可逆结构节省内存；不是直接的 CTR 模型。

## 工程检查
测桶数、哈希轮数、碰撞漏检率、长序列显存及与精确注意力的质量差。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 56–59 页（20.5.1 节）。

## 扩展说明
不是直接的 CTR 模型。
