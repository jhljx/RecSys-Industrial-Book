---
id: unisid-joint-embedding
title: UniSID 语义 ID 与内容表示联合生成
chapter: 9
prerequisites: [one-stage-generative-alignment]
---

## 定义
UniSID 使用统一多模态模型联合产生物品 Embedding 和分层语义 ID，并对不同语义粒度施加训练约束。

## 核心判断
以 argmax 生成离散 ID 不需要 RQ-VAE 最近邻码本；但不同层级的语义区分度和推荐目标仍需联合验收。

## 工程检查
检查各层 ID 的区分能力、碰撞率、多粒度对比目标、内容缺失与反向物品映射。

## 本书依据
《第九章-生成式推荐模块.pdf》，PDF 第 20–22 页（9.6.3 节）。

## 扩展说明
不要把 UniSID 的直接预测 ID 机制解释为另一种 RQ-VAE 码本查找。
