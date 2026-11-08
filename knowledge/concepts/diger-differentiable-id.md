---
id: diger-differentiable-id
title: DIGER 可微语义 ID
chapter: 9
prerequisites: [one-stage-generative-alignment]
---

## 定义
DIGER 通过可微离散化机制让语义 ID 学习与后续生成推荐目标联合优化。

## 核心判断
一阶段可缓解冻结 tokenizer 与推荐目标错配，但离散采样近似可能带来训练方差和码本不稳定。

## 工程检查
与冻结 RQ-VAE 两阶段基线比较生成指标、码本熵、梯度稳定性和推理成本。

## 本书依据
《第九章-生成式推荐模块.pdf》，PDF 第 14–16 页（9.6.1 节）。

## 扩展说明
训练可微化不能替代线上对无效 ID 与索引版本的检查。
