---
id: etegrec-end-to-end-token
title: ETEGRec 端到端物品 Token
chapter: 9
prerequisites: [one-stage-generative-alignment]
---

## 定义
ETEGRec 联合学习物品离散 Token 与生成式推荐，使 Token 的构建受到下游推荐目标约束。

## 核心判断
联合训练目标有助于对齐，但物品编码更新会影响历史行为 Token 与物品反查映射。

## 工程检查
验证编码更新后的历史序列兼容性、Token 利用率、推荐增益与训练开销。

## 本书依据
《第九章-生成式推荐模块.pdf》，PDF 第 17–20 页（9.6.2 节）。

## 扩展说明
应把 Tokenizer 版本迁移计入一阶段方案的工程成本。
