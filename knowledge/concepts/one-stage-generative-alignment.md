---
id: one-stage-generative-alignment
title: 一阶段生成中编码与推荐目标对齐
chapter: 9
prerequisites: [semantic-id, generative-token-objective]
---

## 核心判断
先量化物品再训练生成器可能出现编码目标与推荐目标不一致；一阶段方法尝试联合或交替优化离散表示与生成学习，选型应看对齐收益是否覆盖训练和服务复杂度。

## 工程检查
与冻结 tokenizer 基线比较推荐增益、码本使用、碰撞、训练稳定性和更新成本；分清不同方法的可微化与对齐机制，避免将论文方法直接视为现成生产方案。

## 本书依据
《第九章-生成式推荐模块.pdf》，PDF 第 14–22 页（9.6 DIGER、ETEGRec、UniSID）。

## 扩展说明
在相同延迟和索引维护预算下再与级联方案比较；一阶段训练不免除在线合法性校验。
