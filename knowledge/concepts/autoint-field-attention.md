---
id: autoint-field-attention
title: AutoInt特征域注意力
chapter: 15
prerequisites: [feature-crossing-deep-ranking]
---

## 定义
AutoInt用多头自注意力学习不同特征域之间的交互权重。

## 核心判断
注意力权重易受高频域主导，域交互可解释性也不能直接等同于因果重要性。

## 工程检查
检查域对注意力分布、低频域效果和在线交互计算开销。

## 本书依据
《第十五章-特征工程.pdf》，PDF 第 15–16 页（15.5.3 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
