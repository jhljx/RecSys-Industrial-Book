---
id: pinsage-neighbor-sampling
title: PinSage重要邻居采样
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
PinSage结合随机游走的邻居重要性和图卷积，在大规模物品图上采样聚合。

## 核心判断
邻居预算是规模与召回质量的共同旋钮，热门节点过度进入邻域会吞噬长尾信号。

## 工程检查
核对采样分布、批处理吞吐、节点缓存命中及新增物品邻居的更新滞后。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 46–49 页（10.2.2 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
