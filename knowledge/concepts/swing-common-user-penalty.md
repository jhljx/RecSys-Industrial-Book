---
id: swing-common-user-penalty
title: Swing共同用户去噪
chapter: 10
prerequisites: [collaborative-filtering-recall]
---

## 定义
Swing在物品共现上引入共同用户之间的交互重合惩罚，降低活跃用户制造的虚假相似。

## 核心判断
当热门物品共同用户很多而兴趣区分度差时，比较Swing与ItemCF的长尾覆盖；惩罚过强可能削弱真实热门兴趣。

## 工程检查
按用户活跃度与物品热度分桶检查相似邻居质量，并核对在线共同用户集合更新成本。

## 本书依据
《第十章-召回模型.pdf》，PDF 第 9–15 页（10.1.4 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
