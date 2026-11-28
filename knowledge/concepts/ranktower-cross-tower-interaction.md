---
id: ranktower-cross-tower-interaction
title: RankTower 双向跨塔交互
chapter: 22
prerequisites: [pre-ranking-two-tower, cascade-selection-bias]
---
## 定义
RankTower 用多头门控产生多视角用户和物品表示，双向门控交叉注意力建模两塔关系，再结合全阶段样本和混合排序损失。
## 核心判断
跨塔注意力增加表达能力，但只有单侧表示可预计算；在线配对计算与粗排候选规模共同决定是否可部署。
## 工程检查
检查用户查询物品与物品查询用户两个方向、缓存前后分数一致性、阶段样本范围和混合损失分量；测量候选数下的 P99。
## 本书依据
《第二十二章-链路一致性建模.pdf》，PDF 第 14–18 页（22.1.4 节，RankTower 模型）。
## 扩展说明
章节代码对论文 AM-Rankmax 的零违反情形做了显式数值稳定化，并标注 `[MOCK]` 数据。
