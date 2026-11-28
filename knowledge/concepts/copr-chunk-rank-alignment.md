---
id: copr-chunk-rank-alignment
title: COPR 分块采样与排序对齐
chapter: 22
prerequisites: [pre-ranking-chain-consistency]
---
## 定义
COPR 从精排顺序分块抽样，使用可插拔的排序对齐模块学习粗排与精排的 ECPM 次序，并以 ΔNDCG 加权排序损失。
## 核心判断
直接全列表对齐既昂贵又受简单样本主导；分块和排序增益权重让训练集中在影响下游排序的物品对。
## 工程检查
确认块按同一请求和下游排序构造，核对 ECPM 中 CTR、出价及校准口径；对比分块数、物品对数量、ΔNDCG 权重和线上延迟。
## 本书依据
《第二十二章-链路一致性建模.pdf》，PDF 第 9–12 页（22.1.3 节，COPR 模型）。
## 扩展说明
下游教师排序发生变化时需重新评价对齐目标；教学出价与反馈是 `[MOCK]`。
