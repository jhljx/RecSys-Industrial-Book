---
id: unirec-cross-stage-preference
title: UniRec 跨阶段多任务偏好优化
chapter: 22
prerequisites: [pre-ranking-chain-consistency, multi-objective]
---
## 定义
UniRec 连接粗排和精排的多任务融合塔，在保留阶段输入和在线预算差异的同时，用阶段内及跨阶段偏好联合优化。
## 核心判断
只蒸馏下游分数无法直接协调两阶段融合能力；偏好对、任务权重与属性组奖励共同决定跨阶段对齐方向。
## 工程检查
验证粗排不会读取精排才可用的特征；检查 CPPA 任务权重、CAPP Top-K pair、AGRR 组内奖励与 Adapter 停梯度，并分阶段测 P99。
## 本书依据
《第二十二章-链路一致性建模.pdf》，PDF 第 25–30 页（22.1.5 节，UniRec 模型）。
## 扩展说明
章节代码使用 `[MOCK]` pXTR 和反馈；论文版次差异需要以当前书稿所引版本核对。
