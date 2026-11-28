---
id: hybrid-value-rank-fusion
title: 值序混合融合
chapter: 25
prerequisites: [value-fusion-compensation, rank-fusion-robustness]
---
## 定义
同时使用任务预测值及其请求内排序位置，经各自变换后组合成最终分数。
## 核心判断
Value 保留预测差距，Rank 保留相对次序；两者混合有意义的前提是二者带来互补信息。
## 工程检查
在同一候选集做 Value、Rank、Hybrid 消融，检查校准变化、候选规模变化和额外请求级排序带来的 P99。
## 本书依据
《第二十五章-多任务融合.pdf》，PDF 第 3–4 页（25.2.3 节）。
## 扩展说明
如果输入预测已充分校准且 Rank 信息没有增益，混合公式会增加维护成本。
