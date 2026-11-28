---
id: hccp-hybrid-stage-sampling
title: HCCP 跨阶段分层采样
chapter: 22
prerequisites: [cascade-selection-bias, ranktower-cross-tower-interaction]
---
## 定义
HCCP 组合曝光、下游候选、召回尾部、其他请求与全局物品池等多层样本，以全局/局部排序及对比目标协调粗排。
## 核心判断
不同来源负样本难度和业务语义不同；局部监督只适用于有反馈的样本，不能把全池未曝光当成已观测负反馈。
## 工程检查
逐层核对 N1–N5 数量、来源和去重；检查全局 ListMLE、局部 ListMLE/BCE 与 Margin InfoNCE 的样本掩码及梯度。
## 本书依据
《第二十二章-链路一致性建模.pdf》，PDF 第 18–22 页（22.1.4 节，HCCP 模型）。
## 扩展说明
真实召回尾部和物品池采样需要可追溯的数据服务；章节示例使用 `[MOCK]` 请求日志。
