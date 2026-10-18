---
id: overload-control
title: 过载控制
chapter: 17
prerequisites: [latency-budget]
---

## 核心判断
限流和熔断用于阻止过载与故障扩散，触发条件必须早于资源完全耗尽。

## 工程检查
明确入口限流、依赖熔断、恢复窗口和每种拒绝原因指标。

## 本书依据
《第十七章-推荐系统在线服务.pdf》，PDF 第3页。

## 扩展说明
排队会放大尾延迟，不能作为无限缓冲。
