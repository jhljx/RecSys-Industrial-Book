---
id: latency-budget
title: 延迟预算
chapter: 17
prerequisites: [online-serving]
---

## 核心判断
端到端延迟预算必须分配给各阶段并预留网络、调度和组装余量，不能让子服务各自使用总预算。

## 工程检查
同时设置请求、模块和 RPC 三层 timeout，验证 P99 下仍能完成兜底。

## 本书依据
《第十七章-推荐系统在线服务.pdf》，PDF 第4页。

## 扩展说明
预算数值必须由真实压测确定。
