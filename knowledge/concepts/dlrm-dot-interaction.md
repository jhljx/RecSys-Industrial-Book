---
id: dlrm-dot-interaction
title: DLRM点积交互
chapter: 12
prerequisites: [feature-crossing-deep-ranking]
---

## 定义
DLRM先把稀疏特征转成向量，再用特征间点积构造交互送入顶层网络。

## 核心判断
Embedding查表常比交互层本身更影响线上成本，应同时评估内存访问与模型计算。

## 工程检查
记录各特征表访问频率、批量查表吞吐及交互层增量效果。

## 本书依据
《第十二章-精排模型.pdf》，PDF 第 19–22 页（12.4.1 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
