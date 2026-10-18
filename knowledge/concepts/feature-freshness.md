---
id: feature-freshness
title: 特征时效性
chapter: 15
prerequisites: [feature-engineering]
---

## 核心判断
特征新鲜度是事件发生到在线可见的总延迟，过期特征会使模型响应滞后。

## 工程检查
分解采集、计算、写入、缓存和读取延迟并设置 freshness SLO。

## 本书依据
《第十五章-特征工程.pdf》，PDF 第12页。

## 扩展说明
更实时不一定更优，应权衡稳定性与成本。
