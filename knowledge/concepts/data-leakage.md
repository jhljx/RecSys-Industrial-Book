---
id: data-leakage
title: 样本穿越
chapter: 15
prerequisites: [feature-engineering]
---

## 核心判断
样本穿越指训练特征使用了预测时点之后的信息，会制造无法在线复现的离线收益。

## 工程检查
采用 point-in-time join，并对事件时间、处理时间和回填任务做边界测试。

## 本书依据
《第十五章-特征工程.pdf》，PDF 第37页附近。

## 扩展说明
随机切分不能替代时间穿越检查。
