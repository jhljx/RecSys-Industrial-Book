---
id: point-in-time-join
title: Point-in-time Join
chapter: 16
prerequisites: [data-leakage]
---

## 核心判断
样本关联特征时只能使用标签时刻已经可见的版本；事件发生时间晚于标签或当时尚未到达的数据都会造成穿越。

## 工程检查
同时保存 event time、available time、label time、回填版本和 join 规则，并设置未来特征阻断门禁。

## 本书依据
《第十六章-推荐系统离线服务.pdf》，PDF 第 1–20 页。

## 扩展说明
流式与批式链路应共享时间语义，而不是仅比较最终字段值。
