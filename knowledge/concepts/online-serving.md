---
id: online-serving
title: 推荐系统在线服务
chapter: 17
prerequisites: [recsys-funnel]
---

## 核心判断
在线服务将召回、排序和策略组织成受端到端延迟约束的请求链路。

## 工程检查
为每层定义 timeout、容量、过载行为、监控、版本和回滚。

## 本书依据
《第十七章-推荐系统在线服务.pdf》，PDF 第1页。

## 扩展说明
平均延迟不能代表尾延迟风险，应以 P99 设计。
