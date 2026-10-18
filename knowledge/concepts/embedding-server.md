---
id: embedding-server
title: Embedding Server
chapter: 17
prerequisites: [two-tower]
---

## 核心判断
Embedding Server 将大规模稀疏参数或向量从模型计算中解耦，服务设计重点是版本、缓存和批量访问。

## 工程检查
请求携带版本，缓存键包含版本，模型和向量切换必须原子可回滚。

## 本书依据
《第十七章-推荐系统在线服务.pdf》，PDF 第17页。

## 扩展说明
真实存储选型取决于向量规模、更新频率和容灾要求。
