---
id: mirrn-multigranularity
title: MIRRN 多粒度检索
chapter: 20
prerequisites: [sim-two-stage-search]
---

## 定义
MIRRN 从目标、近期和全局三个粒度检索历史，并在精炼后融合对应兴趣。

## 核心判断
目标、近期和全局兴趣的检索 Query 能互补，单一目标检索可能忽略稳定与局部偏好。

## 工程检查
分别消融三路 Query、精炼和融合，检查多路重复行为、耗时与兴趣转移人群。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 41–45 页（20.4.4 节）。

## 扩展说明
多路检索扩大覆盖，也可能重复提取行为并增加在线开销。
