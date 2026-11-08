---
id: muse-stagewise-modality
title: MUSE 分阶段模态融合
chapter: 20
prerequisites: [multimodal-sim-search]
---

## 定义
GSU 的多模态相似度检索和 ESU 的细粒度模态融合承担不同任务。

## 核心判断
GSU 的多模态相似度检索和 ESU 的细粒度模态融合承担不同任务；复杂融合应放在有实际收益的阶段。

## 工程检查
分别消融 GSU、ESU 的 ID 与模态信号，测模态缺失、重复检索、吞吐与尾延迟。

## 本书依据
《第二十章-用户兴趣与序列建模.pdf》，PDF 第 74–77 页（20.6.4 节）。

## 扩展说明
复杂融合应放在有实际收益的阶段。
