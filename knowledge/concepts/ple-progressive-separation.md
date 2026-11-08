---
id: ple-progressive-separation
title: PLE渐进共享与专属专家
chapter: 12
prerequisites: [feature-crossing-deep-ranking]
---

## 定义
PLE分层设置共享与任务专属专家，让任务共性和差异逐步抽取。

## 核心判断
任务专属容量过低无法缓解冲突，容量过高则损失迁移并增加服务成本。

## 工程检查
逐层检查共享门控与专属专家利用率，对比MMoE的多目标收益和P99。

## 本书依据
《第十二章-精排模型.pdf》，PDF 第 33–36 页（12.4.2 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
