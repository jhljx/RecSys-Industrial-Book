---
id: pepnet-gate-nu
title: PEPNet先验驱动的双层门控
chapter: 24
prerequisites: [personalized-network-parameters, lhuc-hidden-unit-adaptation]
---

## 定义
PEPNet用Gate NU把个性化先验转成隐藏单元门控：EPNet按场景调制共享Embedding，PPNet按用户、物品、作者先验调节多任务塔，并配合大规模稀疏参数管理。

## 核心判断
Gate NU借鉴第15章LHUC的隐藏单元贡献调制，但针对推荐的多场景多任务设计；EPNet门控输入的停止梯度不等于冻结共享Embedding，PPNet与EPNet各司其职。

## 工程检查
分别消融EPNet与PPNet；按场景、用户群及任务监控门控饱和、Embedding更新、梯度路径、稀疏ID覆盖和在线P99。

## 本书依据
《第二十四章-个性化网络参数.pdf》，PDF 第 10–11 页（24.3.1 节Gate NU）；PDF 第 11 页（24.3.2 节EPNet）；PDF 第 11–12 页（24.3.3 节PPNet）；PDF 第 12 页（24.3.4 节工业部署）。LHUC原理另见前置概念卡的第15章引文。

## 扩展说明
LHUC的原始任务是语音自适应，与PEPNet并非相同模型；先验字段需保证在线可用且版本一致。
