---
id: flashattention-io
title: FlashAttention访存优化
chapter: 12
prerequisites: [feature-crossing-deep-ranking]
---

## 定义
FlashAttention通过分块计算精确注意力减少高带宽内存读写，而非改变注意力目标。

## 核心判断
只有序列长度和硬件条件使访存成为瓶颈时，其吞吐优势才能兑现。

## 工程检查
在目标序列长度上比较数值误差、峰值显存与实际训练及推理吞吐。

## 本书依据
《第十二章-精排模型.pdf》，PDF 第 50–51 页（12.4.3 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
