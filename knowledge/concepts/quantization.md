---
id: quantization
title: 向量量化
chapter: 18
prerequisites: [ann-index]
---

## 核心判断
向量量化用精度损失换取内存和距离计算效率，误差可能改变近邻顺序。

## 工程检查
同时比较压缩率、Recall@K、P99 和重建误差。

## 本书依据
《第十八章-向量检索系统.pdf》，PDF 第 33 页起。

## 扩展说明
可用原始向量对小候选集二次精排以回收精度。
