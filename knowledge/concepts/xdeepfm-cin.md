---
id: xdeepfm-cin
title: xDeepFM显式高阶交互
chapter: 12
prerequisites: [feature-crossing-deep-ranking]
---

## 定义
xDeepFM用CIN显式构造高阶特征交互，并与普通深层网络的隐式交互结合。

## 核心判断
高阶交互并非层数越多越好；在固定计算预算下比较CIN和DNN各自贡献。

## 工程检查
按交互阶数消融并记录推理成本，核对低频交叉上的过拟合。

## 本书依据
《第十二章-精排模型.pdf》，PDF 第 16–19 页（12.4.1 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
