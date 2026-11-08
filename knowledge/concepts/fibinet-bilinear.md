---
id: fibinet-bilinear
title: FiBiNet双线性交互
chapter: 15
prerequisites: [feature-crossing-deep-ranking, lhuc-hidden-unit-adaptation]
---

## 定义
FiBiNet结合特征域重要性估计与双线性函数，强化成对域的交互表达。

## 核心判断
域重要性与域对交互是两种不同信号，分别消融才能判断收益来源。LHUC调节隐藏单元贡献，FiBiNet的SE分支调节特征域贡献，二者属于第15章从动态贡献走向特征重加权的相关思想，不能视为相同结构。

## 工程检查
对照关闭重要性模块和双线性模块的效果，核算域数增加时的组合成本。

## 本书依据
《第十五章-特征工程.pdf》，PDF 第 16–18 页（15.5.4 节FiBiNet）；第 32–33 页（15.6.1 节LHUC与动态贡献的联系）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
