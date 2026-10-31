---
id: listwise-ranking-loss
title: Listwise 排序损失
chapter: 12
prerequisites: [ranking]
---

## 核心判断
Listwise 损失把同一请求的候选列表作为联合优化对象，利用列表内标签与分数构造相对分布；它改善的是请求内排序监督，并不天然给出可直接用于多任务融合的绝对点击概率。

## 工程检查
固定请求分组键，核查全局 shuffle 是否打散列表、候选截断和标签是否引入未来信息；确认无正例列表的处理与请求级归一化。与 Pointwise、Pairwise 及组合损失在相同样本、模型、预算下对比列表指标、概率校准、损失尺度和线上排序收益。

## 本书依据
《第十二章-精排模型.pdf》，PDF 第 58–59 页（12.5.3 Listwise Loss）；组合与实践对比见 PDF 第 59–64 页。

## 扩展说明
Listwise 目标不必然优于 Pointwise；收益取决于监督序关系的质量与线上目标一致性。
