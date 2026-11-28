---
id: cs3-online-capability-synergy
title: CS3 在线双塔能力协同
chapter: 22
prerequisites: [pre-ranking-two-tower, version-consistency]
---
## 定义
CS3 以循环自适应结构修正单塔特征，跨塔同步交换用户与物品信息，并复用下游级联模型的中间表示。
## 核心判断
双塔的独立表示限制交互能力；引入在线同步和下游表示后，收益必须覆盖缓存更新、版本一致与尾延迟成本。
## 工程检查
分别消融 CAS、CTS、CMS；核对正负反馈的更新条件、EMA 时序、表示版本、缓存失效和服务降级。
## 本书依据
《第二十二章-链路一致性建模.pdf》，PDF 第 23–25 页（22.1.5 节，CS3 模型）。
## 扩展说明
章节代码用本地字典 `[MOCK]` 缓存，不包含分布式参数服务与并发更新协议。
