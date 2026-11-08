---
id: rocket-launching-training
title: Rocket Launching联合训练
chapter: 11
prerequisites: [pre-ranking-two-tower]
---

## 定义
Rocket Launching以较大辅助网络引导轻量网络训练，再部署轻量学生以控制成本。

## 核心判断
教师收益只有传递到最终学生且上线延迟可接受时才有意义，不能只报告教师指标。

## 工程检查
比较教师、学生和独训学生的同口径指标，并检查训练共享参数与服务模型体积。

## 本书依据
《第十一章-粗排模型.pdf》，PDF 第 25–27 页（11.3.2 节）。

## 扩展说明
将方法收益与同等候选预算、特征可用性和实际在线资源成本共同核验。
