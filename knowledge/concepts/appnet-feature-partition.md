---
id: appnet-feature-partition
title: APPNet自动特征划分与参数门控
chapter: 24
prerequisites: [personalized-network-parameters, pepnet-gate-nu, apg-conditional-parameters]
---

## 定义
APPNet在共享特征表示上以字段级或Embedding维度级AFP学习梯度归属，利用直通估计和停止梯度分别训练个性化门控与主预测分支，结合PPNet式隐藏单元门控和曝光空间的点击转化联合目标。

## 核心判断
两个分支前向仍能看到完整表示，掩码控制的是梯度；维度级划分更细但参数更多。全空间联合目标用CTR×CVR约束曝光到转化的概率，而非把未点击曝光当作CVR负例。

## 工程检查
比较人工先验、字段级与维度级AFP，监控掩码密度、STE权重梯度、门控饱和、CTR/联合转化率分群校准及尾延迟。

## 本书依据
《第二十四章-个性化网络参数.pdf》，PDF 第 14–16 页（24.4.1 节AFP与梯度划分）；PDF 第 16–17 页（24.4.2 节PPNet门控）；PDF 第 16–17 页（24.4.3 节全空间多任务目标）。

## 扩展说明
AFP阈值与二值近似需做稳定性检查；联合零标签不表示观察到点击后的CVR结果。
