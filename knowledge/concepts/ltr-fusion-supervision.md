---
id: ltr-fusion-supervision
title: LTR 融合的输入与监督
chapter: 25
prerequisites: [hybrid-value-rank-fusion, ranking-ltr-choice]
---
## 定义
将多目标预测值、可选 Rank 和上下文送入排序模型，以真实用户反馈或综合效用训练统一融合分数。
## 核心判断
上游 pxtr 是输入而非真实监督；Pointwise、Pairwise、Listwise 的选择取决于标签和请求分组质量。
## 工程检查
按请求组织曝光与反馈，核对延迟标签归因、未曝光候选处理、训练服务预测版本，以及各目标与列表指标。
## 本书依据
《第二十五章-多任务融合.pdf》，PDF 第 10–11 页（25.5.1 节）。
## 扩展说明
综合效用标签和多任务损失都体现业务取舍，LTR 不会自动消除目标冲突。
