---
id: generative-token-objective
title: 生成式推荐序列目标
chapter: 9
prerequisites: [generative-recsys, semantic-id]
---

## 定义
生成式推荐序列目标是根据已有用户行为和离散物品表示，训练模型预测后续 token 的目标函数。

## 核心判断
将用户行为和语义 ID 组织为模型输入后，自回归模型学习下一个 token 的条件分布；训练 token 预测能力不等同于生成合法物品及获得业务收益。

## 工程检查
核对用户行为序列截断、语义 ID 分层顺序、训练目标与线上解码口径；测量非法 token、预测质量及位置相关偏差。

## 本书依据
《第九章-生成式推荐模块.pdf》，PDF 第 10–12 页（9.4 Transformer 模型及生成训练目标）。

## 扩展说明
曝光偏差与业务约束须在评估中单独处理，不能仅以训练损失代替线上检验。
