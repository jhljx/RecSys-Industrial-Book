---
id: rq-kmeans-tokenizer
title: RQ-KMeans 残差聚类编码
chapter: 9
prerequisites: [semantic-id]
---

## 定义
RQ-KMeans 在每层对上一层的残差做 KMeans 聚类，将逐层聚类中心索引作为物品的语义 ID。

## 核心判断
省去 Encoder 和 Decoder 的联合训练及 STE 近似，但码本质量仍依赖输入 Embedding 和聚类初始化。

## 工程检查
在相同码本大小下对比 RQ-VAE 的编码碰撞、重构误差、码本利用率与构建成本；检查 RQ-KMeans++ 初始化收益。

## 本书依据
《第九章-生成式推荐模块.pdf》，PDF 第 8–8 页（9.3.3 节）。

## 扩展说明
章节以 OneRec 为应用例，聚类编码本身不等同于 OneRec 全部生成模型。
