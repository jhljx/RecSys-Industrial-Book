# 核心知识卡目录

本目录是独立撰写的导航层，不替代发布版 PDF。每张卡必须区分 `本书依据` 与 `扩展说明`，并以 PDF 页码作为公开引用坐标。

| ID | 主题 | 主要章节 | 前置知识 |
|---|---|---:|---|
| industrial-recsys-scope | 工业推荐系统全链路 | 0 | 无 |
| cold-start | 冷启动 | 1 | recsys-funnel |
| feedback-loop | 推荐反馈闭环 | 1 | industrial-recsys-scope |
| hybrid-architecture | 混合推荐架构 | 2 | recsys-funnel |
| recsys-funnel | 推荐漏斗 | 2 | 无 |
| ab-experiment | A/B 实验 | 3 | offline-online-evaluation |
| offline-online-evaluation | 离线与在线评价 | 3 | feedback-loop |
| filtering | 过滤 | 4 | multi-recall |
| multi-recall | 多路召回 | 4 | recsys-funnel |
| recall-fusion | 召回融合 | 4 | multi-recall |
| pre-ranking | 粗排 | 5 | recsys-funnel |
| quota-truncation | Quota 与候选截断 | 5 | pre-ranking |
| multi-objective | 多目标建模 | 6 | ranking |
| ranking | 精排 | 6 | pre-ranking |
| ranking-formula | 多目标排序公式 | 6 | multi-objective |
| diversity | 多样性 | 7 | reranking |
| diversity-reranking | 多样性重排 | 7 | reranking |
| reranking | 重排 | 7 | ranking |
| mixed-ranking | 混排 | 8 | reranking |
| score-calibration | 分数校准 | 8 | mixed-ranking |
| generative-recsys | 生成式推荐 | 9 | ranking |
| semantic-id | 语义 ID | 9 | generative-recsys |
| hard-negative | 困难负样本 | 10 | negative-sampling |
| negative-sampling | 负采样 | 10 | two-tower |
| two-tower | 双塔召回 | 10 | multi-recall |
| model-distillation | 粗排模型蒸馏 | 11 | pre-ranking |
| data-leakage | 样本穿越 | 15 | feature-engineering |
| feature-engineering | 特征工程 | 15 | recsys-funnel |
| feature-freshness | 特征时效性 | 15 | feature-engineering |
| feature-governance | 特征治理 | 15 | feature-engineering |
| model-release | 模型原子发布 | 16 | offline-pipeline |
| offline-pipeline | 推荐系统离线链路 | 16 | feature-engineering |
| point-in-time-join | Point-in-time Join | 16 | data-leakage |
| embedding-server | Embedding Server | 17 | two-tower |
| fallback | 降级与兜底 | 17 | overload-control |
| latency-budget | 延迟预算 | 17 | online-serving |
| observability | 推荐可观测性 | 17 | online-serving |
| online-serving | 推荐系统在线服务 | 17 | recsys-funnel |
| overload-control | 过载控制 | 17 | latency-budget |
| version-consistency | 模型与数据版本一致性 | 17 | embedding-server |
| ann-index | ANN 索引选择 | 18 | vector-retrieval |
| hnsw | HNSW | 18 | ann-index |
| quantization | 向量量化 | 18 | ann-index |
| vector-retrieval | 向量检索 | 18 | two-tower |
| exposure-bias | 曝光偏差 | 19 | offline-pipeline |
| position-bias | 位置偏差 | 19 | exposure-bias |
| position-debiasing | 位置偏差校正 | 19 | position-bias |
| consistency-modeling | 链路一致性建模 | 22 | version-consistency |
| duration-distribution | 时长分布建模 | 23 | duration-modeling |
| duration-modeling | 时长预估 | 23 | multi-objective |
| exploration | 探索与利用 | 28 | cold-start |
| rl-recsys | 强化学习推荐 | 28 | exploration |
| safe-exploration | 安全探索 | 28 | exploration |

知识图谱机器可读版本见 [`graph.json`](graph.json)。
