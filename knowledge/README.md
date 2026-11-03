# 核心知识卡目录

本目录是独立撰写的导航层，不替代发布版 PDF。每张卡必须区分 `本书依据` 与 `扩展说明`，并以 PDF 页码作为公开引用坐标。

| ID | 主题 | 主要章节 | 前置知识 |
|---|---|---:|---|
| industrial-recsys-scope | 工业推荐系统全链路 | 0 | 无 |
| cold-start | 冷启动 | 1 | recsys-funnel |
| feedback-loop | 推荐反馈闭环 | 1 | industrial-recsys-scope |
| creator-side-recommendation | 作者侧推荐与供给曝光 | 2 | hybrid-architecture |
| hybrid-architecture | 混合推荐架构 | 2 | recsys-funnel |
| offline-online-architecture | 离线生产与在线请求的交付边界 | 2 | recsys-funnel |
| recommendation-log-contract | 推荐请求日志与反馈数据契约 | 2 | offline-online-architecture |
| recsys-funnel | 推荐漏斗 | 2 | 无 |
| ab-experiment | A/B 实验 | 3 | offline-online-evaluation |
| offline-online-evaluation | 离线与在线评价 | 3 | feedback-loop |
| ranking-evaluation-metrics | 全局区分与列表头部评价 | 3 | offline-online-evaluation |
| filtering | 过滤 | 4 | multi-recall |
| multi-recall | 多路召回 | 4 | recsys-funnel |
| recall-concurrency-isolation | 并行召回的超时与空路隔离 | 4 | multi-recall |
| recall-fusion | 召回融合 | 4 | multi-recall |
| recall-route-selection | 召回技术路线选择 | 4 | multi-recall |
| pre-ranking | 粗排 | 5 | recsys-funnel |
| pre-ranking-chain-consistency | 粗排与精排链路一致性 | 5 | pre-ranking |
| pre-ranking-objective-fusion | 粗排多目标值融合与序融合 | 5 | pre-ranking |
| pre-ranking-set-optimization | 粗排候选集集合优化 | 5 | pre-ranking, quota-truncation |
| quota-truncation | Quota 与候选截断 | 5 | pre-ranking |
| multi-objective | 多目标建模 | 6 | ranking |
| ranking | 精排 | 6 | pre-ranking |
| ranking-formula | 多目标排序公式 | 6 | multi-objective |
| ranking-ltr-choice | 行为预估与 LTR 目标选型 | 6 | ranking |
| ranking-traffic-control | 精排流量调控与强约束 | 6 | ranking-formula |
| diversity | 列表多样性的目标与相似度口径 | 7 | reranking |
| diversity-reranking | MMR、DPP 与 SSD 多样性算法选型 | 7 | diversity |
| reranking | 重排 | 7 | ranking |
| reranking-postprocessing | 重排末端硬约束后处理 | 7 | reranking |
| reranking-sequence-evaluation | 重排序列价值评估 | 7 | reranking-sequence-generation |
| reranking-sequence-generation | 重排候选序列生成 | 7 | reranking |
| load-pid-control | 混排 Load 与 PID 流量调控 | 8 | mixed-ranking, score-calibration |
| mixed-ranking | 混排 | 8 | reranking |
| mixed-sequence-generation | 跨业务混排序列生成 | 8 | mixed-ranking, score-calibration |
| mixed-sequence-ranking | 混排候选序列排序 | 8 | mixed-sequence-generation |
| mixed-value-evaluation | 混排用户平台生态价值评估 | 8 | mixed-sequence-ranking |
| score-calibration | 分数校准 | 8 | mixed-ranking |
| generated-item-lookup | 生成语义 ID 到可服务物品检索 | 9 | semantic-id, generative-token-objective |
| generative-recsys | 生成式推荐 | 9 | ranking |
| generative-token-objective | 生成式推荐序列目标 | 9 | generative-recsys, semantic-id |
| one-stage-generative-alignment | 一阶段生成中编码与推荐目标对齐 | 9 | semantic-id, generative-token-objective |
| semantic-id | 语义 ID | 9 | generative-recsys |
| collaborative-filtering-recall | 协同过滤召回 | 10 | multi-recall |
| graph-recall | 图结构召回 | 10 | collaborative-filtering-recall |
| hard-negative | 困难负样本 | 10 | negative-sampling |
| multi-interest-recall | 多兴趣召回 | 10 | sequential-recall |
| negative-sampling | 负采样 | 10 | two-tower |
| sequential-recall | 序列召回 | 10 | two-tower |
| two-tower | 双塔召回 | 10 | multi-recall |
| lightweight-dnn-pre-ranking | 轻量 DNN 粗排的特征与计算预算 | 11 | pre-ranking-two-tower |
| model-distillation | 粗排模型蒸馏 | 11 | pre-ranking |
| multi-interest-pre-ranking | 粗排多兴趣表示与候选匹配 | 11 | pre-ranking-two-tower |
| multi-tower-pre-ranking | 粗排多塔建模 | 11 | pre-ranking-two-tower |
| pre-ranking-two-tower | 粗排双塔与交互取舍 | 11 | pre-ranking, two-tower |
| feature-crossing-deep-ranking | 精排特征交叉与深层表征融合 | 12 | ranking |
| listwise-ranking-loss | Listwise 排序损失 | 12 | ranking |
| multi-task-experts | 多任务专家模型 | 12 | multi-objective |
| ranking-loss-selection | 精排损失与概率校准 | 12 | ranking, listwise-ranking-loss |
| target-attention-ranking | 精排目标注意力 | 12 | feature-crossing-deep-ranking, sequential-recall |
| data-leakage | 样本穿越 | 15 | feature-engineering |
| feature-engineering | 特征工程 | 15 | recsys-funnel |
| feature-freshness | 特征时效性 | 15 | feature-engineering |
| feature-governance | 特征治理 | 15 | feature-engineering |
| feature-pruning | 冗余特征分组剪枝 | 15 | feature-engineering |
| forward-inverted-index-update | 正排与倒排索引协同更新 | 16 | offline-pipeline |
| late-feedback-labels | 迟到反馈的 Label 合并与样本等待 | 16 | offline-pipeline |
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
| ann-index-lifecycle | ANN 索引换代与一致性 | 18 | ann-index |
| hnsw | HNSW | 18 | ann-index |
| lsh-retrieval | 局部敏感哈希检索 | 18 | ann-index |
| quantization | 向量量化 | 18 | ann-index |
| vector-retrieval | 向量检索 | 18 | two-tower |
| exposure-bias | 曝光偏差 | 19 | offline-pipeline |
| position-bias | 位置偏差 | 19 | exposure-bias |
| position-debiasing | 位置偏差校正 | 19 | position-bias |
| sample-selection-bias | 样本选择偏差与级联漏样 | 19 | exposure-bias |
| user-group-bias | 用户群体偏差 | 19 | exposure-bias |
| consistency-modeling | 链路一致性建模 | 22 | version-consistency |
| duration-bucket-modeling | 时长分桶建模 | 23 | duration-modeling |
| duration-distribution | 时长分布建模 | 23 | duration-modeling |
| duration-modeling | 时长预估 | 23 | multi-objective |
| duration-threshold-labels | 时长阈值标签 | 23 | duration-modeling |
| weighted-lr-duration | Weighted LR 时长反解 | 23 | duration-modeling |
| exploration | 探索与利用 | 28 | cold-start |
| rl-recsys | 强化学习推荐 | 28 | exploration |
| safe-exploration | 安全探索 | 28 | exploration |

知识图谱机器可读版本见 [`graph.json`](graph.json)。
