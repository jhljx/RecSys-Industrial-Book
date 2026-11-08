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
| diger-differentiable-id | DIGER 可微语义 ID | 9 | one-stage-generative-alignment |
| etegrec-end-to-end-token | ETEGRec 端到端物品 Token | 9 | one-stage-generative-alignment |
| fsq-tokenizer | FSQ 有限标量量化 | 9 | semantic-id |
| generated-item-lookup | 生成语义 ID 到可服务物品检索 | 9 | semantic-id, generative-token-objective |
| generative-recsys | 生成式推荐 | 9 | ranking |
| generative-token-objective | 生成式推荐序列目标 | 9 | generative-recsys, semantic-id |
| lfq-tokenizer | LFQ 无查找表量化 | 9 | semantic-id |
| one-stage-generative-alignment | 一阶段生成中编码与推荐目标对齐 | 9 | semantic-id, generative-token-objective |
| pq-semantic-tokenizer | PQ 分组向量量化 | 9 | semantic-id |
| rq-kmeans-tokenizer | RQ-KMeans 残差聚类编码 | 9 | semantic-id |
| rq-vae-quantization | RQ-VAE 残差量化 | 9 | semantic-id |
| semantic-id | 语义 ID | 9 | generative-recsys |
| unisid-joint-embedding | UniSID 语义 ID 与内容表示联合生成 | 9 | one-stage-generative-alignment |
| bert4rec-masked-items | BERT4Rec双向掩码建模 | 10 | collaborative-filtering-recall |
| collaborative-filtering-recall | 协同过滤召回 | 10 | multi-recall |
| eges-side-information | EGES侧信息融合 | 10 | collaborative-filtering-recall |
| graph-recall | 图结构召回 | 10 | collaborative-filtering-recall |
| graph-walk-bias | 图随机游走采样 | 10 | collaborative-filtering-recall |
| hard-negative | 困难负样本 | 10 | negative-sampling |
| lightgcn-propagation | LightGCN邻居传播 | 10 | collaborative-filtering-recall |
| mind-dynamic-routing | MIND动态路由兴趣胶囊 | 10 | collaborative-filtering-recall |
| multi-interest-recall | 多兴趣召回 | 10 | sequential-recall |
| negative-sampling | 负采样 | 10 | two-tower |
| neural-cf-interaction | Neural CF非线性交互 | 10 | collaborative-filtering-recall |
| pdn-path-matching | PDN路径匹配召回 | 10 | collaborative-filtering-recall |
| pinsage-neighbor-sampling | PinSage重要邻居采样 | 10 | collaborative-filtering-recall |
| sasrec-causal-attention | SASRec因果序列注意力 | 10 | collaborative-filtering-recall |
| sdm-long-short-interest | SDM长短期兴趣融合 | 10 | collaborative-filtering-recall |
| semantic-item-embedding | 预训练语义向量召回 | 10 | collaborative-filtering-recall |
| sequence-contrast-augmentation | 序列对比视图增强 | 10 | collaborative-filtering-recall |
| sequential-recall | 序列召回 | 10 | two-tower |
| structured-candidate-search | 结构化候选搜索 | 10 | multi-recall |
| swing-common-user-penalty | Swing共同用户去噪 | 10 | collaborative-filtering-recall |
| tdm-tree-search | TDM树状候选搜索 | 10 | collaborative-filtering-recall |
| two-tower | 双塔召回 | 10 | multi-recall |
| dat-dual-augmentation | DAT双重增强双塔 | 11 | pre-ranking-two-tower |
| delayed-feedback-multitower | 延迟反馈标签成熟窗口 | 11 | pre-ranking-two-tower |
| fscd-feature-cost | FSCD可学习特征选择 | 11 | pre-ranking-two-tower |
| inttower-interaction | IntTower塔间交互效率 | 11 | pre-ranking-two-tower |
| lightweight-dnn-pre-ranking | 轻量 DNN 粗排的特征与计算预算 | 11 | pre-ranking-two-tower |
| model-distillation | 粗排模型蒸馏 | 11 | pre-ranking |
| multi-interest-pre-ranking | 粗排多兴趣表示与候选匹配 | 11 | pre-ranking-two-tower |
| multi-tower-pre-ranking | 粗排多塔建模 | 11 | pre-ranking-two-tower |
| mvke-interest-kernels | MVKE多兴趣虚拟核专家 | 11 | pre-ranking-two-tower |
| pre-ranking-two-tower | 粗排双塔与交互取舍 | 11 | pre-ranking, two-tower |
| privileged-feature-distillation | PFD特权特征蒸馏 | 11 | pre-ranking-two-tower |
| rocket-launching-training | Rocket Launching联合训练 | 11 | pre-ranking-two-tower |
| tdssm-temporal-tower | TDSSM长短期双塔兴趣 | 11 | pre-ranking-two-tower |
| dlrm-dot-interaction | DLRM点积交互 | 12 | feature-crossing-deep-ranking |
| feature-crossing-deep-ranking | 精排特征交叉与深层表征融合 | 12 | ranking |
| flashattention-io | FlashAttention访存优化 | 12 | feature-crossing-deep-ranking |
| listwise-ranking-loss | Listwise 排序损失 | 12 | ranking |
| mmoe-task-gates | MMoE任务门控专家 | 12 | feature-crossing-deep-ranking |
| multi-task-experts | 多任务专家模型 | 12 | multi-objective |
| ple-progressive-separation | PLE渐进共享与专属专家 | 12 | feature-crossing-deep-ranking |
| pnn-product-interaction | PNN显式乘积交互 | 12 | feature-crossing-deep-ranking |
| ranking-loss-selection | 精排损失与概率校准 | 12 | ranking, listwise-ranking-loss |
| target-attention-ranking | 精排目标注意力 | 12 | feature-crossing-deep-ranking, sequential-recall |
| xdeepfm-cin | xDeepFM显式高阶交互 | 12 | feature-crossing-deep-ranking |
| afn-adaptive-order | AFN自适应交互阶数 | 15 | feature-crossing-deep-ranking |
| autoint-field-attention | AutoInt特征域注意力 | 15 | feature-crossing-deep-ranking |
| can-coaction | CAN特征协同作用 | 15 | feature-crossing-deep-ranking |
| data-leakage | 样本穿越 | 15 | feature-engineering |
| dcn-v2-cross | DCN-V2显式交叉矩阵 | 15 | feature-crossing-deep-ranking |
| feature-engineering | 特征工程 | 15 | recsys-funnel |
| feature-freshness | 特征时效性 | 15 | feature-engineering |
| feature-governance | 特征治理 | 15 | feature-engineering |
| feature-interaction-models | 特征交互模型选型 | 15 | feature-engineering |
| feature-pruning | 冗余特征分组剪枝 | 15 | feature-engineering |
| fibinet-bilinear | FiBiNet双线性交互 | 15 | feature-crossing-deep-ranking |
| gdcn-gated-cross | GDCN门控深层交叉 | 15 | feature-crossing-deep-ranking |
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
| hnsw-search-budget | HNSW分层图搜索预算 | 18 | hnsw |
| ivfadc-residual-code | IVFADC残差量化 | 18 | quantization |
| kd-tree-pruning | KD树空间剪枝 | 18 | ann-index |
| lsh-retrieval | 局部敏感哈希检索 | 18 | ann-index |
| minhash-jaccard | MinHash集合相似检索 | 18 | lsh-retrieval |
| quantization | 向量量化 | 18 | ann-index |
| simhash-hamming | SimHash汉明近邻 | 18 | lsh-retrieval |
| vector-retrieval | 向量检索 | 18 | two-tower |
| exposure-bias | 曝光偏差 | 19 | offline-pipeline |
| position-bias | 位置偏差 | 19 | exposure-bias |
| position-debiasing | 位置偏差校正 | 19 | position-bias |
| sample-selection-bias | 样本选择偏差与级联漏样 | 19 | exposure-bias |
| user-group-bias | 用户群体偏差 | 19 | exposure-bias |
| attention-polarization | De-Anchor 注意力极化 | 20 | din-local-activation |
| augmented-behavior-contrast | 增广行为序列对比 | 20 | positive-negative-sequence |
| behavior-feedback-contrast | 行为反馈对比目标 | 20 | target-attention-ranking |
| bst-self-attention | BST 行为自注意力 | 20 | din-local-activation |
| cain-context-aware-history | CAIN 邻域上下文兴趣 | 20 | lcn-cross-domain-lifetime |
| cofars-context-prototype-search | CoFARS 上下文原型筛选 | 20 | context-temporal-prototypes |
| context-temporal-prototypes | 上下文时序图与兴趣原型 | 20 | long-sequence-retrieval |
| dien-interest-evolution | DIEN 兴趣演化 | 20 | din-local-activation |
| din-local-activation | DIN 目标局部激活 | 20 | target-attention-ranking |
| disenctr-dynamic-interest-graph | DisenCTR 动态兴趣图 | 20 | context-temporal-prototypes |
| dsin-session-interest | DSIN Session 兴趣 | 20 | dien-interest-evolution |
| eta-net-serving | ETA-Net 长序列部署 | 20 | eta-simhash-retrieval |
| eta-simhash-retrieval | ETA SimHash 行为检索 | 20 | sequence-hash-aggregation |
| fim-frequency-interest | FIM 频域多视角兴趣 | 20 | mirrn-multigranularity |
| infonce-density-ratio | InfoNCE 与密度比 | 20 | behavior-feedback-contrast |
| lcn-cross-domain-lifetime | LCN 跨域终身行为 | 20 | long-sequence-retrieval |
| lmn-shared-memory | LMN 跨用户共享记忆 | 20 | mimn-user-memory |
| long-history-interest-modeling | 长历史兴趣检索与压缩 | 20 | sequential-recall |
| long-sequence-retrieval | 长行为序列的目标检索 | 20 | target-attention-ranking |
| mimn-user-memory | MIMN 用户兴趣记忆 | 20 | dien-interest-evolution |
| mirrn-multigranularity | MIRRN 多粒度检索 | 20 | sim-two-stage-search |
| miss-multimodal-tree | MISS 多模态树检索 | 20 | multimodal-sim-search |
| moco-queue | MoCo 动量队列 | 20 | infonce-density-ratio |
| multimodal-interest-modeling | 多模态兴趣与语义融合 | 20 | sequential-recall |
| multimodal-sim-search | 多模态 SIM 检索 | 20 | sim-two-stage-search |
| muse-stagewise-modality | MUSE 分阶段模态融合 | 20 | multimodal-sim-search |
| positive-negative-sequence | 正负反馈序列分离 | 20 | behavior-feedback-contrast |
| psrq-semantic-codebooks | PSRQ 语义残差量化 | 20 | multimodal-sim-search |
| reformer-lsh-attention | Reformer LSH 注意力 | 20 | sequence-hash-aggregation |
| sdim-collision-aggregation | SDIM 碰撞聚合 | 20 | eta-simhash-retrieval |
| sequence-attention-overview | 目标注意力与兴趣演化建模 | 20 | sequential-recall |
| sequence-contrast-overview | 对比学习与反馈序列建模 | 20 | sequential-recall |
| sequence-hash-aggregation | 序列哈希检索与聚合 | 20 | long-sequence-retrieval |
| sequence-memory-overview | 长期兴趣记忆建模 | 20 | sequential-recall |
| sim-two-stage-search | SIM 两阶段行为检索 | 20 | long-sequence-retrieval |
| simclr-views | SimCLR 增广视图 | 20 | infonce-density-ratio |
| simsiam-stop-gradient | SimSiam 停止梯度 | 20 | simclr-views |
| tin-temporal-correlation | TIN 语义时间相关 | 20 | din-local-activation |
| twin-search-consistency | TWIN 两阶段一致性 | 20 | sim-two-stage-search |
| twin-v2-lifetime-compression | TWIN-V2 终身压缩 | 20 | twin-search-consistency |
| consistency-modeling | 链路一致性建模 | 22 | version-consistency |
| cread-restoration | CREAD分类恢复时长 | 23 | duration-distribution |
| d2co-noise | D2Co有偏噪声时长信号 | 23 | duration-distribution |
| d2q-deconfounding | D2Q时长偏差解耦 | 23 | duration-distribution |
| duration-bucket-modeling | 时长分桶建模 | 23 | duration-modeling |
| duration-distribution | 时长分布建模 | 23 | duration-modeling |
| duration-modeling | 时长预估 | 23 | multi-objective |
| duration-threshold-labels | 时长阈值标签 | 23 | duration-modeling |
| flowtime-personal-prior | FlowTime个性化流先验 | 23 | duration-distribution |
| tpm-progressive-regression | TPM树状渐进回归 | 23 | duration-distribution |
| weighted-lr-duration | Weighted LR 时长反解 | 23 | duration-modeling |
| exploration | 探索与利用 | 28 | cold-start |
| rl-recsys | 强化学习推荐 | 28 | exploration |
| safe-exploration | 安全探索 | 28 | exploration |

知识图谱机器可读版本见 [`graph.json`](graph.json)。
