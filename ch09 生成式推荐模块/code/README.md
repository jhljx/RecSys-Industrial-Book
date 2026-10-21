# ch09 生成式推荐模块 - 代码示例

本目录对应教材第 9 章「生成式推荐模块」，用 Python 复刻生成式推荐的核心链路（语义ID生成 → 生成式推荐模型 → 物品ID检索 → 一阶段联合训练 → 落地问题与讨论），共 **18 个可运行脚本**。目录编号与正文节次一一对应，每个脚本自包含、互不 import，均可直接 `python3 <文件>.py` 运行。其中 11 个脚本依赖 **PyTorch**（`item_id_vs_semantic_id.py`、`rq_vae.py`、`codebook_collapse.py`、`decoder_only_transformer.py`、`training_objectives.py`、`diger_differentiable_sid.py`、`etegrec_end_to_end.py`、`unisid_one_stage.py`、`diffusion_lm_parallel_decode.py`、`mtp_speculative_decoding.py`、`gradient_conflict_pcgrad.py`），其余只依赖 Python 标准库与 NumPy。正文涉及的工业组件（MLLM 内容理解、ANN 索引、真实推理引擎等）在此一律用进程内 Python mock 表达，不联网、不读本地数据集。

---

## 目录结构

```
code/
├── ch01_overview/                      § 9.1 生成式推荐概述
│   └── item_id_vs_semantic_id.py       Item ID vs 语义 ID：词表规模 / 输出层参数 / NTP 数据形态
├── ch02_architecture/                  § 9.2 生成式推荐架构
│   ├── cascade_vs_generative.py        级联漏斗 vs One-Model：算力 / 候选规模 / 可控性 / 异常影响半径
│   └── gen_rec_pipeline.py             生成式推荐链路端到端仿真（§9.2 五阶段 + 两种部署方式）
├── ch03_semantic_id/                    § 9.3 语义ID生成
│   ├── rq_vae.py                       § 9.3.1 RQ-VAE：残差量化 + STE 直通估计
│   ├── codebook_collapse.py            § 9.3.2 码本坍缩诊断与缓解（熵正则 / 死码本重启 / KMeans++）
│   ├── rq_kmeans.py                    § 9.3.3 RQ-KMeans / RQ-KMeans++（OneRec 方案）
│   └── other_tokenizers.py             § 9.3.4 PQ / FSQ / LFQ：分组量化与无查找表量化
├── ch04_generative_model/              § 9.4 生成式推荐模型
│   ├── decoder_only_transformer.py     § 9.4.1 Decoder-only Transformer：因果掩码 + NTP 训练
│   └── training_objectives.py          § 9.4.2 训练目标对比：NTP vs MTP vs Masked
├── ch05_item_retrieval/                § 9.5 物品ID检索
│   ├── sid_conflict_and_index.py       语义 ID 冲突 + Hash/Trie/ANN 三种索引
│   └── trie_constrained_beam_search.py Trie 前缀约束 Beam Search（只生成合法语义 ID）
├── ch06_one_stage/                     § 9.6 一阶段生成式推荐
│   ├── diger_differentiable_sid.py     § 9.6.1 DIGER：Gumbel-Max + SDUD/FrqUD 可微离散化
│   ├── etegrec_end_to_end.py           § 9.6.2 ETEGRec：交替优化 + SIA/PSA 双对齐
│   └── unisid_one_stage.py             § 9.6.3 UniSID：Embedding 与语义 ID 联合学习
├── ch07_deployment/                     § 9.7 生成式推荐落地问题
│   ├── exposure_bias_and_diversity.py  曝光偏差数据循环 + Beam Search 多样性约束
│   ├── diffusion_lm_parallel_decode.py DLM 掩码离散扩散：并行去噪步数与质量权衡
│   └── mtp_speculative_decoding.py     MTP 多 Token 预测头 + 推测解码加速
└── ch08_discussion/                     § 9.8 生成式推荐讨论
    └── gradient_conflict_pcgrad.py     多任务梯度冲突：等权求和 vs PCGrad vs GradNorm
```

---

## 快速运行

```bash
# § 9.1 生成式推荐概述
python3 ch01_overview/item_id_vs_semantic_id.py

# § 9.2 生成式推荐架构
python3 ch02_architecture/cascade_vs_generative.py
python3 ch02_architecture/gen_rec_pipeline.py

# § 9.3 语义ID生成
python3 ch03_semantic_id/rq_vae.py
python3 ch03_semantic_id/codebook_collapse.py
python3 ch03_semantic_id/rq_kmeans.py
python3 ch03_semantic_id/other_tokenizers.py

# § 9.4 生成式推荐模型
python3 ch04_generative_model/decoder_only_transformer.py
python3 ch04_generative_model/training_objectives.py

# § 9.5 物品ID检索
python3 ch05_item_retrieval/sid_conflict_and_index.py
python3 ch05_item_retrieval/trie_constrained_beam_search.py

# § 9.6 一阶段生成式推荐
python3 ch06_one_stage/diger_differentiable_sid.py
python3 ch06_one_stage/etegrec_end_to_end.py
python3 ch06_one_stage/unisid_one_stage.py

# § 9.7 生成式推荐落地问题
python3 ch07_deployment/exposure_bias_and_diversity.py
python3 ch07_deployment/diffusion_lm_parallel_decode.py
python3 ch07_deployment/mtp_speculative_decoding.py

# § 9.8 生成式推荐讨论
python3 ch08_discussion/gradient_conflict_pcgrad.py
```

全部脚本均在 CPU 上 15 秒内跑完（最慢的是 `etegrec_end_to_end.py`，约 14 秒）。

---

## 各脚本要点

下面的数字均来自 `random.seed(42)` 固定随机种子后的实际运行输出。

### § 9.1 `ch01_overview/item_id_vs_semantic_id.py`

量化「为什么不能直接用 Item ID 当生成目标」：N 达亿级时输出层词表爆炸。正文例子 L=4 层、每层 K=512 码字。

- **词表对比**：Item ID 词表 = 物品数 N，语义 Token 词表恒为 L×K = 2,048。N=10⁶/10⁷/10⁸/10⁹ 时输出层参数从 64.0M 涨到 **64,000.0M**，而语义 ID 方案恒为 0.131M，缩减 **488x → 488,281x**。
- **可表示空间**：语义 ID 组合空间 K^L = 512⁴ ≈ **6.87×10¹⁰**，远大于亿级物品集。
- **实测开销**（受内存限制 Item ID 方案只取 20 万物品）：Item ID 输出头 12.8M 参数、单次前向+softmax **7.463ms**；语义 Token 头 0.13M、**0.142ms** —— 参数量比 97.7x、耗时比 52.6x。且这仅仅是 20 万物品；亿级下 Item ID 输出层根本放不进显存。
- **代价演示**：L=4、K=8 的玩具例子展示「行为序列 → 语义 ID 序列 → NTP 样本 → 逐层自回归生成 → 查表恢复 Item」的完整闭环（生成一个物品需要 L 次自回归前向，这正是 §9.7 讨论 MTP/DLM 的动机）。

### § 9.2 `ch02_architecture/cascade_vs_generative.py`

复刻正文两种架构的算力/可控性对比。

- **级联漏斗**（召回 100 万 → 粗排 6 千 → 精排 600 → 重排 60）：总算力 3530.0 MFLOPs，其中召回占 56.7%、精排占 34.0%。
- **生成式**（beam=256 路径 × L=9 次自回归 Decoder 前向 + SID 查表）：合计 **2304.3** MFLOPs，仅为级联的 **0.653x**；候选规模 64 条 vs 60 条（1.07x）。
- **关键差异**：级联打分涉及 **100 万物品**（算力随全库规模线性增长），生成式只搜 **256 条路径**（算力只与 L、beam 有关）；可插入业务约束的决策点级联 13 个 vs 生成式 6 个。
- **异常影响半径**（§9.8 的分层防护论据）：单路召回故障级联只影响 18% 曝光、One-Model 影响 **100%**；平均影响半径级联 51% vs One-Model 100%。
- **两种部署方式**（正文）：① 并行候选生成——生成式模块作为统一召回-粗排通道汇入重排；② 流量隔离——划 ~5% 实验流量直接由生成式模块决策。

### § 9.2 `ch02_architecture/gen_rec_pipeline.py`

把正文五阶段链路（内容理解 → 语义ID生成 → 序列建模 → 自回归生成 → 检索 + 过滤重排）串成一次可观察的请求。240 个物品、L=3、K=8。

- **[MOCK] MLLM 内容理解** 0.87ms 产出 6 类物品 embedding；链路总耗时 78.7ms（其中离线语义 ID 生成 77.7ms，在线四阶段合计约 1ms）。
- **beam=24 生成后检索**：6/24 = **25.0%** 的 beam 候选是无效语义 ID（直接丢弃），其余 18 条有效 SID 命中 29 个物品（多对一映射平均每条携带约 1.6 个物品）；29 个候选经过滤重排后凑满 Top-10。
- **多对一冲突**：183 个不同 SID 覆盖 240 物品，冲突分布 {1 个物品的 SID×153, 2×32, 3×5, 4×2}。
- Top-10 重排后类别分布 {3:5, 5:3, 0:1, 1:1}，用户历史类别 3 占主导 —— 对应正文「生成式候选聚集度高，需要后续重排打散」。

### § 9.3.1 `ch03_semantic_id/rq_vae.py`

严格按正文损失 $L = L_{recon} + \lambda \sum_l \|z_{l-1} - e_{s^l}\|^2$ 实现 RQ-VAE（Encoder/Decoder 2 层 MLP + STE 直通估计），N=3000、dim=48、L=3、K=64。

- **训练收敛**：300 步 total loss 4.3850 → 0.5044，重构项 4.0259 → 0.1738。
- **残差逼近**：逐层量化后残差能量相对初始 **100% → 28.1% → 5.5% → 3.7%**，直观展示「第一层吸收大部分能量、后续层逐级细化」。
- **码本利用**：三层分别使用 27/43/45 个码字（使用率 42.2%/67.2%/70.3%），Perplexity 24.68/36.36/28.09（上限 64）。
- **教学规模的坍缩现象**：3000 个物品只产生 **330 个唯一语义 ID**（冲突率 89.0%）—— 码本利用不足时大量物品挤进同一 SID，这正是 §9.3.2 讨论码本坍缩与下一个脚本 `codebook_collapse.py` 的入口。

### § 9.3.2 `ch03_semantic_id/codebook_collapse.py`

构造 K=128 码本只被 60 个码字（46.9%）使用的坍缩场景，对比三种缓解手段。

| 配置 | 被用码字 | 使用率 | Perplexity | PPL/K | 量化误差 | Top10% 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| A 随机初始化 + 纯 STE | 60 | 46.9% | 59.22 | 46.3% | 114.01 | 24.4% |
| B A + 熵正则 λ=0.5 | 128 | 100.0% | 116.46 | 91.0% | 184.24 | 18.3% |
| C A + 死码本重启 | 128 | 100.0% | 79.28 | 61.9% | **3.18** | 23.6% |
| D KMeans++ 初始化 + 重启 | 128 | 100.0% | 96.64 | 75.5% | **3.45** | 21.2% |

- 三种手段都把使用率从 46.9% 拉到 100%，代价截然不同：**熵正则（B）把 Perplexity 推得最高（116.46）却把量化误差放大到 A 的 1.62x**；死码本重启（C）与 KMeans++ 初始化+重启（D）在把 PPL 提到 1.34x/1.63x 的同时把量化误差压到 A 的 **0.03x**，重构 MSE 从 0.5012 降到 0.0856 —— 工程上更实用的组合。
- 死码本重启机制：长时间未被任何样本选中的码字被重新播种到当前残差最大的样本处。

### § 9.3.3 `ch03_semantic_id/rq_kmeans.py`

OneRec（快手）采用的方案：RQ 层次化 + KMeans 聚类，无需神经网络与 STE。N=3000、L=3、K=64，ID 空间 262,144。

- **逐层残差能量**：RQ-KMeans（随机初始化）100% → 16.0% → 1.9% → **1.4%**；KMeans++ 初始化后第 2 层就降到 **2.1%**（初始能量被更均匀覆盖）。
- **KMeans++ 的真实收益在冲突率**：

| 指标 | RQ-KMeans（随机） | RQ-KMeans++ |
| --- | --- | --- |
| 重构 MSE | 0.05882 | **0.05220** |
| 相对残差（末层/初始） | 1.43% | **1.27%** |
| 唯一语义 ID 数 | 2671 | **2988** |
| 语义 ID 冲突率 | **10.97%** | **0.40%** |
| 平均 Perplexity | 52.36 | 60.76 |

- KMeans++ 只是把初始质心铺得更均匀，就让 3000 个物品几乎一物一码（冲突率 10.97% → 0.40%）—— 对应正文「码本初始化对语义 ID 质量影响巨大，工程上必须做多轮重启取最优」。

### § 9.3.4 `ch03_semantic_id/other_tokenizers.py`

在同一合成 embedding（N=2000, dim=32）上对比四类离散 Tokenizer（正文 §9.3.4 的范式划分）。

| 方法 | 范式 | 显式码字 | 隐式 ID 空间 | 实际用 ID | 相对 MSE |
| --- | --- | --- | --- | --- | --- |
| RQ (L=3,K=32) | Level-wise | 96 | 32,768 | 905 | **0.019** |
| PQ (S=4,K=32) | Group-wise | 128 | 1,048,576 | 71 | 0.154 |
| FSQ (8,8,5,5) | Lookup-Free | — | 1,600 | 161 | 0.991 |
| LFQ (12 bits) | Lookup-Free | — | 4,096 | 273 | 0.833 |

- **空间利用率反转**：PQ 隐式空间最大（10⁶）但只用到 71 个 ID（**0.01%**）；RQ 空间小却用到 905 个（**2.76%**、覆盖 45.2% 样本）。
- **LFQ 熵正则分解** $L_{entropy} = E_x[H(q(z))] - H(E_x[q(z)])$：单样本熵 6.0724 nats、边缘熵 8.2750 nats（上限 m·ln2 = 8.318）、总项 **−2.2026** —— 训练中最小化该项，鼓励码字使用均匀。
- 结论与正文一致：RQ/PQ 依赖显式码本查找、重构最准；FSQ/LFQ 无查找表、天然不会整块码字死亡，但以表达能力换工程简洁。

### § 9.4.1 `ch04_generative_model/decoder_only_transformer.py`

教学规模 Decoder-only Transformer（d_model=48、4 heads、2 blocks、Pre-LN + 因果掩码），严格按 §9.4.2 的 NTP 目标训练。物品 300、序列 1200、每物品 3 层语义 Token（不同层用互不重叠的 id 区间）。

- **逐层 next-token 可学性差异**：8 个 epoch 后 valid CE 2.9286 → 1.2231（PPL 18.70 → 3.40）；第 1 层 token acc 飙到 **88.3%** 后饱和，第 2/3 层缓慢爬到 32.6%/61.7% —— 第 1 层承载类别级粗信息（可预测），越深层越接近物品级细节（信息量越大越难预测），对应正文「层次化语义 ID 的浅层 token 相对可学」。
- **贪心展开 top-k 命中**（验证集 120 样本）：完整语义 ID 的 HR@1 = **18.3%**、HR@5 = 55.0%、HR@10 = 70.8%；随机基线 1/K^L = 0.024%。
- **合法性**：top-1 生成路径 94.2% 落在合法语义 ID 表内（本实现逐层只在当前层区间取 top-k，但没有全局前缀约束，仍有 5.8% 无效 —— 引出 §9.5 的 Trie 约束解码）。

### § 9.4.2 `ch04_generative_model/training_objectives.py`

三种训练目标（NTP / MTP n=2 / Masked p=15%）在同一道评测题上对比：给定同一前缀预测末尾两个 Token，口径完全一致（NTP 第 2 步用自己的预测而非真值）。

| 目标 | 注意力 | 第 1 个 Token acc | 两个 Token 联合 acc |
| --- | --- | --- | --- |
| A NTP | causal | **28.1%** | **13.3%** |
| B MTP (n=2) | causal | 26.7% | 10.4% |
| C Masked (p=15%) | 双向 | 17.8% | 4.4% |

- **产出同样 2 个 Token 的前向次数**：NTP 2 次、MTP 1 次、Masked 1 次 —— 这正是正文「MTP 提供更密集监督信号并天然适配投机解码；掩码/扩散式目标可并行产出整组 Token」的来源。
- 注：玩具规模下联合 acc 最高的是 NTP（13.3%），正文 §9.4.2 的论点是 MTP 的监督密度与推理并行度，而非小数据集上的绝对精度。

### § 9.5 `ch05_item_retrieval/sid_conflict_and_index.py`

多对一映射（N=400 物品、K^L=216、不同 SID 183 个）下的检索问题。

- **冲突分布**：平均 **2.19 物品/SID**（占位球模型理论期望 1.85），最大一个 SID 挂 6 个物品；15% 的 SID 独占、31.5% 二物品。
- **三种索引**（查询 4000 次，其中 618 次非法 SID）：哈希表 0.106μs / Trie 0.183μs / **[MOCK] ANN 暴力近邻 26.8μs**。合法 SID 三者命中率均 100%；只有 ANN 能给非法 SID 返回近似结果。
- **Top-K 恢复**：每用户 beam=20 路径，丢弃无效 SID 后靠多对一 SID 的额外物品补位（平均 6.5 个/用户），8/8 用户凑满 Top-10。
- **消歧策略**（正文「去重后仅保留得分最高者」）：SID=(0,4,3) 5 个候选 → 保留 item_42（0.659）。

### § 9.5 `ch05_item_retrieval/trie_constrained_beam_search.py`

正文的核心工程问题：无约束 Beam Search 大部分路径是 Invalid SID。N=220 物品、合法路径仅占全空间 34.6%。

| beam | 无约束非法率 | 无约束有效候选 | Trie 约束非法率 | Trie 约束有效候选 |
| --- | --- | --- | --- | --- |
| 5 | 60.5% | 1.98 | **0.0%** | **5.00** |
| 10 | 62.5% | 3.75 | 0.0% | 10.00 |
| 20 | 63.5% | 7.29 | 0.0% | 20.00 |
| 50 | 64.5% | 17.75 | 0.0% | 50.00 |

- Trie 三层节点 1/8/60，深层平均分支从 8.00 收敛到 2.95 —— 前缀树天然把搜索限制在合法路径上，**beam 利用率恒为 100%**。
- 长度归一化实验：L 固定时归一化不改变排序（5/5 用户 top-1 不变），对应正文「长度归一化在变长生成场景才真正起作用」。

### § 9.6.1 `ch06_one_stage/diger_differentiable_sid.py`

DIGER 的可微离散化三件套：Gumbel-Max Trick、SDUD 噪声自适应衰减、FrqUD 频率感知离散化。

- **Gumbel-Max 验证**（20 万次采样，logits=[2,1,0,−1]）：argmax 经验频率 [0.6463, 0.2363, 0.0862, 0.0312] vs Softmax [0.6439, 0.2369, 0.0871, 0.0321]，绝对误差 ≤ 0.0024 —— argmax(l+g) 确实服从 Softmax(l)。
- **SDUD 衰减**：$\sigma^* = \max(0, \sqrt{L_{gen}} - \lambda)$，λ=0.3 下 L_gen 从 6.2246 降到 0.0519 时 σ* 从 2.1949 自动衰减到 0 —— 生成损失下降驱动噪声从探索平滑过渡到利用。
- **离散化对比**（推理统一关噪声）：STE 的 L_gen=3.0334 但 **SID 全命中率仅 4.69%**（梯度直通没有探索）；DRIL（Gumbel+SDUD）L_gen=0.0401、**全命中 100%**、三层 Perplexity 2.65/3.62/2.66（码字覆盖更均衡）。
- **FrqUD**：EMA 频率 > β 的热门码字保留 Gumbel 探索、其余转确定性 Softmax —— 第 1 层 I_high={3}（EMA 0.854）只占 1/8 码字，正文「对过度使用的码字保留探索、对低频码字稳定分配」。

### § 9.6.2 `ch06_one_stage/etegrec_end_to_end.py`

ETEGRec（SIGIR 2025，人大高瓴 × 快手）交替优化：每轮先 L_IT 更新 Tokenizer、再 L_GR 更新推荐模型，四项损失 $L = L_{REC} + \mu L_{SIA} + \lambda L_{PSA} + L_{SQ}$。

- **两阶段 vs 交替优化**：两阶段（RQ-VAE 冻结）L_REC **0.3396**、SID 全命中 0.8687；ETEGRec L_REC 1.4670、全命中 0.5781。Tokenizer 重建损失几乎不变（60.65 vs 60.81）—— 交替优化**牺牲一点重建精度换取更适合生成任务的语义 ID**，L_SIA 把推荐信号回传到 Tokenizer（第 1 轮 1.78 → 第 5 轮 3.18 波动上升，表示对齐压力与重建压力在博弈）。
- **对齐消融**（5 轮）：仅 L_SQ+L_REC 全命中 0.4437；+SIA 提到 0.6062；+PSA 0.5688；SIA&PSA 双开 0.5781 但 L_SIA/L_PSA 同步最低（3.18/8.86 vs 消融时的 24.76/14.40）—— 两个对齐项各管一端（样本对齐 / 路径对齐），缺一则另一端失压。
- 注：教学规模 + 合成数据，绝对数值不代表论文结论，仅展示四项损失的耦合关系。

### § 9.6.3 `ch06_one_stage/unisid_one_stage.py`

UniSID（腾讯广告）：统一多模态大模型同时产出 Item Embedding 与语义 ID，打破「Embedding→RQ-VAE→SID」两阶段。总损失 $L_{total} = L_{sid} + L_{emb} + \lambda L_{rec}$。

- **s^l = argmax(z^l) 的鲁棒性**（正文「只关心维度大小顺序」）：整体 ×10、÷10、每层 +5 平移后 SID 完全一致率 **100%**；只有 σ=0.5 的高斯噪声才翻转 43% 的 SID —— 语义 ID 对缩放/平移不变，只受大噪声影响。
- **多粒度对比学习**：第 1 层 token 对大类纯度 **1.0000**（4 个 token 正好对应 4 个大类）、对（大类,子类）纯度仅 0.4444；第 3 层对子类纯度升到 0.6667 —— 层次解耦成立（浅层管粗粒度、深层管细粒度）。
- **损失消融**（各配置独立训练 40 epoch）：去掉 L_sid 后第 1 层大类纯度崩到 0.7444（L_sid 负责层次解耦）；去掉 L_rec 后子类纯度降到 0.5611（L_rec 迫使 SID 编码抽象语义）；去掉 L_emb 第 3 层纯度反升到 0.7111 但 Embedding 分支失去约束。
- 180 物品只用 22 个不同 SID —— 教学规模下层次 SID 天然把同类物品折叠到同前缀，这也是正文「语义 ID 天然承载层次语义」的体现。

### § 9.7 `ch07_deployment/exposure_bias_and_diversity.py`

两大落地问题各做一个闭环实验。

**数据循环**（12 轮闭环，300 物品、营销号诱导系数 2.6 占比 14.3%）：

| 策略 | 末轮基尼 | 垂类熵(上限 2.079) | 营销号曝光占比 | 覆盖物品 |
| --- | --- | --- | --- | --- |
| 朴素（高反馈=正样本） | 0.873 | 1.085 | **90.0%** | 222 |
| 质量加权+垂类降权 | 0.707 | 2.047 | 20.9% | 287 |
| 质量加权+ε探索(ε=0.15) | **0.540** | **2.065** | **18.3%** | **298** |

- 朴素策略 12 轮内把营销号曝光推到 90%、基尼从 0.141 涨到 0.873 —— 曝光偏差的正反馈循环被完整复现；ε 探索几乎不牺牲真实质量（0.753 vs 朴素的 0.797 是「诱导流量」贡献的虚高）换取最大覆盖。
- **Beam Search 多样性**：单纯把 beam 从 1 加到 20，覆盖垂类数/最大同类占比/ΣlogP 完全不变（贪心解已是概率最优）；必须在路径评分加 α（类别重复罚）+β（新颖性）后，最大同类占比才从 60% 降到 40%，代价仅 ΣlogP +0.076 —— 正文「加大束宽无效，必须显式加约束项」。

### § 9.7 `ch07_deployment/diffusion_lm_parallel_decode.py`

掩码离散扩散（T=12 噪声级）：历史确定部分 + 一个等概率不可预测分支（首层码字 50/50）+ 物品 B 依赖 A —— 结构化构造「跨位置/跨物品依赖」，检验并行解码的协同能力。

| 去噪步数 S | 前向次数 | 串行步数 | 合法 SID 率 | 跨物品一致性 |
| --- | --- | --- | --- | --- |
| 自回归逐 Token | 6 | 6 | 100.0% | 100.0% |
| S=1（一次并行） | 1 | 1 | **74.0%** | **59.0%** |
| S=2 | 2 | 2 | 90.6% | 81.2% |
| S=3 | 3 | 3 | 96.1% | 92.2% |
| S=6 | 6 | 6 | 100.0% | 100.0% |

- **贪心解码下 Token 准确率几乎不随 S 变化**（0.5391~0.5547，天花板 ≈0.55 由 50% 分支决定）—— 真正的差别在**合法 SID 率与跨物品一致性**：S=1 时 6 个位置独立 argmax，会拼出词表里不存在的 SID。
- **采样解码更极端**：S=1 合法率崩到 39.5%、一致性 16.0%；逐步去噪时已定稿 token 成为后续位置的条件，S=6 恢复到自回归水平（99.8%/99.6%）—— 一次前向最快但质量最差，这就是正文「DLM 在生成质量与推理效率之间尚未达到最优平衡」的数值形态。

### § 9.7 `ch07_deployment/mtp_speculative_decoding.py`

MTP（K_MTP=3 个预测头）+ 推测解码，生成 24 个语义 Token（= 8 个物品）。

- **训练目标对比**：NTP 模型第 0 头 acc 0.7830、第 1/2 头接近随机（0.0447/0.0362，词表 1/V=0.027）；MTP 模型三个头 0.7575/0.6788/0.6919 —— **一次前向即可产出完整语义 ID**。
- **推理开销**：自回归 NTP 24 次前向（3.88ms）；MTP 并行 8 次（2.49ms，**3.00x**）但输出与 NTP 不保证一致；推测解码 16 次前向（6.01ms，1.50x）**输出与逐 Token 自回归完全一致** —— 验证步骤买到的是一致性，代价是草稿+验证双份前向。
- **接受长度**：平均 τ=1.56（上限 K_MTP=3），即平均每次前向定稿约 1.5 个 Token —— 这正是实测 1.50x 加速的来源。
- **K 扫描**：K=1/2/3 的前向次数 24/12/8 = T/K，与正文公式一致。

### § 9.8 `ch08_discussion/gradient_conflict_pcgrad.py`

「One-Model 能否替代级联」的优化视角论据：共享 2 维瓶颈 Backbone + 3 个任务头，同时存在**容量竞争**（各任务需要不同方向的非线性特征）与**尺度失衡**（任务 3 标签 ×6）两种冲突来源。

- **初始化时三对梯度余弦全负**（−0.178/−0.065/−0.073，冲突对占比 100%），梯度范数 [1.215, 0.796, 3.790] —— 尺度失衡已现。
- **三种策略最终 MSE**：等权求和被任务 3 主导（它最好 0.0313 但任务 1 最差 0.6994）；**PCGrad 平均 0.2917、最差任务 0.4328 双最优**；GradNorm 为平衡范数把任务 3 权重压到 0.03（w 从 0.96 → 0.13），任务 3 MSE 反而恶化到 0.8808。
- 结论与正文一致：调权类方法（GradNorm）只是**重新分配冲突而非消除它**；PCGrad 投影掉冲突分量也只是缓解（训练中冲突对占比仍有 91%）。推荐系统的生态/商业目标无法统一写成单一损失 —— One-Model 更可能是级联架构的补充而非替代。

---

## 实现约定

1. **节次对应**：子目录名 `chNN_<english_snake_case>` 中的 `NN` 与正文 §9.NN 一一对应（`ch01_overview` ↔ §9.1，`ch08_discussion` ↔ §9.8）。
2. **自包含**：脚本之间不互相 import，每个文件可以单独拷出运行。文件顶部 docstring 用中文写明对应的正文节次与概念、本实现做了什么、关键公式。
3. **可复现**：`if __name__ == "__main__":` 块固定 `random.seed(42)`，用到 NumPy / PyTorch 时同时设 `np.random.seed(42)` / `torch.manual_seed(42)`。全部数据为进程内合成，**不联网、不读本地数据集**。
4. **输出形式**：一律打印数值表格（对齐的列 + 分隔线 + 一句中文结论），不打印 loss 日志、不使用 matplotlib。
5. **Mock 组件**：教学用的假 MLLM 内容理解、假 ANN 索引、假输出层等在 docstring 中以 `[MOCK]` 标注，均为进程内 Python mock，不起真实服务。
6. **以正文为准**：公式、符号、参数（如 RQ-VAE 的损失形式、DIGER 的 $\sigma^*=\max(0,\sqrt{L_{gen}}-\lambda)$、ETEGRec 的四项损失、MTP 的 $K$ 头结构）全部照抄正文；教学规模（小词表、合成数据）下的绝对数值不代表论文或工业结论，脚本输出中均已注明。
7. **依赖**：Python 3.12 标准库、NumPy 1.26.4、PyTorch 2.7.1（11 个深度学习脚本）。

## 参考论文（正文实际引用）

- RQ: Lee et al., *Autoregressive image generation using residual quantization*, CVPR 2022
- VQ-VAE: van den Oord et al., *Neural discrete representation learning*, NeurIPS 2017
- PQ: Jégou et al., *Product quantization for nearest neighbor search*, TPAMI 2010
- OneRec: Deng et al., *OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*, 2025（RQ-KMeans）
- FSQ: Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple*, ICLR 2024
- LFQ: Yu et al., *Language model beats diffusion — tokenizer is key to visual generation*, ICLR 2024
- 离散 Tokenizer 综述: Jia et al., *From principles to applications: A comprehensive survey of discrete tokenizers*, 2025
- DIGER: Fu et al., *Differentiable Semantic ID for Generative Recommendation*, 2026
- ETEGRec: Liu et al., *Generative recommender with end-to-end learnable item tokenization*, SIGIR 2025
- UniSID: Jiang et al., *End-to-End Semantic ID Generation for Generative Advertisement Recommendation*, 2026
- MTP: Gloeckle et al., *Better & faster large language models via multi-token prediction*, ICML 2024
- DLM: Nie et al., *Large language diffusion models*, 2026
- PCGrad: Yu et al., *Gradient surgery for multi-task learning*, NeurIPS 2020
- GradNorm: Chen et al., *GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks*, ICML 2018

（以上条目均取自书稿 `reference.bib` 中正文 §9 实际引用的 `cite` 键。）
