# ch15 特征工程 — 代码示例

本目录对应教材第 15 章，覆盖特征体系、特征表示、行为与实时特征、特征交互、特征筛选与治理五个部分，共 **26 个可运行脚本**（另含 3 个公共模块）。目录编号与正文节次对应，均可直接 `python3 <文件>.py` 运行。

- §15.1–§15.4 的脚本**不读外部文件、不联网**：其中 `user_item_context_features.py`、`multi_window_behavior_features.py`、`realtime_session_features.py` 及 `common.py` 只用标准库，`multimodal_fusion.py`、`representation_compare.py`、`user_profile_interests.py`、`training_serving_skew.py` 额外用到 NumPy；
- §15.5 特征交互**依赖 PyTorch**（实现 10 个模型的网络结构与统一训练循环）；
- §15.6 特征筛选**依赖 NumPy**（手写 Logistic Regression，不调用 sklearn）。

---

## 目录结构

```
code/
├── common.py                              §15.1–§15.4 共用：合成用户/物品/行为事件流
├── ch01_feature_system/                   §15.1 特征体系
│   ├── user_item_context_features.py      §15.1.1–§15.1.4 用户/物品/上下文/行为特征清单与覆盖率
│   └── multimodal_fusion.py               §15.1.5 多模态特征：concat / weighted / attention 融合
├── ch02_representation/                   §15.2 特征表示
│   ├── sparse_dense_cost.py               §15.2.1–§15.2.3 Sparse Embedding 参数量、显存、四种压缩手段
│   └── representation_compare.py          §15.2.1 Sparse / Dense / 分桶 / Sequence 五种表示的 AUC 对比
├── ch03_behavior_and_realtime/            §15.3–§15.4 行为建模与实时特征
│   ├── multi_window_behavior_features.py  §15.3.1 多时间窗口、衰减权重、F/R/I 画像
│   ├── user_profile_interests.py          §15.3.2–§15.3.3 长期 vs 短期兴趣、多兴趣中心
│   ├── realtime_session_features.py       §15.4.1–§15.4.2 离线/准实时/实时/Session 多尺度与更新延迟
│   └── training_serving_skew.py           §15.4.3 训练-服务偏差：泄露、延迟、缺失、归一化
├── ch04_feature_interaction/              §15.5 特征交互（PyTorch）
│   ├── _train_utils.py                    公共：合成 CTR 数据、训练循环、AUC、Embedding 表
│   ├── dcn.py                             §15.5.1 DCN：x_{l+1} = x0·(x_lᵀw) + b + x_l
│   ├── dcn_v2.py                          §15.5.2 DCN-V2：满矩阵 / 低秩（含 r=0 退化为向量版）
│   ├── autoint.py                         §15.5.3 AutoInt：多头自注意力 + 注意力图
│   ├── fibinet.py                         §15.5.4 FiBiNet：SENET 重加权 + Bilinear（All/Each）
│   ├── afn.py                             §15.5.5 AFN：对数神经元与自适应阶数 O_j
│   ├── fed.py                             §15.5.6 FED：DRM（维度级）+ Field-wise + Element-wise
│   ├── can.py                             §15.5.7 CAN：Co-Action Unit + Multi-order
│   ├── fint.py                            §15.5.8 FINT：Field-aware Interaction Layer 堆叠
│   ├── gdcn.py                            §15.5.9 GDCN：门控交叉 + 串联/并联结构
│   ├── ipa.py                             §15.5.10 IPA：交互函数 × 层池化 × 层聚合
│   └── interaction_benchmark.py           全部交互模型的统一对比表
├── ch05_feature_selection/                §15.6 特征筛选与治理（NumPy）
│   ├── _lr_utils.py                       公共：合成 CTR 数据（含真值信号）、Logistic Regression
│   ├── feature_importance.py              §15.6.1 静态重要性 / 置换重要性 / LHUC 式门控
│   ├── feature_redundancy.py              §15.6.2 相关性、前向边际增益、分组消融
│   ├── feature_cost_roi.py                §15.6.3 成本四项拆解与 ROI 排序
│   ├── feature_quality.py                 §15.6.4 覆盖率 / 新鲜度 / 分布异常监控
│   ├── data_drift.py                      §15.6.5 PSI / KL / Label Drift
│   ├── feature_leakage.py                 §15.6.6 未来行为、窗口越界、ActionList 错位
│   └── feature_governance.py              §15.6.7 注册表、生命周期与下线决策
```

---

## 快速运行

```bash
# 依赖：Python 3.12；ch04 需要 torch，ch05 需要 numpy

# §15.1 特征体系
python3 ch01_feature_system/user_item_context_features.py
python3 ch01_feature_system/multimodal_fusion.py

# §15.2 特征表示
python3 ch02_representation/sparse_dense_cost.py
python3 ch02_representation/representation_compare.py

# §15.3–§15.4 行为与实时特征
python3 ch03_behavior_and_realtime/multi_window_behavior_features.py
python3 ch03_behavior_and_realtime/user_profile_interests.py
python3 ch03_behavior_and_realtime/realtime_session_features.py
python3 ch03_behavior_and_realtime/training_serving_skew.py

# §15.5 特征交互（逐个模型）
python3 ch04_feature_interaction/dcn.py
python3 ch04_feature_interaction/dcn_v2.py
python3 ch04_feature_interaction/autoint.py
python3 ch04_feature_interaction/fibinet.py
python3 ch04_feature_interaction/afn.py
python3 ch04_feature_interaction/fed.py
python3 ch04_feature_interaction/can.py
python3 ch04_feature_interaction/fint.py
python3 ch04_feature_interaction/gdcn.py
python3 ch04_feature_interaction/ipa.py

# 统一对比（约 40s）
python3 ch04_feature_interaction/interaction_benchmark.py

# §15.6 特征筛选与治理
python3 ch05_feature_selection/feature_importance.py
python3 ch05_feature_selection/feature_redundancy.py
python3 ch05_feature_selection/feature_cost_roi.py
python3 ch05_feature_selection/feature_quality.py
python3 ch05_feature_selection/data_drift.py
python3 ch05_feature_selection/feature_leakage.py
python3 ch05_feature_selection/feature_governance.py
```

---

## 各脚本要点

### §15.1 特征体系

- **`user_item_context_features.py`**：打印一份 24 行的特征清单，按来源分为 用户=8 / 物品=7 / 上下文=6 / 行为=3，按表示分为 Sparse=10 / Dense=12 / Sequence=1 / Embedding=1。核心是"同一业务信息（7 天内观看体育内容 6 次）可以有四种表示"的对照：Sparse ID 把 6 与 7 当成两个无关类别、Dense 原值保留数值语义但只有 0 个参数、分桶 Embedding 用 9 个参数折中、Sequence 保留次序但需要序列编码器。末尾用一个覆盖率 62% 的 Dense 特征说明默认值填充的代价：38% 的样本会被模型当成"真零"。
- **`multimodal_fusion.py`**：用 `MockEncoder`（随机投影 + ReLU + L2 归一化）模拟文本/视觉/音频三个模态编码器，对比三种融合方式。在"库内已有物品"上，attention 融合的权重集中在 0.31–0.35 之间（query 为用户兴趣向量）；在**冷启动**场景（新物品没有 ID 特征、没有行为历史），三种融合与库内物品的最大相似度为 concat **0.6976** → weighted **0.8425** → attention **0.8635**。结论对应正文：ID 特征只能记忆，多模态特征才能泛化到新物品。

### §15.2 特征表示

- **`sparse_dense_cost.py`**：按 `参数量 = 词表大小 × d` 计算六张 Embedding 表。`user_id` 1 亿词表 × d=64 → **6.4e9 参数 / 23.84 GB**，六张表合计 **2.66e10 参数 / 99.18 GB**。维度扫描给出 d=8 到 d=256 的线性膨胀（2.98 GB → 95.37 GB）。四种压缩手段的收益与代价：d 减半 → 0.5x，频次截断保留 30% 活跃 ID → 0.3x，哈希到 1000 万桶 → 0.1x，int8 量化 → 0.25x。末尾的 Pareto 抽样显示 **54.9% 的头部 ID 覆盖了 90% 的交互**，而 45.1% 的长尾 ID 共享不到 10%，其 Embedding 训练严重不足——这就是频次截断与哈希桶的动机。
- **`representation_compare.py`**：手写 Logistic Regression（400 步、l2=1e-3）与带并列处理的 AUC，在同一份"用户对某类目的 5 天兴趣轨迹"上比较五种表示。只看总量时四种表示差异很小：Sparse ID **0.6099**（6 与 7 是两个独立类别，也看不到趋势）、Dense 总次数 **0.6391**、加 log1p 同为 0.6391、8 分位桶 **0.6351**；保留轨迹的 Sequence **0.6969**；用"真实趋势"作为特征的上界对照 **0.7074**。这说明统计特征与序列特征不是替代关系：趋势信息只在序列里。

### §15.3–§15.4 行为建模与实时特征

- **`multi_window_behavior_features.py`**：对同一批 200 个用户、30 天事件流计算 1d / 7d / 30d 的点击数与观看时长，以及衰减权重 `w = exp(-0.1·Δt)`。三个可读结论：(a) **1d Top-10 与 30d Top-10 只重合 1 人**，短期与长期捕捉的是不同尺度；(b) 衰减权重把"历史多但近期不活跃"的用户从第 134 名拉到第 58 名（上升 76 位）；(c) F/R/I 三维画像能把用户分成"高频+近期+深度""低频+近期""高频+久未活跃（流失预警）"等状态，单一点击次数无法区分。
- **`user_profile_interests.py`**：用 30 天 vs 3 天的类目分布对比长期与短期兴趣，平均 KL **0.611**；KL 大的用户在建模时应更依赖短期序列。用 spherical k-means 把每个用户拆成 K=3 个兴趣中心（例：uid 0 的中心占比 0.16/0.81/0.03，主类目 0/7/1）。召回打分对比显示多兴趣取 max 的最大分显著高于平均向量（uid 0：0.775 → 0.881；uid 1：0.900 → 0.932），小众兴趣不会被平均掉。
- **`realtime_session_features.py`**：构造 18 小时分钟级行为流，在第 300 分钟注入"体育 → 游戏"的兴趣切换，比较四类特征的响应速度。切换后第 8 分钟：离线（日级）仍是体育 100%、准实时（小时）98%、实时（分钟）95%、Session 仅剩 80% 体育；到第 482 分钟，准实时与实时都只降到体育 61%（尚未追上），而 Session 已经是游戏 80%，离线仍停在 100% 体育。更新延迟实验给出 60 分钟窗口特征在滞后 0/5/15/30/60 分钟时的覆盖率 **100% / 85% / 62% / 38% / 0%**——实时特征的价值不是"更新快"，而是在可接受延迟内保持分布一致。
- **`training_serving_skew.py`**：用户在第 7 天发生兴趣切换，比较"特征窗口包含未来"与"严格截止到预测时刻"两种构造。泄露版 AUC **0.6408** vs 干净版 **0.6048**（虚高 0.036）。在线变体表逐项叠加"延迟 + 缺失 + 归一化口径不一致"，AUC 从 0.6048 一路降到 0.6016。Action List 长度实验复现正文的隐蔽泄露：离线读取最新 L 条历史时，L=10 平均含 8.30 条未来行为、AUC 0.5806；L≥50 平均含 14.07 条、AUC 0.6695——序列越长，"最新列表"里混入的未来行为越多，离线指标越虚高。

### §15.5 特征交互

所有模型共用 `_train_utils.py`：4 万训练 / 8 千验证样本、10 个 field、Embedding 维度 4、hidden=(32,)、dropout 0.2、600 步 Adam（lr=2e-3, weight_decay=1e-5）。数据里埋了两处真正的二阶交互（阈值化的用户-物品偏好匹配、用户-物品隐向量点积超过 1.2），所以"能不能显式建模交互"会被拉开差距。**每张表里的 Baseline 都是同一次训练循环下的纯 Embedding+MLP，AUC ≈ 0.7187**；不同脚本的训练步数不完全相同（各自 200–600 步），因此同一模型在不同脚本里会有约 ±0.005 的波动（例如 DCN-V2 满矩阵在单独脚本里是 0.7175，在统一对比的 600 步下是 0.7223）。横向比较请以 `interaction_benchmark.py` 的同一份数据为准。

| 模型 | 参数量 | val AUC | 脚本里的关键诊断输出 |
|---|---|---|---|
| Baseline MLP | 4,025 | 0.7187 | — |
| DCN（2 层 Cross） | 4,225 | **0.7211** | 交叉层只有 40 维向量参数 |
| DCN-V2（满矩阵） | 7,345 | 0.7223 | 低秩 r=8：5,425 参数 / 0.7183 |
| AutoInt（1 block, 2 heads） | 4,113 | 0.7215 | 10×10 注意力图 |
| FiBiNet Field-All | 14,396 | 0.7186 | Field-Each 14,540 参数，AUC 0.7183 |
| AFN（12 神经元） | 4,577 | 0.7193 | 学到的阶数 O_j 在 0.45–1.05 之间（12 个神经元各不相同） |
| FED | 4,387 | **0.7227** | DRM 216 + Field-wise 48 参数 |
| CAN | 12,281 | 0.7174 | 参数量为笛卡尔积方案的 1/400 |
| CAN + Multi-order | 12,281 | 0.7193 | tanh 约束高阶项 |
| FINT（2 层） | 4,245 | 0.7136 | 1/2/3 层对比：0.7152/0.7136/0.7123 |
| GDCN-P（并联） | 14,025 | **0.7232** | 门控均值 0.65/0.57/0.54 |
| IPA（projected/field/layer） | 6,188 | 0.7197 | element 聚合 0.7200 |

- **`dcn.py`**：`CrossLayer` 实现 `x0·(x_lᵀw) + b + x_l`，2 层交叉。交叉层参数量与输入维度线性相关（这里只有 40 维，单层 80 个参数），所以 DCN 与 Baseline 参数量只差 200。
- **`dcn_v2.py`**：三种参数化并排——满矩阵 `W∈R^{40×40}`（3,280 参数）、低秩 `UVᵀ`（r=8 时 1,360 参数）、`r=0` 退化为向量版（实现上单独走 `x_l·w` 的加权求和，既对应 DCN 的向量形式，也避免创建 0 维 Linear）。满矩阵参数量最多，AUC 反而不如低秩：0.7175 vs 0.7183。
- **`autoint.py`**：多头自注意力（2 heads），带残差 + LayerNorm，打印第一个 block 的 10×10 平均注意力图。对角线保留自身信息，非对角元表示 field 间依赖；权重随样本变化，这是它和固定 Cross 结构的本质区别。
- **`fibinet.py`**：SENET 先 squeeze（每 field 求均值）再 excitation（两层线性 + sigmoid）再重加权；Bilinear 同时实现 Field-All / Field-Each。参数量从 All 到 Each 涨了 10 倍，AUC 基本不动（0.7186 vs 0.7183）——正文"表达能力与成本权衡"的直接体现。
- **`afn.py`**：对数神经元 `z_j = exp(Σ_i w_ij ln e_i)`，Embedding 先取绝对值加 1e-3 再取对数，前后各一个 BatchNorm，`exp` 前 clamp 到 [-6, 6] 防溢出。训练后 12 个神经元的有效阶数 `O_j = Σ|w_ij|` 落在 **0.45–1.05**，模型自己选择了从"几乎不交互"到"多阶"的混合，不需要人工指定阶数。
- **`fed.py`**：DRM 在 Embedding 维度上做注意力聚合、Field-wise 在原始 Field 上做聚合、Element-wise 把结果展平送 DNN。实现时按 `E = [S (V W_δ)ᵀ]ᵀ + V` 组织矩阵乘法，使 `d×d` 的注意力矩阵作用在维度轴上（`m×d` 与 `d×d` 直接相乘维度不匹配，脚本注释里写明了这一点）。
- **`can.py`**：`CoActionUnit` 从 Induction 侧参数表里切出每层的 `d×d` 权重与 `d` 维 bias，作用在 Feed 侧上；每层输出都保留并拼接（对应正文 `Concat(h_1..h_L)`）。Multi-order 版本额外把 `tanh(P_feed²)` 送进同一个 Micro-MLP 再求和。参数量对照：笛卡尔积 200×400×4 = **320,000**，CAN 的 induction 表 200×40 = **8,000**（1/400）。初始化 std 取 0.5：实测 0.02 时 Micro-MLP 输出接近常数、AUC 掉约 0.006。
- **`fint.py`**：每层 `v_i^l = Σ_j w_ij (v_i^{l-1} ⊙ v_j^0) + u_i v_i^{l-1}`，w 是 M×M 的 Field 级权重、u 是残差标量。堆 1/2/3 层对应 0.7152/0.7136/0.7123（参数量只线性增长），在这个小数据设定下加深没有收益——正文也强调层数要跟数据量匹配。
- **`gdcn.py`**：`c_{l+1} = c0 ⊙ (W^c c_l + b) ⊙ σ(W^g c_l) + c_l`，3 层。串联 GDCN-S 0.7211、并联 GDCN-P **0.7232**；去掉门控（等价 DCN-V2）掉到 0.7183，说明 gate 是有效项。训练后打印三层 gate 的均值 0.647/0.570/0.540、极值范围，且随样本变化——这是"实例级筛选"的直接证据。
- **`ipa.py`**：三种组件各扫一遍。交互函数 naive/weighted/diagonal/projected 的 AUC 为 0.7104/0.7112/0.7111/0.7197；层池化 field vs global 为 0.7197 vs 0.7145；层聚合 direct/layer/term/element 为 0.7174/0.7197/0.7199/0.7200，参数量从 0 涨到 120。同一个框架下可以拼出 FM、DCN-V2、xDeepFM 等已有模型，上面三组对比就是在设计空间里逐维扫了一遍。
- **`interaction_benchmark.py`**：把上表所有模型放在同一份数据、同一训练配置下跑一遍，约 37–45s。读数提示写在脚本末尾：千分之几的差距很容易被优化噪声淹没；参数量增长远快于 AUC 增长；要看某类交互的真实效果，应回到对应脚本的诊断输出（交叉项、注意力图、阶数、门控值）。

### §15.6 特征筛选与治理

公共数据 `_lr_utils.py` 生成 12 个特征、3 万条样本的合成 CTR 数据，其中 7 个真正影响点击（`user_activity`、`clicks_7d`、`item_ctr`、`cat_match`、`is_new_user`、`hour_bucket`、`device`），3 个纯噪声、`clicks_30d` 与 `clicks_7d` 高度相关。标准化参数只用训练集统计量——这本身就是 §15.6.6 要点名的泄露点。

- **`feature_importance.py`**：全特征基线 val AUC **0.7849**。三种静态重要性对照：`user_activity` 的 |w|=0.809、单特征 AUC 0.7173、置换 ΔAUC **0.1214**；`clicks_7d` 单特征 AUC 0.7148 但置换只有 0.0067——它的信息被 `user_activity` 覆盖了。LHUC 式门控（`g_i = σ(r_i)`，梯度按 batch 幅度归一化）把噪声特征压到 **0.21–0.27**、与活跃度重复的 `hist_len` 压到 **0.006**，而真信号保持在 0.76–1.00；硬删除后重训（8 个特征）AUC **0.7851**，与全量持平。
- **`feature_redundancy.py`**：相关系数表列出 5 组 |r|>0.5 的对子。冗余对处理对照：去掉 `clicks_30d` 掉 0.0001，去掉 `user_activity` 掉 **0.0067**——同样是"冗余对"，保留哪一个要看 AUC 之外的更新频率与成本。前向边际增益逐个加入特征：1→4 个特征把 AUC 从 0.7173 推到 0.7853，第 5 个开始 ΔAUC 基本为 0。分组消融：删除"物品属性"整组下降 **0.0577**，删除"上下文"只下降 0.0001。
- **`feature_cost_roi.py`**：按 `C = C_计算 + C_存储 + C_网络 + C_延迟`（延迟 1ms ≈ 3.0 元/百万请求，全部 [MOCK]）给 12 个特征算账。总成本最高的是 `user_activity` 15.70、`clicks_7d` 7.90、`clicks_30d` 6.65；ROI（每千元换来的 ΔAUC）最高的是 `user_activity` 13.84、`item_ctr` 10.16。四象限里"高收益低成本"只有 `cat_match`，"高收益高成本"有 `user_activity`/`item_ctr`/`clicks_7d`，"低收益高成本（优先下线）"是 `follow_ratio` 与 `clicks_30d`。成本拆解显示计算项占 60–76%，但延迟项往往是最容易忽略、也最难优化的一项。
- **`feature_quality.py`**：14 天逐日快照，第 9 天注入上游故障。覆盖率从 **96% 掉到 49%**，p99 新鲜度从 17 分钟跳到 **287 分钟**，特征均值从 1.67 被系统性低估到 0.53。z-score 用的是"最近 7 个正常日"的滑动均值与标准差（故障期不进入基线），在 D9 直接跌到 **−15**。脚本同时说明：z-score 只看总体均值，如果故障只影响一小群用户，它可能仍然正常——所以覆盖率、分群覆盖率、新鲜度、分布指标必须一起看。
- **`data_drift.py`**：三种时期 + 两种注入。A 平滑漂移（活跃度上移 0.8σ）：`user_activity` 的 PSI **0.623**，label 率几乎不变，AUC 甚至略升（0.7863 → 0.7905），但预测均值从 0.372 抬到 0.491（校准偏差 +0.126）——排序没坏、概率不能直接用了。B 结构漂移（类目匹配比例 35%→65%，并抬高点击率）：`cat_match` PSI **0.349**，label 率 0.367 → 0.407，AUC 掉到 **0.7491**。这组对照说明 PSI 只能说明"输入变了"，无法单独判断"模型是否失效"。
- **`feature_leakage.py`**：三组对照。(a) 未来行为泄露：只用 t 之前 24h 行为 AUC **0.6843**，加入 t 之后 24h 行为后 **0.9470**（虚高 0.2627）。(b) 统计窗口泄露：`[t−7d, t)` 0.6796，越界到 `[t−7d, t+2h)` 0.7369——越界窗口直接把这次曝光之后的点击算了进去。(c) Action List 错位：样本时刻 t 的序列只有 80 条，训练任务读的是当前 14,000 条，两者兴趣分布余弦相似度只有 **0.0054**，主兴趣从类目 1 变成类目 5。
- **`feature_governance.py`**：12 条 [MOCK] 注册表记录，字段覆盖正文列的 定义/负责人/来源/新鲜度/覆盖率/成本/漂移/使用方/生命周期。自动处置结果：下线 2（`clicks_30d`、`follow_ratio`）、保留 7、待讨论 3；"僵尸特征"（低收益 + 使用方 ≤ 1）3 个；覆盖率不足 90% 的 3 个（`follow_ratio`、`is_new_user`、`price_band`）单独标出——低覆盖会让特征在离线评估里天然吃亏，不能直接判死刑。末尾打印下线评审的六项证据清单。

---

## 实现约定

- **自包含**：`ch01`–`ch03` 的脚本之间不互相 import（共享的合成数据放在 `common.py`）；`ch04` 的模型脚本只 import 同目录的 `_train_utils.py`，`ch05` 只 import `_lr_utils.py`。每个文件都能单独运行。
- **可复现**：所有脚本固定随机种子（`random.Random(42)` / `np.random.default_rng` / `torch.manual_seed(0)`），输出可复现。DCN/CAN 等入门的 torch 脚本里额外给 Embedding 表 `torch.manual_seed(0)`，避免模型间初始化差异掩盖结构差异。
- **[MOCK] 标注**：多模态编码器（随机投影模拟）、特征成本单价、告警阈值、注册表与使用方数量都在 docstring 或注释里标了 `[MOCK]`。
- **以已发布 PDF 为准**：公式与流程以第 15 章发布 PDF 的相应小节为准——DCN 的 `x0x_lᵀw + b + x_l`、DCN-V2 的低秩 `UVᵀ`、AutoInt 的缩放点积注意力、FiBiNet 的 squeeze/excitation 与三种 Bilinear、AFN 的 `exp(Σ w ln e)` 与阶数定义、FED 的 DRM/Field-wise/Element-wise、CAN 的 Induction/Feed 角色与 `Concat(h_1..h_L)`、FINT 的 Field 级权重与残差、GDCN 的门控交叉公式、IPA 的三组件与四种聚合方式的参数量（0 / L / LM / LMK）、§15.6 的 ΔM、四次成本分解、Coverage 定义、PSI 与 `F(t) ⊆ I_{≤t}`。
- **数值只用于横向对照**：§15.5 的实验是 4 万样本、Embedding 4 维、隐层 32 维的快速实验，AUC 差异只有千分之几，目的是让不同结构在同一份数据上可比较，不代表线上收益。
- **运行开销**（单机 CPU，串行）：
  - §15.1–§15.3：每个脚本 0.05–2.3s，全部 < 3s；
  - §15.5：单个模型 8–18s（`ipa.py` 因为要扫 10 组结构约 56s）；`interaction_benchmark.py` 约 38s；首次 `import torch` 本身需要 3–9s；
  - §15.6：每个脚本 0.05–2.5s。
