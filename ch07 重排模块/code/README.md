# ch07 重排模块 — 代码示例

本目录对应教材第 7 章，覆盖重排模块的完整链路：为什么需要重排、重排架构、序列生成（Actor）、序列评估与选序（Critic）、多样性打散（MMR / DPP / SSD）、重排后处理，共 **14 个可运行脚本**。目录编号与正文节次一一对应，每个脚本自包含、互不 import，均可直接 `python3 <文件>.py` 运行。依赖：Python 3.12 标准库 + `numpy` 1.26.4，其中 `listwise_evaluator_model.py` 与 `list_value_definitions.py` 额外依赖 **PyTorch**（CPU 即可）。

---

## 目录结构

```
code/
├── ch01_why_rerank/                            § 7.1 为什么需要重排
│   ├── pointwise_vs_listwise.py          精排 pointwise 贪心 vs 序列级最优（全排列枚举验证）
│   └── position_bias.py                  位置偏置曲线拟合 + 位置权重 w_i 的三种形式对比
├── ch02_rerank_architecture/                   § 7.2 重排模块架构
│   └── rerank_pipeline.py                五阶段重排链路：候选接入→序列生成→评估→打散→后处理
├── ch03_list_generation/                       § 7.3 序列生成
│   ├── greedy_fusion_gaussian_sampling.py 融合公式贪心生成 + 权重高斯采样扩大序列空间
│   └── beam_search_generation.py         逐位置 Beam Search + 业务约束剪枝，与穷举最优对齐
├── ch04_list_evaluation/                       § 7.4 序列评估与排序
│   ├── list_value_definitions.py         三种序列价值定义（位置加权和/阈值二分类/下滑概率累乘）
│   ├── listwise_evaluator_model.py       序列评估模型：Pointwise 基线 / DLCM(GRU) / PRM(自注意力)
│   └── list_selection_strategies.py      多路 Actor 并行生成 + Critic 选序策略（argmax/软惩罚/硬过滤/采样）
├── ch05_diversity/                             § 7.5 多样性打散
│   ├── mmr_diversification.py            § 7.5.1 MMR：IR 场景复刻 + 推荐场景 λ 扫描
│   ├── dpp_greedy_map.py                 § 7.5.2 DPP：核矩阵构造 + 贪心 MAP + Cholesky 增量加速
│   ├── ssd_sliding_spectrum.py           § 7.5.3 SSD：GS/MGS/one-step MGS + 滑动窗口谱分解
│   └── diversity_algorithms_comparison.py 三者在同一候选集上的等相关性损失对比与复杂度对比
└── ch06_post_processing/                       § 7.6 重排后处理
    ├── rule_based_shuffle_and_repair.py  规则化打散、序列合法性校验与最小位移修正、兜底补齐
    └── quota_injection_and_data_closure.py 强插/保量 + [MOCK] Redis 回溯 + [MOCK] Kafka 日志闭环
```

---

## 快速运行

```bash
# § 7.1 为什么需要重排
python3 ch01_why_rerank/pointwise_vs_listwise.py
python3 ch01_why_rerank/position_bias.py

# § 7.2 重排模块架构
python3 ch02_rerank_architecture/rerank_pipeline.py

# § 7.3 序列生成
python3 ch03_list_generation/greedy_fusion_gaussian_sampling.py
python3 ch03_list_generation/beam_search_generation.py

# § 7.4 序列评估与排序
python3 ch04_list_evaluation/list_value_definitions.py
python3 ch04_list_evaluation/listwise_evaluator_model.py
python3 ch04_list_evaluation/list_selection_strategies.py

# § 7.5 多样性打散
python3 ch05_diversity/mmr_diversification.py
python3 ch05_diversity/dpp_greedy_map.py
python3 ch05_diversity/ssd_sliding_spectrum.py
python3 ch05_diversity/diversity_algorithms_comparison.py

# § 7.6 重排后处理
python3 ch06_post_processing/rule_based_shuffle_and_repair.py
python3 ch06_post_processing/quota_injection_and_data_closure.py
```

全部脚本在 CPU 上单次运行均 ≤ 20s（最慢为 `listwise_evaluator_model.py`，约 13s），退出码 0，运行表见文末。

---

## 各脚本要点

### § 7.1 为什么需要重排

- **`pointwise_vs_listwise.py`**：N=6 候选、T=4 坑位，枚举 **P(6,4)=360** 个排列。精排按 `f(x)` 降序的贪心序列 value=**106.0442**，在 360 个排列中仅排第 **10 名**；序列级最优 value=**111.9133**，提升 **5.53%**；最差排列 value=**55.5525**（最优/最差相差 **101.45%**）。贪心序列相邻同垂类 **3 对**，序列最优只剩 **1 对**——收益来自把一个萌宠物品插到第 2 位，牺牲单点分（加和 157.92→146.88）换取后续曝光概率。
- **`position_bias.py`**：拟合经验衰减曲线，并对比正文的两种位置权重 `w_i=1/i` 与 `w_i=e^{-(i-1)}`。`e^{-(i-1)}` 到第 5 位权重已不足 **2%**，等价于只优化头部 2~3 个位置；`1/i` 衰减更缓。另外用枚举证明：**只有位置权重、没有物品间交互时，按 `f` 降序即最优**（两种权重下枚举最优序列都等于 `f` 降序序列，value 分别 85.8058 / 69.0923，完全一致）。这正是正文强调的「位置偏置解释了为什么加权，但不构成重排的收益来源」。

### § 7.2 重排模块架构

- **`rerank_pipeline.py`**：进程内 mock 跑通正文 § 7.2 的五阶段链路。精排 Top-30 接入 → 采样 8 组融合权重生成 8 条序列 → listwise 打分选中 value=**521.8192** 的序列0（注意点加和最高的序列5 点加和 1105.5539 但 value 仅 374.0807，**点加和最高 ≠ 序列价值最高**）→ MMR(λ=0.7) 打散：垂类覆盖 **2/5→4/5**，相邻同垂类 **5→3**，平均两两 cos **0.2896→0.0805** → 后处理命中 `relax@pos10` 与 `force_insert@pos3`，最终垂类覆盖 **5/5**、相邻同垂类 **0**、平均 cos **0.0583**。所有外部依赖（存储/RPC/日志）均标 `[MOCK]`。

### § 7.3 序列生成

- **`greedy_fusion_gaussian_sampling.py`**：复刻正文的融合公式贪心生成 + 权重高斯采样。基准权重 `w0=[0.2,0.3,0.2,0.2,0.1]` 贪心序列融合分和 **3.1298**；从 `N(w0, 0.01I)` 采 8 组权重得到 **6 条互不相同**的序列（重复率 **25.0%**）。协方差扫描（各采 200 组）：对角值 1e-4 → 仅 **3** 条不同序列、首位 100% 是 i1；1e-3 → **16** 条；1e-2 → **106** 条；0.1 → **171** 条；1.0 → **187** 条，首位分布从 `i1:200` 散到 `i1:32, i4:26, i3:25, i6:25`。探索范围随协方差单调扩大。
- **`beam_search_generation.py`**：逐位置 Beam Search + 「相邻同垂类禁止」剪枝。beam=1 扩展 **45** 节点（剪掉 11.1%）得序列分 **1.3170**；beam=4 扩展 **150**（剪 23.3%）得 **1.3176**；beam=16 扩展 **516**（剪 29.8%）得 **1.3362**，与穷举最优 **1.3362** 差距 **0.000000**。穷举空间 P(10,6)=**151200** 条排列（其中合法 **35760** 条），beam=16 只做 O(T·B·N)=**960** 次扩展即达到最优。

### § 7.4 序列评估与排序

- **`list_value_definitions.py`**：实现正文给出的三种序列价值定义并横向对比选序能力（40 条候选序列、300 条训练序列）。Spearman / top-1 命中 / 价值损失：**位置加权和 w=1/i** 0.8274 / 48.3% / 4.08%；**w=e^{-(i-1)}** 0.7187 / 26.7% / 6.59%；**阈值二分类 logloss** 0.7366 / 35.0% / 6.09%；**下滑概率累乘**（正文 `p(item_i)=p0·∏p_slide`）**0.7533 / 58.3% / 2.96%**；朴素单点分加和 0.6076 / 18.3% / 10.00%。与数据生成过程同构的「下滑概率累乘」口径 top-1 命中与价值损失最好。
- **`listwise_evaluator_model.py`**：按正文逐物品损失 `L=(1/B)ΣΣ||ŷ-value||²`、标签 `value_i=p(item_i)·f(x_i)` 训练三个评估器（训练 900 / 测试 300 条，N=10，40 epoch）。
  - 测试精度（逐物品 RMSE 秒 / 序列价值 MAE 秒 / MAPE）：PointwiseMLP **9.7474 / 16.4431 / 104.96%**；DLCM(GRU) **0.5627 / 1.7368 / 5.59%**；PRM(SelfAttn) **0.8071 / 1.9070 / 7.03%**。
  - 选序能力（同一池 32 条排列）Spearman / top-1 / regret：Pointwise **0.3054 / 15.0% / 34.18%**；DLCM **0.9852 / 85.0% / 0.84%**；PRM **0.9867 / 86.7% / 0.30%**；单点分朴素加和 0.3124 / 11.7% / 37.37%。
  - 位置敏感性（同一集合不同排列的预测价值 std）：真实环境 **23.6382**，Pointwise **0.0000**（极差/均值 0.00%），DLCM **24.7014**，PRM **25.4839**。这组 0 值就是正文「单点打分无法做序列级优化」的直接数值证据。
  - **与正文的差异**：正文 § 7.4 只规定了监督范式与上述损失，未点名网络结构；DLCM / PRM 两种结构取自公开论文（见下文出处表），训练目标与标签严格按正文公式。
- **`list_selection_strategies.py`**：把 § 7.3 的多路 Actor（greedy / 权重高斯采样 / beam / 交换邻域 / MMR 式）与 § 7.4 的 Critic（真实参数 + `N(0,0.18²)` 扰动）接起来，按正文的 Actor-Critic 视角做选序。单次请求 29 条候选序列中 beam 路 Critic 分最高（210.963）且 **违约束占比 0.0%**（全局 82.8%）。120 次请求的 regret / 约束违例率：**oracle 0.00% / 53.3%**；**argmax 1.54% / 55.8%**；**soft_penalty(β=0.06) 1.80% / 42.5%**；**constrained 硬过滤 8.18% / 1.7%**；**softmax 采样 T=0.05 2.76% / 55.8%**；池内最差 56.52%；**greedy 单路无选序 22.89%**。换 Critic 口径后 regret / top-1 命中率：`Σp(item_i)f(x_i)` **3.35% / 53.3%**，`Σf/i` **16.17% / 5.0%**，`Σe^{-(i-1)}f` **17.21% / 0.0%**，无位置权重加和 **16.93% / 3.3%**。

### § 7.5 多样性打散

- **`mmr_diversification.py`**（§ 7.5.1）：严格按正文 MMR 公式
  `MMR = argmax_{D_i∈R\S} [λ·Sim₁(D_i,Q) − (1−λ)·max_{D_j∈S} Sim₂(D_i,D_j)]`。
  先复刻正文 `code_mmr_ir` 的 7 篇中文文档检索例子：纯相关性（λ=1）选出 doc1/doc0/doc4，集合内最大两两相似度 **0.3484**；MMR(λ=0.6) 选出 doc1/doc4/doc5，最大相似度降到 **0.0581**（doc0 因与 doc1 冗余被剔除，正是正文的预期行为）。再在推荐场景（N=36、M=10）做 λ 扫描：λ=1.0 相关性损失 0.00%、ILD 0.9072、类目熵 1.3322、相邻同类 **3** 对；λ=0.7 损失 **0.22%**、ILD 0.9670、类目熵 1.5571；λ=0.3 损失 8.35%；λ=0.0 损失 **40.71%**、类目数 **6**、类目熵 **1.7481**、相邻同类 **0**。
- **`dpp_greedy_map.py`**（§ 7.5.2）：按正文构造 `B_i=r_i f_i`、`S_ij=(1+⟨f_i,f_j⟩)/2`、`L=Diag(r)·S·Diag(r)`、`L'=Diag(e^{αr})·S·Diag(e^{αr})`，`α=θ/(2(1−θ))`。
  - 性质验证（N=60, d=16）：S 最小特征值 **−9.07e−16**、L 最小特征值 **−1.64e−15**（数值意义上半正定）；正文恒等式 `log det L_Y = Σ log r_i² + log det S_Y` 在前 8 个物品子阵上两边均为 **−18.200985**（|差| **9.3e−08**）。
  - 贪心 MAP（θ=0.7, K=10）：朴素实现与 Cholesky 增量实现选出**完全相同**的集合 `[3,25,6,59,42,40,15,7,45,26]`，耗时 **7.06ms → 1.08ms**，加速比 **6.53×**。规模扫描（朴素/增量/加速比）：N=40,K=8 **4.28/1.03/4.16×**；N=80,K=10 **12.59/1.43/8.81×**；N=120,K=12 **21.02/7.43/2.83×**；N=200,K=15 **37.69/4.51/8.37×**，集合一致性全部 True（单次测时有抖动，故加速比非单调）。
  - 增量量正确性：`log(d_i²)` 与真实边际增益 `log det(L_{Y∪{i}}) − log det(L_Y)` 在抽检的 5 个候选上误差 ≤ **1.7e−10**。
  - 几何解释：纯相关性 Top-K 的 `log det(L_Y)=13.5993`、Vol=**8.98e+02**；DPP 贪心 `log det(L_Y)=15.1318`、Vol=**1.93e+03**，体积大 1 个数量级以上。
  - θ 扫描（相关性损失 / log det S_Y / 类目熵）：θ=0.99 **0.00% / −9.4562 / 1.2799**；θ=0.8 **1.37% / −7.8012 / 1.5048**；θ=0.6 **5.34% / −6.6915 / 1.6957**；θ=0.0 **54.81%**。**注意一个与直觉相反的点**：正文代码在 θ=1 时令 α=0，此时 `L'=S`，相关性被完全抹掉，结果反而等同「纯多样性」（表中 θ=1.00 与 θ=0.00 两行数值完全一致）；真正的纯相关性要 α→∞，脚本用 θ=0.99 近似。
- **`ssd_sliding_spectrum.py`**（§ 7.5.3）：按正文实现 one-step MGS 残差与滑动窗口谱分解。
  - 正交化对照（最大 `|EᵀE−I|`）：良态随机 24×24（cond 2.43e2）GS **1.31e−13** / MGS **8.12e−15** / 1step **1.06e−14**；病态 cond=1e10 的 16×16：GS **9.99e−01**（正交性完全崩塌）/ MGS **1.04e−07** / 1step **1.07e−07**；近线性相关 16×16（cond 1.91e9）：GS **1.08e−01** / MGS **6.77e−09** / 1step **8.54e−08**。one-step MGS 只扣除最新向量的分量，换来的是 **O(d)** 单步开销与「局部时序多样性」语义。
  - γ 扫描（N=60, T=12, W=5，相关性损失 / ILD / 类目熵 / Σσ）：γ=0 **0.00% / 0.8740 / 1.4735 / 4.8073**；γ=5 **6.00% / 0.9738 / 1.5171 / 4.9126**；γ=10 **21.86% / 0.9966 / 1.5454 / 4.9319**；γ≥20 饱和在 **36.81% / 0.9648 / 1.6762 / 4.9503**。
  - 全序列版 vs 滑动窗口（γ=10，相关性 / 相邻同类 / Σσ）：全序列 W=T **0.7576 / 2 / 4.9319**；W=2 **0.9378 / 0 / 4.8389**；W=3 **0.9177 / 1 / 4.8943**；W=5 **0.8621 / 0 / 4.9436**；W=8 **0.8262 / 2 / 4.9202**。窗口越小保留的相关性越多，但窗口外的长程重复不再受约束。
  - 复杂度验证（T=12）：N=200 **7.77ms**（0.03885 ms/N）、N=400 **15.60ms**（0.03901）、N=800 **31.71ms**（0.03963）、N=1600 **63.67ms**（0.03979），ms/N 基本不随 N 增长，印证 O(N·T·d)。
- **`diversity_algorithms_comparison.py`**：三者跑在**同一** N=60 / d=16 / K=12 / 6 类目的候选集上。基线（纯精排 Top-K）avg_rel=0.9696、ILD=0.8740、类目熵=1.4735、`log det S_Y=−13.5474`。
  - **等相关性损失档（各取最接近 5% 的一档）**：MMR λ=0.50 → 损失 **2.30%**、ILD 0.9435、类目熵 1.6326、相邻同类 **4**、`log det S_Y=−14.1632`；DPP θ=0.70 → 损失 **4.25%**、ILD **1.0088**、类目熵 **1.7046**、相邻同类 **0**、`log det S_Y=−9.9303`；SSD γ=5.00 → 损失 **6.00%**、ILD 0.9738、类目熵 1.5171、相邻同类 1、`log det S_Y=−10.6166`。耗时量级 MMR≈DPP≈0.8~1.0ms、SSD≈2.2ms（单次测时有抖动）。同等相关性代价下 DPP 的集合级多样性（ILD、类目熵、log det）最好。
  - 集合重叠（Jaccard）：MMR-DPP **0.500**、DPP-SSD **0.600**、MMR-SSD **0.500**，MMR-基线 **0.714**（MMR 最贴近精排结果）。
  - 耗时随 N 增长（K=12，MMR/DPP/SSD，ms，两次独立运行）：N=60 **0.79/0.89/2.30** 与 **0.83/0.99/2.35**；N=120 **1.66/2.02/4.49** 与 **2.83/2.80/5.03**；N=240 **3.42/4.28/10.00** 与 **3.50/6.91/9.26**；N=480 **8.55/9.00/18.33** 与 **7.16/13.12/23.37**；N=960 **15.63/27.62/47.03** 与 **20.07/30.12/43.00**。绝对值有抖动，但三者都近似随 N 线性增长。

### § 7.6 重排后处理

- **`rule_based_shuffle_and_repair.py`**：实现正文「基于规则约束或轻量级贪心策略，在局部窗口内微调」的四个算子——合法性校验（相邻同作者/同类目、窗口 W=4 内同作者≤1 同类目≤2、头部 3 位类目互异）、滑窗最小位移修正、局部窗口交换、兜底补齐。
  - 单次请求：原始精排序列违例 **22** 个（相邻同作者 5 / 相邻同类目 6 / 窗口同作者 5 / 窗口类目超限 4 / 头部重复 2），w=1/i 加权分 2.9950、相邻异类目比例 0.455；滑窗最小位移修正后违例 **0**、放宽 0 次、加权分 2.9738、相邻异类目比例 **1.000**。
  - 100 次请求统计（违例数 / Kendall τ / 平均位移 / w=1/i 分损失 / 相邻异类目 / 放宽次数）：原始精排 **22.47 / 0 / 0 / 0% / 0.456 / 0**；滑窗最小位移 **0.00 / 0.2728 / 2.97 / 1.23% / 1.000 / 0**；局部交换 R=1 **12.90 / 0.0623 / 0.69 / 0.08% / 0.775 / 5.31**；R=3 **4.70 / 0.1981 / 1.93 / 0.61% / 0.897 / 2.57**；R=6 **1.34 / 0.2590 / 2.63 / 0.94% / 0.972 / 0.63**。位移半径越大违例清得越干净，排序破坏（τ、分损失）同步增大。
  - 兜底补齐：约束卡死导致打散后仅 **7 < 12** 坑位，从兜底池（大小 10）无条件强补 **5** 个（标 `*`），补齐后长度 12、违例 14——说明「坑位优先于约束」的降级口径必须显式存在。
- **`quota_injection_and_data_closure.py`**：实现正文 § 7.6 的强插/保量 + 数据闭环。
  - 单次链路（L=12，保量下限 ad 2 / activity 1 / operation 1，占比上限 ad 0.25 / activity 0.17 / operation 0.17，强插第 4、9 位）：重排输出 → 强插后挤出自然内容 **2** 条 → 保量补入 `{'ad': 2}`、超限剔除 `{}`。三阶段自然占比 **1.000 → 0.833 → 0.667**，强插比例 **0 → 0.167 → 0.167**，w=1/i 分 **2.9129 → 2.8770 → 2.8367**。
  - 60 次请求 A/B：A（纯重排）自然 1.000、作者覆盖 0.436、w=1/i 分 **2.9273**；B（强插+保量）自然 **0.667**、广告 0.167、活动 0.083、运营 0.083、作者覆盖 **0.675**、w=1/i 分 **2.8260**，**序列分损失 3.46%**。执行顺序必须是「强插 → 保量补齐 → 占比上限截断」，否则上限会被强插突破。
  - 数据闭环（均标 `[MOCK]`）：Redis `setex` **60** 次、可按 requestId 回溯序列/来源通路/强插标记，TTL=300s 时模拟 600s 后查询返回 **None（过期 miss）**；Kafka topic `rerank_exposure_log` 3 分区，send 成功 **720** 条、丢弃 **0**，flush **45** 批、平均批大小 **16.0**、分区分布 `{0:192, 1:264, 2:264}`；后处理 + 写入总耗时 **5.2ms**（**0.09ms/请求**）。

---

## 论文出处（仅列正文明确提到、且标题已核实者）

| 算法 | 论文 | 年份/会议 |
| --- | --- | --- |
| MMR | The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries（Carbonell & Goldstein，正文 `cite{carbonell1998use}`） | 1998 |
| DPP 理论源头 | The coincidence approach to stochastic point processes（Macchi，正文 `cite{macchi1975coincidence}`） | 1975 |
| DPP 机器学习化 | Learning Determinantal Point Processes（Kulesza & Taskar，正文 `cite{kulesza2011learning}`，UAI）；Determinantal Point Processes for Machine Learning（正文 `cite{kulesza2012determinantal}`） | 2011 / 2012 |
| DPP 快速贪心 MAP | Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity（Chen 等，正文 `cite{chen2018fast}`） | NeurIPS 2018 |
| DPP 近似贪心 | 正文 `cite{gillenwater2012near}` | — |
| SSD | Sliding Spectrum Decomposition for Diversified Recommendation（正文 `cite{huang2021sliding}`，小红书） | KDD 2021 |
| Beam Search 源头 | 正文 `cite{lowerre1976harpy}`（Lowerre, 1976，Harpy 语音识别）、`cite{rubin1977locus}`（IJCAI 1977） | 1976 / 1977 |
| 序列评估模型结构（正文未点名，本目录实现参考） | Learning a Deep Listwise Context Model for Ranking Refinement（DLCM，Ai 等） | SIGIR 2018 |
| 同上 | Personalized Re-ranking for Recommendation（PRM，Pei 等，阿里） | RecSys 2019 |

正文还提到 GRN（Generative Reranking Network）等一阶段生成式重排方向，本目录未实现对应脚本（正文仅作为趋势介绍，未给公式）。

---

## 实现约定

- **自包含**：每个 `.py` 文件独立可跑，文件之间不互相 import；共用的小工具（合成数据、ILD/熵/Kendall τ 计算）在各脚本内各写一份，便于单文件阅读。
- **docstring 对齐正文**：每个脚本顶部中文 docstring 写明对应节次、实现内容与关键公式；与正文有差异之处（如 DLCM/PRM 结构、DPP θ=1 时 α=0 的退化行为）在 docstring 与本 README 中都显式注明。
- **固定随机种子**：`if __name__ == "__main__":` 块统一 `random.seed(42)`，用到 numpy 加 `np.random.seed(42)`，用到 torch 加 `torch.manual_seed(42)`，输出可复现。
- **纯合成数据**：不联网、不读取本地数据集、不画图；可视化一律以数值表格形式打印。
- **mock 组件**：教学用的假存储 / RPC / 日志流 / 兜底池等，在 docstring 与运行输出里统一标 `[MOCK]`。
- **CPU 友好**：全部脚本单次运行 ≤ 20s。

---

## 运行结果（实测）

```
./ch01_why_rerank/pointwise_vs_listwise.py                 rc=0 t=0s
./ch01_why_rerank/position_bias.py                         rc=0 t=0s
./ch02_rerank_architecture/rerank_pipeline.py              rc=0 t=0s
./ch03_list_generation/beam_search_generation.py           rc=0 t=1s
./ch03_list_generation/greedy_fusion_gaussian_sampling.py  rc=0 t=1s
./ch04_list_evaluation/list_selection_strategies.py        rc=0 t=1s
./ch04_list_evaluation/list_value_definitions.py           rc=0 t=4s
./ch04_list_evaluation/listwise_evaluator_model.py         rc=0 t=14s
./ch05_diversity/diversity_algorithms_comparison.py        rc=0 t=1s
./ch05_diversity/dpp_greedy_map.py                         rc=0 t=1s
./ch05_diversity/mmr_diversification.py                    rc=0 t=0s
./ch05_diversity/ssd_sliding_spectrum.py                   rc=0 t=0s
./ch06_post_processing/quota_injection_and_data_closure.py rc=0 t=0s
./ch06_post_processing/rule_based_shuffle_and_repair.py    rc=0 t=1s
```
