# ch05 粗排模块 — 代码示例

本目录对应教材第 5 章，覆盖粗排模块的完整链路与核心策略算法，共 **15 个可运行脚本**。目录编号与正文节次一一对应，均可直接 `python3 <文件>.py` 运行。除 `ch02_prerank_model/two_tower_multi_task.py` 与 `ch07_candidate_set_optimization/pointwise_listwise_setwise.py` 使用 **PyTorch** 外，其余脚本只依赖标准库与 **NumPy**。

---

## 目录结构

```
code/
├── ch01_why_prerank/                        § 5.1 为什么需要粗排
│   └── prerank_pipeline.py                  粗排完整链路：召回→打分→融合→保量→Quota→截断→精排
├── ch02_prerank_model/                      § 5.2 粗排模型
│   └── two_tower_multi_task.py              轻量双塔多任务：离线预计算物品塔 + 在线点积 + Recall@K 评价
├── ch03_pipeline_consistency/               § 5.3 链路一致性模型
│   └── consistency_metrics_and_funnel.py    一致性度量（ρ/τ/Hit@K）、漏斗损失、一致性强度 α 扫描
├── ch04_multi_objective_fusion/             § 5.4 粗排多目标融合
│   ├── value_based_fusion.py                § 5.4.1 基于值：线性融合 vs 乘法融合
│   ├── rank_based_fusion.py                 § 5.4.2 基于序关系：五种 Rank 位置衰减变换
│   ├── hybrid_value_rank_fusion.py          § 5.4.3 基于值与序关系：s=f(pxtr)·g(Rank)
│   └── bayesian_optimization_auto_fusion.py § 5.4.4 自动化融合：自实现 GP + EI/UCB/PI/TPE
├── ch05_fusion_mechanism_analysis/          § 5.5 多目标融合机制分析
│   ├── additive_vs_multiplicative.py        加法 vs 乘法：等值线、短板效应、对数空间、Top-K 结构
│   ├── generalized_product_taylor.py        广义乘法公式 ∏(β+αp) 的乘积展开与 β 的退化分析
│   └── gamma_traffic_sharpness.py           幂指数 γ 与流量陡峭程度（Gini / 熵 / HHI）
├── ch06_quota_and_truncation/               § 5.6 粗排分 Quota 调控与截断
│   ├── quota_control.py                     § 5.6.1 固定 / 动态 Quota + PI 闭环调控仿真
│   ├── guarantee_mechanism.py               § 5.6.2 保量：分数补偿 Δ vs 预留位置
│   └── truncation_funnel_tradeoff.py        § 5.6.3 截断长度 K 对下游漏斗与算力的影响
└── ch07_candidate_set_optimization/         § 5.7 粗排候选集集合优化
    ├── set_level_candidate_selection.py     集合价值函数 F(S) 的贪心优化 vs 逐点 TopK
    └── pointwise_listwise_setwise.py        Point-wise → List-wise → Set-wise 建模演进对比
```

---

## 快速运行

```bash
# §5.1 为什么需要粗排
python3 ch01_why_prerank/prerank_pipeline.py

# §5.2 粗排模型
python3 ch02_prerank_model/two_tower_multi_task.py

# §5.3 链路一致性
python3 ch03_pipeline_consistency/consistency_metrics_and_funnel.py

# §5.4 多目标融合
python3 ch04_multi_objective_fusion/value_based_fusion.py
python3 ch04_multi_objective_fusion/rank_based_fusion.py
python3 ch04_multi_objective_fusion/hybrid_value_rank_fusion.py
python3 ch04_multi_objective_fusion/bayesian_optimization_auto_fusion.py

# §5.5 融合机制分析
python3 ch05_fusion_mechanism_analysis/additive_vs_multiplicative.py
python3 ch05_fusion_mechanism_analysis/generalized_product_taylor.py
python3 ch05_fusion_mechanism_analysis/gamma_traffic_sharpness.py

# §5.6 Quota 调控与截断
python3 ch06_quota_and_truncation/quota_control.py
python3 ch06_quota_and_truncation/guarantee_mechanism.py
python3 ch06_quota_and_truncation/truncation_funnel_tradeoff.py

# §5.7 候选集集合优化
python3 ch07_candidate_set_optimization/set_level_candidate_selection.py
python3 ch07_candidate_set_optimization/pointwise_listwise_setwise.py
```

批量跑全部脚本：

```bash
find . -name "*.py" | sort | while read f; do echo "=== $f ==="; python3 "$f" >/dev/null && echo OK || echo FAIL; done
```

每个脚本自包含（文件之间不互相 import），`__main__` 块固定 `random.seed(42)`（用到 NumPy / PyTorch 时再加 `np.random.seed(42)` / `torch.manual_seed(42)`），跑合成数据并打印数值表格。

---

## 各脚本要点

> 下列数字均为在本机实际运行脚本后抄录的输出。

### § 5.1 为什么需要粗排

- **`prerank_pipeline.py`**：把正文图「粗排模块内部处理流程」的 6 个阶段串成一条进程内链路（`[MOCK]` 召回源 / 精排开销模型）。5 路召回 5100 条 → 去重后 3434 条（去重率 32.7%）→ 3 个 pxtr 打分 → 线性融合 `s=0.5p_ctr+0.3p_cvr+0.2p_ltr` → 161 条欠投新内容 `Score+Δ(0.01)` → Quota（game 120 / entertainment 90 / ecommerce 60 / new_anchor 30）→ 截断 K=300。全链路 **8.759 ms，占 50 ms 预算的 17.5%**。对照实验：纯 TopK 的 `category_coverage=1`、`max_category_share=1.0000`、平均分 0.1162；加 Quota+保量后 `category_coverage=3`、`max_category_share=0.5000`、平均分 0.0991。末尾给出截断-算力表：K=300 精排开销 54 ms，而 K=5000（不截断）需 900 ms，正是粗排存在的根本原因。

### § 5.2 粗排模型

- **`two_tower_multi_task.py`**：轻量双塔 + 3 个任务头（ctr/cvr/ltr，损失 $$L=\sum_t \lambda_t \mathrm{BCE}_t$$，$$\lambda=(1.0,1.5,0.8)$$）。参数量仅 **2622**，400 step 训练 3.60 s，loss 1.8616 → 1.1175；离线 AUC 分别为 **ctr 0.8965 / cvr 0.8512 / ltr 0.8647**。解耦优势实测：3000 条候选下「在线重算物品塔」1.832 ms/请求 vs「离线缓存 + 在线点积」1.328 ms/请求（**加速 1.38×**），两路打分最大数值差 0.00e+00（物品塔与用户/任务无关）。粗排评价用 Recall@K：N=5000、Relevant=Top50 时 **Recall@300 = 0.9000**（候选压到 6%），对应正文「不是看得最准，而是淘汰得足够准」。

### § 5.3 链路一致性模型

- **`consistency_metrics_and_funnel.py`**：模块层的一致性度量（模型层实现在 ch22）。N=1000、40 维特征（共享 [0,8)、精排独有 [8,32)、粗排独有 [32,40)）下，Spearman ρ=+0.2908、Kendall τ=+0.1920、Hit@10=0.1000、Hit@100=0.2500。漏斗分析：K=50 时漏斗损失 0.750（75% 的精排 Top20 被永久丢弃）、收益保留 0.742；K=500 时损失降到 0.250、收益保留 0.964。一致性强度 α 扫描是本脚本最关键的结论——α 从 0 升到 1.0 时 ρ 单调升到 1.0000、Hit@60 升到 1.0000，但**曝光真实收益在 α=0.3 见顶 13.4729，α=1.0 回落到 12.8847（相对峰值 −4.37%）**，数值验证了正文「一致性是系统内部指标、与业务指标无严格因果关系」以及 α=1 时链路退化为「精排复读机」。

### § 5.4 粗排多目标融合

- **`value_based_fusion.py`**（§5.4.1）：先复刻正文三点算例（A(0.9,0.1)、B(0.5,0.5)、C(0.1,0.9)：加法全为 1.00，乘法 B=0.2500 > A=C=0.0900）。再用 N=500 候选暴露线性融合的量纲敏感性——各目标量级相差悬殊（p_evtr 均值 0.5262 是 p_cvr 均值 0.0244 的 **21.6×**）。乘性漂移实验：p_cvr 整体 ×3 时 β=0 的乘法融合排序完全不变（ρ=1），线性融合的排序显著改变，对应正文「原有权重失效，从而增加调参成本」。短板效应：加法下 (0.9,0.1) 与 (0.99,0.01) 同在 c=1.0 等值线，乘法下 0.0900 vs 0.0099，**相差 9.1 倍**。
- **`rank_based_fusion.py`**（§5.4.2）：五种位置衰减变换的数值表（N=400）：`1/Rank` 从 R=1 的 1.0000 掉到 R=10 的 0.1000（头部极陡）；幂函数 `1/(R^0.7+2)`、归一化幂 `1-(R/N)^1.5`（R=1 时 0.9999、R=100 时 0.8750）、指数 `2sinh(kt)`、平滑衰减 `1-(R/(N+b))^1.5` 依次更平缓。信息损失实验给出两组三路 Rank 完全相同、数值分布迥异的候选集（「紧凑型」与「长尾型（头部断层）」），**两组的 1/Rank 融合分逐项相同（True）**，而值融合分之差为 `[0.0, 0.1737, 0.3119, 0.3492, 0.3315]`——这正是正文说的「无法感知真实分数差异」。
- **`hybrid_value_rank_fusion.py`**（§5.4.3）：实现 $$s_{agg}=f(pxtr)\cdot g(Rank)$$ 的四种组合（$$f\in\{\log(1+p),\,(\beta+\alpha p)^\gamma\}$$ × $$g\in\{1/Rank^\lambda,\,1-(Rank/N)^\lambda\}$$）。N=400、Top-50、λ=0.5 时，纯值融合与纯 Rank 融合彼此 ρ 只有 **+0.8961**，而值序融合对两端同时保持高相关（如 `(β+αp)^γ × 1/Rank^λ`：ρ(vs 值)=+0.9483、ρ(vs Rank)=+0.9849），确实落在两种机制中间。λ 扫描显示头尾增益比 g(1)/g(N) 从 λ=0.1 的 1.8 涨到 λ=2.0 的 1.6e5，排序逐渐被 Rank 项接管。鲁棒性实验（×3 乘性漂移 / 平方压缩 / sigmoid 重标定三种保序变换）：纯 Rank 融合恒为 `+1.0000 / 1.00`，纯值融合最差（×3 漂移下 `+0.9642 / 0.75`），值序融合居中（`+0.9934 / 0.89`）。
- **`bayesian_optimization_auto_fusion.py`**（§5.4.4）：自己实现高斯过程（RBF 核 + Cholesky 求解）与 EI / UCB / PI 采集函数，另加 TPE 与随机、网格搜索作对照，黑盒目标 `Target(w)=AUC_click+AUC_follow`，搜索空间 $$w\in[0,1]^3$$，预算 48 次评估。手调基线 (0.5,0.3,0.2) 的 Target = **1.3685**；48 次评估后 GP+UCB(κ=2) 达 **1.4126**（+0.0441）、GP+EI/PI 1.4118、TPE 1.4123、随机搜索 1.4113、网格搜索(4 档/维) 1.4121。追上「随机搜索 48 次终点水平」所需评估次数：**GP+EI/UCB 21 次、GP+PI 24、网格 26、TPE 34、随机 43**。由于 AUC 只依赖排序、Target(w) 对 w 整体缩放不变，表里同时给出归一化方向，各策略一致收敛到约 **(0.24, 0.51, 0.25)**。GP 代理模型质量（48 观测点、300 留出点）：后验均值 MAE 0.01328、RMSE 0.02260、平均 σ_n 0.00491、Pearson 相关 0.7898。

### § 5.5 多目标融合机制分析

- **`additive_vs_multiplicative.py`**：等值线分析显示 c=1.0 时乘法只有 p_ctr=p_cvr=1.0 这一孤点可行（p_ctr=0.9 需要 p_cvr=1.111 > 1）。沿加法等值线 p_ctr+p_cvr=1.0 扫描，加法分恒为 1.0000，乘法分从 (0.5,0.5) 的 0.250000 单调掉到 (0.999,0.001) 的 0.000999（峰值的 0.0040）。对数空间表：p 从 0.9 到 0.1 同为 Δp=0.2 的间隔，Δlog p 从 0.2513 放大到 1.0986（**差异放大 5.49 倍**），p=0.01 时放大 40.24 倍。N=600 候选的 Top-K 结构对比：K=100 时加法 Top-K 里偏科型占 **0.460**、平均 min/max 均衡度 0.6189，乘法只有 **0.010**、均衡度 0.9175，两者 Top-100 重叠仅 0.550。
- **`generalized_product_taylor.py`**：复刻正文 $$\prod_i(\beta_i+\alpha_i p_i)$$ 的乘积展开。β=0 时 (0.9,0.1)=0.0900 vs (0.5,0.5)=0.2500；**β=10 时为 110.09 vs 110.25**（与正文算例一致），相对差降到 1.45e-3。各阶项占比（n=3, p=(0.8,0.5,0.2)）：β=0.1 时高阶（二阶+三阶）占比 **0.901235**，β=10 时降到 **0.005775**，β=50 时 0.000257，数值验证「$$\beta_i\to\infty$$ 时高阶交互项消失」；各阶求和 3.240000 = 精确 Score 3.240000（展开是恒等式）。一阶截断 $$Score \approx Const+\sum_i w_i p_i,\ w_i=Const\cdot\alpha_i/\beta_i$$ 与精确乘法的一致性：β=0.2 时平均相对误差 0.653058、Spearman 0.9323；β=100 时误差 5.0e-5、Spearman 1.0000、**Top-50 Jaccard = 1.0000**（完全退化为线性加法）。梯度视角：$$\partial Score/\partial p_1 = \beta_2+\alpha_2p_2$$，p_2 从 0.0 到 1.0 时解析值 1.0→2.0，数值差分完全吻合，而加法恒为 w_1=1.0。
- **`gamma_traffic_sharpness.py`**：正文算例复刻——γ=2 时 0.8²=0.6400 vs 0.2²=0.0400（比值 16.0），γ=0.5 时 0.8944 vs 0.4472（比值 2.0）。N=2000 合成候选的流量分布：γ 从 0.25 升到 8.0，Top1% 流量占比 0.0131 → **0.3636**，Gini 0.0896 → 0.9193，归一化熵 0.9983 → 0.6930，HHI×1e3 从 0.5126 → 10.0979，单调验证「γ 控制流量陡峭程度」。排序影响区分两种情形：**单目标** $$(\beta+\alpha p)^\gamma$$ 是严格单调变换，γ=0.5/2.0/8.0 时 Spearman 恒为 1.0000（只改尺度不改序）；**多目标**时改某一个 γ_i 才真正改序（γ=(4.0,1.0) 时 Spearman 0.9461、Top100 Jaccard 0.5152）。γ_i 的权重含义：γ=(6,1) 时 Top-100 平均 p_ctr 0.6620 / p_cvr 0.4644（比值 1.4254），γ=(1,6) 时 0.5122 / 0.5985（比值 0.8558）。

### § 5.6 粗排分 Quota 调控与截断

- **`quota_control.py`**（§5.6.1）：先复刻正文表「无 Quota 约束时 Top5 全部来自游戏池」（0.98/0.96/0.95/0.94/0.93）。再复刻正文 `code_prerank_quota`：召回 5000 → 打分 → 0.5/0.3/0.2 融合 → 固定配额 {game:40, entertainment:30, ecommerce:20, new:10}。无约束 TopK 的 game 占比 0.960、池熵 0.1374、新主播 3 个；固定 Quota 后占比精确落在 0.400/0.300/0.200/0.100，池熵升到 0.9232、新主播 10 个，**代价是平均融合分 0.09234 → 0.08283（相对损失 10.29%）**。动态 Quota（$$Quota_i=f(User,Context,BusinessGoal)$$，凸组合 + 最大余数法取整 + 生态保底 floor）：重度游戏用户 P(Game|U)=0.9 时游戏配额 40 → **66**、电商 20 → 16；电商意图用户 ecommerce 20 → **56**；冷启新用户近似均匀 27/27/27/19（池熵 0.9926）。闭环 PI 控制（`[MOCK]` 下游通过率 game 0.95 / entertainment 0.8 / ecommerce 0.45 / new 0.35）：开环固定配额的曝光占比 MAE 均值 **0.0663** 始终不收敛，闭环**第 3 轮**即把 MAE 压到 ≤0.02，稳态配额为 game=28 / entertainment=26 / ecommerce=28 / new=18（控制律放大高损耗池的配额来补偿下游损失）。
- **`guarantee_mechanism.py`**（§5.6.2）：T=24 个时间片 × 每片 40 请求 × 每请求 600 候选、输出 Top100；保量对象占候选池 12% 但 score 分布 N(0.042,0.016) 低于普通内容 N(0.062,0.020)，全天目标 = 5% 总曝光。无保量时完成率仅 **0.128**。实现一（$$Exposure_i<Target_i \Rightarrow Score_i = Score_i+\Delta$$）扫描：Δ=0.005 → 0.292、Δ=0.010 → 0.566、Δ=0.020 → **1.054（超投 0.054，大盘分损失 0.66%）**、Δ=0.050 → 1.232（超投 0.232，损失 1.94%）。三种策略对比：自适应 Δ_t（按进度缺口，Δ_max=0.050）完成率 0.901、超投 0、**大盘分损失仅 0.38%**；实现二「Top100 预留 5 个位置」完成率钉死 **1.000**、超投 0、损失 0.49%，且进度曲线与理想匀速几乎重合（时间片 16 时 0.667 vs 理想 0.667）。
- **`truncation_funnel_tradeoff.py`**（§5.6.3）：召回 5000 → 粗排打分（噪声 σ=0.016）→ TopK → 精排 → Top10，200 次请求。K=50 时 Recall@K=0.5060、Top10 收益率 0.9648；**K=300 时 Recall 0.7795、收益率 0.9906，精排 FLOPs 只有不截断的 6.0%**；K=500 时 0.8370 / 0.9940（FLOPs 10.0%）；K=5000 延迟 25.5 ms **超出 25.0 ms 预算**。边际收益单调递减：50→100 区间每 1e9 FLOPs 换 0.9125 Recall，300→500 降到 0.1198，2000→5000 仅 0.0047。与 §5.3 的折算：σ=0.004 时 K=50 即可满足 Recall≥0.95，σ=0.016 需要 K=300，σ=0.032 需要 **K=2000**——一致性越差就要用越长的候选集兜底。自适应截断（用头部区分度 margin 作难度代理，margin 与 Recall 正相关：tau=0.008 组 margin 0.02240 / Recall 0.4140，tau=0.034 组 margin 0.04620 / Recall 0.9962）：平均 K=292.9（K 范围 120~540、FLOPs 0.98×）下把 Recall 从 0.7795 提到 **0.7825**、收益率 0.9906 → 0.9916。

### § 5.7 粗排候选集集合优化

- **`set_level_candidate_selection.py`**：优化目标从 $$P(click|u,i)$$ 换成 $$P(Value|CandidateSet)$$。`[MOCK]` 召回池 3000（篮球/足球/游戏/科技各 750，粗排分被篮球系统性高估 +0.030、科技低估 −0.008，Relevant 按正文兴趣分布 40%/30%/20%/10% 分配），候选集 K=300。五种策略横评：**A 逐点 TopK** Recall@K=0.7333、最大类目占比 0.8733、有效兴趣数 2/4（篮球 262 条、科技 0 条，即正文的 Candidate Concentration）；**B 兴趣配额** Recall=1.0000、TV 距离 0.0000；**C Count(Cat)≤120** Recall=0.9667；**D Sim(Ci,Cj)<0.8** Recall=0.7500；**E 集合贪心 F(S) + 10% 探索** Recall=1.0000、有效兴趣 4/4、对内相似度 0.4728 → **0.2477**、探索率 0.1933 → 0.2767。关键反直觉结论：贪心集合的平均粗排分比逐点 TopK 低 5.86%，但**集合真值均值反而从 0.06622 升到 0.07239**——粗排分高只是偏置带来的幻觉。ε 扫描（正文建议 5%~20%）：ε 从 0 到 0.20 探索率 0.2067 → 0.3633，代价是 Recall 1.0000 → 0.9833、TV 距离 0.0067 → 0.0833。λ_cov 敏感性：λ_cov 0 → 10 把 TV 距离从 0.0700 压到 0.0400，λ_cov ≥ 10 后饱和。
- **`pointwise_listwise_setwise.py`**：`[MOCK]` 标签带集合级耦合 $$p_i=\sigma(w\cdot x_i)/(1+\rho(m_c-1))$$（ρ=1.2，|L|=20），同簇条数 m_c=1 时平均真值 0.4001、m_c=7 时掉到 0.0551，直接证伪 Point-wise 的独立性假设。四种建模横评（400 个验证请求，Relevant=真值 Top5）：A Point-wise+BCE（AUC 0.7451 / NDCG@5 0.7865 / Top5 真实价值 0.9396）、B ListNet（0.7397 / 0.7582 / 0.9066）、C ListMLE（0.7412 / 0.7836 / 0.9432）、**D Set-wise Deep Sets（0.7436 / 0.7964 / 0.9519，唯一置换不变）**，Oracle 上界 1.1173。置换不变性数值验证：3 次随机排列下打分向量最大逐元素偏差 ≤ 4.77e-07、Top-5 集合完全一致。最有说服力的是「往集合里再塞同簇内容」实验：同簇条数从 8 加到 12 时 Point-wise 打分恒为 **−1.3261**（纹丝不动），Set-wise 从 −1.3819 单调降到 **−1.4160**，与真值折扣 0.1064 → 0.0704 同向。

---

## 论文出处

正文第 5 章引用、且本目录实现涉及的文献（书稿 `reference.bib` 已收录，标题/年份/出处照抄该文件）：

| 文献 key | 标题 | 年份 | 出处 | 对应脚本 |
|---|---|---|---|---|
| `hinton2015distilling` | Distilling the knowledge in a neural network | 2015 | arXiv preprint arXiv:1503.02531 | §5.3 一致性蒸馏思路（`consistency_metrics_and_funnel.py` 的 α 融合） |
| `burges2010ranknet` | From ranknet to lambdarank to lambdamart: An overview | 2010 | Learning | `rank_based_fusion.py` / `pointwise_listwise_setwise.py` |
| `taylor2008softrank` | Softrank: optimizing non-smooth rank metrics | 2008 | Proceedings of the 2008 international conference on web search and data mining | `rank_based_fusion.py` 的位置衰减 |
| `qin2010general` | A general approximation framework for direct optimization of information retrieval measures | 2010 | Information retrieval | `rank_based_fusion.py` 的位置衰减 |
| `cao2007learning` | Learning to rank: from pairwise approach to listwise approach | 2007 | Proceedings of the 24th international conference on Machine learning | `pointwise_listwise_setwise.py` 的 ListNet |
| `xia2008listwise` | Listwise approach to learning to rank: theory and algorithm | 2008 | — | `pointwise_listwise_setwise.py` 的 ListMLE |
| `zaheer2017deep` | Deep sets | 2017 | Advances in neural information processing systems | `pointwise_listwise_setwise.py` 的 Set-wise 打分器 |
| `lee2019set` | Set transformer: A framework for attention-based permutation-invariant neural networks | 2019 | International conference on machine learning | §5.7 置换不变建模 |
| `frazier2018tutorial` | A tutorial on Bayesian optimization | 2018 | arXiv preprint arXiv:1807.02811 | `bayesian_optimization_auto_fusion.py` 的 EI/UCB/PI |
| `williams1995gaussian` | Gaussian processes for regression | 1995 | Advances in neural information processing systems | 同上，GP 代理模型 |
| `rasmussen2003gaussian` | Gaussian processes in machine learning | 2003 | Summer school on machine learning | 同上 |
| `ebden2015gaussian` | Gaussian processes: A quick introduction | 2015 | arXiv preprint arXiv:1505.02965 | 同上 |
| `watanabe2023tree` | Tree-structured parzen estimator: Understanding its algorithm components and their roles for better empirical performance | 2023 | arXiv preprint arXiv:2304.11127 | `bayesian_optimization_auto_fusion.py` 的 TPE 对照 |

---

## 实现约定

- **纯 Python**：模块链路也用 Python 表达（进程内 mock，不起真实服务、不发真实 RPC）。所有教学性假组件（假存储、假下游通过率、假精排、假竞价/供给）均在 docstring 里以 `[MOCK]` 标注。
- **自包含**：脚本之间不互相 import，每个文件可单独运行；顶部 docstring 用中文写明对应节次、本实现做了什么以及关键公式。
- **可复现**：`__main__` 块固定 `random.seed(42)`，用到 NumPy / PyTorch 时同步固定 `np.random.seed(42)` / `torch.manual_seed(42)`。
- **离线无依赖**：只用 Python 3.12 标准库、NumPy 与 PyTorch，不联网下载数据或权重、不读取本地数据集、不弹绘图窗口（可视化一律改成数值表）。
- **小规模**：候选规模典型为 N=300~5000、训练 ≤400 step，CPU 上单文件运行最慢的 `bayesian_optimization_auto_fusion.py` 约 9 秒，其余均在数秒内结束。
- **以正文为准**：公式符号、参数取值、算例数字一律照抄 `chapters/ch05.tex`。两处需要说明的处理：
  1. §5.4.2 正文把指数衰减写作 $$2\sinh(kt)$$ 形式，脚本按 docstring 说明取负后展示，以保持「名次越靠前增益越大」的方向一致；
  2. §5.6.1 正文只给出固定配额与动态配额的概念式 $$Quota_i=f(User,Context,BusinessGoal)$$，脚本在此基础上补了一个最小 PI 闭环控制律作为 $$\pi(s)\to Quota$$ 的可运行形态，属正文概念的工程化具体化，非正文原文算法。
