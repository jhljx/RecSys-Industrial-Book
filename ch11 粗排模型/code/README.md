# ch11 粗排模型 — 代码示例

本目录对应教材第 11 章，覆盖粗排阶段主流模型的 Python 实现，共 **16 个可运行脚本**。目录编号与正文节次一一对应，均可直接 `python3 <文件>.py` 运行。除纯统计的 `delayed_feedback.py` 外，其余脚本均为神经网络模型，依赖 **PyTorch**（`pip install torch`）。

---

## 目录结构

```
code/
├── ch01_two_tower/                    § 11.1 双塔模型
│   ├── dssm.py                  DSSM：双塔 + In-Batch Sampled Softmax
│   ├── tdssm.py                 TDSSM：静态 MLP（长期）+ LSTM（短期）
│   ├── dssm_transformer.py      DSSM+Transformer：序列编码增强用户塔
│   └── dat.py                   DAT：双塔增强向量 + 自适应模仿 + 类别对齐
├── ch02_multi_tower/                  § 11.2 多塔模型
│   ├── mv_dssm.py               MV-DSSM：共享用户塔 + 多域独立物品塔
│   ├── dssm_fm.py               DSSM+FM：双塔内积 + FM 二阶交叉
│   ├── delayed_feedback.py      延迟反馈：假负样本纠偏 + 回填窗口
│   └── cross_feature_tower.py   带交叉特征的多塔：双塔 + 轻量交叉塔
├── ch03_distillation/                 § 11.3 粗排蒸馏模型
│   ├── teacher_student_rocket.py  Teacher-Student（KL 蒸馏）+ Rocket Launching
│   └── pfd.py                   PFD：特权特征蒸馏（Privileged Feature Distillation）
├── ch04_prerank_dnn/                  § 11.4 粗排 DNN 模型
│   ├── two_tower_dnn.py         双塔DNN：交叉特征 + 顶部轻量 DNN
│   ├── cold.py                  COLD：SE 特征组选择 + 算力代价感知
│   ├── fscd.py                  FSCD：复杂度感知的变分 Dropout 特征筛选
│   └── inttower.py              IntTower：Light-SE + FE-Block + CIR
└── ch05_multi_interest/               § 11.5 粗排模型多兴趣建模
    ├── mvke.py                  MVKE：虚拟核专家混合 + VKG 门控
    └── hit.py                   HIT：粗粒度生成 + 细粒度匹配
```

---

## 快速运行

```bash
# 双塔模型（§11.1）
python3 ch01_two_tower/dssm.py
python3 ch01_two_tower/tdssm.py
python3 ch01_two_tower/dssm_transformer.py
python3 ch01_two_tower/dat.py

# 多塔模型（§11.2）
python3 ch02_multi_tower/mv_dssm.py
python3 ch02_multi_tower/dssm_fm.py
python3 ch02_multi_tower/delayed_feedback.py
python3 ch02_multi_tower/cross_feature_tower.py

# 粗排蒸馏（§11.3）
python3 ch03_distillation/teacher_student_rocket.py
python3 ch03_distillation/pfd.py

# 粗排 DNN（§11.4）
python3 ch04_prerank_dnn/two_tower_dnn.py
python3 ch04_prerank_dnn/cold.py
python3 ch04_prerank_dnn/fscd.py
python3 ch04_prerank_dnn/inttower.py

# 多兴趣建模（§11.5）
python3 ch05_multi_interest/mvke.py
python3 ch05_multi_interest/hit.py
```

批量跑全部脚本：

```bash
find . -name "*.py" | sort | while read f; do echo "=== $f ==="; python3 "$f" >/dev/null && echo OK || echo FAIL; done
```

每个脚本的 `__main__` 块固定 `random.seed(42)`，用合成数据完整演示训练 + 粗排推理流程。所有模型的推理接口统一为 `pre_rank(uid, candidates, top_k)`，返回 `[(item_id, score), ...]`。

---

## 各模型要点

### § 11.1 双塔模型

- **`dssm.py`**：用户塔和物品塔各为独立 MLP，In-Batch Sampled Softmax 训练，温度系数 $$\tau$$ 控制分布锐度，匹配函数为余弦相似度。线上离线预计算物品塔 Embedding，只跑用户塔 + 内积排序。
- **`tdssm.py`**：长期兴趣用静态 Embedding 过 MLP，短期兴趣用 `nn.GRU` 编码近期行为序列，两者 element-wise 相加融合。
- **`dssm_transformer.py`**：用单层 Transformer Encoder（自注意力 + Add&Norm + FFN）替换 LSTM，对序列做 mean pooling 后与静态兴趣融合。物品塔不依赖序列，仍可独立离线预计算。
- **`dat.py`**：完整覆盖正文四个子节——(a) 双塔增强结构：给每个用户/物品配可学习增强向量 $$a_u$$ / $$a_v$$ 拼入对侧塔输入；(b) 自适应模仿机制 $$\mathcal{L}_u=\sum w\|a_u-\text{sg}(z_v)\|^2$$（用值拷贝实现 stop-gradient，只在正样本上拟合）；(c) 类别对齐损失，对齐不同类别物品表示的二阶统计量；(d) 三项损失联合训练。实测 $$\mathcal{L}_{\text{mimic}}$$ 从 1.0915 降到 0.8296。

### § 11.2 多塔模型

- **`mv_dssm.py`**：一个共享用户塔 + news/app/video 三个独立物品塔（物品 Embedding 维度分别 16/12/20，刻意异构）。视频域只给 10 条样本以体现跨域知识迁移。演示块用同一个 $$e_u$$（只算一次）在三个域分别出 Top-3。
- **`dssm_fm.py`**：FM 二阶交叉用「和平方减平方和」技巧实现，复杂度 $$O(kn)$$；FM 隐向量与双塔 Embedding 共享参数。同时打印 $$|s_{\text{DSSM}}|$$ 与 $$|s_{\text{FM}}|$$ 两分支量级，并对比 `with_fm=True/False`。
- **`delayed_feedback.py`**：合成 87 条曝光、真实转化 23 条，实时流只观测到 3 条（20 条假负样本）。真实 CVR 0.2644 vs 朴素观测 0.0345；指数分布 MLE 估出延迟 rate=0.3527（期望 2.835 天）。**纠偏效果**：不纠偏平均预估 0.0398（偏差 −0.2245），重要性加权纠偏后 0.0676（偏差 −0.1967）。离线流用 $$W=3$$ 天窗口回填出 14/87 正样本。
- **`cross_feature_tower.py`**：双塔 + 轻量交叉塔三塔结构，交叉塔只吃交叉特征并输出偏置分。演示两阶段打分的精度/开销权衡：Top-N = 5/10/20 时命中全量 Top-5 的比例为 1/5、2/5、4/5，交叉塔调用次数相应为 5/10/20（候选集 30）。

### § 11.3 粗排蒸馏模型

- **`teacher_student_rocket.py`**：
  - Teacher-Student：教师输出温度软化的 soft label，学生用 KL 散度对齐 + BPR 硬标签，$$\mathcal{L}=(1-\alpha)\mathcal{L}_{\text{BPR}}+\alpha\,\text{KL}(p_T\|p_S)$$。
  - Rocket Launching：共享 Booster 层 + Heavy Head（深）/ Light Head（浅），训练时让 Light Net 对齐 Heavy Net 中间表示（hint loss），推理只走 Light Net。
- **`pfd.py`**：PFD 全称 **Privileged Feature Distillation**（阿里 KDD 2020）。思想源自 LUPI，但关键改进在于**教师同时使用在线特征与特权特征** $$f_T(\mathbf{x}, \mathbf{x}_{\text{privileged}})$$，而非传统 LUPI 只用特权特征（后者的教师未必比学生强，甚至可能误导）。学生仍只用在线可得特征 $$f_S(\mathbf{x})$$。目标函数按正文三项：
  $$\mathcal{L}=(1-\lambda)\mathcal{L}_S(y,f_S(\mathbf{x}))+\lambda\mathcal{L}_{KD}(f_T,f_S)+\mathcal{L}_T(y,f_T)$$
  配套实现正文强调的两项工程机制：**同步训练 + Warm-up**（初期只优化各自监督损失，教师收敛后 $$\lambda$$ 渐进上升到 0.80，避免早期不稳定监督信号）和**参数共享**（教师与学生共享 Embedding 层，仅高层预测分支独立，大幅降低训练开销）。特权特征按正文举例设为详情页停留时长、是否查看评论、是否咨询客服等点击后才可得的后验行为。实测 AUC：教师 0.9968 / PFD 学生 0.8405 / 无蒸馏学生 0.8350。
  > 注：代码额外实现了一个中间层 Hint 对齐（投影层 + MSE），这不是正文内容，可通过 `beta_hint=0.0` 关闭。

### § 11.4 粗排 DNN 模型

- **`two_tower_dnn.py`**：在双塔基础上引入用户-物品交叉特征（类目历史 CTR 的 Beta 平滑、最偏好类目匹配、SIM 检索相似度，均用 leave-one-out 统计 + z-score 归一化），拼接后送入顶部轻量 DNN：$$h=[h_u,h_i,h_{\text{cross}}]$$，$$\hat{y}=\sigma(W_2\,\text{ReLU}(W_1h+b_1)+b_2)$$。纯内积头 AUC 0.7886 vs 双塔DNN 头 0.8289。docstring 讨论了打分函数不再可分解为内积、因而破坏 ANN 检索前提，只适用于候选已收敛到千级的粗排阶段。两阶段演示中昂贵 DNN 打分次数下降 66.7%。
- **`cold.py`**：SE 特征组选择 + 全连接 DNN（COLD 的关键是突破双塔必须内积匹配的限制）。含算力代价表和预算约束下的贪心特征选择（收益密度 = 重要性/代价）。预算 2.0/5.0/14.5 对应 AUC 0.6968/0.7779/0.8882，完整呈现算力-效果曲线。
- **`fscd.py`**：FSCD = **Feature Selection method based on feature Complexity and variational Dropout**（阿里 SIGIR 2021）。它不重新设计网络结构，而是直接继承精排的交互式网络、只通过减少输入特征来降低在线开销。按正文公式链完整实现：
  - 特征复杂度 $$c_j=\gamma_1 o_j+\gamma_2 e_j+\gamma_3 n_j$$（$$o_j$$ 在线计算复杂度、$$e_j$$ Embedding 维度、$$n_j$$ 取值数量）
  - 先验保留概率 $$\theta_j=1-\sigma(c_j)$$，复杂度越高越倾向被移除
  - 伯努利变量 $$z_j\sim\text{Bern}(\theta_j)$$，正则权重 $$\alpha_j=\log(1-\theta_j)-\log\theta_j$$
  - 统一目标 $$\mathcal{L}=-\frac{1}{N}\log P(\mathcal{D}|\mathbf{v},\mathbf{w},\mathbf{z})+\frac{\lambda}{N}(\|\mathbf{v}\|^2+\|\mathbf{w}\|^2)+\frac{1}{N}\sum_j\alpha_j z_j$$
  - 伯努利不可导 → **Gumbel-Sigmoid 连续松弛**，收敛后按 $$\delta_j$$ 取 Top-K
  - **两阶段**：stage1 联合学 $$\delta_j$$ 与 DNN，stage2 移除未选中特征、以 stage1 参数初始化重新微调（目标退化为纯 CTR 损失）

  实测：全特征 AUC 0.8884@cost 9.8；FSCD 选 Top-4 后 0.7861@cost 1.8，微调后回升到 0.8970，在线复杂度仅 18%。对照组「按高复杂度选 Top-4」只有 0.7684@cost 8.0，印证了把计算成本纳入选择目标的价值。
- **`inttower.py`**：完整实现正文四个子模块。
  - (a) **Light-SE**：轻量通道注意力，低成本筛选重要特征（演示中权重和归一化为 1.000）
  - (b) **FE-Block**：从用户塔**多个中间层**提取表示，与物品塔最后一层做细粒度交互（2 层 × 4 头 × $$d_h$$=8），而非等到最终 Embedding 才交互；用无额外参数的 Max Similarity 实现早期交互
  - (c) **CIR** = **Contrastive Interaction Regularization**（对比交互正则化），从**隐式**角度补充 FE-Block 的显式交互。在两塔最后一层多头表示 $$M_u^L$$ / $$M_v^L$$ 上做 InfoNCE：
    $$L_{\text{cir}}=-\frac{1}{Q}\sum_{(u,v)\in Q}\log\frac{\exp(\text{sim}(M_u^L,M_v^L)/\tau)}{\sum_{(u',v')\in N}\exp(\text{sim}(M_{u'}^L,M_{v'}^L)/\tau)}$$
  - (d) 联合优化目标 $$L=L_{\text{ctr}}+\lambda_1 L_{\text{cir}}+\lambda_2\|\Theta\|_2^2$$

  演示打印完整维度流转 16→Light-SE→24→16，$$M_u^L/M_v^L$$=32/32，AUC 0.8968。正文强调 CIR 是 FE-Block 的**补充而非替代**——两者分别从显式交互和隐式交互两个角度增强双塔表达能力。
> **关于 RankTower**：§11.4 引言的模型清单中还提到了 RankTower，但 ch11.tex 没有对应小节正文，本章小结也只列了上述 4 个模型。该模型的占位小节在 `chapters/ch22.tex` §22.1.4（第 22 章 链路一致性建模）下——它的落脚点是跨阶段排序一致性而非单点的效果-算力权衡，因此代码放在 `ch22 链路一致性建模/code/ch01_prerank_consistency/ranktower.py`，不在本目录。

### § 11.5 粗排模型多兴趣建模

- **`mvke.py`**：MVKE 全称 **Mixture of Virtual-Kernel Experts**（腾讯 KDD 2022）。$$K$$ 个虚拟核专家分别建模不同兴趣维度，VKG 门控根据候选目标动态聚合。演示打印每个 VKE 的字段注意力差异（用户 0 的 VKE0 字段2 权重 0.341 vs VKE1 均匀 0.250）和逐目标物品的门控分布（item2=[0.506,0.346,0.148]），24 个目标上门控极差均值 0.115，证明不同视角确实被区分开。AUC 0.8109。
- **`hit.py`**：HIT = **Hierarchical Interaction-Enhanced Two-Tower**（腾讯 CIKM 2025）。按正文实现两个子模块：
  - **粗粒度生成模块**（受 DAT 启发）：Generator 网络 $$\mathbf{r}_m^k=g_k(\mathbf{x}_b^\delta)$$ 只吃**静态特征**（避免实时行为特征推高线上开销），生成模拟对侧信息的高层语义表示；广告场景下分设正/负样本 Generator。生成表示先做 L2 归一化，再与原始 Embedding 拼接成增强输入 $$\mathbf{e}'=f(\mathbf{r}_m^1,\cdots,\mathbf{r}_m^K,\mathbf{e})$$ 过 DNN 得 $$\mathbf{h}_u$$ / $$\mathbf{h}_a$$
  - **细粒度匹配模块**：Multi-head Representer 把两侧投到 $$J$$ 个子空间（$$\mathbf{r}_u^{(j)}=\mathbf{W}_u^{(j)}\mathbf{h}_u+\mathbf{b}_u^{(j)}$$），再用 IntTower 式 Max Similarity 聚合：
    $$\hat{y}=\sum_{j_u=1}^{J}\max_{j_a}\{(\mathbf{r}_u^{(j_u)})^\top \mathbf{r}_a^{(j_a)}\}$$
  - 优化目标 $$L=L_c+\alpha(L_{g_u}+L_{g_a})$$，生成损失 $$L_g=-\frac{1}{N}\sum_i[y_i\text{Dist}(\mathbf{r}_u,\mathbf{r}_m^1)+(1-y_i)\text{Dist}(\mathbf{r}_u,\mathbf{r}_m^2)]$$（Dist 取余弦；保留正文的前置负号，等价于最大化相似度）

  实测生成表示对齐度 $$\cos(r_u,r_m^1)$$ 从 0.1244 升到 0.6764（Generator 确实学到了对侧语义），AUC 0.8740。物品侧的多头表示与生成表示均不依赖用户，可离线预计算，线上只需 $$J\times J$$ 次低维内积取 max——这是它区别于 COLD / FSCD 全交互模型的关键。

---

## 实现约定

- **PyTorch 实现**：所有神经网络模型（双塔、多塔、蒸馏、粗排 DNN、多兴趣）均以 `nn.Module` + autograd 实现，`python3 <file>.py` 即可运行；文件之间不互相 import，每个文件自包含。唯一的非神经网络文件 `delayed_feedback.py` 为纯标准库实现。
- **统一推理接口**：`pre_rank(uid, candidates, top_k)` → `[(item_id, score), ...]`，接收候选集并返回 Top-K 排序结果，契合粗排「候选集已由召回缩小」的语义。
- **自动微分**：训练入口统一为 `fit()` / `train_model()`（避开 `nn.Module.train(mode)` 同名方法），梯度由 `loss.backward()` 自动求得，多层链式法则、注意力权重、Embedding 等所有路径均得到精确梯度，无需任何近似。CPU 上即可运行，不依赖 GPU。
- **合成数据**：每个 `__main__` 块固定 `random.seed(42)`，输出可复现。
- **小规模**：典型配置为 8 用户 / 20~30 物品 / emb_dim=16 / 5~6 epochs，全部脚本单文件运行均在 4 秒内完成。
- **以正文为准**：若通用论文印象与本书正文口径冲突，实现一律以 `chapters/ch11.tex` 的论文讲解为准。已核对过的几处容易混淆的命名：

  | 模型 | 正文口径（已据此实现） | 容易误认成 |
  |------|----------------------|-----------|
  | PFD | Privileged Feature Distillation，教师同时用在线特征 + 特权特征 | 渐进融合蒸馏（Progressive Fused） |
  | FSCD | feature Complexity + variational Dropout 的可学习特征选择 | 交叉特征蒸馏（Cross-feature Distillation） |
  | HIT | Hierarchical Interaction-Enhanced Two-Tower，Generator + 多头 Max Similarity | 层次兴趣塔（Hierarchical Interest Tower） |
  | IntTower 的 CIR | Contrastive Interaction Regularization（对比学习正则） | 跨塔交互模块（Cross-Interaction） |
  | COLD | 突破内积限制、可直接上全连接 DNN + SE 特征组选择 | 仅做 SE 门控的轻量双塔 |
