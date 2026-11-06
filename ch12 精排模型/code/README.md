# ch12 精排模型 — 代码示例

本目录对应教材第 12 章，覆盖精排阶段主流模型的 Python 实现，共 **16 个可运行脚本**。目录编号与正文节次一一对应，均可直接 `python3 <文件>.py` 运行。全部脚本仅依赖 **PyTorch**（`pip install torch`）与 Python 标准库；`gbdt_lr.py` 的 GBDT 由文件内自行实现，不依赖 scikit-learn。

---

## 目录结构

```
code/
├── ch01_traditional_ranking/          § 12.3 传统精排模型
│   ├── lr.py                    LR：逻辑回归 + 人工交叉特征 + PCOC 校准
│   ├── gbdt_lr.py               GBDT+LR：树叶节点特征 + 线性分类器
│   ├── fm.py                    FM：因子分解机（和平方−平方和技巧）
│   └── ffm.py                   FFM：场感知因子分解机
├── ch02_deep_ranking/                 § 12.4.1 深度精排模型
│   ├── wide_deep.py             Wide&Deep：广义线性 + Embedding MLP
│   ├── pnn.py                   PNN：乘积层（IPNN 内积 / OPNN 外积）
│   ├── deepfm.py                DeepFM：FM + DNN，共享 Embedding
│   ├── xdeepfm.py               xDeepFM：CIN 显式高阶交叉 + DNN
│   ├── dlrm.py                  DLRM：两两内积显式交叉 + top MLP
│   ├── deepmcp.py               DeepMCP：预测 + 匹配 + 相关子网络
│   └── dfn.py                   DFN：深度反馈网络（正/负反馈交互）
├── ch03_multi_task/                   § 12.4.2 多任务精排模型
│   ├── mmoe.py                  MMoE：共享专家 + 任务独立门控
│   └── ple.py                   PLE/CGC：任务专属专家 + 共享专家逐层分离
├── ch04_attention/                    § 12.4.3 注意力与 MoE 精排
│   ├── target_attention.py      Target Attention（代码 12.3）+ 缩放点积对照
│   └── moe_ranking.py           MoE 精排：Top-K 稀疏路由 + 负载均衡损失
└── ch05_ranking_loss/                 § 12.5 精排损失函数
    └── pointwise.py             BCE / Pairwise / BCE+Pair / Focal / ListNet 对比
```

---

## 快速运行

```bash
# 传统精排模型（§12.3）
python3 ch01_traditional_ranking/lr.py
python3 ch01_traditional_ranking/gbdt_lr.py
python3 ch01_traditional_ranking/fm.py
python3 ch01_traditional_ranking/ffm.py

# 深度精排模型（§12.4.1）
python3 ch02_deep_ranking/wide_deep.py
python3 ch02_deep_ranking/pnn.py
python3 ch02_deep_ranking/deepfm.py
python3 ch02_deep_ranking/xdeepfm.py
python3 ch02_deep_ranking/dlrm.py
python3 ch02_deep_ranking/deepmcp.py
python3 ch02_deep_ranking/dfn.py

# 多任务精排模型（§12.4.2）
python3 ch03_multi_task/mmoe.py
python3 ch03_multi_task/ple.py

# 注意力与 MoE（§12.4.3）
python3 ch04_attention/target_attention.py
python3 ch04_attention/moe_ranking.py

# 损失函数对比（§12.5）
python3 ch05_ranking_loss/pointwise.py
```

批量跑全部脚本：

```bash
find . -name "*.py" | sort | while read f; do echo "=== $f ==="; python3 "$f" >/dev/null && echo OK || echo FAIL; done
```

每个脚本的 `__main__` 块固定 `random.seed(42)`，用合成数据完整演示训练 + 精排推理流程。各脚本的推理入口按模型输入形态命名：`lr.py` / `fm.py` / `ffm.py` / `wide_deep.py` / `deepfm.py` / `xdeepfm.py` / `dlrm.py` 提供 `rank(...) -> [(item_id, score), ...]`；`deepmcp.py` 提供 `ctr_score(...)` / `match_score(...)`；`dfn.py` / `mmoe.py` / `ple.py` / `moe_ranking.py` / `target_attention.py` 提供 `predict_proba(...)`；`pnn.py` 与 `pointwise.py` 直接调用模型前向或指标函数。

---

## 各模型要点

### § 12.3 传统精排模型

- **`lr.py`**：$$z=w^\top[x;\varphi(x)]+b$$，$$p=\sigma(z)$$，交叉熵 + L2。演示人工交叉特征的作用：无交叉特征 AUC 0.9606 / PCOC 1.108，加入二阶交叉特征后 AUC 0.9619 / PCOC 1.113。同时打印「实际 CTR vs 预测平均 CTR」，说明 PCOC 接近 1 代表概率校准良好——这是精排输出可直接用于竞价的前提。
- **`gbdt_lr.py`**：两阶段结构。先用梯度提升树拟合，把样本落到各树的叶节点上（演示中共 158 个叶节点，即 LR 的输入维度），再对叶节点做 One-Hot 后接 LR。GBDT 与 LR 均为文件内纯 Python 实现（不依赖 scikit-learn / XGBoost）。实测 GBDT 独立精排 AUC 0.8464，GBDT+LR 0.8488——收益主要来自「树自动学非线性划分、线性模型在叶节点特征上高效学习高阶组合」的分工，而非堆叠本身。
- **`fm.py`**：用「和平方 − 平方和」恒等式把二阶交互复杂度从 $$O(d^2k)$$ 降到 $$O(dk)$$。脚本同时实现暴力枚举版作为对照，两者最大误差 1.91e-06，验证恒等式实现正确。测试集 AUC 0.9790。
- **`ffm.py`**：为「特征 × 目标特征域」各学一个隐向量，因此二阶参数是 FM 的 $$F$$ 倍（$$k=4$$、5 个字段时 FM 120 个、FFM 600 个，正好 5.0 倍）。合成数据刻意构造「字段 0×字段 1 为正交互、字段 0×字段 2 为负交互」的域相关模式，给 FFM 区分不同域交互的动机。该脚本 AUC 0.8265，参数与开销的代价在正文中亦有说明（PDF §12.3.4）；注意它与 `fm.py` 使用的是不同的合成数据，两者数值不可横向比较。

### § 12.4.1 深度精排模型

- **`wide_deep.py`**：Wide 分支吃原始稀疏特征 + 人工交叉特征做线性「记忆」，Deep 分支吃 Embedding + 数值特征做 MLP「泛化」，两分支得分相加后统一过 Sigmoid，**共享同一个损失联合训练**（不是分别训练再集成）。AUC 0.9721。脚本末尾打印 Wide 分支中人工交叉特征对应的权重，直观呈现记忆能力的来源。
- **`pnn.py`**：在 Embedding 与 DNN 之间插入 Product Layer 显式构造二阶乘性交互。两种形式都实现：
  - **IPNN**：$$p_{i,j}=\langle f_i,f_j\rangle$$，用秩一分解 $$l_p=\lVert\sum_i\theta_i f_i\rVert^2$$ 把 $$O(N^2)$$ 降到 $$O(N)$$；顺带给出低秩推广（$$K=4$$ 时交互参数 120→480）。
  - **OPNN**：$$p_{i,j}=f_if_j^\top$$，用特征叠加 $$f_\Sigma=\sum_i f_i$$ 压成 $$M\times M$$ 矩阵。

  实测 IPNN（K=1）AUC 0.9020、IPNN（K=4）0.8803、OPNN 0.7770。OPNN 参数最多反而最弱，正好说明「二阶交互的建模方式」比「参数量」更关键。脚本同时打印两种形式的复杂度。
- **`deepfm.py`**：用 FM 组件替换 Wide&Deep 中依赖人工交叉的 Wide 分支，FM 与 DNN **共享同一张 Embedding 表**、无需预训练（这是它与 FNN 的区别）。共享带来的梯度效应是：同一组嵌入同时接收来自 FM 与 Deep 两条路径的梯度。AUC 0.9469。
- **`xdeepfm.py`**：线性组件 + CIN 组件 + DNN 组件。CIN 每层把上一层的交互表示与原始字段嵌入再做逐元素乘积，显式构造**任意阶**交互，各层 sum-pooling 后拼接：$$p^+=[p^{(1)};\dots;p^{(T)}]$$。AUC 0.9745。
- **`dlrm.py`**：稠密特征过 bottom MLP 投到与类别嵌入同维，类别嵌入两两做内积得到显式二阶交互，再与稠密嵌入拼接送入 top MLP 学高阶非线性。交互模块只做内积、不构造高阶交互，这是它与 CIN 类模型的边界。AUC 0.9594。
- **`deepmcp.py`**：三个子网络共享底层嵌入——预测子网（BCE）、匹配子网（用户/查询侧与广告侧分别编码后取内积再过 Sigmoid，点式匹配复用点击标签）、相关性子网（Skip-gram 式，正样本对 + 负采样）。联合目标 $$\mathcal{L}=L_p+\alpha L_m+\beta L_c$$。只有预测子网在线上推理，辅助子网纯属离线训练期的表示学习。实测仅主任务 AUC 0.7439，联合训练 0.8169（+0.0729）；匹配子网单独 AUC 0.9500。
- **`dfn.py`**：把用户行为分成隐式正反馈 C（点击）、显式负反馈 D（不喜欢）、隐式负反馈 U（曝光未点击）。IntraFeedback 先把目标物品与各序列做「加位置嵌入 + 多头自注意力 + 平均池化」，InterFeedback 再拿可靠的 D 与 C 分别对语义不确定的 U 做注意力（匹配函数 $$\text{MLP}([a;b;a-b;a\odot b])$$），得到潜在负/正兴趣。演示数据刻意让 U 序列更长、正负混杂，使 f_u 平均后信息被抹平——此时外部交互是唯一能分离信号的路径：无外部交互 AUC 0.6618，DFN 完整 0.7178（+0.0560）。脚本打印的正/负引导注意力权重分布明显不同。

### § 12.4.2 多任务精排模型

- **`mmoe.py`**：$$n$$ 个共享专家 + 每个任务一个独立门控 $$g^{(k)}=\text{softmax}(W_g^{(k)}x)$$，任务表示 $$f^{(k)}=\sum_i g_i^{(k)}e_i$$ 再进各自塔网络。对比 Shared-Bottom / OMoE / MMoE 三种结构。实测平均 AUC：Shared-Bottom 0.9777 → MMoE 0.9807。门控权重（前 20 样本平均）显示两任务确实选择了不同专家组合：Task0 偏好 E2（0.429），Task1 偏好 E3（0.370）。
- **`ple.py`**：CGC 在共享专家之外给每个任务配专属专家，门控只在本任务专属专家 + 共享专家上做混合；PLE 再把多个 CGC 逐层堆叠，使任务特异性随层深递增。实测平均 AUC：Shared-Bottom 0.9777 / MMoE 0.9807 / PLE 0.9844。最后一层门控显示任务差异越大、越依赖专属专家：Task0 专属专家权重合计 0.993、共享仅 0.007；Task1 专属 0.683、共享 0.317。这正是正文「减少负迁移」机制的直接证据。

### § 12.4.3 注意力与 MoE

- **`target_attention.py`**：对应正文 §12.4.3.2 与代码 12.3。打分网络输入 $$[e_a;e_i;e_a-e_i;e_a\odot e_i]$$ 而不是简单内积，因此能利用差分（方向）与逐元素乘积（交互）信息。脚本做四路对照，均在同一份合成数据、同一 MLP 头上训练：

  | 汇聚方式 | AUC | Δ vs 平均池化 |
  |---|---|---|
  | 平均池化（基线，与候选无关） | 0.6089 | – |
  | 缩放点积 Cross-Attention | 0.6080 | −0.0009 |
  | Target Attention + softmax（代码 12.3） | 0.6307 | +0.0218 |
  | Target Attention，不归一化（DIN 原文） | **0.7208** | **+0.1119** |

  额外打印一个诊断量：固定同一序列、遍历 16 个候选，汇聚表示的 L2 变化量。平均池化恒为 0.0000（结构上就与候选无关），Target Attention 为 0.6872——这是「按候选动态汇聚兴趣」的可观测证据。脚本同时把 softmax 归一化的问题摊开讲：总权重固定为 1，序列中只有少数行为真正相关时注意力会被摊薄；DIN 原始论文的局部激活单元直接用原始打分为权重，保留兴趣强度。

  > **章节边界**：完整的 DIN 与 BST 模型不在第 12 章。正文明确说明它们属于第 20 章《用户兴趣与序列建模》，本章只讲 Target Attention 这一模块在精排特征交互框架中的用法。

- **`moe_ranking.py`**：用 Top-K 稀疏路由的专家混合替换 FFN，对应 §12.4.3.7（Top-K 路由与工程约束）与 §12.4.3.8（共享专家、负载均衡）。演示两组对照：
  - 无负载均衡损失：专家利用率极不均衡，最大/最小比 8.39，标准差 0.1704，设备 token 占比 1.135 / 0.865。
  - 加专家级 + 设备级均衡损失：最大/最小比降到 2.09，标准差 0.0564，设备占比收敛到 0.925 / 1.075；AUC 同时从 0.9386 升到 0.9420。

  即：均衡损失不只是「让利用率好看」的正则项，它回收了被闲置专家的容量，也是稀疏化在分布式环境下兑现「算力与专家数解耦」的前提。脚本另打印共享专家（544 参数，全部 token 激活）与路由专家（4352 参数，每 token 只激活 8 个中的 2 个）的参数结构。

### § 12.5 精排损失函数

- **`pointwise.py`**：复现正文 §12.5 的五损失对比，协议与正文一致——1800 训练请求 × 16 候选、600 验证请求，线性评分模型 $$s(x)=w^\top x+b$$ 从零初始化，22 个 epoch，SGD，学习率 5e-2。逐点目标（BCE / Focal）把「请求 × 候选」展平成 64×16 条曝光一批；排序类目标（Pairwise / BCE+Pair / ListNet）保留请求维度、64 个请求一批，以便在批内构造物品对与列表级分布。

  实测（验证集 CTR = 0.2399）：

  | 损失函数 | AUC | NDCG@5 | Pair准确率 | LogLoss | μ(sigmoid) | PCOC |
  |---|---|---|---|---|---|---|
  | BCE | 0.8074 | 0.6346 | 0.7790 | 0.4252 | 0.2447 | 1.020 |
  | Pairwise | 0.7738 | 0.6264 | 0.7781 | 0.6291 | 0.4997 | 2.083 |
  | BCE+0.20·Pair | 0.8072 | 0.6308 | 0.7787 | 0.4251 | 0.2430 | 1.013 |
  | Focal(γ=2, α=0.25) | 0.8075 | 0.6349 | 0.7788 | 0.4871 | 0.2980 | 1.242 |
  | ListNet | 0.7730 | 0.6394 | 0.7797 | – | – | – |

  三处值得注意的实现细节：

  1. **平移不变性**：纯 Pairwise 与 ListNet 的梯度只作用于分数差分，给所有分数加同一常数不改变损失，因此每步后把 `w[:2]` 与 `b` 清零，模型只靠差分项定序。这也是它们 PCOC 远离 1 的根源（μ=0.4997）。BCE+Pair 不施加该约束——BCE 负责锚定绝对水平——所以校准仍然正常（PCOC 1.013）。
  2. **ListNet 的「–」**：它输出的是请求内归一化的相对分布，不能解释为单个物品的点击概率，故不报告 LogLoss / μ(sigmoid) / PCOC。
  3. **AUC 用 Wilcoxon-Mann-Whitney 秩和**计算，秩必须从最小分数开始编号（升序），否则得到的是 1−AUC；NDCG@5 只对至少含一个正样本的请求求均值（全负请求的 ideal DCG 为 0）。

  脚本后半段单独演示 PCOC：验证集真实 CTR 0.2399，纯 BCE 的 PCOC=1.0198，叠加 $$5.0\cdot(\bar p-\text{CTR})^2$$ 后为 1.0136，AUC 不变（0.8074）——说明校准正则改善的是概率水平而非排序能力，这正是竞价场景需要单独约束 PCOC 的原因。

---

## 实现约定

- **PyTorch 实现**：16 个脚本全部以 `nn.Module` + autograd 实现，`python3 <file>.py` 即可运行；文件之间不互相 import，每个文件自包含。`gbdt_lr.py` 的 GBDT 部分为纯 Python 实现，不依赖 scikit-learn 或 XGBoost。
- **推理入口**：不强行统一签名，按模型输入形态提供——多数字典特征模型用 `rank(feat_ids, top_k)` 返回 `[(item_id, score), ...]`；`deepmcp.py` 用 `ctr_score` / `match_score`；序列类与多任务类模型用 `predict_proba`。
- **自动微分**：训练入口统一为 `fit()` / `train_model()`（避开 `nn.Module.train(mode)` 同名方法），梯度由 `loss.backward()` 自动求得。CPU 上即可运行，不依赖 GPU。
- **合成数据**：每个 `__main__` 块固定 `random.seed(42)`，输出可复现。**各脚本使用各自的合成数据，AUC 只在同一脚本内部横向可比，跨脚本不可比**（例如 `fm.py` 二值特征任务 AUC 0.9790 与 `target_attention.py` 序列任务 AUC 0.7208 不是同一难度）。
- **小规模**：典型配置为数十个字段 / 数百到数千条样本 / emb_dim 8~16 / 20~100 epochs，全部脚本单文件运行均在 12 秒内完成。
- **以已发布 PDF 为准**：若通用论文印象与本书正文口径冲突，以第 12 章发布 PDF 的对应小节为准。以下两条最容易混淆，已按正文处理：

  | 主题 | 正文口径（已据此实现） | 容易误认成 |
  |------|----------------------|-----------|
  | Target Attention | §12.4.3.2 的模块，打分函数为 MLP，**不是**完整 DIN | DIN 精排模型 |
  | DIN / BST | 属于第 20 章《用户兴趣与序列建模》，**不在第 12 章** | 第 12 章的注意力模型 |
  | MoE 精排 | §12.4.3.7（Top-K 路由）与 §12.4.3.8（共享专家 + 负载均衡） | §12.4.3.2 / §12.4.3.3 |
