# ch22 链路一致性建模 — 代码示例

本目录按第 22 章的方法与工业实践组织，包含 **8 个论文方法脚本、1 个工业实践脚本和统一训练/验证入口**。各模型采用小型 PyTorch 计算图和结构化合成级联日志，展示正文强调的机制；`[MOCK]` 表示日志、教师输出、特征、缓存或服务由本地数据代替。每个方法脚本均可单独执行。脚本没有复现论文实验规模、完整网络配置或线上效果。章节 PDF 共 38 页：粗排链路方法见第 1–30 页，粗排与召回工业实践分别见第 30–35 页和第 35–37 页。

---

## 目录结构

```text
code/
├── README.md
├── cascade_data.py                       请求级级联日志与教师信号生成
├── train_examples.py                     9 个方法的训练/验证循环
├── verify_all.py                         机制断言 + 逐脚本训练入口
├── ch01_prerank_consistency/              粗排链路一致性模型
│   ├── ecmm.py                       全链路跨域样本与多任务路由
│   ├── asmol.py                      全场景目标与选择性蒸馏
│   ├── copr.py                       分块采样与 ECPM 排序对齐
│   ├── grace.py                      多阈值 Top-K 一致性与物品对比
│   ├── ranktower.py                  多头门控、双向 GCAN、混合损失
│   ├── hccp.py                       N1–N5 样本与混合目标
│   ├── cs3.py                        双塔 CAS、CTS、CMS
│   └── unirec.py                     两阶段融合与偏好联合优化
└── ch02_industrial_practice/
    └── cascade.py                    Rank/Final 与级联召回
```

---

## 快速运行

Python 3.10+，安装 PyTorch（`python3 -m pip install torch`）。从 `code/` 目录执行：

```bash
python3 -m ch01_prerank_consistency.ecmm
python3 -m ch01_prerank_consistency.asmol
python3 -m ch01_prerank_consistency.copr
python3 -m ch01_prerank_consistency.grace
python3 -m ch01_prerank_consistency.ranktower
python3 -m ch01_prerank_consistency.hccp
python3 -m ch01_prerank_consistency.cs3
python3 -m ch01_prerank_consistency.unirec
python3 -m ch02_industrial_practice.cascade
python3 verify_all.py
```

每个入口通过 `cascade_data.py` 生成固定随机种子的 16 条请求日志，按时间顺序切为 12 条训练请求和 4 条留出验证请求，训练 3 轮。每条请求包含用户 ID 与 8 维用户特征、3 维上下文、18 个召回候选、8 维物品特征、粗排分数、送精排标记、精排分数、精排保留标记、曝光、点击、购买、观看时长、出价、属性和样本来源；链路规模依次为召回 18、送精排 14、精排保留 7、曝光 5，点击蕴含曝光、购买蕴含点击。行为标签依据潜在效用采样，阶段筛选由排序分数产生，未曝光样本不等同于观测到的未点击。教师概率由精排分数生成并标为 `[MOCK]`。

训练循环以请求为一个 mini-batch，逐请求计算方法损失、梯度裁剪并更新参数；输出训练目标均值和留出请求目标均值。不同方法只读取相应阶段可用的字段。`verify_all.py` 先检查机制不变量，再逐进程运行九个训练入口。目标值仅用于确认训练和验证数据流可运行，不代表收敛、论文指标或泛化能力。所有示例在 CPU 上运行，无须内网服务。

---

## 各模型要点

### 粗排目标重构与跨阶段样本建模

- **`ecmm.py` — ECMM：**对召回后候选构造 `ETCTR=点击`、`ETR=实际曝光`、`ITR=未曝光` 三个监督目标。特征先经逐维 Sigmoid 门控，再由 Hard-Concrete 路由到三个任务塔；三个子概率用 Softmax 约束，使 `ETR=ETCTR+曝光未点击`、`ETR+ITR=1`。损失是三个二分类项和期望非零路由数的稀疏正则。训练字段取自请求日志的点击和实际曝光标签；合成特征和较小子网络不还原论文的完整跨域数据流。
- **`asmol.py` — ASMOL：**购买标签蕴含点击标签，点击标签又进入自适应曝光标签。购买任务使用单正例 Softmax；点击和曝光任务对每个正例只与负例竞争，避免正例互相压低。精排 CTR 与 `CTCVR=CTR×CVR` 软标签仅对**实际曝光**样本蒸馏，`ash()` 计算粗排候选对精排 Top-K 的保留率。跨场景行为和教师概率由示例给定，没有真实 RC/PRC 日志构造器。

### 面向下游排序的偏好一致性优化

- **`copr.py` — COPR：**按精排顺序分块，每块随机取一个广告；基础 CTR 网络之外的松弛网络输出 `α=ReLU(MLP(x))+10⁻⁶`，构成 `ECPM=CTR×α×bid`。跨块 pair 的 Logistic 排序损失使用 ΔNDCG 加权，并对 `max(α,1/α)−1` 正则。演示中出价与点击为合成值，线上粗排和精排的出价口径需要由日志保证。
- **`grace.py` — GRACE：**共享专家与任务专家输出 CTR、CVR 及多个精排 Top-K 成员预测；仅对带预训练表示的正例使用物品对比项，批内物品表示构成分母。代码的专家层是缩小版 PLE，不含真实预训练物品编码器。

### 跨阶段模型与样本协同

- **`ranktower.py` — RankTower：**单侧 Multi-Head Gated Network 产生多个表示，门控分支不反传到输入 Embedding。GCAN 用用户表示查询物品子空间、物品表示查询用户子空间，经过门控注意力、残差和 LayerNorm 后，用 `Σ_p max_q cosine(u_p,i_q)/τ` 打分。训练按样本范围分别使用曝光蒸馏、曝光加候选 SoftSort、全阶段 AM-Rankmax。为使约束全部满足时损失有定义，教学版 AM-Rankmax 使用 `log1p(Σ violation)`；这是相对于论文印刷 `log(Σ violation)` 的**显式数值稳定化**。可预计算的是单侧门控表示，跨塔注意力仍需在线配对执行。
- **`hccp.py` — HCCP：**N1 曝光样本始终保留；N2 按下游排名区间非均匀采样，点击/曝光/未曝光依反馈重排；N3 从召回尾部采样，N4 来自其他请求，N5 来自未召回物品池。全局 ListMLE 作用于重排后的 N1/N2，局部 ListMLE 与 BCE 仅作用于 N1。Margin InfoNCE 对正例和 N2/N3 用 `v(cosθ−m)/τ`，对 N4/N5 用 `v·cosθ/τ`；两者不能互换。三塔和 MMoE 为小型教学网络，缺少真实级联日志与池服务。

### 在线能力协同与多阶段联合优化

- **`cs3.py` — CS3：**CAS 用同一线性层先算初始表示，再预测重要性并以 `2×Sigmoid` 修正输入，执行第二次前向；CTS 仅在正反馈时以 EMA 交换用户/物品表示；CMS 对正负反馈均以 EMA 缓存下游模型中间表示。字典缓存是 `[MOCK]`，没有参数服务器、嵌入服务器、并发更新或缓存失效协议。
- **`unirec.py` — UniRec：**粗排与精排分别拥有融合塔；粗排输入只含该阶段可用的 pXTR，精排加入本阶段 pXTR 和上游表示。融合塔含 DCN，粗排后接轻量 Mixer，精排以单头 AutoInt-Lite 在**候选物品之间**做注意力；Adapter 消费停止梯度的粗排隐藏表示。先验 pXTR、曝光反馈和下游排序方向分别形成阶段内/跨阶段偏好；CPPA 聚合两个方向的任务权重，CAPP 为每个候选保留高置信度 Top-K pair，AGRR 在属性分组内拟合相对奖励分布。脚本只训练融合模块，pXTR 网络与业务奖励为合成输入；单头注意力和两字段 Mixer 是教学简化。

### 工业实践

- **`cascade.py`：**由精排 Top-K 与最终 Top-K 分别构造 Rank/Final 目标；正样本位置权重作用于 Focal Loss 而非预测概率。双塔向量既可用于粗排点积，也可作为级联召回的索引向量。脚本用精确点积代替 ANN，未实现索引增量更新；真实服务必须保证用户向量与物品索引来自兼容模型版本。

---

## 实现约定与可信边界

- **训练数据边界：**`cascade_data.py` 是确定性教学生成器，不代表真实线上日志采样。粗排/精排分数与行为反馈来自同一个潜在效用过程，数据存在刻意设置的相关性；没有延迟反馈、位置倾向校正、负采样权重、跨天漂移或请求级分布式批处理。训练/验证按请求切分，避免同一请求候选拆到两侧。
- **训练过程边界：**9 个入口都执行真实前向、反向传播和参数更新，并在留出请求上计算同一个方法目标。小样本、少轮数、方法损失量纲不同，因此不能横向比较打印的目标值，也不能据此声称模型收敛或验证效果提升。

- **真实入口：**九个脚本均在 `__main__` 中执行若干次梯度更新或一次可观察的在线路径；`verify_all.py` 还覆盖 CPPA 与逐任务公式的等价、CAPP pair 数、AGRR 组内平移不变性、Adapter 停梯度、CS3 更新条件、HCCP 样本层次与 RankTower 缓存分数等性质。机制断言不能替代官方代码对拍。
- **接口范围：**这里不强行提供统一 `pre_rank()` 接口。UniRec 联合训练与分阶段推理、CS3 的缓存读写、HCCP 的请求级样本构造需要不同输入；每个脚本均在类和函数签名中显式给出输入。
- **生产缺口：**没有真实训练/在线日志、完整特征系统、教师模型、分布式参数服务、跨服务超时与降级、容量压测、模型与索引版本切换、业务指标回归或回滚实现。合成 loss 不能作为上线验收。
- **已核对的论文：**[RankTower](https://arxiv.org/abs/2407.12385)、[HCCP](https://arxiv.org/abs/2502.10284)、[GRACE](https://arxiv.org/abs/2405.05606)、[UniRec 当前版本](https://arxiv.org/abs/2609.11052)。UniRec 早期 arXiv v1 与当前正文标题和方法有差异，代码采用当前版本的偏好优化口径。
