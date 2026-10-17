# ch10 召回模型 — 代码示例

本目录对应教材第 10 章，覆盖召回阶段主流模型的 Python 实现，共 **38 个可运行脚本**。目录编号与正文节次一一对应，均可直接 `python3 <文件>.py` 运行。神经网络类脚本（含图网络、向量召回、序列召回、多兴趣等）依赖 **PyTorch**（`pip install torch`）；协同过滤与图嵌入中的纯统计模型（`item_cf`、`user_cf`、`swing`、`online_cf`、`svd`、`deepwalk`、`node2vec`）仅用标准库，无需额外安装。

---

## 目录结构

```
code/
├── ch01_collaborative_filtering/      § 10.1 协同过滤召回
│   ├── item_cf.py               ItemCF：基于物品的协同过滤
│   ├── user_cf.py               UserCF：基于用户的协同过滤
│   ├── online_cf.py             Online CF：增量更新 + 时间衰减 + 惰性重算
│   ├── swing.py                 Swing：共现用户重叠度惩罚
│   ├── svd.py                   SVD：Jacobi 旋转法实现截断奇异值分解
│   ├── mf.py                    BPR-MF：贝叶斯个性化排序矩阵分解
│   ├── neumf.py                 NeuMF：GMF + MLP 双分支神经协同过滤
│   └── heterogeneous_cf.py      Heterogeneous CF：多行为加权共现 + EHCF
├── ch02_graph_embedding/              § 10.2.1 图嵌入召回
│   ├── deepwalk.py              DeepWalk：随机游走 + Skip-Gram
│   ├── node2vec.py              Node2Vec：p/q 偏置二阶游走 + Alias 采样
│   └── eges.py                  EGES：side info 加权融合（权重逐物品可学习）
├── ch03_graph_network/                § 10.2.2 图网络召回
│   ├── gcn.py                   GCN：对称归一化邻接矩阵逐层传播
│   ├── gat.py                   GAT：多头注意力聚合邻居
│   ├── graphsage.py             GraphSAGE：邻居采样 + mean/pool 聚合（归纳式）
│   ├── lightgcn.py              LightGCN：去掉特征变换的轻量图卷积
│   ├── pinsage.py               PinSage：RWR 重要性采样 + Margin Ranking Loss
│   └── hybridgnn.py             HybridGNN：异构图分关系聚合 + 跨关系注意力
├── ch04_vector_retrieval/             § 10.3 向量召回
│   ├── youtube_dnn.py           YoutubeDNN：历史行为平均池化 + Sampled Softmax
│   ├── two_tower.py             双塔召回：In-Batch 负采样 Softmax
│   ├── two_tower_debias.py      负采样偏差处理：LogQ 矫正 + 流式频率估计
│   ├── fasttext.py              FastText：字符 n-gram 哈希，解决 OOV
│   ├── mini_bert.py             BERT：极简 Transformer Encoder + MLM 预训练
│   ├── sentence_bert.py         Sentence-BERT：孪生网络 + mean pooling
│   ├── simcse.py                SimCSE：dropout 双视角对比学习（InfoNCE）
│   └── e5.py                    E5：query/passage 非对称前缀 + 难负样本
├── ch05_sequence_retrieval/           § 10.4 序列召回
│   ├── item2vec.py              Item2Vec：session 内两两共现（无滑动窗口）
│   ├── list2vec.py              List2Vec：item-item + item-list 联合建模
│   ├── gru4rec.py               GRU4Rec：nn.GRU + session-parallel + TOP1/BPR
│   ├── sasrec.py                SASRec：因果单向自注意力
│   ├── bert4rec.py              BERT4Rec：双向注意力 + Cloze 完形填空
│   ├── sdm.py                   SDM：长短期兴趣 + 门控融合
│   ├── cl4srec.py               CL4SRec：crop/mask/reorder 三增强 + 对比学习
│   └── icsrec.py                ICSRec：意图原型聚类 + CICL/FICL 双对比
├── ch06_multi_interest/               § 10.5 多兴趣召回
│   ├── mind.py                  MIND：胶囊网络动态路由 + Label-aware Attention
│   ├── comirec.py               ComiRec：DR/SA 双变体 + 可控多样性聚合
│   └── remi.py                  REMI：兴趣感知难负样本 + 路由正则化
└── ch07_others/                       § 10.6 其他召回模型
    ├── pdn.py                   PDN：Trigger Net + Similarity Net 路径建模
    └── tdm.py                   TDM：树检索 + 逐层 Beam Search + 树重建
```

---

## 快速运行

```bash
# 协同过滤（§10.1）
python3 ch01_collaborative_filtering/item_cf.py
python3 ch01_collaborative_filtering/user_cf.py
python3 ch01_collaborative_filtering/online_cf.py
python3 ch01_collaborative_filtering/swing.py
python3 ch01_collaborative_filtering/svd.py
python3 ch01_collaborative_filtering/mf.py
python3 ch01_collaborative_filtering/neumf.py
python3 ch01_collaborative_filtering/heterogeneous_cf.py

# 图嵌入与图网络（§10.2）
python3 ch02_graph_embedding/deepwalk.py
python3 ch02_graph_embedding/node2vec.py
python3 ch02_graph_embedding/eges.py
python3 ch03_graph_network/gcn.py
python3 ch03_graph_network/gat.py
python3 ch03_graph_network/graphsage.py
python3 ch03_graph_network/lightgcn.py
python3 ch03_graph_network/pinsage.py
python3 ch03_graph_network/hybridgnn.py

# 向量召回（§10.3）
python3 ch04_vector_retrieval/youtube_dnn.py
python3 ch04_vector_retrieval/two_tower.py
python3 ch04_vector_retrieval/two_tower_debias.py
python3 ch04_vector_retrieval/fasttext.py
python3 ch04_vector_retrieval/mini_bert.py
python3 ch04_vector_retrieval/sentence_bert.py
python3 ch04_vector_retrieval/simcse.py
python3 ch04_vector_retrieval/e5.py

# 序列召回（§10.4）
python3 ch05_sequence_retrieval/item2vec.py
python3 ch05_sequence_retrieval/list2vec.py
python3 ch05_sequence_retrieval/gru4rec.py
python3 ch05_sequence_retrieval/sasrec.py
python3 ch05_sequence_retrieval/bert4rec.py
python3 ch05_sequence_retrieval/sdm.py
python3 ch05_sequence_retrieval/cl4srec.py
python3 ch05_sequence_retrieval/icsrec.py

# 多兴趣召回（§10.5）
python3 ch06_multi_interest/mind.py
python3 ch06_multi_interest/comirec.py
python3 ch06_multi_interest/remi.py

# 其他召回模型（§10.6）
python3 ch07_others/pdn.py
python3 ch07_others/tdm.py
```

批量跑全部脚本：

```bash
find . -name "*.py" | sort | while read f; do echo "=== $f ==="; python3 "$f" >/dev/null && echo OK || echo FAIL; done
```

每个脚本的 `if __name__ == "__main__":` 块会固定随机种子（`random.seed(42)`）、生成合成数据、训练若干轮并打印 Top-K 召回结果。

---

## 各模型要点

### § 10.1 协同过滤召回

| 文件 | 核心公式 / 机制 |
|------|----------------|
| `item_cf.py` | $$S(i,j) = \frac{\|N(i)\cap N(j)\|}{\sqrt{\|N(i)\|\cdot\|N(j)\|}}$$ |
| `user_cf.py` | $$S(A,B) = \frac{\|N(A)\cap N(B)\|}{\sqrt{\|N(A)\|\cdot\|N(B)\|}}$$ |
| `online_cf.py` | 增量更新 $$\text{Score}(i,j) \leftarrow \text{Decay}(\text{Score}(i,j)) + w(j)$$，时间衰减 $$\text{Decay}(x)=x e^{-\lambda\Delta t}$$ |
| `swing.py` | $$S(i,j) = \sum_{u,v\in N(i)\cap N(j)} \frac{1}{\alpha + \|N(u)\cap N(v)\|}$$ |
| `svd.py` | $$R = U\Sigma V^\top$$，截断为 rank-$$d$$ 近似；Jacobi 旋转法对角化 Gram 矩阵 |
| `mf.py` | BPR 损失 $$\mathcal{L} = -\log\sigma(\hat{r}_{ui^+} - \hat{r}_{ui^-})$$ |
| `neumf.py` | GMF 逐元素乘积分支 + MLP 拼接分支，拼接后 Sigmoid + BCE |
| `heterogeneous_cf.py` | 多行为置信度加权共现 $$S(i,j)=\sum_k \lambda_k S^{(k)}(i,j)$$，含 EHCF 行为迁移矩阵 |

- `online_cf.py` 的演示展示了热门物品注入后相似度榜单的重排，以及静默 3600 秒后分数的衰减过程，并统计惰性重算节省的计算量。
- `svd.py` 用「零填充 vs 均值填充」的 RMSE 对比量化了 SVD 在稀疏矩阵上的缺陷，这正是后续 MF 只在观测集 $$\Omega$$ 上优化的动机。

### § 10.2 图召回

- `node2vec.py`：二阶游走的非归一化转移概率 $$\pi_{vx}=\alpha_{pq}(t,x)\cdot w_{vx}$$，其中 $$\alpha$$ 按上一跳 $$t$$ 到候选 $$x$$ 的距离取 $$1/p$$、$$1$$、$$1/q$$。演示对比了 BFS 式 $$(p{=}1,q{=}4)$$ 与 DFS 式 $$(p{=}1,q{=}0.25)$$ 下同社区节点占比 0.950 vs 0.703。
- `eges.py`：融合权重 $$H_v = \sum_s e^{a_v^s} W_s(v) / \sum_s e^{a_v^s}$$，**权重矩阵 $$A$$ 逐物品私有且可学习**。演示包含冷启动物品（无交互、仅有 side info）仍能召回到同品类物品。
- `gcn.py` / `gat.py` / `graphsage.py` / `lightgcn.py` / `pinsage.py` / `hybridgnn.py`：每个文件的 docstring 都点明了与其他 GNN 的关键区别——GCN 聚合权重由度数固定、GAT 改为可学习注意力、GraphSAGE 改为采样 + 归纳式（演示了训练时不存在的新用户直接前向推理）、LightGCN 去掉特征变换、PinSage 用随机游走访问频次做重要性池化、HybridGNN 按关系类型分别聚合。

### § 10.3 向量召回

- `two_tower_debias.py`：演示 In-Batch 负采样的热门物品偏置。未矫正时 Top-5 热门占比 50.0%，LogQ 矫正后升至 96.7%，直观体现未矫正会过度惩罚热门物品。
- `fasttext.py`：OOV 词（如 `playful`、`runners`）在 word2vec 中无向量，FastText 靠字符 n-gram 仍能检索到同词根词。
- `mini_bert.py` → `sentence_bert.py` → `simcse.py` → `e5.py` 构成一条递进链：MLM 预训练 → 孪生网络解决 $$O(n^2)$$ 推理问题 → dropout 对比学习缓解各向异性（跨主题相似度 0.5417→0.0462）→ 非对称前缀 + 难负样本（Top-1 准确率 12.5%→100%）。

### § 10.4 序列召回

- `item2vec.py`：末尾断言「序列顺序颠倒后共现样本完全一致」，直接验证 Item2Vec 丢弃了时序信息。
- `gru4rec.py`：实现真正的 session-parallel mini-batch（每槽位独立推进一个 session，结束后换入新 session 并清零隐状态），TOP1 与 BPR 两种 ranking 损失。
- `bert4rec.py`：与 SASRec 的对比写在 docstring 中——SASRec 因果 mask 单向预测，BERT4Rec 双向注意力 + 随机 mask，推理时末尾追加 `[MASK]` token。
- `sdm.py`：核心是门控融合 $$G=\sigma(W_1 e_u + W_2 s + W_3 p + b)$$，$$o=(1-G)\odot p + G\odot s$$。
- `cl4srec.py`：三种增强算子（crop / mask / reorder）+ InfoNCE，总损失 $$\mathcal{L}_{\text{rec}} + \lambda\mathcal{L}_{\text{cl}}$$。
- `icsrec.py`：引入意图原型（自写 k-means），CICL（同目标物品子序列互为正样本）+ FICL（与最近邻原型拉近）。

### § 10.5 多兴趣召回

- `mind.py`：胶囊网络动态路由抽取 K 个兴趣向量 + Label-aware Attention。
- `comirec.py`：DR 与 SA 两个变体，核心贡献是**可控多样性聚合** $$Q(u,S)=\sum f(u,i) + \lambda\sum\sum g(i,j)$$。演示显示 $$\lambda$$ 从 0 增到 1.0 时类目覆盖 5/8→6/8、相关性 +0.135→+0.058，权衡清晰可见。
- `remi.py`：五组配置同种子对比，验证兴趣感知难负样本把负样本相似度从 −0.2413 推高到 +0.1589，路由正则化把路由熵从 1.2147 提到 1.3377（缓解兴趣坍缩）。

### § 10.6 其他召回模型

- `pdn.py`：路径打分 $$y_{ui}=\text{AGG}_j(t_{uj}\cdot s_{ji})$$，每条推荐都能回溯到具体 trigger 物品（演示输出附主路径 `u→1→9`）。
- `tdm.py`：完整树检索框架。演示打印 Beam Search 逐层下探过程（候选 2→4→6→6，每层保留 3 个），并外推说明 $$|I|=2^{20}$$ 时加速约 8738 倍。还包含基于叶子 Embedding 的 k-means 树重建 + 重训，HR@5 从 0.781 提升到 0.906。

---

## 实现约定

- **两种实现风格**：涉及神经网络与 Loss 计算、需要反向传播的文件（图网络、向量召回、序列召回、多兴趣、`mf`、`neumf`、`eges` 等）统一用 **PyTorch**（`nn.Module` + autograd）实现，梯度由 `loss.backward()` 自动求得，无需手写反向传播；纯统计/矩阵分解类模型（`item_cf`、`user_cf`、`swing`、`online_cf`、`svd`、`deepwalk`、`node2vec`）仍用标准库实现，因为它们不含可训练的神经网络。所有文件之间不互相 import，每个文件自包含（即使多个文件都需要 Transformer Encoder，也各自内嵌一份精简实现）。
- **统一训练接口**：模型类继承 `nn.Module`，训练入口统一为 `train_model()` / `fit()`（避开 `nn.Module.train(mode)` 同名方法），内部为标准 `optimizer.zero_grad(); loss.backward(); optimizer.step()` 三步。CPU 上即可运行，不依赖 GPU。
- **合成数据**：每个 `__main__` 块固定 `random.seed(42)`，内置随机生成的交互数据，可直接运行观察输出。
- **小规模**：典型配置为 8 用户 / 20 物品 / emb_dim=16 / 5~6 epochs，全部脚本单文件运行均在 4 秒内完成。
- **超参数**：均通过构造函数参数暴露，便于调整实验。
