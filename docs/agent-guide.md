# Agent 使用指南

本仓库把发布 PDF 作为权威正文，把 `knowledge/` 作为导航层，把命令行工具和 Skills 作为工作流层。无需 MCP、数据库或云端知识库。

本地也可直接打开 [`../knowledge-map/index.html`](../knowledge-map/index.html) 浏览全书知识蓝图、概念前置关系与课程路径；它与 GitHub Pages 使用同一份公开导航数据，不含 PDF 全文或个人进度。知识内容更新后运行 `python3 tools/build_knowledge_map.py` 重建页面数据。

## 1. 五分钟开始

环境：Python 3.10+；索引需要系统命令 `pdftotext`。在仓库根目录执行：

```bash
python3 tools/recsys_agent.py index
python3 tools/recsys_agent.py doctor
python3 tools/recsys_agent.py search "双塔召回的负采样偏差"
python3 tools/recsys_agent.py page --chapter 10 --page 60
```

`doctor` 应显示 `index=ready`，且其模式判定与检索命令一致：无知识源、源文件变更、索引损坏或 manifest 模式不匹配时为 `index=stale`。索引写入被忽略的 `.recsys-agent/`；PDF 更新后重新执行 `index`。检索、讲解、学习计划、工程报告和自测会拒绝过期或来源模式不匹配的索引。所有命令可通过以下入口发现：

```bash
python3 tools/recsys_agent.py --help
python3 tools/recsys_agent.py search --help
```

## 2. 命令总览

| 命令 | 用途 | 典型输出 |
|---|---|---|
| `index` | 构建或刷新本地 PDF 索引 | 分块数与 manifest |
| `doctor` | 检查依赖、知识源和索引新鲜度 | `index=ready/stale` |
| `search` | BM25 词项评分与特征哈希相似度检索 PDF | 文件、PDF 页、节、摘录 |
| `page` | 阅读一个命中页 | 该 PDF 页的本地提取文本 |
| `eval` | 运行检索与引用基准 | Hit@1、Recall@5、引用有效率 |
| `learn` | 生成自由主题限时学习计划 | 前置、阅读页、主题化时间预算、实践与验收 |
| `complete` | 用提交物、实践、失败路径和测验分数完成自由主题 | `assessed/completed` 状态 |
| `course` | 运行完整课程学习闭环 | 今日任务、反馈、提交、进度与延迟复习 |
| `next` | 按前置与掌握状态推荐下一主题 | 概念 ID 列表 |
| `graph` | 输出全局或课程 Mermaid 知识图谱 | 节点与依赖边 |
| `self-eval` | 生成读者本地自测报告 | 检索、引用、学习与复习状态 |
| `overview` | 汇总课程、自由主题、复习和陪练状态 | 本地学习概览 |
| `session-clear` | 清除指定问答会话 | 显式结束一个问答主题 |
| `explain` | 按受众组织讲解提纲和 PDF 证据 | 受众提纲、引用页码、证据摘录 |
| `design` | 生成可机读系统设计报告 | 链路、P99 预算、容量、稳定性、版本与发布契约 |
| `review` | 生成带位置的领域评审报告 | 发现、风险、修复与验证要求 |
| `debug` | 生成证据优先诊断报告 | 假设、证据、结论、修复与恢复验证 |

注意：文档中的 `/learn`、`/design` 等表示 Agent 工作流名称，不是 shell 命令。没有安装客户端 Slash Command；终端中必须写完整 CLI，如 `python3 tools/recsys_agent.py learn ...`。在 Claude Code 等客户端中，用自然语言要求 Agent 执行对应 Skill 即可。

## 3. 面向读者的问书、追问与导航

```bash
python3 tools/recsys_agent.py index  # 首次使用或 PDF 更新后
python3 tools/recsys_agent.py ask "双塔召回的 Batch 内负采样为什么有偏差？" --session two-tower
python3 tools/recsys_agent.py ask "书中如何估计采样概率？" --session two-tower
python3 tools/recsys_agent.py navigate --session two-tower
```

`ask` 使用本地已发布 PDF 索引挑选少量页面，再通过本机 `claude -p` 或 `codex exec` 组织回答。提示词只包含当前问题、上一轮短摘要及候选页摘录；Claude 调用禁用工具和会话持久化，Codex 调用使用只读沙箱（不等于禁用其读取能力）。程序把每条书中论断的引用 ID 绑定到这些已存在的 PDF 文件和页码。若检索不到可信候选或模型证据不足，则明确拒答；`--json` 可供其他客户端消费。`--session` 在被 Git 忽略的 `.recsys-workspace/reader-sessions/` 保存主题、最近问题和论断摘要，不保存 PDF 提取文本；可用 `session-clear --session <name>` 清除。通过 `--agent codex` 可显式使用另一运行时，`coach` 也支持同一参数；`doctor` 会分别报告两种 CLI 是否可用。模型调用可能耗费额度且需要本机可用认证；自动绑定页码不能证明每句推理正确，重要公式仍应打开 PDF 原页核对。

`navigate` 把当前主题连接到已发布章节、概念卡、前置概念、相关教学代码和人工精选的代表论文；空缺资源会明确留空，不会凭关联猜测书中结论。静态网站仍是导航页，不在线运行模型。

## 3a. 原始检索与证据检查

```bash
# 普通检索
python3 tools/recsys_agent.py search "HNSW efSearch 如何影响召回和延迟"

# 限定章节和返回数量
python3 tools/recsys_agent.py search "负采样" --chapter 10 --limit 3

# 生成面向不同读者的讲解提纲和证据包
python3 tools/recsys_agent.py explain "双塔召回" --audience new-engineer

# 供脚本消费
python3 tools/recsys_agent.py search "Embedding 版本" --json

# 命中后读取完整页
python3 tools/recsys_agent.py page --chapter 17 --page 17
```

检索结果是发现证据，不是最终答案。公式、表格、限定条件或相邻上下文重要时，必须执行 `page` 或打开对应 PDF 页面。回答格式：

精确输入某张概念卡的标题（如「负采样」）时，检索优先展示该卡「本书依据」标注的已发布 PDF 页；加上具体场景的查询仍按检索分数排序。概念卡引用只用于导航，最终论断必须核对 PDF 原页。

- `本书依据`：只陈述发布 PDF 支持的内容并标注文件和 PDF 页；
- `扩展说明`：工程经验、外部资料或推导；
- 找不到时明确说发布章节未覆盖，不补造书中结论。

`explain` 只生成受众化讲解提纲和证据包，不会自行调用语言模型生成答案。无关查询低于保守分数门槛时返回 `needs_source`；`ready` 也只是候选证据，需阅读 PDF 核对相关性，不能当成答案正确。CLI 的相似度特征基于词项哈希，不是 embedding 或真正的语义检索；跨措辞检索效果应以固定正例、页级与负例评测集为准。

章节级学习入口是 [`knowledge/chapters/README.md`](../knowledge/chapters/README.md)；概念入口是 [`knowledge/README.md`](../knowledge/README.md)。章节地图不是正文替代品。

## 4. 学习工作流

完整课程从一次初始化开始：

```bash
python3 tools/recsys_agent.py course list
python3 tools/recsys_agent.py course start new-engineer  # 或 graduate
python3 tools/recsys_agent.py course today --minutes 90
python3 tools/recsys_agent.py course feedback --message "本周只有 45 分钟；负采样校正仍不清楚"
python3 tools/recsys_agent.py course status
# 对实际提交物生成针对性追问和下一步任务（不会自动给出掌握分数）
python3 tools/recsys_agent.py coach --course new-engineer --evidence /tmp/ne-01.md --feedback "负采样校正仍不清楚"
```

两门 12 周课程覆盖全部 20 个已发布章节：

- [`learning-paths/new-engineer.md`](../learning-paths/new-engineer.md)：全链路工程、可靠性、故障与毕业项目；
- [`learning-paths/graduate.md`](../learning-paths/graduate.md)：可信实验、强基线、偏差、复现与迁移。

完成单元不能只口头声明。提交物必须包含 PDF 证据、问题答案、主路径和失败路径的真实命令与输出：

```bash
python3 tools/recsys_agent.py course submit \
  --course new-engineer --unit ne-01 --evidence /tmp/ne-01.md \
  --score 0.85 --practice-passed --failure-path
python3 tools/recsys_agent.py course review
python3 tools/recsys_agent.py self-eval
```

证据不全或分数低于 0.8 时只记为 `assessed`，不会进入 `completed`；仅允许提交当前未完成单元，已完成状态不会被低分重复提交撤销。代码实验单元需要主路径和失败路径；方案单元在提交物中写 `## 方案验证`，逐行列出 `检验方法`、`反例或失败条件`、`验收依据`，不强制命令日志。`course today` 会给大单元生成可顺序完成的概念检查点；完成阅读后可用 `course checkpoint --course new-engineer --id ne-04-1` 记录位置，下一次 `today` 从尚未记录的检查点继续。检查点仅表示已访问，不是掌握评分。CLI 仅检查结构，仍需评阅者按 `acceptance` 核对质量。七天后 `course review` 安排回忆、纠错、参数预测和迁移任务；核查四节答案后，可用 `course review --course new-engineer --unit ne-01 --evidence /tmp/ne-01-review.md --score 0.85` 提交复习，通过后不再重复提醒。课程外补缺可运行：

```bash
python3 tools/recsys_agent.py learn "HNSW" --level graduate --minutes 120
python3 tools/recsys_agent.py complete hnsw --evidence /tmp/hnsw.md \
  --score 0.9 --practice-passed --failure-path
python3 tools/recsys_agent.py next
```

`coach` 会对当前单元提交物做结构检查，并在模型可用时结合课程阅读页、学习者的解释与反馈，提出具体待核对的概念缺口和下一道追问；反馈写入本机忽略的陪练记录，下一次会话可继续。可传 `--model <可用模型ID>`；离线或不愿发送学习提交物给模型时使用 `--no-model`，此时只做结构性检查。它不自动判分、不执行提交物中的命令，也不把结构完整视作掌握；正式完成仍需人工核对并使用 `course submit`。

详细的读者提示词、证据模板、反馈方法、状态语义和隐私边界见 [`learning-agent-guide.md`](learning-agent-guide.md)。个人状态只写入被 Git 忽略的 `.recsys-workspace/`。

## 5. 工程工作流

现有 `design`、`review`、`debug` 支持可选的 `--project` 指向读者自己的项目，并通过 `--domain`、`--evidence` 核对用户提供的本地 JSON 观测。书籍索引仍由仓库根目录提供；目标项目的代码和证据不会写入公开仓库。`--project` 仅列出有限的入口与测试候选，Agent 还必须阅读实际实现、调用方和目标项目的运行说明。对目标项目的改动由宿主 Agent 在用户要求修复时执行，CLI 只做只读检查。

若目标项目能输出单条 JSON 观测，可以用 `verify` 运行三个明确的本地脚本，检查基线、故障和恢复：

```bash
python3 tools/recsys_agent.py verify \
  --project /path/to/my-recommender --domain feature-time \
  --baseline "python3 emit_observation.py baseline" \
  --fault "python3 emit_observation.py fault" \
  --recovery "python3 emit_observation.py recovery"
```

`verify` 只接受目标项目目录内的 Python 脚本，单条命令最长 300 秒以内，要求标准输出为一个 JSON 对象。`feature-time` 可输出带 `rows` 的对象（每行含 `label_time`、`feature_time`）；`release-version` 的正常输出也可用 `status=ok`、`version` 表示当前版本；其他场景输出相应契约字段。故障命令可用退出码 2 加 `status=rejected` 和领域匹配的 `error`（特征穿越含 `future feature`，版本错配含 `version mismatch`）表示显式拒绝，这种结果仍需阅读代码确认拒绝原因。输出会区分 `fault_reproduced_and_recovered`、`fault_rejected_and_recovered`、`baseline_or_recovery_failed` 和 `fault_not_reproduced`；这证明了本地脚本本次运行的契约行为，不等于线上根因或性能达标。

三个首批契约为 `feature-time`、`serving`、`release-version`。证据文件格式是 `{ "domain": "...", "records": [...] }`；每条记录的 `phase` 为 `baseline`、`fault` 或 `recovery`。这是用户提供的样本，不自动认定为现场遥测。推荐提供三阶段记录；`coverage` 会显示缺失阶段。具体字段如下：

| 领域 | 每条记录必需字段 | 检查结果 |
|---|---|---|
| `feature-time` | `label_time`、`feature_time`（同单位的数值，或带时区 ISO 时间）、`selected`（布尔） | 入样特征不能晚于标签时刻 |
| `serving` | `budget_ms`（正数）、`elapsed_ms`（非负数）、`candidates`（非负整数） | 请求不超预算且候选非空 |
| `release-version` | `model_release`、`embedding_release`、`index_release`（非空字符串）、`accepted`（布尔） | 不兼容版本必须拒绝 |

例如，将脱敏的故障记录保存在目标项目以外的本地文件 `/tmp/feature-observations.json` 后运行：

```bash
python3 tools/recsys_agent.py review src/producer.py \
  --project /path/to/my-recommender \
  --domain feature-time --evidence /tmp/feature-observations.json
python3 tools/recsys_agent.py debug "样本包含未来特征" \
  --project /path/to/my-recommender \
  --domain feature-time --evidence /tmp/feature-observations.json
```

`evidence_check` 返回违规记录序号和三阶段覆盖情况。工程 Agent 应继续核对证据来源、阅读源码和调用方，在本地安全样本上复现；用户要求修复时修改目标项目，并用同一个故障场景验证修复前失败、修复后通过、边界样本行为正确。只凭 `no_violation_in_samples` 不能宣称代码正确或线上指标达标。`design`、`review`、`debug` 的证据检查只读本地文件；`verify` 会执行指定脚本。当前不做跨文件数据流证明，也不自动接入监控平台。

### 设计

```bash
python3 tools/recsys_agent.py design "在线推荐服务" \
  objective=watch_time qps=10000 p99=200ms items=1e8 \
  freshness=5m availability=99.99% cost=bounded
```

输出 JSON 包含请求链路、数值 P99 预算、容量规划估算、容量公式、版本契约、过载/降级、观测、灰度、回滚、来源和仍待确认约束。约束齐全且格式有效时仅返回 `status=draft` 与 `verification=not_run`：这是一份待验证的设计草案，不表示运行、压测或上线验收通过。给出 `qps` 与 `p99` 时会计算 `minimum_inflight_at_p99 = ceil(qps × p99_seconds)`；这里用 P99 作保守规划参考，不是 Little 定律意义上的最小并发量（平均在途数需使用平均时延）。单副本安全 QPS 等实测参数不会臆造。部署前需结合真实业务约束、PDF 来源、可运行实现、故障注入、负载与回滚演练单独记录验证结果。

`qps`、`p99`、`items` 必须是正数（支持 `k`/`m` 后缀和科学计数法；P99 可带 `ms`）；`availability` 必须在 0 到 100 之间。无效输入会列入 `invalid` 并将状态设为 `needs_input`。

### 评审

```bash
python3 tools/recsys_agent.py review templates/recommendation-pipeline/app.py
```

输出 JSON 包含精确文件位置、静态线索、待执行验证和 PDF 来源；`status=needs_runtime_verification` 不是代码通过评审。静态检查只识别仓库内显式模式，不能替代运行验证。

当前 PIT 检查只通过 Python AST 识别少数明确比较表达式；识别到表达式不代表运行时行为正确。未识别代码或语法错误时不会出具通过结论。
对于 Markdown 等非 `.py` 文件，返回 `unsupported_target` 和文件类型，不执行 Python 静态评审。

### 排障

```bash
python3 tools/recsys_agent.py debug "精排 CTR 突降"
```

输出 JSON 包含有序待核查假设、所需证据、可能的修复方向、恢复验证和 PDF 来源；无现场遥测时返回 `status=needs_evidence`，不能当作根因结论。只有显式加 `--fixture multi-recall-tail-latency` 才展示仓库教学案例；该案例不是当前故障的遥测。

开发交付必须区分 `本书依据`、`工程实现`、`验证` 和 `待业务确认`。

## 6. 让章节 `code/` 成为 Agent 的教学与代码参考

四个 `templates/` 是跨章节、端到端的练习；各已发布章节下的 `code/` 则是细粒度方法实验，两者互补。Agent 不应只围绕四个模板工作。先用统一入口按主题或章节找真实存在的源码及该章说明：

```bash
python3 tools/recsys_agent.py code "负采样偏差" --chapter 10
python3 tools/recsys_agent.py code "HNSW"
python3 tools/recsys_agent.py code --chapter 3 --limit 8
python3 tools/recsys_agent.py navigate "负采样"  # 概念卡、PDF 页、代码、代表论文
```

`code` 只根据各章 `code/README.md` 的文件说明与已存在的源码做轻量检索，不运行、不导入示例；没有准确匹配就返回 `not_found`，没有 README 描述的文件也可按文件名搜索。`explain`、`learn` 和 `course today` 也会在有关联时给出有限的章节代码候选。结果是**代码候选**，不是“这段代码实现了该论文”的证据。章节尚未提供 `code/` 或只有 PDF 时，不要虚构对应示例。

**讲解任务**：先看概念对应的 PDF 页，再读章节 README 与选中的源码；在回答中分开写 `本书依据`、`代码演示`、`扩展说明`，指出脚本的输入、核心运算、输出、假设、可改参数和反例。若要让学生复现，检查该脚本的依赖及耗时，运行选定脚本的正常/异常路径并记录真实输出；不能自动批量执行各章所有文件。教学代码可能依赖 torch、NumPy、外部权重或服务，测试覆盖并不等于全部脚本可运行。

**新代码任务**：先定位最贴近需求的章节示例和 README，阅读实际函数与数据契约，把它当作参考而非直接复制生产实现；说明复用点、改动点、输入输出、离在线版本、超时/失败处理和 `[MOCK]` 替代部分，再为新代码写测试并实际运行。需求超出已发布章节时明确标为工程假设或扩展知识，绝不能用示例代码代替书中 PDF 的出处。

例如可对 Agent 说：「结合第 10 章 PDF 与 `two_tower_debias.py` 给我讲 Batch 负采样；先引用 PDF 页，再用代码中的得分修正做一个小实验。最后参考该示例写一个新场景的采样校正模块，标清复用和未实现的生产依赖，并测试。」

## 7. 可运行项目

| 项目 | 正常路径 | 故障路径 |
|---|---|---|
| `templates/recommendation-pipeline/` | 多路召回到重排 | timeout、过载降级 |
| `templates/two-tower-ann/` | 固定示例向量的精确排序与版本契约 | 模型/索引版本错配 |
| `templates/feature-consistency/` | point-in-time join | 样本穿越拒绝 |
| `templates/serving-reliability/` | 容量内个性化服务 | 慢依赖、过载、fallback |

每个目录的 README 给出精确运行命令。`[MOCK]` 表示教学替代，不是生产集成。

## 8. 在 Claude Code 或其他 Agent 中安装

Canonical Skills 位于 `.agents/skills/`。复制到目标项目：

```bash
python3 tools/install_skills.py --client claude --target /path/to/project
python3 tools/install_skills.py --client agents --target /path/to/project
python3 tools/install_skills.py --client all --target /path/to/project
```

安装后重新启动或新开客户端会话。自然语言示例：

- “依据这本书解释双塔负采样偏差，给出 PDF 页码。”
- “为 QPS 10000、P99 200ms 的推荐服务做系统设计。”
- “评审这个特征生产任务是否存在数据穿越。”
- “按新人路线安排本周 5 小时学习，并记录完成状态。”

Skills 只编排仓库 CLI；它们不会复制 PDF、索引或检索实现。目标项目要能访问本仓库，或直接在本仓库会话中使用。

## 9. 如何验证 AI-native 改造

```bash
python3 tools/recsys_agent.py eval
python3 tools/check_knowledge.py
python3 tools/check_skills.py
python3 tools/check_public_boundary.py
```

- `eval`：当前 88 条固定查询的 Hit@1、Recall@5、引用有效率；默认要求 Hit@1 = 100%，可用 `--min-hit-at-1` 显式调整；
- `check_knowledge.py`：20 章知识地图覆盖、PDF 文件和页码边界；
- `check_skills.py`：Skill 元数据和 canonical 目录；
- `check_public_boundary.py`：阻止 LaTeX、提取缓存和正文等价 Markdown 进入公开仓库。

“比以前好用”不能只看新增文件数。[`evals/usability.md`](../evals/usability.md) 定义冻结任务、旧/新系统对照、盲评量表和报告格式。当前能自动证明的是检索与引用；学习计划质量、任务完成时间、答案正确性和工程行动性需要按该协议进行人评，不能用 Recall@5 替代。

正式旧/新系统盲评可复现地准备和汇总：

```bash
python3 tools/usability_eval.py prepare \
  --new-revision <committed-revision> \
  --model-runtime '<model-and-runtime>' \
  --system-prompt AGENTS.md \
  --output /tmp/recsys-usability

python3 tools/usability_eval.py summarize \
  --reviews /tmp/recsys-usability/reviews.jsonl \
  --assignment /tmp/recsys-usability/private/assignment.json
```

`prepare` 在独立 worktree 构建旧版索引，冻结 revision、模型/运行时和提示词哈希，并为每项任务生成随机化 A/B 记录。`private/assignment.json` 只交给实验管理员；评审者只看 A/B。`summarize` 强制每个答案至少两名不同评审、0–2 分量表和首次成功时间；任一维度两位评审分差大于 1 时要求第三人裁决。工作区改造未提交时不能作为 `--new-revision`，避免不可复现实验。

本仓库当前一次完整执行记录见 [`ai-native-validation-report.md`](ai-native-validation-report.md)。报告区分自动验证、冻结任务预检和仍待双盲完成的人评结论。

## 10. 内容与隐私边界

公开仓库允许发布 PDF、独立撰写的导航元数据、代码、测试、模板、Agent 指令和 Skills。禁止提交：

- LaTeX、私有草稿、作者编辑记录或源文件映射；
- 正文的 Markdown/纯文本转写；
- PDF 提取文本、`.recsys-agent/` 或个人 `.recsys-workspace/`；
- 未匿名化的公司、用户、内部系统、真实流量或故障信息。
