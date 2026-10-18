# Agent 使用指南

本仓库把发布 PDF 作为权威正文，把 `knowledge/` 作为导航层，把命令行工具和 Skills 作为工作流层。无需 MCP、数据库或云端知识库。

## 1. 五分钟开始

环境：Python 3.10+；索引需要系统命令 `pdftotext`。在仓库根目录执行：

```bash
python3 tools/recsys_agent.py index
python3 tools/recsys_agent.py doctor
python3 tools/recsys_agent.py search "双塔召回的负采样偏差"
python3 tools/recsys_agent.py page --chapter 10 --page 60
```

`doctor` 应显示 `index=ready`。索引写入被忽略的 `.recsys-agent/`；PDF 更新后重新执行 `index`。所有命令可通过以下入口发现：

```bash
python3 tools/recsys_agent.py --help
python3 tools/recsys_agent.py search --help
```

## 2. 命令总览

| 命令 | 用途 | 典型输出 |
|---|---|---|
| `index` | 构建或刷新本地 PDF 索引 | 分块数与 manifest |
| `doctor` | 检查依赖、知识源和索引新鲜度 | `index=ready/stale` |
| `search` | 混合检索 PDF | 文件、PDF 页、节、摘录 |
| `page` | 阅读一个命中页 | 该 PDF 页的本地提取文本 |
| `eval` | 运行检索与引用基准 | Hit@1、Recall@5、引用有效率 |
| `learn` | 生成自由主题限时学习计划 | 前置、阅读页、主题化时间预算、实践与验收 |
| `complete` | 用提交物、实践、失败路径和测验分数完成自由主题 | `assessed/completed` 状态 |
| `course` | 运行完整课程学习闭环 | 今日任务、反馈、提交、进度与延迟复习 |
| `next` | 按前置与掌握状态推荐下一主题 | 概念 ID 列表 |
| `graph` | 输出全局或课程 Mermaid 知识图谱 | 节点与依赖边 |
| `self-eval` | 生成读者本地自测报告 | 检索、引用、学习与复习状态 |
| `explain` | 按受众组织解释 | 解释结构与 PDF 证据 |
| `design` | 生成可机读系统设计报告 | 链路、P99 预算、容量、稳定性、版本与发布契约 |
| `review` | 生成带位置的领域评审报告 | 发现、风险、修复与验证要求 |
| `debug` | 生成证据优先诊断报告 | 假设、证据、结论、修复与恢复验证 |

注意：文档中的 `/learn`、`/design` 等表示 Agent 工作流名称，不是 shell 命令。没有安装客户端 Slash Command；终端中必须写完整 CLI，如 `python3 tools/recsys_agent.py learn ...`。在 Claude Code 等客户端中，用自然语言要求 Agent 执行对应 Skill 即可。

## 3. 查书与引用

```bash
# 普通检索
python3 tools/recsys_agent.py search "HNSW efSearch 如何影响召回和延迟"

# 限定章节和返回数量
python3 tools/recsys_agent.py search "负采样" --chapter 10 --limit 3

# 供脚本消费
python3 tools/recsys_agent.py search "Embedding 版本" --json

# 命中后读取完整页
python3 tools/recsys_agent.py page --chapter 17 --page 17
```

检索结果是发现证据，不是最终答案。公式、表格、限定条件或相邻上下文重要时，必须执行 `page` 或打开对应 PDF 页面。回答格式：

- `本书依据`：只陈述发布 PDF 支持的内容并标注文件和 PDF 页；
- `扩展说明`：工程经验、外部资料或推导；
- 找不到时明确说发布章节未覆盖，不补造书中结论。

章节级学习入口是 [`knowledge/chapters/README.md`](../knowledge/chapters/README.md)；概念入口是 [`knowledge/README.md`](../knowledge/README.md)。章节地图不是正文替代品。

## 4. 学习工作流

完整课程从一次初始化开始：

```bash
python3 tools/recsys_agent.py course list
python3 tools/recsys_agent.py course start new-engineer  # 或 graduate
python3 tools/recsys_agent.py course today --minutes 90
python3 tools/recsys_agent.py course feedback --message "本周只有 45 分钟；负采样校正仍不清楚"
python3 tools/recsys_agent.py course status
```

两门 12 周课程覆盖全部 19 个已发布章节：

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

证据不全或分数低于 0.8 时只记为 `assessed`，不会进入 `completed`。七天后 `course review` 安排回忆、纠错、参数预测和迁移任务。课程外补缺可运行：

```bash
python3 tools/recsys_agent.py learn "HNSW" --level graduate --minutes 120
python3 tools/recsys_agent.py complete hnsw --evidence /tmp/hnsw.md \
  --score 0.9 --practice-passed --failure-path
python3 tools/recsys_agent.py next
```

详细的读者提示词、证据模板、反馈方法、状态语义和隐私边界见 [`learning-agent-guide.md`](learning-agent-guide.md)。个人状态只写入被 Git 忽略的 `.recsys-workspace/`。

## 5. 工程工作流

### 设计

```bash
python3 tools/recsys_agent.py design "在线推荐服务" \
  objective=watch_time qps=10000 p99=200ms items=1e8 \
  freshness=5m availability=99.99% cost=bounded
```

输出 JSON 包含请求链路、数值 P99 预算、最小并发量、容量公式、版本契约、过载/降级、观测、灰度、回滚、来源和仍待确认约束。给出 `qps` 与 `p99` 时会计算 `minimum_inflight_at_p99 = ceil(qps × p99_seconds)`；单副本安全 QPS 等实测参数不会臆造。

### 评审

```bash
python3 tools/recsys_agent.py review templates/recommendation-pipeline/app.py
```

输出 JSON 包含精确文件位置、风险、建议修复、可杀死变异的验证要求和 PDF 来源。静态检查只识别仓库内显式模式，不能替代运行验证。

### 排障

```bash
python3 tools/recsys_agent.py debug "精排 CTR 突降"
```

输出 JSON 包含有序假设、逐项取证结果、结论、修复、恢复验证和 PDF 来源；没有实际遥测时会明确保留未确认假设，不把症状猜成根因。

开发交付必须区分 `本书依据`、`工程实现`、`验证` 和 `待业务确认`。

## 6. 可运行项目

| 项目 | 正常路径 | 故障路径 |
|---|---|---|
| `templates/recommendation-pipeline/` | 多路召回到重排 | timeout、过载降级 |
| `templates/two-tower-ann/` | 训练玩具双塔并查询 ANN | 模型/索引版本错配 |
| `templates/feature-consistency/` | point-in-time join | 样本穿越拒绝 |
| `templates/serving-reliability/` | 容量内个性化服务 | 慢依赖、过载、fallback |

每个目录的 README 给出精确运行命令。`[MOCK]` 表示教学替代，不是生产集成。

## 7. 在 Claude Code 或其他 Agent 中安装

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

## 8. 如何验证 AI-native 改造

```bash
python3 tools/recsys_agent.py eval
python3 tools/check_knowledge.py
python3 tools/check_skills.py
python3 tools/check_public_boundary.py
```

- `eval`：72 条固定查询的 Hit@1、Recall@5、引用有效率；
- `check_knowledge.py`：19 章知识地图覆盖、PDF 文件和页码边界；
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

## 9. 内容与隐私边界

公开仓库允许发布 PDF、独立撰写的导航元数据、代码、测试、模板、Agent 指令和 Skills。禁止提交：

- LaTeX、私有草稿、作者编辑记录或源文件映射；
- 正文的 Markdown/纯文本转写；
- PDF 提取文本、`.recsys-agent/` 或个人 `.recsys-workspace/`；
- 未匿名化的公司、用户、内部系统、真实流量或故障信息。
