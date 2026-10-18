# AI-native 可用性评测

## 目的

回答两个不同问题：

1. **能力是否存在并可运行**：CLI、知识地图、Skills、项目和安全边界可自动验证。
2. **读者是否比改造前更高效、更正确**：必须用冻结任务和盲评验证；不能用文件数、功能数或检索 Recall 代替。

## 对照系统

- **旧系统**：Git revision `1af34d036fa22f21e69b254bb96949ed28b28156`。只允许该 revision 的 README、发布 PDF 与当时的 `tools/recsys_agent.py`。
- **新系统**：待评 revision。允许 README、发布 PDF、`knowledge/`、`learning-paths/`、CLI、Skills、cases 和 templates。
- 两组使用同一模型版本、系统提示、机器、时间上限和任务文本；清空会话，不共享旧组答案。
- PDF 与题目冻结后再评测。调参不能查看隐藏任务结果。

旧系统的已有 `.recsys-agent/index.jsonl` 与当前结构不兼容，因此不能复用缓存。复现实验应在独立 worktree 中执行旧版 `index` 后测试，避免把新索引误当旧能力。

使用仓库工具冻结实验资产：

```bash
python3 tools/usability_eval.py prepare \
  --new-revision <committed-revision> \
  --model-runtime '<model-and-runtime>' \
  --system-prompt AGENTS.md \
  --output /tmp/recsys-usability
```

管理员保管 `private/assignment.json` 和 `private/runs/`；评审者只收到去除 `system`、`revision` 等身份字段后的 A/B 答案。评分写入 `reviews.jsonl`，字段为 `task_id`、`label`、`reviewer`、五个量表分和 `first_success_seconds`。汇总命令：

```bash
python3 tools/usability_eval.py summarize \
  --reviews /tmp/recsys-usability/reviews.jsonl \
  --assignment /tmp/recsys-usability/private/assignment.json
```

## 冻结任务

| ID | 角色 | 任务 | 成功证据 | 上限 |
|---|---|---|---|---:|
| Q1 | 新人工程师 | 解释双塔负采样偏差，并给出书中来源 | 结论正确；至少一个有效 PDF 页码；区分扩展 | 10 分钟 |
| Q2 | 研究生 | 比较 HNSW 与 PQ 的适用条件 | 覆盖 Recall/延迟/内存/更新；引用有效 | 12 分钟 |
| L1 | 新人工程师 | 制定 5 小时召回学习计划 | 前置、阅读页、实践、验收均可执行 | 12 分钟 |
| L2 | 研究生 | 制定时长建模研究计划 | 假设、强基线、偏差、消融、复现条件完整 | 15 分钟 |
| D1 | 校招生 | 为 `QPS=10k, P99=200ms, 1e8 items` 设计在线链路 | 容量、预算、降级、版本、监控、灰度、回滚 | 20 分钟 |
| G1 | 工程师 | 诊断“召回 P99 上升且空结果率增加” | 先取证；覆盖供给、依赖、版本、过载和降级 | 15 分钟 |
| R1 | 工程师 | 评审一个含未来特征的样本任务 | 找到穿越；解释时间语义；给出修复与验证 | 15 分钟 |
| P1 | 新人工程师 | 运行服务并证明过载仍非空 | 实际命令、输出、故障原因和恢复条件 | 15 分钟 |

任务中的设计输入、代码样本和故障夹具在评测开始前冻结；评测者不能临时修改任务迎合当前实现。

## 评分量表

每项 0–2 分，总分 10：

| 维度 | 0 | 1 | 2 |
|---|---|---|---|
| 正确性 | 关键结论错误 | 大体正确但有实质遗漏 | 正确且边界清楚 |
| 证据 | 无来源或伪引用 | 有来源但页码/对应关系弱 | 有效 PDF 页码且可复核 |
| 完整性 | 未完成核心任务 | 完成主路径 | 主路径、约束和失败路径完整 |
| 行动性 | 泛泛建议 | 有步骤但不可直接执行 | 命令、产出和验收明确 |
| 边界安全 | 泄露/混淆权威来源 | 边界表达不稳定 | 本书、扩展、MOCK、隐私边界明确 |

工程任务另记录“是否实际运行”和“首次成功时间”；未实际运行时，行动性最高 1 分。两名评审者独立盲评；分差超过 1 的维度由第三人裁决。

## 自动指标

```bash
python3 tools/recsys_agent.py eval
python3 tools/check_knowledge.py
python3 tools/check_skills.py
python3 tools/check_public_boundary.py
```

记录：

- 检索：Hit@1、Recall@5、有效 PDF 引用率；
- 导航：发布 19 章覆盖率、引用页码合法率；
- Skills：canonical Skill 数和 lint；
- 安全：公开边界通过/失败；
- 工程：四个 template 的正常路径与指定故障路径是否实际成功。

## 当前可复现结果

评测日期：2026-10-18。数据集：`evals/retrieval.jsonl` 72 条。

| 指标 | 旧系统 | 新系统 | 解释 |
|---|---:|---:|---|
| 固定 72 查询可执行 | 否 | 是 | 旧 revision 没有该评测集或 `eval` 命令 |
| Hit@1 | N/A | 100% | 新系统实测 72/72 |
| Recall@5 | N/A | 100% | 新系统实测 72/72 |
| 有效 PDF 引用率 | N/A | 100% | Top 结果均含发布 PDF 页码 |
| 章节知识地图覆盖 | 0/19 | 19/19 | 文件、PDF 存在且引用页码范围合法 |
| 可执行课程 | 0 | 2 | 新人、研究生各 8 周 |
| Canonical Skills lint | N/A | 7/7 | 元数据和目录名合法 |
| 公开内容边界 | 未覆盖新资产 | 通过 | 未发现 LaTeX、提取缓存或越界 Markdown |
| 四个项目正常路径 | N/A | 4/4 | 实际运行均产生预期结构化输出 |
| 已运行故障路径 | N/A | 5/5 | 漏斗过载、版本错配、样本穿越、服务过载、依赖超时均按契约降级或拒绝 |

这里的 `N/A` 不等于 0 分：旧系统没有相同协议，无法诚实计算同口径指标。仓库已提供独立旧版 worktree、冻结控制、A/B 隐藏映射和双评审汇总工具；真正的“改造前后好用程度”结论仍须完成 8 个任务、两名独立评审及必要裁决后填写。

## 报告模板

```text
revision:
model/runtime:
task_id:
start/end:
commands_run:
actual_output_evidence:
score_correctness_0_2:
score_evidence_0_2:
score_completeness_0_2:
score_actionability_0_2:
score_boundary_0_2:
reviewer_notes:
```

发布结论至少报告每个任务的两组得分、首次成功时间、中位数、失败原因和评审一致率。若只完成自动指标，应写“检索/边界验证通过”，不得写“读者效率已提升”。
