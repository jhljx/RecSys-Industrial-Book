# AI-native 改造验证报告

## 1. 结论

本次验证依据 [`docs/agent-guide.md`](agent-guide.md) 第 8 节和 [`evals/usability.md`](../evals/usability.md) 执行。

- **自动验证通过**：72 条固定检索全部首位命中，Hit@1、Recall@5 和有效 PDF 引用率均为 100%；19 章知识地图、7 个 canonical Skills 和公开内容边界全部通过。
- **四个教学项目通过运行验证**：4/4 正常路径成功；5/5 指定故障路径按契约降级或拒绝；服务过载和依赖超时时均保持 100/100 非空返回。
- **六项预检问题已修复**：学习预算守恒与 `graduate` 参数兼容；“时长建模”解析到 `duration-modeling`；`design`、`debug`、`review` 输出可机读完整报告；固定 72 查询 Hit@1 从 65/72 提升至 72/72；盲评准备/汇总工具可运行。
- **正式人评仍未完成**：当前没有 8 个任务的真实旧/新答案、两名独立盲评者评分和首次成功时间，因此不得声称“读者效率已提升”。

评测日期：2026-10-18。

## 2. 被测版本与环境

```text
repository: RecSys-Industrial-Book
base revision: 1af34d036fa22f21e69b254bb96949ed28b28156
working tree: 包含本轮尚未提交的 AI-native 改造
runtime: Python 3.12.7
system: Darwin 27.0.0 arm64
pdftotext: /opt/homebrew/bin/pdftotext
knowledge mode: public
PDF sources: 19
index chunks: 808
```

由于新系统改造仍位于该 revision 之上的工作区，报告以 base revision 加当前工作区文件作为被测对象，不把 revision 哈希误写成新系统的独立提交版本。

## 3. 自动指标

执行命令：

```bash
python3 tools/recsys_agent.py index
python3 tools/recsys_agent.py doctor
python3 tools/recsys_agent.py eval
python3 tools/check_knowledge.py
python3 tools/check_skills.py
python3 tools/check_public_boundary.py
```

实测结果：

| 指标 | 结果 | 判定 |
|---|---:|---|
| PDF 来源 | 19 | 完整 |
| 索引状态 | `ready` | 通过 |
| 固定查询 | 72 | 全部可执行 |
| Hit@1 | 72/72（100%） | 通过 |
| Recall@5 | 72/72（100%） | 通过 |
| 有效 PDF 引用率 | 72/72（100%） | 通过 |
| 章节知识地图 | 19/19 | 通过 |
| Canonical Skills | 7/7 | 通过 |
| 公开内容边界 | 无违规项 | 通过 |

原始摘要：

```text
indexed mode=public sources=19 chunks=808
index=ready
{"cases":72,"hit_at_1":1.0,"recall_at_5":1.0,"citation_valid":1.0,"failures":[]}
knowledge check passed: 19 chapter maps with valid PDF citations
skill lint passed: 7 canonical skills
privacy check passed: public files match the source and Markdown boundary
```

自动指标只能证明检索、引用、元数据和边界可运行，不能替代冻结任务的人评。

## 4. 可运行项目验证

### 4.1 推荐漏斗

执行：

```bash
python3 templates/recommendation-pipeline/app.py --user u1 --limit 5
python3 templates/recommendation-pipeline/app.py --user u1 --limit 5 --simulate-overload
python3 -m unittest discover -s templates/recommendation-pipeline/tests -v
```

证据：

- 正常路径返回 `item-1`、`item-2`、`item-3`，`degraded=false`。
- 过载路径返回 5 个热门候选，`degraded=true`、`reason=overload`。
- 合并去重、过载兜底、超时后不启动后续阶段三个测试全部通过。

### 4.2 双塔与 ANN 版本契约

执行：

```bash
python3 templates/two-tower-ann/app.py --user u1 --limit 3
python3 templates/two-tower-ann/app.py --inject-version-mismatch
```

证据：正常路径返回 `status=ok`、`version=v1`；错配路径返回 `status=rejected`、`model/index version mismatch`，退出码为 2。

### 4.3 特征时间一致性

执行：

```bash
python3 templates/feature-consistency/app.py
python3 templates/feature-consistency/app.py --inject-leakage
```

证据：正常样本均满足 `feature_time <= label_time`；注入未来特征后返回 `future features detected: 1`，退出码为 2。

### 4.4 在线服务稳定性

执行：

```bash
python3 templates/serving-reliability/app.py --requests 100 --capacity 80
python3 templates/serving-reliability/app.py --requests 100 --capacity 80 --dependency-latency-ms 100
```

| 场景 | 个性化 | fallback | 非空 | fallback 率 | P99 |
|---|---:|---:|---:|---:|---:|
| 容量 80、请求 100 | 80 | 20 | 100 | 20% | 20 ms |
| 依赖延迟 100 ms | 0 | 100 | 100 | 100% | 5 ms |

项目汇总：正常路径 4/4；指定故障路径 5/5（漏斗过载、模型/索引错配、样本穿越、服务过载、依赖超时）。这些实现均含 `[MOCK]` 教学依赖，不构成生产集成证明。

## 5. 冻结任务修复复验

本节仍是工作流预检，不冒充 `evals/usability.md` 要求的正式双盲评分。

| ID | 修复后实际执行与证据 | 结果 |
|---|---|---|
| L1 | `learn "双塔召回" --level beginner --minutes 4` | 四步各 1 分钟，总和严格等于预算；小预算无负数或超配 |
| L2 | `learn "时长建模" --level graduate --minutes 300` | `graduate` 映射为 `advanced`；`concept_id=duration-modeling`；步骤覆盖强基线、偏差、消融和复现条件，分钟数 `75+105+75+45=300` |
| D1 | `design` 输入 objective、QPS、P99、items、freshness、availability、cost | `status=complete`；请求链路、分阶段 P99、`minimum_inflight_at_p99=2000`、容量公式、版本、降级、监控、灰度和回滚均结构化输出；未实测的单副本容量仍列为 unresolved |
| G1 | `debug "召回 P99 上升且空结果率增加"` | `status=complete`；输出假设、所需证据、未确认结论、修复和恢复验证，不臆断根因 |
| R1 | `review templates/feature-consistency/app.py` | `status=complete`；定位 point-in-time 条件和拒绝门禁，给出风险、修复与可杀死错误的验证要求 |
| 检索 | `python3 tools/recsys_agent.py eval` | 固定题目未改；Hit@1、Recall@5、引用有效率均为 100% |

## 6. 盲评工作流复验

新增 `tools/usability_eval.py`：

- `prepare` 校验 old/new commit，在临时独立 worktree 构建旧版索引并记录 SHA-256；冻结模型/运行时、提示词哈希和 8 项任务；为每项任务生成 A/B 隐藏映射及 16 个隔离记录。
- `summarize` 只接受 A/B 标签，强制每个答案至少两名不同评审、五维 0–2 分和非负首次成功时间；分差超过 1 且缺第三评审时拒绝发布结论。
- 实际 smoke 已观察：空评审集返回 `status=incomplete`、列出 16 个缺失双评审单元并退出 1；32 条合成协议夹具返回 `status=complete`、一致率 1.0。合成夹具只验证汇总逻辑，不是人评结果。

正式结论仍缺少：将当前工作区提交为可复现 `new_revision`，执行 16 次隔离任务，收集两名独立评审及必要裁决。

## 7. 可发布结论边界

本轮可以发布的结论：

> 新系统固定 72 查询的 Hit@1、Recall@5 和有效 PDF 引用率均为 100%；知识地图、Skills lint、公开边界和四个教学项目均可运行；六项预检缺陷已修复并完成对应 smoke。

本轮不能发布的结论：

> “相对旧系统，读者完成任务更快、答案更正确或工程行动性更强。”

该结论仍缺少真实旧/新任务答案、两名独立盲评者评分、首次成功时间和必要裁决。
