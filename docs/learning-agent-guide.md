# 如何用好推荐系统学习 Agent

本指南面向 Claude Code、Codex 等仓库 Agent 用户。目标不是让 Agent 替你总结 PDF，而是让它管理“阅读证据 → 推导 → 实践 → 验收 → 反馈 → 复习”的闭环。

## 1. 先建立可复核的知识源

```bash
python3 tools/recsys_agent.py index
python3 tools/recsys_agent.py doctor
python3 tools/recsys_agent.py eval
```

`doctor` 必须显示 `index=ready`。PDF 是权威正文；`knowledge/` 仅负责导航；`.recsys-agent/` 是可删除的本地提取缓存。向 Agent 提问时要求它区分：

- `本书依据`：发布 PDF 的章节和 PDF 页码；
- `扩展说明`：不在本书中的经验、论文或推断；
- `[MOCK]`：教学模板中的替代依赖，不得声称可直接上线。

推荐提示词：

```text
先检索发布 PDF，再回答。给出 PDF 文件和 PDF 页码；
把本书依据与扩展说明分开；不确定时说明边界，不要补写书中没有的内容。
```

## 2. 选择完整课程

```bash
python3 tools/recsys_agent.py course list
python3 tools/recsys_agent.py course start new-engineer  # 校招入职/初级工程师
# 或
python3 tools/recsys_agent.py course start graduate      # 研究生
```

两门课程各 12 个单元，覆盖 19 个已发布章节。人读入口是 `learning-paths/*.md`；机器可读课程在 `learning-paths/courses.json`。课程单元包含章节、概念、指定阅读、实践命令、提交物、验收条件和检索/应用题。

## 3. 每次只生成一个可完成任务

```bash
python3 tools/recsys_agent.py course today --minutes 90
```

输出按主题特征分配阅读、推导、实践和验收时间。实践型主题会给实践更多时间；实验设计型主题会给推导更多时间。不要让 Agent 一次生成整周长文；让它依次做：

1. 根据 `reading` 打开 PDF 指定页，提取问题相关证据；
2. 回答 `questions`，先独立作答再让 Agent 纠错；
3. 运行 `practice` 主路径，保存命令、退出码和关键输出；
4. 修改参数或注入故障，运行失败路径；
5. 按 `acceptance` 检查提交物。

可直接给 Agent：

```text
执行 course today 给出的当前单元。先让我回答问题，不要直接泄露答案；
然后检查我的推理。实践必须真实运行主路径和失败路径，把命令、输出、原因、恢复条件写进提交物。
```

## 4. 提交真实证据，而不是口头“学完了”

建议提交物模板：

```markdown
# 单元 ID
## 本书依据
- PDF 文件、PDF 第 N 页、支持的结论
## 我的解释
- 核心机制、目标、取舍和适用边界
## 实践
- 环境与版本
- 主路径命令：`python3 ...`
- 主路径退出码：0
- 主路径输出：关键输出
- 失败路径命令：`python3 ... --inject-failure`
- 失败路径退出码：2
- 失败路径输出：关键拒绝或降级输出
- 失败原因：实际观察到的故障机制
- 恢复条件：重新满足的契约
## 问题答案
- 逐题回答
## 仍不确定
- 概念或工程约束
```

提交：

```bash
python3 tools/recsys_agent.py course submit \
  --course new-engineer --unit ne-01 \
  --evidence /tmp/ne-01.md --score 0.85 \
  --practice-passed --failure-path
```

`submit` 和 `complete` 会检查上述章节、PDF 页码及主/失败路径的命令、退出码、输出、原因和恢复条件；缺项仅记 `assessed`。这只是结构门禁，无法证明输出真实或答案正确，仍需 Agent 阅读提交物并按单元验收条件评分。`--score`、`--practice-passed`、`--failure-path` 均为学习者声明。

状态语义：

```mermaid
stateDiagram-v2
    [*] --> started: course start / today
    started --> assessed: 提交或测验
    assessed --> started: 低于 0.8 或证据不全
    assessed --> completed: 提交物 + 主路径 + 失败路径 + score≥0.8
    completed --> review_due: 7 天后
    review_due --> completed: 延迟复习通过
```

`--practice-passed` 和 `--failure-path` 是学习者声明；提交物应保留可复核输出。Agent 必须读取提交物并按单元 `acceptance` 评分，不能仅因文件存在就给高分。

## 5. 随时反馈，下一次再调整

```bash
python3 tools/recsys_agent.py course feedback \
  --message "负采样偏差仍不清楚；本周只有 45 分钟；希望增加手算例子"
python3 tools/recsys_agent.py course today --minutes 45
```

反馈应写具体事实：可用时间、卡住的问题、实践输出、错误理解、资源约束。不要只写“太难”。Agent 下一轮应：

- 保留当前单元，不因时间不足跳过验收；
- 针对低分题安排重读和小实验；
- 已掌握部分减少重复；
- 只推荐前置已满足的下一主题。

查看状态和下一主题：

```bash
python3 tools/recsys_agent.py course status
python3 tools/recsys_agent.py next
python3 tools/recsys_agent.py graph --course new-engineer --output /tmp/new-engineer.mmd
```

个人记录只写入被 Git 忽略的 `.recsys-workspace/progress.jsonl`，不得提交真实公司数据、用户信息或内部事故详情。

## 6. 做延迟复习和迁移测试

完成单元 7 天后：

```bash
python3 tools/recsys_agent.py course review
```

复习不再重复原答案，而是完成四类任务：

1. 不看资料解释核心机制；
2. 找出一个错误设计；
3. 修改一个参数并预测结果；
4. 迁移到新业务场景并说明取舍。

这用于区分短期记忆与真正掌握。复习提交物需分别写出 `## 机制回忆`、`## 错误设计`、`## 参数预测`、`## 迁移取舍` 四节。Agent 核查内容后提交评分：

```bash
python3 tools/recsys_agent.py course review --course new-engineer \
  --unit ne-01 --evidence /tmp/ne-01-review.md --score 0.85
```

通过后该次到期任务不再重复出现；低分或缺项仍保留待复习。CLI 只能检查四节结构，不能自动判定理解正确。



## 7. 生成自己的可复核报告

```bash
python3 tools/recsys_agent.py self-eval \
  --output .recsys-workspace/self-eval.json
```

报告包含固定检索集 Recall@5、引用有效率、已完成记录和到期复习数。它不等于教学效果证明；课程完成率、首次成功时间、一周后保持率和迁移任务仍需真实学习者数据。

若要评价仓库新版 Agent 是否优于旧版，按 `evals/usability.md` 做冻结 revision、相同模型/提示词、独立会话和盲评，不使用当前学习记录替代 A/B 证据。

## 8. 何时使用自由主题模式

课程外临时学习一个主题：

```bash
python3 tools/recsys_agent.py learn "HNSW" --level graduate --minutes 120
```

完成时仍需要证据：

```bash
python3 tools/recsys_agent.py complete hnsw \
  --evidence /tmp/hnsw.md --score 0.9 \
  --practice-passed --failure-path
```

自由模式适合补缺，不替代完整课程。若前置概念不满足，先运行 `next` 或回到课程当前单元。

## 9. Agent 的边界

Agent 可以检索、拆任务、检查结构化验收条件、运行仓库模板和记录反馈；它不能仅凭一份文件可靠判断你是否真正理解，也不能替代生产数据、线上实验、隐私审查和容量压测。对生产工程问题，必须补齐目标、QPS、P99、规模、新鲜度、一致性、运行时、部署、可用性、隐私和成本约束。

## 10. 用 coach 做基于证据的针对性追问

在仓库根目录先运行 `python3 tools/recsys_agent.py index`。开始课程后，先完成当前单元自己的解释和实践，再让陪练针对提交物提出追问：

```bash
python3 tools/recsys_agent.py course start new-engineer
python3 tools/recsys_agent.py course today --minutes 45
python3 tools/recsys_agent.py coach --course new-engineer --evidence /tmp/ne-01.md \
  --feedback "我不确定反馈闭环如何影响训练数据"
# 若本机默认 Claude 模型不可用，添加 --model <可用模型ID>；不调用模型则添加 --no-model。
```

提交物可按四节组织：`本书依据`（PDF 文件和页码）、`我的解释`、`实践`（主路径与失败路径的命令、退出码、输出、失败原因与恢复条件）、`问题答案`。`coach` 会结合课程问题和本地已发布 PDF 候选页，指出最多两处需核对的概念或推理缺口，先给追问而非直接揭晓答案；下次调用同课程当前单元时读取上一轮任务，继续调整。只有反馈而无提交物也可用，但会明确要求补交证据。若反馈明确命中已编目的其他单元概念而不属于当前单元，返回 `topic_mismatch`，不运行语义诊断、不写入陪练状态；可用 `ask` / `navigate` 自由学习，或先完成当前单元。未被概念目录识别的跨主题反馈仍需学习者自行判断。

不希望模型看到提交物时使用 `--no-model`，只进行本地结构检查。模型反馈是待核对的学习提示，不会自动评分、运行提交物中的命令、证明输出真实或宣告掌握。正式课程完成仍须人工/可信 Agent 核对，再用 `course submit` 显式提交；七天后用 `course review` 做延迟回忆和迁移。

陪练状态追加到 Git 忽略的 `.recsys-workspace/coach.jsonl`；只包含状态、反馈和 evidence 文件路径，不保存 PDF 摘录或答案正文。`ask --session` 的主题也仅存于本机忽略目录。勿在提交物或反馈中放凭据和内部数据；模型调用会使用本机配置的服务，可能产生费用。

底层函数 `recsys_coach.coach(root, course_id, evidence=None, feedback="", semantic=False, model=None)` 可供其他客户端调用。无模型时仍会返回 `diagnosis`、`follow_up`、`next_tasks`、课程验收条件与 PDF 阅读导航，但结构完整不等于答案正确。
