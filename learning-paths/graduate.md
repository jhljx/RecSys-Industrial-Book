# 研究生建模与可信实验路线

这是一门 12 周研究课程，不是论文题目清单。权威正文仍是发布 PDF；完整机器可读课程、阅读页、研究产物、问题和验收条件在 [`courses.json`](courses.json) 的 `graduate` 中。

## 目标

围绕“可证伪、可复现、可部署”建立研究闭环：问题定义 → 数据生成机制 → 强基线 → 模型与消融 → 偏差和不确定性 → 系统约束 → 安全实验 → 迁移复现。

课程持续覆盖 `knowledge/chapters/` 中全部已发布章节；新增章节必须同步进入本课程及新人课程，并通过概念引用和 PDF 页码检查。生成式推荐、蒸馏、偏差校正、时长分布建模和强化学习都必须与传统强基线在同一数据与预算下比较。

## 使用

```bash
python3 tools/recsys_agent.py course start graduate
python3 tools/recsys_agent.py course today --minutes 120
python3 tools/recsys_agent.py course feedback --message "实验数据只能做时间切分，GPU 预算为单卡"
```

提交必须包含配置、版本、实际输出、失败案例和问题答案：

```bash
python3 tools/recsys_agent.py course submit \
  --course graduate --unit gr-01 \
  --evidence /tmp/gr-01.md --score 0.9 \
  --practice-passed --failure-path
python3 tools/recsys_agent.py course status
```

分数低于 0.8 或证据不全时不会完成。结论必须区分本书依据、实验观察和推断。完整工作流见 [`../docs/learning-agent-guide.md`](../docs/learning-agent-guide.md)。
