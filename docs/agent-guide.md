# Agent 使用指南

本仓库同时支持人工阅读和本地 AI Agent，但不公开 LaTeX 或与正文等价的 Markdown 源码。

## 两种模式

### 1. 基于本书的答疑

```bash
python3 tools/recsys_agent.py doctor
python3 tools/recsys_agent.py index
python3 tools/recsys_agent.py search "双塔召回的负采样偏差"
```

索引从已发布 PDF 在本机临时生成，写入 `.recsys-agent/`。该目录已被 Git 忽略，不是本书的再发行格式。Claude Code 会读取 `CLAUDE.md` 和 `.claude/skills/recsys-book-qa/SKILL.md`；Codex 等兼容 Agent 会读取 `AGENTS.md`。

回答应引用章节 PDF 与 PDF 页码，并区分“本书依据”和“扩展说明”。

### 2. 工业推荐系统开发

Agent 先从 PDF 检索本书中的设计依据，再检查配套代码或 `templates/` 下的工程模板。开发结果应区分：

- `本书依据`：来自已发布 PDF；
- `工程实现`：本次设计和代码；
- `验证`：实际运行的命令或场景；
- `待业务确认`：只有业务方能决定的约束。

开始设计前，应明确会改变方案的约束，例如 QPS、P99 延迟、候选规模、特征规模、更新时效、一致性、硬件、部署环境、可用性与成本。

## 内容边界

公开仓库允许提交 PDF、独立编写的导航元数据、代码、测试、模板、Agent 指令和 Skills。禁止提交：

- LaTeX 和私有草稿；
- 正文的 Markdown/纯文本转写；
- PDF 提取文本与 `.recsys-agent/`；
- 作者编辑记录、私有注释和源文件映射。

发布前执行：

```bash
python3 tools/check_public_boundary.py
```
