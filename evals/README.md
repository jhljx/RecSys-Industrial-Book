# 检索与引用评测

`retrieval.jsonl` 当前含 89 条中英文高频查询。每条指定允许命中的章节；评测同时报告 Hit@1、Recall@5 和 PDF 页码引用有效性。题目数以文件实际记录为准，新增题目后同步更新本文。

```bash
python3 tools/recsys_agent.py eval
```

退出码非零表示 Hit@1 低于最低通过率、至少一条 Recall@5 失败，或 PDF 引用无效。默认要求 Hit@1 = 100%；需要宽松基线时可显式传入 `--min-hit-at-1 0.95`（范围 0–1），报告中的 `top1_misses` 会列出首位未命中的题目。CI 使用默认严格门槛。查询必须对应真实读者意图，不得仅为迎合当前排序结果而改写。PDF 更新、别名或排序逻辑变化后都应运行。

别名表 `knowledge/aliases.json` 按术语族组织：中文复合术语与英文缩写按原查询中的最长、互不重叠的出现位置识别；同一段文本不会同时触发短词和长词的不同术语族，分开的出现位置仍可表达混合意图。例如「协同过滤」不等于「候选过滤」，但「协同过滤召回后过滤已看内容」应识别两个意图。新增别名时应补充正反查询扩展用例，并运行上述固定检索集；不要仅靠章节加权或改写评测查询掩盖误匹配。

`unanswerable.jsonl` 收录无关主题及明确索取尚未发布章节 PDF 的负例，`expected_chapters: []` 表示应拒绝作为书中证据。执行 `python3 tools/recsys_agent.py eval --dataset evals/unanswerable.jsonl`；报告的 `negative_rejection_rate` 必须为 100%。对具体未发布章节 PDF 的请求先按已发布文件目录拒答；其他问题仍使用保守检索分数门槛。这不能证明所有书外问题都能正确拒答，尤其是与已发布主题相近的虚构结论；新增负例与歧义例应人工确认。

`page-evidence.jsonl` 是另一个独立的小型页级门禁：人工对照已发布 PDF 为查询标定 `expected_pages`（章节号 → 允许的 PDF 页范围），要求 Top-1 同时落在对应章节和页范围。执行 `python3 tools/recsys_agent.py eval --dataset evals/page-evidence.jsonl`。当前 7 条覆盖 6 个章节，仍只能证明定位页，不能证明 Agent 的最终解释正确；后续扩充歧义与未覆盖查询，并按 `evals/usability.md` 盲评答案的每条论断及其引用。
