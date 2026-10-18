# 检索与引用评测

`retrieval.jsonl` 含 72 条中英文高频查询。每条指定允许命中的章节；评测同时检查 Top-1、Recall@5 和 PDF 页码引用有效性。

```bash
python3 tools/recsys_agent.py eval
```

退出码非零表示至少一条 Recall@5 失败或引用无效。查询必须对应真实读者意图，不得仅为迎合当前排序结果而改写。PDF 更新、别名或排序逻辑变化后都应运行。
