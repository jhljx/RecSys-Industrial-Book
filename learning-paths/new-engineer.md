# 推荐算法工程师全链路学习路线

这是一门 12 周课程，不是章节摘录。权威正文仍是发布 PDF；完整机器可读课程、阅读页、提交物、问题和验收条件在 [`courses.json`](courses.json) 的 `new-engineer` 中。

## 目标

把知识串成一条可运行链路：目标与反馈 → 架构与评价 → 召回 → 粗排/精排 → 重排/混排 → 生成式推荐 → 特征与离线生产 → 在线服务 → ANN → 偏差 → 时长与探索 → 毕业项目。

课程持续覆盖 `knowledge/chapters/` 中全部已发布章节；新增章节只有在阅读地图、概念卡和本课程单元同步通过发布检查后才会进入公开仓库。

## 使用

```bash
python3 tools/recsys_agent.py course start new-engineer
python3 tools/recsys_agent.py course today --minutes 90
python3 tools/recsys_agent.py course feedback --message "双塔负采样校正仍不清楚，本周只有 45 分钟"
python3 tools/recsys_agent.py course today --minutes 45
```

完成今日任务后，把命令、输出、设计和问题答案写入一个本地文件，再提交：

```bash
python3 tools/recsys_agent.py course submit \
  --course new-engineer --unit ne-01 \
  --evidence /tmp/ne-01.md --score 0.85 \
  --practice-passed --failure-path
python3 tools/recsys_agent.py course status
```

分数低于 0.8、缺少提交物、未运行主路径或失败路径时，单元只会进入 `assessed`，不会完成。完整流程与证据模板见 [`../docs/learning-agent-guide.md`](../docs/learning-agent-guide.md)。
