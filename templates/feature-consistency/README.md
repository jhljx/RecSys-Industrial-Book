# 特征与样本一致性项目

## 场景
按事件时间为训练标签关联当时可见的最新特征。任何未来特征进入样本均阻断产出。

## 运行与验收

```bash
python3 templates/feature-consistency/app.py
python3 templates/feature-consistency/app.py --inject-leakage  # 预期退出 2
```

验收：所有 `feature_time <= label_time`；注入穿越后门禁失败。生产替换需加入水位线、迟到数据、回填版本、快照审计与离在线特征对账。

## 可观测指标
特征年龄分位数、迟到率、缺失率、穿越行数、回填影响样本数。

## 扩展任务
增加多实体特征、允许延迟和回填重算策略。

## 本书依据
运行 `python3 tools/recsys_agent.py search "特征时效性 样本穿越"` 获取发布 PDF 页码。
