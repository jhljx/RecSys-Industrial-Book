# 双塔训练与 ANN 服务项目

## 场景
用双塔向量产物构建检索服务，并保证模型与索引版本原子切换。代码使用 `[MOCK]` 小数据，但版本契约和失败路径真实可运行。

## 运行与验收

```bash
python3 templates/two-tower-ann/app.py --user u1 --limit 3
python3 templates/two-tower-ann/app.py --inject-version-mismatch  # 预期拒绝并退出 2
```

验收：正样本排在检索前列；负样本不含正样本；版本不匹配不可提供结果。生产替换需加入真实训练、批量导出、ANN 基准、灰度切换、指标和回滚。

## 可观测指标
Recall@K、索引构建耗时、查询 P99、QPS、内存、版本拒绝数。

## 扩展任务
加入困难负样本、量化索引及新旧版本双读对比。

## 本书依据
运行 `python3 tools/recsys_agent.py search "双塔负采样向量召回"` 和 `search "ANN 向量召回系统架构"` 获取发布 PDF 页码。
