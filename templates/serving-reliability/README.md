# 在线服务稳定性项目

## 场景
在容量限制和依赖超时下验证过载降级、非空返回及尾延迟。模拟器确定性运行，适合作为容量推演起点。

## 运行与验收

```bash
python3 templates/serving-reliability/app.py --requests 100 --capacity 80
python3 templates/serving-reliability/app.py --requests 100 --capacity 80 --dependency-latency-ms 100
```

验收：`non_empty == requests`；超容量和超时请求进入 fallback；输出 P99 与降级率。生产替换需使用真实压测器、并发模型、依赖隔离、告警、灰度和回滚。

## 可观测指标
QPS、P50/P99、拒绝率、超时率、熔断状态、降级率、非空返回率。

## 扩展任务
加入突发流量、熔断恢复窗口和多级兜底。

## 本书依据
运行 `python3 tools/recsys_agent.py search "在线服务 延迟预算 限流 熔断 降级"` 获取发布 PDF 页码。
