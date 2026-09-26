-- sample_join.sql
-- 基于 Hive SQL 的非实时样本拼接
-- 对应书中 code_hive_sample_join
--
-- [MOCK] 以下 SQL 在 Hive/Spark SQL 环境下可运行。
-- 实际任务还需明确日期分区、Label 去重规则，
-- 以及同一 Request ID 出现多个特征版本时的选版策略。

WITH latest_label AS (
    SELECT request_id, item_id, click, watch_time_ms, event_time_ms
    FROM (
        SELECT request_id, item_id, click, watch_time_ms, event_time_ms,
            ROW_NUMBER() OVER (
                PARTITION BY request_id, item_id
                ORDER BY event_time_ms DESC) AS rn
        FROM dwd_item_feedback
        WHERE dt BETWEEN '${start_dt}' AND '${end_dt}'
    ) t
    WHERE rn = 1
)
INSERT OVERWRITE TABLE dws_training_sample PARTITION (dt='${biz_dt}')
SELECT f.request_id, f.item_id, f.user_features, f.context_features,
    f.item_features, f.request_time_ms,
    l.click, l.watch_time_ms, l.event_time_ms AS label_time_ms
FROM dwd_request_features f
JOIN latest_label l
    ON f.request_id = l.request_id AND f.item_id = l.item_id
WHERE f.dt = '${biz_dt}';
