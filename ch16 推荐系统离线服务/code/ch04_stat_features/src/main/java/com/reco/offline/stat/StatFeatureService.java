package com.reco.offline.stat;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 统计特征生产服务 — 完整可运行示例
 *
 * 包含：
 *  1. ActionLog              - 用户行为日志（Kafka 消息体）
 *  2. CounterEventConsumer   - Kafka 消费者骨架（书中 code_counter_event_writer）
 *  3. TimeBucketCounter        - 时间分桶计数器（书中 code_user_counter_writer / code_item_counter_writer）
 *  4. ItemCounterSnapshotService - 物品快照服务（书中 code_counter_snapshot_service）
 *  5. EmpiricalXtrService      - 经验 CTR/VTR（贝叶斯平滑）（书中 code_counter_snapshot_service）
 *  6. CounterIndexTrigger      - 计数变化触发 Kafka（书中 code_counter_index_trigger）
 *
 * [MOCK] TimeBucketCounter 使用 HashMap，工业落地时替换为 Redis HASH + TTL 管理。
 * [MOCK] Kafka 写使用内存队列，工业落地时替换为 KafkaProducer。
 */
public class StatFeatureService {

    // ─────────────────────────────────────────────
    //  ActionLog：用户行为日志（Kafka 消息体）
    // ─────────────────────────────────────────────

    public static class ActionLog {
        public final long userId;
        public final String deviceId;
        public final List<String> requestIds;
        public final List<Long> itemIds;
        /** 与 itemIds 一一对应的作者 ID；长度可短于 itemIds，缺位视为无作者。 */
        public final List<Long> authorIds;
        public final List<Long> actionValues;
        public final String actionName;
        /** 请求类型，如 "home_feed" / "search" / "live_feed" 等。 */
        public final String requestType;
        /** 场景 ID，用于区分不同推荐场景的计数命名空间。 */
        public final String scenarioId;
        public final long eventTimeMs;

        public ActionLog(long userId, String deviceId, List<String> requestIds,
                         List<Long> itemIds, List<Long> authorIds,
                         List<Long> actionValues,
                         String actionName, String requestType, String scenarioId,
                         long eventTimeMs) {
            this.userId = userId;
            this.deviceId = deviceId;
            this.requestIds = requestIds != null ? requestIds : List.of();
            this.itemIds = itemIds != null ? itemIds : List.of();
            this.authorIds = authorIds != null ? authorIds : List.of();
            this.actionValues = actionValues != null ? actionValues : List.of();
            this.actionName = actionName;
            this.requestType = requestType != null ? requestType : "";
            this.scenarioId = scenarioId != null ? scenarioId : "";
            this.eventTimeMs = eventTimeMs;
        }
    }

    // ─────────────────────────────────────────────
    //  数据结构
    // ─────────────────────────────────────────────

    public static class CounterEvent {
        public final long subjectId;
        /** "user" or "item" */
        public final String subjectType;
        /** 行为类型: "click" / "show" / "like" */
        public final String actionName;
        /** 行为数值（发生型为 1，数值型为实际值） */
        public final long delta;
        public final long eventTimeMs;

        public CounterEvent(long subjectId, String subjectType, String actionName,
                            long delta, long eventTimeMs) {
            this.subjectId = subjectId;
            this.subjectType = subjectType;
            this.actionName = actionName;
            this.delta = delta;
            this.eventTimeMs = eventTimeMs;
        }
    }

    public static class ItemStatistics {
        public final long itemId;
        public final Map<String, Long> showsByBucket;   // bucket_hours -> count
        public final Map<String, Long> clicksByBucket;
        public final double empiricalCtr;
        public final double empiricalVtr;

        public ItemStatistics(long itemId, Map<String, Long> showsByBucket,
                              Map<String, Long> clicksByBucket,
                              double empiricalCtr, double empiricalVtr) {
            this.itemId = itemId;
            this.showsByBucket = Map.copyOf(showsByBucket);
            this.clicksByBucket = Map.copyOf(clicksByBucket);
            this.empiricalCtr = empiricalCtr;
            this.empiricalVtr = empiricalVtr;
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] 时间分桶计数器
    //  工业落地时替换为 Redis HASH + 定时清理
    // ─────────────────────────────────────────────

    public static class TimeBucketCounter {
        /** 支持的时间窗口（小时） */
        private static final int[] WINDOWS_HOURS = {1, 6, 24, 72, 168};

        // key = subjectId + ":" + actionName + ":" + bucketHours, value = count
        private final Map<String, Long> counters = new ConcurrentHashMap<>();
        // 去重 Set: key = subjectId + ":" + actionName + ":" + dedupId + ":" + bucketHours
        private final Map<String, Boolean> dedupSets = new ConcurrentHashMap<>();

        /**
         * 增加计数（含对重复 ID 的去重）
         * @param subjectId   主体 ID（用户或物品）
         * @param actionName  行为名称
         * @param delta       增量
         * @param dedupId     去重 ID（如 requestId，-1 表示不去重）
         * @param eventTimeMs 事件时间
         */
        public void increment(long subjectId, String actionName, long delta,
                              long dedupId, long eventTimeMs) {
            Instant eventTime = Instant.ofEpochMilli(eventTimeMs);
            Instant now = Instant.now();
            for (int windowHours : WINDOWS_HOURS) {
                Instant cutoff = now.minus(windowHours, ChronoUnit.HOURS);
                if (eventTime.isBefore(cutoff)) continue; // 超出窗口跳过

                String counterKey = subjectId + ":" + actionName + ":" + windowHours;

                // 去重检查
                if (dedupId >= 0) {
                    String dedupKey = subjectId + ":" + actionName + ":" + dedupId + ":" + windowHours;
                    if (dedupSets.putIfAbsent(dedupKey, Boolean.TRUE) != null) {
                        continue; // 已处理过
                    }
                }

                counters.merge(counterKey, delta, Long::sum);
            }
        }

        public long get(long subjectId, String actionName, int windowHours) {
            return counters.getOrDefault(subjectId + ":" + actionName + ":" + windowHours, 0L);
        }

        public Map<String, Long> getAll(long subjectId, String actionName) {
            Map<String, Long> result = new LinkedHashMap<>();
            for (int w : WINDOWS_HOURS) {
                result.put(w + "h", get(subjectId, actionName, w));
            }
            return result;
        }
    }

    // ─────────────────────────────────────────────
    //  Kafka 消费者骨架
    //  对应书中 code_counter_event_writer
    // ─────────────────────────────────────────────

    /**
     * 统计特征 Kafka 消费者 — 对应 tex 代码片段 code_counter_event_writer。
     * 消费用户 Action Log，实时更新用户侧和物品侧窗口计数。
     */
    public static class CounterEventConsumer {
        private final TimeBucketCounter counter;
        private final UserCounterWriter userWriter;
        private final ItemCounterWriter itemWriter;

        public CounterEventConsumer(TimeBucketCounter counter) {
            this.counter = counter;
            this.userWriter = new UserCounterWriter(counter);
            this.itemWriter = new ItemCounterWriter(counter);
        }

        public void consume(ActionLog actionLog) {
            if (actionLog == null || actionLog.actionName == null) return;
            long eventTimeMs = actionLog.eventTimeMs;
            for (int i = 0; i < actionLog.itemIds.size(); i++) {
                long itemId = actionLog.itemIds.get(i);
                long reqId  = firstRequestId(actionLog.requestIds);
                long delta  = i < actionLog.actionValues.size()
                        ? actionLog.actionValues.get(i) : 1L;
                long authorId = i < actionLog.authorIds.size()
                        ? actionLog.authorIds.get(i) : -1L;
                // 用户侧计数（按 userId，不按 requestId 去重）
                if (actionLog.userId > 0) {
                    userWriter.process(new CounterEvent(
                            actionLog.userId, "user", actionLog.actionName, delta, eventTimeMs));
                }
                // 物品侧计数（按 requestId 去重，同一次刷新只计一次）
                itemWriter.process(new CounterEvent(
                        itemId, "item", actionLog.actionName, delta, eventTimeMs), reqId, authorId);
            }
        }

        private long firstRequestId(List<String> ids) {
            if (ids == null || ids.isEmpty()) return -1L;
            try { return Long.parseLong(ids.get(0)); } catch (NumberFormatException e) { return -1L; }
        }
    }

    // ─────────────────────────────────────────────
    //  用户侧计数器写入
    //  对应书中 code_user_counter_writer
    // ─────────────────────────────────────────────

    public static class UserCounterWriter {
        private final TimeBucketCounter counter;

        public UserCounterWriter(TimeBucketCounter counter) {
            this.counter = counter;
        }

        public void process(CounterEvent event) {
            if (event == null || event.subjectType == null
                    || !event.subjectType.equals("user")) return;
            if (event.delta <= 0) return;
            counter.increment(event.subjectId, event.actionName,
                    event.delta, -1L /* 用户侧不去重 */, event.eventTimeMs);
        }
    }

    // ─────────────────────────────────────────────
    //  物品侧计数器写入（含 requestId 级去重）
    //  对应书中 code_item_counter_writer
    // ─────────────────────────────────────────────

    public static class ItemCounterWriter {
        private final TimeBucketCounter counter;

        public ItemCounterWriter(TimeBucketCounter counter) {
            this.counter = counter;
        }

        public void process(CounterEvent event, long requestId, long authorId) {
            if (event == null || event.subjectType == null
                    || !event.subjectType.equals("item")) return;
            if (event.delta <= 0) return;
            counter.increment(event.subjectId, event.actionName,
                    event.delta, requestId /* 物品侧按 requestId 去重 */,
                    event.eventTimeMs);
            if (authorId > 0) {
                // 作者侧：以 authorId 为主体键累加，requestId 复用同一去重标记
                counter.increment(authorId, "author_" + event.actionName,
                        event.delta, requestId, event.eventTimeMs);
            }
        }
    }

    // ─────────────────────────────────────────────
    //  经验 XTR（贝叶斯平滑）
    //  书中 ExperienceXtrService 的简化实现
    // ─────────────────────────────────────────────

    public static class EmpiricalXtrService {
        private final double ctrPrior;
        private final double vtrPrior;

        public EmpiricalXtrService(double ctrPrior, double vtrPrior) {
            if (ctrPrior <= 0 || vtrPrior <= 0)
                throw new IllegalArgumentException("prior must be positive");
            this.ctrPrior = ctrPrior;
            this.vtrPrior = vtrPrior;
        }

        /**
         * 贝叶斯平滑 CTR = (click + alpha) / (show + alpha / ctrPrior)
         * 其中 alpha = ctrPrior * beta，beta 为等效样本数（此处固定 beta=10）
         */
        public double smoothedCtr(long shows, long clicks) {
            double beta = 10.0;
            double alpha = ctrPrior * beta;
            return (clicks + alpha) / (shows + beta);
        }

        public double smoothedVtr(long shows, long views) {
            double beta = 10.0;
            double alpha = vtrPrior * beta;
            return (views + alpha) / (shows + beta);
        }

        public Map<Long, ItemStatistics> load(Collection<Long> itemIds,
                                               TimeBucketCounter counter) {
            Map<Long, ItemStatistics> result = new HashMap<>();
            for (long itemId : itemIds) {
                Map<String, Long> shows  = counter.getAll(itemId, "show");
                Map<String, Long> clicks = counter.getAll(itemId, "click");
                Map<String, Long> views  = counter.getAll(itemId, "view");

                long show24h  = shows.getOrDefault("24h", 0L);
                long click24h = clicks.getOrDefault("24h", 0L);
                long view24h  = views.getOrDefault("24h", 0L);

                double eCtr = smoothedCtr(show24h, click24h);
                double eVtr = smoothedVtr(show24h, view24h);

                result.put(itemId, new ItemStatistics(itemId, shows, clicks, eCtr, eVtr));
            }
            return result;
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] 物品计数快照服务
    //  对应书中 code_counter_snapshot_service
    // ─────────────────────────────────────────────

    public static class ItemCounterSnapshotService {
        private final TimeBucketCounter counter;
        private final EmpiricalXtrService xtrService;

        public ItemCounterSnapshotService(TimeBucketCounter counter,
                                          double ctrPrior, double vtrPrior) {
            if (counter == null) throw new IllegalArgumentException("counter must not be null");
            this.counter = counter;
            this.xtrService = new EmpiricalXtrService(ctrPrior, vtrPrior);
        }

        public Map<Long, ItemStatistics> load(Collection<Long> itemIds) {
            return xtrService.load(itemIds, counter);
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] 计数索引触发器
    //  对应书中 code_counter_index_trigger
    // ─────────────────────────────────────────────

    public static class CounterIndexTrigger {
        private final List<String> kafkaMessages; // [MOCK]，工业落地时替换为 KafkaProducer

        public CounterIndexTrigger(List<String> kafkaMessages) {
            this.kafkaMessages = kafkaMessages;
        }

        public void trigger(long itemId, String actionName) {
            String msg = "item_counter_update:" + itemId + ":" + actionName
                    + ":ts=" + System.currentTimeMillis();
            kafkaMessages.add(msg);
        }
    }

    // ─────────────────────────────────────────────
    //  端到端 Demo
    // ─────────────────────────────────────────────

    public static Map<Long, ItemStatistics> runDemo() {
        TimeBucketCounter counter = new TimeBucketCounter();
        // 使用 CounterEventConsumer 消费 ActionLog（对应书中 code_counter_event_writer）
        CounterEventConsumer eventConsumer = new CounterEventConsumer(counter);
        List<String> kafkaMsgs = new ArrayList<>();
        CounterIndexTrigger trigger = new CounterIndexTrigger(kafkaMsgs);
        ItemCounterSnapshotService snapshotService =
                new ItemCounterSnapshotService(counter, 0.02, 0.5);

        long now = System.currentTimeMillis();

        // 通过 ActionLog 模拟用户与物品的行为事件
        long[] items = {501L, 502L, 503L};
        long[] authors = {601L, 602L, 603L}; // 物品对应的作者
        for (int idx = 0; idx < items.length; idx++) {
            long itemId = items[idx];
            long authorId = authors[idx];
            // 多次 show
            for (int i = 0; i < 100; i++) {
                eventConsumer.consume(new ActionLog(0L, "dev_A",
                        List.of(String.valueOf(10000L + i)),
                        List.of(itemId), List.of(authorId),
                        List.of(1L), "show", "home_feed", "home", now - i * 60_000));
            }
            // 少量 click
            int clicks = (int) (itemId - 500);
            for (int c = 0; c < clicks; c++) {
                long reqId = 20000L + itemId * 10 + c;
                eventConsumer.consume(new ActionLog(100L, "dev_A",
                        List.of(String.valueOf(reqId)),
                        List.of(itemId), List.of(authorId),
                        List.of(1L), "click", "home_feed", "home", now - c * 60_000));
                trigger.trigger(itemId, "click");
            }
        }

        // 用户行为（userId 维度）
        eventConsumer.consume(new ActionLog(100L, "dev_A", List.of("30001"),
                List.of(501L), List.of(601L), List.of(5L), "click", "home_feed", "home", now - 1000));
        eventConsumer.consume(new ActionLog(100L, "dev_A", List.of("30002"),
                List.of(501L), List.of(601L), List.of(50L), "show", "home_feed", "home", now - 2000));

        Map<Long, ItemStatistics> stats = snapshotService.load(
                Arrays.asList(501L, 502L, 503L));

        System.out.println("=== 物品统计特征 ===");
        for (var entry : stats.entrySet()) {
            ItemStatistics s = entry.getValue();
            System.out.printf("item %d: eCTR=%.4f  shows_24h=%d clicks_24h=%d%n",
                    s.itemId,
                    s.empiricalCtr,
                    s.showsByBucket.getOrDefault("24h", 0L),
                    s.clicksByBucket.getOrDefault("24h", 0L));
        }
        System.out.println("Kafka trigger messages: " + kafkaMsgs.size());
        return stats;
    }

    public static void main(String[] args) {
        System.out.println("=== ch04 统计特征生产服务 Demo ===");
        runDemo();
    }
}
