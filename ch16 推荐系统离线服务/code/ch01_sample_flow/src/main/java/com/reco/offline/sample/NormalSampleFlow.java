package com.reco.offline.sample;

import java.util.*;
import com.reco.offline.sample.SampleFlowTypes.*;

/**
 * 普通任务样本流 — 完整可运行示例
 *
 * 数据流：
 *   RawRequestWithFeatures (Kafka: raw_request) -> 特征预处理消费者 (FeaturePreprocessConsumer)
 *           -> [Kafka: sample_join_waiting]  (FeatureReadyRequest，含 slotId/sign 整数对)
 *   ActionLog (Kafka: action_log) -> 反馈标签消费者 (FeedbackLabelConsumer) -> [Redis: fb_occ / fb_amt]
 *   FeatureReadyRequest (Kafka: sample_join_waiting) -> 样本拼接消费者 -> [Kafka: training_sample_stream]
 *
 * [MOCK] Redis 和 Kafka 均使用内存模拟实现；
 *        工业落地时请替换为真实 Jedis / KafkaProducer / KafkaConsumer 客户端。
 */
public class NormalSampleFlow {

    // ─────────────────────────────────────────────
    //  核心数据结构
    // ─────────────────────────────────────────────

    /**
     * ActionLog：用户行为日志（Kafka 消息体）。
     * 统一输入来源：反馈标签消费者、序列服务、计数消费者均消费此消息。
     */
    public static class ActionLog {
        public final long userId;
        public final String deviceId;
        public final List<String> requestIds;
        public final List<Long> itemIds;
        public final List<Long> authorIds;
        public final List<Long> actionValues;
        public final String actionName;
        public final String requestType;
        public final String scenarioId;
        public final long eventTimeMs;

        public ActionLog(long userId, String deviceId, List<String> requestIds,
                         List<Long> itemIds, List<Long> authorIds, List<Long> actionValues,
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

        public long firstRequestId() {
            if (requestIds == null || requestIds.isEmpty()) return 0L;
            try { return Long.parseLong(requestIds.get(0)); }
            catch (NumberFormatException e) { return 0L; }
        }
    }

    /** 用户行为事件（反馈，FeedbackStore 内部使用） */
    public static class FeedbackEvent {
        public final long requestId;
        public final long itemId;
        /** 行为类型: click / watch_time / like */
        public final String action;
        /** 数值（发生型行为置 1，数值型行为填实际值） */
        public final long value;
        public final long eventTimeMs;

        public FeedbackEvent(long requestId, long itemId, String action, long value, long eventTimeMs) {
            this.requestId = requestId;
            this.itemId = itemId;
            this.action = action;
            this.value = value;
            this.eventTimeMs = eventTimeMs;
        }
    }

    /** 单刷次训练批次（样本拼接消费者输出） */
    public static class TrainingBatch {
        public final long requestId;
        public final String scene;
        public final long eventTimeMicros;
        /** itemId -> {action -> value} */
        public final Map<Long, Map<String, Long>> labels;

        public TrainingBatch(long requestId, String scene, long eventTimeMicros,
                             Map<Long, Map<String, Long>> labels) {
            this.requestId = requestId;
            this.scene = scene;
            this.eventTimeMicros = eventTimeMicros;
            this.labels = Map.copyOf(labels);
        }

        public boolean hasValidFeedback() {
            return labels.values().stream().anyMatch(m -> !m.isEmpty());
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] Redis 模拟实现
    //  工业落地时替换为 Jedis / Lettuce 客户端
    // ─────────────────────────────────────────────

    public static class MockRedis {
        // 发生型行为: key -> Set<action>
        private final Map<String, Set<String>> bitmaps = new HashMap<>();
        // 数值型行为: key -> value
        private final Map<String, Long> counters = new HashMap<>();
        // 时间戳: key -> ms
        private final Map<String, Long> timestamps = new HashMap<>();

        public void setOccur(long reqId, long itemId, String action) {
            String key = "fb_occ_" + reqId + "_" + itemId;
            bitmaps.computeIfAbsent(key, k -> new HashSet<>()).add(action);
        }

        public Set<String> getOccur(long reqId, long itemId) {
            return bitmaps.getOrDefault("fb_occ_" + reqId + "_" + itemId, Collections.emptySet());
        }

        public void incrby(long reqId, long itemId, String action, long delta) {
            String key = "fb_amt_" + reqId + "_" + itemId + "_" + action;
            counters.merge(key, Math.max(0, delta), Long::sum);
        }

        public long getAmount(long reqId, long itemId, String action) {
            return counters.getOrDefault("fb_amt_" + reqId + "_" + itemId + "_" + action, 0L);
        }

        public void setTimestamp(long reqId, long ts) {
            timestamps.put("fb_ts_" + reqId, ts);
        }

        public long getTimestamp(long reqId) {
            return timestamps.getOrDefault("fb_ts_" + reqId, 0L);
        }

        public void clear(long reqId, List<Long> itemIds) {
            for (long itemId : itemIds) {
                bitmaps.remove("fb_occ_" + reqId + "_" + itemId);
            }
            timestamps.remove("fb_ts_" + reqId);
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] Kafka 模拟实现
    //  工业落地时替换为 KafkaProducer / KafkaConsumer
    // ─────────────────────────────────────────────

    public static class MockKafka {
        private final Map<String, List<Object>> topics = new HashMap<>();

        @SuppressWarnings("unchecked")
        public <T> void send(String topic, T message) {
            topics.computeIfAbsent(topic, k -> new ArrayList<>()).add(message);
        }

        @SuppressWarnings("unchecked")
        public <T> List<T> poll(String topic) {
            List<Object> msgs = topics.getOrDefault(topic, Collections.emptyList());
            List<T> result = new ArrayList<>((List<T>) msgs);
            msgs.clear();
            return result;
        }
    }

    // ─────────────────────────────────────────────
    //  特征预处理消费者
    //  对应书中代码 code_request_feature_preprocess
    // ─────────────────────────────────────────────

    /**
     * Kafka 消费者：消费原始请求特征日志（RawRequestWithFeatures），
     * 完成过滤、特征裁剪与哈希编码后，将 FeatureReadyRequest 输出到 sample_join_waiting Topic。
     * FeatureReadyRequest 保留 requestId/deviceId/sessionId/scene 等关键元数据，
     * 特征部分已编码为 slotId/sign 整数对，不再包含原始字符串特征。
     */
    public static class FeaturePreprocessConsumer {
        private final SampleFlowService.FeaturePreprocessService preprocessService;
        private final MockKafka kafka;

        public FeaturePreprocessConsumer(SampleFlowService.FeaturePreprocessService preprocessService,
                                         MockKafka kafka) {
            this.preprocessService = preprocessService;
            this.kafka = kafka;
        }

        public void consume(RawRequestWithFeatures rawRequest) {
            FeatureReadyRequest result = preprocessService.preprocess(rawRequest);
            if (result != null) {
                // 直接把编码完成的 FeatureReadyRequest 发到等待 Topic
                // 下游样本拼接消费者消费此消息，无需再次访问原始特征
                kafka.send("sample_join_waiting", result);
            }
        }
    }

    // ─────────────────────────────────────────────
    //  反馈暂存层
    //  对应书中代码 code_ordinary_feedback_store 内层
    // ─────────────────────────────────────────────

    public static class FeedbackStore {
        private static final long BASE_WAIT_MS  = 30_000L;
        private static final long MAX_WAIT_MS   = 120_000L;
        private static final long ACTIVE_EXT_MS = 10_000L;

        private final MockRedis redis;

        public FeedbackStore(MockRedis redis) {
            this.redis = redis;
        }

        /**
         * 写入一条用户反馈。
         * 发生型行为（click/like）用 bitmap 记录；
         * 数值型行为（watch_time）用计数器累加。
         */
        public void record(FeedbackEvent event) {
            if (event.action == null || event.action.isBlank()) return;

            redis.setOccur(event.requestId, event.itemId, event.action);

            if (isNumeric(event.action)) {
                redis.incrby(event.requestId, event.itemId, event.action, event.value);
            }

            redis.setTimestamp(event.requestId, event.eventTimeMs);
        }

        public Set<String> getActions(long reqId, long itemId) {
            return redis.getOccur(reqId, itemId);
        }

        public long getValue(long reqId, long itemId, String action) {
            long v = redis.getAmount(reqId, itemId, action);
            return v > 0 ? v : 1L;
        }

        public long firstKeyFeedbackTime(long reqId) {
            return redis.getTimestamp(reqId);
        }

        public long lastFeedbackTime(long reqId) {
            return redis.getTimestamp(reqId);
        }

        public boolean shouldWait(FeatureReadyRequest request, long nowMs, int replayCount) {
            long firstTs = firstKeyFeedbackTime(request.getRequestId());
            long lastTs  = lastFeedbackTime(request.getRequestId());

            long windowMs = firstTs > 0 ? BASE_WAIT_MS : MAX_WAIT_MS;
            long deadline = request.getRequestTimeMs() + windowMs;
            long activeDeadline = lastTs > 0 ? lastTs + ACTIVE_EXT_MS : 0;

            return nowMs < Math.max(deadline, activeDeadline) && replayCount < 5;
        }

        public void clear(long reqId, List<Long> itemIds) {
            redis.clear(reqId, itemIds);
        }

        private boolean isNumeric(String action) {
            return action.equals("watch_time");
        }
    }

    // ─────────────────────────────────────────────
    //  反馈标签消费者（Kafka ActionLog 消费者）
    //  对应书中代码 code_ordinary_feedback_store 的外层框架
    // ─────────────────────────────────────────────

    /**
     * 消费 ActionLog Kafka 消息，通过 requestId + itemId 索引写入反馈存储。
     * 发生型行为写入 Bitmap，数值型行为累加计数器；
     * 两侧均更新最近反馈时间戳，供等待窗口判断使用。
     */
    public static class FeedbackLabelConsumer {
        private final FeedbackStore feedbackStore;

        public FeedbackLabelConsumer(FeedbackStore feedbackStore) {
            this.feedbackStore = feedbackStore;
        }

        public void consume(ActionLog actionLog) {
            long reqId = actionLog.firstRequestId();
            if (reqId <= 0 || actionLog.itemIds.isEmpty()) return;
            for (int i = 0; i < actionLog.itemIds.size(); i++) {
                long itemId = actionLog.itemIds.get(i);
                long value = i < actionLog.actionValues.size()
                        ? actionLog.actionValues.get(i) : 1L;
                feedbackStore.record(new FeedbackEvent(
                        reqId, itemId, actionLog.actionName, value, actionLog.eventTimeMs));
            }
        }
    }

    // ─────────────────────────────────────────────
    //  样本拼接消费者
    //  对应书中代码 code_ordinary_sample_join / code_ordinary_wait_window
    // ─────────────────────────────────────────────

    /**
     * 消费 FeatureReadyRequest（来自 sample_join_waiting Topic）。
     * 等待窗口内写回队列；窗口结束后按 itemId 逐一拼接反馈，输出 TrainingBatch。
     */
    public static class SampleJoinConsumer {
        private final FeedbackStore feedbackStore;
        private final MockKafka kafka;

        // 重投次数计数（实际系统中放在消息元数据里）
        private final Map<Long, Integer> replayCount = new HashMap<>();

        public SampleJoinConsumer(FeedbackStore feedbackStore, MockKafka kafka) {
            this.feedbackStore = feedbackStore;
            this.kafka = kafka;
        }

        public void consume(FeatureReadyRequest request) {
            long nowMs = System.currentTimeMillis();
            int rc = replayCount.getOrDefault(request.getRequestId(), 0);

            if (feedbackStore.shouldWait(request, nowMs, rc)) {
                replayCount.put(request.getRequestId(), rc + 1);
                kafka.send("sample_join_waiting", request);
                return;
            }

            joinAndEmit(request);
        }

        private void joinAndEmit(FeatureReadyRequest request) {
            Map<Long, Map<String, Long>> allLabels = new HashMap<>();
            boolean hasValid = false;

            for (long itemId : request.getItemIds()) {
                Set<String> actions = feedbackStore.getActions(request.getRequestId(), itemId);
                Map<String, Long> labelMap = new HashMap<>();
                for (String action : actions) {
                    labelMap.put(action, feedbackStore.getValue(request.getRequestId(), itemId, action));
                    hasValid = true;
                }
                allLabels.put(itemId, labelMap);
            }

            if (!hasValid) {
                System.out.println("[SampleJoinConsumer] drop_no_feedback scene=" + request.getScene());
                return;
            }

            TrainingBatch batch = new TrainingBatch(
                request.getRequestId(),
                request.getScene(),
                request.getRequestTimeMs() * 1000L,
                allLabels
            );
            kafka.send("training_sample_stream", batch);
            feedbackStore.clear(request.getRequestId(), request.getItemIds());
            System.out.println("[SampleJoinConsumer] output requestId=" + request.getRequestId()
                    + " items=" + request.getItemIds().size());
        }
    }

    // ─────────────────────────────────────────────
    //  级联训练批次
    // ─────────────────────────────────────────────

    /** 级联样本拼接输出：曝光 item 含真实 label，未曝光候选只含阶段信息 */
    public static class CascadeTrainingBatch {
        public final long requestId;
        public final String scene;
        public final long eventTimeMicros;
        /** 曝光 item: itemId -> {action -> value}（有真实反馈） */
        public final Map<Long, Map<String, Long>> exposedLabels;
        /** 未曝光候选: itemId -> stage label */
        public final List<CandidateRecord> candidates;

        public CascadeTrainingBatch(long requestId, String scene, long eventTimeMicros,
                                    Map<Long, Map<String, Long>> exposedLabels,
                                    List<CandidateRecord> candidates) {
            this.requestId = requestId;
            this.scene = scene;
            this.eventTimeMicros = eventTimeMicros;
            this.exposedLabels = Map.copyOf(exposedLabels);
            this.candidates = List.copyOf(candidates);
        }

        public boolean hasExposedFeedback() {
            return exposedLabels.values().stream().anyMatch(m -> !m.isEmpty());
        }

        public int totalItemCount() { return exposedLabels.size() + candidates.size(); }
    }

    /** 未曝光候选的阶段记录 */
    public static class CandidateRecord {
        public final long itemId;
        public final SampleFlowTypes.ItemStage stage;
        public final int rank;
        public final float score;

        public CandidateRecord(long itemId, SampleFlowTypes.ItemStage stage, int rank, float score) {
            this.itemId = itemId; this.stage = stage; this.rank = rank; this.score = score;
        }
    }

    // ─────────────────────────────────────────────
    //  级联样本拼接消费者
    //  对应书中代码 code_cascade_join
    // ─────────────────────────────────────────────

    /**
     * 消费同一个 sample_join_waiting Topic 中的 FeatureReadyRequest。
     * 与 SampleJoinConsumer 共享等待窗口逻辑；区别在于额外读取
     * cascadeCandidates，将其展开为 item list 并写入独立的
     * cascade_training_sample_stream Topic。
     *
     * 曝光 item（itemIds 中出现）拼接真实用户反馈；
     * 未曝光候选（cascadeCandidates）只填阶段/排序/分数，不读取反馈。
     */
    public static class CascadeSampleJoinConsumer {
        private final FeedbackStore feedbackStore;
        private final MockKafka kafka;

        private final Map<Long, Integer> replayCount = new HashMap<>();

        public CascadeSampleJoinConsumer(FeedbackStore feedbackStore, MockKafka kafka) {
            this.feedbackStore = feedbackStore;
            this.kafka = kafka;
        }

        public void consume(SampleFlowTypes.FeatureReadyRequest request) {
            // 没有级联候选，直接跳过（避免占用普通样本流的输出）
            if (request.cascadeCandidates.isEmpty()) return;

            long nowMs = System.currentTimeMillis();
            int rc = replayCount.getOrDefault(request.getRequestId(), 0);

            if (feedbackStore.shouldWait(request, nowMs, rc)) {
                replayCount.put(request.getRequestId(), rc + 1);
                kafka.send("sample_join_waiting", request);
                return;
            }

            joinAndEmit(request);
        }

        private void joinAndEmit(SampleFlowTypes.FeatureReadyRequest request) {
            // ── 曝光 item：拼接真实反馈 ──
            Map<Long, Map<String, Long>> exposedLabels = new HashMap<>();
            boolean hasExposedFeedback = false;

            for (long itemId : request.getItemIds()) {
                Set<String> actions = feedbackStore.getActions(request.getRequestId(), itemId);
                Map<String, Long> labelMap = new HashMap<>();
                for (String action : actions) {
                    labelMap.put(action, feedbackStore.getValue(request.getRequestId(), itemId, action));
                    hasExposedFeedback = true;
                }
                exposedLabels.put(itemId, labelMap);
            }

            if (!hasExposedFeedback) {
                System.out.println("[CascadeSampleJoinConsumer] drop_no_exposed_feedback scene="
                        + request.getScene());
                return;
            }

            // ── 未曝光候选：只填阶段信息，不读反馈 ──
            List<CandidateRecord> candidates = new ArrayList<>();
            for (SampleFlowTypes.EncodedStagedItem c : request.cascadeCandidates) {
                candidates.add(new CandidateRecord(
                        c.getItemId(), c.getStage(), c.getRank(), c.getScore()));
            }

            CascadeTrainingBatch batch = new CascadeTrainingBatch(
                    request.getRequestId(),
                    request.getScene(),
                    request.getRequestTimeMs() * 1000L,
                    exposedLabels,
                    candidates
            );
            kafka.send("cascade_training_sample_stream", batch);
            feedbackStore.clear(request.getRequestId(), request.getItemIds());
            System.out.println("[CascadeSampleJoinConsumer] output requestId=" + request.getRequestId()
                    + " exposed=" + request.getItemIds().size()
                    + " candidates=" + candidates.size());
        }
    }

    // ─────────────────────────────────────────────
    //  驱动示例 — 模拟一次完整的样本拼接流程
    // ─────────────────────────────────────────────

    public static TrainingBatch runDemo() {
        MockRedis redis = new MockRedis();
        MockKafka kafka = new MockKafka();
        FeedbackStore store = new FeedbackStore(redis);
        FeaturePreprocessConsumer preprocessConsumer =
                new FeaturePreprocessConsumer(new SampleFlowService.FeaturePreprocessService(), kafka);
        FeedbackLabelConsumer labelConsumer = new FeedbackLabelConsumer(store);
        SampleJoinConsumer joiner = new SampleJoinConsumer(store, kafka);

        long now = System.currentTimeMillis();

        // 1. 原始请求日志到达 → 特征预处理消费者 → FeatureReadyRequest 写入 sample_join_waiting
        RawRequestWithFeatures rawReq = new RawRequestWithFeatures(
                1001L, now - 500_000L, "dev_A", "sess_1", "home_feed", "v1",
                List.of(RawFeature.of("device_type", "android"),
                        RawFeature.of("age_bucket", 25.0)),
                List.of(new RawItem(501L, List.of(RawFeature.of("category", "sports"))),
                        new RawItem(502L, List.of(RawFeature.of("category", "music"))),
                        new RawItem(503L, List.of(RawFeature.of("category", "news")))));
        preprocessConsumer.consume(rawReq);

        // 2. 反馈 ActionLog 到达 → 反馈标签消费者 → 写入 Redis
        labelConsumer.consume(new ActionLog(
                100L, "dev_A", List.of("1001"),
                List.of(501L), List.of(201L), List.of(1L),
                "click", "home_feed", "home", now - 60_000));
        labelConsumer.consume(new ActionLog(
                100L, "dev_A", List.of("1001"),
                List.of(502L), List.of(202L), List.of(30_000L),
                "watch_time", "home_feed", "home", now - 50_000));

        // 3. 样本拼接消费者从 sample_join_waiting 消费 FeatureReadyRequest（等待窗口已过）
        List<FeatureReadyRequest> waiting = kafka.poll("sample_join_waiting");
        System.out.println("waiting queue size: " + waiting.size());
        for (FeatureReadyRequest req : waiting) {
            joiner.consume(req);
        }

        // 4. 读取训练样本
        List<TrainingBatch> batches = kafka.poll("training_sample_stream");
        System.out.println("training samples produced: " + batches.size());
        if (!batches.isEmpty()) {
            TrainingBatch b = batches.get(0);
            System.out.println("  requestId=" + b.requestId + " labels=" + b.labels);
            return b;
        }
        return null;
    }

    public static void main(String[] args) {
        System.out.println("=== ch01 普通样本流 Demo ===");
        TrainingBatch result = runDemo();
        if (result != null && result.hasValidFeedback()) {
            System.out.println("SUCCESS: 训练样本已产出");
        } else {
            System.out.println("WARN: 未产出有效训练样本");
        }
    }
}
