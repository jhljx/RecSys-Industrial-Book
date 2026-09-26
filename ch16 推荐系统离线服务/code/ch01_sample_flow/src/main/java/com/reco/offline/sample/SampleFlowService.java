package com.reco.offline.sample;

import java.util.*;
import java.util.concurrent.TimeUnit;

import com.reco.offline.sample.SampleFlowTypes.*;

/**
 * 普通样本流完整服务实现 — 对应 test_chapter.tex 中 ch01 的所有代码片段。
 *
 * 包含以下 tex 代码片段的完整可运行实现：
 *   code_request_feature_preprocess  - 请求侧特征裁剪与哈希编码
 *   code_ordinary_feedback_store     - 普通样本流的反馈暂存与更新
 *   code_ordinary_wait_window        - 普通样本流的等待窗口与延迟再次投递
 *   code_ordinary_sample_join        - 普通任务样本的逐物品拼接主流程
 *   code_ordinary_history_features   - 按候选实体回溯历史反馈的可选增强层
 *   code_cascade_join                - 保留级联候选的统一样本组装
 *
 * [MOCK] Redis、Kafka 均使用内存模拟；工业落地时替换为真实客户端。
 */
public class SampleFlowService {

    // ─────────────────────────────────────────────────────────
    //  [MOCK] 依赖：Redis、Kafka 内存模拟
    // ─────────────────────────────────────────────────────────

    /** [MOCK] Redis 内存模拟。工业落地时替换为 Jedis / Lettuce 客户端。 */
    public static class MockRedisClient {
        private final Map<String, Set<Integer>> bitmaps = new HashMap<>();
        private final Map<String, Long> counters = new HashMap<>();
        private final Map<String, Long> timestamps = new HashMap<>();

        public interface Pipeline {
            void setbit(String key, int bit, boolean value);
            void expire(String key, int ttl);
            void incrby(String key, long delta);
            void setex(String key, int ttl, String value);
        }

        public void pipeline(java.util.function.Consumer<Pipeline> fn) {
            fn.accept(new Pipeline() {
                public void setbit(String key, int bit, boolean value) {
                    if (value) bitmaps.computeIfAbsent(key, k -> new HashSet<>()).add(bit);
                }
                public void expire(String key, int ttl) { /* TTL mock no-op */ }
                public void incrby(String key, long delta) {
                    counters.merge(key, Math.max(0, delta), Long::sum);
                }
                public void setex(String key, int ttl, String value) {
                    try { timestamps.put(key, Long.parseLong(value)); } catch (NumberFormatException ignored) {}
                }
            });
        }

        public void setex(String key, int ttl, String value) {
            try { timestamps.put(key, Long.parseLong(value)); } catch (NumberFormatException ignored) {}
        }

        public Set<Integer> getBits(String key) {
            return bitmaps.getOrDefault(key, Collections.emptySet());
        }

        public long getCounter(String key) {
            return counters.getOrDefault(key, 0L);
        }

        public long getTimestamp(String key) {
            return timestamps.getOrDefault(key, 0L);
        }

        public boolean exists(String key) {
            return bitmaps.containsKey(key) || counters.containsKey(key) || timestamps.containsKey(key);
        }

        /** 模拟 binaryPipeline：返回 itemId -> bits */
        public Map<Long, byte[]> binaryPipeline(List<Long> itemIds,
                java.util.function.BiConsumer<Object, Long> fn) {
            Map<Long, byte[]> result = new HashMap<>();
            for (long itemId : itemIds) {
                // fn 只是 hint；我们直接按 key 读取位图
                String key = "fb_occ__" + itemId; // 简化 mock 不含 reqId；实际实现参 MockRedisWithReqId
                Set<Integer> bits = bitmaps.get(key);
                if (bits != null && !bits.isEmpty()) {
                    int maxBit = bits.stream().mapToInt(i -> i).max().orElse(0);
                    byte[] bytes = new byte[(maxBit / 8) + 1];
                    for (int b : bits) {
                        bytes[b / 8] |= (byte) (1 << (7 - (b % 8)));
                    }
                    result.put(itemId, bytes);
                }
            }
            return result;
        }

        public String get(String key) {
            Long v = timestamps.get(key);
            return v != null ? String.valueOf(v) : null;
        }
    }

    /** [MOCK] Kafka 内存模拟。工业落地时替换为 KafkaProducer。 */
    public static class MockKafkaClient {
        private final Map<String, List<Object>> topics = new HashMap<>();

        @SuppressWarnings("unchecked")
        public <T> void send(String topic, String key, T message) {
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

    // ─────────────────────────────────────────────────────────
    //  [MOCK] 辅助依赖 stubs
    // ─────────────────────────────────────────────────────────

    public static class RequestValidator {
        public boolean isTrainable(RawRequestWithFeatures r) { return r.requestId > 0; }
    }

    public static class Deduplicator {
        private final Set<Long> seen = new HashSet<>();
        public boolean seen(long reqId) { return !seen.add(reqId); }
    }

    public static class FeatureSchemaProvider {
        private final FeatureSchema defaultSchema;
        public FeatureSchemaProvider() {
            Map<String, FeatureRule> cr = new HashMap<>();
            cr.put("device_type", new FeatureRule(1, "user", 42L, ValueType.CATEGORICAL, true));
            cr.put("age_bucket", new FeatureRule(2, "user", 43L, ValueType.NUMERIC, true));
            Map<String, FeatureRule> ir = new HashMap<>();
            ir.put("category", new FeatureRule(10, "item", 44L, ValueType.CATEGORICAL, true));
            ir.put("score", new FeatureRule(11, "item", 45L, ValueType.NUMERIC, true));
            this.defaultSchema = new FeatureSchema("v1", cr, ir);
        }
        public FeatureSchema forVersion(String version) { return defaultSchema; }
    }

    public static class PolicyProvider {
        public JoinPolicy forScene(String scene) {
            return new JoinPolicy(30_000L, 120_000L, 10_000L, 5);
        }
    }

    public static class FeedbackVersionStore {
        public void keepLatest(Object event) { /* no-op */ }
    }

    // ─────────────────────────────────────────────────────────
    //  code_request_feature_preprocess
    //  请求侧特征裁剪与哈希编码
    // ─────────────────────────────────────────────────────────

    /**
     * 请求预处理服务 — 对应 tex 代码片段 code_request_feature_preprocess。
     * 将原始请求的字符串特征哈希编码为 slotId/sign 整数对，
     * 使下游训练任务可以直接构造 Embedding 输入。
     */
    public static class FeaturePreprocessService {
        private final RequestValidator requestValidator = new RequestValidator();
        private final Deduplicator deduplicator = new Deduplicator();
        private final FeatureSchemaProvider featureSchemaProvider = new FeatureSchemaProvider();
        private final Metrics metrics = new Metrics();

        // ── tex snippet: code_request_feature_preprocess ──
        public FeatureReadyRequest preprocess(RawRequestWithFeatures request) {
            if (!requestValidator.isTrainable(request) || deduplicator.seen(request.getRequestId())) {
                metrics.count("drop_invalid_or_duplicate_request");
                return null;
            }

            FeatureSchema schema = featureSchemaProvider.forVersion(request.getFeatureSchemaVersion());
            FeatureReadyRequest.Builder output = FeatureReadyRequest.newBuilder()
                    .setRequestId(request.getRequestId())
                    .setRequestTimeMs(request.getRequestTimeMs())
                    .setDeviceId(request.getDeviceId())
                    .setSessionId(request.getSessionId())
                    .setScene(request.getScene())
                    .setFeatureSchemaVersion(schema.version());

            for (RawFeature feature : request.getCommonFeatures()) {
                EncodedFeature encoded = encode(feature, schema.commonRule(feature.name()));
                if (encoded != null) {
                    output.addCommonFeature(encoded); // slotId + sign + optional numeric value
                }
            }
            for (RawItem item : request.getItems()) {
                EncodedItem.Builder encodedItem = output.addItemBuilder().setItemId(item.getItemId());
                for (RawFeature feature : item.getFeatures()) {
                    EncodedFeature encoded = encode(feature, schema.itemRule(feature.name()));
                    if (encoded != null) {
                        encodedItem.addFeature(encoded);
                    }
                }
            }
            for (RawStagedItem candidate : request.cascadeCandidates) {
                EncodedStagedItem.Builder encodedCandidate = new EncodedStagedItem.Builder()
                        .setItemId(candidate.getItemId())
                        .setStage(candidate.getStage())
                        .setRank(candidate.getRank())
                        .setScore(candidate.getScore());
                for (RawFeature feature : candidate.getFeatures()) {
                    EncodedFeature encoded = encode(feature, schema.itemRule(feature.name()));
                    if (encoded != null) {
                        encodedCandidate.addFeature(encoded);
                    }
                }
                output.addCascadeCandidate(encodedCandidate.build());
            }
            return output.build();
        }

        private EncodedFeature encode(RawFeature feature, FeatureRule rule) {
            if (rule == null || !rule.isEnabled() || feature.isEmpty()) {
                return null;
            }
            long sign = rule.valueType() == ValueType.NUMERIC
                    ? (long) feature.numericValue()
                    : SampleFlowTypes.stableHash(rule.namespace(), feature.stringValue(), rule.hashSeed());
            return EncodedFeature.of(rule.slotId(), sign, rule.valueType());
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_ordinary_feedback_store
    //  普通样本流的反馈暂存与更新
    // ─────────────────────────────────────────────────────────

    /**
     * 反馈暂存服务 — 对应 tex 代码片段 code_ordinary_feedback_store。
     * 使用 Redis Bitmap 存储发生型反馈，使用 INCRBY 计数器存储数值型反馈。
     */
    public static class FeedbackStore {
        private static final int FEEDBACK_TTL_SECONDS = 3 * 3600;

        private final MockRedisClient redis;
        private final FeedbackVersionStore feedbackVersionStore = new FeedbackVersionStore();
        private final Metrics metrics = new Metrics();

        // [MOCK] reqId -> itemId -> actions (bitmap simulation)
        private final Map<String, Set<FeedbackKind>> bitmapState = new HashMap<>();
        // [MOCK] reqId -> itemId -> action -> value
        private final Map<String, Long> counterState = new HashMap<>();
        // [MOCK] reqId -> lastFeedbackMs
        private final Map<Long, Long> timestampState = new HashMap<>();
        // [MOCK] reqId -> itemId -> showPosition
        private final Map<String, Long> positionState = new HashMap<>();

        public FeedbackStore(MockRedisClient redis) {
            this.redis = redis;
        }

        // ── tex snippet: code_ordinary_feedback_store ──
        public void record(FeedbackEventPb event) {
            FeedbackKind kind = FeedbackKind.fromName(event.getAction());
            if (kind == null || !kind.isTrainingLabel()) {
                metrics.count("unknown_feedback", event.getAction());
                return;
            }

            String occurKey = String.format("fb:%d:%d", event.getRequestId(), event.getItemId());
            redis.pipeline(pipe -> {
                pipe.setbit(occurKey, kind.bitIndex(), true);  // true 表示将该位置设为 1
                pipe.expire(occurKey, FEEDBACK_TTL_SECONDS);
            });

            if (kind.valueMode() == ValueMode.SUM) {
                String valueKey = String.format(
                        "fbv:%d:%d:%s", event.getRequestId(), event.getItemId(), kind.name());
                redis.pipeline(pipe -> {
                    pipe.incrby(valueKey, Math.max(0, event.getValue()));
                    pipe.expire(valueKey, FEEDBACK_TTL_SECONDS);
                });
            } else if (kind.valueMode() == ValueMode.LATEST) {
                feedbackVersionStore.keepLatest(event);
            }
            redis.setex("fb_ts:" + event.getRequestId(), FEEDBACK_TTL_SECONDS,
                    String.valueOf(event.getEventTimeMs()));

            // [MOCK] also update in-memory state for query methods
            String bitKey = String.format("fb:%d:%d", event.getRequestId(), event.getItemId());
            bitmapState.computeIfAbsent(bitKey, k -> new HashSet<>()).add(kind);
            if (kind.valueMode() == ValueMode.SUM) {
                String cntKey = String.format("fbv:%d:%d:%s",
                        event.getRequestId(), event.getItemId(), kind.featureName());
                counterState.merge(cntKey, Math.max(0, event.getValue()), Long::sum);
            }
            timestampState.merge(event.getRequestId(), event.getEventTimeMs(), Math::max);
        }

        // ── batch query helpers (used by code_ordinary_sample_join) ──
        public Map<Long, Set<FeedbackKind>> batchGetActions(long requestId, List<Long> itemIds) {
            Map<Long, Set<FeedbackKind>> result = new HashMap<>();
            for (long itemId : itemIds) {
                String key = String.format("fb:%d:%d", requestId, itemId);
                Set<FeedbackKind> kinds = bitmapState.get(key);
                if (kinds != null && !kinds.isEmpty()) {
                    result.put(itemId, new HashSet<>(kinds));
                }
            }
            return result;
        }

        public Map<Long, Map<FeedbackKind, Long>> batchGetValues(
                long requestId, Map<Long, Set<FeedbackKind>> actions) {
            Map<Long, Map<FeedbackKind, Long>> result = new HashMap<>();
            for (Map.Entry<Long, Set<FeedbackKind>> e : actions.entrySet()) {
                long itemId = e.getKey();
                for (FeedbackKind kind : e.getValue()) {
                    if (kind.valueMode() == ValueMode.SUM) {
                        String cntKey = String.format("fbv:%d:%d:%s",
                                requestId, itemId, kind.featureName());
                        long v = counterState.getOrDefault(cntKey, 0L);
                        if (v > 0) {
                            result.computeIfAbsent(itemId, k -> new HashMap<>()).put(kind, v);
                        }
                    }
                }
            }
            return result;
        }

        public Map<Long, Long> batchGetShowPositions(long requestId, List<Long> itemIds) {
            Map<Long, Long> result = new HashMap<>();
            for (long itemId : itemIds) {
                String key = String.format("pos:%d:%d", requestId, itemId);
                Long pos = positionState.get(key);
                if (pos != null) result.put(itemId, pos);
            }
            return result;
        }

        public long firstKeyFeedbackTime(long requestId) {
            return timestampState.getOrDefault(requestId, 0L);
        }

        public long lastFeedbackTime(long requestId) {
            return timestampState.getOrDefault(requestId, 0L);
        }

        public void clear(long requestId, List<Long> itemIds) {
            for (long itemId : itemIds) {
                bitmapState.remove(String.format("fb:%d:%d", requestId, itemId));
            }
            timestampState.remove(requestId);
        }

        /** 记录真实展示位置 */
        public void recordShowPosition(long requestId, long itemId, long position) {
            positionState.put(String.format("pos:%d:%d", requestId, itemId), position);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_ordinary_wait_window + code_ordinary_sample_join
    //  普通样本流的等待窗口与样本拼接主流程
    // ─────────────────────────────────────────────────────────

    /**
     * 样本拼接消费者 — 对应 tex 代码片段
     *   code_ordinary_wait_window  （shouldWait / consume）
     *   code_ordinary_sample_join  （joinAndEmit）
     */
    public static class SampleJoinService {
        private final FeedbackStore feedbackStore;
        private final PolicyProvider policyProvider = new PolicyProvider();
        private final MockKafkaClient kafka;
        private final Metrics metrics = new Metrics();
        /** 可选的历史特征增强服务；null 表示跳过增强逻辑 */
        private HistoryFeatureService historyFeatureService;
        /** deviceId -> 该用户的历史行为序列（工业落地时替换为远程序列服务调用） */
        private final Map<String, BehaviorSequence> historyStore = new HashMap<>();

        public SampleJoinService(FeedbackStore feedbackStore, MockKafkaClient kafka) {
            this.feedbackStore = feedbackStore;
            this.kafka = kafka;
        }

        /** 注入历史特征服务与行为序列快照，二者同时设置才会启用历史特征增强。 */
        public void setHistoryFeatureService(HistoryFeatureService service,
                                             Map<String, BehaviorSequence> history) {
            this.historyFeatureService = service;
            this.historyStore.putAll(history);
        }

        // ── tex snippet: code_ordinary_wait_window ──
        private boolean shouldWait(JoinRequest request, long nowMs) {
            JoinPolicy policy = policyProvider.forScene(request.getScene());
            long firstFeedbackMs = feedbackStore.firstKeyFeedbackTime(request.getRequestId());
            long recentFeedbackMs = feedbackStore.lastFeedbackTime(request.getRequestId());

            long requestDeadline = request.getRequestTimeMs()
                    + (firstFeedbackMs > 0 ? policy.baseWaitMs() : policy.maxWaitMs());
            long activeDeadline = recentFeedbackMs > 0
                    ? recentFeedbackMs + policy.activeExtensionMs() : 0;
            return nowMs < Math.max(requestDeadline, activeDeadline)
                    && request.getReplayCount() < policy.maxReplayCount();
        }

        public void consume(byte[] payload) {
            JoinRequest request = JoinRequest.parseFrom(payload);
            if (shouldWait(request, System.currentTimeMillis())) {
                JoinRequest replay = request.toBuilder()
                        .setReplayCount(request.getReplayCount() + 1)
                        .build();
                kafka.send("sample_join_waiting", request.getRequestKey(), replay);
                sleepUninterruptibly(100, TimeUnit.MILLISECONDS);
                return;
            }
            joinAndEmit(request);
        }

        /** Overload for direct use in tests without byte[] serialization */
        public void consume(JoinRequest request) {
            if (shouldWait(request, System.currentTimeMillis())) {
                JoinRequest replay = request.toBuilder()
                        .setReplayCount(request.getReplayCount() + 1)
                        .build();
                kafka.send("sample_join_waiting", request.getRequestKey(), replay);
                return;
            }
            joinAndEmit(request);
        }

        // ── tex snippet: code_ordinary_sample_join ──
        public void joinAndEmit(JoinRequest request) {
            long requestId = request.getRequestId();
            List<Long> itemIds = request.getItemIds();

            Map<Long, Set<FeedbackKind>> actions = feedbackStore.batchGetActions(requestId, itemIds);
            if (actions.isEmpty()) {
                metrics.count("drop_no_feedback", request.getScene());
                return;
            }
            Map<Long, Map<FeedbackKind, Long>> values = feedbackStore.batchGetValues(requestId, actions);
            Map<Long, Long> showPositions = feedbackStore.batchGetShowPositions(requestId, itemIds);

            TrainingBatchPb.Builder output = TrainingBatchPb.newBuilder()
                    .setEventTimeMicros(TimeUnit.MILLISECONDS.toMicros(request.getRequestTimeMs()))
                    .addAllCommonFeatures(request.getCommonFeatures());

            int validLabelCount = 0;
            for (int i = 0; i < itemIds.size(); i++) {
                long itemId = itemIds.get(i);
                ItemSample.Builder sample = output.addItemBuilder().setItemId(itemId)
                        .addAllItemFeatures(request.getItemFeatures(i));

                for (FeedbackKind action : actions.getOrDefault(itemId, Collections.emptySet())) {
                    if (action == FeedbackKind.EXPOSURE_ONLY) {
                        continue; // 只有曝光占位, 不足以形成有效反馈样本
                    }
                    long value = values.getOrDefault(itemId, Collections.emptyMap())
                            .getOrDefault(action, 1L);
                    sample.addLabel(LabelValue.of(action.trainingName(), value));
                    validLabelCount++;
                }
                Long showPosition = showPositions.get(itemId);
                if (showPosition != null) {
                    sample.addFeature(IntFeature.of("show_position", showPosition));
                }
                if (historyFeatureService != null) {
                    BehaviorSequence history = historyStore.get(request.getDeviceId());
                    if (history != null) {
                        historyFeatureService.appendHistoryFeatures(
                                sample, itemId, history, request.getRequestTimeMs());
                    }
                }
            }

            if (validLabelCount == 0) {
                metrics.count("drop_exposure_without_feedback", request.getScene());
                return;
            }
            TrainingBatchPb batch = output.build();
            kafka.send("training_sample_stream", request.getRequestKey(), batch);
            feedbackStore.clear(requestId, itemIds); // 发送成功后主动清理，TTL 处理异常遗留的数据
            metrics.count("output_items", request.getScene(), batch.getItemCount());
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_ordinary_history_features
    //  按候选实体回溯历史反馈的可选增强层
    // ─────────────────────────────────────────────────────────

    /**
     * 历史特征增强服务 — 对应 tex 代码片段 code_ordinary_history_features。
     * 在基础样本之外补充候选实体的历史行为特征。
     */
    public static class HistoryFeatureService {
        private final long historyWindowMs;

        public HistoryFeatureService(long historyWindowMs) {
            this.historyWindowMs = historyWindowMs;
        }

        // ── tex snippet: code_ordinary_history_features ──
        public void appendHistoryFeatures(ItemSample.Builder sample, long entityId,
                BehaviorSequence history, long anchorTimeMs) {
            List<BehaviorEvent> events = history.eventsSortedByTime();
            int start = lowerBound(events, anchorTimeMs - historyWindowMs);

            Map<FeedbackKind, Long> amounts = new EnumMap<>(FeedbackKind.class);
            Set<FeedbackKind> occurred = EnumSet.noneOf(FeedbackKind.class);
            for (int i = start; i < events.size(); i++) {
                BehaviorEvent event = events.get(i);
                if (event.getEventTimeMs() > anchorTimeMs || event.getEntityId() != entityId) {
                    continue;
                }
                occurred.add(event.getKind());
                if (event.getKind().valueMode() == ValueMode.SUM) {
                    amounts.merge(event.getKind(), event.getValue(), Long::sum);
                }
            }
            occurred.forEach(kind -> sample.addFeature(IntFeature.of("history_" + kind.featureName(), 1)));
            amounts.forEach((kind, value) -> sample.addFeature(
                    LongFeature.of("history_" + kind.featureName() + "_value", value)));
        }

        /** 二分查找：在已排序事件列表中找到第一个事件时间 >= targetMs 的位置 */
        private int lowerBound(List<BehaviorEvent> events, long targetMs) {
            int lo = 0, hi = events.size();
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (events.get(mid).getEventTimeMs() < targetMs) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_cascade_join
    //  保留级联候选的统一样本组装
    // ─────────────────────────────────────────────────────────

    /** 级联候选元数据 */
    public static class StageCandidate {
        private final long itemId;
        private final String stage;
        private final int rank;
        private final float score;
        private final String sampleReason;
        private final float sampleProbability;
        private final boolean enteredNextStage;
        private final List<EncodedFeature> features;

        public StageCandidate(long itemId, String stage, int rank, float score,
                              String sampleReason, float sampleProbability,
                              boolean enteredNextStage, List<EncodedFeature> features) {
            this.itemId = itemId; this.stage = stage; this.rank = rank; this.score = score;
            this.sampleReason = sampleReason; this.sampleProbability = sampleProbability;
            this.enteredNextStage = enteredNextStage;
            this.features = features != null ? features : Collections.emptyList();
        }

        public long itemId() { return itemId; }
        public String stage() { return stage; }
        public int rank() { return rank; }
        public float score() { return score; }
        public String sampleReason() { return sampleReason; }
        public float sampleProbability() { return sampleProbability; }
        public boolean enteredNextStage() { return enteredNextStage; }
        public List<EncodedFeature> features() { return features; }
    }

    /** 已曝光物品（含排序ID） */
    public static class RankedItem {
        private final long id;
        private final int rank;
        public RankedItem(long id, int rank) { this.id = id; this.rank = rank; }
        public long id() { return id; }
        public int rank() { return rank; }
    }

    /** 级联追踪信息 */
    public static class CascadeTrace {
        private final List<RankedItem> exposedItems;
        public CascadeTrace(List<RankedItem> exposedItems) { this.exposedItems = exposedItems; }
        public List<RankedItem> exposedItems() { return exposedItems; }
    }

    /**
     * 级联样本组装服务 — 对应 tex 代码片段 code_cascade_join。
     * 保留多阶段候选并仅对实际曝光物品读取真实用户反馈。
     */
    public static class CascadeSampleService {
        private final FeedbackStore feedbackStore;
        private final Metrics metrics = new Metrics();

        public CascadeSampleService(FeedbackStore feedbackStore) {
            this.feedbackStore = feedbackStore;
        }

        // ── tex snippet: code_cascade_join ──
        public TrainingBatchPb buildCascadeBatch(JoinRequest request, CascadeTrace trace,
                List<StageCandidate> candidates) {
            // 只有最终曝光物品可以读取真实用户反馈。
            Set<Long> exposedIds = new HashSet<>();
            for (RankedItem item : trace.exposedItems()) {
                exposedIds.add(item.id());
            }
            Map<Long, Set<FeedbackKind>> feedback = feedbackStore.batchGetActions(
                    request.getRequestId(), new ArrayList<>(exposedIds));
            Map<Long, Map<FeedbackKind, Long>> values = feedbackStore.batchGetValues(
                    request.getRequestId(), feedback);
            TrainingBatchPb.Builder batch = TrainingBatchPb.newBuilder().addAllCommonFeatures(
                    request.getCommonFeatures());

            for (StageCandidate candidate : candidates) {
                boolean exposed = exposedIds.contains(candidate.itemId());
                // 所有候选都保留链路阶段信息，供不同训练任务使用。
                ItemSample.Builder sample = batch.addItemBuilder()
                        .setItemId(candidate.itemId())
                        .addAllItemFeatures(candidate.features())
                        .addFeature(StringFeature.of("stage", candidate.stage()))
                        .addFeature(IntFeature.of("stage_rank", candidate.rank()))
                        .addFeature(FloatFeature.of("stage_score", candidate.score()))
                        .addFeature(StringFeature.of("sample_reason", candidate.sampleReason()))
                        .addFeature(FloatFeature.of("sample_probability", candidate.sampleProbability()))
                        .addLabel(LabelValue.of("entered_next_stage", candidate.enteredNextStage() ? 1 : 0))
                        .addLabel(LabelValue.of("is_exposed", exposed ? 1 : 0));

                // 未曝光候选没有真实用户反馈，不能伪造点击、观看时长等 Label。
                if (exposed) {
                    appendObservedFeedback(sample, candidate.itemId(), feedback, values);
                }
            }
            return batch.build();
        }

        // ── tex snippet: code_cascade_sampling ──
        /**
         * 从完整候选列表中按阶段配额采样未曝光物品，
         * 同时全量保留所有最终曝光物品（可以获取真实用户反馈）。
         * quotaByStage 控制精排尾部、粗排淘汰、召回淘汰各阶段的采样上限。
         */
        public List<StageCandidate> sampleCascadeCandidates(
                List<StageCandidate> allCandidates, CascadeTrace trace,
                Map<String, Integer> quotaByStage) {
            Set<Long> exposedIds = new HashSet<>();
            for (RankedItem item : trace.exposedItems()) exposedIds.add(item.id());
            List<StageCandidate> result = new ArrayList<>();
            // 最终曝光物品全量保留
            for (StageCandidate c : allCandidates) {
                if (exposedIds.contains(c.itemId())) result.add(c);
            }
            // 未曝光候选按阶段配额随机采样
            Map<String, List<StageCandidate>> byStage = new LinkedHashMap<>();
            for (StageCandidate c : allCandidates) {
                if (!exposedIds.contains(c.itemId()))
                    byStage.computeIfAbsent(c.stage(), k -> new ArrayList<>()).add(c);
            }
            Random rng = new Random(42);
            for (Map.Entry<String, Integer> entry : quotaByStage.entrySet()) {
                List<StageCandidate> pool = byStage.getOrDefault(entry.getKey(), List.of());
                int quota = Math.min(entry.getValue(), pool.size());
                Collections.shuffle(pool, rng);
                result.addAll(pool.subList(0, quota));
            }
            return result;
        }

        private void appendObservedFeedback(ItemSample.Builder sample, long itemId,
                Map<Long, Set<FeedbackKind>> feedback,
                Map<Long, Map<FeedbackKind, Long>> values) {
            Set<FeedbackKind> kinds = feedback.getOrDefault(itemId, Collections.emptySet());
            for (FeedbackKind kind : kinds) {
                if (kind == FeedbackKind.EXPOSURE_ONLY) continue;
                long value = values.getOrDefault(itemId, Collections.emptyMap())
                        .getOrDefault(kind, 1L);
                sample.addLabel(LabelValue.of(kind.trainingName(), value));
            }
        }
    }

    // ─────────────────────────────────────────────────────────
    //  工具方法
    // ─────────────────────────────────────────────────────────

    private static void sleepUninterruptibly(long duration, TimeUnit unit) {
        try { unit.sleep(duration); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    // ─────────────────────────────────────────────────────────
    //  Demo / 集成测试入口
    // ─────────────────────────────────────────────────────────

    public static TrainingBatchPb runDemo() {
        MockRedisClient redis = new MockRedisClient();
        MockKafkaClient kafka = new MockKafkaClient();
        FeedbackStore store = new FeedbackStore(redis);
        SampleJoinService joiner = new SampleJoinService(store, kafka);

        long now = System.currentTimeMillis();

        // 1. 构造 JoinRequest（通常由 FeaturePreprocessService 产出）
        JoinRequest request = JoinRequest.newBuilder()
                .setRequestId(2001L)
                .setDeviceId("device_A")
                .setSessionId("session_1")
                .setScene("home_feed")
                .setRequestTimeMs(now - 300_000L)  // 5 分钟前，确保超出等待窗口
                .addAllItemIds(Arrays.asList(601L, 602L, 603L))
                .build();

        // 2. 反馈写入
        store.record(new FeedbackEventPb(2001L, 601L, "click", 1, now - 200_000L, true));
        store.record(new FeedbackEventPb(2001L, 602L, "watch_time", 45_000L, now - 180_000L, false));

        // 3. 拼接（等待窗口已过）
        joiner.consume(request);

        // 4. 读取输出
        List<TrainingBatchPb> batches = kafka.poll("training_sample_stream");
        if (!batches.isEmpty()) {
            TrainingBatchPb b = batches.get(0);
            System.out.println("[SampleFlowService] output items=" + b.getItemCount()
                    + " hasValidFeedback=" + b.hasValidFeedback());
            return b;
        }
        return null;
    }

    public static void main(String[] args) {
        System.out.println("=== ch01 SampleFlowService Demo ===");
        TrainingBatchPb result = runDemo();
        if (result != null && result.hasValidFeedback()) {
            System.out.println("SUCCESS: 训练样本已产出");
        } else {
            System.out.println("WARN: 未产出有效训练样本");
        }
    }
}
