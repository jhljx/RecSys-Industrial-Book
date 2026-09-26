package com.reco.offline.sample;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * 工业级 API 类型定义（供 tex 代码片段参考）
 *
 * 这些类模拟工业级 Protobuf Builder / Enum 风格的 API，
 * 与 NormalSampleFlow.java 中的 Mock 实现配合使用，
 * 使书中 tex 代码片段在完整代码中有对应实现。
 *
 * [MOCK] 所有存储、序列化均为内存实现；
 *        工业落地时替换为 Protobuf 生成代码 + Jedis + KafkaProducer。
 */
public class SampleFlowTypes {

    // ─────────────────────────────────────────────
    //  特征编码类型
    // ─────────────────────────────────────────────

    public enum ValueType { NUMERIC, CATEGORICAL }

    public static class RawFeature {
        public final String name;
        public final String stringValue;
        public final double numericValue;
        public final boolean empty;

        private RawFeature(String name, String sv, double nv, boolean empty) {
            this.name = name; this.stringValue = sv; this.numericValue = nv; this.empty = empty;
        }

        public static RawFeature of(String name, String value) {
            return new RawFeature(name, value, 0, value == null || value.isBlank());
        }

        public static RawFeature of(String name, double value) {
            return new RawFeature(name, null, value, false);
        }

        public String name() { return name; }
        public boolean isEmpty() { return empty; }
        public String stringValue() { return stringValue; }
        public double numericValue() { return numericValue; }
    }

    public static class EncodedFeature {
        public final int slotId;
        public final long sign;
        public final ValueType valueType;

        private EncodedFeature(int slotId, long sign, ValueType vt) {
            this.slotId = slotId; this.sign = sign; this.valueType = vt;
        }

        public static EncodedFeature of(int slotId, long sign, ValueType vt) {
            return new EncodedFeature(slotId, sign, vt);
        }
    }

    public static class FeatureRule {
        public final int slotId;
        public final String namespace;
        public final long hashSeed;
        public final ValueType valueType;
        private final boolean enabled;

        public FeatureRule(int slotId, String namespace, long hashSeed, ValueType vt, boolean enabled) {
            this.slotId = slotId; this.namespace = namespace;
            this.hashSeed = hashSeed; this.valueType = vt; this.enabled = enabled;
        }

        public int slotId() { return slotId; }
        public String namespace() { return namespace; }
        public long hashSeed() { return hashSeed; }
        public ValueType valueType() { return valueType; }
        public boolean isEnabled() { return enabled; }
    }

    public static class FeatureSchema {
        private final String version;
        private final Map<String, FeatureRule> commonRules;
        private final Map<String, FeatureRule> itemRules;

        public FeatureSchema(String version, Map<String, FeatureRule> commonRules,
                             Map<String, FeatureRule> itemRules) {
            this.version = version;
            this.commonRules = commonRules;
            this.itemRules = itemRules;
        }

        public String version() { return version; }
        public FeatureRule commonRule(String name) { return commonRules.get(name); }
        public FeatureRule itemRule(String name) { return itemRules.get(name); }
    }

    public static class RawItem {
        public final long itemId;
        public final List<RawFeature> features;

        public RawItem(long itemId, List<RawFeature> features) {
            this.itemId = itemId; this.features = features;
        }

        public long getItemId() { return itemId; }
        public List<RawFeature> getFeatures() { return features; }
    }

    /** 原始请求（含特征，用于特征预处理） */
    public static class RawRequestWithFeatures {
        public final long requestId;
        public final long requestTimeMs;
        public final String deviceId;
        public final String sessionId;
        public final String scene;
        public final String featureSchemaVersion;
        public final List<RawFeature> commonFeatures;
        public final List<RawItem> items;
        /** 级联样本流扩展：未曝光候选（可选，普通样本流传 null/空列表） */
        public final List<RawStagedItem> cascadeCandidates;

        /** 向后兼容：不含 scene 的构造器 */
        public RawRequestWithFeatures(long requestId, long requestTimeMs, String deviceId,
                                      String sessionId, String schemaVersion,
                                      List<RawFeature> commonFeatures, List<RawItem> items) {
            this(requestId, requestTimeMs, deviceId, sessionId, "", schemaVersion,
                    commonFeatures, items, Collections.emptyList());
        }

        public RawRequestWithFeatures(long requestId, long requestTimeMs, String deviceId,
                                      String sessionId, String scene, String schemaVersion,
                                      List<RawFeature> commonFeatures, List<RawItem> items) {
            this(requestId, requestTimeMs, deviceId, sessionId, scene, schemaVersion,
                    commonFeatures, items, Collections.emptyList());
        }

        public RawRequestWithFeatures(long requestId, long requestTimeMs, String deviceId,
                                      String sessionId, String scene, String schemaVersion,
                                      List<RawFeature> commonFeatures, List<RawItem> items,
                                      List<RawStagedItem> cascadeCandidates) {
            this.requestId = requestId;
            this.requestTimeMs = requestTimeMs;
            this.deviceId = deviceId;
            this.sessionId = sessionId;
            this.scene = scene != null ? scene : "";
            this.featureSchemaVersion = schemaVersion;
            this.commonFeatures = commonFeatures;
            this.items = items;
            this.cascadeCandidates = cascadeCandidates != null ? cascadeCandidates : Collections.emptyList();
        }

        public long getRequestId() { return requestId; }
        public long getRequestTimeMs() { return requestTimeMs; }
        public String getDeviceId() { return deviceId; }
        public String getSessionId() { return sessionId; }
        public String getScene() { return scene; }
        public String getFeatureSchemaVersion() { return featureSchemaVersion; }
        public List<RawFeature> getCommonFeatures() { return commonFeatures; }
        public List<RawItem> getItems() { return items; }
    }

    /** 预处理完成的请求（已编码特征） */
    public static class FeatureReadyRequest {
        public final long requestId;
        public final long requestTimeMs;
        public final String deviceId;
        public final String sessionId;
        public final String featureSchemaVersion;
        public final List<EncodedFeature> commonFeatures;
        public final List<EncodedItem> items;
        public final List<Long> itemIds;
        public final String scene;
        /** 级联样本流扩展：编码后的未曝光候选（普通样本流为空列表） */
        public final List<EncodedStagedItem> cascadeCandidates;

        private FeatureReadyRequest(Builder b) {
            this.requestId = b.requestId;
            this.requestTimeMs = b.requestTimeMs;
            this.deviceId = b.deviceId;
            this.sessionId = b.sessionId;
            this.featureSchemaVersion = b.featureSchemaVersion;
            this.commonFeatures = List.copyOf(b.commonFeatures);
            this.items = List.copyOf(b.items);
            this.itemIds = items.stream().map(e -> e.itemId).toList();
            this.scene = b.scene;
            this.cascadeCandidates = List.copyOf(b.cascadeCandidates);
        }

        public long getRequestId() { return requestId; }
        public long getRequestTimeMs() { return requestTimeMs; }
        public String getDeviceId() { return deviceId; }
        public String getSessionId() { return sessionId; }
        public List<EncodedFeature> getCommonFeatures() { return commonFeatures; }
        public List<Long> getItemIds() { return itemIds; }
        public List<EncodedFeature> getItemFeatures(int idx) {
            return idx < items.size() ? items.get(idx).features : List.of();
        }
        public boolean hasValidIdentity() { return requestId > 0 && deviceId != null; }
        public String getScene() { return scene; }
        public int getReplayCount() { return 0; }
        public String getRequestKey() { return deviceId + "#" + requestId; }

        public static Builder newBuilder() { return new Builder(); }

        public static class Builder {
            long requestId, requestTimeMs;
            String deviceId = "", sessionId = "", featureSchemaVersion = "", scene = "";
            List<EncodedFeature> commonFeatures = new ArrayList<>();
            List<EncodedItem> items = new ArrayList<>();
            List<EncodedItem.Builder> itemBuilders = new ArrayList<>();
            List<EncodedStagedItem> cascadeCandidates = new ArrayList<>();

            public Builder setRequestId(long v) { requestId = v; return this; }
            public Builder setRequestTimeMs(long v) { requestTimeMs = v; return this; }
            public Builder setDeviceId(String v) { deviceId = v; return this; }
            public Builder setSessionId(String v) { sessionId = v; return this; }
            public Builder setFeatureSchemaVersion(String v) { featureSchemaVersion = v; return this; }
            public Builder setScene(String v) { scene = v; return this; }
            public Builder addCommonFeature(EncodedFeature f) { commonFeatures.add(f); return this; }
            public EncodedItem.Builder addItemBuilder() {
                EncodedItem.Builder b = new EncodedItem.Builder();
                itemBuilders.add(b);
                return b;
            }
            public Builder addCascadeCandidate(EncodedStagedItem c) { cascadeCandidates.add(c); return this; }
            public FeatureReadyRequest build() {
                for (EncodedItem.Builder b : itemBuilders) items.add(b.build());
                return new FeatureReadyRequest(this);
            }
        }
    }

    public static class EncodedItem {
        public final long itemId;
        public final List<EncodedFeature> features;

        private EncodedItem(Builder b) { itemId = b.itemId; features = List.copyOf(b.features); }

        public static class Builder {
            long itemId;
            List<EncodedFeature> features = new ArrayList<>();

            public Builder setItemId(long v) { itemId = v; return this; }
            public Builder addFeature(EncodedFeature f) { features.add(f); return this; }
            public EncodedItem build() { return new EncodedItem(this); }
        }
    }

    // ─────────────────────────────────────────────
    //  反馈行为枚举
    // ─────────────────────────────────────────────

    public enum FeedbackKind {
        CLICK(0, ValueMode.OCCUR, "click"),
        WATCH_TIME(1, ValueMode.SUM, "watch_time"),
        LIKE(2, ValueMode.OCCUR, "like"),
        SHARE(3, ValueMode.OCCUR, "share"),
        EXPOSURE_ONLY(4, ValueMode.OCCUR, "exposure_only");

        private final int bitIndex;
        private final ValueMode valueMode;
        private final String featureName;

        FeedbackKind(int bitIndex, ValueMode valueMode, String featureName) {
            this.bitIndex = bitIndex;
            this.valueMode = valueMode;
            this.featureName = featureName;
        }

        public int bitIndex() { return bitIndex; }
        public ValueMode valueMode() { return valueMode; }
        public String trainingName() { return featureName; }
        public String featureName() { return featureName; }
        public boolean isTrainingLabel() { return this != EXPOSURE_ONLY; }

        public static FeedbackKind fromName(String name) {
            for (FeedbackKind k : values()) {
                if (k.featureName.equals(name)) return k;
            }
            return null;
        }
    }

    public enum ValueMode { OCCUR, SUM, LATEST }

    // ─────────────────────────────────────────────
    //  样本 Protobuf 风格类型
    // ─────────────────────────────────────────────

    public static class LabelValue {
        public final String name;
        public final long value;

        private LabelValue(String name, long value) { this.name = name; this.value = value; }
        public static LabelValue of(String name, long value) { return new LabelValue(name, value); }
    }

    public static class IntFeature {
        public final String name;
        public final long value;
        private IntFeature(String n, long v) { name = n; value = v; }
        public static IntFeature of(String name, long value) { return new IntFeature(name, value); }
    }

    public static class LongFeature {
        public final String name;
        public final long value;
        private LongFeature(String n, long v) { name = n; value = v; }
        public static LongFeature of(String name, long value) { return new LongFeature(name, value); }
    }

    // ─────────────────────────────────────────────
    //  行为序列类型（供历史特征增强层使用）
    // ─────────────────────────────────────────────

    /** 单次行为事件 */
    public static class BehaviorEvent {
        private final long entityId;
        private final FeedbackKind kind;
        private final long value;
        private final long eventTimeMs;

        public BehaviorEvent(long entityId, FeedbackKind kind, long value, long eventTimeMs) {
            this.entityId = entityId;
            this.kind = kind;
            this.value = value;
            this.eventTimeMs = eventTimeMs;
        }

        public long getEntityId() { return entityId; }
        public FeedbackKind getKind() { return kind; }
        public long getValue() { return value; }
        public long getEventTimeMs() { return eventTimeMs; }
    }

    /** 用户历史行为序列（已按事件时间排序） */
    public static class BehaviorSequence {
        private final List<BehaviorEvent> events;

        public BehaviorSequence(List<BehaviorEvent> events) {
            // 保证按事件时间升序排列
            List<BehaviorEvent> sorted = new ArrayList<>(events);
            sorted.sort(java.util.Comparator.comparingLong(BehaviorEvent::getEventTimeMs));
            this.events = Collections.unmodifiableList(sorted);
        }

        public List<BehaviorEvent> eventsSortedByTime() { return events; }
        public int size() { return events.size(); }
    }

    // ─────────────────────────────────────────────
    //  级联样本流类型
    // ─────────────────────────────────────────────

    /** 候选物品所处的推荐链路阶段 */
    public enum ItemStage {
        EXPOSED("exposed"),
        RANKED_NOT_SHOWN("ranked_not_shown"),
        RETRIEVED_NOT_RANKED("retrieved_not_ranked");

        private final String label;
        ItemStage(String label) { this.label = label; }
        public String label() { return label; }
    }

    /**
     * 原始级联候选（含阶段元数据 + 原始字符串特征），
     * 由推荐服务在请求产生时附加到 RawRequestWithFeatures。
     */
    public static class RawStagedItem {
        public final long itemId;
        public final ItemStage stage;
        public final int rank;
        public final float score;
        public final List<RawFeature> features;

        public RawStagedItem(long itemId, ItemStage stage, int rank, float score,
                             List<RawFeature> features) {
            this.itemId = itemId; this.stage = stage; this.rank = rank; this.score = score;
            this.features = features != null ? features : Collections.emptyList();
        }

        public long getItemId() { return itemId; }
        public ItemStage getStage() { return stage; }
        public int getRank() { return rank; }
        public float getScore() { return score; }
        public List<RawFeature> getFeatures() { return features; }
    }

    /**
     * 编码后的级联候选（slotId/sign 整数对特征 + 阶段元数据），
     * 由 FeaturePreprocessService 产出，存入 FeatureReadyRequest。
     */
    public static class EncodedStagedItem {
        public final long itemId;
        public final ItemStage stage;
        public final int rank;
        public final float score;
        public final List<EncodedFeature> features;

        private EncodedStagedItem(Builder b) {
            itemId = b.itemId; stage = b.stage; rank = b.rank; score = b.score;
            features = List.copyOf(b.features);
        }

        public long getItemId() { return itemId; }
        public ItemStage getStage() { return stage; }
        public int getRank() { return rank; }
        public float getScore() { return score; }
        public List<EncodedFeature> getFeatures() { return features; }

        public static class Builder {
            long itemId;
            ItemStage stage;
            int rank;
            float score;
            List<EncodedFeature> features = new ArrayList<>();

            public Builder setItemId(long v) { itemId = v; return this; }
            public Builder setStage(ItemStage v) { stage = v; return this; }
            public Builder setRank(int v) { rank = v; return this; }
            public Builder setScore(float v) { score = v; return this; }
            public Builder addFeature(EncodedFeature f) { features.add(f); return this; }
            public EncodedStagedItem build() { return new EncodedStagedItem(this); }
        }
    }

    public static class FloatFeature {
        public final String name;
        public final float value;
        private FloatFeature(String n, float v) { name = n; value = v; }
        public static FloatFeature of(String name, float value) { return new FloatFeature(name, value); }
    }

    public static class StringFeature {
        public final String name;
        public final String value;
        private StringFeature(String n, String v) { name = n; value = v; }
        public static StringFeature of(String name, String value) { return new StringFeature(name, value); }
    }

    /** 单物品训练样本 */
    public static class ItemSample {
        public final long itemId;
        public final List<EncodedFeature> itemFeatures;
        public final List<LabelValue> labels;
        public final List<IntFeature> intFeatures;

        private ItemSample(Builder b) {
            itemId = b.itemId;
            itemFeatures = List.copyOf(b.itemFeatures);
            labels = List.copyOf(b.labels);
            intFeatures = List.copyOf(b.intFeatures);
        }

        public static class Builder {
            long itemId;
            List<EncodedFeature> itemFeatures = new ArrayList<>();
            List<LabelValue> labels = new ArrayList<>();
            List<IntFeature> intFeatures = new ArrayList<>();
            List<StringFeature> stringFeatures = new ArrayList<>();
            List<FloatFeature> floatFeatures = new ArrayList<>();

            public Builder setItemId(long v) { itemId = v; return this; }
            public Builder addAllItemFeatures(List<EncodedFeature> f) { itemFeatures.addAll(f); return this; }
            public Builder addLabel(LabelValue l) { labels.add(l); return this; }
            public Builder addFeature(IntFeature f) { intFeatures.add(f); return this; }
            public Builder addFeature(LongFeature f) { intFeatures.add(IntFeature.of(f.name, f.value)); return this; }
            public Builder addFeature(StringFeature f) { stringFeatures.add(f); return this; }
            public Builder addFeature(FloatFeature f) { floatFeatures.add(f); return this; }
            public ItemSample build() { return new ItemSample(this); }
        }
    }

    /** 整刷训练批次 */
    public static class TrainingBatchPb {
        public final long eventTimeMicros;
        public final List<EncodedFeature> commonFeatures;
        public final List<ItemSample> items;
        public final long requestId;
        public final String scene;
        public final long requestTimeMs;
        public final String deviceId;
        public final String sessionId;

        private TrainingBatchPb(Builder b) {
            eventTimeMicros = b.eventTimeMicros;
            commonFeatures = List.copyOf(b.commonFeatures);
            items = List.copyOf(b.itemSamples);
            requestId = b.requestId;
            scene = b.scene;
            requestTimeMs = b.requestTimeMs;
            deviceId = b.deviceId;
            sessionId = b.sessionId;
        }

        public boolean hasValidFeedback() { return items.stream().anyMatch(i -> !i.labels.isEmpty()); }
        public boolean hasCompleteLabels() { return hasValidFeedback(); }
        public int getItemCount() { return items.size(); }
        public long getRequestTimeMs() { return requestTimeMs; }
        public long getRequestId() { return requestId; }
        public String getDeviceId() { return deviceId; }
        public String getSessionId() { return sessionId; }

        public static Builder newBuilder() { return new Builder(); }

        public static class Builder {
            long eventTimeMicros, requestId, requestTimeMs;
            String scene = "", deviceId = "", sessionId = "";
            List<EncodedFeature> commonFeatures = new ArrayList<>();
            List<ItemSample.Builder> itemBuilders = new ArrayList<>();
            List<ItemSample> itemSamples = new ArrayList<>();

            public Builder setEventTimeMicros(long v) { eventTimeMicros = v; return this; }
            public Builder setRequestId(long v) { requestId = v; return this; }
            public Builder setRequestTimeMs(long v) { requestTimeMs = v; return this; }
            public Builder setScene(String v) { scene = v; return this; }
            public Builder setDeviceId(String v) { deviceId = v; return this; }
            public Builder setSessionId(String v) { sessionId = v; return this; }
            public Builder addAllCommonFeatures(List<EncodedFeature> f) { commonFeatures.addAll(f); return this; }
            public ItemSample.Builder addItemBuilder() {
                ItemSample.Builder b = new ItemSample.Builder();
                itemBuilders.add(b);
                return b;
            }
            public TrainingBatchPb build() {
                for (ItemSample.Builder b : itemBuilders) itemSamples.add(b.build());
                return new TrainingBatchPb(this);
            }
        }
    }

    // ─────────────────────────────────────────────
    //  JoinRequest（样本拼接消费者使用）
    // ─────────────────────────────────────────────

    public static class JoinRequest {
        public final long requestId;
        public final String deviceId;
        public final String sessionId;
        public final String scene;
        public final long requestTimeMs;
        public final List<Long> itemIds;
        public final List<EncodedFeature> commonFeatures;
        public final List<List<EncodedFeature>> itemFeaturesList;
        public final int replayCount;

        private JoinRequest(Builder b) {
            requestId = b.requestId; deviceId = b.deviceId; sessionId = b.sessionId;
            scene = b.scene; requestTimeMs = b.requestTimeMs; itemIds = List.copyOf(b.itemIds);
            commonFeatures = List.copyOf(b.commonFeatures);
            itemFeaturesList = List.copyOf(b.itemFeaturesList);
            replayCount = b.replayCount;
        }

        public long getRequestId() { return requestId; }
        public String getDeviceId() { return deviceId; }
        public String getSessionId() { return sessionId; }
        public String getScene() { return scene; }
        public long getRequestTimeMs() { return requestTimeMs; }
        public List<Long> getItemIds() { return itemIds; }
        public List<EncodedFeature> getCommonFeatures() { return commonFeatures; }
        public List<EncodedFeature> getItemFeatures(int idx) {
            return idx < itemFeaturesList.size() ? itemFeaturesList.get(idx) : List.of();
        }
        public int getReplayCount() { return replayCount; }
        public boolean hasValidIdentity() { return requestId > 0 && deviceId != null; }
        public String getRequestKey() { return deviceId + "#" + requestId; }

        public Builder toBuilder() {
            Builder b = new Builder();
            b.requestId = requestId; b.deviceId = deviceId; b.sessionId = sessionId;
            b.scene = scene; b.requestTimeMs = requestTimeMs; b.itemIds.addAll(itemIds);
            b.commonFeatures.addAll(commonFeatures); b.itemFeaturesList.addAll(itemFeaturesList);
            b.replayCount = replayCount;
            return b;
        }

        public static Builder newBuilder() { return new Builder(); }

        /**
         * [MOCK] 模拟 Protobuf parseFrom(byte[])。
         * 工业落地时替换为 JoinRequest.parseFrom(bytes)。
         * 由于此为内存演示版本，直接返回空 JoinRequest。
         */
        public static JoinRequest parseFrom(byte[] bytes) {
            // [MOCK] 无法真正反序列化，返回空请求。
            // 工业落地时此方法由 Protobuf 编译器生成。
            return newBuilder().setRequestId(0).setDeviceId("unknown").build();
        }

        public static class Builder {
            long requestId, requestTimeMs;
            String deviceId = "", sessionId = "", scene = "";
            List<Long> itemIds = new ArrayList<>();
            List<EncodedFeature> commonFeatures = new ArrayList<>();
            List<List<EncodedFeature>> itemFeaturesList = new ArrayList<>();
            int replayCount;

            public Builder setRequestId(long v) { requestId = v; return this; }
            public Builder setDeviceId(String v) { deviceId = v; return this; }
            public Builder setSessionId(String v) { sessionId = v; return this; }
            public Builder setScene(String v) { scene = v; return this; }
            public Builder setRequestTimeMs(long v) { requestTimeMs = v; return this; }
            public Builder addItemId(long v) { itemIds.add(v); return this; }
            public Builder addAllItemIds(List<Long> v) { itemIds.addAll(v); return this; }
            public Builder addAllCommonFeatures(List<EncodedFeature> f) { commonFeatures.addAll(f); return this; }
            public Builder setReplayCount(int v) { replayCount = v; return this; }
            public JoinRequest build() { return new JoinRequest(this); }
        }
    }

    // ─────────────────────────────────────────────
    //  FeedbackEvent（统一的反馈事件类型）
    // ─────────────────────────────────────────────

    public static class FeedbackEventPb {
        public final long requestId;
        public final long itemId;
        public final String action;
        public final long value;
        public final long eventTimeMs;
        public final boolean keyFeedback;

        public FeedbackEventPb(long requestId, long itemId, String action,
                               long value, long eventTimeMs, boolean keyFeedback) {
            this.requestId = requestId; this.itemId = itemId; this.action = action;
            this.value = value; this.eventTimeMs = eventTimeMs; this.keyFeedback = keyFeedback;
        }

        public long getRequestId() { return requestId; }
        public long getItemId() { return itemId; }
        public String getAction() { return action; }
        public long getValue() { return value; }
        public long getEventTimeMs() { return eventTimeMs; }
        public boolean isKeyFeedback() { return keyFeedback; }
    }

    // ─────────────────────────────────────────────
    //  FeedbackValue（Flink MapState 中合并用）
    // ─────────────────────────────────────────────

    public static class FeedbackValue {
        public final long count;
        public final long sumValue;
        public final long lastEventTimeMs;

        public FeedbackValue(long count, long sumValue, long lastEventTimeMs) {
            this.count = count; this.sumValue = sumValue; this.lastEventTimeMs = lastEventTimeMs;
        }

        public static FeedbackValue merge(FeedbackValue existing, FeedbackEventPb event) {
            FeedbackKind kind = FeedbackKind.fromName(event.getAction());
            long sumV = (existing != null ? existing.sumValue : 0)
                    + (kind != null && kind.valueMode() == ValueMode.SUM ? event.getValue() : 1L);
            long cnt = (existing != null ? existing.count : 0) + 1;
            return new FeedbackValue(cnt, sumV, event.getEventTimeMs());
        }
    }

    // ─────────────────────────────────────────────
    //  等待窗口策略
    // ─────────────────────────────────────────────

    public static class JoinPolicy {
        public final long baseWaitMs;
        public final long maxWaitMs;
        public final long activeExtensionMs;
        public final int maxReplayCount;

        public JoinPolicy(long baseWaitMs, long maxWaitMs, long activeExtensionMs, int maxReplayCount) {
            this.baseWaitMs = baseWaitMs; this.maxWaitMs = maxWaitMs;
            this.activeExtensionMs = activeExtensionMs; this.maxReplayCount = maxReplayCount;
        }

        public long baseWaitMs() { return baseWaitMs; }
        public long maxWaitMs() { return maxWaitMs; }
        public long activeExtensionMs() { return activeExtensionMs; }
        public int maxReplayCount() { return maxReplayCount; }
    }

    // ─────────────────────────────────────────────
    //  Metrics / 监控 (空实现，避免编译错误)
    // ─────────────────────────────────────────────

    public static class Metrics {
        public void count(String key, Object... tags) { /* no-op */ }
    }

    // ─────────────────────────────────────────────
    //  工具方法
    // ─────────────────────────────────────────────

    /** 将字符串稳定哈希为 long（namespace + value + seed） */
    public static long stableHash(String namespace, String value, long seed) {
        long h = seed;
        for (char c : (namespace + ":" + value).toCharArray()) {
            h = h * 31 + c;
        }
        return h;
    }
}
