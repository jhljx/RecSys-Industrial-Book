package com.reco.offline.rl;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * RL 样本流完整服务实现 — 对应 test_chapter.tex 中 ch02 的所有 Redis 方案代码片段。
 *
 * 包含以下 tex 代码片段的完整可运行实现：
 *   code_label_writer      - 反馈标签消费者写入逻辑骨架（FeedbackLabelConsumer + FeedbackSinkService）
 *   code_label_reader      - 行为中转区批量读取与 Bitmap 解码（FeedbackQueryService）
 *   code_join_label        - Label 拼入样本（appendFeedbackLabels）
 *   code_wait_window       - 自适应等待窗口判定（isInWaitWindow）
 *   code_replay_join       - Label 拼接消费者的延迟再次投递骨架（LabelJoinConsumer）
 *   code_seq_index         - 真实曝光请求序列与下一刷定位（appendRealShow / findNextRequest）
 *   code_state_write       - 请求状态的压缩写入与结果标记（saveRequestState）
 *   code_state_read        - 请求状态读取：解压与解析容错（loadRequestState）
 *   code_state_clear       - 拼接后的状态清理与标记（evictRequestState）
 *   code_join_next_main    - 状态转移拼接消费者主流程（consume）
 *
 * [MOCK] Redis、Kafka、Snappy 均使用内存模拟；工业落地时替换为真实客户端。
 */
public class RLSampleFlowService {

    // ─────────────────────────────────────────────────────────
    //  ActionLog：用户行为日志（Kafka 消息体）
    // ─────────────────────────────────────────────────────────

    public static class ActionLog {
        public final long userId;
        public final String deviceId;
        public final List<String> requestIds;
        public final List<Long> itemIds;
        public final List<Long> authorIds;
        public final List<Long> actionValues;
        public final String actionName;
        /** 请求类型，如 "home_feed" / "search" / "live_feed" 等。 */
        public final String requestType;
        /** 场景 ID，用于区分不同推荐场景。 */
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

        /** 取第一个 requestId 的长整型值，解析失败返回 0。 */
        public long firstRequestId() {
            if (requestIds == null || requestIds.isEmpty()) return 0L;
            try { return Long.parseLong(requestIds.get(0)); }
            catch (NumberFormatException e) { return 0L; }
        }
    }

    // ─────────────────────────────────────────────────────────
    //  内联 FeedbackKind 枚举（ch02 包独立，不依赖 ch01）
    // ─────────────────────────────────────────────────────────

    public enum FeedbackKind {
        CLICK(0, "click"),
        WATCH_TIME(1, "watch_time"),
        LIVE_ENTER(2, "live_enter"),
        LIVE_WATCH(3, "live_watch_time"),
        LIKE(4, "like"),
        SHARE(5, "share"),
        EXPOSURE_ONLY(6, "exposure_only");

        private final int bitIndex;
        private final String featureName;

        FeedbackKind(int bitIndex, String featureName) {
            this.bitIndex = bitIndex;
            this.featureName = featureName;
        }

        public int bitIndex() { return bitIndex; }
        public String featureName() { return featureName; }
        public boolean isTrainingLabel() { return this != EXPOSURE_ONLY; }
        public boolean isNumericValue() { return this == WATCH_TIME || this == LIVE_WATCH; }

        public static boolean isNumericAction(String name) {
            FeedbackKind k = fromName(name);
            return k != null && k.isNumericValue();
        }

        public static FeedbackKind fromName(String name) {
            for (FeedbackKind k : values()) {
                if (k.featureName.equals(name)) return k;
            }
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────
    //  TrainRecord：对应 Protobuf TrainRecord，此为内存实现
    // ─────────────────────────────────────────────────────────

    public static class TrainRecord {
        public final long requestId;
        public final String deviceId;
        public final String sessionId;
        public final long requestTimeMs;
        public final String action;  // "insert_live" / "no_insert"
        public double reward;
        public final Map<String, Object> features;
        // has_feedback 公共属性
        public boolean hasFeedback;
        // has_join_next
        public boolean hasJoinNext;
        // 状态标记
        public String replayFlag;
        private int ingestSequence;
        private int entryCount;
        private List<Long> itemIds;

        private TrainRecord(long requestId, String deviceId, String sessionId,
                            long requestTimeMs, String action, Map<String, Object> features,
                            List<Long> itemIds) {
            this.requestId = requestId;
            this.deviceId = deviceId;
            this.sessionId = sessionId;
            this.requestTimeMs = requestTimeMs;
            this.action = action;
            this.features = new HashMap<>(features);
            this.itemIds = itemIds != null ? new ArrayList<>(itemIds) : new ArrayList<>();
            this.entryCount = this.itemIds.size();
        }

        public long getRequestId() { return requestId; }
        public String getDeviceId() { return deviceId; }
        public String getSessionId() { return sessionId; }
        public long getTimestamp() { return requestTimeMs; }
        public long getIngestSequence() { return ingestSequence; }
        public int getEntryCount() { return entryCount; }
        public List<Long> getItemIds() { return itemIds; }
        public boolean hasValidIdentity() { return requestId > 0 && deviceId != null; }

        public void setReplayFlag(String flag) { this.replayFlag = flag; }

        /** [MOCK] 序列化为 bytes */
        public byte[] toByteArray() { return ("record:" + requestId).getBytes(); }

        /** [MOCK] 从 bytes 解析（实际为 Protobuf 反序列化） */
        public static TrainRecord parseFrom(byte[] bytes) {
            // [MOCK] 返回空记录，实际应是 Protobuf 解析
            return new TrainRecord(0L, "", "", 0L, "", new HashMap<>(), new ArrayList<>());
        }

        public static TrainRecord of(long requestId, String deviceId, String sessionId,
                                      long requestTimeMs, String action, List<Long> itemIds) {
            return new TrainRecord(requestId, deviceId, sessionId, requestTimeMs, action,
                                   new HashMap<>(), itemIds);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  [MOCK] Redis 内存模拟
    // ─────────────────────────────────────────────────────────

    public static class MockRedis {
        // Bitmap 存储: key -> set of bit indices
        private final Map<String, Set<Integer>> bitmaps = new HashMap<>();
        // 计数器: key -> value
        private final Map<String, Long> counters = new HashMap<>();
        // 字符串: key -> value
        private final Map<String, String> strings = new HashMap<>();
        // 行为时间戳: key -> ms
        private final Map<String, Long> timestamps = new HashMap<>();
        // 请求序列: deviceId -> list of reqIds
        private final Map<String, List<Long>> showLists = new HashMap<>();
        // 已登记曝光: flagKey -> 1
        private final Set<String> existsKeys = new HashSet<>();
        // 状态存储: reqId -> bytes
        private final Map<Long, byte[]> stateData = new HashMap<>();
        // 状态标记: reqId -> flag
        private final Map<Long, String> stateFlags = new HashMap<>();
        // session: reqId -> sessionId
        private final Map<Long, String> sessions = new HashMap<>();

        private static final int TIMELINE_MAX_LEN = 100;
        private static final int TIMELINE_TTL_SECONDS = 7200;
        private static final int ACTION_TTL_SECONDS = 3600;

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
                public void expire(String key, int ttl) { /* TTL no-op */ }
                public void incrby(String key, long delta) {
                    counters.merge(key, Math.max(0, delta), Long::sum);
                }
                public void setex(String key, int ttl, String value) {
                    strings.put(key, value);
                }
            });
        }

        public void setex(String key, int ttl, String value) { strings.put(key, value); }

        public String get(String key) {
            if (strings.containsKey(key)) return strings.get(key);
            if (counters.containsKey(key)) return String.valueOf(counters.get(key));
            return null;
        }

        public boolean exists(String key) { return existsKeys.contains(key); }

        public void setNxEx(byte[] key, byte[] value, int ttl) {
            String k = new String(key);
            if (!existsKeys.contains(k)) {
                existsKeys.add(k);
                stateData.put((long) k.hashCode(), value);
            }
        }

        public byte[] binaryGet(byte[] key) {
            Long k = (long) new String(key).hashCode();
            return stateData.get(k);
        }

        public void delete(String key) {
            strings.remove(key);
            existsKeys.remove(key);
        }

        /** 模拟 binaryPipeline：itemId -> bitmap bytes */
        public Map<Long, byte[]> binaryPipeline(List<Long> itemIds,
                java.util.function.BiConsumer<MockRedis, Long> fn) {
            // [MOCK] 直接返回空，调用方通过 getOccurActions 补充
            return Collections.emptyMap();
        }

        public long getCounter(String key) { return counters.getOrDefault(key, 0L); }

        public Set<Integer> getBits(String key) {
            return bitmaps.getOrDefault(key, Collections.emptySet());
        }

        // ── 真实曝光序列操作 ──
        public void appendRealShow(String deviceId, long reqId) {
            String flagKey = String.format("real_show_%s_%d", deviceId, reqId);
            if (existsKeys.contains(flagKey)) return;
            existsKeys.add(flagKey);
            setex(flagKey, ACTION_TTL_SECONDS, "1");

            String listKey = String.format("real_show_list_%s", deviceId);
            List<Long> ids = parseIds(get(listKey));
            ids.remove(reqId);
            ids.add(reqId);
            if (ids.size() > TIMELINE_MAX_LEN) {
                ids = ids.subList(ids.size() - TIMELINE_MAX_LEN, ids.size());
            }
            setex(listKey, TIMELINE_TTL_SECONDS, joinIds(ids));
        }

        public List<Long> getRealShowList(String deviceId) {
            return parseIds(get(String.format("real_show_list_%s", deviceId)));
        }

        // ── 状态存储操作 ──
        public void saveState(long reqId, TrainRecord record, String sessionId) {
            byte[] data = record.toByteArray();
            String dataKey = String.format("state_pb_%d", reqId);
            stateData.put(reqId, data);
            existsKeys.add(dataKey);
            stateFlags.put(reqId, "write_success");
            sessions.put(reqId, sessionId);
        }

        public TrainRecord loadState(long reqId,
                Map<Long, TrainRecord> registry) {
            if (!existsKeys.contains(String.format("state_pb_%d", reqId))) return null;
            return registry.get(reqId);
        }

        public String getStateFlag(long reqId) { return stateFlags.get(reqId); }

        public void evictState(long reqId) {
            String dataKey = String.format("state_pb_%d", reqId);
            if (existsKeys.contains(dataKey)) {
                existsKeys.remove(dataKey);
                stateData.remove(reqId);
                stateFlags.put(reqId, "clear_success");
            }
        }

        public String getSession(long reqId) { return sessions.get(reqId); }

        // ── 行为存储 ──
        public void recordOccur(long reqId, long itemId, String action) {
            bitmaps.computeIfAbsent(String.format("fb_occ_%d_%d", reqId, itemId),
                    k -> new HashSet<>()).add(action.hashCode() & 0x7F);
            // store action name for retrieval
            strings.put(String.format("fb_action_%d_%d_%s", reqId, itemId, action), "1");
        }

        public Set<String> getOccurActions(long reqId, long itemId) {
            Set<String> result = new HashSet<>();
            String prefix = String.format("fb_action_%d_%d_", reqId, itemId);
            for (String key : strings.keySet()) {
                if (key.startsWith(prefix)) {
                    result.add(key.substring(prefix.length()));
                }
            }
            return result;
        }

        public long getActionTimestamp(long reqId, String actionType) {
            String key = String.format("fb_ts_%d_%s", reqId, actionType);
            String v = strings.get(key);
            return v != null ? Long.parseLong(v) : 0L;
        }

        public void recordActionTimestamp(long reqId, String actionType, long ts) {
            strings.put(String.format("fb_ts_%d_%s", reqId, actionType), String.valueOf(ts));
        }

        // ── 工具方法 ──
        private List<Long> parseIds(String val) {
            List<Long> ids = new ArrayList<>();
            if (val == null || val.isBlank()) return ids;
            for (String s : val.split(",")) {
                try { ids.add(Long.parseLong(s.trim())); } catch (NumberFormatException ignored) {}
            }
            return ids;
        }

        private String joinIds(List<Long> ids) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < ids.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(ids.get(i));
            }
            return sb.toString();
        }
    }

    /** [MOCK] Kafka 内存模拟 */
    public static class MockKafka {
        private final Map<String, List<Object>> topics = new HashMap<>();

        @SuppressWarnings("unchecked")
        public <T> void send(String topic, T msg) {
            topics.computeIfAbsent(topic, k -> new ArrayList<>()).add(msg);
        }

        @SuppressWarnings("unchecked")
        public <T> List<T> poll(String topic) {
            List<Object> msgs = topics.getOrDefault(topic, Collections.emptyList());
            List<T> result = new ArrayList<>((List<T>) msgs);
            msgs.clear();
            return result;
        }

        public String selfTopic() { return "rl_joint_sample"; }
    }

    /** [MOCK] Metrics */
    public static class Metrics {
        public void count(String key, Object... tags) { /* no-op */ }
    }

    // ─────────────────────────────────────────────────────────
    //  code_label_writer
    //  反馈标签消费者写入逻辑骨架
    // ─────────────────────────────────────────────────────────

    /**
     * 反馈标签写入服务 — 对应 tex 代码片段 code_label_writer。
     * 将反馈写入 Redis Bitmap（发生型）和计数器（数值型）。
     */
    public static class FeedbackSinkService {

        private static final int LABEL_TTL_SECONDS = 7200;

        private final MockRedis redis;
        private final Metrics metrics = new Metrics();

        public FeedbackSinkService(MockRedis redis) {
            this.redis = redis;
        }

        // ── tex snippet: code_label_writer ──
        public void recordOccurLabel(long reqId, long itemId, String actionName) {
            FeedbackKind action = FeedbackKind.fromName(actionName);
            // 未知行为直接丢弃并打点, 防止客户端新版本新行为枚举冲垮老样本流
            if (action == null || !action.isTrainingLabel()) {
                metrics.count("sink_unknown_action", actionName);
                return;
            }
            String key = String.format("fb_occ_%d_%d", reqId, itemId);
            redis.pipeline(pipe -> {
                pipe.setbit(key, action.bitIndex(), true);     // 行为 -> 比特位，true表示将该比特位设为 1
                pipe.expire(key, LABEL_TTL_SECONDS);
            });
            // [MOCK] also store action name for retrieval
            redis.recordOccur(reqId, itemId, actionName);
        }

        // 数值型行为：观看时长、播放次数等，计数器累加
        public void recordAmountLabel(long reqId, long itemId, String actionName, long delta) {
            FeedbackKind action = FeedbackKind.fromName(actionName);
            if (action == null) {
                return;
            }
            String key = String.format("fb_amt_%d_%d_%s", reqId, itemId, actionName);
            redis.pipeline(pipe -> {
                pipe.incrby(key, delta);
                pipe.expire(key, LABEL_TTL_SECONDS);
            });
            // 每次写入刷新行为时间戳, 自适应等待窗口的锚点输入
            String tsKey = String.format("fb_ts_%d", reqId);
            redis.setex(tsKey, LABEL_TTL_SECONDS, String.valueOf(System.currentTimeMillis()));
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_label_writer (外层 consumer 框架)
    //  FeedbackLabelConsumer：Kafka 消费用户 Action Log，写入 Redis 中转区
    // ─────────────────────────────────────────────────────────

    /**
     * Kafka 行为日志消费者 — 对应 tex 代码片段 code_label_writer（外层框架部分）。
     * 消费用户 Action Log，将 Label 写入 Redis 供样本拼接使用。
     */
    public static class FeedbackLabelConsumer {
        private final FeedbackSinkService sinkService;

        public FeedbackLabelConsumer(FeedbackSinkService sinkService) {
            this.sinkService = sinkService;
        }

        public void consume(ActionLog actionLog) {
            long reqId = actionLog.firstRequestId();
            if (reqId <= 0 || actionLog.itemIds.isEmpty()) return;
            for (int i = 0; i < actionLog.itemIds.size(); i++) {
                long itemId = actionLog.itemIds.get(i);
                if (FeedbackKind.isNumericAction(actionLog.actionName)) {
                    long value = i < actionLog.actionValues.size()
                            ? actionLog.actionValues.get(i) : 0L;
                    sinkService.recordAmountLabel(reqId, itemId, actionLog.actionName, value);
                } else {
                    sinkService.recordOccurLabel(reqId, itemId, actionLog.actionName);
                }
            }
        }

        private long firstRequestId(List<String> ids) {
            if (ids == null || ids.isEmpty()) return 0L;
            try { return Long.parseLong(ids.get(0)); } catch (NumberFormatException e) { return 0L; }
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_label_reader
    //  行为中转区批量读取与 Bitmap 解码
    // ─────────────────────────────────────────────────────────

    /**
     * 反馈查询服务 — 对应 tex 代码片段 code_label_reader。
     * 批量读取反馈并解码 Bitmap。
     */
    public static class FeedbackQueryService {
        private final MockRedis redis;

        public FeedbackQueryService(MockRedis redis) {
            this.redis = redis;
        }

        // ── tex snippet: code_label_reader ──
        /**
         * 批量读取一次请求下所有候选物品的行为集合.
         * 返回: itemId -> 该物品发生的所有行为名列表
         */
        public Map<Long, List<String>> queryOccurLabels(long reqId, List<Long> itemIds) {
            Map<Long, List<String>> result = new HashMap<>();
            if (reqId <= 0 || itemIds.isEmpty()) {
                return result;   // 非法请求不读取, 直接返回空, 后续按"Label缺失"处理
            }
            // Pipeline: 一次网络往返读取全部物品的 Bitmap
            // [MOCK] binaryPipeline returns empty; actions read via getOccurActions below
            Map<Long, byte[]> bitmapMap = redis.binaryPipeline(itemIds, (pipe, itemId) -> {});
            // [MOCK] Supplement from in-memory action store since binaryPipeline is no-op
            for (long itemId : itemIds) {
                Set<String> actions = redis.getOccurActions(reqId, itemId);
                if (!actions.isEmpty()) {
                    result.put(itemId, new ArrayList<>(actions));
                }
            }
            if (bitmapMap.isEmpty()) {
                return result;   // 全部 Key 不存在: 无任何行为
            }
            // Bitmap -> 行为枚举: 遍历枚举而不是比特位, 枚举新增行为时老样本自动被忽略
            for (Map.Entry<Long, byte[]> entry : bitmapMap.entrySet()) {
                long itemId = entry.getKey();
                byte[] bytes = entry.getValue();
                if (bytes == null) {
                    continue;
                }
                BitSet bits = fromRedisBitmap(bytes);
                if (bits.isEmpty()) {
                    continue;
                }
                for (FeedbackKind action : FeedbackKind.values()) {
                    if (bits.get(action.bitIndex())) {
                        result.computeIfAbsent(itemId, k -> new ArrayList<>()).add(action.featureName());
                    }
                }
            }
            return result;
        }

        /** 数值型 Label 批量读取: 与 Bitmap 同理, Key 维度为 (reqId, itemId, actionName) */
        public Map<Long, Map<String, Long>> queryAmountLabels(long reqId, Map<Long, List<String>> itemActionMap) {
            Map<Long, Map<String, Long>> result = new HashMap<>();
            // 展开为 (itemId, actionName) 对列表, Pipeline 批量 GET
            List<long[]> pairs = new ArrayList<>();
            List<String> actionNames = new ArrayList<>();
            itemActionMap.forEach((itemId, actions) ->
                    actions.forEach(action -> {
                        pairs.add(new long[]{itemId});
                        actionNames.add(action);
                    }));
            for (int idx = 0; idx < pairs.size(); idx++) {
                long itemId = pairs.get(idx)[0];
                String action = actionNames.get(idx);
                String key = String.format("fb_amt_%d_%d_%s", reqId, itemId, action);
                String value = redis.get(key);
                if (value != null) {
                    result.computeIfAbsent(itemId, k -> new HashMap<>())
                            .put(action, Long.parseLong(value));
                }
            }
            return result;
        }

        public long getActionTimestamp(long reqId, String actionType) {
            return redis.getActionTimestamp(reqId, actionType);
        }

        public void clearOccurLabels(long reqId, List<Long> itemIds) {
            // [MOCK] no-op; TTL handles cleanup
        }

        /** Redis SETBIT 为大端比特序, 与 JDK BitSet 的内存布局相反, 必须显式反转 */
        private BitSet fromRedisBitmap(byte[] bytes) {
            BitSet bits = new BitSet();
            for (int i = 0; i < bytes.length * 8; i++) {
                if ((bytes[i / 8] & (1 << (7 - (i % 8)))) != 0) {
                    bits.set(i);
                }
            }
            return bits;
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_join_label
    //  Label 拼入样本逻辑
    // ─────────────────────────────────────────────────────────

    public static final long MAX_DURATION_MS = 24L * 3600 * 1000;

    /** 将 Redis 行为中转区读回的行为写入每个候选物品的样本结构（code_join_label）*/
    private static void appendFeedbackLabels(TrainRecordBuilder builder, long reqId,
            List<Long> itemIds, FeedbackQueryService feedbackService) {
        // 1. 批量读取是否发生与数值两类行为
        Map<Long, List<String>> actionMap = feedbackService.queryOccurLabels(reqId, itemIds);
        Map<Long, Map<String, Long>> amountMap = feedbackService.queryAmountLabels(reqId, actionMap);
        boolean hasAnyLabel = false;
        // 2. 逐物品写入 label 名与 label 值
        for (int i = 0; i < builder.entryCount; i++) {
            long itemId = itemIds.get(i);
            List<String> actions = actionMap.getOrDefault(itemId, Collections.emptyList());
            for (String action : actions) {
                builder.addLabelName(i, action);
                // 观看时长等数值型行为读计数器；只表示是否发生的行为默认值为 1
                long value = amountMap.getOrDefault(itemId, Collections.emptyMap())
                        .getOrDefault(action, 1L);
                // 时长类 Label 夹紧到物理合理区间, 防御客户端脏数据
                if (action.endsWith("duration") || action.endsWith("watch_time")) {
                    value = Math.max(0, Math.min(value, MAX_DURATION_MS));
                }
                builder.addLabelValue(i, value);
                hasAnyLabel = true;
            }
        }
        // 3. 请求级标记
        builder.hasFeedback = hasAnyLabel;
    }

    /** TrainRecord.Builder 辅助类 */
    public static class TrainRecordBuilder {
        // Builder 封装 TrainRecord 的构建
        private long requestId;
        private String deviceId = "";
        private String sessionId = "";
        private long requestTimeMs;
        private String action = "";
        private final List<Long> itemIds = new ArrayList<>();
        private final Map<Integer, List<String>> labelNames = new HashMap<>();
        private final Map<Integer, List<Long>> labelValues = new HashMap<>();
        public int entryCount;
        public boolean hasFeedback;

        public TrainRecordBuilder setRequestId(long v) { requestId = v; return this; }
        public TrainRecordBuilder setDeviceId(String v) { deviceId = v; return this; }
        public TrainRecordBuilder setSessionId(String v) { sessionId = v; return this; }
        public TrainRecordBuilder setTimestamp(long v) { requestTimeMs = v; return this; }
        public TrainRecordBuilder setAction(String v) { action = v; return this; }
        public TrainRecordBuilder addItemId(long v) { itemIds.add(v); entryCount++; return this; }

        public void addLabelName(int idx, String name) {
            labelNames.computeIfAbsent(idx, k -> new ArrayList<>()).add(name);
        }
        public void addLabelValue(int idx, long val) {
            labelValues.computeIfAbsent(idx, k -> new ArrayList<>()).add(val);
        }

        public TrainRecord build() {
            TrainRecord r = TrainRecord.of(requestId, deviceId, sessionId, requestTimeMs, action, itemIds);
            r.hasFeedback = hasFeedback;
            return r;
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_wait_window
    //  自适应等待窗口判定
    // ─────────────────────────────────────────────────────────

    public static final long BASE_DELAY = 30_000L;
    public static final long BASE_MAX_DELAY = 120_000L;
    public static final long ACTION_EXTEND_DELAY = 10_000L;

    private static long seconds(long ms) { return ms; }

    /**
     * 判断请求是否仍处于 Label 拼接等待窗口内（code_wait_window）。
     */
    public static class WaitWindowService {
        private final FeedbackQueryService feedbackService;

        public WaitWindowService(FeedbackQueryService feedbackService) {
            this.feedbackService = feedbackService;
        }

        // ── tex snippet: code_wait_window ──
        /**
         * 判断请求是否仍处于 Label 拼接等待窗口内.
         * 窗口锚点随行为时间戳逐级切换: requestTime -> keyActionTs -> innerActionTs
         */
        public boolean isInWaitWindow(long requestId, long requestTimeMs) {
            long nowMs = System.currentTimeMillis();

            // 第一级锚点: 请求时间. 若已观察到关键行为(如进入消费态), 锚点前移至该行为
            long keyActionTs = feedbackService.getActionTimestamp(requestId, "key_action");
            long baseTs = keyActionTs > 0 ? keyActionTs : requestTimeMs;
            // 无关键行为时，等待到最大窗口；有关键行为时，使用基准窗口
            long windowMs = seconds(keyActionTs > 0 ? BASE_DELAY : BASE_MAX_DELAY);
            if (nowMs - baseTs < windowMs) {
                return true;   // 仍在窗口内, 继续等待
            }
            // 若关键行为之后产生了间内行为(停留/滑动), 继续延长窗口
            long innerActionTs = feedbackService.getActionTimestamp(requestId, "inner_action");
            if (innerActionTs > 0 && nowMs - innerActionTs < seconds(ACTION_EXTEND_DELAY)) {
                return true;
            }
            return false;      // 超出所有窗口, 允许出样
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_replay_join
    //  Label 拼接消费者的延迟再次投递骨架
    // ─────────────────────────────────────────────────────────

    /**
     * Label 拼接消费者 — 对应 tex 代码片段 code_replay_join。
     */
    public static class LabelJoinConsumer {
        private final WaitWindowService waitWindowService;
        private final FeedbackQueryService feedbackService;
        private final MockKafka kafka;
        private final Metrics metrics = new Metrics();
        // registry: reqId -> TrainRecord (for state store)
        private final Map<Long, TrainRecord> registry = new HashMap<>();

        public LabelJoinConsumer(WaitWindowService waitWindowService,
                                  FeedbackQueryService feedbackService,
                                  MockKafka kafka) {
            this.waitWindowService = waitWindowService;
            this.feedbackService = feedbackService;
            this.kafka = kafka;
        }

        // ── tex snippet: code_replay_join ──
        public void consume(TrainRecord message) {
            long reqId = message.getRequestId();
            long requestMs = message.getTimestamp();

            if (waitWindowService.isInWaitWindow(reqId, requestMs)) {
                // 仍在等待窗口内：记录再次投递原因，然后写回自身 topic
                message.setReplayFlag("wait_window");
                kafka.send(kafka.selfTopic(), message);
                sleepUninterruptibly(100, TimeUnit.MILLISECONDS); // 轻微限流
                return;
            }
            // 超出窗口: 从行为中转区读取行为并写入样本
            TrainRecordBuilder builder = new TrainRecordBuilder()
                    .setRequestId(message.getRequestId())
                    .setDeviceId(message.getDeviceId())
                    .setSessionId(message.getSessionId())
                    .setTimestamp(message.getTimestamp())
                    .setAction(message.action);
            for (long itemId : message.getItemIds()) {
                builder.addItemId(itemId);
            }
            appendFeedbackLabels(builder, reqId, message.getItemIds(), feedbackService);
            // 拼接完成后主动清理该请求的行为 Key；异常遗留数据由 TTL 到期后清理
            feedbackService.clearOccurLabels(reqId, message.getItemIds());
            TrainRecord joined = builder.build();
            kafka.send("rl_joint_sample", joined);
            registry.put(reqId, joined);
        }

        public Map<Long, TrainRecord> getRegistry() { return registry; }
    }

    // ─────────────────────────────────────────────────────────
    //  code_seq_index
    //  真实曝光请求序列与下一刷定位
    // ─────────────────────────────────────────────────────────

    public static class LookupResult {
        public final long requestId;
        public final String status;
        private final boolean found;

        private LookupResult(long requestId, String status, boolean found) {
            this.requestId = requestId;
            this.status = status;
            this.found = found;
        }

        public static LookupResult found(long reqId) { return new LookupResult(reqId, "found", true); }
        public static LookupResult missing(String status) { return new LookupResult(0L, status, false); }
        public boolean isFound() { return found; }
    }

    /**
     * 曝光序列索引服务 — 对应 tex 代码片段 code_seq_index。
     */
    public static class SeqIndexService {
        private final MockRedis redis;

        private static final int ACTION_TTL_SECONDS = 3600;
        private static final int TIMELINE_TTL_SECONDS = 7200;
        private static final int TIMELINE_MAX_LEN = 100;

        public SeqIndexService(MockRedis redis) {
            this.redis = redis;
        }

        // ── tex snippet: code_seq_index ──
        public void appendRealShow(String deviceId, long reqId) {
            String flagKey = String.format("real_show_%s_%d", deviceId, reqId);
            if (redis.exists(flagKey)) {
                return;                         // 同一请求只登记一次
            }
            redis.setex(flagKey, ACTION_TTL_SECONDS, "1");

            String listKey = String.format("real_show_list_%s", deviceId);
            List<Long> ids = parseIds(redis.get(listKey));
            ids.remove(reqId);                  // 先删后加，避免检查与写入分开时重复
            ids.add(reqId);
            if (ids.size() > TIMELINE_MAX_LEN) {
                ids = ids.subList(ids.size() - TIMELINE_MAX_LEN, ids.size());
            }
            redis.setex(listKey, TIMELINE_TTL_SECONDS, joinIds(ids));
        }

        public LookupResult findNextRequest(String deviceId, long reqId) {
            List<Long> ids = parseIds(redis.get("real_show_list_" + deviceId));
            int pos = ids.indexOf(reqId);
            if (pos < 0) {
                return LookupResult.missing("req_id_not_found");
            }
            if (pos + 1 == ids.size()) {
                return LookupResult.missing("next_req_id_not_found");
            }
            return LookupResult.found(ids.get(pos + 1));
        }

        private List<Long> parseIds(String val) {
            List<Long> ids = new ArrayList<>();
            if (val == null || val.isBlank()) return ids;
            for (String s : val.split(",")) {
                try { ids.add(Long.parseLong(s.trim())); } catch (NumberFormatException ignored) {}
            }
            return ids;
        }

        private String joinIds(List<Long> ids) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < ids.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(ids.get(i));
            }
            return sb.toString();
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_state_write / code_state_read / code_state_clear
    //  请求状态的写入、读取与清理
    // ─────────────────────────────────────────────────────────

    /** 单位：秒 */
    private static final int STATE_TTL_SECONDS = 7200;
    private static final int FLAG_TTL_SECONDS = 3 * 3600;

    /**
     * 请求状态存储服务 — 对应 tex 代码片段 code_state_write / code_state_read / code_state_clear。
     */
    public static class RequestStateStore {
        private final MockRedis redis;
        // [MOCK] registry: reqId -> TrainRecord
        private final Map<Long, TrainRecord> registry = new HashMap<>();
        private final Metrics metrics = new Metrics();

        public RequestStateStore(MockRedis redis) {
            this.redis = redis;
        }

        /** 将 TrainRecord 存入 registry（供 loadRequestState 使用） */
        public void register(TrainRecord record) {
            registry.put(record.getRequestId(), record);
        }

        // ── tex snippet: code_state_write ──
        public void saveRequestState(long reqId, TrainRecord state) {
            String dataKey = String.format("state_pb_%d", reqId);
            String flagKey = String.format("state_flag_%d", reqId);
            final byte[] compressed;
            try {
                // [MOCK] Snappy.compress(state.toByteArray()) — 用 bytes 模拟
                compressed = state.toByteArray();
            } catch (Exception e) {
                redis.setex(flagKey, FLAG_TTL_SECONDS, "write_fail");
                metrics.count("save_state_error", e.getClass().getSimpleName());
                return;
            }
            redis.setNxEx(dataKey.getBytes(), compressed, STATE_TTL_SECONDS);
            redis.setex(flagKey, FLAG_TTL_SECONDS, "write_success");
            redis.saveState(reqId, state, state.getSessionId());
            registry.put(reqId, state);
        }

        // ── tex snippet: code_state_read ──
        public TrainRecord loadRequestState(long reqId) {
            if (reqId <= 0) {
                return null;
            }
            byte[] compressed = redis.binaryGet(String.format("state_pb_%d", reqId).getBytes());
            if (compressed == null) {
                return null;
            }
            try {
                // [MOCK] TrainRecord.parseFrom(Snappy.uncompress(compressed)) — 从 registry 查找
                return registry.get(reqId);
            } catch (Exception e) {
                metrics.count("load_state_error", e.getClass().getSimpleName());
                return null;
            }
        }

        // ── tex snippet: code_state_clear ──
        public void evictRequestState(long reqId) {
            if (reqId <= 0) {
                return;
            }
            String dataKey = String.format("state_pb_%d", reqId);
            String flagKey = String.format("state_flag_%d", reqId);
            if (redis.exists(dataKey)) {
                redis.delete(dataKey);
                redis.setex(flagKey, FLAG_TTL_SECONDS, "clear_success");
            }
            registry.remove(reqId);
        }

        public String queryStateFlag(long nextReqId) {
            return redis.getStateFlag(nextReqId);
        }

        public String lookupSession(long reqId) {
            return redis.getSession(reqId);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  code_join_next_main
    //  状态转移拼接消费者主流程
    // ─────────────────────────────────────────────────────────

    private static final long JOIN_WINDOW_SECONDS = 120L;
    private static final long REPLAY_JOIN_WINDOW_SECONDS = 30L;
    private static final long MAX_JOIN_WINDOW_SECONDS = 300L;

    /**
     * 状态转移拼接消费者 — 对应 tex 代码片段 code_join_next_main。
     */
    public static class TransitionJoinConsumer {
        private final SeqIndexService seqIndexService;
        private final RequestStateStore stateStore;
        private final MockKafka kafka;
        private final Metrics metrics = new Metrics();

        public TransitionJoinConsumer(SeqIndexService seqIndexService,
                                       RequestStateStore stateStore,
                                       MockKafka kafka) {
            this.seqIndexService = seqIndexService;
            this.stateStore = stateStore;
            this.kafka = kafka;
        }

        // ── tex snippet: code_join_next_main ──
        public void consume(TrainRecord message) {
            long reqId = message.getRequestId();
            String deviceId = message.getDeviceId();
            String sessionId = message.getSessionId();

            LookupResult lookup = seqIndexService.findNextRequest(deviceId, reqId);
            long nextReqId = lookup.requestId;
            TrainRecord nextState = stateStore.loadRequestState(nextReqId);

            if (nextState != null) {
                String nextSession = stateStore.lookupSession(nextReqId);
                if (equals(sessionId, nextSession) && !isBlank(sessionId)) {
                    emitJoined(message, nextState);       // has_join_next=1
                } else {
                    emitTerminal(message, "session_mismatch");
                }
                stateStore.evictRequestState(nextReqId);  // 已处理的下一刷状态清理
                return;
            }

            String stateFlag = stateStore.queryStateFlag(nextReqId);
            if (isInJoinWindow(message.getTimestamp(), stateFlag)) {
                kafka.send(kafka.selfTopic(), message);
                sleepUninterruptibly(100, TimeUnit.MILLISECONDS);
                return;
            }
            emitTerminal(message, nextReqId > 0 ? stateFlag : lookup.status);
        }

        private boolean isInJoinWindow(long requestMs, String stateFlag) {
            long windowSec = JOIN_WINDOW_SECONDS;
            if ("replay_data".equals(stateFlag)) {
                windowSec = REPLAY_JOIN_WINDOW_SECONDS;
            } else if ("missing_data".equals(stateFlag)) {
                windowSec = MAX_JOIN_WINDOW_SECONDS;
            }
            return System.currentTimeMillis() - requestMs < TimeUnit.SECONDS.toMillis(windowSec);
        }

        private void emitJoined(TrainRecord current, TrainRecord next) {
            current.hasJoinNext = true;
            kafka.send("rl_transition_topic", current);
        }

        private void emitTerminal(TrainRecord record, String reason) {
            record.hasJoinNext = false;
            kafka.send("rl_transition_topic", record);
        }

        private static boolean equals(String a, String b) {
            return a != null && a.equals(b);
        }

        private static boolean isBlank(String s) {
            return s == null || s.isBlank();
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

    public static List<TrainRecord> runDemo() {
        MockRedis redis = new MockRedis();
        MockKafka kafka = new MockKafka();

        FeedbackSinkService sinkService = new FeedbackSinkService(redis);
        FeedbackQueryService queryService = new FeedbackQueryService(redis);
        WaitWindowService waitWindowService = new WaitWindowService(queryService);
        SeqIndexService seqIndex = new SeqIndexService(redis);
        RequestStateStore stateStore = new RequestStateStore(redis);

        LabelJoinConsumer labelJoinConsumer = new LabelJoinConsumer(waitWindowService, queryService, kafka);
        TransitionJoinConsumer transitionConsumer = new TransitionJoinConsumer(seqIndex, stateStore, kafka);

        long now = System.currentTimeMillis();

        // 1. 写入用户反馈
        sinkService.recordOccurLabel(1001L, 601L, "click");
        sinkService.recordAmountLabel(1001L, 601L, "watch_time", 30_000L);
        redis.recordActionTimestamp(1001L, "key_action", now - 300_000L);

        // 2. 构造请求样本（等待窗口已过）
        TrainRecord req1 = TrainRecord.of(1001L, "device_A", "sess_1",
                now - 400_000L, "no_insert", Arrays.asList(601L, 602L));
        TrainRecord req2 = TrainRecord.of(1002L, "device_A", "sess_1",
                now - 200_000L, "insert_live", Arrays.asList(701L));

        // 3. 登记真实曝光序列
        seqIndex.appendRealShow("device_A", 1001L);
        seqIndex.appendRealShow("device_A", 1002L);

        // 4. Label 拼接
        labelJoinConsumer.consume(req1);
        labelJoinConsumer.consume(req2);

        // 5. 状态存储（模拟 Label 拼接后保存状态）
        List<TrainRecord> joined = kafka.poll("rl_joint_sample");
        for (TrainRecord r : joined) {
            stateStore.saveRequestState(r.getRequestId(), r);
        }

        // 6. 状态转移拼接
        for (TrainRecord r : joined) {
            transitionConsumer.consume(r);
        }

        List<TrainRecord> transitions = kafka.poll("rl_transition_topic");
        System.out.println("[RLSampleFlowService] transitions=" + transitions.size());
        return transitions;
    }

    public static void main(String[] args) {
        System.out.println("=== ch02 RLSampleFlowService Demo ===");
        List<TrainRecord> transitions = runDemo();
        System.out.println("Total RL transition samples: " + transitions.size());
    }
}
