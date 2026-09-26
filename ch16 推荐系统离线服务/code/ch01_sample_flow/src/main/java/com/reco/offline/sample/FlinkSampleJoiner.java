package com.reco.offline.sample;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * 基于 Flink 框架的普通样本拼接实现（可运行仿真）
 *
 * 完整 Flink 作业需要引入 flink-streaming-java 依赖并部署到集群；
 * 本文件通过内存仿真演示 Flink 双流 Join 的核心逻辑：
 *   - KeyedCoProcessFunction 双流合并
 *   - ValueState / MapState 存储请求与反馈
 *   - Event-time Timer 驱动等待截止
 *   - Watermark 处理乱序
 *
 * [MOCK] 所有 Flink 状态 / 定时器均用 Java 数据结构模拟；
 *        工业落地时替换为实际 Flink StreamExecutionEnvironment 运行时。
 */
public class FlinkSampleJoiner {

    // ─────────────────────────────────────────────
    //  Flink Stub 类型（简化，避免引入 Flink 依赖）
    //  工业落地时从 flink-streaming-java 中获取真实类型
    // ─────────────────────────────────────────────

    /** [STUB] 模拟 Flink ValueState<T>。工业落地替换为真实 ValueState。 */
    public static class ValueState<T> {
        private T value;
        public T value() { return value; }
        public void update(T v) { this.value = v; }
        public void clear() { this.value = null; }
    }

    /** [STUB] 模拟 Flink MapState<K,V>。工业落地替换为真实 MapState。 */
    public static class MapState<K, V> {
        private final Map<K, V> map = new HashMap<>();
        public V get(K key) { return map.get(key); }
        public void put(K key, V value) { map.put(key, value); }
        public boolean contains(K key) { return map.containsKey(key); }
        public Iterable<Map.Entry<K, V>> entries() { return map.entrySet(); }
        public void remove(K key) { map.remove(key); }
        public void clear() { map.clear(); }
    }

    /** [STUB] 模拟 Flink TimerService。工业落地替换为 KeyedCoProcessFunction.Context.timerService()。 */
    public static class TimerService {
        private final PriorityQueue<Long> timers = new PriorityQueue<>();
        public void registerEventTimeTimer(long timestampMs) { timers.add(timestampMs); }
        public void deleteEventTimeTimer(long timestampMs) { timers.remove(timestampMs); }
        public Long pollNextTimer() { return timers.isEmpty() ? null : timers.poll(); }
        public boolean hasTimer(long ts) { return timers.contains(ts); }
    }

    /** [STUB] 模拟 Flink Collector<T>。工业落地替换为真实 Collector。 */
    public static class Collector<T> {
        private final List<T> collected = new ArrayList<>();
        public void collect(T value) { collected.add(value); }
        public List<T> getCollected() { return Collections.unmodifiableList(collected); }
    }

    // ─────────────────────────────────────────────
    //  数据类型
    // ─────────────────────────────────────────────

    /** 请求流元素（已完成特征预处理） */
    public static class RequestEvent {
        public final long requestId;
        public final String deviceId;
        public final String sessionId;
        public final String scene;
        public final long requestTimeMs;    // Event Time
        public final List<Long> itemIds;

        public RequestEvent(long requestId, String deviceId, String sessionId,
                            String scene, long requestTimeMs, List<Long> itemIds) {
            this.requestId = requestId;
            this.deviceId = deviceId;
            this.sessionId = sessionId;
            this.scene = scene;
            this.requestTimeMs = requestTimeMs;
            this.itemIds = List.copyOf(itemIds);
        }

        public long getRequestId() { return requestId; }
        public String getDeviceId() { return deviceId; }
        public String getScene() { return scene; }
        public long getRequestTimeMs() { return requestTimeMs; }
        public List<Long> getItemIds() { return itemIds; }
    }

    /** 反馈流元素 */
    public static class FeedbackEvent {
        public final long requestId;
        public final long itemId;
        public final String action;
        public final long value;
        public final long eventTimeMs;    // Event Time
        public final boolean keyFeedback;

        public FeedbackEvent(long requestId, long itemId, String action,
                             long value, long eventTimeMs, boolean keyFeedback) {
            this.requestId = requestId;
            this.itemId = itemId;
            this.action = action;
            this.value = value;
            this.eventTimeMs = eventTimeMs;
            this.keyFeedback = keyFeedback;
        }

        public long getRequestId() { return requestId; }
        public long getItemId() { return itemId; }
        public String getAction() { return action; }
        public long getValue() { return value; }
        public long getEventTimeMs() { return eventTimeMs; }
        public boolean isKeyFeedback() { return keyFeedback; }

        /** 构造一个表示"无反馈"的占位事件，用于侧输出排查 */
        public static FeedbackEvent missingFor(long requestId) {
            return new FeedbackEvent(requestId, 0L, "missing_feedback", 0L, System.currentTimeMillis(), false);
        }
    }

    /** 拼接输出的训练样本 */
    public static class JoinedSample {
        public final long requestId;
        public final String sessionId;
        public final String scene;
        public final long requestTimeMs;
        /** itemId -> {action -> value} */
        public final Map<Long, Map<String, Long>> labels;
        public final boolean hasKeyFeedback;

        public JoinedSample(long requestId, String sessionId, String scene, long requestTimeMs,
                            Map<Long, Map<String, Long>> labels, boolean hasKeyFeedback) {
            this.requestId = requestId;
            this.sessionId = sessionId;
            this.scene = scene;
            this.requestTimeMs = requestTimeMs;
            this.labels = Map.copyOf(labels);
            this.hasKeyFeedback = hasKeyFeedback;
        }

        public boolean hasValidFeedback() {
            return labels.values().stream().anyMatch(m -> !m.isEmpty());
        }
    }

    // ─────────────────────────────────────────────
    //  KeyedCoProcessFunction 核心实现
    //  以 requestId 为 Key，合并请求流与反馈流
    // ─────────────────────────────────────────────

    /**
     * 基于 Flink KeyedCoProcessFunction 的双流样本拼接。
     *
     * 工业落地时继承：
     *   org.apache.flink.streaming.api.functions.co.KeyedCoProcessFunction
     *     <Long, RequestEvent, FeedbackEvent, JoinedSample>
     *
     * [MOCK] 状态和定时器通过内存结构模拟，逻辑与真实 Flink 一致。
     */
    public static class SampleJoinFunction {

        // 基础等待：有关键反馈时使用
        private static final long BASE_WAIT_MS = 30_000L;
        // 最大等待：始终无关键反馈时使用
        private static final long MAX_WAIT_MS  = 120_000L;

        // ── Keyed State ──

        /** 请求状态：requestId -> RequestEvent */
        // [MOCK] 工业落地替换为: ValueState<RequestEvent> requestState;
        private final Map<Long, ValueState<RequestEvent>> requestStates = new HashMap<>();

        /** 反馈状态：requestId -> (itemId:action -> value) */
        // [MOCK] 工业落地替换为: MapState<String, Long> feedbackState;
        private final Map<Long, MapState<String, Long>> feedbackStates = new HashMap<>();

        /** 关键反馈到达时间：requestId -> eventTimeMs */
        // [MOCK] 工业落地替换为: ValueState<Long> keyFeedbackTs;
        private final Map<Long, ValueState<Long>> keyFeedbackTsStates = new HashMap<>();

        /** 定时器服务（每个 requestId 独立） */
        private final Map<Long, TimerService> timerServices = new HashMap<>();

        /** 输出收集器 */
        private final Collector<JoinedSample> out = new Collector<>();

        // ── 请求流处理（processElement1）──

        /**
         * 处理来自请求流的事件。
         * 工业落地时对应：
         *   public void processElement1(RequestEvent req,
         *       Context ctx, Collector<JoinedSample> out)
         */
        public void processElement1(RequestEvent req) {
            long reqId = req.requestId;

            // 存储请求状态，供 onTimer 使用
            getRequestState(reqId).update(req);

            // 注册等待截止定时器（Event Time）
            long deadline = req.requestTimeMs + MAX_WAIT_MS;
            getTimerService(reqId).registerEventTimeTimer(deadline);
        }

        // ── 反馈流处理（processElement2）──

        /**
         * 处理来自反馈流的事件。
         * 工业落地时对应：
         *   public void processElement2(FeedbackEvent event,
         *       Context ctx, Collector<JoinedSample> out)
         */
        public void processElement2(FeedbackEvent event) {
            long reqId = event.requestId;
            String feedbackKey = event.itemId + ":" + event.action;

            // 合并反馈值（数值型累加，发生型取 1）
            MapState<String, Long> state = getFeedbackState(reqId);
            Long existing = state.get(feedbackKey);
            long merged = isNumeric(event.action)
                ? (existing == null ? 0L : existing) + event.value
                : 1L;
            state.put(feedbackKey, merged);

            // 记录关键反馈到达时间
            if (event.keyFeedback) {
                ValueState<Long> kfTs = getKeyFeedbackTsState(reqId);
                if (kfTs.value() == null) {
                    kfTs.update(event.eventTimeMs);
                    // 提前注册较短的 BASE_WAIT 定时器
                    RequestEvent req = getRequestState(reqId).value();
                    if (req != null) {
                        long baseDeadline = event.eventTimeMs + BASE_WAIT_MS;
                        TimerService timerService = getTimerService(reqId);
                        timerService.registerEventTimeTimer(baseDeadline);
                    }
                }
            }

            // 若请求已到达且等待窗口已关闭，立即出样
            RequestEvent req = getRequestState(reqId).value();
            if (req != null && isWindowClosed(req, System.currentTimeMillis())) {
                emitAndClear(reqId, out);
            }
        }

        // ── 定时器触发（onTimer）──

        /**
         * Event-time 定时器触发，执行等待截止出样。
         * 工业落地时对应：
         *   public void onTimer(long timestamp,
         *       OnTimerContext ctx, Collector<JoinedSample> out)
         */
        public void onTimer(long timestamp, long reqId) {
            RequestEvent req = getRequestState(reqId).value();
            if (req == null) return;   // 已被提前出样清理

            // 窗口截止，无论是否有足够反馈，强制出样（带终止标记）
            emitAndClear(reqId, out);
        }

        // ── 内部工具 ──

        private boolean isWindowClosed(RequestEvent req, long nowMs) {
            Long kfTs = getKeyFeedbackTsState(req.requestId).value();
            long deadline = kfTs != null
                ? req.requestTimeMs + BASE_WAIT_MS
                : req.requestTimeMs + MAX_WAIT_MS;
            return nowMs >= deadline;
        }

        private void emitAndClear(long reqId, Collector<JoinedSample> out) {
            RequestEvent req = getRequestState(reqId).value();
            if (req == null) return;

            Map<Long, Map<String, Long>> labels = new HashMap<>();
            for (long itemId : req.itemIds) {
                labels.put(itemId, new HashMap<>());
            }

            MapState<String, Long> state = getFeedbackState(reqId);
            for (Map.Entry<String, Long> entry : state.entries()) {
                String[] parts = entry.getKey().split(":", 2);
                if (parts.length == 2) {
                    long itemId = Long.parseLong(parts[0]);
                    if (labels.containsKey(itemId)) {
                        labels.get(itemId).put(parts[1], entry.getValue());
                    }
                }
            }

            Long kfTs = getKeyFeedbackTsState(reqId).value();
            boolean hasKeyFeedback = kfTs != null;

            JoinedSample sample = new JoinedSample(
                req.requestId, req.sessionId, req.scene, req.requestTimeMs,
                labels, hasKeyFeedback
            );
            out.collect(sample);

            // 清理状态
            getRequestState(reqId).clear();
            getFeedbackState(reqId).clear();
            getKeyFeedbackTsState(reqId).clear();
        }

        private boolean isNumeric(String action) {
            return action != null && (action.contains("time") || action.contains("count"));
        }

        // ── State 懒加载辅助 ──

        private ValueState<RequestEvent> getRequestState(long reqId) {
            return requestStates.computeIfAbsent(reqId, k -> new ValueState<>());
        }

        private MapState<String, Long> getFeedbackState(long reqId) {
            return feedbackStates.computeIfAbsent(reqId, k -> new MapState<>());
        }

        private ValueState<Long> getKeyFeedbackTsState(long reqId) {
            return keyFeedbackTsStates.computeIfAbsent(reqId, k -> new ValueState<>());
        }

        private TimerService getTimerService(long reqId) {
            return timerServices.computeIfAbsent(reqId, k -> new TimerService());
        }

        /** 触发所有已到期的定时器（模拟 Watermark 推进） */
        public void advanceWatermark(long watermarkMs) {
            // 收集需要触发的 (reqId, timerTs) 对
            List<long[]> toFire = new ArrayList<>();
            for (Map.Entry<Long, TimerService> entry : timerServices.entrySet()) {
                Long ts;
                while ((ts = entry.getValue().pollNextTimer()) != null && ts <= watermarkMs) {
                    toFire.add(new long[]{entry.getKey(), ts});
                }
            }
            for (long[] pair : toFire) {
                onTimer(pair[1], pair[0]);
            }
        }

        public List<JoinedSample> getOutput() {
            return out.getCollected();
        }
    }

    // ─────────────────────────────────────────────
    //  buildJob — 真实 Flink 作业主干（对应书中 code_flink_ordinary_join）
    //
    //  工业落地步骤：
    //    1. pom.xml 加 flink-streaming-java / flink-connector-kafka（provided scope）
    //    2. 实现 RequestEventDeserializer / FeedbackEventDeserializer / JoinedSampleSerializer
    //    3. mvn package && flink run -c com.reco.offline.sample.FlinkSampleJoiner job.jar
    //
    //  完整代码如下（取消注释后可直接使用）：
    //
    //  public static void buildJob(String bootstrapServers) throws Exception {
    //      StreamExecutionEnvironment env =
    //          StreamExecutionEnvironment.getExecutionEnvironment();
    //
    //      // EXACTLY_ONCE Checkpoint：外部化快照，任务取消后保留
    //      env.enableCheckpointing(60_000L, CheckpointingMode.EXACTLY_ONCE);
    //      env.getCheckpointConfig().setCheckpointTimeout(300_000L);
    //      env.getCheckpointConfig().enableExternalizedCheckpoints(
    //          ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
    //      env.setMaxParallelism(1024);
    //
    //      Properties kafkaProps = new Properties();
    //      kafkaProps.setProperty("bootstrap.servers", bootstrapServers);
    //      kafkaProps.setProperty("group.id", "sample-joiner");
    //
    //      // ── 请求流：feature_ready_requests ──
    //      // 特征预处理消费者将完整请求写入此 topic；按 requestTimeMs 分配 Watermark
    //      DataStream<RequestEvent> requestStream = env
    //          .addSource(new FlinkKafkaConsumer<>(
    //              "feature_ready_requests", new RequestEventDeserializer(), kafkaProps))
    //          .name("RequestSource")
    //          .assignTimestampsAndWatermarks(
    //              WatermarkStrategy.<RequestEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    //                  .withTimestampAssigner((e, t) -> e.requestTimeMs)
    //                  .withIdleness(Duration.ofSeconds(30)));
    //
    //      // ── 反馈流：user_feedback_events ──
    //      // 行为日志服务将点击/播放/点赞等用户行为写入此 topic
    //      DataStream<FeedbackEvent> feedbackStream = env
    //          .addSource(new FlinkKafkaConsumer<>(
    //              "user_feedback_events", new FeedbackEventDeserializer(), kafkaProps))
    //          .name("FeedbackSource")
    //          .assignTimestampsAndWatermarks(
    //              WatermarkStrategy.<FeedbackEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    //                  .withTimestampAssigner((e, t) -> e.eventTimeMs)
    //                  .withIdleness(Duration.ofSeconds(30)));
    //
    //      // ── 核心拼接：双流按 requestId 分区后 connect ──
    //      // SampleJoinFunction 在 Keyed State 中等待反馈，定时器触发后出样
    //      DataStream<JoinedSample> output = requestStream
    //          .keyBy(e -> e.requestId)
    //          .connect(feedbackStream.keyBy(e -> e.requestId))
    //          .process(new SampleJoinFunction())
    //          .name("SampleJoinProcessor").uid("SampleJoinProcessor");
    //
    //      // ── 结果写回 training_samples：事务型 Sink 与 Checkpoint 对齐 ──
    //      // 保证故障恢复后不重复输出；若同时写 Redis 等外部系统须另行去重
    //      output.sinkTo(KafkaSink.<JoinedSample>builder()
    //          .setBootstrapServers(bootstrapServers)
    //          .setRecordSerializer(KafkaRecordSerializationSchema.builder()
    //              .setTopic("training_samples")
    //              .setValueSerializationSchema(new JoinedSampleSerializer())
    //              .build())
    //          .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)
    //          .build()).name("TrainingSampleSink");
    //
    //      env.execute("SampleFlowJoiner");
    //  }
    // ─────────────────────────────────────────────

    // ─────────────────────────────────────────────
    //  buildPipeline — 内存仿真入口（单元测试用）
    // ─────────────────────────────────────────────

    /**
     * 以内存 List 模拟 Kafka Source / Sink 的仿真入口，供单元测试使用。
     * 生产作业入口结构见上方 buildJob 注释块。
     */
    public static class RedisLabelJoinFunction {

        /** [MOCK] 内存 Redis 替代真实 Jedis 连接 */
        private final Map<String, Long> redisStore;

        public RedisLabelJoinFunction(Map<String, Long> redisStore) {
            this.redisStore = redisStore;
        }

        /**
         * 对输出样本补充 Redis 中的反馈标签。
         * 工业落地时对应 RichMapFunction.map() 或 AsyncFunction.asyncInvoke()。
         */
        public JoinedSample enrich(JoinedSample sample) {
            Map<Long, Map<String, Long>> enriched = new HashMap<>(sample.labels);
            for (long itemId : enriched.keySet()) {
                // 从 Redis 读取发生型反馈
                for (String action : List.of("click", "like", "share")) {
                    String key = "fb_occ_" + sample.requestId + "_" + itemId + "_" + action;
                    Long val = redisStore.get(key);
                    if (val != null && val > 0) {
                        enriched.get(itemId).put(action, val);
                    }
                }
                // 从 Redis 读取数值型反馈
                for (String action : List.of("watch_time")) {
                    String key = "fb_amt_" + sample.requestId + "_" + itemId + "_" + action;
                    Long val = redisStore.get(key);
                    if (val != null && val > 0) {
                        enriched.get(itemId).merge(action, val, Long::sum);
                    }
                }
            }
            return new JoinedSample(
                sample.requestId, sample.sessionId, sample.scene, sample.requestTimeMs,
                enriched, sample.hasKeyFeedback
            );
        }
    }

    // ─────────────────────────────────────────────
    //  FlinkSessionJoinJob — Kafka 输入流 → Session 组装 → Kafka 输出流
    //  对应书中 code_session_assembly
    // ─────────────────────────────────────────────

    /** [STUB] Kafka 消费者桩。工业落地替换为 KafkaSource<JoinedSample>。 */
    public static class KafkaSourceStub {
        private final List<JoinedSample> records;
        public KafkaSourceStub(List<JoinedSample> records) { this.records = new ArrayList<>(records); }
        public List<JoinedSample> poll() { return Collections.unmodifiableList(records); }
    }

    /** [STUB] Kafka 生产者桩。工业落地替换为 KafkaSink<SessionSample>。 */
    public static class KafkaSinkStub {
        private final List<SessionSample> output = new ArrayList<>();
        public void emit(SessionSample s) { output.add(s); }
        public List<SessionSample> getOutput() { return Collections.unmodifiableList(output); }
    }

    /** Session 组装后的输出样本：一个会话内所有刷次的有序列表。 */
    public static class SessionSample {
        public final String sessionId;
        public final List<JoinedSample> requests; // 按 requestTimeMs 有序，已去重

        public SessionSample(String sessionId, List<JoinedSample> requests) {
            this.sessionId = sessionId;
            this.requests  = List.copyOf(requests);
        }

        public int size() { return requests.size(); }
    }

    /**
     * Session 组装函数 — 对应 Flink KeyedProcessFunction<String, JoinedSample, SessionSample>。
     *
     * 每条 JoinedSample 到达时追加到 session 缓冲区并滑动重置 gap 定时器；
     * gap 超时后关闭会话：排序 + 去重后写入输出 Kafka topic。
     *
     * [MOCK] sessionBuffer -> ListState<JoinedSample>
     *        activeTimer   -> ValueState<Long>（当前活跃 gap timer 时间戳）
     */
    public static class SessionAssemblyFunction {

        static final long SESSION_GAP_MS  = 30 * 60 * 1_000L; // 30 分钟 gap
        static final int  MAX_SESSION_LEN = 200;               // 单会话最大刷次

        // [MOCK] 工业落地替换为 ListState<JoinedSample>
        private final Map<String, List<JoinedSample>> sessionBuffer = new HashMap<>();
        // [MOCK] 工业落地替换为 ValueState<Long>
        private final Map<String, Long> activeTimer = new HashMap<>();

        private final TimerService          timerService = new TimerService();
        private final Collector<SessionSample> out       = new Collector<>();

        /**
         * 处理单条到达的拼接样本。
         * 工业落地时对应：
         *   public void processElement(JoinedSample value, Context ctx, Collector<SessionSample> out)
         */
        public void processElement(JoinedSample value, long eventTimeMs) {
            String sid = value.sessionId;
            List<JoinedSample> buf = sessionBuffer.computeIfAbsent(sid, k -> new ArrayList<>());

            if (buf.size() < MAX_SESSION_LEN) {
                buf.add(value);
            } else {
                // 超长截断；工业落地时写入侧输出或上报 metrics
                System.out.printf("[WARN] session=%s 超过最大长度 %d，丢弃多余刷次%n",
                        sid, MAX_SESSION_LEN);
            }

            // 删除旧 gap timer，注册新 gap timer（每条消息到达均滑动延迟）
            Long old = activeTimer.get(sid);
            if (old != null) timerService.deleteEventTimeTimer(old);
            long deadline = eventTimeMs + SESSION_GAP_MS;
            timerService.registerEventTimeTimer(deadline);
            activeTimer.put(sid, deadline);
        }

        /**
         * gap 定时器到期，关闭会话并写入 Kafka 输出 topic。
         * 工业落地时对应：
         *   public void onTimer(long timestamp, OnTimerContext ctx, Collector<SessionSample> out)
         */
        public void onTimer(long timestamp, String sessionId) {
            // 若定时器已被更新则跳过（避免老 timer 误触发）
            if (!Long.valueOf(timestamp).equals(activeTimer.get(sessionId))) return;

            List<JoinedSample> buf = sessionBuffer.getOrDefault(sessionId, Collections.emptyList());
            if (buf.isEmpty()) { cleanup(sessionId); return; }

            // 按 requestTimeMs 升序排列，再按 requestId 去重（保留最后到达的）
            Map<Long, JoinedSample> dedup = new LinkedHashMap<>();
            buf.stream()
               .sorted(Comparator.comparingLong(s -> s.requestTimeMs))
               .forEach(s -> dedup.put(s.requestId, s));

            out.collect(new SessionSample(sessionId, new ArrayList<>(dedup.values())));
            cleanup(sessionId);
        }

        private void cleanup(String sessionId) {
            sessionBuffer.remove(sessionId);
            activeTimer.remove(sessionId);
        }

        /** 推进 Watermark，触发所有已到期的 gap 定时器（仿真用）。 */
        public void advanceWatermark(long watermarkMs) {
            for (Map.Entry<String, Long> e : new ArrayList<>(activeTimer.entrySet())) {
                if (e.getValue() <= watermarkMs) {
                    onTimer(e.getValue(), e.getKey());
                }
            }
        }

        public List<SessionSample> getOutput() { return out.getCollected(); }
    }

    /**
     * Flink Session 组装作业入口（仿真）。
     *
     * 工业落地等价：
     *   StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
     *   env.fromSource(KafkaSource.<JoinedSample>builder()...build(),
     *                  WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofSeconds(5)),
     *                  "joined-sample-source")
     *      .keyBy(s -> s.sessionId)
     *      .process(new SessionAssemblyFunction())
     *      .sinkTo(KafkaSink.<SessionSample>builder()...build());
     *   env.execute("session-assembly");
     */
    public static class FlinkSessionJoinJob {
        static final String INPUT_TOPIC  = "joined_sample_stream";
        static final String OUTPUT_TOPIC = "session_sample_stream";

        public static List<SessionSample> run(KafkaSourceStub source, KafkaSinkStub sink) {
            SessionAssemblyFunction fn = new SessionAssemblyFunction();

            // 模拟 Flink：逐条从 Kafka source 读取并交给 processElement
            for (JoinedSample sample : source.poll()) {
                fn.processElement(sample, sample.requestTimeMs);
            }

            // 推进 Watermark 使所有 gap 定时器到期并触发
            long maxTs = source.poll().stream()
                    .mapToLong(s -> s.requestTimeMs).max().orElse(0L);
            fn.advanceWatermark(maxTs + SessionAssemblyFunction.SESSION_GAP_MS + 1_000L);

            // 模拟 Flink：将输出写入 Kafka sink
            for (SessionSample ss : fn.getOutput()) {
                sink.emit(ss);
            }
            return sink.getOutput();
        }
    }

    // ─────────────────────────────────────────────
    //  端到端 Demo（仿真 Flink 双流 Join）
    // ─────────────────────────────────────────────

    public static List<JoinedSample> runDemo() {
        SampleJoinFunction joinFn = new SampleJoinFunction();

        long now = System.currentTimeMillis();
        long req1Time = now - 200_000L;
        long req2Time = now - 180_000L;

        // 请求流
        joinFn.processElement1(new RequestEvent(
            1001L, "device_A", "session_X", "home_feed",
            req1Time, List.of(501L, 502L)
        ));
        joinFn.processElement1(new RequestEvent(
            1002L, "device_A", "session_X", "home_feed",
            req2Time, List.of(601L, 602L)
        ));

        // 反馈流（关键反馈：click）
        joinFn.processElement2(new FeedbackEvent(
            1001L, 501L, "click", 1L, req1Time + 5_000L, true
        ));
        joinFn.processElement2(new FeedbackEvent(
            1001L, 502L, "watch_time", 45_000L, req1Time + 10_000L, false
        ));
        joinFn.processElement2(new FeedbackEvent(
            1002L, 601L, "click", 1L, req2Time + 3_000L, true
        ));

        // 推进 Watermark，触发所有已到期定时器
        joinFn.advanceWatermark(now + 10_000L);

        List<JoinedSample> result = joinFn.getOutput();
        System.out.println("=== Flink 双流 Join 输出 ===");
        for (JoinedSample s : result) {
            System.out.printf("  reqId=%d sessionId=%s scene=%s hasKeyFeedback=%b labels=%s%n",
                s.requestId, s.sessionId, s.scene, s.hasKeyFeedback, s.labels);
        }
        return result;
    }

    /** Session 组装作业端到端演示。 */
    public static List<SessionSample> runSessionDemo() {
        // 复用双流 Join 产出的 JoinedSample 作为 Session 组装的输入
        List<JoinedSample> joined = runDemo();

        KafkaSourceStub source = new KafkaSourceStub(joined);
        KafkaSinkStub   sink   = new KafkaSinkStub();
        List<SessionSample> sessions = FlinkSessionJoinJob.run(source, sink);

        System.out.println("=== FlinkSessionJoinJob 输出 ===");
        for (SessionSample ss : sessions) {
            System.out.printf("  sessionId=%s requestCount=%d%n",
                ss.sessionId, ss.size());
        }
        return sessions;
    }

    public static void main(String[] args) {
        System.out.println("=== ch01 Flink 双流 Join Demo ===");
        List<JoinedSample> samples = runDemo();
        System.out.println("Join output: " + samples.size());

        System.out.println("\n=== ch01 Flink Session 组装 Demo ===");
        List<SessionSample> sessions = runSessionDemo();
        System.out.println("Session output: " + sessions.size());
        long withFeedback = samples.stream().filter(JoinedSample::hasValidFeedback).count();
        System.out.println("Joined samples with feedback: " + withFeedback);
    }
}
