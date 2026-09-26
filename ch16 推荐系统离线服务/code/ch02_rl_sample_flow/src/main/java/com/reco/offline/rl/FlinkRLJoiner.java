package com.reco.offline.rl;

import java.util.*;

/**
 * 基于 Flink 框架的 RL 状态转移样本拼接（可运行仿真）
 *
 * 上游基于 Redis 的 Label 拼接消费者已产出 (s_t, a_t, r_t) 基础样本；
 * 本 Flink 作业负责跨请求的时序关联，补全 s_{t+1}，产出完整状态转移样本。
 *
 * 数据流：
 *   (s_t, a_t, r_t) 基础样本流 + 真实曝光事件流
 *   -> keyBy(deviceId)
 *   -> KeyedCoProcessFunction（维护设备请求序列 + 等待定时器）
 *   -> (s_t, a_t, r_t, s_{t+1}) 状态转移样本流
 *
 * [MOCK] 所有 Flink 状态和定时器均用 Java 数据结构模拟；
 *        工业落地时替换为实际 Flink StreamExecutionEnvironment 运行时。
 */
public class FlinkRLJoiner {

    // ─────────────────────────────────────────────
    //  Flink Stub 类型（简化）
    // ─────────────────────────────────────────────

    /** [STUB] Flink ValueState<T>。工业落地替换为真实 ValueState。 */
    public static class ValueState<T> {
        private T value;
        public T value() { return value; }
        public void update(T v) { value = v; }
        public void clear() { value = null; }
    }

    /** [STUB] Flink ListState<T>。工业落地替换为真实 ListState。 */
    public static class ListState<T> {
        private final List<T> list = new ArrayList<>();
        public void add(T v) { list.add(v); }
        public Iterable<T> get() { return Collections.unmodifiableList(list); }
        public void update(List<T> v) { list.clear(); list.addAll(v); }
        public void clear() { list.clear(); }
    }

    /** [STUB] Flink MapState<K,V>。工业落地替换为真实 MapState。 */
    public static class MapState<K, V> {
        private final Map<K, V> map = new HashMap<>();
        public V get(K k) { return map.get(k); }
        public void put(K k, V v) { map.put(k, v); }
        public boolean contains(K k) { return map.containsKey(k); }
        public Iterable<Map.Entry<K, V>> entries() { return map.entrySet(); }
        public void remove(K k) { map.remove(k); }
        public void clear() { map.clear(); }
    }

    /** [STUB] Flink TimerService。工业落地替换为真实 TimerService。 */
    public static class TimerService {
        private final PriorityQueue<Long> timers = new PriorityQueue<>();
        public void registerEventTimeTimer(long ts) { timers.add(ts); }
        public void deleteEventTimeTimer(long ts) { timers.remove(ts); }
        public Long pollNextExpired(long watermark) {
            Long t = timers.peek();
            return (t != null && t <= watermark) ? timers.poll() : null;
        }
    }

    /** [STUB] Flink Collector<T>。工业落地替换为真实 Collector。 */
    public static class Collector<T> {
        private final List<T> list = new ArrayList<>();
        public void collect(T v) { list.add(v); }
        public List<T> getAll() { return Collections.unmodifiableList(list); }
    }

    // ─────────────────────────────────────────────
    //  数据类型
    // ─────────────────────────────────────────────

    /** 上游 Label 拼接后的基础 RL 样本 (s_t, a_t, r_t) */
    public static class BaseRLSample {
        public final long requestId;
        public final String deviceId;
        public final String sessionId;
        public final long requestTimeMs;    // Event Time
        public final String action;
        public final double reward;
        public final Map<String, Object> stateFeatures;

        public BaseRLSample(long requestId, String deviceId, String sessionId,
                            long requestTimeMs, String action, double reward,
                            Map<String, Object> stateFeatures) {
            this.requestId = requestId; this.deviceId = deviceId;
            this.sessionId = sessionId; this.requestTimeMs = requestTimeMs;
            this.action = action; this.reward = reward;
            this.stateFeatures = Map.copyOf(stateFeatures);
        }

        public long getRequestId() { return requestId; }
        public String getDeviceId() { return deviceId; }
        public String getSessionId() { return sessionId; }
        public long getRequestTimeMs() { return requestTimeMs; }
    }

    /** 真实曝光事件（客户端回调确认后写入） */
    public static class RealShowEvent {
        public final long requestId;
        public final String deviceId;
        public final long showTimeMs;    // Event Time

        public RealShowEvent(long requestId, String deviceId, long showTimeMs) {
            this.requestId = requestId; this.deviceId = deviceId; this.showTimeMs = showTimeMs;
        }

        public long getRequestId() { return requestId; }
        public String getDeviceId() { return deviceId; }
        public long getShowTimeMs() { return showTimeMs; }
    }

    /** 完整 RL 状态转移样本 (s_t, a_t, r_t, s_{t+1}) */
    public static class RLTransitionSample {
        public final BaseRLSample current;
        /** null 表示 terminal */
        public final BaseRLSample next;
        public final boolean hasJoinNext;
        /** 终止原因（terminal 时有值） */
        public final String terminalReason;

        private RLTransitionSample(BaseRLSample current, BaseRLSample next,
                                   boolean hasJoinNext, String terminalReason) {
            this.current = current; this.next = next;
            this.hasJoinNext = hasJoinNext; this.terminalReason = terminalReason;
        }

        public static RLTransitionSample joined(BaseRLSample cur, BaseRLSample nxt) {
            return new RLTransitionSample(cur, nxt, true, null);
        }

        public static RLTransitionSample terminal(BaseRLSample cur, String reason) {
            return new RLTransitionSample(cur, null, false, reason);
        }
    }

    // ─────────────────────────────────────────────
    //  KeyedCoProcessFunction 核心实现
    //  以 deviceId 为 Key，合并基础样本流与真实曝光事件流
    // ─────────────────────────────────────────────

    /**
     * 基于 Flink KeyedCoProcessFunction 的 RL 状态转移拼接。
     *
     * 工业落地时继承：
     *   org.apache.flink.streaming.api.functions.co.KeyedCoProcessFunction
     *     <String, BaseRLSample, RealShowEvent, RLTransitionSample>
     *
     * [MOCK] 所有状态通过设备维度的内存 Map 管理，逻辑与真实 Flink 一致。
     */
    public static class RLStateJoinFunction {

        // 等待下一刷状态的最长时间（Event Time）
        private static final long JOIN_WINDOW_MS = 300_000L;

        // ── Keyed State（以 deviceId 为 Key）──

        /**
         * 已确认真实曝光的请求序列：deviceId -> [reqId1, reqId2, ...]
         * [MOCK] 工业落地替换为: ListState<Long> realShowList;
         */
        private final Map<String, ListState<Long>> realShowLists = new HashMap<>();

        /**
         * 已完成 Label 拼接的基础样本：deviceId -> (reqId -> BaseRLSample)
         * [MOCK] 工业落地替换为: MapState<Long, BaseRLSample> sampleState;
         */
        private final Map<String, MapState<Long, BaseRLSample>> sampleStates = new HashMap<>();

        /**
         * 等待下一刷的请求状态：deviceId -> (reqId -> requestTimeMs)
         * [MOCK] 工业落地替换为: MapState<Long, Long> pendingRequests;
         */
        private final Map<String, MapState<Long, Long>> pendingRequests = new HashMap<>();

        /** 定时器服务（每个 deviceId 独立） */
        private final Map<String, TimerService> timerServices = new HashMap<>();

        /** 输出收集器 */
        private final Collector<RLTransitionSample> out = new Collector<>();

        // ── 基础样本流处理（processElement1）──

        /**
         * 处理来自上游 Label 拼接消费者的基础样本。
         * 工业落地时对应：
         *   public void processElement1(BaseRLSample sample,
         *       Context ctx, Collector<RLTransitionSample> out)
         */
        public void processElement1(BaseRLSample sample) {
            String deviceId = sample.deviceId;
            long reqId = sample.requestId;

            // 存储样本状态，等待真实曝光序列定位下一刷
            getSampleState(deviceId).put(reqId, sample);

            // 尝试立即拼接（若真实曝光序列已有下一刷）
            if (!tryJoin(deviceId, reqId, sample)) {
                // 注册等待截止定时器
                long deadline = sample.requestTimeMs + JOIN_WINDOW_MS;
                getPendingRequests(deviceId).put(reqId, sample.requestTimeMs);
                getTimerService(deviceId).registerEventTimeTimer(deadline);
            }
        }

        // ── 真实曝光事件流处理（processElement2）──

        /**
         * 处理来自行为日志的真实曝光事件。
         * 工业落地时对应：
         *   public void processElement2(RealShowEvent event,
         *       Context ctx, Collector<RLTransitionSample> out)
         */
        public void processElement2(RealShowEvent event) {
            String deviceId = event.deviceId;
            long reqId = event.requestId;

            // 追加到真实曝光序列（避免重复）
            ListState<Long> showList = getRealShowList(deviceId);
            List<Long> ids = new ArrayList<>();
            for (Long id : showList.get()) ids.add(id);
            if (!ids.contains(reqId)) {
                ids.add(reqId);
                showList.update(ids);
            }

            // 尝试触发所有等待此设备下一刷的样本拼接
            MapState<Long, Long> pending = getPendingRequests(deviceId);
            List<Long> toProcess = new ArrayList<>();
            for (Map.Entry<Long, Long> entry : pending.entries()) {
                toProcess.add(entry.getKey());
            }
            for (long pendingReqId : toProcess) {
                BaseRLSample sample = getSampleState(deviceId).get(pendingReqId);
                if (sample != null) {
                    tryJoin(deviceId, pendingReqId, sample);
                }
            }
        }

        // ── 定时器触发（onTimer）──

        /**
         * Event-time 定时器触发，处理等待超时的样本。
         * 工业落地时对应：
         *   public void onTimer(long timestamp,
         *       OnTimerContext ctx, Collector<RLTransitionSample> out)
         */
        public void onTimer(long timestamp, String deviceId) {
            MapState<Long, Long> pending = getPendingRequests(deviceId);
            List<Long> expired = new ArrayList<>();
            for (Map.Entry<Long, Long> entry : pending.entries()) {
                long deadline = entry.getValue() + JOIN_WINDOW_MS;
                if (timestamp >= deadline) {
                    expired.add(entry.getKey());
                }
            }
            for (long reqId : expired) {
                BaseRLSample sample = getSampleState(deviceId).get(reqId);
                if (sample != null) {
                    // 最后尝试一次拼接
                    if (!tryJoin(deviceId, reqId, sample)) {
                        // 真正超时，输出 terminal 样本
                        Long nextReqId = findNextRequest(deviceId, reqId);
                        String reason = nextReqId != null ? "state_not_ready" : "next_req_not_found";
                        out.collect(RLTransitionSample.terminal(sample, reason));
                        cleanupRequest(deviceId, reqId);
                    }
                }
                pending.remove(reqId);
            }
        }

        // ── 内部工具 ──

        /**
         * 尝试从真实曝光序列中定位下一刷并完成拼接。
         * @return 是否成功拼接（true 表示已出样并清理状态）
         */
        private boolean tryJoin(String deviceId, long reqId, BaseRLSample sample) {
            Long nextReqId = findNextRequest(deviceId, reqId);
            if (nextReqId == null) return false;

            BaseRLSample nextState = getSampleState(deviceId).get(nextReqId);
            if (nextState == null) return false;   // 下一刷样本尚未到达

            // Session 一致性校验
            if (sample.sessionId.equals(nextState.sessionId)) {
                out.collect(RLTransitionSample.joined(sample, nextState));
            } else {
                out.collect(RLTransitionSample.terminal(sample, "session_mismatch"));
            }
            cleanupRequest(deviceId, reqId);
            return true;
        }

        /** 在真实曝光序列中找到 reqId 的下一个请求 ID */
        private Long findNextRequest(String deviceId, long reqId) {
            List<Long> ids = new ArrayList<>();
            for (Long id : getRealShowList(deviceId).get()) ids.add(id);
            int pos = ids.indexOf(reqId);
            if (pos < 0 || pos + 1 >= ids.size()) return null;
            return ids.get(pos + 1);
        }

        private void cleanupRequest(String deviceId, long reqId) {
            getSampleState(deviceId).remove(reqId);
            getPendingRequests(deviceId).remove(reqId);
        }

        /** 推进 Watermark，触发所有到期定时器（仿真用） */
        public void advanceWatermark(long watermarkMs) {
            for (Map.Entry<String, TimerService> entry : timerServices.entrySet()) {
                Long ts;
                while ((ts = entry.getValue().pollNextExpired(watermarkMs)) != null) {
                    onTimer(ts, entry.getKey());
                }
            }
        }

        // ── State 懒加载辅助 ──

        private ListState<Long> getRealShowList(String deviceId) {
            return realShowLists.computeIfAbsent(deviceId, k -> new ListState<>());
        }

        private MapState<Long, BaseRLSample> getSampleState(String deviceId) {
            return sampleStates.computeIfAbsent(deviceId, k -> new MapState<>());
        }

        private MapState<Long, Long> getPendingRequests(String deviceId) {
            return pendingRequests.computeIfAbsent(deviceId, k -> new MapState<>());
        }

        private TimerService getTimerService(String deviceId) {
            return timerServices.computeIfAbsent(deviceId, k -> new TimerService());
        }

        public List<RLTransitionSample> getOutput() { return out.getAll(); }
    }

    // ─────────────────────────────────────────────
    //  Session Window 函数
    //  对应书中 code_flink_window_fn
    // ─────────────────────────────────────────────

    /**
     * Session Window 中处理完整会话片段，按事件时间排序后构建状态转移对。
     * 工业落地时对应：
     *   ProcessWindowFunction<BaseRLSample, RLTransitionSample, String, TimeWindow>.process()
     * [MOCK] 以下实现与 RLStateJoinFunction 的逻辑等价，但改为攒窗后批量处理。
     */
    public static class SessionWindowFunction {
        public List<RLTransitionSample> process(String sessionKey,
                Iterable<BaseRLSample> records) {
            Map<Long, BaseRLSample> latestByRequest = new HashMap<>();
            for (BaseRLSample record : records) {
                BaseRLSample prev = latestByRequest.get(record.requestId);
                if (prev == null || record.requestTimeMs > prev.requestTimeMs)
                    latestByRequest.put(record.requestId, record);
            }
            List<BaseRLSample> ordered = new ArrayList<>(latestByRequest.values());
            ordered.sort(Comparator.comparingLong(s -> s.requestTimeMs));
            List<RLTransitionSample> output = new ArrayList<>();
            for (int i = 0; i + 1 < ordered.size(); i++) {
                output.add(RLTransitionSample.joined(ordered.get(i), ordered.get(i + 1)));
            }
            if (!ordered.isEmpty()) {
                output.add(RLTransitionSample.terminal(
                    ordered.get(ordered.size() - 1), "session_window_closed"));
            }
            return output;
        }
    }

    // ─────────────────────────────────────────────
    //  buildJob — 真实 Flink 作业主干（对应书中 code_flink_rl_join）
    //
    //  工业落地步骤：
    //    1. pom.xml 加 flink-streaming-java / flink-connector-kafka（provided scope）
    //    2. 实现 BaseRLSampleDeserializer / RealShowEventDeserializer / RLSampleSerializer
    //    3. mvn package && flink run -c com.reco.offline.rl.FlinkRLJoiner job.jar
    //
    //  完整代码如下（取消注释后可直接使用）：
    //
    //  public static void buildJob(String bootstrapServers) throws Exception {
    //      StreamExecutionEnvironment env =
    //          StreamExecutionEnvironment.getExecutionEnvironment();
    //
    //      // EXACTLY_ONCE Checkpoint；外部化快照，任务取消后保留
    //      env.enableCheckpointing(60_000L, CheckpointingMode.EXACTLY_ONCE);
    //      env.getCheckpointConfig().setCheckpointTimeout(300_000L);
    //      env.getCheckpointConfig().enableExternalizedCheckpoints(
    //          ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
    //      env.setMaxParallelism(4096);
    //
    //      Properties kafkaProps = new Properties();
    //      kafkaProps.setProperty("bootstrap.servers", bootstrapServers);
    //      kafkaProps.setProperty("group.id", "rl-sample-joiner");
    //
    //      // ── 基础样本流：rl_base_samples ──
    //      // 上游 Label 拼接消费者输出 (s_t, a_t, r_t)，按 requestTimeMs 分配 Watermark
    //      DataStream<BaseRLSample> sampleStream = env
    //          .addSource(new FlinkKafkaConsumer<>(
    //              "rl_base_samples", new BaseRLSampleDeserializer(), kafkaProps))
    //          .name("BaseRLSampleSource")
    //          .assignTimestampsAndWatermarks(
    //              WatermarkStrategy.<BaseRLSample>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    //                  .withTimestampAssigner((s, t) -> s.requestTimeMs)
    //                  .withIdleness(Duration.ofSeconds(30)));
    //
    //      // ── 真实曝光流：real_show_events ──
    //      // 客户端回调确认后写入，用于恢复请求顺序
    //      DataStream<RealShowEvent> showStream = env
    //          .addSource(new FlinkKafkaConsumer<>(
    //              "real_show_events", new RealShowEventDeserializer(), kafkaProps))
    //          .name("RealShowEventSource")
    //          .assignTimestampsAndWatermarks(
    //              WatermarkStrategy.<RealShowEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    //                  .withTimestampAssigner((e, t) -> e.showTimeMs)
    //                  .withIdleness(Duration.ofSeconds(30)));
    //
    //      // ── 核心拼接：双流按 deviceId 分区后 connect ──
    //      // RLStateJoinFunction 维护曝光序列和等待样本，定时器超时后输出 terminal 样本
    //      DataStream<RLTransitionSample> output = sampleStream
    //          .keyBy(s -> s.deviceId)
    //          .connect(showStream.keyBy(e -> e.deviceId))
    //          .process(new RLStateJoinFunction())
    //          .name("RLStateJoinProcessor").uid("RLStateJoinProcessor");
    //
    //      // ── 结果写回 rl_transition_samples：事务型 Sink 与 Checkpoint 对齐 ──
    //      output.sinkTo(KafkaSink.<RLTransitionSample>builder()
    //          .setBootstrapServers(bootstrapServers)
    //          .setRecordSerializer(KafkaRecordSerializationSchema.builder()
    //              .setTopic("rl_transition_samples")
    //              .setValueSerializationSchema(new RLSampleSerializer())
    //              .build())
    //          .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)
    //          .build()).name("RLTransitionSampleSink");
    //
    //      env.execute("RLSampleJoiner");
    //  }
    // ─────────────────────────────────────────────

    // ─────────────────────────────────────────────
    //  buildPipeline — 内存仿真入口（单元测试用）
    // ─────────────────────────────────────────────

    /**
     * 以内存 List 模拟 Kafka Source / Sink 的仿真入口，供单元测试使用。
     * 生产作业入口结构见上方 buildJob 注释块。
     */
    public static List<RLTransitionSample> buildPipeline(
            List<BaseRLSample> sampleStream,
            List<RealShowEvent> showStream) {
        // 1. Watermark 策略: 有界乱序，允许最大 5s 乱序
        //    工业落地对应:
        //    WatermarkStrategy.<BaseRLSample>forBoundedOutOfOrderness(Duration.ofSeconds(5))
        //       .withTimestampAssigner((s, t) -> s.requestTimeMs)
        //       .withIdleness(Duration.ofSeconds(30))
        long allowedLatenessMs = 5_000L;

        // 2. [MOCK] 按 deviceId 分流，等同于 keyBy(BaseRLSample::getDeviceId)
        RLStateJoinFunction joinFn = new RLStateJoinFunction();

        // 3. 处理真实曝光流（processElement2）和基础样本流（processElement1）
        for (RealShowEvent event : showStream) {
            joinFn.processElement2(event);
        }
        for (BaseRLSample sample : sampleStream) {
            joinFn.processElement1(sample);
        }

        // 4. 推进 Watermark，触发超时定时器（onTimer）
        //    工业落地对应: Event-time Timer 在 Watermark >= deadline 时自动触发
        long maxEventTime = sampleStream.stream()
                .mapToLong(s -> s.requestTimeMs).max().orElse(0L);
        joinFn.advanceWatermark(maxEventTime + allowedLatenessMs + 1);

        // 5. Checkpoint + 事务型 Kafka Sink（注释说明）
        //    env.enableCheckpointing(60_000, CheckpointingMode.EXACTLY_ONCE);
        //    KafkaSink.<RLTransitionSample>builder()
        //       .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)...

        return joinFn.getOutput();
    }

    // ─────────────────────────────────────────────
    //  端到端 Demo
    // ─────────────────────────────────────────────

    public static List<RLTransitionSample> runDemo() {
        RLStateJoinFunction joinFn = new RLStateJoinFunction();

        long now = System.currentTimeMillis();
        long t1Ms = now - 500_000L;
        long t2Ms = now - 490_000L;

        // 基础样本（上游 Label 拼接消费者产出）
        BaseRLSample req1 = new BaseRLSample(
            1001L, "device_A", "session_X", t1Ms,
            "insert_live", 1.5, Map.of("user_id", 100L));
        BaseRLSample req2 = new BaseRLSample(
            1002L, "device_A", "session_X", t2Ms,
            "no_insert", 0.3, Map.of("user_id", 100L));

        // 真实曝光事件（先写入）
        joinFn.processElement2(new RealShowEvent(1001L, "device_A", t1Ms + 1000L));
        joinFn.processElement2(new RealShowEvent(1002L, "device_A", t2Ms + 1000L));

        // 基础样本流
        joinFn.processElement1(req1);
        joinFn.processElement1(req2);

        // 推进 Watermark，触发到期定时器
        joinFn.advanceWatermark(now + 10_000L);

        List<RLTransitionSample> result = joinFn.getOutput();
        System.out.println("=== Flink RL 状态转移输出 ===");
        for (RLTransitionSample ts : result) {
            if (ts.hasJoinNext) {
                System.out.printf("  JOINED reqId=%d -> nextReqId=%d reward=%.2f%n",
                    ts.current.requestId, ts.next.requestId, ts.current.reward);
            } else {
                System.out.printf("  TERMINAL reqId=%d reason=%s%n",
                    ts.current.requestId, ts.terminalReason);
            }
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println("=== ch02 Flink RL 状态转移 Demo ===");
        List<RLTransitionSample> transitions = runDemo();
        System.out.println("Total: " + transitions.size());
        long joined = transitions.stream().filter(t -> t.hasJoinNext).count();
        System.out.println("Joined: " + joined + ", Terminal: " + (transitions.size() - joined));
    }
}
