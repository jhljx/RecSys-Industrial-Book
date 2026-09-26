package com.reco.offline.rl;

import java.util.*;

/**
 * 强化学习样本流 — 完整可运行示例（基于 Redis 的 TD Learning 状态转移拼接）
 *
 * 数据流：
 *   请求日志 -> 特征预处理消费者 -> Label拼接消费者 -> 状态转移拼接消费者 -> 训练样本
 *
 * 状态转移：
 *   (s_t, a_t, r_t) -> 找到 ReqID_{t+1} -> 读取 s_{t+1} -> (s_t, a_t, r_t, s_{t+1})
 *
 * [MOCK] Redis 和 Kafka 均使用内存模拟实现。
 *        工业落地时请替换为真实 Jedis / Kafka 客户端，
 *        Snappy 压缩替换为实际 org.xerial:snappy-java。
 */
public class RLSampleFlow {

    // ─────────────────────────────────────────────
    //  核心数据结构
    // ─────────────────────────────────────────────

    /** RL 基础样本 (s_t, a_t, r_t) */
    public static class BaseRLSample {
        public final long requestId;
        public final String deviceId;
        public final String sessionId;
        public final long requestTimeMs;
        /** 决策动作: "insert_live" / "no_insert" */
        public final String action;
        /** 奖励（观看时长或点击次数等） */
        public final double reward;
        /** 状态特征（简化为 key-value map） */
        public final Map<String, Object> stateFeatures;

        public BaseRLSample(long requestId, String deviceId, String sessionId,
                            long requestTimeMs, String action, double reward,
                            Map<String, Object> stateFeatures) {
            this.requestId = requestId;
            this.deviceId = deviceId;
            this.sessionId = sessionId;
            this.requestTimeMs = requestTimeMs;
            this.action = action;
            this.reward = reward;
            this.stateFeatures = Map.copyOf(stateFeatures);
        }
    }

    /** 完整 RL 样本 (s_t, a_t, r_t, s_{t+1}) */
    public static class RLTransitionSample {
        public final BaseRLSample current;
        /** null 表示 terminal */
        public final BaseRLSample next;
        /** 是否成功关联下一刷 */
        public final boolean hasJoinNext;
        /** 截断原因（terminal 时有值） */
        public final String terminalReason;

        private RLTransitionSample(BaseRLSample current, BaseRLSample next,
                                   boolean hasJoinNext, String terminalReason) {
            this.current = current;
            this.next = next;
            this.hasJoinNext = hasJoinNext;
            this.terminalReason = terminalReason;
        }

        public static RLTransitionSample joined(BaseRLSample current, BaseRLSample next) {
            return new RLTransitionSample(current, next, true, null);
        }

        public static RLTransitionSample terminal(BaseRLSample current, String reason) {
            return new RLTransitionSample(current, null, false, reason);
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] Redis 模拟
    //  工业落地时替换为 Jedis 或 Lettuce
    // ─────────────────────────────────────────────

    public static class MockRedis {
        // 真实曝光序列: deviceId -> list of requestIds
        private final Map<String, List<Long>> showLists = new HashMap<>();
        // 请求状态: reqId -> state bytes (序列化后，此处用 map 模拟)
        private final Map<Long, byte[]> stateStore = new HashMap<>();
        // 状态标记: reqId -> flag
        private final Map<Long, String> stateFlags = new HashMap<>();
        // session id: reqId -> sessionId
        private final Map<Long, String> sessions = new HashMap<>();
        // 用户反馈: "fb_occ_{reqId}_{itemId}" -> set of actions
        private final Map<String, Set<String>> feedbackOcc = new HashMap<>();
        // 数值反馈: "fb_amt_{reqId}_{itemId}_{action}" -> value
        private final Map<String, Long> feedbackAmt = new HashMap<>();
        // 时间戳
        private final Map<Long, Long> feedbackTs = new HashMap<>();

        private static final int TIMELINE_MAX_LEN = 100;

        public void appendRealShow(String deviceId, long reqId) {
            String flagKey = "real_show_" + deviceId + "_" + reqId;
            if (stateFlags.containsKey((long) flagKey.hashCode())) return;
            stateFlags.put((long) flagKey.hashCode(), "1");

            List<Long> ids = showLists.computeIfAbsent(deviceId, k -> new ArrayList<>());
            ids.remove(reqId);
            ids.add(reqId);
            if (ids.size() > TIMELINE_MAX_LEN) {
                ids.subList(0, ids.size() - TIMELINE_MAX_LEN).clear();
            }
        }

        public Long findNextRequest(String deviceId, long reqId) {
            List<Long> ids = showLists.getOrDefault(deviceId, Collections.emptyList());
            int pos = ids.indexOf(reqId);
            if (pos < 0 || pos + 1 >= ids.size()) return null;
            return ids.get(pos + 1);
        }

        // [MOCK] 序列化为 bytes（实际应使用 Protobuf + Snappy 压缩）
        public void saveState(long reqId, BaseRLSample sample, String sessionId) {
            // 简化：用 reqId 的 bytes 表示（实际应是 Protobuf 序列化后 Snappy 压缩）
            stateStore.put(reqId, ("state:" + reqId).getBytes());
            stateFlags.put(reqId, "write_success");
            sessions.put(reqId, sessionId);
        }

        public BaseRLSample loadState(long reqId, Map<Long, BaseRLSample> sampleRegistry) {
            if (!stateStore.containsKey(reqId)) return null;
            // [MOCK] 实际应解压 + Protobuf 解析，此处直接从注册表查找
            return sampleRegistry.get(reqId);
        }

        public String getSessionId(long reqId) {
            return sessions.get(reqId);
        }

        public void evictState(long reqId) {
            stateStore.remove(reqId);
            stateFlags.put(reqId, "clear_success");
        }

        public void recordOccurFeedback(long reqId, long itemId, String action) {
            feedbackOcc.computeIfAbsent("fb_occ_" + reqId + "_" + itemId, k -> new HashSet<>())
                       .add(action);
        }

        public void recordAmountFeedback(long reqId, long itemId, String action, long delta) {
            feedbackAmt.merge("fb_amt_" + reqId + "_" + itemId + "_" + action,
                              Math.max(0, delta), Long::sum);
        }

        public void recordAmountFeedbackWithTs(long reqId, long itemId, String action, long delta, long eventTimeMs) {
            feedbackAmt.merge("fb_amt_" + reqId + "_" + itemId + "_" + action,
                              Math.max(0, delta), Long::sum);
            // 只在 eventTimeMs 比当前记录的更新时才更新时间戳
            feedbackTs.merge(reqId, eventTimeMs, Math::max);
        }

        public Set<String> getOccurFeedback(long reqId, long itemId) {
            return feedbackOcc.getOrDefault("fb_occ_" + reqId + "_" + itemId, Collections.emptySet());
        }

        public long getAmountFeedback(long reqId, long itemId, String action) {
            return feedbackAmt.getOrDefault("fb_amt_" + reqId + "_" + itemId + "_" + action, 0L);
        }

        public long getLastFeedbackTs(long reqId) {
            return feedbackTs.getOrDefault(reqId, 0L);
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] Kafka 模拟
    // ─────────────────────────────────────────────

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
    }

    // ─────────────────────────────────────────────
    //  反馈标签消费者
    //  对应书中 code_label_writer
    // ─────────────────────────────────────────────

    public static class FeedbackSinkService {
        private static final long LABEL_TTL_SECONDS = 7200L;

        private final MockRedis redis;

        public FeedbackSinkService(MockRedis redis) {
            this.redis = redis;
        }

        /** 发生型行为（click、进入直播间等） */
        public void recordOccurLabel(long reqId, long itemId, String actionName) {
            if (actionName == null || actionName.isBlank()) return;
            redis.recordOccurFeedback(reqId, itemId, actionName);
        }

        /** 数值型行为（观看时长、播放次数等） */
        public void recordAmountLabel(long reqId, long itemId, String actionName, long delta) {
            if (actionName == null) return;
            redis.recordAmountFeedback(reqId, itemId, actionName, delta);
        }

        /** 数值型行为，带事件时间（用于自适应等待窗口） */
        public void recordAmountLabelWithTs(long reqId, long itemId, String actionName, long delta, long eventTimeMs) {
            if (actionName == null) return;
            redis.recordAmountFeedbackWithTs(reqId, itemId, actionName, delta, eventTimeMs);
        }
    }

    // ─────────────────────────────────────────────
    //  Label 拼接消费者
    //  对应书中 code_wait_window + code_replay_join + code_join_label
    // ─────────────────────────────────────────────

    public static class LabelJoinConsumer {
        private static final long BASE_DELAY_MS      = 30_000L;
        private static final long BASE_MAX_DELAY_MS  = 120_000L;
        private static final long ACTION_EXTEND_MS   = 10_000L;

        private final MockRedis redis;
        private final MockKafka kafka;
        // 全局样本注册表，用于 mock 状态存储
        private final Map<Long, BaseRLSample> sampleRegistry;

        private final Map<Long, Integer> replayCounts = new HashMap<>();

        public LabelJoinConsumer(MockRedis redis, MockKafka kafka,
                                 Map<Long, BaseRLSample> sampleRegistry) {
            this.redis = redis;
            this.kafka = kafka;
            this.sampleRegistry = sampleRegistry;
        }

        public void consume(BaseRLSample sample, List<Long> itemIds) {
            long reqId = sample.requestId;
            long nowMs = System.currentTimeMillis();

            if (isInWaitWindow(reqId, sample.requestTimeMs, nowMs)) {
                int rc = replayCounts.merge(reqId, 1, Integer::sum);
                if (rc < 5) {
                    kafka.send("label_join_waiting", sample);
                    return;
                }
                // 达到重投上限，强制出样
            }

            // 拼接反馈 Label
            Map<Long, Map<String, Long>> labels = new HashMap<>();
            for (long itemId : itemIds) {
                Map<String, Long> lblMap = new HashMap<>();
                for (String action : redis.getOccurFeedback(reqId, itemId)) {
                    long v = redis.getAmountFeedback(reqId, itemId, action);
                    lblMap.put(action, v > 0 ? v : 1L);
                }
                labels.put(itemId, lblMap);
            }

            // 计算奖励（简化：live 观看时长 + click 权重）
            double reward = computeReward(reqId, itemIds);

            BaseRLSample withReward = new BaseRLSample(
                sample.requestId, sample.deviceId, sample.sessionId,
                sample.requestTimeMs, sample.action, reward, sample.stateFeatures
            );

            // 将状态写入 Redis 供后续状态转移消费者读取
            redis.saveState(reqId, withReward, sample.sessionId);
            sampleRegistry.put(reqId, withReward);

            // 将基础样本发给状态转移消费者
            kafka.send("rl_joint_sample", withReward);
            System.out.println("[LabelJoin] reqId=" + reqId + " reward=" + reward);
        }

        private boolean isInWaitWindow(long reqId, long requestTimeMs, long nowMs) {
            long keyActionTs = redis.getLastFeedbackTs(reqId);
            // keyActionTs == 0 表示无反馈，使用 requestTimeMs 作为基础锚点
            long baseTs = keyActionTs > 0 ? keyActionTs : requestTimeMs;
            long windowMs = keyActionTs > 0 ? BASE_DELAY_MS : BASE_MAX_DELAY_MS;
            if (nowMs - baseTs < windowMs) return true;
            // 有内层行为时进一步延长
            return keyActionTs > 0 && nowMs - keyActionTs < ACTION_EXTEND_MS;
        }

        private double computeReward(long reqId, List<Long> itemIds) {
            double total = 0.0;
            for (long itemId : itemIds) {
                total += redis.getAmountFeedback(reqId, itemId, "live_watch_time") * 0.001;
                total += redis.getAmountFeedback(reqId, itemId, "video_watch_time") * 0.0005;
                if (redis.getOccurFeedback(reqId, itemId).contains("click")) total += 1.0;
            }
            return total;
        }
    }

    // ─────────────────────────────────────────────
    //  状态转移拼接消费者
    //  对应书中 code_seq_index + code_state_write + code_state_read + code_join_next_main
    // ─────────────────────────────────────────────

    public static class TransitionJoinConsumer {
        private static final long JOIN_WINDOW_MS = 300_000L;

        private final MockRedis redis;
        private final MockKafka kafka;
        private final Map<Long, BaseRLSample> sampleRegistry;

        private final Map<Long, Integer> replayCounts = new HashMap<>();

        public TransitionJoinConsumer(MockRedis redis, MockKafka kafka,
                                      Map<Long, BaseRLSample> sampleRegistry) {
            this.redis = redis;
            this.kafka = kafka;
            this.sampleRegistry = sampleRegistry;
        }

        public void consume(BaseRLSample sample) {
            long reqId = sample.requestId;
            Long nextReqId = redis.findNextRequest(sample.deviceId, reqId);

            if (nextReqId != null) {
                BaseRLSample nextState = redis.loadState(nextReqId, sampleRegistry);
                if (nextState != null) {
                    String nextSession = redis.getSessionId(nextReqId);
                    if (sample.sessionId.equals(nextSession)) {
                        RLTransitionSample ts = RLTransitionSample.joined(sample, nextState);
                        kafka.send("rl_transition_topic", ts);
                        redis.evictState(nextReqId);
                        System.out.println("[TransitionJoin] SUCCESS reqId=" + reqId
                                + " -> nextReqId=" + nextReqId);
                        return;
                    } else {
                        RLTransitionSample ts = RLTransitionSample.terminal(sample, "session_mismatch");
                        kafka.send("rl_transition_topic", ts);
                        redis.evictState(nextReqId);
                        return;
                    }
                }
            }

            // 下一刷尚未到达，判断是否继续等待
            long nowMs = System.currentTimeMillis();
            if (nowMs - sample.requestTimeMs < JOIN_WINDOW_MS) {
                int rc = replayCounts.merge(reqId, 1, Integer::sum);
                if (rc < 5) {
                    kafka.send("transition_join_waiting", sample);
                    return;
                }
            }

            // 超时，输出 terminal 样本
            String reason = nextReqId != null ? "state_not_ready" : "next_req_not_found";
            RLTransitionSample ts = RLTransitionSample.terminal(sample, reason);
            kafka.send("rl_transition_topic", ts);
            System.out.println("[TransitionJoin] TERMINAL reqId=" + reqId + " reason=" + reason);
        }
    }

    // ─────────────────────────────────────────────
    //  端到端 Demo
    // ─────────────────────────────────────────────

    public static List<RLTransitionSample> runDemo() {
        MockRedis redis = new MockRedis();
        MockKafka kafka = new MockKafka();
        Map<Long, BaseRLSample> registry = new HashMap<>();

        FeedbackSinkService feedbackSink = new FeedbackSinkService(redis);
        LabelJoinConsumer labelJoin = new LabelJoinConsumer(redis, kafka, registry);
        TransitionJoinConsumer transJoin = new TransitionJoinConsumer(redis, kafka, registry);

        long now = System.currentTimeMillis();
        // 请求 t1 (很久以前，等待窗口已过)
        long t1Ms = now - 500_000L;
        BaseRLSample req1 = new BaseRLSample(
                1001L, "device_A", "session_X", t1Ms,
                "insert_live", 0.0, Map.of("user_id", 100L, "ctx", "home"));

        // 请求 t2 (同一 session，稍晚)
        long t2Ms = now - 490_000L;
        BaseRLSample req2 = new BaseRLSample(
                1002L, "device_A", "session_X", t2Ms,
                "no_insert", 0.0, Map.of("user_id", 100L, "ctx", "home"));

        // 真实曝光序列写入（顺序先 req1 再 req2）
        redis.appendRealShow("device_A", 1001L);
        redis.appendRealShow("device_A", 1002L);

        // 反馈: req1 有直播点击 + 观看时长（时间设为很久以前）
        feedbackSink.recordOccurLabel(1001L, 501L, "click");
        feedbackSink.recordAmountLabelWithTs(1001L, 501L, "live_watch_time", 120_000L, now - 480_000L);

        // 反馈: req2 有视频观看（时间也设为很久以前）
        feedbackSink.recordAmountLabelWithTs(1002L, 601L, "video_watch_time", 45_000L, now - 470_000L);

        // Label 拼接
        labelJoin.consume(req1, List.of(501L, 502L));
        labelJoin.consume(req2, List.of(601L, 602L));

        // 读取基础样本
        List<BaseRLSample> baseSamples = kafka.poll("rl_joint_sample");
        System.out.println("base samples: " + baseSamples.size());

        // 状态转移拼接
        for (BaseRLSample s : baseSamples) {
            transJoin.consume(s);
        }

        // 处理等待队列中可能有的重投
        List<BaseRLSample> waiting = kafka.poll("transition_join_waiting");
        for (BaseRLSample s : waiting) {
            transJoin.consume(s);
        }

        List<RLTransitionSample> result = kafka.poll("rl_transition_topic");
        System.out.println("transition samples: " + result.size());
        for (RLTransitionSample ts : result) {
            if (ts.hasJoinNext) {
                System.out.println("  JOINED reqId=" + ts.current.requestId
                        + " reward=" + ts.current.reward
                        + " nextReqId=" + ts.next.requestId);
            } else {
                System.out.println("  TERMINAL reqId=" + ts.current.requestId
                        + " reason=" + ts.terminalReason);
            }
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println("=== ch02 RL 样本流 Demo ===");
        List<RLTransitionSample> transitions = runDemo();
        System.out.println("Total transitions: " + transitions.size());
        long joined = transitions.stream().filter(t -> t.hasJoinNext).count();
        System.out.println("Joined: " + joined + ", Terminal: " + (transitions.size() - joined));
    }
}
