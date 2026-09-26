package com.reco.offline.rl;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

public class RLSampleFlowTest {

    // ── 原有 RLSampleFlow (Redis) 测试 ──

    @Test
    void testFeedbackSinkOccurAndAmount() {
        RLSampleFlow.MockRedis redis = new RLSampleFlow.MockRedis();
        RLSampleFlow.FeedbackSinkService sink = new RLSampleFlow.FeedbackSinkService(redis);

        sink.recordOccurLabel(1L, 10L, "click");
        sink.recordAmountLabel(1L, 10L, "live_watch_time", 30000L);

        assertTrue(redis.getOccurFeedback(1L, 10L).contains("click"));
        assertEquals(30000L, redis.getAmountFeedback(1L, 10L, "live_watch_time"));
    }

    @Test
    void testRealShowSequence() {
        RLSampleFlow.MockRedis redis = new RLSampleFlow.MockRedis();
        redis.appendRealShow("dev1", 100L);
        redis.appendRealShow("dev1", 101L);
        redis.appendRealShow("dev1", 102L);

        assertEquals(101L, redis.findNextRequest("dev1", 100L));
        assertEquals(102L, redis.findNextRequest("dev1", 101L));
        assertNull(redis.findNextRequest("dev1", 102L));
        assertNull(redis.findNextRequest("dev1", 999L));
    }

    @Test
    void testLabelJoinOutputsBaseRL() {
        RLSampleFlow.MockRedis redis = new RLSampleFlow.MockRedis();
        RLSampleFlow.MockKafka kafka = new RLSampleFlow.MockKafka();
        Map<Long, RLSampleFlow.BaseRLSample> registry = new HashMap<>();
        RLSampleFlow.FeedbackSinkService sink = new RLSampleFlow.FeedbackSinkService(redis);
        RLSampleFlow.LabelJoinConsumer joiner = new RLSampleFlow.LabelJoinConsumer(redis, kafka, registry);

        long now = System.currentTimeMillis();
        RLSampleFlow.BaseRLSample sample = new RLSampleFlow.BaseRLSample(
                9001L, "dev_test", "sess_1", now - 400_000L,
                "insert_live", 0.0, Map.of("uid", 1L));

        sink.recordOccurLabel(9001L, 10L, "click");
        joiner.consume(sample, List.of(10L, 11L));

        List<RLSampleFlow.BaseRLSample> base = kafka.poll("rl_joint_sample");
        assertEquals(1, base.size());
        assertTrue(base.get(0).reward > 0, "click 应产生正 reward");
    }

    @Test
    void testTransitionJoinSameSession() {
        RLSampleFlow.MockRedis redis = new RLSampleFlow.MockRedis();
        RLSampleFlow.MockKafka kafka = new RLSampleFlow.MockKafka();
        Map<Long, RLSampleFlow.BaseRLSample> registry = new HashMap<>();

        long now = System.currentTimeMillis();
        RLSampleFlow.BaseRLSample s1 = new RLSampleFlow.BaseRLSample(
                2001L, "dev_A", "sess_same", now - 600_000L, "insert_live", 1.5, Map.of());
        RLSampleFlow.BaseRLSample s2 = new RLSampleFlow.BaseRLSample(
                2002L, "dev_A", "sess_same", now - 590_000L, "no_insert", 0.5, Map.of());

        redis.appendRealShow("dev_A", 2001L);
        redis.appendRealShow("dev_A", 2002L);
        redis.saveState(2001L, s1, "sess_same");
        redis.saveState(2002L, s2, "sess_same");
        registry.put(2001L, s1);
        registry.put(2002L, s2);

        RLSampleFlow.TransitionJoinConsumer tc = new RLSampleFlow.TransitionJoinConsumer(redis, kafka, registry);
        tc.consume(s1);

        List<RLSampleFlow.RLTransitionSample> transitions = kafka.poll("rl_transition_topic");
        assertEquals(1, transitions.size());
        assertTrue(transitions.get(0).hasJoinNext, "同一 session 下应成功关联下一刷");
        assertEquals(2002L, transitions.get(0).next.requestId);
    }

    @Test
    void testTransitionJoinSessionMismatch() {
        RLSampleFlow.MockRedis redis = new RLSampleFlow.MockRedis();
        RLSampleFlow.MockKafka kafka = new RLSampleFlow.MockKafka();
        Map<Long, RLSampleFlow.BaseRLSample> registry = new HashMap<>();

        long now = System.currentTimeMillis();
        RLSampleFlow.BaseRLSample s1 = new RLSampleFlow.BaseRLSample(
                3001L, "dev_B", "sess_old", now - 600_000L, "no_insert", 0.0, Map.of());
        RLSampleFlow.BaseRLSample s2 = new RLSampleFlow.BaseRLSample(
                3002L, "dev_B", "sess_new", now - 200_000L, "insert_live", 0.0, Map.of());

        redis.appendRealShow("dev_B", 3001L);
        redis.appendRealShow("dev_B", 3002L);
        redis.saveState(3001L, s1, "sess_old");
        redis.saveState(3002L, s2, "sess_new");
        registry.put(3001L, s1);
        registry.put(3002L, s2);

        RLSampleFlow.TransitionJoinConsumer tc = new RLSampleFlow.TransitionJoinConsumer(redis, kafka, registry);
        tc.consume(s1);

        List<RLSampleFlow.RLTransitionSample> transitions = kafka.poll("rl_transition_topic");
        assertEquals(1, transitions.size());
        assertFalse(transitions.get(0).hasJoinNext, "不同 session 应输出 terminal 样本");
        assertEquals("session_mismatch", transitions.get(0).terminalReason);
    }

    @Test
    void testDemoEndToEnd() {
        List<RLSampleFlow.RLTransitionSample> result = RLSampleFlow.runDemo();
        assertFalse(result.isEmpty(), "Demo 应产出状态转移样本");

        long joined = result.stream().filter(t -> t.hasJoinNext).count();
        assertTrue(joined > 0, "至少有一个成功拼接的转移样本");
    }

    // ── Flink RL 状态转移测试 ──

    @Test
    void testFlinkRLJoinSameSession() {
        FlinkRLJoiner.RLStateJoinFunction fn = new FlinkRLJoiner.RLStateJoinFunction();

        long now = System.currentTimeMillis();
        long t1Ms = now - 500_000L;
        long t2Ms = now - 490_000L;

        FlinkRLJoiner.BaseRLSample s1 = new FlinkRLJoiner.BaseRLSample(
            4001L, "dev_X", "sess_A", t1Ms, "insert_live", 1.2, Map.of("uid", 200L));
        FlinkRLJoiner.BaseRLSample s2 = new FlinkRLJoiner.BaseRLSample(
            4002L, "dev_X", "sess_A", t2Ms, "no_insert", 0.4, Map.of("uid", 200L));

        // 先注册真实曝光序列
        fn.processElement2(new FlinkRLJoiner.RealShowEvent(4001L, "dev_X", t1Ms + 500L));
        fn.processElement2(new FlinkRLJoiner.RealShowEvent(4002L, "dev_X", t2Ms + 500L));

        // 基础样本流（两者均已完成 Label 拼接）
        fn.processElement1(s1);
        fn.processElement1(s2);

        // 推进 Watermark
        fn.advanceWatermark(now + 10_000L);

        List<FlinkRLJoiner.RLTransitionSample> output = fn.getOutput();
        assertFalse(output.isEmpty(), "Flink RL 应产出状态转移样本");

        FlinkRLJoiner.RLTransitionSample ts = output.stream()
            .filter(t -> t.current.requestId == 4001L).findFirst().orElse(null);
        assertNotNull(ts, "reqId=4001 应有输出");
        assertTrue(ts.hasJoinNext, "同一 session 应成功关联下一刷");
        assertEquals(4002L, ts.next.requestId);
    }

    @Test
    void testFlinkRLJoinSessionMismatch() {
        FlinkRLJoiner.RLStateJoinFunction fn = new FlinkRLJoiner.RLStateJoinFunction();

        long now = System.currentTimeMillis();
        long t1Ms = now - 600_000L;
        long t2Ms = now - 200_000L;

        FlinkRLJoiner.BaseRLSample s1 = new FlinkRLJoiner.BaseRLSample(
            5001L, "dev_Y", "sess_OLD", t1Ms, "no_insert", 0.0, Map.of());
        FlinkRLJoiner.BaseRLSample s2 = new FlinkRLJoiner.BaseRLSample(
            5002L, "dev_Y", "sess_NEW", t2Ms, "insert_live", 0.0, Map.of());

        fn.processElement2(new FlinkRLJoiner.RealShowEvent(5001L, "dev_Y", t1Ms + 500L));
        fn.processElement2(new FlinkRLJoiner.RealShowEvent(5002L, "dev_Y", t2Ms + 500L));
        fn.processElement1(s1);
        fn.processElement1(s2);
        fn.advanceWatermark(now + 10_000L);

        List<FlinkRLJoiner.RLTransitionSample> output = fn.getOutput();
        FlinkRLJoiner.RLTransitionSample ts = output.stream()
            .filter(t -> t.current.requestId == 5001L).findFirst().orElse(null);
        assertNotNull(ts, "reqId=5001 应有输出");
        assertFalse(ts.hasJoinNext, "不同 session 应为 terminal");
        assertEquals("session_mismatch", ts.terminalReason);
    }

    @Test
    void testFlinkRLDemoEndToEnd() {
        List<FlinkRLJoiner.RLTransitionSample> transitions = FlinkRLJoiner.runDemo();
        assertFalse(transitions.isEmpty(), "Flink RL Demo 应产出样本");
        long joined = transitions.stream().filter(t -> t.hasJoinNext).count();
        assertTrue(joined > 0, "至少 1 个成功关联的状态转移样本");
    }

    // ── RLSampleFlowService 测试（对应 tex 代码片段）──

    @Test
    void testRLFeedbackSinkService() {
        RLSampleFlowService.MockRedis redis = new RLSampleFlowService.MockRedis();
        RLSampleFlowService.FeedbackSinkService sink = new RLSampleFlowService.FeedbackSinkService(redis);

        sink.recordOccurLabel(100L, 50L, "click");
        sink.recordAmountLabel(100L, 50L, "watch_time", 20000L);

        Set<String> actions = redis.getOccurActions(100L, 50L);
        assertTrue(actions.contains("click"), "click 行为应被记录");
    }

    @Test
    void testRLFeedbackQueryService() {
        RLSampleFlowService.MockRedis redis = new RLSampleFlowService.MockRedis();
        RLSampleFlowService.FeedbackSinkService sink = new RLSampleFlowService.FeedbackSinkService(redis);
        RLSampleFlowService.FeedbackQueryService query = new RLSampleFlowService.FeedbackQueryService(redis);

        sink.recordOccurLabel(200L, 60L, "click");
        sink.recordOccurLabel(200L, 60L, "like");

        Map<Long, List<String>> result = query.queryOccurLabels(200L, List.of(60L, 61L));
        assertTrue(result.containsKey(60L), "物品 60 应有行为记录");
        assertTrue(result.get(60L).contains("click"), "click 应被读取");
        assertFalse(result.containsKey(61L), "物品 61 应无记录");
    }

    @Test
    void testRLSeqIndexService() {
        RLSampleFlowService.MockRedis redis = new RLSampleFlowService.MockRedis();
        RLSampleFlowService.SeqIndexService seqIndex = new RLSampleFlowService.SeqIndexService(redis);

        seqIndex.appendRealShow("user_A", 1001L);
        seqIndex.appendRealShow("user_A", 1002L);
        seqIndex.appendRealShow("user_A", 1003L);

        RLSampleFlowService.LookupResult result = seqIndex.findNextRequest("user_A", 1002L);
        assertTrue(result.isFound(), "应能找到下一刷");
        assertEquals(1003L, result.requestId, "下一刷 requestId 应为 1003");

        RLSampleFlowService.LookupResult missing = seqIndex.findNextRequest("user_A", 1003L);
        assertFalse(missing.isFound(), "最后一刷没有下一刷");
    }

    @Test
    void testRLRequestStateStore() {
        RLSampleFlowService.MockRedis redis = new RLSampleFlowService.MockRedis();
        RLSampleFlowService.RequestStateStore stateStore = new RLSampleFlowService.RequestStateStore(redis);

        long reqId = 3001L;
        RLSampleFlowService.TrainRecord record = RLSampleFlowService.TrainRecord.of(
                reqId, "device_B", "sess_2", System.currentTimeMillis(), "insert_live", List.of(801L));

        stateStore.saveRequestState(reqId, record);
        RLSampleFlowService.TrainRecord loaded = stateStore.loadRequestState(reqId);
        assertNotNull(loaded, "应能读取已保存的状态");
        assertEquals(reqId, loaded.getRequestId(), "requestId 应匹配");

        stateStore.evictRequestState(reqId);
        assertNull(stateStore.loadRequestState(reqId), "清理后状态应为 null");
    }

    @Test
    void testRLWaitWindow() {
        RLSampleFlowService.MockRedis redis = new RLSampleFlowService.MockRedis();
        RLSampleFlowService.FeedbackQueryService queryService = new RLSampleFlowService.FeedbackQueryService(redis);
        RLSampleFlowService.WaitWindowService waitWindow = new RLSampleFlowService.WaitWindowService(queryService);

        long now = System.currentTimeMillis();
        // 刚到的请求应在窗口内
        assertTrue(waitWindow.isInWaitWindow(5001L, now - 1000L), "新请求应在窗口内");
        // 400秒前的请求应超出窗口
        assertFalse(waitWindow.isInWaitWindow(5002L, now - 400_000L), "旧请求应超出窗口");
    }

    @Test
    void testRLSampleFlowServiceDemo() {
        List<RLSampleFlowService.TrainRecord> transitions = RLSampleFlowService.runDemo();
        assertNotNull(transitions, "Demo 应返回非 null 结果");
        // transitions 可能为空（等待窗口未过），这里只验证不报错
    }
}
