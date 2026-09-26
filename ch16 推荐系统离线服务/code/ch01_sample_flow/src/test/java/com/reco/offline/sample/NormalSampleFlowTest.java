package com.reco.offline.sample;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

public class NormalSampleFlowTest {

    // ── 原有 NormalSampleFlow 测试 ──

    @Test
    void testFeedbackStoreRecord() {
        NormalSampleFlow.MockRedis redis = new NormalSampleFlow.MockRedis();
        NormalSampleFlow.FeedbackStore store = new NormalSampleFlow.FeedbackStore(redis);

        long now = System.currentTimeMillis();
        store.record(new NormalSampleFlow.FeedbackEvent(1001L, 501L, "click", 1L, now));
        store.record(new NormalSampleFlow.FeedbackEvent(1001L, 502L, "watch_time", 15000L, now));

        Set<String> actions501 = store.getActions(1001L, 501L);
        assertTrue(actions501.contains("click"), "click 行为应被记录");

        long watchTime = store.getValue(1001L, 502L, "watch_time");
        assertEquals(15000L, watchTime, "watch_time 数值应正确累加");
    }

    @Test
    void testSampleJoinWhenNoFeedback() {
        NormalSampleFlow.MockRedis redis = new NormalSampleFlow.MockRedis();
        NormalSampleFlow.MockKafka kafka = new NormalSampleFlow.MockKafka();
        NormalSampleFlow.FeedbackStore store = new NormalSampleFlow.FeedbackStore(redis);
        NormalSampleFlow.SampleJoinConsumer joiner = new NormalSampleFlow.SampleJoinConsumer(store, kafka);

        long now = System.currentTimeMillis();
        SampleFlowTypes.FeatureReadyRequest req = SampleFlowTypes.FeatureReadyRequest.newBuilder()
                .setRequestId(9999L).setDeviceId("dev_test").setScene("test")
                .setRequestTimeMs(now - 500_000L)
                .build();

        joiner.consume(req);

        List<NormalSampleFlow.TrainingBatch> batches = kafka.poll("training_sample_stream");
        assertTrue(batches.isEmpty(), "无反馈时不应产出训练样本");
    }

    @Test
    void testSampleJoinWithFeedback() {
        NormalSampleFlow.MockRedis redis = new NormalSampleFlow.MockRedis();
        NormalSampleFlow.MockKafka kafka = new NormalSampleFlow.MockKafka();
        NormalSampleFlow.FeedbackStore store = new NormalSampleFlow.FeedbackStore(redis);
        NormalSampleFlow.SampleJoinConsumer joiner = new NormalSampleFlow.SampleJoinConsumer(store, kafka);

        long now = System.currentTimeMillis();
        SampleFlowService.FeaturePreprocessService svc = new SampleFlowService.FeaturePreprocessService();
        SampleFlowTypes.RawRequestWithFeatures raw = new SampleFlowTypes.RawRequestWithFeatures(
                2001L, now - 300_000L, "dev_a", "sess_a", "home", "v1",
                List.of(),
                List.of(new SampleFlowTypes.RawItem(10L, List.of()),
                        new SampleFlowTypes.RawItem(20L, List.of())));
        SampleFlowTypes.FeatureReadyRequest req2 = svc.preprocess(raw);
        assertNotNull(req2);

        store.record(new NormalSampleFlow.FeedbackEvent(2001L, 10L, "click", 1L, now - 300_000));
        store.record(new NormalSampleFlow.FeedbackEvent(2001L, 10L, "watch_time", 5000L, now - 290_000));

        joiner.consume(req2);

        List<NormalSampleFlow.TrainingBatch> batches = kafka.poll("training_sample_stream");
        assertEquals(1, batches.size(), "有反馈时应产出 1 个训练样本");

        NormalSampleFlow.TrainingBatch b = batches.get(0);
        assertEquals(2001L, b.requestId);
        assertTrue(b.labels.containsKey(10L), "物品 10 应有 Label");
        assertTrue(b.labels.get(10L).containsKey("click"), "click Label 应存在");
    }

    @Test
    void testWaitWindowReplay() {
        NormalSampleFlow.MockRedis redis = new NormalSampleFlow.MockRedis();
        NormalSampleFlow.MockKafka kafka = new NormalSampleFlow.MockKafka();
        NormalSampleFlow.FeedbackStore store = new NormalSampleFlow.FeedbackStore(redis);
        NormalSampleFlow.SampleJoinConsumer joiner = new NormalSampleFlow.SampleJoinConsumer(store, kafka);

        long now = System.currentTimeMillis();
        SampleFlowService.FeaturePreprocessService svc = new SampleFlowService.FeaturePreprocessService();
        SampleFlowTypes.RawRequestWithFeatures raw = new SampleFlowTypes.RawRequestWithFeatures(
                3001L, now - 1_000L, "dev_b", "sess_b", "live", "v1",
                List.of(),
                List.of(new SampleFlowTypes.RawItem(100L, List.of())));
        SampleFlowTypes.FeatureReadyRequest req = svc.preprocess(raw);
        assertNotNull(req);

        joiner.consume(req);

        List<SampleFlowTypes.FeatureReadyRequest> waiting = kafka.poll("sample_join_waiting");
        assertEquals(1, waiting.size(), "在等待窗口内应写回等待队列");
        List<NormalSampleFlow.TrainingBatch> batches = kafka.poll("training_sample_stream");
        assertTrue(batches.isEmpty(), "等待期间不应输出训练样本");
    }

    @Test
    void testDemoEndToEnd() {
        NormalSampleFlow.TrainingBatch b = NormalSampleFlow.runDemo();
        assertNotNull(b, "Demo 应产出训练样本");
        assertTrue(b.hasValidFeedback(), "训练样本应包含有效反馈");
    }

    // ── Flink 样本拼接测试 ──

    @Test
    void testFlinkJoinWithKeyFeedback() {
        FlinkSampleJoiner.SampleJoinFunction fn = new FlinkSampleJoiner.SampleJoinFunction();

        long now = System.currentTimeMillis();
        long reqTime = now - 200_000L;

        fn.processElement1(new FlinkSampleJoiner.RequestEvent(
            2001L, "device_B", "sess_1", "home_feed",
            reqTime, List.of(501L, 502L)
        ));
        // 关键反馈：click
        fn.processElement2(new FlinkSampleJoiner.FeedbackEvent(
            2001L, 501L, "click", 1L, reqTime + 5000L, true
        ));
        // 非关键反馈：watch_time
        fn.processElement2(new FlinkSampleJoiner.FeedbackEvent(
            2001L, 502L, "watch_time", 30_000L, reqTime + 8000L, false
        ));

        // 推进 Watermark（超过 BASE_WAIT 截止时间）
        fn.advanceWatermark(reqTime + 200_000L);

        List<FlinkSampleJoiner.JoinedSample> output = fn.getOutput();
        assertFalse(output.isEmpty(), "Flink 应产出至少 1 个样本");

        FlinkSampleJoiner.JoinedSample s = output.get(0);
        assertEquals(2001L, s.requestId);
        assertTrue(s.hasValidFeedback(), "样本应包含有效反馈");
    }

    @Test
    void testFlinkJoinTimeout() {
        FlinkSampleJoiner.SampleJoinFunction fn = new FlinkSampleJoiner.SampleJoinFunction();

        long now = System.currentTimeMillis();
        long reqTime = now - 300_000L;

        fn.processElement1(new FlinkSampleJoiner.RequestEvent(
            3001L, "device_C", "sess_2", "search",
            reqTime, List.of(701L)
        ));
        // 不发送任何反馈，仅推进 Watermark 触发超时
        fn.advanceWatermark(reqTime + 200_000L);

        List<FlinkSampleJoiner.JoinedSample> output = fn.getOutput();
        // 超时后出样，但无反馈（hasValidFeedback = false）
        assertEquals(1, output.size(), "超时后应出样一条");
        assertFalse(output.get(0).hasValidFeedback(), "无反馈的样本 hasValidFeedback 应为 false");
    }

    @Test
    void testFlinkDemoEndToEnd() {
        List<FlinkSampleJoiner.JoinedSample> samples = FlinkSampleJoiner.runDemo();
        assertFalse(samples.isEmpty(), "Flink Demo 应产出样本");
        long withFeedback = samples.stream().filter(FlinkSampleJoiner.JoinedSample::hasValidFeedback).count();
        assertTrue(withFeedback > 0, "至少 1 个样本应有有效反馈");
    }

    // ── SampleFlowService 测试（对应 tex 代码片段）──

    @Test
    void testFeaturePreprocessService() {
        SampleFlowService.FeaturePreprocessService service = new SampleFlowService.FeaturePreprocessService();
        long now = System.currentTimeMillis();
        SampleFlowTypes.RawRequestWithFeatures raw = new SampleFlowTypes.RawRequestWithFeatures(
                5001L, now, "device_X", "sess_X", "v1",
                List.of(SampleFlowTypes.RawFeature.of("device_type", "android"),
                        SampleFlowTypes.RawFeature.of("age_bucket", 25.0)),
                List.of(new SampleFlowTypes.RawItem(901L,
                        List.of(SampleFlowTypes.RawFeature.of("category", "sports"),
                                SampleFlowTypes.RawFeature.of("score", 0.8))))
        );
        SampleFlowTypes.FeatureReadyRequest result = service.preprocess(raw);
        assertNotNull(result, "preprocess 不应返回 null");
        assertEquals(5001L, result.getRequestId(), "requestId 应保留");
        assertFalse(result.getCommonFeatures().isEmpty(), "公共特征不应为空");
    }

    @Test
    void testFeedbackStorePbRecord() {
        SampleFlowService.MockRedisClient redis = new SampleFlowService.MockRedisClient();
        SampleFlowService.FeedbackStore store = new SampleFlowService.FeedbackStore(redis);
        long now = System.currentTimeMillis();

        store.record(new SampleFlowTypes.FeedbackEventPb(3001L, 601L, "click", 1L, now, true));
        store.record(new SampleFlowTypes.FeedbackEventPb(3001L, 601L, "watch_time", 12000L, now, false));

        Map<Long, Set<SampleFlowTypes.FeedbackKind>> actions =
                store.batchGetActions(3001L, List.of(601L, 602L));
        assertTrue(actions.containsKey(601L), "物品 601 应有行为记录");
        assertTrue(actions.get(601L).contains(SampleFlowTypes.FeedbackKind.CLICK), "click 行为应被记录");
    }

    @Test
    void testSampleJoinServiceEndToEnd() {
        SampleFlowService.MockRedisClient redis = new SampleFlowService.MockRedisClient();
        SampleFlowService.MockKafkaClient kafka = new SampleFlowService.MockKafkaClient();
        SampleFlowService.FeedbackStore store = new SampleFlowService.FeedbackStore(redis);
        SampleFlowService.SampleJoinService joiner = new SampleFlowService.SampleJoinService(store, kafka);

        long now = System.currentTimeMillis();

        // 先写反馈
        store.record(new SampleFlowTypes.FeedbackEventPb(4001L, 801L, "click", 1L, now - 50_000L, true));

        // 构造请求（等待窗口已过）
        SampleFlowTypes.JoinRequest request = SampleFlowTypes.JoinRequest.newBuilder()
                .setRequestId(4001L).setDeviceId("device_Y").setScene("feed")
                .setRequestTimeMs(now - 300_000L)
                .addAllItemIds(List.of(801L, 802L))
                .build();

        joiner.consume(request);

        List<SampleFlowTypes.TrainingBatchPb> batches = kafka.poll("training_sample_stream");
        assertEquals(1, batches.size(), "应产出 1 个训练样本");
        assertTrue(batches.get(0).hasValidFeedback(), "训练样本应有有效反馈");
    }

    @Test
    void testHistoryFeatureService() {
        SampleFlowService.HistoryFeatureService service = new SampleFlowService.HistoryFeatureService(3_600_000L);
        long now = System.currentTimeMillis();

        SampleFlowTypes.BehaviorSequence history = new SampleFlowTypes.BehaviorSequence(List.of(
                new SampleFlowTypes.BehaviorEvent(101L, SampleFlowTypes.FeedbackKind.CLICK, 1L, now - 1000L),
                new SampleFlowTypes.BehaviorEvent(101L, SampleFlowTypes.FeedbackKind.WATCH_TIME, 30000L, now - 500L)
        ));

        SampleFlowTypes.ItemSample.Builder sample = new SampleFlowTypes.ItemSample.Builder().setItemId(101L);
        service.appendHistoryFeatures(sample, 101L, history, now);

        SampleFlowTypes.ItemSample built = sample.build();
        assertFalse(built.intFeatures.isEmpty(), "历史特征不应为空");
    }

    @Test
    void testCascadeSampleService() {
        SampleFlowService.MockRedisClient redis = new SampleFlowService.MockRedisClient();
        SampleFlowService.FeedbackStore store = new SampleFlowService.FeedbackStore(redis);
        SampleFlowService.CascadeSampleService service = new SampleFlowService.CascadeSampleService(store);

        long now = System.currentTimeMillis();
        store.record(new SampleFlowTypes.FeedbackEventPb(5001L, 201L, "click", 1L, now, true));

        SampleFlowTypes.JoinRequest request = SampleFlowTypes.JoinRequest.newBuilder()
                .setRequestId(5001L).setDeviceId("device_Z").setScene("feed")
                .setRequestTimeMs(now - 60_000L).addAllItemIds(List.of(201L, 202L, 203L))
                .build();

        SampleFlowService.CascadeTrace trace = new SampleFlowService.CascadeTrace(
                List.of(new SampleFlowService.RankedItem(201L, 1))
        );
        List<SampleFlowService.StageCandidate> candidates = List.of(
                new SampleFlowService.StageCandidate(201L, "rank", 1, 0.9f, "rank_top", 1.0f, true, List.of()),
                new SampleFlowService.StageCandidate(202L, "prerank", 5, 0.7f, "prerank_drop", 0.5f, false, List.of()),
                new SampleFlowService.StageCandidate(203L, "recall", 10, 0.5f, "recall_drop", 0.3f, false, List.of())
        );

        SampleFlowTypes.TrainingBatchPb batch = service.buildCascadeBatch(request, trace, candidates);
        assertNotNull(batch, "级联样本不应为 null");
        assertEquals(3, batch.getItemCount(), "应包含 3 个候选物品");
        assertTrue(batch.hasValidFeedback(), "实际曝光物品应有有效反馈");
    }

    @Test
    void testCascadeSampleJoinConsumerSkipsWithoutCandidates() {
        NormalSampleFlow.MockRedis redis = new NormalSampleFlow.MockRedis();
        NormalSampleFlow.MockKafka kafka = new NormalSampleFlow.MockKafka();
        NormalSampleFlow.FeedbackStore store = new NormalSampleFlow.FeedbackStore(redis);
        NormalSampleFlow.CascadeSampleJoinConsumer consumer =
                new NormalSampleFlow.CascadeSampleJoinConsumer(store, kafka);

        long now = System.currentTimeMillis();
        // 无 cascadeCandidates 的普通请求
        SampleFlowTypes.FeatureReadyRequest req = SampleFlowTypes.FeatureReadyRequest.newBuilder()
                .setRequestId(8001L).setDeviceId("dev_c").setScene("feed")
                .setRequestTimeMs(now - 500_000L)
                .build();

        consumer.consume(req);

        List<NormalSampleFlow.CascadeTrainingBatch> batches =
                kafka.poll("cascade_training_sample_stream");
        assertTrue(batches.isEmpty(), "无 cascadeCandidates 时不应产出级联样本");
    }

    @Test
    void testCascadeSampleJoinConsumerWithCandidates() {
        NormalSampleFlow.MockRedis redis = new NormalSampleFlow.MockRedis();
        NormalSampleFlow.MockKafka kafka = new NormalSampleFlow.MockKafka();
        NormalSampleFlow.FeedbackStore store = new NormalSampleFlow.FeedbackStore(redis);
        NormalSampleFlow.CascadeSampleJoinConsumer consumer =
                new NormalSampleFlow.CascadeSampleJoinConsumer(store, kafka);

        long now = System.currentTimeMillis();

        // 写入曝光 item 的真实反馈
        store.record(new NormalSampleFlow.FeedbackEvent(9001L, 301L, "click", 1L, now - 400_000L));
        store.record(new NormalSampleFlow.FeedbackEvent(9001L, 301L, "watch_time", 20_000L, now - 390_000L));

        // 构造带 cascadeCandidates 的 FeatureReadyRequest（含曝光 item + 未曝光候选）
        SampleFlowTypes.EncodedStagedItem candidate1 = new SampleFlowTypes.EncodedStagedItem.Builder()
                .setItemId(401L).setStage(SampleFlowTypes.ItemStage.RANKED_NOT_SHOWN)
                .setRank(8).setScore(0.6f).build();
        SampleFlowTypes.EncodedStagedItem candidate2 = new SampleFlowTypes.EncodedStagedItem.Builder()
                .setItemId(402L).setStage(SampleFlowTypes.ItemStage.RETRIEVED_NOT_RANKED)
                .setRank(20).setScore(0.4f).build();

        SampleFlowTypes.FeatureReadyRequest.Builder reqBuilder =
                SampleFlowTypes.FeatureReadyRequest.newBuilder()
                        .setRequestId(9001L).setDeviceId("dev_d").setScene("feed")
                        .setRequestTimeMs(now - 500_000L);
        reqBuilder.addItemBuilder().setItemId(301L);  // 曝光 item
        reqBuilder.addCascadeCandidate(candidate1);
        reqBuilder.addCascadeCandidate(candidate2);
        SampleFlowTypes.FeatureReadyRequest reqWithItems = reqBuilder.build();

        consumer.consume(reqWithItems);

        List<NormalSampleFlow.CascadeTrainingBatch> batches =
                kafka.poll("cascade_training_sample_stream");
        assertEquals(1, batches.size(), "应产出 1 个级联训练批次");

        NormalSampleFlow.CascadeTrainingBatch batch = batches.get(0);
        assertEquals(9001L, batch.requestId);
        assertTrue(batch.hasExposedFeedback(), "曝光 item 应有有效反馈");
        assertEquals(2, batch.candidates.size(), "应包含 2 个未曝光候选");
        assertEquals(SampleFlowTypes.ItemStage.RANKED_NOT_SHOWN, batch.candidates.get(0).stage);
        assertEquals(SampleFlowTypes.ItemStage.RETRIEVED_NOT_RANKED, batch.candidates.get(1).stage);
    }

    @Test
    void testCascadeSampleJoinConsumerWaitWindow() {
        NormalSampleFlow.MockRedis redis = new NormalSampleFlow.MockRedis();
        NormalSampleFlow.MockKafka kafka = new NormalSampleFlow.MockKafka();
        NormalSampleFlow.FeedbackStore store = new NormalSampleFlow.FeedbackStore(redis);
        NormalSampleFlow.CascadeSampleJoinConsumer consumer =
                new NormalSampleFlow.CascadeSampleJoinConsumer(store, kafka);

        long now = System.currentTimeMillis();

        SampleFlowTypes.EncodedStagedItem candidate = new SampleFlowTypes.EncodedStagedItem.Builder()
                .setItemId(501L).setStage(SampleFlowTypes.ItemStage.RANKED_NOT_SHOWN)
                .setRank(5).setScore(0.7f).build();

        // 请求时间为 1 秒前，在等待窗口内
        SampleFlowTypes.FeatureReadyRequest.Builder reqBuilder2 =
                SampleFlowTypes.FeatureReadyRequest.newBuilder()
                        .setRequestId(7001L).setDeviceId("dev_e").setScene("feed")
                        .setRequestTimeMs(now - 1_000L);
        reqBuilder2.addItemBuilder().setItemId(601L);
        reqBuilder2.addCascadeCandidate(candidate);
        SampleFlowTypes.FeatureReadyRequest req = reqBuilder2.build();

        consumer.consume(req);

        List<SampleFlowTypes.FeatureReadyRequest> waiting = kafka.poll("sample_join_waiting");
        assertEquals(1, waiting.size(), "在等待窗口内应写回等待队列");
        List<NormalSampleFlow.CascadeTrainingBatch> batches =
                kafka.poll("cascade_training_sample_stream");
        assertTrue(batches.isEmpty(), "等待期间不应输出级联训练样本");
    }
}
