package com.reco.offline.behavior;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

public class BehaviorLogServiceTest {

    @Test
    void testActionRecordSerdes() {
        BehaviorLogService.ActionRecord rec = new BehaviorLogService.ActionRecord(
                1001L, 501L, 201L, "click", 1L, "home", "recommend", 1700000000000L);
        byte[] bytes = rec.toBytes();
        BehaviorLogService.ActionRecord decoded = BehaviorLogService.ActionRecord.fromBytes(bytes);
        assertEquals(rec.requestId, decoded.requestId);
        assertEquals(rec.itemId, decoded.itemId);
        assertEquals(rec.actionName, decoded.actionName);
        assertEquals(rec.eventTimeMs, decoded.eventTimeMs);
    }

    @Test
    void testActionListStoreAppendAndGet() {
        BehaviorLogService.ActionListStore store = new BehaviorLogService.ActionListStore();
        BehaviorLogService.ActionRecord rec = new BehaviorLogService.ActionRecord(
                1L, 2L, 3L, "click", 1L, "feed", "reco", System.currentTimeMillis());

        store.append("key:click", rec.toBytes(), 5, java.time.Duration.ofHours(1));
        assertEquals(1, store.size("key:click"));

        List<BehaviorLogService.ActionRecord> records = store.get("key:click");
        assertEquals(1, records.size());
        assertEquals(2L, records.get(0).itemId);
    }

    @Test
    void testActionListStoreMaxLength() {
        BehaviorLogService.ActionListStore store = new BehaviorLogService.ActionListStore();
        String key = "test:click";
        int maxLen = 3;
        for (int i = 1; i <= 5; i++) {
            BehaviorLogService.ActionRecord rec = new BehaviorLogService.ActionRecord(
                    (long) i, (long) i, 0L, "click", 1L, "", "", System.currentTimeMillis());
            store.append(key, rec.toBytes(), maxLen, java.time.Duration.ofHours(1));
        }
        assertEquals(maxLen, store.size(key));
        // 最旧的 3 条应被淘汰，保留最近 3 条
        List<BehaviorLogService.ActionRecord> recs = store.get(key);
        assertEquals(3L, recs.get(0).requestId); // 最老保留的是第 3 条
    }

    @Test
    void testShortSequenceServiceAppendAndGet() {
        BehaviorLogService.ActionListStore store = new BehaviorLogService.ActionListStore();
        BehaviorLogService.ShortSequenceService svc = new BehaviorLogService.ShortSequenceService(store);

        long now = System.currentTimeMillis();
        svc.appendAction(1001L, 100L, "dev_X", 501L, 201L, "click", 1L, "feed", "reco", now);
        svc.appendAction(1002L, 100L, "dev_X", 502L, 202L, "click", 1L, "feed", "reco", now + 1000);

        List<BehaviorLogService.ActionRecord> actions = svc.getRecentActions("dev_X", "click");
        assertEquals(2, actions.size());
        assertEquals(501L, actions.get(0).itemId);
        assertEquals(502L, actions.get(1).itemId);
    }

    @Test
    void testActionLogConsumerSkipsInvalidLogs() throws InterruptedException {
        BehaviorLogService.ActionListStore store = new BehaviorLogService.ActionListStore();
        BehaviorLogService.ShortSequenceService svc = new BehaviorLogService.ShortSequenceService(store);
        BehaviorLogService.InMemoryItemUpdateBuffer buf = new BehaviorLogService.InMemoryItemUpdateBuffer();

        try (BehaviorLogService.SingleThreadSerialExecutor exec =
                     new BehaviorLogService.SingleThreadSerialExecutor()) {

            BehaviorLogService.ActionLogConsumer consumer =
                    new BehaviorLogService.ActionLogConsumer(exec, svc, buf, Set.of());

            // invalid: no userId and no deviceId
            BehaviorLogService.ActionLog invalid = new BehaviorLogService.ActionLog(
                    0L, null, List.of("10001"), List.of(1L), List.of(1L),
                    List.of(10L), List.of(), "click", "feed", System.currentTimeMillis());
            consumer.consume(invalid);
            Thread.sleep(100);

            assertEquals(0, svc.getRecentActions("", "click").size(), "无效日志不应写入序列");
        }
    }

    @Test
    void testDemoEndToEnd() throws InterruptedException {
        List<BehaviorLogService.ActionRecord> records = BehaviorLogService.runDemo();
        assertFalse(records.isEmpty(), "Demo 应产出行为记录");

        boolean hasClick = records.stream().anyMatch(r -> r.actionName.equals("click"));
        boolean hasWatch = records.stream().anyMatch(r -> r.actionName.equals("watch_time"));
        assertTrue(hasClick, "应包含 click 行为");
        assertTrue(hasWatch, "应包含 watch_time 行为");
    }
}
