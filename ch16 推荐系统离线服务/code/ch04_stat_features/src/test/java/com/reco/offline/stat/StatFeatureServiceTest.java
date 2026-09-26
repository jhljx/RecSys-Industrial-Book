package com.reco.offline.stat;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

public class StatFeatureServiceTest {

    @Test
    void testTimeBucketCounterBasic() {
        StatFeatureService.TimeBucketCounter counter = new StatFeatureService.TimeBucketCounter();
        long now = System.currentTimeMillis();
        counter.increment(100L, "click", 1L, -1L, now);
        counter.increment(100L, "click", 1L, -1L, now - 1000);
        assertEquals(2L, counter.get(100L, "click", 1));
        assertEquals(2L, counter.get(100L, "click", 24));
    }

    @Test
    void testTimeBucketCounterDeduplication() {
        StatFeatureService.TimeBucketCounter counter = new StatFeatureService.TimeBucketCounter();
        long now = System.currentTimeMillis();
        // 同一 dedupId 只计一次
        counter.increment(200L, "show", 1L, 9001L, now);
        counter.increment(200L, "show", 1L, 9001L, now);  // 重复，应被去重
        counter.increment(200L, "show", 1L, 9002L, now);  // 不同 dedupId，正常计数
        assertEquals(2L, counter.get(200L, "show", 1));
    }

    @Test
    void testTimeBucketCounterExpiredEvents() {
        StatFeatureService.TimeBucketCounter counter = new StatFeatureService.TimeBucketCounter();
        long now = System.currentTimeMillis();
        long oldTs = now - 200 * 3600_000L; // 200 小时前，超出所有窗口
        counter.increment(300L, "click", 1L, -1L, oldTs);
        assertEquals(0L, counter.get(300L, "click", 168), "超出168h窗口的事件不应被计数");
    }

    @Test
    void testEmpiricalXtrSmoothing() {
        StatFeatureService.EmpiricalXtrService xtr =
                new StatFeatureService.EmpiricalXtrService(0.02, 0.5);

        // 0 曝光时应回落到 prior
        double ctr0 = xtr.smoothedCtr(0, 0);
        assertTrue(Math.abs(ctr0 - 0.02) < 1e-6, "零曝光 CTR 应等于 prior");

        // 大量曝光后应收敛到实际 CTR
        double ctr = xtr.smoothedCtr(10000L, 200L);
        assertTrue(ctr > 0.019 && ctr < 0.021, "大量样本后 CTR 应接近实际值 0.02");
    }

    @Test
    void testItemCounterSnapshotServiceRejectsInvalid() {
        assertThrows(IllegalArgumentException.class, () ->
                new StatFeatureService.ItemCounterSnapshotService(
                        new StatFeatureService.TimeBucketCounter(), -1.0, 0.5));
        assertThrows(IllegalArgumentException.class, () ->
                new StatFeatureService.ItemCounterSnapshotService(null, 0.02, 0.5));
    }

    @Test
    void testCounterIndexTrigger() {
        List<String> messages = new ArrayList<>();
        StatFeatureService.CounterIndexTrigger trigger =
                new StatFeatureService.CounterIndexTrigger(messages);
        trigger.trigger(501L, "click");
        trigger.trigger(502L, "show");
        assertEquals(2, messages.size());
        assertTrue(messages.get(0).contains("501"));
    }

    @Test
    void testDemoEndToEnd() {
        Map<Long, StatFeatureService.ItemStatistics> stats = StatFeatureService.runDemo();
        assertFalse(stats.isEmpty(), "Demo 应产出统计特征");
        for (var s : stats.values()) {
            assertTrue(s.empiricalCtr > 0, "经验 CTR 应大于 0（有 prior 平滑）");
        }
    }
}
