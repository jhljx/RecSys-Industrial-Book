package com.reco.offline.index;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

public class ItemIndexServiceTest {

    @Test
    void testDualBufferReload() {
        ItemIndexService.DualBufferIndex idx = new ItemIndexService.DualBufferIndex();
        List<ItemIndexService.ItemFeature> data = new ArrayList<>();
        for (int i = 1; i <= 10; i++) {
            data.add(new ItemIndexService.ItemFeature((long) i, i * 0.1, Map.of()));
        }
        idx.reload(data);
        assertEquals(10, idx.size());
        assertNotNull(idx.get(5L));
        assertNull(idx.get(99L));
    }

    @Test
    void testDualBufferAtomicSwap() throws InterruptedException {
        ItemIndexService.DualBufferIndex idx = new ItemIndexService.DualBufferIndex();
        idx.reload(List.of(new ItemIndexService.ItemFeature(1L, 0.5, Map.of())));

        // 并发读，不应崩溃
        Thread reader = new Thread(() -> {
            for (int i = 0; i < 100; i++) idx.get(1L);
        });
        Thread writer = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                idx.reload(List.of(new ItemIndexService.ItemFeature((long) i, i * 0.1, Map.of())));
            }
        });
        reader.start();
        writer.start();
        reader.join(2000);
        writer.join(2000);
    }

    @Test
    void testOrderedInvertedIndexQuery() {
        ItemIndexService.OrderedInvertedIndex idx = new ItemIndexService.OrderedInvertedIndex();
        idx.upsert(1L, 0.9, Set.of("cat:food"));
        idx.upsert(2L, 0.7, Set.of("cat:food"));
        idx.upsert(3L, 0.5, Set.of("cat:food"));
        idx.upsert(4L, 0.3, Set.of("cat:game"));

        List<ItemIndexService.InvertedEntry> result = idx.query("cat:food", 2);
        assertEquals(2, result.size());
        assertEquals(1L, result.get(0).itemId, "最高分 item 应排第一");
        assertEquals(2L, result.get(1).itemId);
    }

    @Test
    void testOrderedInvertedIndexUpsertUpdatesScore() {
        ItemIndexService.OrderedInvertedIndex idx = new ItemIndexService.OrderedInvertedIndex();
        idx.upsert(1L, 0.3, Set.of("tag:A"));
        idx.upsert(2L, 0.9, Set.of("tag:A"));
        // 更新 item 1 的分数为 1.0，应排第一
        idx.upsert(1L, 1.0, Set.of("tag:A"));

        List<ItemIndexService.InvertedEntry> result = idx.query("tag:A", 5);
        assertEquals(1L, result.get(0).itemId);
        assertEquals(1.0, result.get(0).sortScore, 1e-6);
    }

    @Test
    void testOrderedInvertedIndexRemove() {
        ItemIndexService.OrderedInvertedIndex idx = new ItemIndexService.OrderedInvertedIndex();
        idx.upsert(1L, 0.8, Set.of("tag:X"));
        idx.upsert(2L, 0.6, Set.of("tag:X"));
        idx.remove(1L);

        List<ItemIndexService.InvertedEntry> result = idx.query("tag:X", 10);
        assertEquals(1, result.size());
        assertEquals(2L, result.get(0).itemId);
    }


    @Test
    void testDemoEndToEnd() {
        Map<String, Object> result = ItemIndexService.runDemo();
        assertEquals(100, result.get("forward_index_size"));
        assertTrue((int) result.get("food_tag_size") > 0);
    }
}
