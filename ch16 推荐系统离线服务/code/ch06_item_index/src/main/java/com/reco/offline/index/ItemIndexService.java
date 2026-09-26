package com.reco.offline.index;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 物品索引服务 — 完整可运行示例
 *
 * 包含：
 *  1. DualBufferIndex        - 双 Buffer 全量索引发布（书中 code_dual_buffer_publish）
 *  2. OrderedInvertedIndex   - 有序倒排索引（书中 code_ordered_inverted_index）
 *  3. RealtimeIndexUpdater   - 实时增量更新（书中 code_realtime_index_update）
 *
 * [MOCK] 所有持久化操作使用内存存储；
 *        向量索引替换为 Faiss 或专有 ANN 库。
 */
public class ItemIndexService {

    // ─────────────────────────────────────────────
    //  数据结构
    // ─────────────────────────────────────────────

    public static class ItemFeature {
        public final long itemId;
        public final double score;
        public final Map<String, String> attributes;

        public ItemFeature(long itemId, double score, Map<String, String> attributes) {
            this.itemId = itemId;
            this.score = score;
            this.attributes = Map.copyOf(attributes);
        }
    }

    /** 倒排索引节点（itemId + 排序分） */
    public static class InvertedEntry implements Comparable<InvertedEntry> {
        public final long itemId;
        public final double sortScore;

        public InvertedEntry(long itemId, double sortScore) {
            this.itemId = itemId;
            this.sortScore = sortScore;
        }

        @Override
        public int compareTo(InvertedEntry o) {
            // 降序排列
            return Double.compare(o.sortScore, this.sortScore);
        }
    }

    // ─────────────────────────────────────────────
    //  双 Buffer 全量索引发布
    //  对应书中 code_dual_buffer_publish
    // ─────────────────────────────────────────────

    public static class DualBufferIndex {
        /** 当前读 Buffer（原子引用实现零拷贝切换） */
        private final AtomicReference<Map<Long, ItemFeature>> readBuffer =
                new AtomicReference<>(new ConcurrentHashMap<>());
        /** 写 Buffer */
        private volatile Map<Long, ItemFeature> writeBuffer = new ConcurrentHashMap<>();

        /**
         * 全量加载新索引到写 Buffer，然后原子切换。
         * 调用期间读请求仍可访问旧 Buffer，无锁等待。
         */
        public void reload(List<ItemFeature> newData) {
            Map<Long, ItemFeature> newBuffer = new ConcurrentHashMap<>();
            for (ItemFeature f : newData) {
                newBuffer.put(f.itemId, f);
            }
            writeBuffer = newBuffer;
            // 原子切换：旧 buffer 由 GC 回收
            readBuffer.set(newBuffer);
            System.out.println("[DualBufferIndex] reloaded items=" + newBuffer.size());
        }

        /** 并发安全的读取 */
        public ItemFeature get(long itemId) {
            return readBuffer.get().get(itemId);
        }

        /** 实时插入或更新单个物品文档（供 RealtimeIndexUpdater 调用） */
        public void upsert(ItemFeature doc) {
            readBuffer.get().put(doc.itemId, doc);
        }

        /** 从正排删除单个物品（供 RealtimeIndexUpdater 调用） */
        public void remove(long itemId) {
            readBuffer.get().remove(itemId);
        }

        public int size() {
            return readBuffer.get().size();
        }
    }

    // ─────────────────────────────────────────────
    //  有序倒排索引
    //  对应书中 code_ordered_inverted_index
    // ─────────────────────────────────────────────

    public static class OrderedInvertedIndex {
        /** tag -> sorted list of InvertedEntry */
        private final Map<String, TreeSet<InvertedEntry>> invertedMap = new ConcurrentHashMap<>();
        /** itemId -> set of tags */
        private final Map<Long, Set<String>> forwardMap = new ConcurrentHashMap<>();

        private static final int MAX_PER_TAG = 10_000;

        public void upsert(long itemId, double score, Set<String> tags) {
            Set<String> oldTags = forwardMap.get(itemId);
            // 移除旧标签中该 item 的倒排条目
            if (oldTags != null) {
                for (String tag : oldTags) {
                    removeFromInverted(tag, itemId);
                }
            }
            // 插入新条目
            for (String tag : tags) {
                TreeSet<InvertedEntry> bucket = invertedMap.computeIfAbsent(
                        tag, k -> new TreeSet<>());
                bucket.add(new InvertedEntry(itemId, score));
                // 超出上限时移除分数最低的
                while (bucket.size() > MAX_PER_TAG) {
                    bucket.pollLast();
                }
            }
            forwardMap.put(itemId, new HashSet<>(tags));
        }

        public void remove(long itemId) {
            Set<String> tags = forwardMap.remove(itemId);
            if (tags != null) {
                for (String tag : tags) removeFromInverted(tag, itemId);
            }
        }

        private void removeFromInverted(String tag, long itemId) {
            TreeSet<InvertedEntry> bucket = invertedMap.get(tag);
            if (bucket != null) {
                bucket.removeIf(e -> e.itemId == itemId);
            }
        }

        /** 按分数降序返回 topK 结果 */
        public List<InvertedEntry> query(String tag, int topK) {
            TreeSet<InvertedEntry> bucket = invertedMap.get(tag);
            if (bucket == null || bucket.isEmpty()) return List.of();
            List<InvertedEntry> result = new ArrayList<>();
            int count = 0;
            for (InvertedEntry e : bucket) {
                if (count++ >= topK) break;
                result.add(e);
            }
            return result;
        }

        public int tagSize(String tag) {
            TreeSet<InvertedEntry> b = invertedMap.get(tag);
            return b == null ? 0 : b.size();
        }
    }

    // ─────────────────────────────────────────────
    //  实时增量索引更新（Kafka Consumer）
    //  对应书中 code_realtime_index_update
    // ─────────────────────────────────────────────

    /**
     * 索引更新触发消息（来自 Kafka 主题 index-update-trigger）。
     * 消息只携带物品 ID、变化原因和产生时间，不包含完整特征；
     * 消费者收到后需重新读取当前文档并重建索引。
     */
    public static class IndexUpdateTrigger {
        public final long itemId;
        public final String reason;   // e.g. "CONTENT_UPDATED", "INTERACTION_CHANGED", "OFFLINE"
        public final long eventTimeMs;

        public IndexUpdateTrigger(long itemId, String reason, long eventTimeMs) {
            this.itemId = itemId;
            this.reason = reason;
            this.eventTimeMs = eventTimeMs;
        }
    }

    /**
     * 实时索引更新 Kafka Consumer。
     * [MOCK] MockKafkaConsumer 为内存队列；工业落地时替换为真实 KafkaConsumer。
     * consume() 正常返回后由外部 Kafka 框架提交位点；异常必须向上传播，不可吞掉。
     */
    public static class RealtimeIndexUpdater {
        private final DualBufferIndex forwardIndex;
        private final OrderedInvertedIndex invertedIndex;
        // [MOCK] 模拟 Kafka Consumer 拉取触发消息
        private final Queue<List<IndexUpdateTrigger>> mockKafkaQueue = new ArrayDeque<>();

        public RealtimeIndexUpdater(DualBufferIndex forwardIndex,
                                    OrderedInvertedIndex invertedIndex) {
            this.forwardIndex  = forwardIndex;
            this.invertedIndex = invertedIndex;
        }

        /**
         * 消费一批 IndexUpdateTrigger 消息并同步更新正排与倒排索引。
         * 1. 提取并去重本批物品 ID
         * 2. 重新加载每个物品的当前特征文档（模拟 RPC 读取）
         * 3. 不可见/已下线物品 → 同时从正排和倒排删除
         * 4. 可见物品 → 写入正排，并更新倒排
         * 异常直接向上传播，调用方（Kafka 框架）负责提交或回退位点。
         */
        public void consume(List<IndexUpdateTrigger> triggers) {
            // Step 1: 提取并去重物品 ID
            Set<Long> itemIds = new LinkedHashSet<>();
            for (IndexUpdateTrigger t : triggers) {
                if (t.itemId > 0) itemIds.add(t.itemId);
            }
            if (itemIds.isEmpty()) return;

            // Step 2: 批量重新加载当前特征文档（mock: 从 forwardIndex 读取已有文档）
            for (long itemId : itemIds) {
                ItemFeature doc = loadCurrentDoc(itemId);

                if (doc == null || isOffline(doc)) {
                    // Step 3: 不可见或已下线 → 从正排和倒排同时删除
                    removeFromForwardIndex(itemId);
                    invertedIndex.remove(itemId);
                    System.out.printf("[RealtimeIndexUpdater] deleted itemId=%d%n", itemId);
                } else {
                    // Step 4: 写入正排，并更新倒排
                    upsertForwardIndex(doc);
                    Set<String> tags = deriveTags(doc.score, doc.attributes);
                    invertedIndex.upsert(doc.itemId, doc.score, tags);
                    System.out.printf("[RealtimeIndexUpdater] updated itemId=%d score=%.4f tags=%s%n",
                            doc.itemId, doc.score, tags);
                }
            }
        }

        /**
         * [MOCK] 模拟从上游 RPC / Redis 重新读取物品当前完整文档。
         * 工业落地时调用内容理解服务、状态服务和统计服务，聚合为完整文档。
         */
        private ItemFeature loadCurrentDoc(long itemId) {
            // 先查正排中已有文档，作为 mock 数据源
            ItemFeature existing = forwardIndex.get(itemId);
            if (existing != null) return existing;
            // 若正排中不存在，模拟新增物品的特征文档
            return new ItemFeature(itemId, 0.5, Map.of("category", "general"));
        }

        /** [MOCK] 判断物品是否已下线 */
        private boolean isOffline(ItemFeature doc) {
            return "offline".equalsIgnoreCase(doc.attributes.getOrDefault("status", ""));
        }

        /** 写入正排索引 */
        private void upsertForwardIndex(ItemFeature doc) {
            forwardIndex.upsert(doc);
        }

        /** 从正排索引删除 */
        private void removeFromForwardIndex(long itemId) {
            forwardIndex.remove(itemId);
        }

        private Set<String> deriveTags(double score, Map<String, String> attributes) {
            Set<String> tags = new HashSet<>();
            String category = attributes.getOrDefault("category", "");
            if (!category.isBlank()) tags.add("cat:" + category);
            String author = attributes.getOrDefault("author_id", "");
            if (!author.isBlank()) tags.add("author:" + author);
            if (score > 0.8) tags.add("top_score");
            return tags;
        }

        /** 向 mock Kafka 队列注入触发消息（测试用） */
        public void mockPush(List<IndexUpdateTrigger> triggers) {
            mockKafkaQueue.offer(triggers);
        }

        /** 拉取并消费一批（测试用） */
        public void pollAndConsume() {
            List<IndexUpdateTrigger> batch = mockKafkaQueue.poll();
            if (batch != null) consume(batch);
        }
    }
    

    // ─────────────────────────────────────────────
    //  离线全量索引构建入口
    //  对应书中 code_offline_runner
    // ─────────────────────────────────────────────

    public static class OfflineRun {
        public final String id;
        public final java.time.Instant inputCutoff;
        public final String outputVersion;
        public OfflineRun(String id, java.time.Instant inputCutoff, String outputVersion) {
            this.id            = id;
            this.inputCutoff   = inputCutoff;
            this.outputVersion = outputVersion;
        }
    }

    /**
     * 离线全量索引构建器 — 对应书中 code_offline_runner。
     * [MOCK] 从快照读取物品特征，填充双 Buffer 后原子切换；
     *        工业落地时快照数据从 Hive / HDFS 读取，
     *        通过 RPC 通知在线服务刷新索引版本。
     */
    public static class OfflineRunner {
        private final DualBufferIndex forwardIndex;
        private final OrderedInvertedIndex invertedIndex;

        public OfflineRunner(DualBufferIndex forwardIndex,
                             OrderedInvertedIndex invertedIndex) {
            this.forwardIndex  = forwardIndex;
            this.invertedIndex = invertedIndex;
        }

        public OfflineRun run(List<ItemFeature> snapshot) {
            String runId = "run-" + System.currentTimeMillis();
            java.time.Instant cutoff = java.time.Instant.now();
            // 1. 全量写入写 Buffer 并原子切换
            forwardIndex.reload(snapshot);
            // 2. 重建倒排索引
            for (ItemFeature f : snapshot) {
                Set<String> tags = deriveTags(f.score, f.attributes);
                invertedIndex.upsert(f.itemId, f.score, tags);
            }
            String version = "v" + cutoff.getEpochSecond();
            System.out.printf("[OfflineRunner] run=%s version=%s items=%d%n",
                    runId, version, snapshot.size());
            return new OfflineRun(runId, cutoff, version);
        }

        private Set<String> deriveTags(double score, Map<String, String> attributes) {
            Set<String> tags = new HashSet<>();
            String cat = attributes.getOrDefault("category", "");
            if (!cat.isBlank()) tags.add("cat:" + cat);
            if (score > 0.8) tags.add("top_score");
            return tags;
        }
    }

    // ─────────────────────────────────────────────
    //  端到端 Demo
    // ─────────────────────────────────────────────

    public static Map<String, Object> runDemo() {
        DualBufferIndex forwardIndex = new DualBufferIndex();
        OrderedInvertedIndex invertedIndex = new OrderedInvertedIndex();
        RealtimeIndexUpdater updater = new RealtimeIndexUpdater(forwardIndex, invertedIndex);

        // 1. 全量加载
        List<ItemFeature> bulk = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            bulk.add(new ItemFeature((long) i, i * 0.01,
                    Map.of("category", i % 2 == 0 ? "game" : "food",
                           "author_id", String.valueOf(i % 10))));
        }
        forwardIndex.reload(bulk);

        // 2. 实时更新：模拟 Kafka 触发消息
        updater.consume(List.of(
                new IndexUpdateTrigger(1L, "CONTENT_UPDATED", System.currentTimeMillis()),
                new IndexUpdateTrigger(2L, "INTERACTION_CHANGED", System.currentTimeMillis()),
                new IndexUpdateTrigger(3L, "CONTENT_UPDATED", System.currentTimeMillis())
        ));
        // 模拟下线触发（status=offline 的物品会走删除路径）
        updater.consume(List.of(
                new IndexUpdateTrigger(99L, "OFFLINE", System.currentTimeMillis())
        ));

        // 3. 倒排查询
        List<InvertedEntry> foodItems = invertedIndex.query("cat:food", 5);
        List<InvertedEntry> topItems = invertedIndex.query("top_score", 10);

        System.out.println("Food category top 5: " + foodItems.stream()
                .map(e -> e.itemId + "(" + String.format("%.2f", e.sortScore) + ")")
                .toList());
        System.out.println("Top score items: " + topItems.stream()
                .map(e -> e.itemId + "(" + String.format("%.2f", e.sortScore) + ")")
                .toList());


        return Map.of(
                "forward_index_size", forwardIndex.size(),
                "food_tag_size", invertedIndex.tagSize("cat:food")
        );
    }

    public static void main(String[] args) {
        System.out.println("=== ch06 物品索引服务 Demo ===");
        Map<String, Object> result = runDemo();
        System.out.println("Result: " + result);
    }
}
