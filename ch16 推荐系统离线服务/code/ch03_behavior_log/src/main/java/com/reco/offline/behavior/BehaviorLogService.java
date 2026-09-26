package com.reco.offline.behavior;

import java.util.*;
import java.util.concurrent.*;

/**
 * 用户行为日志解析服务 — 完整可运行示例
 *
 * 包含：
 *  1. ActionLogConsumer    - Kafka 行为消费者（书中 code_action_log_consumer）
 *  2. ShortSequenceService - 短期兴趣序列服务（书中 code_short_sequence_store）
 *  3. ActionListStore      - 内存存储（[MOCK]，工业落地时替换为 Redis）
 *
 * [MOCK] ActionListStore 使用进程内 HashMap + Deque，进程重启后数据丢失。
 *        工业落地时需替换为 Redis LPUSH / LTRIM / EXPIRE 原子操作。
 *
 * [MOCK] KeySerialExecutor 使用单线程线程池模拟串行执行，
 *        工业落地时替换为按 subjectKey 哈希到专用线程的并发队列。
 */
public class BehaviorLogService {

    // ─────────────────────────────────────────────
    //  数据结构
    // ─────────────────────────────────────────────

    public static class ActionLog {
        public final long userId;
        public final String deviceId;
        public final List<String> requestIds;
        public final List<Long> itemIds;
        public final List<Long> actionValues;
        public final List<Long> authorIds;
        public final List<Map<String, String>> itemAttributes;
        public final String actionName;
        public final String requestType;
        public final long eventTimeMs;

        public ActionLog(long userId, String deviceId, List<String> requestIds,
                         List<Long> itemIds, List<Long> actionValues, List<Long> authorIds,
                         List<Map<String, String>> itemAttributes, String actionName,
                         String requestType, long eventTimeMs) {
            this.userId = userId;
            this.deviceId = deviceId;
            this.requestIds = requestIds != null ? requestIds : List.of();
            this.itemIds = itemIds != null ? itemIds : List.of();
            this.actionValues = actionValues != null ? actionValues : List.of();
            this.authorIds = authorIds != null ? authorIds : List.of();
            this.itemAttributes = itemAttributes != null ? itemAttributes : List.of();
            this.actionName = actionName;
            this.requestType = requestType != null ? requestType : "";
            this.eventTimeMs = eventTimeMs;
        }
    }

    public static class ActionRecord {
        public final long requestId;
        public final long itemId;
        public final long authorId;
        public final String actionName;
        public final long actionValue;
        public final String requestType;
        public final String sourceType;
        public final long eventTimeMs;

        public ActionRecord(long requestId, long itemId, long authorId, String actionName,
                            long actionValue, String requestType, String sourceType, long eventTimeMs) {
            this.requestId = requestId;
            this.itemId = itemId;
            this.authorId = authorId;
            this.actionName = actionName;
            this.actionValue = actionValue;
            this.requestType = requestType;
            this.sourceType = sourceType;
            this.eventTimeMs = eventTimeMs;
        }

        public byte[] toBytes() {
            String s = requestId + "|" + itemId + "|" + authorId + "|" + actionName + "|"
                    + actionValue + "|" + requestType + "|" + sourceType + "|" + eventTimeMs;
            return s.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        }

        public static ActionRecord fromBytes(byte[] bytes) {
            String[] p = new String(bytes, java.nio.charset.StandardCharsets.UTF_8).split("\\|", 8);
            return new ActionRecord(
                Long.parseLong(p[0]), Long.parseLong(p[1]), Long.parseLong(p[2]),
                p[3], Long.parseLong(p[4]), p[5], p[6], Long.parseLong(p[7]));
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] 内存 ActionListStore
    //  工业落地时替换为 Redis
    // ─────────────────────────────────────────────

    public static class ActionListStore {
        private final Map<String, Deque<byte[]>> lists = new HashMap<>();
        private final Map<String, Long> expiresAtMs = new HashMap<>();

        public synchronized void append(String key, byte[] value, int maxLength,
                                        java.time.Duration ttl) {
            if (key == null || key.isBlank() || value == null || value.length == 0) return;
            if (maxLength <= 0 || ttl.isNegative() || ttl.isZero()) {
                throw new IllegalArgumentException("invalid config");
            }
            removeIfExpired(key, System.currentTimeMillis());
            Deque<byte[]> list = lists.computeIfAbsent(key, k -> new ArrayDeque<>());
            list.addLast(value);
            while (list.size() > maxLength) list.removeFirst();
            expiresAtMs.put(key, System.currentTimeMillis() + ttl.toMillis());
        }

        public synchronized List<ActionRecord> get(String key) {
            removeIfExpired(key, System.currentTimeMillis());
            List<ActionRecord> result = new ArrayList<>();
            Deque<byte[]> list = lists.getOrDefault(key, new ArrayDeque<>());
            for (byte[] b : list) result.add(ActionRecord.fromBytes(b));
            return result;
        }

        private void removeIfExpired(String key, long nowMs) {
            Long exp = expiresAtMs.get(key);
            if (exp != null && exp <= nowMs) {
                lists.remove(key);
                expiresAtMs.remove(key);
            }
        }

        public synchronized int size(String key) {
            removeIfExpired(key, System.currentTimeMillis());
            return lists.getOrDefault(key, new ArrayDeque<>()).size();
        }
    }

    // ─────────────────────────────────────────────
    //  短期兴趣序列服务
    //  对应书中 code_short_sequence_store
    // ─────────────────────────────────────────────

    public static class ShortSequenceService {
        private static final int MAX_ACTIONS_PER_LIST = 200;
        private static final java.time.Duration ACTION_LIST_TTL = java.time.Duration.ofDays(14);

        private final ActionListStore actionListStore;

        public ShortSequenceService(ActionListStore actionListStore) {
            this.actionListStore = actionListStore;
        }

        public void appendAction(long requestId, long userId, String deviceId, long itemId,
                                 long authorId, String actionName, long actionValue,
                                 String requestType, String sourceType, long eventTimeMs) {
            if (requestId <= 0 || itemId <= 0 || actionName == null || actionName.isBlank()) return;
            if (userId <= 0 && (deviceId == null || deviceId.isBlank())) return;
            if (deviceId == null || deviceId.isBlank()) return;

            ActionRecord record = new ActionRecord(requestId, itemId, authorId, actionName,
                    actionValue, requestType == null ? "" : requestType,
                    sourceType == null ? "" : sourceType, eventTimeMs);

            actionListStore.append(deviceKey(deviceId, actionName), record.toBytes(),
                    MAX_ACTIONS_PER_LIST, ACTION_LIST_TTL);
        }

        public List<ActionRecord> getRecentActions(String deviceId, String actionName) {
            return actionListStore.get(deviceKey(deviceId, actionName));
        }

        private String deviceKey(String deviceId, String actionName) {
            return "recent-action:" + deviceId + ":" + actionName;
        }
    }

    // ─────────────────────────────────────────────
    //  [MOCK] 串行执行器
    //  工业落地时替换为按 key 哈希到固定线程的并发队列
    // ─────────────────────────────────────────────

    public static class SingleThreadSerialExecutor implements AutoCloseable {
        private final ExecutorService executor = Executors.newSingleThreadExecutor();

        public void execute(String key, Runnable task) {
            // [MOCK] 单线程池，实际按 key 路由到专用线程可减少锁竞争
            executor.submit(task);
        }

        @Override
        public void close() {
            executor.shutdown();
            try { executor.awaitTermination(5, TimeUnit.SECONDS); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }
    }

    // ─────────────────────────────────────────────
    //  物品更新触发缓冲区（简化版）
    // ─────────────────────────────────────────────

    public static class InMemoryItemUpdateBuffer {
        private final Set<Long> buffer = Collections.synchronizedSet(new HashSet<>());

        public void add(long itemId) { buffer.add(itemId); }

        public Set<Long> drain() {
            Set<Long> result = new HashSet<>(buffer);
            buffer.clear();
            return result;
        }
    }

    // ─────────────────────────────────────────────
    //  ActionLogConsumer
    //  对应书中 code_action_log_consumer
    // ─────────────────────────────────────────────

    public static class ActionLogConsumer {
        private final SingleThreadSerialExecutor serialExecutor;
        private final ShortSequenceService sequenceService;
        private final InMemoryItemUpdateBuffer itemUpdateBuffer;
        private final Set<String> ignoredForItemUpdate;

        public ActionLogConsumer(SingleThreadSerialExecutor serialExecutor,
                                 ShortSequenceService sequenceService,
                                 InMemoryItemUpdateBuffer itemUpdateBuffer,
                                 Set<String> ignoredForItemUpdate) {
            this.serialExecutor = serialExecutor;
            this.sequenceService = sequenceService;
            this.itemUpdateBuffer = itemUpdateBuffer;
            this.ignoredForItemUpdate = new HashSet<>(ignoredForItemUpdate);
        }

        public void consume(ActionLog actionLog) {
            String subjectKey = subjectKeyOf(actionLog);
            if (subjectKey == null) return;
            serialExecutor.execute(subjectKey, () -> appendActions(actionLog));
        }

        private void appendActions(ActionLog actionLog) {
            long requestId = firstRequestId(actionLog.requestIds);
            if (requestId <= 0 || actionLog.itemIds.isEmpty()) return;

            boolean hasAlignedValue = actionLog.actionValues.size() == actionLog.itemIds.size();
            boolean hasAlignedAuthor = actionLog.authorIds.size() == actionLog.itemIds.size();

            for (int i = 0; i < actionLog.itemIds.size(); i++) {
                long itemId = actionLog.itemIds.get(i);
                if (itemId <= 0) continue;

                long actionValue = hasAlignedValue ? actionLog.actionValues.get(i) : 1L;
                long authorId = hasAlignedAuthor ? actionLog.authorIds.get(i) : 0L;
                String sourceType = sourceTypeAt(actionLog.itemAttributes, i);

                sequenceService.appendAction(requestId, actionLog.userId, actionLog.deviceId,
                        itemId, authorId, actionLog.actionName, actionValue,
                        actionLog.requestType, sourceType, actionLog.eventTimeMs);

                if (!ignoredForItemUpdate.contains(actionLog.actionName)) {
                    itemUpdateBuffer.add(itemId);
                }
            }
        }

        private String subjectKeyOf(ActionLog actionLog) {
            if (actionLog.userId > 0) return "user:" + actionLog.userId;
            if (actionLog.deviceId != null && !actionLog.deviceId.isBlank())
                return "device:" + actionLog.deviceId;
            return null;
        }

        private long firstRequestId(List<String> requestIds) {
            if (requestIds == null || requestIds.isEmpty()) return 0L;
            try { return Long.parseLong(requestIds.get(0)); }
            catch (NumberFormatException e) { return 0L; }
        }

        private String sourceTypeAt(List<Map<String, String>> attributes, int index) {
            if (attributes == null || index >= attributes.size()) return "";
            return attributes.get(index).getOrDefault("source_type", "");
        }
    }

    // ─────────────────────────────────────────────
    //  Demo 入口
    // ─────────────────────────────────────────────

    public static List<ActionRecord> runDemo() throws InterruptedException {
        ActionListStore store = new ActionListStore();
        ShortSequenceService seqService = new ShortSequenceService(store);
        InMemoryItemUpdateBuffer itemBuf = new InMemoryItemUpdateBuffer();

        try (SingleThreadSerialExecutor executor = new SingleThreadSerialExecutor()) {
            ActionLogConsumer consumer = new ActionLogConsumer(
                    executor, seqService, itemBuf, Set.of("show"));

            long now = System.currentTimeMillis();

            // 模拟一批用户行为日志
            ActionLog log1 = new ActionLog(
                    100L, "dev_A",
                    List.of("20010001"),
                    List.of(501L, 502L),
                    List.of(1L, 1L),
                    List.of(201L, 202L),
                    List.of(Map.of("source_type", "recommend"), Map.of("source_type", "search")),
                    "click", "home_feed", now - 5000L);

            ActionLog log2 = new ActionLog(
                    100L, "dev_A",
                    List.of("20010002"),
                    List.of(501L),
                    List.of(30000L),
                    List.of(201L),
                    List.of(Map.of()),
                    "watch_time", "home_feed", now - 3000L);

            consumer.consume(log1);
            consumer.consume(log2);

            // 等待异步任务完成
            Thread.sleep(200);

            List<ActionRecord> clicks = seqService.getRecentActions("dev_A", "click");
            List<ActionRecord> watches = seqService.getRecentActions("dev_A", "watch_time");
            System.out.println("click records: " + clicks.size());
            System.out.println("watch records: " + watches.size());
            System.out.println("item update buffer: " + itemBuf.drain());

            List<ActionRecord> all = new ArrayList<>(clicks);
            all.addAll(watches);
            return all;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== ch03 行为日志解析服务 Demo ===");
        List<ActionRecord> records = runDemo();
        System.out.println("Total records: " + records.size());
    }
}
