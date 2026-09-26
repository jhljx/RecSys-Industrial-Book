// test_graph_service.cpp — 图存储服务单元测试
//
// 覆盖：
//   code_graph_kafka_consumer  → ActionLogGraphConsumer
//   code_graph_neighbor_service → GraphNeighborService
//
// 编译（macOS，需通过 build_and_test.sh）：
//   ./build_and_test.sh
//
// 不依赖任何第三方库。

#include "graph_types.h"
#include "graph_store.h"
#include "graph_kafka_consumer.h"
#include "graph_neighbor_service.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

using namespace reco::graph;

// ─────────────────────────────────────────────────────────
//  极简测试框架
// ─────────────────────────────────────────────────────────

#include <functional>
#include <vector>
static std::vector<std::pair<std::string, std::function<void()>>> tests;

struct TestRegistrar {
    TestRegistrar(std::string name, std::function<void()> fn) {
        tests.push_back({std::move(name), std::move(fn)});
    }
};

#define TEST(name) \
    static void _test_fn_##name(); \
    static TestRegistrar _reg_##name(#name, _test_fn_##name); \
    static void _test_fn_##name()

#define ASSERT_TRUE(expr) \
    do { if (!(expr)) { std::cerr << "FAIL: " #expr " at " __FILE__ ":" << __LINE__ << "\n"; std::abort(); } } while(0)
#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { std::cerr << "FAIL: " #a " == " #b " (" << (a) << " != " << (b) << ") at " __FILE__ ":" << __LINE__ << "\n"; std::abort(); } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

// ─────────────────────────────────────────────────────────
//  code_graph_kafka_consumer 测试
// ─────────────────────────────────────────────────────────

TEST(test_consumer_writes_nodes_and_edges) {
    InMemorySwingGraphStore store;
    MockKafkaConsumer kafka("action-log");
    ActionLogGraphConsumer consumer(kafka, store);

    kafka.publish(ActionLog("evt1", 1001L, 201L, 1000L));
    kafka.publish(ActionLog("evt2", 1001L, 202L, 2000L));
    kafka.publish(ActionLog("evt3", 1001L, 203L, 3000L));

    consumer.drain();

    ASSERT_TRUE(store.has_node("user:1001"));
    ASSERT_TRUE(store.has_node("item:201"));
    ASSERT_TRUE(store.has_node("item:202"));
    ASSERT_TRUE(store.has_node("item:203"));
    // 三个物品两两之间应有双向边：3 对 × 2 方向 = 6 条
    ASSERT_EQ(store.edge_count(), 6u);
    std::cout << "  PASS: test_consumer_writes_nodes_and_edges\n";
}

TEST(test_consumer_window_size_limit) {
    InMemorySwingGraphStore store;
    MockKafkaConsumer kafka("action-log");
    ActionLogGraphConsumer consumer(kafka, store);

    // 写入 MAX_WINDOW_SIZE + 5 条行为，验证窗口不超限
    for (int i = 1; i <= ActionLogGraphConsumer::MAX_WINDOW_SIZE + 5; i++) {
        kafka.publish(ActionLog("evt" + std::to_string(i), 2001L,
                                static_cast<int64_t>(100 + i), i * 1000LL));
    }
    consumer.drain();

    const auto& window = consumer.recent_items(2001L);
    ASSERT_EQ(static_cast<int>(window.size()), ActionLogGraphConsumer::MAX_WINDOW_SIZE);
    std::cout << "  PASS: test_consumer_window_size_limit\n";
}

TEST(test_consumer_idempotent_same_event) {
    // 同一 event_id 重复投递，边权重不应增加
    InMemorySwingGraphStore store;
    MockKafkaConsumer kafka("action-log");
    ActionLogGraphConsumer consumer(kafka, store);

    kafka.publish(ActionLog("dup_evt", 3001L, 301L, 1000L));
    consumer.drain();
    // 第一条写完后再写含相同物品对的相同 event
    kafka.publish(ActionLog("evt_b", 3001L, 302L, 2000L));  // 302 与 301 建边
    consumer.drain();
    // 重复相同 event（幂等测试）
    kafka.publish(ActionLog("evt_b", 3001L, 302L, 2000L));
    consumer.drain();

    GraphRead gr = store.read_neighbors(301L, "SWING", 10);
    ASSERT_EQ(gr.edges.size(), 1u);
    ASSERT_EQ(gr.edges[0].weight, 1.0);  // 幂等，不应加到 2.0
    std::cout << "  PASS: test_consumer_idempotent_same_event\n";
}

TEST(test_consumer_invalid_action_throws) {
    // event_id 为空的 ActionLog 构造应抛异常
    bool threw = false;
    try {
        ActionLog("", 1001L, 201L, 1000L);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
    std::cout << "  PASS: test_consumer_invalid_action_throws\n";
}

// ─────────────────────────────────────────────────────────
//  code_graph_neighbor_service 测试
// ─────────────────────────────────────────────────────────

// 辅助：在已有 store 中建 N 条 seed_item 的邻居边
static void populate_edges(InMemorySwingGraphStore& store,
                            int64_t seed_item, int n_neighbors) {
    MockKafkaConsumer kafka("action-log");
    ActionLogGraphConsumer consumer(kafka, store);
    for (int i = 1; i <= n_neighbors; i++) {
        int64_t item_b = seed_item + i;
        kafka.publish(ActionLog("sa" + std::to_string(seed_item) + "_" + std::to_string(i),
                                5001L + i, seed_item, i * 100LL));
        kafka.publish(ActionLog("sb" + std::to_string(item_b) + "_" + std::to_string(i),
                                5001L + i, item_b, i * 100LL + 50LL));
    }
    consumer.drain();
}

TEST(test_neighbor_service_returns_candidates) {
    InMemorySwingGraphStore store;
    populate_edges(store, 401L, 5);
    AlwaysVisibleItems visibility;
    GraphNeighborService service(store, visibility);

    NeighborQuery query(401L, "SWING", 3, 0L);
    auto result = service.neighbors(query);

    ASSERT_FALSE(result.degraded);
    ASSERT_EQ(static_cast<int>(result.item_ids.size()), 3);
    std::cout << "  PASS: test_neighbor_service_returns_candidates\n";
}

TEST(test_neighbor_service_filters_invisible) {
    InMemorySwingGraphStore store;
    populate_edges(store, 501L, 4);
    // 将所有邻居加入黑名单
    BlocklistItemVisibility visibility({502L, 503L, 504L, 505L});
    GraphNeighborService service(store, visibility);

    NeighborQuery query(501L, "SWING", 10, 0L);
    auto result = service.neighbors(query);

    ASSERT_FALSE(result.degraded);
    ASSERT_EQ(static_cast<int>(result.item_ids.size()), 0);
    std::cout << "  PASS: test_neighbor_service_filters_invisible\n";
}

TEST(test_neighbor_service_limit_exceeded_degrades) {
    InMemorySwingGraphStore store;
    AlwaysVisibleItems visibility;
    GraphNeighborService service(store, visibility);

    // limit > MAX_RESULT_SIZE → 立即降级
    NeighborQuery query(601L, "SWING", GraphNeighborService::MAX_RESULT_SIZE + 1, 0L);
    auto result = service.neighbors(query);

    ASSERT_TRUE(result.degraded);
    ASSERT_EQ(static_cast<int>(result.item_ids.size()), 0);
    std::cout << "  PASS: test_neighbor_service_limit_exceeded_degrades\n";
}

TEST(test_neighbor_service_version_check_degrades) {
    InMemorySwingGraphStore store;
    populate_edges(store, 701L, 3);
    AlwaysVisibleItems visibility;
    GraphNeighborService service(store, visibility);

    // 要求版本 > 当前图版本 → 降级
    int64_t too_high_version = InMemorySwingGraphStore::CURRENT_GRAPH_VERSION + 9999;
    NeighborQuery query(701L, "SWING", 5, too_high_version);
    auto result = service.neighbors(query);

    ASSERT_TRUE(result.degraded);
    std::cout << "  PASS: test_neighbor_service_version_check_degrades\n";
}

TEST(test_neighbor_service_unknown_item_returns_empty) {
    InMemorySwingGraphStore store;
    AlwaysVisibleItems visibility;
    GraphNeighborService service(store, visibility);

    NeighborQuery query(999999L, "SWING", 5, 0L);
    auto result = service.neighbors(query);

    ASSERT_FALSE(result.degraded);
    ASSERT_EQ(static_cast<int>(result.item_ids.size()), 0);
    std::cout << "  PASS: test_neighbor_service_unknown_item_returns_empty\n";
}

// ─────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────

int main() {
    std::cout << "=== ch07 Graph Storage C++ Tests ===\n";
    int passed = 0;
    for (auto& [name, fn] : tests) {
        std::cout << "[" << name << "]\n";
        fn();
        passed++;
    }
    std::cout << "\n" << passed << "/" << tests.size() << " tests passed.\n";
    return 0;
}
