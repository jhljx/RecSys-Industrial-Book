// graph_types.h — 图存储公共类型定义
//
// 对应 test_chapter.tex 1.7 节图存储服务
// tex 代码片段：code_graph_kafka_consumer、code_graph_neighbor_service
//
// [MOCK] 本文件为独立 C++ 实现，不依赖外部库（Kafka/JDBC 用内存模拟）。
//        工业落地时：
//          - KafkaConsumer → librdkafka / confluent-kafka-cpp
//          - GraphStore    → JDBC / NebulaGraph / JanusGraph C++ SDK
//          - DataSource    → 真实 JDBC 连接池

#pragma once

#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <chrono>

namespace reco::graph {

// ─────────────────────────────────────────────────────────
//  ActionLog：行为日志事件
// ─────────────────────────────────────────────────────────

struct ActionLog {
    std::string event_id;
    int64_t user_id;
    int64_t item_id;
    int64_t occurred_at_ms;  // epoch ms

    ActionLog(std::string event_id, int64_t user_id, int64_t item_id, int64_t occurred_at_ms)
        : event_id(std::move(event_id))
        , user_id(user_id)
        , item_id(item_id)
        , occurred_at_ms(occurred_at_ms)
    {
        if (this->event_id.empty() || user_id <= 0 || item_id <= 0) {
            throw std::invalid_argument("invalid action log");
        }
    }
};

// ─────────────────────────────────────────────────────────
//  GraphEdge：图边（Online Swing ITEM-ITEM 边）
// ─────────────────────────────────────────────────────────

struct GraphEdge {
    int64_t target_item_id;
    double  weight;

    GraphEdge() : target_item_id(0), weight(0.0) {}  // default ctor for vector resize

    GraphEdge(int64_t target_item_id, double weight)
        : target_item_id(target_item_id), weight(weight)
    {
        if (target_item_id <= 0 || weight <= 0) {
            throw std::invalid_argument("invalid graph edge");
        }
    }
};

// ─────────────────────────────────────────────────────────
//  NeighborQuery：一跳邻居查询请求
// ─────────────────────────────────────────────────────────

struct NeighborQuery {
    int64_t     source_item_id;
    std::string edge_type;
    int         limit;
    int64_t     min_graph_version;

    NeighborQuery(int64_t source_item_id, std::string edge_type,
                  int limit, int64_t min_graph_version)
        : source_item_id(source_item_id)
        , edge_type(std::move(edge_type))
        , limit(limit)
        , min_graph_version(min_graph_version)
    {
        if (source_item_id <= 0 || this->edge_type.empty() || limit <= 0) {
            throw std::invalid_argument("invalid neighbor query");
        }
    }
};

// ─────────────────────────────────────────────────────────
//  NeighborResult：一跳邻居查询结果
// ─────────────────────────────────────────────────────────

struct NeighborResult {
    std::vector<int64_t> item_ids;
    bool degraded;
    int64_t served_graph_version;

    NeighborResult(std::vector<int64_t> item_ids, bool degraded, int64_t served_graph_version)
        : item_ids(std::move(item_ids))
        , degraded(degraded)
        , served_graph_version(served_graph_version)
    {}

    // 降级结果工厂方法
    static NeighborResult degraded_result() {
        return NeighborResult({}, true, 0LL);
    }
};

} // namespace reco::graph
