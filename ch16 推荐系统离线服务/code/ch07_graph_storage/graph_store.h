// graph_store.h — 图存储后端接口与内存模拟实现
//
// 对应 test_chapter.tex 1.7 节图存储服务
// tex 代码片段：code_graph_kafka_consumer（JdbcSwingGraphStore）
//
// [MOCK] 生产环境替换为：
//   - JDBC DataSource → MySQL / PostgreSQL 连接池
//   - upsertEdge/upsertNode → 真实 SQL 执行
//   - GraphRead / readNeighbors → 带超时的数据库查询

#pragma once

#include "graph_types.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>

namespace reco::graph {

// ─────────────────────────────────────────────────────────
//  GraphRead：readNeighbors 的内部返回结构
// ─────────────────────────────────────────────────────────

struct GraphRead {
    std::vector<GraphEdge> edges;
    int64_t served_graph_version;
};

// ─────────────────────────────────────────────────────────
//  [MOCK] InMemorySwingGraphStore
//  对应 tex 代码片段 code_graph_kafka_consumer 中的 JdbcSwingGraphStore。
//  工业落地时替换为真实 JDBC 实现。
// ─────────────────────────────────────────────────────────

class InMemorySwingGraphStore {
public:
    static constexpr int64_t CURRENT_GRAPH_VERSION = 1L;

    // ── tex snippet: code_graph_kafka_consumer（write 方法骨架） ──
    /**
     * 写入用户节点、物品节点和双向 SWING 边。
     * 边表以 (source_item_id, target_item_id) 为唯一键；
     * 同一 event_id 再次写入不增加权重（幂等）。
     */
    void write(const ActionLog& action, const std::deque<int64_t>& recent_items) {
        std::lock_guard<std::mutex> lock(mutex_);
        upsert_node("user:" + std::to_string(action.user_id), "USER");
        upsert_node("item:" + std::to_string(action.item_id), "ITEM");
        for (int64_t prev_item_id : recent_items) {
            if (prev_item_id != action.item_id) {
                upsert_edge(prev_item_id, action.item_id, action.event_id);
                upsert_edge(action.item_id, prev_item_id, action.event_id);
            }
        }
    }

    /**
     * 读取 source_item_id 的一跳邻居，按权重降序，最多 scan_limit 条。
     * 对应 JdbcGraphRepository::readNeighbors。
     */
    GraphRead read_neighbors(int64_t source_item_id,
                              const std::string& /*edge_type*/,
                              int scan_limit) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = edges_.find(source_item_id);
        if (it == edges_.end()) {
            return GraphRead{{}, CURRENT_GRAPH_VERSION};
        }
        // 拷贝后排序（避免修改内部状态）
        std::vector<GraphEdge> sorted = it->second;
        std::sort(sorted.begin(), sorted.end(),
                  [](const GraphEdge& a, const GraphEdge& b) {
                      return a.weight > b.weight;
                  });
        if (static_cast<int>(sorted.size()) > scan_limit) {
            sorted.resize(scan_limit);
        }
        return GraphRead{std::move(sorted), CURRENT_GRAPH_VERSION};
    }

    // 查询节点是否存在
    bool has_node(const std::string& node_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_.count(node_id) > 0;
    }

    // 查询边数
    size_t edge_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        for (auto& kv : edges_) total += kv.second.size();
        return total;
    }

private:
    void upsert_node(const std::string& node_id, const std::string& node_type) {
        nodes_[node_id] = node_type;
    }

    void upsert_edge(int64_t src, int64_t tgt, const std::string& event_id) {
        // 同一 event_id 不重复计数（幂等）
        auto& vec = edges_[src];
        for (auto& e : vec) {
            if (e.target_item_id == tgt) {
                if (last_event_id_[{src, tgt}] != event_id) {
                    e.weight += 1.0;
                    last_event_id_[{src, tgt}] = event_id;
                }
                return;
            }
        }
        vec.emplace_back(tgt, 1.0);
        last_event_id_[{src, tgt}] = event_id;
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> nodes_;
    std::unordered_map<int64_t, std::vector<GraphEdge>> edges_;
    // 幂等去重: (src, tgt) -> last event_id
    struct PairHash {
        size_t operator()(const std::pair<int64_t, int64_t>& p) const {
            return std::hash<int64_t>{}(p.first) ^ (std::hash<int64_t>{}(p.second) << 32);
        }
    };
    std::unordered_map<std::pair<int64_t, int64_t>, std::string, PairHash> last_event_id_;
};

} // namespace reco::graph
