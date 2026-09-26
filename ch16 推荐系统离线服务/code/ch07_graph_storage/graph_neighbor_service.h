// graph_neighbor_service.h — 面向在线召回的一跳图邻居服务
//
// 对应 test_chapter.tex 1.7 节图存储服务
// tex 代码片段：code_graph_neighbor_service
//
// [MOCK] ItemVisibility 使用内存集合模拟；
//        JdbcGraphRepository 替换为 InMemorySwingGraphStore。
//        工业落地时将 read_neighbors 替换为真实 SQL 查询。

#pragma once

#include "graph_types.h"
#include "graph_store.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <unordered_set>

namespace reco::graph {

// ─────────────────────────────────────────────────────────
//  ItemVisibility：物品可见性接口
// ─────────────────────────────────────────────────────────

class ItemVisibility {
public:
    virtual ~ItemVisibility() = default;
    virtual bool is_visible(int64_t item_id) const = 0;
};

/** [MOCK] 所有物品均可见 */
class AlwaysVisibleItems final : public ItemVisibility {
public:
    bool is_visible(int64_t item_id) const override { return item_id > 0; }
};

/** [MOCK] 基于黑名单集合的可见性 */
class BlocklistItemVisibility final : public ItemVisibility {
public:
    explicit BlocklistItemVisibility(std::unordered_set<int64_t> blocklist)
        : blocklist_(std::move(blocklist)) {}

    bool is_visible(int64_t item_id) const override {
        return item_id > 0 && blocklist_.find(item_id) == blocklist_.end();
    }

private:
    std::unordered_set<int64_t> blocklist_;
};

// ─────────────────────────────────────────────────────────
//  GraphNeighborService
//  对应 tex 代码片段 code_graph_neighbor_service
// ─────────────────────────────────────────────────────────

/**
 * 面向在线召回的一跳图邻居服务。
 *
 * 查询流程（对应 tex GraphNeighborService.neighbors()）：
 *   1. 校验 limit ≤ MAX_RESULT_SIZE
 *   2. 从底层存储读取最多 scan_limit = min(MAX_SCAN_SIZE, limit * 5) 条边
 *   3. 校验图版本 ≥ query.min_graph_version
 *   4. 过滤不可见物品，累积 limit 个候选后截止
 *   5. 异常时返回空候选 + degraded = true
 *
 * 降级标记告知召回编排服务可以改用向量或倒排通道。
 */
class GraphNeighborService {
public:
    static constexpr int MAX_RESULT_SIZE = 200;
    static constexpr int MAX_SCAN_SIZE   = 1000;
    static constexpr int READ_TIMEOUT_MS = 20;

    GraphNeighborService(
        InMemorySwingGraphStore& store,
        const ItemVisibility& visibility)
        : store_(store)
        , visibility_(visibility)
    {}

    // ── tex snippet: code_graph_neighbor_service ──
    NeighborResult neighbors(const NeighborQuery& query) const {
        if (query.limit > MAX_RESULT_SIZE) {
            return NeighborResult::degraded_result();
        }

        try {
            int scan_limit = std::min(MAX_SCAN_SIZE, query.limit * 5);
            GraphRead graph_read = store_.read_neighbors(
                query.source_item_id, query.edge_type, scan_limit);

            // 版本检查
            if (graph_read.served_graph_version < query.min_graph_version) {
                return NeighborResult({}, true, graph_read.served_graph_version);
            }

            // 过滤不可见物品，收集 limit 个候选
            std::vector<int64_t> item_ids;
            item_ids.reserve(query.limit);
            for (const auto& edge : graph_read.edges) {
                if (visibility_.is_visible(edge.target_item_id)) {
                    item_ids.push_back(edge.target_item_id);
                    if (static_cast<int>(item_ids.size()) == query.limit) {
                        break;
                    }
                }
            }
            return NeighborResult(std::move(item_ids), false,
                                  graph_read.served_graph_version);

        } catch (const std::exception& /*e*/) {
            return NeighborResult::degraded_result();
        }
    }

private:
    InMemorySwingGraphStore& store_;
    const ItemVisibility& visibility_;
};

} // namespace reco::graph
