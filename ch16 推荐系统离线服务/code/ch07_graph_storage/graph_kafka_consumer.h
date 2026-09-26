// graph_kafka_consumer.h — Action Log 消费者，写入 Online Swing 图边
//
// 对应 test_chapter.tex 1.7 节图存储服务
// tex 代码片段：code_graph_kafka_consumer
//
// [MOCK] Kafka 消费使用内存消息队列模拟（MockKafkaConsumer）。
//        工业落地时替换为 librdkafka / confluent-kafka-cpp KafkaConsumer。

#pragma once

#include "graph_types.h"
#include "graph_store.h"

#include <functional>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <iostream>

namespace reco::graph {

// ─────────────────────────────────────────────────────────
//  [MOCK] MockKafkaConsumer：内存消息队列，模拟 Kafka 拉取
// ─────────────────────────────────────────────────────────

class MockKafkaConsumer {
public:
    explicit MockKafkaConsumer(std::string topic) : topic_(std::move(topic)) {}

    void publish(const ActionLog& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(msg);
    }

    // poll(timeout_ms) → 返回最多 max_records 条消息
    std::vector<ActionLog> poll(int /*timeout_ms*/ = 500, int max_records = 100) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<ActionLog> result;
        while (!queue_.empty() && static_cast<int>(result.size()) < max_records) {
            result.push_back(queue_.front());
            queue_.pop();
        }
        return result;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    const std::string& topic() const { return topic_; }

private:
    std::string topic_;
    mutable std::mutex mutex_;
    std::queue<ActionLog> queue_;
};

// ─────────────────────────────────────────────────────────
//  ActionLogGraphConsumer
//  对应 tex 代码片段 code_graph_kafka_consumer
// ─────────────────────────────────────────────────────────

/**
 * Action Log 消费者 — 消费 action-log Kafka Topic 并写入 Online Swing 图边。
 *
 * 处理流程（对应 tex ActionLogGraphConsumer.run()）：
 *   1. poll(500ms) 拉取一批消息
 *   2. 对每条消息：
 *      a. 从 recentItemsByUser 取该用户的最近 MAX_WINDOW_SIZE 个物品
 *      b. 调用 graphStore.write(action, recentItems)
 *      c. 追加新物品到窗口末尾，超长时移除头部
 *      d. 成功后提交 Kafka 位点（[MOCK] 此处 no-op）
 *      e. 写库失败时 break，等待下次 poll 重试
 *
 * 注意：内存窗口（recentItemsByUser）在进程重启或分区迁移后会丢失；
 *       生产环境应将用户近期序列保存到可恢复的状态存储（如 Redis）。
 */
class ActionLogGraphConsumer {
public:
    static constexpr int MAX_WINDOW_SIZE = 50;

    explicit ActionLogGraphConsumer(
        MockKafkaConsumer& consumer,
        InMemorySwingGraphStore& graph_store)
        : consumer_(consumer)
        , graph_store_(graph_store)
        , running_(false)
    {}

    // ── tex snippet: code_graph_kafka_consumer ──
    void run_once() {
        auto records = consumer_.poll(500);
        for (const auto& action : records) {
            auto& recent_items = recent_items_by_user_[action.user_id];
            try {
                // 写入节点 + 双向 SWING 边
                graph_store_.write(action, recent_items);
                // 追加物品到用户窗口
                recent_items.push_back(action.item_id);
                if (static_cast<int>(recent_items.size()) > MAX_WINDOW_SIZE) {
                    recent_items.pop_front();
                }
                // [MOCK] commit_sync — 工业落地时提交 Kafka offset
            } catch (const std::exception& e) {
                std::cerr << "[ActionLogGraphConsumer] write failed: " << e.what() << "\n";
                break;  // 失败时停止处理本批，等待重试
            }
        }
    }

    // 便捷方法：处理所有消息直到队列为空
    void drain() {
        while (!consumer_.empty()) {
            run_once();
        }
    }

    // 获取用户最近物品窗口（用于测试验证）
    const std::deque<int64_t>& recent_items(int64_t user_id) const {
        static const std::deque<int64_t> empty;
        auto it = recent_items_by_user_.find(user_id);
        return it != recent_items_by_user_.end() ? it->second : empty;
    }

private:
    MockKafkaConsumer& consumer_;
    InMemorySwingGraphStore& graph_store_;
    std::atomic<bool> running_;
    std::unordered_map<int64_t, std::deque<int64_t>> recent_items_by_user_;
};

} // namespace reco::graph
