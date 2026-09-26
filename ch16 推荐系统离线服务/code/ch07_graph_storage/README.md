# ch07 图存储 C++ 实现

对应 `test_chapter.tex` 1.7 节图存储服务。

## 代码结构

```
ch07_graph_storage/
├── graph_types.h              # 公共类型：ActionLog、GraphEdge、NeighborQuery、NeighborResult
├── graph_store.h              # InMemorySwingGraphStore（mock JDBC 存储）
├── graph_kafka_consumer.h     # ActionLogGraphConsumer（code_graph_kafka_consumer）
├── graph_neighbor_service.h   # GraphNeighborService（code_graph_neighbor_service）
├── test_graph_service.cpp     # 单元测试（10 个测试用例）
└── build_and_test.sh          # 编译 + 运行脚本
```

## tex 代码片段对应关系

| tex 标签 | C++ 实现位置 |
|---|---|
| `code_graph_kafka_consumer` | `ActionLogGraphConsumer::run_once()` in `graph_kafka_consumer.h` |
| `code_graph_neighbor_service` | `GraphNeighborService::neighbors()` in `graph_neighbor_service.h` |

## 编译与运行

```bash
chmod +x build_and_test.sh
./build_and_test.sh
```

要求：`g++` 支持 C++17（`-std=c++17`），无需第三方库。

## 工业落地替换点

| [MOCK] 组件 | 工业实现 |
|---|---|
| `MockKafkaConsumer` | librdkafka / confluent-kafka-cpp |
| `InMemorySwingGraphStore` | MySQL/PostgreSQL JDBC 连接池（写图边） |
| `InMemorySwingGraphStore::read_neighbors` | NebulaGraph / JanusGraph C++ SDK |
| `AlwaysVisibleItems` | 查询审核系统可见性接口 |
