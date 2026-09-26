# 推荐系统离线服务 — 代码示例

本目录包含书中每个章节涉及的完整可运行代码。每个子目录对应章节中的一种数据流或服务。

## 目录结构

```
code/
├── ch01_sample_flow/              # 普通任务样本流
│   ├── src/                       # Java 源代码
│   ├── sample_join.sql            # Hive SQL 非实时样本拼接示例
│   └── pom.xml
├── ch02_rl_sample_flow/           # 强化学习样本流
│   ├── src/
│   └── pom.xml
├── ch03_behavior_log/             # 用户行为日志解析服务（短期/长期兴趣序列）
│   ├── src/
│   └── pom.xml
├── ch04_stat_features/            # 统计特征生产服务
│   ├── src/
│   └── pom.xml
├── ch05_content_understanding/    # 物品侧内容理解生产服务
│   ├── java/                      # Java gRPC 客户端
│   │   └── pom.xml
│   ├── src/                       # Python 向量服务
│   ├── proto/                     # Protobuf 定义
│   └── requirements.txt
├── ch06_item_index/               # 物品索引特征服务
│   ├── src/
│   └── pom.xml
├── ch07_graph_storage/            # 图存储服务（Online Swing）
│   ├── graph_types.h              # 公共类型定义
│   ├── graph_store.h              # 图存储 mock 实现
│   ├── graph_kafka_consumer.h     # Action Log 图边消费者
│   ├── graph_neighbor_service.h   # 一跳邻居查询服务
│   ├── test_graph_service.cpp     # 单元测试
│   ├── build_and_test.sh          # 编译 + 运行脚本
│   └── README.md
└── ch08_offline_training/         # 离线训练任务
    ├── single_machine_train.py    # 单机 DNN 训练（PyTorch）
    ├── distributed_train.py       # 分布式 DDP 训练 worker
    └── test_training.py           # 单元测试
```

## 说明

- ch01–ch04、ch06：Java，使用 Maven 构建，Java 17+
- ch05：Python（向量服务）+ Java（gRPC 客户端），Python 3.10+，Java 17+
- ch07：C++17，无第三方依赖，使用 `build_and_test.sh` 编译
- ch08：Python 3.10+，使用 PyTorch，通过 pytest 运行测试
- 涉及 **Redis、Kafka、Flink** 等中间件的部分采用**内存模拟实现**（Mock），注释中标注了 `[MOCK]`；
  工业落地时需替换为真实客户端，详见各文件中的注释

## 快速开始

```bash
# Java 项目（以 ch01 为例）
cd ch01_sample_flow
mvn clean test

# Python 项目（以 ch05 为例）
cd ch05_content_understanding
pip install -r requirements.txt
python -m pytest src/test_*.py -v

# C++ 项目（ch07）
cd ch07_graph_storage
chmod +x build_and_test.sh
./build_and_test.sh

# Python 训练项目（ch08）
cd ch08_offline_training
python -m pytest test_training.py -v
```
