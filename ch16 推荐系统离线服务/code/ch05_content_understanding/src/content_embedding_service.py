#!/usr/bin/env python3
"""
内容理解生产服务 — 完整可运行示例（Python）

包含：
  1. MockQwenVLClient  - Qwen2.5-VL 模型调用客户端（Mock）
  2. ContentEmbeddingConsumer - 直播切片 Embedding 消费者
                               （对应书中 code_content_embedding + code_qwen_vl_rpc）
  3. EmbeddingStore    - Embedding 向量存储（Mock，实际应存入向量数据库或 Redis）

[MOCK] MockQwenVLClient 返回固定维度的随机向量；
       工业落地时替换为调用真实 Qwen2.5-VL 推理服务（gRPC 或 HTTP）。
[MOCK] EmbeddingStore 使用内存 dict；
       工业落地时替换为 Faiss / Redis Stack / Milvus 等向量存储。
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────

@dataclass
class LiveSlice:
    """直播切片事件（从 Kafka 消费）"""
    item_id: int
    author_id: int
    slice_id: str          # 唯一切片 ID，例如 "item_501_ts_1700000000"
    image_url: str         # 切片截图 URL（Mock 时为伪造路径）
    title: str             # 直播标题
    category: str          # 类目
    event_time_ms: int


@dataclass
class ContentEmbedding:
    """内容向量结果"""
    item_id: int
    slice_id: str
    embedding: list[float]   # 维度 = EMBEDDING_DIM
    generated_at_ms: int
    # 模型附加文本标签
    tags: list[str] = field(default_factory=list)
    description: str = ""


# ─────────────────────────────────────────────
#  [MOCK] Qwen2.5-VL 客户端
#  工业落地时替换为真实 gRPC / HTTP 推理服务调用
# ─────────────────────────────────────────────

EMBEDDING_DIM = 256   # 向量维度


class MockQwenVLClient:
    """
    [MOCK] 模拟 Qwen2.5-VL 多模态模型。
    实际推理服务接受 (image_bytes, text_prompt) -> (embedding, description, tags)。
    工业落地时：
      1. 将图片下载 + base64 编码发送到推理服务
      2. 文本 prompt 填入 LiveSlice.title 等字段
      3. 解析返回的 JSON / Protobuf 得到 embedding
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, seed: int = 42):
        self._dim = embedding_dim
        self._rng = random.Random(seed)

    def encode(self, image_url: str, text: str) -> tuple[list[float], str, list[str]]:
        """
        返回 (embedding_vector, description, tags)
        [MOCK] 使用 hash(image_url + text) 作为随机种子，保证相同输入得到相同向量。
        """
        seed = int(hashlib.md5((image_url + text).encode()).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        embedding = [rng.gauss(0.0, 1.0) for _ in range(self._dim)]
        # 归一化
        norm = sum(x * x for x in embedding) ** 0.5
        embedding = [x / (norm + 1e-8) for x in embedding]
        description = f"[MOCK] Scene of {text[:30]}"
        tags = ["live", text.split()[0].lower() if text else "default"]
        return embedding, description, tags


# ─────────────────────────────────────────────
#  [MOCK] Embedding 存储
#  工业落地时替换为向量数据库（如 Milvus / Faiss + Redis）
# ─────────────────────────────────────────────

class EmbeddingStore:
    """
    [MOCK] 内存 Embedding 存储。
    工业落地时：
      - 向量检索：Milvus / Faiss
      - 精确查找：Redis (item_id -> latest embedding bytes)
    """

    def __init__(self):
        self._store: dict[str, ContentEmbedding] = {}   # slice_id -> embedding
        self._item_latest: dict[int, str] = {}           # item_id -> latest slice_id

    def save(self, emb: ContentEmbedding) -> None:
        self._store[emb.slice_id] = emb
        self._item_latest[emb.item_id] = emb.slice_id

    def get_latest(self, item_id: int) -> Optional[ContentEmbedding]:
        slice_id = self._item_latest.get(item_id)
        if slice_id is None:
            return None
        return self._store.get(slice_id)

    def get_by_slice(self, slice_id: str) -> Optional[ContentEmbedding]:
        return self._store.get(slice_id)

    def count(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
#  内容理解 Embedding 消费者
#  对应书中 code_content_embedding
# ─────────────────────────────────────────────

class ContentEmbeddingConsumer:
    """
    消费 Kafka 直播切片事件，调用 Qwen2.5-VL 生成 Embedding 后写入存储。
    """

    def __init__(self, vl_client: MockQwenVLClient, store: EmbeddingStore):
        self._client = vl_client
        self._store = store
        self._processed = 0
        self._errors = 0

    def consume(self, slice_event: LiveSlice) -> Optional[ContentEmbedding]:
        """
        处理单个切片事件：
          1. 幂等检查（已处理过的 slice_id 跳过）
          2. 调用多模态模型生成向量
          3. 写入 EmbeddingStore
        """
        # 幂等性检查
        if self._store.get_by_slice(slice_event.slice_id) is not None:
            return self._store.get_by_slice(slice_event.slice_id)

        # 拼接 text prompt
        text_prompt = f"{slice_event.title} {slice_event.category}"

        try:
            embedding, description, tags = self._client.encode(
                slice_event.image_url, text_prompt)

            result = ContentEmbedding(
                item_id=slice_event.item_id,
                slice_id=slice_event.slice_id,
                embedding=embedding,
                generated_at_ms=int(time.time() * 1000),
                tags=tags,
                description=description,
            )
            self._store.save(result)
            self._processed += 1
            return result

        except Exception as e:  # pylint: disable=broad-except
            self._errors += 1
            print(f"[ContentEmbeddingConsumer] ERROR slice={slice_event.slice_id}: {e}")
            return None

    def stats(self) -> dict:
        return {"processed": self._processed, "errors": self._errors}


# ─────────────────────────────────────────────
#  余弦相似度工具
# ─────────────────────────────────────────────

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    return dot / (n1 * n2 + 1e-8)


# ─────────────────────────────────────────────
#  端到端 Demo
# ─────────────────────────────────────────────

def run_demo() -> dict:
    client = MockQwenVLClient(embedding_dim=EMBEDDING_DIM)
    store = EmbeddingStore()
    consumer = ContentEmbeddingConsumer(client, store)

    now_ms = int(time.time() * 1000)

    # 模拟 3 个直播切片
    slices = [
        LiveSlice(501, 201, "item_501_ts_01", "/live/501/thumb_01.jpg",
                  "美食直播 厨神现场教学", "美食", now_ms - 3000),
        LiveSlice(501, 201, "item_501_ts_02", "/live/501/thumb_02.jpg",
                  "美食直播 厨神现场教学", "美食", now_ms - 1000),
        LiveSlice(502, 202, "item_502_ts_01", "/live/502/thumb_01.jpg",
                  "游戏直播 王者荣耀电竞赛事解说", "游戏", now_ms - 2000),
    ]

    results = []
    for s in slices:
        emb = consumer.consume(s)
        if emb:
            results.append(emb)
            print(f"  item={emb.item_id} slice={emb.slice_id} "
                  f"dim={len(emb.embedding)} tags={emb.tags}")

    # 幂等性：重复处理相同切片不应增加存储
    consumer.consume(slices[0])
    assert store.count() == 3, f"Expected 3, got {store.count()}"

    # 同一物品的两个切片应高度相似（同一直播不同时刻）
    e1 = store.get_by_slice("item_501_ts_01")
    e2 = store.get_by_slice("item_501_ts_02")
    sim_same = cosine_similarity(e1.embedding, e2.embedding)

    # 不同物品相似度应较低
    e3 = store.get_latest(502)
    sim_diff = cosine_similarity(e1.embedding, e3.embedding)

    print(f"\n  相似度(同物品两切片)  = {sim_same:.4f}")
    print(f"  相似度(不同物品切片)  = {sim_diff:.4f}")

    stats = consumer.stats()
    print(f"\n  Consumer stats: {stats}")

    return {
        "processed": stats["processed"],
        "stored": store.count(),
        "sim_same_item": sim_same,
        "sim_diff_item": sim_diff,
    }


if __name__ == "__main__":
    print("=== ch05 内容理解 Embedding 服务 Demo ===")
    result = run_demo()
    print("\nResult:", result)
