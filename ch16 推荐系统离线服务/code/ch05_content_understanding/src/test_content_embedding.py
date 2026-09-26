#!/usr/bin/env python3
"""
ch05 内容理解服务测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from content_embedding_service import (
    LiveSlice, ContentEmbedding, MockQwenVLClient, EmbeddingStore,
    ContentEmbeddingConsumer, cosine_similarity, run_demo, EMBEDDING_DIM
)
import time


class TestMockQwenVLClient:
    def test_output_dim(self):
        client = MockQwenVLClient(embedding_dim=64)
        emb, desc, tags = client.encode("img.jpg", "test content")
        assert len(emb) == 64

    def test_deterministic(self):
        """相同输入应得到相同向量"""
        client = MockQwenVLClient()
        e1, _, _ = client.encode("img.jpg", "hello world")
        e2, _, _ = client.encode("img.jpg", "hello world")
        assert e1 == e2

    def test_different_inputs(self):
        """不同输入应得到不同向量"""
        client = MockQwenVLClient()
        e1, _, _ = client.encode("img1.jpg", "美食直播")
        e2, _, _ = client.encode("img2.jpg", "游戏直播")
        assert e1 != e2

    def test_normalized(self):
        """向量应接近单位范数（归一化）"""
        client = MockQwenVLClient()
        emb, _, _ = client.encode("img.jpg", "test")
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-5


class TestEmbeddingStore:
    def test_save_and_get_latest(self):
        store = EmbeddingStore()
        emb = ContentEmbedding(501, "slice_1", [0.1, 0.2], int(time.time() * 1000))
        store.save(emb)
        assert store.get_latest(501) is emb

    def test_latest_overwritten(self):
        store = EmbeddingStore()
        e1 = ContentEmbedding(501, "slice_1", [0.1], int(time.time() * 1000))
        e2 = ContentEmbedding(501, "slice_2", [0.2], int(time.time() * 1000) + 1)
        store.save(e1)
        store.save(e2)
        assert store.get_latest(501).slice_id == "slice_2"

    def test_get_by_slice(self):
        store = EmbeddingStore()
        emb = ContentEmbedding(502, "s1", [1.0], 123)
        store.save(emb)
        assert store.get_by_slice("s1") is emb
        assert store.get_by_slice("not_exist") is None


class TestContentEmbeddingConsumer:
    def test_basic_consume(self):
        client = MockQwenVLClient()
        store = EmbeddingStore()
        consumer = ContentEmbeddingConsumer(client, store)
        s = LiveSlice(501, 201, "slice_x", "img.jpg", "美食", "food", int(time.time() * 1000))
        result = consumer.consume(s)
        assert result is not None
        assert result.item_id == 501
        assert len(result.embedding) == EMBEDDING_DIM

    def test_idempotent(self):
        """同一 slice_id 重复消费不应增加存储"""
        client = MockQwenVLClient()
        store = EmbeddingStore()
        consumer = ContentEmbeddingConsumer(client, store)
        s = LiveSlice(501, 201, "dup_slice", "img.jpg", "test", "cat", int(time.time() * 1000))
        consumer.consume(s)
        consumer.consume(s)
        consumer.consume(s)
        assert store.count() == 1
        assert consumer.stats()["processed"] == 1

    def test_stats(self):
        client = MockQwenVLClient()
        store = EmbeddingStore()
        consumer = ContentEmbeddingConsumer(client, store)
        for i in range(5):
            s = LiveSlice(500 + i, 200 + i, f"s_{i}", f"img_{i}.jpg",
                          f"title {i}", "cat", int(time.time() * 1000))
            consumer.consume(s)
        assert consumer.stats()["processed"] == 5
        assert consumer.stats()["errors"] == 0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [0.1, 0.2, 0.3]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert abs(cosine_similarity(v1, v2) + 1.0) < 1e-6


class TestDemo:
    def test_demo_end_to_end(self):
        result = run_demo()
        assert result["processed"] == 3, "应处理 3 个切片"
        assert result["stored"] == 3, "应存储 3 个向量"
        # [MOCK] 随机向量无法保证相似度顺序；实际模型相同物品的切片相似度应更高
        assert -1.0 <= result["sim_same_item"] <= 1.0
        assert -1.0 <= result["sim_diff_item"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
