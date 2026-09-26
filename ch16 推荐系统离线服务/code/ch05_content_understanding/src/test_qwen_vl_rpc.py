"""
test_qwen_vl_rpc.py — 对应 tex 代码片段 code_qwen_vl_rpc 的单元测试。

测试策略：
  - 全部使用 USE_MOCK_MODEL=True（无需 GPU / transformers）
  - 通过直接调用 ImageEmbeddingService 方法（不启动 gRPC server）
"""
import sys
import os
import types

# ── 为没有安装 grpc 的环境提供 stub ──
try:
    import grpc
except ImportError:
    grpc = types.ModuleType("grpc")
    grpc.StatusCode = types.SimpleNamespace(INVALID_ARGUMENT="INVALID_ARGUMENT")
    sys.modules["grpc"] = grpc

# ── 确保以 MOCK 模式运行 ──
os.environ["USE_MOCK_MODEL"] = "true"

import pytest
from pathlib import Path
from unittest.mock import MagicMock

# 将 src 目录加入 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qwen_vl_rpc_service import (
    VisionEngine,
    ImageEmbeddingService,
    InMemoryAnnotationRepository,
    Annotation,
)


# ─────────────────────────────────────────────────────────
#  VisionEngine 测试
# ─────────────────────────────────────────────────────────

class TestVisionEngine:
    def setup_method(self):
        self.engine = VisionEngine("mock", use_mock=True)

    def test_decode_returns_image_list(self):
        # 小批次解码：不依赖像素值，只验证数量
        images = self.engine.decode([0.0] * (3 * 448 * 448 * 2), 3 * 448 * 448, 2)
        assert len(images) == 2

    def test_infer_returns_correct_batch_size(self):
        images = [MagicMock() for _ in range(3)]
        annotations, vectors = self.engine.infer(images)
        assert len(annotations) == 3
        assert len(vectors) == 3

    def test_infer_vectors_are_nonempty(self):
        images = [MagicMock()]
        _, vectors = self.engine.infer(images)
        assert len(vectors[0]) > 0

    def test_parse_valid_json(self):
        text = '{"category": "游戏", "confidence": 0.9, "tags": ["直播"]}'
        ann = VisionEngine._parse(text)
        assert ann.category == "游戏"
        assert ann.confidence == pytest.approx(0.9)
        assert "直播" in ann.tags

    def test_parse_invalid_json_returns_default(self):
        ann = VisionEngine._parse("not json at all")
        assert ann.category == "其他"
        assert ann.confidence == pytest.approx(0.0)

    def test_parse_missing_field_returns_default(self):
        ann = VisionEngine._parse('{"foo": "bar"}')
        assert ann.category == "其他"


# ─────────────────────────────────────────────────────────
#  InMemoryAnnotationRepository 测试
# ─────────────────────────────────────────────────────────

class TestInMemoryAnnotationRepository:
    def test_write_and_read(self):
        repo = InMemoryAnnotationRepository()
        rows = [Annotation("游戏", 0.8, ["tag1"]), Annotation("美食", 0.7, [])]
        repo.write_batch(rows)
        assert repo.count() == 2
        assert repo.all()[0].category == "游戏"

    def test_multiple_writes_accumulate(self):
        repo = InMemoryAnnotationRepository()
        repo.write_batch([Annotation("游戏", 0.8, [])])
        repo.write_batch([Annotation("美食", 0.7, []), Annotation("户外", 0.6, [])])
        assert repo.count() == 3


# ─────────────────────────────────────────────────────────
#  ImageEmbeddingService 测试
# ─────────────────────────────────────────────────────────

class TestImageEmbeddingService:
    def _make_service(self):
        engine = VisionEngine("mock", use_mock=True)
        repo = InMemoryAnnotationRepository()
        return ImageEmbeddingService(engine, repo), repo

    def _make_request(self, batch_size: int = 2):
        """构造 mock RPC request。"""
        req = MagicMock()
        req.batch_size = batch_size
        req.image_dim = 3 * 448 * 448
        req.image_list = [0.0] * (req.image_dim * batch_size)
        return req

    def test_valid_request_returns_embeddings(self):
        service, repo = self._make_service()
        context = MagicMock()
        request = self._make_request(batch_size=2)

        response = service.Get(request, context)

        assert response is not None
        assert response.batch_size == 2
        assert response.emb_dim > 0
        assert len(response.image_embedding) == response.batch_size * response.emb_dim

    def test_annotations_written(self):
        service, repo = self._make_service()
        context = MagicMock()
        request = self._make_request(batch_size=3)

        service.Get(request, context)
        assert repo.count() == 3

    def test_invalid_batch_size_zero(self):
        service, _ = self._make_service()
        context = MagicMock()
        request = self._make_request(batch_size=0)

        service.Get(request, context)
        context.abort.assert_called_once()

    def test_invalid_batch_size_too_large(self):
        service, _ = self._make_service()
        context = MagicMock()
        request = self._make_request(batch_size=9)

        service.Get(request, context)
        context.abort.assert_called_once()
