"""
Qwen2.5-VL 图片向量 RPC 服务 — 对应 test_chapter.tex 中 code_qwen_vl_rpc。

本文件包含：
  - VisionEngine:            模型加载、图片解码、批量推理（标签 + 向量）
  - JsonlAnnotationRepository: 标注持久化（线程安全 JSONL 写入）
  - ImageEmbeddingService:   gRPC Servicer，处理请求并写入标注
  - serve():                 启动 gRPC 服务

[MOCK] 说明：
  - 当 USE_MOCK_MODEL=True（默认）时，VisionEngine 使用随机向量模拟推理，
    无需安装 transformers / torch，可直接运行测试。
  - 工业落地时设 USE_MOCK_MODEL=False，并安装 requirements_gpu.txt。

Proto 文件：image_embedding_service.proto
生成命令：
  python -m grpc_tools.protoc -I. \
    --python_out=. --grpc_python_out=. \
    image_embedding_service.proto
"""
from __future__ import annotations

import json
import os
import threading
from concurrent import futures
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import importlib.util

# ── 控制是否使用 MOCK 模型（无需 GPU / transformers） ──
USE_MOCK_MODEL: bool = os.environ.get("USE_MOCK_MODEL", "true").lower() != "false"

import grpc
import numpy as np

# ── 尝试导入生成的 gRPC stub；如无则用内存模拟 ──
try:
    import image_embedding_service_pb2 as image_api
    import image_embedding_service_pb2_grpc as image_rpc
    _GRPC_STUBS_AVAILABLE = True
except ImportError:
    _GRPC_STUBS_AVAILABLE = False
    # [MOCK] 用 SimpleNamespace 模拟 proto message
    from types import SimpleNamespace as _NS

    class _MockApi:
        @staticmethod
        def ImageEmbeddingResponse(**kwargs):
            return _NS(**kwargs)

    class _MockRpc:
        class ImageEmbeddingServiceServicer:
            pass

        @staticmethod
        def add_ImageEmbeddingServiceServicer_to_server(servicer, server):
            pass

    image_api = _MockApi()
    image_rpc = _MockRpc()


IMAGE_SIDE = 448
MAX_BATCH_SIZE = 8
PROMPT = (
    "仅依据图片可见画面判断垂类。"
    "从[游戏、才艺、户外、体育、知识、购物、美食、其他]中选择 category，"
    "给出 confidence 和不超过三个 tags。只输出 JSON。"
)


# ─────────────────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Annotation:
    """单张图片的垂类标注结果。"""
    category: str
    confidence: float
    tags: list[str]


# ─────────────────────────────────────────────────────────
#  标注持久化
# ─────────────────────────────────────────────────────────

class JsonlAnnotationRepository:
    """
    线程安全的 JSONL 标注写入器。
    工业落地时可替换为写 Kafka、MySQL 或对象存储。
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.lock = threading.Lock()

    def write_batch(self, rows: list[Annotation]) -> None:
        payload = "".join(
            json.dumps(asdict(row), ensure_ascii=False) + "\n"
            for row in rows
        )
        with self.lock, self.output_path.open("a", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())


class InMemoryAnnotationRepository:
    """[MOCK] 内存标注仓库，用于测试。"""

    def __init__(self) -> None:
        self._rows: list[Annotation] = []
        self.lock = threading.Lock()

    def write_batch(self, rows: list[Annotation]) -> None:
        with self.lock:
            self._rows.extend(rows)

    def all(self) -> list[Annotation]:
        with self.lock:
            return list(self._rows)

    def count(self) -> int:
        return len(self._rows)


# ─────────────────────────────────────────────────────────
#  视觉引擎
# ─────────────────────────────────────────────────────────

class VisionEngine:
    """
    Qwen2.5-VL 视觉引擎。

    USE_MOCK_MODEL=True: 随机向量模拟，无需 GPU。
    USE_MOCK_MODEL=False: 加载真实 Qwen2.5-VL 模型。
    """

    def __init__(self, model_id: str, use_mock: bool = USE_MOCK_MODEL) -> None:
        self.use_mock = use_mock
        self._emb_dim: int = 512

        if not use_mock:
            # ── 真实模型加载（工业落地路径） ──
            import torch
            from PIL import Image as _PIL
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            ).eval()
            self._emb_dim = self.model.config.hidden_size
        else:
            # [MOCK] 不加载任何模型权重
            self.processor = None
            self.model = None

    def decode(
        self,
        values: list[float],
        image_dim: int,
        batch_size: int,
    ):
        """将展平的归一化像素数组恢复为 PIL Image 列表。"""
        if self.use_mock:
            # [MOCK] 返回纯色 PIL Image
            try:
                from PIL import Image
                return [Image.new("RGB", (IMAGE_SIDE, IMAGE_SIDE), (128, 128, 128))
                        for _ in range(batch_size)]
            except ImportError:
                return [None] * batch_size

        # ── 真实解码 ──
        import torch
        from PIL import Image

        expected = 3 * IMAGE_SIDE * IMAGE_SIDE
        if image_dim != expected or len(values) != image_dim * batch_size:
            raise ValueError("inconsistent image dimensions")
        array = np.asarray(values, dtype=np.float32).reshape(
            batch_size, 3, IMAGE_SIDE, IMAGE_SIDE
        )
        pixels = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return [
            Image.fromarray(row.transpose(1, 2, 0), "RGB")
            for row in pixels
        ]

    def infer(
        self, images: list
    ) -> tuple[list[Annotation], list[list[float]]]:
        """
        批量推理：
        - 文本生成路径 → 垂类标注
        - 视觉编码路径 → 内容向量
        """
        if self.use_mock:
            return self._mock_infer(images)

        # ── tex snippet: code_qwen_vl_rpc ──
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
            for image in images
        ]
        prompts = [
            self.processor.apply_chat_template(
                [message], tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]
        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            tokens = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )
            answers = self.processor.batch_decode(
                tokens[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            visual = self.model.visual(
                inputs.pixel_values, grid_thw=inputs.image_grid_thw
            )
            lengths = (
                inputs.image_grid_thw[:, 0]
                * inputs.image_grid_thw[:, 1]
                * inputs.image_grid_thw[:, 2]
            ).tolist()
            vectors = [
                part.float().mean(dim=0).cpu().tolist()
                for part in torch.split(visual, lengths, dim=0)
            ]

        return [self._parse(answer) for answer in answers], vectors

    def _mock_infer(
        self, images: list
    ) -> tuple[list[Annotation], list[list[float]]]:
        """[MOCK] 随机生成标注和向量，不调用任何模型。"""
        annotations = [
            Annotation("游戏", 0.85, ["直播", "手游"]) for _ in images
        ]
        rng = np.random.default_rng(seed=42)
        vectors = [
            rng.standard_normal(self._emb_dim).tolist() for _ in images
        ]
        return annotations, vectors

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    @staticmethod
    def _parse(text: str) -> Annotation:
        """将模型生成文本解析为 Annotation；解析失败时返回安全默认值。"""
        try:
            data = json.loads(text[text.find("{") : text.rfind("}") + 1])
            return Annotation(
                category=str(data["category"]),
                confidence=float(data["confidence"]),
                tags=[str(tag) for tag in data.get("tags", [])[:3]],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return Annotation("其他", 0.0, [])


# ─────────────────────────────────────────────────────────
#  gRPC Servicer
# ─────────────────────────────────────────────────────────

class ImageEmbeddingService(image_rpc.ImageEmbeddingServiceServicer):
    """
    gRPC 图片向量服务 — 对应 tex 代码片段 code_qwen_vl_rpc。

    单次 RPC 流程：
      1. 校验 batch_size
      2. 恢复图片（decode）
      3. 批量推理（infer）→ 标注 + 向量
      4. 写入标注仓库
      5. 组装并返回 ImageEmbeddingResponse
    """

    def __init__(
        self,
        engine: VisionEngine,
        annotations,  # JsonlAnnotationRepository | InMemoryAnnotationRepository
    ) -> None:
        self.engine = engine
        self.annotations = annotations

    def Get(self, request, context):
        if not 1 <= request.batch_size <= MAX_BATCH_SIZE:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"invalid batch_size: {request.batch_size}",
            )
            return None

        try:
            images = self.engine.decode(
                list(request.image_list), request.image_dim, request.batch_size
            )
            labels, vectors = self.engine.infer(images)
        except (ValueError, OSError, RuntimeError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return None

        self.annotations.write_batch(labels)
        return image_api.ImageEmbeddingResponse(
            image_embedding=[value for vector in vectors for value in vector],
            emb_dim=len(vectors[0]) if vectors else 0,
            batch_size=len(vectors),
        )


# ─────────────────────────────────────────────────────────
#  服务启动
# ─────────────────────────────────────────────────────────

def serve(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    port: int = 50051,
    annotation_path: Path = Path("image_annotations.jsonl"),
    use_mock: bool = USE_MOCK_MODEL,
) -> None:
    """
    启动 gRPC 服务。

    工业落地：
      USE_MOCK_MODEL=false python qwen_vl_rpc_service.py
    测试模式：
      python qwen_vl_rpc_service.py  （默认使用 MOCK）
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16),
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),   # 64 MB
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
    )
    engine = VisionEngine(model_id, use_mock=use_mock)

    if use_mock:
        repo = InMemoryAnnotationRepository()
    else:
        repo = JsonlAnnotationRepository(annotation_path)

    service = ImageEmbeddingService(engine, repo)
    image_rpc.add_ImageEmbeddingServiceServicer_to_server(service, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[QwenVLRpc] gRPC server started on port {port} (mock={use_mock})")
    server.wait_for_termination()


if __name__ == "__main__":
    # 支持通过环境变量 GRPC_PORT 覆盖端口（供集成测试使用）
    _port = int(os.environ.get("GRPC_PORT", "50051"))
    serve(port=_port)
