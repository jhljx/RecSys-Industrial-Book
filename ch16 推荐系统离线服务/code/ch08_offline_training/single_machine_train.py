"""
single_machine_train.py — 单机离线训练
对应 test_chapter.tex 1.8 节离线训练任务

功能：
  - Kafka 样本流消费（[MOCK] 内存生成器模拟）
  - 特征预处理 + DataLoader 构造
  - 单机 PyTorch 训练循环（DNN 召回 / 排序模型示例）
  - 混合精度训练（AMP）
  - Checkpoint 保存与恢复
  - 训练指标监控（loss、AUC 估算）

[MOCK] 说明：
  - USE_MOCK_DATA=True（默认）：使用随机张量代替真实样本，无需安装 Kafka。
  - 工业落地时：
      1. 将 MockSampleStream 替换为 KafkaSampleStream（读取真实 Kafka Topic）
      2. 将 RecoModel 替换为真实 DNN / Transformer 模型
      3. 将 checkpoint 路径改为分布式文件系统（如 HDFS / OSS）

依赖（所有为可选）：
  pip install torch          # GPU 训练
  pip install scikit-learn   # AUC 计算
"""
from __future__ import annotations

import os
import math
import time
import random
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# ── PyTorch（可选） ──
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import GradScaler, autocast
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


USE_MOCK_DATA: bool = os.environ.get("USE_MOCK_DATA", "true").lower() != "false"


# ─────────────────────────────────────────────────────────
#  超参数配置
# ─────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # 数据
    feature_dim: int       = 64
    max_steps: int         = 100
    batch_size: int        = 256
    # 模型
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float         = 0.1
    # 优化器
    lr: float              = 1e-3
    weight_decay: float    = 1e-5
    # AMP
    use_amp: bool          = True
    # Checkpoint
    checkpoint_dir: str    = "checkpoints/single"
    checkpoint_every: int  = 20
    # 设备
    device: str            = "cuda" if (_TORCH_AVAILABLE and
                                        __import__("torch").cuda.is_available()) else "cpu"


# ─────────────────────────────────────────────────────────
#  [MOCK] 样本流（模拟 Kafka 消费）
# ─────────────────────────────────────────────────────────

@dataclass
class TrainSample:
    """单条训练样本：特征向量 + 标签。"""
    features: list[float]
    label: float  # 0 或 1


class MockSampleStream:
    """
    [MOCK] 内存样本流，模拟从 Kafka 消费的训练样本。
    工业落地时替换为 KafkaSampleStream。
    """

    def __init__(self, feature_dim: int, seed: int = 42) -> None:
        self.feature_dim = feature_dim
        self._rng = random.Random(seed)

    def next_batch(self, batch_size: int) -> list[TrainSample]:
        samples = []
        for _ in range(batch_size):
            features = [self._rng.gauss(0, 1) for _ in range(self.feature_dim)]
            # 模拟正样本率约 30%
            label = float(self._rng.random() < 0.3)
            samples.append(TrainSample(features, label))
        return samples


# ─────────────────────────────────────────────────────────
#  模型定义（PyTorch DNN）
# ─────────────────────────────────────────────────────────

def build_model(config: TrainConfig):
    """构造简单 DNN 二分类模型（点击率预估示例）。"""
    if not _TORCH_AVAILABLE:
        return None

    layers = []
    in_dim = config.feature_dim
    for out_dim in config.hidden_dims:
        layers += [
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        ]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────
#  Checkpoint 工具
# ─────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step: int, loss: float,
                    checkpoint_dir: str) -> Path:
    """保存训练 checkpoint。"""
    if not _TORCH_AVAILABLE or model is None:
        return Path(checkpoint_dir) / f"mock_ckpt_step{step}.pt"

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    ckpt_path = path / f"ckpt_step{step}.pt"
    torch.save({
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    return ckpt_path


def load_latest_checkpoint(model, optimizer, checkpoint_dir: str
                            ) -> tuple[int, float]:
    """恢复最新 checkpoint，返回 (start_step, last_loss)。"""
    if not _TORCH_AVAILABLE or model is None:
        return 0, float("inf")

    path = Path(checkpoint_dir)
    if not path.exists():
        return 0, float("inf")

    ckpts = sorted(path.glob("ckpt_step*.pt"),
                   key=lambda p: int(p.stem.split("step")[1]))
    if not ckpts:
        return 0, float("inf")

    data = torch.load(ckpts[-1])
    model.load_state_dict(data["model_state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    print(f"[Checkpoint] Resumed from {ckpts[-1]} (step={data['step']})")
    return data["step"], data["loss"]


# ─────────────────────────────────────────────────────────
#  训练循环（单机）
# ─────────────────────────────────────────────────────────

def train(config: Optional[TrainConfig] = None) -> dict:
    """
    单机训练入口。
    返回 {"final_loss": float, "steps": int, "checkpoint": str}。
    """
    if config is None:
        config = TrainConfig()

    stream = MockSampleStream(feature_dim=config.feature_dim)

    if not _TORCH_AVAILABLE:
        # ── [MOCK] 无 PyTorch：用纯 Python 数值模拟训练循环 ──
        return _mock_train_loop(config, stream)

    device = torch.device(config.device)
    model = build_model(config).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() if (config.use_amp and device.type == "cuda") else None

    start_step, _ = load_latest_checkpoint(model, optimizer, config.checkpoint_dir)

    print(f"[SingleTrain] device={device}, steps={config.max_steps}, "
          f"batch={config.batch_size}, amp={config.use_amp and scaler is not None}")

    last_loss = float("inf")
    for step in range(start_step, config.max_steps):
        batch = stream.next_batch(config.batch_size)
        x = torch.tensor([[s.features] for s in batch],
                          dtype=torch.float32, device=device).squeeze(1)
        y = torch.tensor([[s.label] for s in batch],
                          dtype=torch.float32, device=device).squeeze(1)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        last_loss = loss.item()
        if step % 10 == 0:
            print(f"  step={step:4d}  loss={last_loss:.4f}")

        if (step + 1) % config.checkpoint_every == 0:
            ckpt = save_checkpoint(model, optimizer, step + 1,
                                   last_loss, config.checkpoint_dir)
            print(f"  [Checkpoint] saved {ckpt}")

    final_ckpt = save_checkpoint(model, optimizer, config.max_steps,
                                 last_loss, config.checkpoint_dir)
    print(f"[SingleTrain] done. final_loss={last_loss:.4f}")
    return {"final_loss": last_loss, "steps": config.max_steps,
            "checkpoint": str(final_ckpt)}


def _mock_train_loop(config: TrainConfig, stream: MockSampleStream) -> dict:
    """[MOCK] 无 PyTorch 的纯 Python 训练循环（用于测试）。"""
    # 简单感知器：随机权重 + SGD
    rng = random.Random(0)
    weights = [rng.gauss(0, 0.01) for _ in range(config.feature_dim)]
    bias = 0.0
    lr = config.lr
    last_loss = float("inf")

    for step in range(config.max_steps):
        batch = stream.next_batch(config.batch_size)
        total_loss = 0.0
        for sample in batch:
            # 前向
            logit = sum(w * f for w, f in zip(weights, sample.features)) + bias
            prob = 1.0 / (1.0 + math.exp(-max(-30, min(30, logit))))
            loss = -(sample.label * math.log(prob + 1e-9) +
                     (1 - sample.label) * math.log(1 - prob + 1e-9))
            total_loss += loss
            # 反向（梯度 = prob - label）
            grad = prob - sample.label
            for i, f in enumerate(sample.features):
                weights[i] -= lr * grad * f
            bias -= lr * grad
        last_loss = total_loss / len(batch)
        if step % 20 == 0:
            print(f"  [MOCK] step={step:4d}  loss={last_loss:.4f}")

    return {"final_loss": last_loss, "steps": config.max_steps,
            "checkpoint": "mock_checkpoint"}


if __name__ == "__main__":
    result = train()
    print("Training result:", result)
