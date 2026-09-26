"""
distributed_train.py — 分布式离线训练（PyTorch DDP）
对应 test_chapter.tex 1.8 节离线训练任务

本文件包含：
  1. DistributedTrainConfig  - 分布式训练超参数（nnodes、nproc_per_node 等）
  2. setup_dist / cleanup    - NCCL/GLOO 进程组初始化
  3. DistributedSampleStream - 每个 rank 消费独立分区（[MOCK] 内存模拟）
  4. train_worker()          - 单个 rank 的训练主函数（DDP + AMP + Checkpoint）
  5. launch_distributed()    - 多进程启动入口（mp.spawn 或 torchrun 兼容）

启动方式：
  单机多卡（2 GPU）：
    torchrun --nproc_per_node=2 distributed_train.py

  多机多卡（2 机各 2 GPU）：
    # node 0:
    torchrun --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
             --nproc_per_node=2 distributed_train.py
    # node 1:
    torchrun --nnodes=2 --node_rank=1 --master_addr=10.0.0.1 --master_port=29500 \
             --nproc_per_node=2 distributed_train.py

  无 GPU 单机测试（GLOO 后端）：
    python distributed_train.py --mock

[MOCK] 说明：
  - USE_MOCK_DATA=True（默认）：随机数据，无需 Kafka。
  - --mock 标志：使用 GLOO 后端在 CPU 上运行 DDP，无需 GPU。

依赖：
  pip install torch
"""
from __future__ import annotations

import os
import sys
import math
import random
import argparse
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.cuda.amp import GradScaler, autocast
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


USE_MOCK_DATA: bool = os.environ.get("USE_MOCK_DATA", "true").lower() != "false"


# ─────────────────────────────────────────────────────────
#  配置
# ─────────────────────────────────────────────────────────

@dataclass
class DistributedTrainConfig:
    # 分布式
    backend: str             = "nccl"   # "nccl" for GPU, "gloo" for CPU mock
    init_method: str         = "env://"
    # 数据
    feature_dim: int         = 64
    max_steps: int           = 50
    batch_size: int          = 256      # per-rank batch size
    # 模型
    hidden_dims: list[int]   = field(default_factory=lambda: [256, 128])
    dropout: float           = 0.1
    # 优化器
    lr: float                = 1e-3
    weight_decay: float      = 1e-5
    # AMP
    use_amp: bool            = True
    # Checkpoint（只由 rank-0 写入）
    checkpoint_dir: str      = "checkpoints/distributed"
    checkpoint_every: int    = 20
    # Gradient clipping
    max_grad_norm: float     = 1.0


# ─────────────────────────────────────────────────────────
#  分布式进程组管理
# ─────────────────────────────────────────────────────────

def setup_dist(rank: int, world_size: int, backend: str,
               init_method: str = "env://") -> None:
    """初始化 NCCL/GLOO 进程组。"""
    if not _TORCH_AVAILABLE:
        return
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    if backend == "nccl":
        torch.cuda.set_device(rank)
    print(f"[rank={rank}] dist setup done (backend={backend}, world_size={world_size})")


def cleanup_dist() -> None:
    """销毁进程组。"""
    if _TORCH_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────
#  [MOCK] 分布式样本流
#  每个 rank 按自身 rank_id 为 seed，独立生成不重叠的样本分区。
#  工业落地时：根据 Kafka 分区 ID（= rank_id）拉取对应 partition。
# ─────────────────────────────────────────────────────────

class DistributedSampleStream:
    """模拟每个 rank 独立读取 Kafka 分区的样本流。"""

    def __init__(self, feature_dim: int, rank: int, seed: int = 2024) -> None:
        self.feature_dim = feature_dim
        self._rng = random.Random(seed + rank * 1000)

    def next_batch(self, batch_size: int) -> list:
        """返回 batch_size 个 TrainSample（延迟导入避免循环依赖）。"""
        from single_machine_train import TrainSample
        samples = []
        for _ in range(batch_size):
            features = [self._rng.gauss(0, 1) for _ in range(self.feature_dim)]
            label = float(self._rng.random() < 0.3)
            samples.append(TrainSample(features, label))
        return samples


# ─────────────────────────────────────────────────────────
#  DNN 模型（与单机版相同，由 DDP 包装）
# ─────────────────────────────────────────────────────────

def build_model(config: DistributedTrainConfig):
    """构造 DNN 模型，与 single_machine_train 中结构相同。"""
    if not _TORCH_AVAILABLE:
        return None
    layers = []
    in_dim = config.feature_dim
    for out_dim in config.hidden_dims:
        layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(config.dropout)]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────
#  Checkpoint（仅 rank-0 写入）
# ─────────────────────────────────────────────────────────

def save_checkpoint_rank0(model_module, optimizer, step: int, loss: float,
                           config: DistributedTrainConfig, rank: int) -> Optional[Path]:
    """仅 rank-0 写入 checkpoint，其他 rank 跳过。"""
    if not _TORCH_AVAILABLE or rank != 0 or model_module is None:
        return None
    path = Path(config.checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    ckpt_path = path / f"ckpt_step{step}.pt"
    torch.save({
        "step": step,
        "loss": loss,
        "model_state_dict": model_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    return ckpt_path


# ─────────────────────────────────────────────────────────
#  AllReduce 指标同步（跨 rank 平均 loss）
# ─────────────────────────────────────────────────────────

def all_reduce_mean(tensor: "torch.Tensor") -> float:
    """对 loss 张量执行 AllReduce Sum，再除以 world_size，得到全局均值。"""
    if not _TORCH_AVAILABLE or not dist.is_initialized():
        return tensor.item()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


# ─────────────────────────────────────────────────────────
#  单 rank 训练主函数
# ─────────────────────────────────────────────────────────

def train_worker(rank: int, world_size: int, config: DistributedTrainConfig) -> None:
    """
    DDP 训练 worker — 每个进程运行此函数。

    关键设计点（对应 tex 1.8 节）：
      - DDP 自动在 backward() 时执行 AllReduce 梯度同步
      - Checkpoint 仅由 rank-0 写入
      - 混合精度（AMP）减少显存占用、提升吞吐
      - 梯度裁剪防止梯度爆炸
    """
    setup_dist(rank, world_size, config.backend, config.init_method)

    # 设备分配
    if _TORCH_AVAILABLE and torch.cuda.is_available() and config.backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu") if _TORCH_AVAILABLE else None

    # 模型 + DDP 包装
    model = build_model(config)
    if model is not None:
        model = model.to(device)
        ddp_device_ids = [rank] if (device is not None and str(device).startswith("cuda")) else None
        model = DDP(model, device_ids=ddp_device_ids)

    optimizer = (optim.Adam(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)
                 if model is not None else None)
    criterion = nn.BCEWithLogitsLoss() if _TORCH_AVAILABLE else None
    use_amp = config.use_amp and (device is not None and str(device).startswith("cuda"))
    scaler = GradScaler() if (use_amp and _TORCH_AVAILABLE) else None

    stream = DistributedSampleStream(config.feature_dim, rank)

    if rank == 0:
        print(f"[DistTrain] world_size={world_size}, device={device}, "
              f"steps={config.max_steps}, batch/rank={config.batch_size}, "
              f"amp={use_amp}")

    last_loss = float("inf")
    for step in range(config.max_steps):
        batch = stream.next_batch(config.batch_size)

        if model is not None and criterion is not None and device is not None:
            x = torch.tensor([s.features for s in batch],
                              dtype=torch.float32, device=device)
            y = torch.tensor([s.label for s in batch],
                              dtype=torch.float32, device=device)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    logits = model(x).squeeze(1)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            # AllReduce：同步各 rank loss，打印全局均值
            loss_t = torch.tensor(loss.item(), dtype=torch.float32, device=device)
            last_loss = all_reduce_mean(loss_t)
        else:
            # [MOCK] 无 PyTorch：数值模拟
            last_loss = 0.5 + 0.4 * math.exp(-step * 0.05)

        if step % 10 == 0 and rank == 0:
            print(f"  step={step:4d}  global_loss={last_loss:.4f}")

        # Checkpoint：仅 rank-0 写入
        if (step + 1) % config.checkpoint_every == 0 and rank == 0:
            ckpt = save_checkpoint_rank0(
                model.module if model is not None else None,
                optimizer, step + 1, last_loss, config, rank)
            if ckpt:
                print(f"  [Checkpoint] rank-0 saved {ckpt}")

    # 最终 checkpoint
    if rank == 0:
        save_checkpoint_rank0(
            model.module if model is not None else None,
            optimizer, config.max_steps, last_loss, config, rank)
        print(f"[DistTrain] done. global_loss={last_loss:.4f}")

    cleanup_dist()


# ─────────────────────────────────────────────────────────
#  多进程启动
# ─────────────────────────────────────────────────────────

def launch_distributed(
    world_size: int = 2,
    config: Optional[DistributedTrainConfig] = None,
    mock_cpu: bool = False,
) -> None:
    """
    在本机启动 world_size 个训练进程（mp.spawn）。
    - 用于测试或单机多卡场景。
    - 多机多卡场景请使用 torchrun（见文件头注释）。
    """
    if config is None:
        config = DistributedTrainConfig()

    if mock_cpu:
        config.backend = "gloo"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".lock") as f:
            init_file = f.name
        config.init_method = f"file://{init_file}"

    if not _TORCH_AVAILABLE:
        # [MOCK] 无 PyTorch：顺序模拟各 rank
        for rank in range(world_size):
            print(f"[MOCK] simulating rank={rank}/{world_size}")
            train_worker(rank, world_size, config)
        return

    mp.spawn(
        fn=train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed DDP Training")
    parser.add_argument("--mock", action="store_true",
                        help="Use CPU GLOO backend for testing (no GPU needed)")
    parser.add_argument("--world_size", type=int, default=2,
                        help="Number of processes (= GPUs for NCCL)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Training steps per rank")
    args = parser.parse_args()

    cfg = DistributedTrainConfig(max_steps=args.steps)
    launch_distributed(
        world_size=args.world_size,
        config=cfg,
        mock_cpu=args.mock,
    )
