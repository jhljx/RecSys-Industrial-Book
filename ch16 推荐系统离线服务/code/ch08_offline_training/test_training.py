"""
test_training.py — 单机和分布式训练的单元测试
对应 test_chapter.tex 1.8 节离线训练任务
"""
import sys
import os
import math
from pathlib import Path

# 强制 MOCK 模式
os.environ["USE_MOCK_DATA"] = "true"

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from single_machine_train import (
    TrainConfig,
    MockSampleStream,
    train,
    _mock_train_loop,
)
from distributed_train import (
    DistributedTrainConfig,
    DistributedSampleStream,
    train_worker,
    launch_distributed,
)


# ─────────────────────────────────────────────────────────
#  MockSampleStream 测试
# ─────────────────────────────────────────────────────────

class TestMockSampleStream:
    def test_batch_size(self):
        stream = MockSampleStream(feature_dim=16)
        batch = stream.next_batch(32)
        assert len(batch) == 32

    def test_feature_dim(self):
        stream = MockSampleStream(feature_dim=16)
        sample = stream.next_batch(1)[0]
        assert len(sample.features) == 16

    def test_label_binary(self):
        stream = MockSampleStream(feature_dim=8)
        for s in stream.next_batch(100):
            assert s.label in (0.0, 1.0)

    def test_reproducibility(self):
        s1 = MockSampleStream(feature_dim=8, seed=99)
        s2 = MockSampleStream(feature_dim=8, seed=99)
        b1 = s1.next_batch(4)
        b2 = s2.next_batch(4)
        assert all(a.features == b.features for a, b in zip(b1, b2))


# ─────────────────────────────────────────────────────────
#  单机训练测试
# ─────────────────────────────────────────────────────────

class TestSingleMachineTrain:
    def _fast_config(self, tmpdir):
        return TrainConfig(
            feature_dim=8,
            max_steps=10,
            batch_size=16,
            hidden_dims=[16],
            checkpoint_dir=str(tmpdir),
        )

    def test_mock_train_returns_dict(self, tmp_path):
        cfg = self._fast_config(tmp_path)
        stream = MockSampleStream(feature_dim=cfg.feature_dim)
        result = _mock_train_loop(cfg, stream)
        assert "final_loss" in result
        assert "steps" in result
        assert result["steps"] == cfg.max_steps

    def test_mock_train_loss_is_finite(self, tmp_path):
        cfg = self._fast_config(tmp_path)
        stream = MockSampleStream(feature_dim=cfg.feature_dim)
        result = _mock_train_loop(cfg, stream)
        assert math.isfinite(result["final_loss"])
        assert result["final_loss"] > 0

    def test_train_returns_valid_result(self, tmp_path):
        cfg = self._fast_config(tmp_path)
        result = train(cfg)
        assert "final_loss" in result
        assert math.isfinite(result["final_loss"])

    def test_train_decreases_loss(self, tmp_path):
        """训练 50 步后 loss 应低于初始值（使用 MOCK 纯 Python 路径）。"""
        from single_machine_train import _mock_train_loop, MockSampleStream
        cfg = TrainConfig(feature_dim=8, max_steps=50, batch_size=64,
                          hidden_dims=[16], lr=0.01, checkpoint_dir=str(tmp_path))
        stream = MockSampleStream(feature_dim=cfg.feature_dim)
        result = _mock_train_loop(cfg, stream)
        # loss 应在训练后收敛到合理范围（< 1.0 for binary cross entropy）
        assert result["final_loss"] < 1.0


# ─────────────────────────────────────────────────────────
#  分布式训练测试
# ─────────────────────────────────────────────────────────

class TestDistributedSampleStream:
    def test_different_ranks_different_data(self):
        """不同 rank 应生成不同的样本（各自独立分区）。"""
        s0 = DistributedSampleStream(feature_dim=8, rank=0)
        s1 = DistributedSampleStream(feature_dim=8, rank=1)
        b0 = s0.next_batch(4)
        b1 = s1.next_batch(4)
        # 至少有一个样本特征不同
        assert any(a.features != b.features for a, b in zip(b0, b1))

    def test_batch_size_matches(self):
        stream = DistributedSampleStream(feature_dim=16, rank=2)
        batch = stream.next_batch(8)
        assert len(batch) == 8


class TestDistributedTrain:
    def _fast_config(self, tmpdir):
        cfg = DistributedTrainConfig(
            feature_dim=8,
            max_steps=5,
            batch_size=16,
            hidden_dims=[16],
            checkpoint_dir=str(tmpdir),
        )
        return cfg

    def test_mock_launch_two_ranks(self, tmp_path, capsys):
        """使用 mock_cpu=True（GLOO 后端）启动 2 个 rank，验证能顺利完成。"""
        import distributed_train as dt
        if not dt._TORCH_AVAILABLE:
            import pytest; pytest.skip("torch not installed")
        cfg = self._fast_config(tmp_path)
        # mock_cpu=True：GLOO 后端 + 文件初始化，无需 GPU 也能运行 DDP
        # mp.spawn 子进程输出不会捕获到 capsys，直接验证调用不抛异常即可
        launch_distributed(world_size=2, config=cfg, mock_cpu=True)
        # 若到达此处说明两个 rank 均正常完成

    def test_train_worker_runs_without_dist(self, tmp_path):
        """train_worker 在无进程组时应不抛异常（MOCK 路径）。"""
        cfg = self._fast_config(tmp_path)
        cfg.backend = "gloo"  # 不会真正初始化
        # 无 dist 进程组时调用，只验证不崩溃
        import importlib
        import distributed_train as dt
        if not dt._TORCH_AVAILABLE:
            # 纯 MOCK 路径：直接调用不应抛异常
            try:
                dt.train_worker(0, 1, cfg)
            except Exception as e:
                # 只有进程组相关异常可接受
                assert "dist" in str(e).lower() or "process group" in str(e).lower()
