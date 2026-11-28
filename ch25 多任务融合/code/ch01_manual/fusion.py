"""第25章：人工多任务融合，预测值/名次/值序混合；对应§25.2。"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_data, normalized_ranks


def value_fusion(pxtr, weights=None, mode="add", alpha=None, beta=None, gamma=None):
    """加法或含缩放、偏置和指数的乘法公式。"""
    if mode == "add":
        if weights is None:
            raise ValueError("加法融合需要 weights")
        weights = torch.as_tensor(weights, dtype=pxtr.dtype, device=pxtr.device)
        return (pxtr * weights).sum(-1)
    if mode != "multiply":
        raise ValueError("mode 必须是 add 或 multiply")
    if weights is not None or alpha is None or beta is None or gamma is None:
        raise ValueError("乘法融合只接收 alpha、beta、gamma，不接收加法权重")
    alpha = torch.as_tensor(alpha, dtype=pxtr.dtype, device=pxtr.device)
    beta = torch.as_tensor(beta, dtype=pxtr.dtype, device=pxtr.device)
    gamma = torch.as_tensor(gamma, dtype=pxtr.dtype, device=pxtr.device)
    if torch.any(beta + alpha * pxtr <= 0):
        raise ValueError("乘法底数必须为正，避免非整数指数产生 NaN")
    return (beta + alpha*pxtr).pow(gamma).prod(-1)


def rank_fusion(pxtr, weights):
    rank = normalized_ranks(pxtr)
    return (torch.as_tensor(weights, dtype=pxtr.dtype) / rank).sum(-1)


def hybrid_fusion(pxtr, weights):
    rank = normalized_ranks(pxtr)
    return (torch.as_tensor(weights, dtype=pxtr.dtype) * pxtr / rank).sum(-1)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch = make_data(8)
    for name, score in (
        ("加权值", value_fusion(batch.pxtr, [0.5, 0.3, 0.2])),
        ("乘法值", value_fusion(batch.pxtr, mode="multiply", alpha=[1]*3, beta=[0.2]*3, gamma=[1]*3)),
        ("名次", rank_fusion(batch.pxtr, [0.5, 0.3, 0.2])),
        ("值序混合", hybrid_fusion(batch.pxtr, [0.5, 0.3, 0.2])),
    ):
        print(name, "首请求 Top-3 位置:", score[0].topk(3).indices.tolist())
