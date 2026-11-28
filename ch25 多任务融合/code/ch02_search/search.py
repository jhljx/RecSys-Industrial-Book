"""第25章：固定融合公式的网格/随机/贝叶斯/交叉熵参数寻优；对应§25.3。"""
from pathlib import Path
import itertools
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_data, split_data
from ch01_manual.fusion import value_fusion


def objective(pxtr, labels, exposed, weights):
    """[MOCK] 离线代理：只对有反馈的已曝光候选计算请求内排序一致性。"""
    score = value_fusion(pxtr, weights)
    diff = score.unsqueeze(2) - score.unsqueeze(1)
    target = labels.unsqueeze(2) - labels.unsqueeze(1)
    valid = exposed.unsqueeze(2) & exposed.unsqueeze(1) & (target > 0)
    if not valid.any():
        return torch.tensor(0.)
    return (torch.sigmoid(diff[valid])).mean()


def grid_search(evaluate):
    candidates = [torch.tensor(w) for w in itertools.product([0.2, 0.5, 0.8], repeat=3)]
    return max(candidates, key=lambda w: float(evaluate(w)))


def random_search(evaluate, trials=24):
    return max((torch.rand(3) for _ in range(trials)), key=lambda w: float(evaluate(w)))


def bayesian_search(evaluate, trials=16, warmup=5):
    """小规模 RBF 高斯过程代理 + 置信上界；核超参为教学配置。"""
    points = [torch.rand(3) for _ in range(warmup)]
    values = [float(evaluate(p)) for p in points]
    for _ in range(trials - warmup):
        x = torch.stack(points)
        y = torch.tensor(values)
        pool = torch.rand(128, 3)
        kernel = torch.exp(-torch.cdist(x, x).square() / 0.25) + 1e-4*torch.eye(len(x))
        cross = torch.exp(-torch.cdist(pool, x).square() / 0.25)
        chol = torch.linalg.cholesky(kernel)
        mean = cross @ torch.cholesky_solve(y[:, None], chol).squeeze(1)
        variance = 1 - (torch.linalg.solve_triangular(chol, cross.T, upper=False).square().sum(0))
        candidate = pool[(mean + 0.2*variance.clamp_min(0).sqrt()).argmax()]
        points.append(candidate)
        values.append(float(evaluate(candidate)))
    return points[max(range(len(values)), key=values.__getitem__)]


def cross_entropy_search(evaluate, rounds=5, population=24, elite=6):
    mean, std = torch.full((3,), 0.5), torch.full((3,), 0.25)
    best, best_value = None, float("-inf")
    for _ in range(rounds):
        draws = (mean + std*torch.randn(population, 3)).clamp(0, 1)
        values = torch.tensor([float(evaluate(p)) for p in draws])
        idx = values.topk(elite).indices
        mean = draws[idx].mean(0)
        std = draws[idx].std(0).clamp_min(0.03)
        if float(values.max()) > best_value:
            best_value = float(values.max())
            best = draws[values.argmax()]
    if best is None:
        raise ValueError("rounds must be positive")
    return best


if __name__ == "__main__":
    torch.manual_seed(42)
    train, valid = split_data(make_data())
    train_score = lambda w: objective(train.pxtr, train.relevance, train.exposed, w)
    for name, search in (("网格", grid_search), ("随机", random_search),
                         ("贝叶斯", bayesian_search), ("交叉熵", cross_entropy_search)):
        weight = search(train_score)
        print(name, "参数:", weight.tolist(), "验证一致性:",
              round(float(objective(valid.pxtr, valid.relevance, valid.exposed, weight)), 4))
