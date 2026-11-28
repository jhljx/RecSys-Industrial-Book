"""第25章：全局/上下文权重及 Pointwise、Pairwise、Listwise 融合；对应§25.4–25.5。"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_data, request_batches, split_data


class GlobalFusion(nn.Module):
    def __init__(self, tasks=3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(tasks))

    def forward(self, data):
        weights = self.logits.softmax(-1)
        return (data.pxtr * weights).sum(-1), weights


class DynamicFusion(nn.Module):
    def __init__(self, tasks=3):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(4 + 4 + 3, 24), nn.ReLU(), nn.Linear(24, tasks))

    def forward(self, data):
        user = data.user_dense[:, None, :].expand(-1, data.pxtr.shape[1], -1)
        context = data.context[:, None, :].expand_as(user[:, :, :3])
        weights = self.gate(torch.cat((user, data.item_dense, context), -1)).softmax(-1)
        return (data.pxtr*weights).sum(-1), weights


class LTRFusion(nn.Module):
    def __init__(self, tasks=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(tasks + 4 + 4 + 3, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, data):
        n = data.pxtr.shape[1]
        user = data.user_dense[:, None, :].expand(-1, n, -1)
        context = data.context[:, None, :].expand(-1, n, -1)
        x = torch.cat((data.pxtr, user, data.item_dense, context), -1)
        return self.net(x).squeeze(-1)


def ranking_loss(score, relevance, exposed, kind):
    """只使用已曝光真实反馈；Listwise 用稳定破平局的 P-L 排列似然。"""
    if kind == "pointwise":
        return F.mse_loss(score[exposed], relevance[exposed] / 2)
    if kind == "pairwise":
        diff = relevance.unsqueeze(2) - relevance.unsqueeze(1)
        valid = exposed.unsqueeze(2) & exposed.unsqueeze(1) & (diff > 0)
        return F.softplus(-(score.unsqueeze(2) - score.unsqueeze(1))[valid]).mean()
    if kind == "listwise":
        losses = []
        for q in range(len(score)):
            chosen = torch.where(exposed[q])[0]
            if len(chosen) < 2:
                continue
            # P-L 需要完整排列；同级反馈按原候选顺序破平局（教学约定）。
            ordered = chosen[relevance[q, chosen].argsort(descending=True, stable=True)]
            logits = score[q, ordered]
            terms = torch.logcumsumexp(logits.flip(0), 0).flip(0) - logits
            losses.append(terms.mean())
        return torch.stack(losses).mean()
    raise ValueError(kind)


def train(model, train_data, objective="pairwise", epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for batch in request_batches(train_data):
            output = model(batch)
            score = output[0] if isinstance(output, tuple) else output
            loss = ranking_loss(score, batch.relevance, batch.exposed, objective)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    train_data, valid = split_data(make_data())
    for name, model, objective in (
        ("全局权重", GlobalFusion(), "pointwise"),
        ("动态权重", DynamicFusion(), "pairwise"),
        ("LTR-Pointwise", LTRFusion(), "pointwise"),
        ("LTR-Pairwise", LTRFusion(), "pairwise"),
        ("LTR-Listwise", LTRFusion(), "listwise"),
    ):
        train(model, train_data, objective)
        with torch.no_grad():
            output = model(valid)
            score = output[0] if isinstance(output, tuple) else output
            print(name, "验证排序损失:", round(float(ranking_loss(score, valid.relevance, valid.exposed, objective)), 4))
