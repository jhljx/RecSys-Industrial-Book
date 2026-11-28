"""第25章 EMER：请求级 Transformer、后验/先验 Pairwise、自演化 AE 与 IPUT；arXiv:2508.05093。"""
from pathlib import Path
import sys
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_data, normalized_ranks, request_batches, split_train_adapt_test


def prior_targets(data, use_iput=True):
    pxtr = data.pxtr.clone()
    if use_iput:
        # 第三个目标模拟互动率；观看时长已转成秒，避免除零。
        pxtr[..., 2] = pxtr[..., 2] / data.watchtime.clamp_min(1e-6)
    return pxtr


class EMER(nn.Module):
    def __init__(self, tasks=3, hidden=48):
        super().__init__()
        self.input = nn.Linear(4 + 4 + 3 + 2*tasks, hidden)
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=96,
                                           dropout=0.0, batch_first=True)
        self.list_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(hidden, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, data):
        n = data.pxtr.shape[1]
        prior = prior_targets(data)
        user = data.user_dense[:, None, :].expand(-1, n, -1)
        context = data.context[:, None, :].expand(-1, n, -1)
        features = torch.cat((user, context, data.item_dense, prior, normalized_ranks(prior)), -1)
        representation = self.list_encoder(self.input(features))
        return self.head(representation).squeeze(-1)


def pairwise_losses(score, data, weights=None):
    """后验仅用已曝光真实反馈；先验对全体候选逐目标比较。"""
    n = score.shape[1]
    i, j = torch.triu_indices(n, n, offset=1, device=score.device)
    delta = score[:, i] - score[:, j]
    satisfaction = data.feedback.sum(-1)
    posterior_diff = satisfaction[:, i] - satisfaction[:, j]
    observed = data.exposed[:, i] & data.exposed[:, j] & (posterior_diff != 0)
    posterior = F.softplus(-delta[observed]*posterior_diff[observed].sign()).mean() if observed.any() else delta.sum()*0
    prior = prior_targets(data)
    losses = []
    for task in range(prior.shape[-1]):
        target = (prior[:, i, task] > prior[:, j, task]).float()
        unequal = prior[:, i, task] != prior[:, j, task]
        losses.append(F.binary_cross_entropy_with_logits(delta[unequal], target[unequal]))
    losses = torch.stack(losses)
    if weights is None:
        weights = torch.ones_like(losses)
    total = posterior + (weights*losses).mean()
    return total, posterior, losses


def per_task_auc(score, feedback, exposed):
    metrics = []
    for task in range(feedback.shape[-1]):
        measurements = []
        for q in range(len(score)):
            pos = score[q][exposed[q] & (feedback[q, :, task] == 1)]
            neg = score[q][exposed[q] & (feedback[q, :, task] == 0)]
            if len(pos) and len(neg):
                measurements.append(((pos[:, None] > neg).float() +
                                     0.5*(pos[:, None] == neg).float()).mean())
        metrics.append(torch.stack(measurements).mean() if measurements else torch.tensor(0.5))
    return torch.stack(metrics)


def advantage_weights(previous_metric, current_metric, epsilon=1e-4):
    """原文 AE=上一版本 Metric / 当前版本 Metric；零指标做数值保护。"""
    return ((previous_metric + epsilon)/(current_metric + epsilon)).clamp(0.25, 4.0)


def train(model, train_data, adapt, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    weights = torch.ones(train_data.pxtr.shape[-1])
    previous = None
    for _ in range(epochs):
        for batch in request_batches(train_data):
            loss, _, _ = pairwise_losses(model(batch), batch, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            current = per_task_auc(model(adapt), adapt.feedback, adapt.exposed)
        if previous is not None:
            weights = advantage_weights(previous, current)
        previous = current
    return weights


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(1)
    train_data, adapt, test = split_train_adapt_test(make_data())
    model = EMER()
    with torch.no_grad():
        initial_score = model(test)
    head = cast(nn.Linear, model.head[0])
    initial = head.weight.detach().clone()
    weights = train(model, train_data, adapt)
    with torch.no_grad():
        total, posterior, prior = pairwise_losses(model(test), test, weights)
        before = pairwise_losses(initial_score, test, weights)[0]
        print("EMER 独立测试损失(均用最终AE权重，训练前/后):", round(float(before), 4),
              round(float(total), 4), "参数变化:", round(float((head.weight-initial).norm()), 4),
              "后验:", round(float(posterior), 4))
        print("各先验损失:", [round(float(x), 4) for x in prior], "AE 权重:", weights.tolist())
