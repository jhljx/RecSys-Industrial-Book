"""第25章 HarmonRank：目标间关系/门控/线性分支与置换多面体投影排序；arXiv:2601.02955。"""
from pathlib import Path
import math
import sys
from typing import cast

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_data, request_batches, split_train_adapt_test


class _PermutahedronProjection(torch.autograd.Function):
    """投影到 conv(permutations(1..N))，升序排序后用 PAVA 解等距回归。"""

    @staticmethod
    def forward(ctx, scores):
        order = scores.argsort(stable=True)
        sorted_scores = scores[order]
        n = len(scores)
        z = sorted_scores - torch.arange(1, n+1, device=scores.device, dtype=scores.dtype)
        blocks = []  # [起点, 终点, 和, 个数]；均值必须单调非降
        for i in range(n):
            blocks.append([i, i+1, float(z[i]), 1])
            while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
                right = blocks.pop()
                left = blocks[-1]
                left[1], left[2], left[3] = right[1], left[2]+right[2], left[3]+right[3]
        projected_z = torch.empty_like(z)
        groups = torch.empty(n, dtype=torch.long, device=scores.device)
        for group, (start, end, total, count) in enumerate(blocks):
            projected_z[start:end] = total/count
            groups[start:end] = group
        rank = torch.empty_like(scores)
        rank[order] = sorted_scores - projected_z
        ctx.save_for_backward(order, groups)
        return rank

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        order, groups = ctx.saved_tensors
        sorted_grad = grad_output[order]
        # 在固定 PAVA 分块的局部区域，Jacobian = I - 每块均值算子。
        block_sums = torch.zeros_like(sorted_grad).scatter_add(0, groups, sorted_grad)
        block_sizes = torch.zeros_like(sorted_grad).scatter_add(
            0, groups, torch.ones_like(sorted_grad))
        centered = sorted_grad - block_sums[groups] / block_sizes[groups]
        return torch.zeros_like(grad_output).scatter(0, order, centered)


def soft_rank(scores: torch.Tensor) -> torch.Tensor:
    return cast(torch.Tensor, _PermutahedronProjection.apply(scores))


def auc_rank_loss(score, labels, exposed):
    """以升序 rank 的正样本秩和最大化 AUC；仅使用已曝光标签。"""
    losses = []
    flat_score = score[exposed]
    flat_label = labels[exposed]
    rank = soft_rank(flat_score)
    for task in range(flat_label.shape[-1]):
        target = flat_label[:, task]
        positives = target.sum()
        negatives = len(target) - positives
        if positives > 0 and negatives > 0:
            # AUC=(正样本秩和 - n_pos(n_pos+1)/2)/(n_pos*n_neg)。
            losses.append(-((rank*target).sum() - positives*(positives+1)/2)/(positives*negatives))
    return torch.stack(losses).mean()


class HarmonRank(nn.Module):
    def __init__(self, tasks=3, buckets=16, dim=8):
        super().__init__()
        self.buckets = buckets
        self.embeddings = nn.ModuleList(nn.Embedding(buckets, dim) for _ in range(tasks))
        self.q_relation = nn.Linear(dim, dim)
        self.k_relation = nn.Linear(dim, dim)
        self.v_relation = nn.Linear(dim, dim)
        self.q_personal = nn.Linear(4+3, dim)
        self.k_personal = nn.Linear(dim, dim)
        self.v_personal = nn.Linear(dim, dim)
        self.aware_head = nn.Linear(dim, 1)
        self.gate = nn.Linear(tasks*dim, tasks)
        self.gated_head = nn.Linear(tasks*dim, 1)
        self.linear_head = nn.Linear(tasks*dim, 1)

    def forward(self, data, return_parts=False):
        n = data.pxtr.shape[1]
        # 原文 floor(s/B) 与「B 个等距桶」文字有量纲冲突；在 [0,1] 上用 floor(s*B)。
        bucket = (data.pxtr.clamp(0, 1-1e-7)*self.buckets).long()
        tokens = torch.stack([emb(bucket[..., k]) for k, emb in enumerate(self.embeddings)], -2)
        x = tokens.flatten(-2)
        q, k, v = self.q_relation(tokens), self.k_relation(tokens), self.v_relation(tokens)
        relation = torch.softmax(q @ k.transpose(-1, -2)/math.sqrt(k.shape[-1]), -1)
        aligned = relation @ v
        user = data.user_dense[:, None, :].expand(-1, n, -1)
        context = data.context[:, None, :].expand(-1, n, -1)
        personal = torch.cat((user, context), -1)
        query = self.q_personal(personal).unsqueeze(-2)
        key, value = self.k_personal(aligned), self.v_personal(aligned)
        personal_weights = torch.softmax(query @ key.transpose(-1, -2)/math.sqrt(key.shape[-1]), -1)
        s1 = self.aware_head((personal_weights @ value).squeeze(-2)).squeeze(-1)
        gate = torch.sigmoid(self.gate(x))
        s2 = self.gated_head((gate.unsqueeze(-1)*tokens).flatten(-2)).squeeze(-1)
        s3 = self.linear_head(x).squeeze(-1)
        if return_parts:
            return s1+s2+s3, (s1, s2, s3), relation, gate
        return s1+s2+s3


def train(model, train_data, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(epochs):
        for batch in request_batches(train_data):
            loss = auc_rank_loss(model(batch), batch.feedback, batch.exposed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(1)
    train_data, _, test = split_train_adapt_test(make_data())
    model = HarmonRank()
    with torch.no_grad():
        before = float(auc_rank_loss(model(test), test.feedback, test.exposed))
    initial = model.aware_head.weight.detach().clone()
    train(model, train_data)
    with torch.no_grad():
        score, parts, relation, gate = model(test, return_parts=True)
        print("HarmonRank 独立测试负AUC秩和(训练前/后):", round(before, 4),
              round(float(auc_rank_loss(score, test.feedback, test.exposed)), 4),
              "参数变化:", round(float((model.aware_head.weight-initial).norm()), 4))
        print("三个分支平均分:", [round(float(p.mean()), 4) for p in parts])
        print("目标关系矩阵首行:", relation[0, 0, 0].tolist(), "门控:", gate[0, 0].tolist())
