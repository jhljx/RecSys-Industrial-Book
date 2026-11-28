"""第25章 UMRE：逐目标正值积分 UMNN、GRU 历史、双交叉注意力和动态奖励；arXiv:2508.07613。"""
from pathlib import Path
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_data, request_batches, split_train_adapt_test


def clenshaw_curtis(n=32):
    """Clenshaw–Curtis 在 [-1,1] 上的积分节点/权重，n=32 对应论文附录。"""
    N = n - 1
    theta = torch.arange(n, dtype=torch.float64)*math.pi/N
    nodes = theta.cos()
    weights = torch.zeros(n, dtype=torch.float64)
    interior = theta[1:-1]
    v = torch.ones(N-1, dtype=torch.float64)
    if N % 2 == 0:
        weights[0] = weights[-1] = 1/(N*N-1)
        for k in range(1, N//2):
            v -= 2*torch.cos(2*k*interior)/(4*k*k-1)
        v -= torch.cos(N*interior)/(N*N-1)
    else:
        weights[0] = weights[-1] = 1/(N*N)
        for k in range(1, (N-1)//2 + 1):
            v -= 2*torch.cos(2*k*interior)/(4*k*k-1)
    weights[1:-1] = 2*v/N
    return nodes.float(), weights.float()


class PositiveIntegrand(nn.Module):
    """三层带残差的 MLP；积分位置用一对正弦/余弦特征编码。"""

    def __init__(self, context_dim=8, hidden=64):
        super().__init__()
        self.input = nn.Linear(context_dim+3, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))
        self.output = nn.Linear(hidden, 1)

    def forward(self, t, context):
        position = torch.stack((torch.sin(2*math.pi*t), torch.cos(2*math.pi*t)), -1)
        h = F.relu(self.input(torch.cat((t.unsqueeze(-1), position, context), -1)))
        for layer in self.layers:
            h = F.relu(h + layer(h))
        return self.output(h).squeeze(-1)


class MonotoneIntegral(nn.Module):
    nodes: torch.Tensor
    weights: torch.Tensor

    def __init__(self, context_dim=8, hidden=64, nodes=32):
        super().__init__()
        self.integrand = PositiveIntegrand(context_dim, hidden)
        self.bias = nn.Parameter(torch.empty(()).uniform_(-0.5, 0.5))
        x, w = clenshaw_curtis(nodes)
        self.register_buffer("nodes", x)
        self.register_buffer("weights", w)

    def positive_integrand(self, t, context):
        return F.elu(self.integrand(t, context)) + 1 + 1e-5

    def forward(self, p, context):
        # [-1,1] -> [0,p] 的变换连同 p/2 雅可比；p 在 [0,1]。
        t = p.unsqueeze(-1)*(self.nodes + 1)/2
        h = context.unsqueeze(-2).expand(*t.shape, context.shape[-1])
        area = (self.positive_integrand(t, h)*self.weights).sum(-1)*p/2
        return area + self.bias


class UMRE(nn.Module):
    def __init__(self, n_items=180, n_categories=5, n_actions=3, tasks=3, dim=24):
        super().__init__()
        self.transforms = nn.ModuleList(MonotoneIntegral() for _ in range(tasks))
        self.item_history_embed = nn.Embedding(n_items, dim)
        self.category_embed = nn.Embedding(n_categories, dim)
        self.action_embed = nn.Embedding(n_actions, dim)
        self.history_gru = nn.GRU(3*dim, dim, batch_first=True)
        self.task_embed = nn.Embedding(tasks, dim)
        self.score_embed = nn.Linear(1, dim)
        self.score_attention = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.category_attention = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.weight_head = nn.Linear(3*dim, tasks)

    def forward(self, data):
        history = torch.cat((self.item_history_embed(data.history_item),
                             self.category_embed(data.history_category),
                             self.action_embed(data.history_action)), -1)
        _, user = self.history_gru(history)
        user = user[0]
        context = torch.cat((data.user_dense[:, None, :].expand_as(data.item_dense), data.item_dense), -1)
        transformed = torch.stack([module(data.pxtr[..., k], context)
                                   for k, module in enumerate(self.transforms)], -1)
        shape = transformed.shape
        task_ids = torch.arange(shape[-1], device=transformed.device)
        score_tokens = self.score_embed(transformed.unsqueeze(-1)) + self.task_embed(task_ids)
        # Key/Value 含当前类别和历史类别，避免单 Key 注意力退化为恒等映射。
        hist_cat = self.category_embed(data.history_category)[:, None].expand(-1, shape[1], -1, -1)
        current_cat = self.category_embed(data.category).unsqueeze(2)
        category_tokens = torch.cat((current_cat, hist_cat), 2)
        query = user[:, None, None, :].expand(-1, shape[1], 1, -1)
        flat = shape[0]*shape[1]
        score_context, _ = self.score_attention(query.reshape(flat, 1, -1), score_tokens.reshape(flat, shape[-1], -1),
                                                  score_tokens.reshape(flat, shape[-1], -1))
        category_context, _ = self.category_attention(query.reshape(flat, 1, -1), category_tokens.reshape(flat, -1, user.shape[-1]),
                                                        category_tokens.reshape(flat, -1, user.shape[-1]))
        user_rep = user[:, None, :].expand(-1, shape[1], -1)
        weights = self.weight_head(torch.cat((user_rep, score_context.reshape(*shape[:2], -1),
                                              category_context.reshape(*shape[:2], -1)), -1))
        return (weights*transformed).sum(-1), transformed, weights


def update_reward_weights(weights, previous_uauc, current_uauc, step=0.4, low=0.05, high=0.8):
    """按第25章的 ΔUAUC=本轮-上轮 更新；下降目标增权。"""
    delta = current_uauc - previous_uauc
    updated = (weights - step*delta).clamp(low, high)
    return updated / updated.sum()


def user_uauc(score, labels, exposed, user_id):
    """按用户汇总全部已曝光样本，再对有效用户的 AUC 取均值。"""
    values = []
    for task in range(labels.shape[-1]):
        auc = []
        for user in user_id.unique():
            selected = (user_id == user)[:, None] & exposed
            pos = score[selected & (labels[:, :, task] == 1)]
            neg = score[selected & (labels[:, :, task] == 0)]
            if len(pos) and len(neg):
                auc.append(((pos[:, None] > neg).float() + 0.5*(pos[:, None] == neg).float()).mean())
        values.append(torch.stack(auc).mean() if auc else score.new_tensor(0.5))
    return torch.stack(values)


def train(model, train_data, adapt, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    reward_weights = torch.tensor([0.45, 0.30, 0.25])
    previous = None
    for epoch in range(epochs):
        for batch in request_batches(train_data):
            score, _, _ = model(batch)
            reward = (batch.feedback*reward_weights).sum(-1)
            loss = F.mse_loss(score[batch.exposed], reward[batch.exposed])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            current = user_uauc(model(adapt)[0], adapt.feedback, adapt.exposed, adapt.user_id)
        if epoch >= 2 and previous is not None:
            reward_weights = update_reward_weights(reward_weights, previous, current)
        previous = current
    return reward_weights


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(1)
    train_data, adapt, test = split_train_adapt_test(make_data())
    model = UMRE()
    with torch.no_grad():
        initial_score = model(test)[0]
    initial = model.weight_head.weight.detach().clone()
    reward_weights = train(model, train_data, adapt)
    with torch.no_grad():
        score, transformed, weights = model(test)
        reward = (test.feedback*reward_weights).sum(-1)
        before = F.mse_loss(initial_score[test.exposed], reward[test.exposed])
        print("UMRE 独立测试 MSE(均用最终奖励权重，训练前/后):", round(float(before), 4),
              round(float(F.mse_loss(score[test.exposed], reward[test.exposed])), 4),
              "参数变化:", round(float((model.weight_head.weight-initial).norm()), 4))
        print("UMRE 独立测试 UAUC:", user_uauc(score, test.feedback, test.exposed, test.user_id).tolist())
        print("奖励权重:", reward_weights.tolist(), "首候选融合权重:", weights[0, 0].tolist())
