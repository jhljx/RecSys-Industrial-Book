"""第25章 IntEL：双粒度历史意图、列表注意力、物品级权重及 EA-MSE；SIGIR 2023。"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_data, request_batches, split_train_adapt_test


class IntEL(nn.Module):
    def __init__(self, n_items=180, n_categories=5, n_actions=3, tasks=3, dim=24):
        super().__init__()
        self.n_actions = n_actions
        self.intent_encoder = nn.Linear(n_categories*n_actions, dim)
        self.context_encoder = nn.Linear(3, dim)
        self.session_gru = nn.GRU(2*dim, dim, batch_first=True)
        self.item_gru = nn.GRU(dim, dim, batch_first=True)
        self.intent_head = nn.Linear(3*dim, n_categories*n_actions)
        self.intent_project = nn.Linear(n_categories*n_actions, dim)
        self.score_embed = nn.Linear(1, dim)
        self.task_embed = nn.Embedding(tasks, dim)
        self.category_embed = nn.Embedding(n_categories, dim)
        self.score_attention = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.category_attention = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.shared_query = nn.Linear(dim, dim)
        self.score_cross = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.category_cross = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.weight_head = nn.Linear(3*dim, tasks)

    def predict_intent(self, data):
        current_context = self.context_encoder(data.context)
        session_context = self.context_encoder(data.session_context)
        session_input = torch.cat((self.intent_encoder(data.session_intent), session_context), -1)
        _, session_final = self.session_gru(session_input)
        hist_index = data.history_category*self.n_actions + data.history_action
        hist_onehot = F.one_hot(hist_index, self.intent_head.out_features).float()
        _, item_final = self.item_gru(self.intent_encoder(hist_onehot))
        logits = self.intent_head(torch.cat((current_context, session_final[0], item_final[0]), -1))
        return logits.softmax(-1)

    def forward(self, data):
        intent = self.predict_intent(data)
        n = data.pxtr.shape[1]
        intent_rep = self.intent_project(intent)[:, None, :].expand(-1, n, -1)
        # 每个基础目标列表分别做候选间自注意力，再聚合目标表示。
        batch, candidates, tasks = data.pxtr.shape
        task_id = torch.arange(tasks, device=data.pxtr.device)
        per_list = self.score_embed(data.pxtr.permute(0, 2, 1).unsqueeze(-1))
        per_list = per_list + self.task_embed(task_id)[None, :, None, :]
        per_list = per_list.reshape(batch*tasks, candidates, -1)
        attended, _ = self.score_attention(per_list, per_list, per_list)
        score_list = (per_list+attended).reshape(batch, tasks, candidates, -1).mean(1)
        category_local = self.category_embed(data.category)
        category_list, _ = self.category_attention(category_local, category_local, category_local)
        category_list = category_local + category_list
        query = self.shared_query(intent_rep)
        score_cross, _ = self.score_cross(query, score_list, score_list)
        category_cross, _ = self.category_cross(query, category_list, category_list)
        # 原文重复的意图 Query 会使各物品交叉注意力输出完全相同；
        # 保留标准注意力残差里的局部表示，才可形成论文要求的物品级权重。
        score_cross = score_cross + score_list
        category_cross = category_cross + category_list
        weights = self.weight_head(torch.cat((score_cross, category_cross, intent_rep), -1)).softmax(-1)
        score = (weights*data.pxtr).sum(-1)
        return score, weights, intent


def intel_loss(output, data, alpha=0.05, gamma=0.2):
    score, weights, predicted_intent = output
    # EA 等式严格适用于 Pointwise MSE 与非负归一化权重；不把该项误用为 BPR 的精确分解。
    mse = F.mse_loss(score[data.exposed], (data.relevance/2)[data.exposed])
    ambiguity = (weights*(data.pxtr - score.unsqueeze(-1)).square()).sum(-1)[data.exposed].mean()
    intent_kl = F.kl_div(predicted_intent.clamp_min(1e-8).log(), data.intent, reduction="batchmean")
    return mse - alpha*ambiguity + gamma*intent_kl


def train(model, train_data, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(epochs):
        for batch in request_batches(train_data):
            loss = intel_loss(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(1)
    train_data, _, test = split_train_adapt_test(make_data())
    model = IntEL()
    with torch.no_grad():
        before = float(intel_loss(model(test), test))
    initial = model.weight_head.weight.detach().clone()
    train(model, train_data)
    with torch.no_grad():
        score, weights, intent = model(test)
        print("IntEL 独立测试损失(训练前/后):", round(before, 4),
              round(float(intel_loss((score, weights, intent), test)), 4),
              "参数变化:", round(float((model.weight_head.weight-initial).norm()), 4))
        print("首请求不同候选的权重:", weights[0, :2].tolist())
