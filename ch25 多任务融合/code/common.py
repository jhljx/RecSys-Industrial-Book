"""[MOCK] 第25章共用的请求级多目标曝光数据；不复刻任何论文数据集。"""
from dataclasses import dataclass, fields
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FusionBatch:
    user_id: torch.Tensor          # [请求]
    item_id: torch.Tensor          # [请求, 候选]
    category: torch.Tensor         # [请求, 候选]
    item_dense: torch.Tensor       # [请求, 候选, 4]
    user_dense: torch.Tensor       # [请求, 4]
    context: torch.Tensor          # [请求, 3]
    pxtr: torch.Tensor             # [请求, 候选, 3]；上游模型固定输出
    exposed: torch.Tensor          # [请求, 候选]；未曝光无真实反馈
    feedback: torch.Tensor         # [请求, 候选, 3]
    relevance: torch.Tensor        # [请求, 候选]；购买 > 点击 > 曝光
    intent: torch.Tensor           # [请求, 类别×行为]；当前请求监督
    history_item: torch.Tensor     # [请求, 历史长度]
    history_category: torch.Tensor
    history_action: torch.Tensor
    session_intent: torch.Tensor   # [请求, 历史Session, 类别×行为]
    watchtime: torch.Tensor        # [请求, 候选]；用于 IPUT
    session_context: torch.Tensor  # [请求, 历史Session, 3]；真实历史上下文
    history_time: torch.Tensor     # [请求, 历史长度]；必须早于请求时间
    request_time: torch.Tensor     # [请求]
    position: torch.Tensor         # [请求, 候选]
    exposure_propensity: torch.Tensor

    def take(self, index):
        return FusionBatch(*(getattr(self, field.name)[index] for field in fields(self)))


class Upstream(nn.Module):
    """仅在请求时间窗以前的独立日志上拟合的基础目标预测器。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, user, item, category, context):
        x = torch.cat((user, item, context, F.one_hot(category, 5).float()), -1)
        return self.net(x).sigmoid()


@lru_cache(maxsize=8)
def _world(seed):
    g = torch.Generator().manual_seed(seed + 901)
    users = torch.randn(32, 4, generator=g)
    items = torch.randn(180, 4, generator=g)
    categories = torch.randint(5, (180,), generator=g)
    preference = users @ (torch.randn(4, 5, generator=g)*0.42)

    def probability(uid, iid, ctx):
        affinity = preference[uid, categories[iid]]
        logits = torch.stack((
            -0.2 + 0.65*affinity + 0.45*items[iid, 0] + 0.25*ctx[..., 0],
            -0.45 + 0.35*affinity + 0.5*items[iid, 1] - 0.2*ctx[..., 1],
            -0.8 + 0.3*affinity + 0.5*items[iid, 2] + 0.2*ctx[..., 2]), -1)
        return logits.sigmoid()

    uid = torch.randint(32, (2048,), generator=g)
    iid = torch.randint(180, (2048,), generator=g)
    ctx = torch.randn(2048, 3, generator=g)
    true = probability(uid, iid, ctx)
    labels = (torch.rand(true.shape, generator=g) < true).float()
    labels[:, 2] *= labels[:, 0]
    with torch.random.fork_rng():
        torch.manual_seed(seed+902)
        upstream = Upstream()
    opt = torch.optim.Adam(upstream.parameters(), lr=0.015)
    for _ in range(30):
        loss = F.binary_cross_entropy(upstream(users[uid], items[iid], categories[iid], ctx), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    upstream.eval()
    for p in upstream.parameters():
        p.requires_grad_(False)
    return users, items, categories, probability, upstream


def make_data(n_requests=256, n_candidates=8, seed=42):
    """按时间生成曝光与正反馈；每个请求只读取过去的历史。"""
    if not 6 <= n_candidates <= 90:
        raise ValueError("候选数须在 6 到 90 之间")
    users, items, categories, probability, upstream = _world(seed)
    g = torch.Generator().manual_seed(seed+1001)
    histories = [[] for _ in range(32)]
    for uid in range(32):
        for t in range(-12, 0):
            # 早期日志先采样反馈，只把实际发生的正反馈放入历史。
            for _ in range(100):
                iid = int(torch.randint(180, (1,), generator=g))
                ctx = torch.randn(3, generator=g)
                outcomes = torch.rand(3, generator=g) < probability(uid, iid, ctx)
                outcomes[2] &= outcomes[0]
                if outcomes.any():
                    action = int(torch.where(outcomes)[0][-1])
                    histories[uid].append((iid, action, ctx, t))
                    break
            else:
                raise RuntimeError("无法生成早期正反馈")
    rows = []
    for request in range(n_requests):
        uid = request % 32
        ctx = torch.randn(3, generator=g) + 0.15*users[uid, :3]
        pool = torch.randperm(180, generator=g)[:max(2*n_candidates, 12)]
        with torch.no_grad():
            upstream_scores = upstream(users[uid].expand(len(pool), -1), items[pool],
                                       categories[pool], ctx.expand(len(pool), -1))
        selected = []
        for task in range(3):
            for index in upstream_scores[:, task].topk(2).indices.tolist():
                if index not in selected:
                    selected.append(index)
        for index in upstream_scores.mean(-1).argsort(descending=True).tolist():
            if index not in selected:
                selected.append(index)
            if len(selected) >= n_candidates:
                break
        ids = pool[selected[:n_candidates]]
        pxtr = upstream_scores[selected[:n_candidates]].detach()
        true = probability(torch.full_like(ids, uid), ids, ctx.expand(len(ids), -1))
        pos = torch.arange(n_candidates)
        propensity = (0.9 - 0.55*pos/(n_candidates-1)).clamp(0.05, 0.95)
        propensity[0] = 1.0  # 保证每次请求至少有一个已曝光候选
        exposed = torch.rand(n_candidates, generator=g) < propensity
        label = ((torch.rand(true.shape, generator=g) < true) & exposed[:, None]).float()
        label[:, 2] *= label[:, 0]
        recent = histories[uid][-9:]
        hist_id = torch.tensor([e[0] for e in recent])
        hist_action = torch.tensor([e[1] for e in recent])
        hist_time = torch.tensor([e[3] for e in recent])
        session = F.one_hot(categories[hist_id]*3+hist_action, 15).float().reshape(3, 3, 15).mean(1)
        session_ctx = torch.stack([e[2] for e in recent]).reshape(3, 3, 3).mean(1)
        counts = torch.full((15,), 0.01)
        for action in range(3):
            counts.scatter_add_(0, categories[ids]*3+action, label[:, action])
        rows.append((uid, ids, categories[ids], items[ids], users[uid], ctx, pxtr,
                     exposed, label, label[:, 0]+label[:, 2], counts/counts.sum(),
                     hist_id, categories[hist_id], hist_action, session, 3+38*pxtr[:, 1],
                     session_ctx, hist_time, request, pos, propensity))
        for index in range(n_candidates):
            if exposed[index]:
                for action in range(3):
                    if label[index, action]:
                        histories[uid].append((int(ids[index]), action, ctx.clone(), request))
    columns = list(zip(*rows))
    return FusionBatch(*(torch.tensor(col) if k in (0, 18) else torch.stack(col)
                         for k, col in enumerate(columns)))


def split_data(data, train_fraction=0.75):
    cut = int(len(data.user_id)*train_fraction)
    return data.take(slice(None, cut)), data.take(slice(cut, None))


def split_train_adapt_test(data, train_fraction=0.6, adapt_fraction=0.2):
    train_end = int(len(data.user_id)*train_fraction)
    adapt_end = int(len(data.user_id)*(train_fraction+adapt_fraction))
    return data.take(slice(None, train_end)), data.take(slice(train_end, adapt_end)), data.take(slice(adapt_end, None))


def request_batches(data, size=8):
    for start in range(0, len(data.user_id), size):
        yield data.take(slice(start, start + size))


def normalized_ranks(scores):
    """每个请求、每个目标内排序；分数越高，名次越靠前。"""
    order = scores.argsort(dim=1, descending=True)
    rank = order.argsort(dim=1).float() + 1
    return rank / scores.shape[1]
