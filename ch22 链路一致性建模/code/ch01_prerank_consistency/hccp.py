"""[MOCK] HCCP: five sample levels, reordered lists, and hybrid objectives.

Paper: https://arxiv.org/abs/2502.10284, methodology 3.2--3.3.
Synthetic request lists replace production stage logs and item servers.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def rank_sequence(ranks, exposed, clicked, bin_ends, rates, seed=7):
    """N2: nonuniformly keep downstream items; N1 exposure is always kept.

    Sort by r-(exposed+clicked)*N, stable within each feedback tier.
    """
    if len(bin_ends) != len(rates) or not all(0 <= x <= 1 for x in rates):
        raise ValueError("one valid sampling rate per rank bin is required")
    n = len(ranks)
    if any(len(v) != n for v in (exposed, clicked)):
        raise ValueError("request fields have different lengths")
    g = torch.Generator().manual_seed(seed)
    kept = []
    for j, r in enumerate(ranks):
        b = next((k for k, end in enumerate(bin_ends) if r <= end), None)
        if b is None:
            raise ValueError("rank outside sampling bins")
        if exposed[j] or torch.rand((), generator=g).item() < rates[b]:
            kept.append(j)
    return sorted(kept, key=lambda j: (ranks[j] - (int(exposed[j]) + int(clicked[j])) * n, ranks[j]))


def listmle(scores, order):
    """Plackett--Luce top-one likelihood over each remaining suffix."""
    z = scores[order]
    if not z.numel():
        return scores.sum() * 0
    return (torch.logcumsumexp(z.flip(0), 0).flip(0) - z).sum()


def local_order(global_order, exposed, task_labels, ranks):
    n = len(ranks)
    return sorted((j for j in global_order if exposed[j]),
                  key=lambda j: (ranks[j] - (1 + int(task_labels[j])) * n, ranks[j]))


def margin_infonce(user, positive, hard, easy, margin=.2, temperature=.7):
    """v(cos-m)/tau for positive and N2/N3; v*cos/tau for N4/N5."""
    if positive.ndim != 2 or not len(positive):
        raise ValueError("at least one exposed positive is required")
    if margin <= 0 or temperature <= 0:
        raise ValueError("margin and temperature must be positive")

    def score(items, use_margin):
        if not len(items):
            return user.new_empty(0)
        norms = user.norm() * items.norm(dim=-1)
        cosine = F.cosine_similarity(user.unsqueeze(0), items, dim=-1)
        return norms * (cosine - (margin if use_margin else 0)) / temperature

    hard_logits, easy_logits = score(hard, True), score(easy, False)
    negative = torch.cat((hard_logits, easy_logits))
    losses = []
    for logit in score(positive, True):
        losses.append(torch.logsumexp(torch.cat((logit.view(1), negative)), 0) - logit)
    return torch.stack(losses).sum()


def inbatch_negatives(request_items, request_index):
    """[MOCK] Other request lists, excluding IDs already in this request."""
    mine = set(request_items[request_index])
    return sorted({i for k, row in enumerate(request_items) if k != request_index
                   for i in row if i not in mine})


def sample_retrieval_tail(retrieved, sent_downstream, count, seed=11):
    """N3: uniform sample from retrieved items not passed to ranking."""
    available = sorted(set(retrieved) - set(sent_downstream))
    if count > len(available) or count < 0:
        raise ValueError("N3 sample size outside retrieval tail")
    g = torch.Generator().manual_seed(seed)
    return [available[i] for i in torch.randperm(len(available), generator=g)[:count].tolist()]


def sample_item_pool(pool, retrieved, count, seed=13):
    """N5: uniform sample from item space excluding current retrieved set."""
    available = sorted(set(pool) - set(retrieved))
    if count > len(available) or count < 0:
        raise ValueError("N5 sample size outside nonretrieved item pool")
    g = torch.Generator().manual_seed(seed)
    return [available[i] for i in torch.randperm(len(available), generator=g)[:count].tolist()]


class HCCP(nn.Module):
    """Small three-tower, two-task MMoE model; training objectives are central."""
    def __init__(self, users=3, items=32, dim=8):
        super().__init__()
        self.user_emb = nn.Embedding(users, dim)
        self.item_emb = nn.Embedding(items, dim)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(2))
        self.gates = nn.ModuleList(nn.Linear(dim, 2) for _ in range(2))
        self.cross = nn.Linear(dim * 2, 2)
        self.global_head = nn.Linear(dim * 2, 1)

    def forward(self, user_id, item_ids):
        u = self.user_emb(torch.tensor(user_id)).expand(len(item_ids), -1)
        v = self.item_emb(item_ids)
        experts = torch.stack([F.relu(e(u)) for e in self.experts], 1)
        task_u = torch.stack([torch.einsum('be,bed->bd',
                                          F.softmax(g(u), -1), experts)
                              for g in self.gates], 1)
        dot = (task_u * v.unsqueeze(1)).sum(-1) / math.sqrt(v.shape[-1])
        task_logits = dot + self.cross(torch.cat((u, v), -1))
        global_logits = self.global_head(torch.cat((u, v), -1)).squeeze(-1)
        return global_logits, task_logits, task_u[0]


def hybrid_loss(model, user_id, ids, ranks, exposed, task_labels,
                n3_ids, n4_ids, n5_ids, rates=(1., .6, .2)):
    """One request; N1/N2 in ids, N3/N4/N5 supplied as disjoint ID lists."""
    if len(ids) != len(ranks) or len(ids) != len(exposed) or task_labels.shape != (len(ids), 2):
        raise ValueError("misaligned request")
    order = rank_sequence(ranks, exposed, task_labels[:, 0].tolist(), (3, 6, len(ids)), rates)
    global_z, task_z, task_u = model(user_id, torch.tensor(ids))
    global_loss = listmle(global_z, order)
    terms = []
    n2 = [ids[j] for j in order if not exposed[j]]
    hard_ids = sorted(set(n2 + list(n3_ids)))
    easy_ids = sorted(set(n4_ids) | set(n5_ids))
    for t in range(2):
        exposed_idx = torch.tensor([j for j in order if exposed[j]], dtype=torch.long)
        cali = F.binary_cross_entropy_with_logits(task_z[exposed_idx, t],
                                                   task_labels[exposed_idx, t], reduction='sum')
        local = listmle(task_z[:, t], local_order(order, exposed, task_labels[:, t].tolist(), ranks))
        positives = [ids[j] for j in exposed_idx.tolist() if task_labels[j, t] > 0]
        contrast = (margin_infonce(task_u[0, t], model.item_emb(torch.tensor(positives)),
                                   model.item_emb(torch.tensor(hard_ids)),
                                   model.item_emb(torch.tensor(easy_ids))) if positives
                    else global_z.sum() * 0)
        terms.append(cali + .2 * local + .2 * contrast)
    return global_loss + sum(terms), dict(global_listmle=global_loss, hard=hard_ids, easy=easy_ids)


def demo():
    from train_examples import main
    main('hccp')


if __name__ == '__main__':
    demo()
