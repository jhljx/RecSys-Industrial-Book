"""[MOCK] UniRec fusion agents and dual-axis preference optimization.

Current paper: https://arxiv.org/abs/2609.11052 (not its earlier v1).
Upstream pXTRs and feedback are synthetic; this does not train pXTR networks.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Linear(dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # One explicit DCN cross: x1 = x0 * (w^T x0) + b + x0.
        return x + x * self.weight(x) + self.bias


class UniRec(nn.Module):
    """Shared feature embedding, separate fusion towers, stop-gradient adapter."""
    def __init__(self, n_items=32, d=8, pre_tasks=2, rank_tasks=2):
        super().__init__()
        self.item = nn.Embedding(n_items, d)
        self.pre_pxtr = nn.Linear(pre_tasks, d)
        self.rank_pxtr = nn.Linear(rank_tasks, d)
        self.pre_cross = CrossLayer(2 * d)
        self.rank_cross = CrossLayer(3 * d)
        self.token_mix = nn.Linear(2, 2)
        self.feature_mix = nn.Linear(d, d)
        self.pre_out = nn.Linear(2 * d, 1)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.query = nn.Linear(d, d)
        self.attention_out = nn.Linear(d, 3 * d)
        self.adapter = nn.Linear(2 * d, d)
        self.rank_out = nn.Linear(4 * d, 1)
        self.d = d

    def pre_score(self, ids, pre_pxtr):
        """Online pre-ranking path: no ranking-only feature dependency."""
        tokens = torch.stack((self.item(ids), self.pre_pxtr(pre_pxtr)), 1)
        crossed = self.pre_cross(tokens.flatten(1))
        mixed = self.token_mix(tokens.transpose(1, 2)).transpose(1, 2)
        mixed = self.feature_mix(mixed).flatten(1)
        hidden = crossed + mixed
        return self.pre_out(hidden).squeeze(-1), hidden

    def rank_score(self, ids, pre_pxtr, rank_pxtr, upstream_hidden):
        """Online ranking path consumes an upstream feature; no pre model call."""
        tokens = torch.stack((self.item(ids), self.pre_pxtr(pre_pxtr),
                              self.rank_pxtr(rank_pxtr)), 1)
        crossed = self.rank_cross(tokens.flatten(1))
        # AutoInt-Lite operates over ranking candidates, not over the three
        # feature tokens of an individual item. K/V projections are shared.
        q, k, v = self.query(tokens), self.key(tokens), self.value(tokens)
        q, k, v = q.mean(1), k.mean(1), v.mean(1)
        attention = F.softmax(q @ k.T / self.d ** .5, -1) @ v
        interacted = tokens.flatten(1) + self.attention_out(attention)
        adapted = self.adapter(upstream_hidden.detach())
        return self.rank_out(torch.cat((crossed + interacted, adapted), -1)).squeeze(-1)


def evidence(pxtr, weights=None):
    """E+ and E- for every ordered candidate pair; ties contribute neither."""
    n, tasks = pxtr.shape
    if weights is None:
        weights = pxtr.new_ones(tasks)
    sign = torch.sign(pxtr[:, None, :] - pxtr[None, :, :])
    positive = ((sign > 0) * weights).sum(-1)
    negative = ((sign < 0) * weights).sum(-1)
    return positive, negative


def add_subset_evidence(whole, subset, extra):
    plus, minus = (x.clone() for x in whole)
    idx = subset[:, None], subset[None, :]
    plus[idx] += extra[0]
    minus[idx] += extra[1]
    return plus, minus


def capp_mask(plus, minus, k):
    """Keep each candidate's top-K confidence neighbors; deduplicate pairs."""
    n = plus.shape[0]
    if n < 2:
        return torch.zeros_like(plus, dtype=torch.bool)
    confidence = (plus - minus).abs().clone()
    confidence.fill_diagonal_(-1)
    neighbors = confidence.topk(min(k, n - 1), dim=1).indices
    directed = torch.zeros_like(confidence, dtype=torch.bool)
    directed.scatter_(1, neighbors, True)
    return torch.triu(directed | directed.T, diagonal=1)


def cppa_loss(scores, plus, minus, pair_mask=None):
    """Grouped Softplus, algebraically equal to the per-task objective."""
    if pair_mask is None:
        pair_mask = torch.triu(torch.ones_like(plus, dtype=torch.bool), diagonal=1)
    delta = scores[:, None] - scores[None, :]
    return (plus[pair_mask] * F.softplus(-delta[pair_mask]) +
            minus[pair_mask] * F.softplus(delta[pair_mask])).sum()


def pair_sign_loss(scores, desired):
    """desired is an external pXTR/feedback/teacher score, detached as label."""
    i, j = torch.triu_indices(len(scores), len(scores), 1)
    signs = torch.sign((desired[i] - desired[j]).detach())
    active = signs != 0
    return F.softplus(-signs[active] * (scores[i[active]] - scores[j[active]])).sum()


def agrr_loss(scores, plus, minus, attribute, buckets=2, min_count=2,
              sigma_min=.05, max_advantage=2., temperature=1.):
    """Group-relative reward, clipped advantage, KL(q || pi) within bins."""
    if len(scores) != len(attribute):
        raise ValueError("attribute and score lengths differ")
    reward = (plus - minus).sum(1) / (plus + minus + 1e-6).sum(1).clamp_min(1e-6)
    boundaries = torch.quantile(attribute.float(),
                                torch.linspace(0, 1, buckets + 1, device=attribute.device)[1:-1])
    group = torch.bucketize(attribute.contiguous().float(), boundaries)
    losses = []
    for g in range(buckets):
        idx = (group == g).nonzero().flatten()
        if len(idx) < min_count:
            continue
        r = reward[idx]
        advantage = ((r - r.mean()) / r.std(unbiased=False).clamp_min(sigma_min)).clamp(
            -max_advantage, max_advantage)
        q = F.softmax(advantage / temperature, 0).detach()
        log_pi = F.log_softmax(scores[idx], 0)
        losses.append((q * (q.log() - log_pi)).sum())
    return torch.stack(losses).mean() if losses else scores.sum() * 0


def train_step(model, ids, pre_pxtr, rank_pxtr, rank_ids, exposed_ids,
               feedback, attribute, k=2):
    """All indices are local to ids; ranking and exposed are nested subsets."""
    if not set(exposed_ids.tolist()) <= set(rank_ids.tolist()):
        raise ValueError("exposures must have passed the ranking stage")
    pre_score, pre_hidden = model.pre_score(ids, pre_pxtr)
    rank_score = model.rank_score(ids[rank_ids], pre_pxtr[rank_ids], rank_pxtr,
                                  pre_hidden[rank_ids])
    pre_evidence = add_subset_evidence(evidence(pre_pxtr), rank_ids, evidence(rank_pxtr))
    rank_evidence = evidence(rank_pxtr)
    pre_prior = cppa_loss(pre_score, *pre_evidence,
                          capp_mask(*pre_evidence, k))
    rank_prior = cppa_loss(rank_score, *rank_evidence,
                           capp_mask(*rank_evidence, k))
    local = {int(g): i for i, g in enumerate(rank_ids.tolist())}
    exposed_rank = torch.tensor([local[int(x)] for x in exposed_ids], dtype=torch.long)
    feedback_pre = pair_sign_loss(pre_score[exposed_ids], feedback)
    feedback_rank = pair_sign_loss(rank_score[exposed_rank], feedback)
    alignment = pair_sign_loss(pre_score[rank_ids], rank_score.detach())
    pre_reg = agrr_loss(pre_score, *pre_evidence, attribute)
    rank_reg = agrr_loss(rank_score, *rank_evidence, attribute[rank_ids])
    pre_loss = pre_prior + .2 * feedback_pre + .3 * alignment + .1 * pre_reg
    rank_loss = rank_prior + .2 * feedback_rank + .1 * rank_reg
    return pre_loss + rank_loss, dict(pre=pre_loss, rank=rank_loss, align=alignment,
                                      pre_pairs=int(capp_mask(*pre_evidence, k).sum()))


def demo():
    from train_examples import main
    main('unirec')


if __name__ == '__main__':
    demo()
