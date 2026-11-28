"""[MOCK] RankTower MHGN + bidirectional GCAN + MaxSim and hybrid loss.

Paper: https://arxiv.org/abs/2407.12385. Synthetic full-stage list data.
Only MHGN single-side heads can be precomputed: GCAN remains pair-dependent.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadGated(nn.Module):
    def __init__(self, n_entities, heads=3, dim=8):
        super().__init__()
        self.embedding = nn.Embedding(n_entities, dim)
        self.value = nn.Linear(dim, heads * dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                  nn.Linear(dim, heads * dim))
        self.heads, self.dim = heads, dim

    def forward(self, ids):
        x = self.embedding(ids)
        # The gating branch must not backpropagate to the input embedding.
        return (self.value(x) * torch.sigmoid(self.gate(x.detach()))).reshape(
            -1, self.heads, self.dim)


class GAU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query, other):
        q, k, v = F.silu(self.q(query)), F.silu(self.k(other)), F.silu(self.v(other))
        a = F.softmax(q @ k.transpose(1, 2) / q.shape[-1] ** .5, -1)
        return self.out(torch.sigmoid(self.gate(query)) * (a @ v))


class RankTower(nn.Module):
    def __init__(self, users=4, items=20, heads=3, dim=8):
        super().__init__()
        self.user = MultiHeadGated(users, heads, dim)
        self.item = MultiHeadGated(items, heads, dim)
        self.u_from_i, self.i_from_u = GAU(dim), GAU(dim)
        self.u_norm, self.i_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def interact(self, u_heads, i_heads):
        u = self.u_norm(u_heads + self.u_from_i(u_heads, i_heads))
        i = self.i_norm(i_heads + self.i_from_u(i_heads, u_heads))
        cosine = F.normalize(u, dim=-1) @ F.normalize(i, dim=-1).transpose(1, 2)
        return cosine.max(-1).values.sum(-1) / self.log_temperature.exp()

    def forward(self, uid, item_ids):
        ids = torch.full_like(item_ids, uid)
        return self.interact(self.user(ids), self.item(item_ids))

    def precompute_item_heads(self, item_ids):
        """[MOCK] Store only MHGN output; cross attention is still online."""
        return self.item(item_ids).detach()

    def score_with_precomputed(self, uid, item_heads):
        ids = torch.full((len(item_heads),), uid, dtype=torch.long)
        return self.interact(self.user(ids), item_heads)


def softsort(values, temperature=.5):
    sorted_values = values.sort(descending=True).values
    return F.softmax(-(sorted_values[:, None] - values[None, :]).abs() / temperature, -1)


def am_rankmax(logits, labels):
    """AM-Rankmax with log1p stabilization at zero violations [MOCK]."""
    terms = []
    for j in range(len(logits)):
        if labels[j] <= 0:
            continue
        lower = (labels < labels[j]).nonzero().flatten()
        if len(lower):
            margin = .3 * (labels[lower] == 0).float() + .2 * (
                labels[j] - labels[lower]).abs()
            violation = F.relu(logits[lower] - logits[j] + margin)
            # The printed log(sum ReLU) is undefined once all constraints are
            # satisfied. log1p keeps the same monotone penalty and a zero floor.
            terms.append(torch.log1p(violation.sum()))
    return torch.stack(terms).sum() if terms else logits.sum() * 0


def hybrid_loss(logits, hard_labels, teacher, exposed, candidates,
                distill_weight=.2, sorting_weight=.1, rankmax_weight=.2):
    """Distill N1; SoftSort N1+candidate; AM-Rankmax all stage samples."""
    if not torch.all(exposed <= candidates):
        raise ValueError("exposed items must belong to candidate set")
    distill = -(teacher[exposed].detach() * F.log_softmax(logits[exposed], 0)).sum()
    included = candidates.nonzero().flatten()
    sort_target = softsort(hard_labels[included]).detach()
    sort_prediction = softsort(logits[included])
    sorting = -(sort_target * sort_prediction.clamp_min(1e-9).log()).sum()
    rankmax = am_rankmax(logits, hard_labels)
    return distill_weight * distill + sorting_weight * sorting + rankmax_weight * rankmax


def demo():
    from train_examples import main
    main('ranktower')


if __name__ == '__main__':
    demo()
