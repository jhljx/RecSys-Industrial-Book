"""[MOCK] Rank/Final cascade targets, weighted focal loss and recall reuse.

This module illustrates chapter engineering choices, not an external service.
"""
import torch
from torch import nn
from torch.nn import functional as F


def make_targets(candidate_ids, rank_top, final_top):
    return torch.tensor([int(x in rank_top) for x in candidate_ids], dtype=torch.float), torch.tensor(
        [int(x in final_top) for x in candidate_ids], dtype=torch.float)


def weighted_focal(logits, labels, positions, gamma=2., alpha=.5, positive_scale=1.):
    """Positive order weight applies to loss, never to predicted probability."""
    p = torch.sigmoid(logits)
    p_t = torch.where(labels.bool(), p, 1 - p)
    alpha_t = torch.where(labels.bool(), alpha, 1 - alpha)
    weight = torch.where(labels.bool(), positive_scale / positions.float().clamp_min(1), 1.)
    return (weight * alpha_t * (1 - p_t).pow(gamma) * F.softplus(-torch.where(
        labels.bool(), logits, -logits))).mean()


class CascadeTwinTower(nn.Module):
    def __init__(self, users=4, items=24, dim=8):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        self.rank_head = nn.Linear(dim, dim)
        self.final_head = nn.Linear(dim, dim)

    def user_vector(self, uid):
        return self.user(torch.tensor(uid))

    def item_vectors(self, item_ids):
        return self.item(item_ids)

    def forward(self, uid, item_ids):
        u = self.user_vector(uid)
        v = self.item_vectors(item_ids)
        return v @ self.rank_head(u), v @ self.final_head(u)

    def ann_like_recall(self, uid, indexed_items, indexed_vectors, top_k):
        """Exact dot-product search stands in for ANN; vectors must match model version."""
        if len(indexed_items) != len(indexed_vectors):
            raise ValueError("item IDs and index vectors differ in length")
        scores = indexed_vectors @ self.rank_head(self.user_vector(uid))
        selected = scores.topk(top_k).indices
        return [(int(indexed_items[k]), float(scores[k])) for k in selected]


def demo():
    from train_examples import main
    main('cascade')


if __name__ == '__main__':
    demo()
