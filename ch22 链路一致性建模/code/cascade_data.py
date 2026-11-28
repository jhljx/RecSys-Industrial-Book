"""Deterministic request level cascade logs used by the chapter examples.

This is a teaching data generator, not a production feature or label service.  It
keeps the data flow explicit: a request has user/context fields, a retrieved
candidate set, downstream scores, exposure, feedback, and sample provenance.
"""
from dataclasses import dataclass

import torch


@dataclass
class RequestLog:
    user_id: int
    user_features: torch.Tensor
    context: torch.Tensor
    item_ids: torch.Tensor
    features: torch.Tensor
    retrieved: torch.Tensor
    sent_to_rank: torch.Tensor
    rank_selected: torch.Tensor
    prerank_score: torch.Tensor
    rank_score: torch.Tensor
    exposed: torch.Tensor
    clicked: torch.Tensor
    purchased: torch.Tensor
    cross_click: torch.Tensor
    cross_purchase: torch.Tensor
    watch_time: torch.Tensor
    bid: torch.Tensor
    attributes: torch.Tensor
    sample_source: tuple[str, ...]

    @property
    def rank_positions(self):
        order = torch.argsort(self.rank_score, descending=True)
        positions = torch.full_like(self.item_ids, self.item_ids.numel() + 1, dtype=torch.long)
        positions[order] = torch.arange(1, len(order) + 1)
        return positions


def make_dataset(num_requests=12, users=8, items=64, candidates=18,
                 retrieve_k=14, rank_k=7, expose_k=5, seed=22):
    """Create correlated request logs with all cascade stages and labels.

    Items outside ``retrieved`` remain in the item pool.  ``exposed`` is caused by
    the rank stage, while click and purchase are sampled from latent utility; this
    deliberately preserves the distinction between selection and behavior labels.
    """
    g = torch.Generator().manual_seed(seed)
    item_table = torch.randn(items, 8, generator=g)
    item_table[:, 4] = torch.sigmoid(item_table[:, 4])  # quality
    item_table[:, 5] = torch.sigmoid(item_table[:, 5])  # freshness
    bids = torch.rand(items, generator=g) * .8 + .2
    attributes = torch.arange(items) % 4
    logs = []
    for request_id in range(num_requests):
        uid = request_id % users
        user = torch.randn(8, generator=g)
        context = torch.randn(3, generator=g)
        base = item_table + .10 * torch.randn(item_table.shape, generator=g)
        utility = base @ user / 3 + .25 * base[:, 4] + .15 * context[0] * base[:, 5]
        retrieval_score = utility + .18 * torch.randn(items, generator=g)
        retrieved = torch.topk(retrieval_score, candidates).indices
        pre_score = utility[retrieved] + .10 * torch.randn(candidates, generator=g)
        pre_order = torch.argsort(pre_score, descending=True)
        rank_candidates = pre_order[:retrieve_k]
        sent_to_rank = torch.zeros(candidates, dtype=torch.bool)
        sent_to_rank[rank_candidates] = True
        rank_ids = retrieved[rank_candidates]
        rank_score = torch.full((candidates,), float('-inf'))
        rank_score[rank_candidates] = utility[rank_ids] + .12 * base[rank_ids, 6]
        rank_order = torch.argsort(rank_score[rank_candidates], descending=True)
        rank_selected = torch.zeros(candidates, dtype=torch.bool)
        rank_selected[rank_candidates[rank_order[:rank_k]]] = True
        exposed_local = rank_order[:rank_k][:expose_k]
        exposed = torch.zeros(candidates, dtype=torch.bool)
        exposed[rank_candidates[exposed_local]] = True
        logits = utility[retrieved] + .25 * context[1]
        click_prob = torch.sigmoid(logits - .35)
        clicked = exposed & (torch.rand(candidates, generator=g) < click_prob)
        purchase_prob = torch.sigmoid(logits - .9)
        purchased = clicked & (torch.rand(candidates, generator=g) < purchase_prob)
        # Independent observation channel for ASMOL; not inferred from non-exposure.
        cross_click = torch.rand(candidates, generator=g) < click_prob * .35
        cross_purchase = cross_click & (torch.rand(candidates, generator=g) < purchase_prob * .3)
        watch_time = exposed.float() * torch.relu(8 + 3 * logits + torch.randn(candidates, generator=g))
        source = tuple("exposed" if exposed[i] else
                       "rank_selected_unexposed" if rank_selected[i] else
                       "rank_candidate_unselected" if sent_to_rank[i] else "retrieval_tail"
                       for i in range(candidates))
        features = base[retrieved] + torch.randn((candidates, 8), generator=g) * .02
        logs.append(RequestLog(uid, user, context, retrieved, features, retrieved,
                               sent_to_rank, rank_selected, pre_score, rank_score,
                               exposed, clicked, purchased,
                               cross_click, cross_purchase,
                               watch_time, bids[retrieved], attributes[retrieved], source))
    return logs


def request_features(log, width=8):
    """Return fixed width item features, including request context in-place."""
    x = log.features[:, :width].clone()
    x[:, 0] += log.context[0]
    x[:, 1] += log.context[1]
    x[:, 2] += log.context[2]
    x[:, 3] += .1 * log.user_features[0]
    x[:, 4] += .1 * log.user_features[1]
    x[:, 5] += .1 * log.user_features[2]
    return x


def teacher_probabilities(log):
    """[MOCK] Downstream teacher probabilities from logged feedback and scores."""
    ctr = torch.sigmoid(log.rank_score - .2)
    cvr = torch.sigmoid(log.rank_score - .8)
    ctr = torch.where(log.clicked, torch.maximum(ctr, torch.full_like(ctr, .65)), ctr)
    cvr = torch.where(log.purchased, torch.maximum(cvr, torch.full_like(cvr, .55)), cvr)
    return ctr, cvr
