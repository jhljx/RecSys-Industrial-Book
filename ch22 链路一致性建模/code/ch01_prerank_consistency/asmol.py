"""[MOCK] ASMOL all-scenario targets, multi-positive ranking and selective distillation."""
import torch
from torch import nn
from torch.nn import functional as F


def all_scenario_labels(exposed, cross_click, cross_purchase):
    purchase = cross_purchase.bool()
    click = cross_click.bool() | purchase
    adaptive_exposure = exposed.bool() | click
    return adaptive_exposure, click, purchase


def multi_positive_loss(scores, positives):
    """Every positive competes only against negatives, never other positives."""
    positives = positives.bool()
    if not positives.any():
        return scores.sum() * 0
    if (~positives).any():
        return (torch.logaddexp(scores[positives],
                                torch.logsumexp(scores[~positives], 0)) - scores[positives]).sum()
    return scores.sum() * 0


def selective_distill(scores, exposure, teacher_ctr, teacher_cvr):
    """Teacher is only trusted on actual exposure; CTCVR=CTR*conditional CVR."""
    s = scores[exposure.bool()]
    if not len(s):
        return scores.sum() * 0
    log_probs = F.log_softmax(s, 0)
    ctr = teacher_ctr[exposure.bool()].detach()
    ctcvr = ctr * teacher_cvr[exposure.bool()].detach()
    return -(ctr * log_probs).sum() - (ctcvr * log_probs).sum()


def ash(pre_top, rank_top):
    if not rank_top:
        raise ValueError("rank Top-K cannot be empty")
    return len(set(pre_top) & set(rank_top)) / len(set(rank_top))


class ASMOL(nn.Module):
    def __init__(self, features=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 12), nn.ReLU(), nn.Linear(12, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def loss(self, x, exposed, cross_click, cross_purchase, teacher_ctr, teacher_cvr):
        ael, ascl, aspl = all_scenario_labels(exposed, cross_click, cross_purchase)
        scores = self(x)
        return (multi_positive_loss(scores, ael) + multi_positive_loss(scores, ascl)
                + multi_positive_loss(scores, aspl)
                + .2 * selective_distill(scores, exposed, teacher_ctr, teacher_cvr))


def demo():
    from train_examples import main
    main('asmol')


if __name__ == '__main__':
    demo()
