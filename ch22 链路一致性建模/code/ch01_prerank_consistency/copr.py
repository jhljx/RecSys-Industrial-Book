"""[MOCK] COPR: random chunk representatives, ECPM alignment and delta-NDCG."""
import random

import torch
from torch import nn
from torch.nn import functional as F


def chunk_sample(downstream_order, size=10, seed=2):
    if size < 1:
        raise ValueError("chunk size must be positive")
    rng = random.Random(seed)
    return [rng.choice(downstream_order[k:k + size])
            for k in range(0, len(downstream_order), size)]


class COPR(nn.Module):
    def __init__(self, features=6):
        super().__init__()
        self.ctr = nn.Sequential(nn.Linear(features, 8), nn.ReLU(), nn.Linear(8, 1))
        self.relax = nn.Sequential(nn.Linear(features, 8), nn.ReLU(), nn.Linear(8, 1))
        nn.init.zeros_(self.relax[-1].weight)
        nn.init.ones_(self.relax[-1].bias)  # alpha begins near the neutral value 1

    def forward(self, x, bid):
        ctr = torch.sigmoid(self.ctr(x).squeeze(-1))
        alpha = F.relu(self.relax(x).squeeze(-1)) + 1e-6
        ecpm = ctr * alpha * bid
        return ctr, alpha, ecpm


def ndcg_swap_weight(high, low, count):
    gain_hi, gain_lo = 2 ** (count - high) - 1, 2 ** (count - low) - 1
    ideal = sum((2 ** (count - i) - 1) / torch.log(torch.tensor(float(i + 1))).item()
                for i in range(1, count + 1))
    return ((gain_hi - gain_lo) / ideal *
            (1 / torch.log(torch.tensor(float(high + 1))).item() -
             1 / torch.log(torch.tensor(float(low + 1))).item()))


def alignment_loss(ecpm):
    """Input is one representative per chunk in downstream descending order."""
    loss = ecpm.sum() * 0
    for i in range(len(ecpm)):
        for j in range(i + 1, len(ecpm)):
            ratio = ecpm[i] / ecpm[j].clamp_min(1e-6)
            loss = loss + ndcg_swap_weight(i + 1, j + 1, len(ecpm)) * F.softplus(-(ratio - 1))
    return loss


def demo():
    from train_examples import main
    main('copr')


if __name__ == '__main__':
    demo()
