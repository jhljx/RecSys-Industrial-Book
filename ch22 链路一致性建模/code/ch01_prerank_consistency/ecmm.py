"""[MOCK] ECMM cross-domain targets, feature gate and sparse subnetwork routing."""
import torch
from torch import nn
from torch.nn import functional as F


class ECMM(nn.Module):
    def __init__(self, features=8, hidden=12, paths=3):
        super().__init__()
        self.feature_gate = nn.Linear(features, features, bias=False)
        self.paths = nn.ModuleList(nn.Linear(features, hidden) for _ in range(paths))
        self.log_alpha = nn.Parameter(torch.zeros(3, paths))
        self.towers = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(3))

    def routing(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.log_alpha) / .67)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * 1.2 - .1).clamp(0, 1)

    def forward(self, x):
        selected = x * torch.sigmoid(self.feature_gate(x))
        paths = torch.stack([F.relu(p(selected)) for p in self.paths], 1)
        mask = self.routing(self.training)
        logits = torch.cat([self.towers[t]((paths * mask[t, None, :, None]).sum(1))
                            for t in range(3)], 1)
        # t1+t2+t3=1; losses on ETCTR=t1, ETR=t1+t2, ITR=t3.
        t1, t2, t3 = F.softmax(logits, -1).unbind(-1)
        return t1, t1 + t2, t3

    def loss(self, x, exposed, clicked, sparse_weight=.01):
        if torch.any(clicked > exposed):
            raise ValueError("click must imply exposure")
        click, exposure, implication = self(x)
        bce = (F.binary_cross_entropy(click, clicked) +
               F.binary_cross_entropy(exposure, exposed) +
               F.binary_cross_entropy(implication, 1 - exposed))
        # P(z>0) for Hard-Concrete gates, not the sampled mask itself.
        expected_l0 = torch.sigmoid(self.log_alpha - .67 * torch.log(torch.tensor(.1 / 1.1))).sum()
        return bce + sparse_weight * expected_l0


def demo():
    from train_examples import main
    main('ecmm')


if __name__ == '__main__':
    demo()
