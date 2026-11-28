"""[MOCK] GRACE: feedback heads, multi-Top-K ranking and item contrastive loss."""
import torch
from torch import nn
from torch.nn import functional as F


class GRACE(nn.Module):
    """Small shared/task expert [MOCK] PLE stand-in; objectives follow the chapter."""
    def __init__(self, features=6, dim=8, cutoffs=(2, 4)):
        super().__init__()
        self.shared = nn.Linear(features, dim)
        self.task_experts = nn.ModuleList(nn.Linear(features, dim) for _ in range(2 + len(cutoffs)))
        self.gates = nn.ModuleList(nn.Linear(features, 2) for _ in self.task_experts)
        self.heads = nn.ModuleList(nn.Linear(dim, 1) for _ in self.task_experts)
        self.item_projection = nn.Linear(features, dim)
        self.cutoffs = cutoffs

    def forward(self, x):
        shared = F.relu(self.shared(x))
        outputs = []
        for expert, gate, head in zip(self.task_experts, self.gates, self.heads):
            private = F.relu(expert(x))
            w = F.softmax(gate(x), -1)
            outputs.append(head(w[:, :1] * shared + w[:, 1:] * private).squeeze(-1))
        return outputs, self.item_projection(x)

    def loss(self, x, ctr, cvr, downstream_positions, pretrained_items,
             pretrained_positive_mask=None, temperature=.8, exposure_mask=None):
        logits, item_repr = self(x)
        observed = (exposure_mask.bool() if exposure_mask is not None
                    else torch.ones_like(ctr, dtype=torch.bool))
        feedback = (F.binary_cross_entropy_with_logits(logits[0][observed], ctr[observed]) +
                    F.binary_cross_entropy_with_logits(logits[1][observed], cvr[observed]))
        ranking = sum(F.binary_cross_entropy_with_logits(logits[i + 2],
                           (downstream_positions <= k).float()) for i, k in enumerate(self.cutoffs))
        # Only positive instances with a pre-trained embedding enter the
        # contrastive numerator; all batch items remain denominator negatives.
        eligible = ctr.bool() & observed
        if pretrained_positive_mask is not None:
            eligible = eligible & pretrained_positive_mask.bool()
        positive = (item_repr[eligible] * pretrained_items[eligible].detach()).sum(-1) / temperature
        negatives = item_repr[eligible] @ item_repr.T / temperature
        contrast = ((torch.logsumexp(negatives, -1) - positive).mean() if eligible.any()
                    else item_repr.sum() * 0)
        return feedback + .2 * ranking + .1 * contrast


def demo():
    from train_examples import main
    main('grace')


if __name__ == '__main__':
    demo()
