"""[MOCK] CS3 CAS/CTS/CMS online capability synergy for two towers.

Dictionary caches stand in for PS and Embedding Server; no network requests.
The same entity version must be used on both towers in real serving.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CAS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.base = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        first = F.relu(self.base(x))
        importance = torch.sigmoid(self.gate(first))
        return F.relu(self.base(x * 2 * importance))


class CS3(nn.Module):
    def __init__(self, users=4, items=12, dim=8, cross_decay=.8, cascade_decay=.9):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        self.user_cas = CAS(dim)
        self.item_cas = CAS(dim)
        self.user_project = nn.Linear(dim * 3, dim)
        self.item_project = nn.Linear(dim * 3, dim)
        self.cross_decay, self.cascade_decay = cross_decay, cascade_decay
        # [MOCK] Local caches: keyed by entity ID, not trained by gradients.
        self.cross_user, self.cross_item = {}, {}
        self.cascade_user, self.cascade_item = {}, {}
        self.dim = dim

    def _get(self, cache, key):
        return cache.get(int(key), torch.zeros(self.dim)).detach()

    def representations(self, uid, iid):
        ux = self.user_cas(self.user(torch.tensor(uid)))
        ix = self.item_cas(self.item(torch.tensor(iid)))
        u = self.user_project(torch.cat((ux, self._get(self.cross_user, uid),
                                         self._get(self.cascade_user, uid))))
        i = self.item_project(torch.cat((ix, self._get(self.cross_item, iid),
                                         self._get(self.cascade_item, iid))))
        return u, i

    def forward(self, uid, iid):
        u, i = self.representations(uid, iid)
        return u.dot(i)

    @staticmethod
    def _ema(cache, key, new, decay):
        old = cache.get(int(key), torch.zeros_like(new))
        cache[int(key)] = decay * old + (1 - decay) * new.detach()

    def observe(self, uid, iid, positive, cascade_hidden):
        """CTS positive only; CMS every logged event, including negatives."""
        if cascade_hidden.shape != (self.dim,):
            raise ValueError("cascade intermediate representation has wrong dimension")
        u, i = self.representations(uid, iid)
        if positive:
            self._ema(self.cross_user, uid, i, self.cross_decay)
            self._ema(self.cross_item, iid, u, self.cross_decay)
        self._ema(self.cascade_user, uid, cascade_hidden, self.cascade_decay)
        self._ema(self.cascade_item, iid, cascade_hidden, self.cascade_decay)


def demo():
    from train_examples import main
    main('cs3')


if __name__ == '__main__':
    demo()
