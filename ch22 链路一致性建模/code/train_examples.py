"""Request-level training and held-out validation for chapter 22 examples."""
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ch01_prerank_consistency'))
sys.path.insert(0, str(ROOT / 'ch02_industrial_practice'))

from cascade_data import make_dataset, request_features, teacher_probabilities
import asmol, copr, cs3, ecmm, grace, hccp, ranktower, unirec, cascade


def _batch_loss(name, model, log, logs, training):
    x = request_features(log)
    exposed = log.exposed
    if name == 'ecmm':
        return model.loss(x, exposed.float(), log.clicked.float())
    if name == 'asmol':
        ctr, cvr = teacher_probabilities(log)
        ctr = torch.where(log.sent_to_rank, ctr, torch.zeros_like(ctr))
        cvr = torch.where(log.sent_to_rank, cvr, torch.zeros_like(cvr))
        return model.loss(x[:, :6], exposed, log.cross_click, log.cross_purchase,
                          ctr, cvr)
    if name == 'copr':
        ctr, _, ecpm = model(x[:, :6], log.bid)
        observed = exposed
        click_loss = F.binary_cross_entropy(ctr[observed], log.clicked[observed].float())
        rank_ids = log.sent_to_rank.nonzero().flatten()
        rank_order = rank_ids[torch.argsort(log.rank_score[rank_ids], descending=True)]
        selected_global = copr.chunk_sample(rank_order.tolist(), size=3,
                                             seed=log.user_id + 4)
        selected = torch.tensor(selected_global, dtype=torch.long)
        selected_ecpm = ecpm[selected]
        return (click_loss + .2 * copr.alignment_loss(selected_ecpm) +
                .01 * (torch.maximum(model(x[:, :6], log.bid)[1][selected],
                                     1 / model(x[:, :6], log.bid)[1][selected]) - 1).sum())
    if name == 'grace':
        ctr_teacher, _ = teacher_probabilities(log)
        pretrained = torch.cat((x[:, :6], torch.zeros(len(x), 2)), dim=1)
        return model.loss(x[:, :6], log.clicked.float(), log.purchased.float(),
                          log.rank_positions, F.normalize(pretrained, dim=-1),
                          pretrained_positive_mask=log.sent_to_rank,
                          exposure_mask=exposed)
    if name == 'ranktower':
        ids = log.item_ids
        logits = model(log.user_id, ids)
        labels = log.clicked.float() + log.purchased.float()
        teacher = torch.zeros_like(log.rank_score)
        teacher[log.sent_to_rank] = torch.sigmoid(log.rank_score[log.sent_to_rank])
        return ranktower.hybrid_loss(logits, labels, teacher, exposed,
                                     log.sent_to_rank)
    if name == 'hccp':
        rank_ids = log.sent_to_rank.nonzero().flatten()
        ids = log.item_ids[rank_ids].tolist()
        order = torch.argsort(log.rank_score[rank_ids], descending=True)
        ranks = [0] * len(ids)
        for pos, local in enumerate(order.tolist(), 1):
            ranks[local] = pos
        exp = log.exposed[rank_ids].tolist()
        labels = torch.stack((log.clicked[rank_ids].float(),
                              log.purchased[rank_ids].float()), dim=1)
        n3 = log.item_ids[~log.sent_to_rank].tolist()
        request_index = next(k for k, row in enumerate(logs) if row is log)
        n4 = hccp.inbatch_negatives([row.item_ids.tolist() for row in logs], request_index)
        n5 = hccp.sample_item_pool(list(range(64)), log.item_ids.tolist(), 4,
                                   seed=log.user_id + 12)
        loss, _ = hccp.hybrid_loss(model, log.user_id, ids, ranks, exp,
                                   labels, n3, n4, n5,
                                   rates=(1., .7, .35))
        return loss
    if name == 'cs3':
        losses = []
        hidden = torch.tanh(x.mean(0))
        for i in exposed.nonzero().flatten().tolist():
            logit = model(log.user_id, int(log.item_ids[i]))
            losses.append(F.binary_cross_entropy_with_logits(
                logit, log.clicked[i].float()))
            if training:
                model.observe(log.user_id, int(log.item_ids[i]),
                              bool(log.clicked[i]), hidden)
        return torch.stack(losses).mean()
    if name == 'unirec':
        ids = log.item_ids
        pre = torch.stack((torch.sigmoid(log.prerank_score),
                           torch.sigmoid(x[:, 4])), dim=1)
        rank_ids = log.sent_to_rank.nonzero().flatten()
        rank = torch.stack((torch.sigmoid(log.rank_score[rank_ids]),
                            x[rank_ids, 5]), dim=1)
        exposed_ids = exposed.nonzero().flatten()
        feedback = log.clicked[exposed_ids].float() + 2 * log.purchased[exposed_ids].float()
        loss, _ = unirec.train_step(model, ids, pre, rank, rank_ids,
                                    exposed_ids, feedback, log.attributes, k=3)
        return loss
    if name == 'cascade':
        rank_target = log.rank_selected.float()
        final_target = exposed.float()
        pos = log.rank_positions.clamp_max(len(log.item_ids))
        rank_logits, final_logits = model(log.user_id, log.item_ids)
        return (cascade.weighted_focal(rank_logits, rank_target, pos) +
                cascade.weighted_focal(final_logits, final_target, pos))
    raise ValueError(name)


def _new_model(name):
    return {
        'ecmm': ecmm.ECMM(features=8),
        'asmol': asmol.ASMOL(features=6),
        'copr': copr.COPR(features=6),
        'grace': grace.GRACE(features=6, dim=8, cutoffs=(3, 5)),
        'ranktower': ranktower.RankTower(users=8, items=64),
        'hccp': hccp.HCCP(users=8, items=64),
        'cs3': cs3.CS3(users=8, items=64),
        'unirec': unirec.UniRec(n_items=64),
        'cascade': cascade.CascadeTwinTower(users=8, items=64),
    }[name]


def train_method(name, epochs=3):
    torch.manual_seed(2026)
    logs = make_dataset(num_requests=16, seed=22)
    train, valid = logs[:12], logs[12:]
    model = _new_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    train_loss = 0.
    for _ in range(epochs):
        model.train()
        total = 0.
        for log in train:
            optimizer.zero_grad()
            loss = _batch_loss(name, model, log, train, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())
        train_loss = total / len(train)
    model.eval()
    with torch.no_grad():
        val_loss = sum(float(_batch_loss(name, model, log, valid, False))
                       for log in valid) / len(valid)
    return train_loss, val_loss


def main(name):
    train_loss, val_loss = train_method(name)
    print(f'{name.upper()} requests train=12 valid=4 epochs=3 '
          f'train_objective={train_loss:.4f} valid_objective={val_loss:.4f}')
