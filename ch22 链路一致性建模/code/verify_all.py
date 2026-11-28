"""Run every chapter 22 demo and verify distinctive method invariants."""
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'ch01_prerank_consistency'))
sys.path.insert(0, str(ROOT / 'ch02_industrial_practice'))

import asmol
import copr
import cs3
import ecmm
import grace
import hccp
import ranktower
import unirec
import cascade
from cascade_data import make_dataset


def check_mechanisms():
    checks = 0
    request = make_dataset(num_requests=2)[0]
    assert request.features.shape == (18, 8)
    assert request.sent_to_rank.sum().item() == 14
    assert request.rank_selected.sum().item() == 7
    assert request.exposed.sum().item() == 5
    assert torch.all(request.exposed <= request.sent_to_rank)
    assert torch.all(request.rank_selected <= request.sent_to_rank)
    assert torch.all(request.exposed <= request.rank_selected)
    assert torch.all(request.clicked <= request.exposed)
    assert torch.all(request.purchased <= request.clicked)
    assert torch.all(request.watch_time[~request.exposed] == 0)
    checks += 1
    order = hccp.rank_sequence([1, 2, 3, 4], [0, 1, 1, 0], [0, 0, 1, 0],
                               (2, 4), (1., 1.))
    assert order == [2, 1, 0, 3]
    checks += 1
    scores = torch.tensor([4., 3., 2.], requires_grad=True)
    good = hccp.listmle(scores, [0, 1, 2])
    bad = hccp.listmle(scores, [2, 1, 0])
    assert good < bad
    checks += 1
    user = torch.tensor([1., 0.]); item = torch.tensor([[1., 0.]])
    # Same negative in hard vs easy category must receive a different margin.
    hard = hccp.margin_infonce(user, item, item, item[:0])
    easy = hccp.margin_infonce(user, item, item[:0], item)
    assert not torch.isclose(hard, easy)
    checks += 1
    assert hccp.inbatch_negatives([[1, 2], [2, 3], [4]], 0) == [3, 4]
    checks += 1
    assert set(hccp.sample_retrieval_tail([1, 2, 3, 4], [1, 2], 2)) == {3, 4}
    assert set(hccp.sample_item_pool([1, 2, 3, 4], [1, 2], 2)) == {3, 4}
    checks += 1

    pxtr = torch.tensor([[.2, .8], [.5, .4], [.1, .9], [.9, .1]])
    weights = torch.tensor([2., .5])
    score = torch.tensor([.1, -.2, .4, .3], requires_grad=True)
    plus, minus = unirec.evidence(pxtr, weights)
    grouped = unirec.cppa_loss(score, plus, minus)
    direct = score.sum() * 0
    for i in range(len(score)):
        for j in range(i + 1, len(score)):
            for t in range(pxtr.shape[1]):
                sign = torch.sign(pxtr[i, t] - pxtr[j, t])
                if sign:
                    direct = direct + weights[t] * F.softplus(-sign * (score[i] - score[j]))
    assert torch.allclose(grouped, direct)
    checks += 1
    mask = unirec.capp_mask(plus, minus, 1)
    assert 0 < int(mask.sum()) <= len(score) and not torch.diag(mask).any()
    checks += 1
    attrs = torch.tensor([1., 2., 3., 4.])
    base = unirec.agrr_loss(score, plus, minus, attrs, buckets=2)
    assert torch.allclose(base, unirec.agrr_loss(score + 10, plus, minus, attrs, buckets=2))
    checks += 1
    model = unirec.UniRec()
    ids = torch.arange(4); pre = torch.rand(4, 2); ranking = torch.rand(4, 2)
    _, upstream = model.pre_score(ids, pre)
    model.rank_score(ids, pre, ranking, upstream).sum().backward()
    assert model.pre_out.weight.grad is None and model.pre_cross.weight.weight.grad is None
    assert model.item.weight.grad is not None
    checks += 1
    cross = unirec.CrossLayer(3)
    with torch.no_grad():
        cross.weight.weight.copy_(torch.tensor([[1., 2., 3.]]))
        cross.bias.zero_()
    x = torch.tensor([[1., 2., 3.]])
    assert torch.allclose(cross(x), x * (1 + 14))
    checks += 1
    with torch.no_grad():
        ids = torch.arange(4)
        pre, rank = torch.rand(4, 2), torch.rand(4, 2)
        upstream = torch.rand(4, 16)
        original = model.rank_score(ids, pre, rank, upstream)[0]
        rank[1] += 2
        changed = model.rank_score(ids, pre, rank, upstream)[0]
        assert not torch.allclose(original, changed), 'ranking attention must mix candidates'
    checks += 1

    exposure = torch.tensor([True, True, False])
    scores = torch.randn(3)
    tctr = torch.tensor([.6, .3, .9]); tcvr = torch.tensor([.1, .2, .9])
    original = asmol.selective_distill(scores, exposure, tctr, tcvr)
    tctr[-1], tcvr[-1] = 0, 0
    assert torch.allclose(original, asmol.selective_distill(scores, exposure, tctr, tcvr))
    checks += 1
    assert asmol.all_scenario_labels(torch.zeros(2), torch.zeros(2),
                                     torch.tensor([1, 0]))[1][0]
    checks += 1

    click, exp, imp = ecmm.ECMM()(torch.randn(3, 8))
    assert torch.allclose(exp + imp, torch.ones_like(exp)) and torch.all(click <= exp)
    checks += 1
    assert len(set(copr.chunk_sample(list(range(12)), size=3))) == 4
    checks += 1
    grace_model = grace.GRACE()
    assert len(grace_model(torch.randn(4, 6))[0]) == 4
    gx = torch.randn(4, 6)
    learned = torch.randn(4, 8)
    ctr = torch.tensor([1., 0., 0., 0.])
    positions = torch.tensor([1, 2, 3, 4])
    base = grace_model.loss(gx, ctr, ctr, positions, learned)
    learned[1:] += 100
    assert torch.allclose(base, grace_model.loss(gx, ctr, ctr, positions, learned))
    checks += 1
    # Availability metadata cannot make a non-clicked item a positive pair.
    all_available = torch.ones(4, dtype=torch.bool)
    assert torch.allclose(base, grace_model.loss(
        gx, ctr, ctr, positions, learned, all_available))
    checks += 1

    c = cs3.CS3()
    c.observe(0, 1, False, torch.ones(8))
    assert 0 not in c.cross_user and 0 in c.cascade_user
    c.observe(0, 1, True, torch.ones(8))
    assert 0 in c.cross_user and 1 in c.cross_item
    checks += 1
    torch.manual_seed(2)
    r = ranktower.RankTower()
    items = torch.arange(5)
    with torch.no_grad():
        assert torch.allclose(r(0, items),
                              r.score_with_precomputed(0, r.precompute_item_heads(items)))
    checks += 1
    assert ranktower.am_rankmax(torch.tensor([5., 0.]), torch.tensor([1., 0.])).item() == 0
    checks += 1
    yr, yf = cascade.make_targets([0, 1, 2], {1, 2}, {2})
    assert yr.tolist() == [0., 1., 1.] and yf.tolist() == [0., 0., 1.]
    checks += 1
    return checks


def main():
    n = check_mechanisms()
    scripts = sorted(ROOT.glob('ch*/*.py'))
    for file in scripts:
        module = f'{file.parent.name}.{file.stem}'
        result = subprocess.run([sys.executable, '-m', module], cwd=ROOT, capture_output=True,
                                text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f'{file.name} failed:\n{result.stderr}')
        print(file.name, result.stdout.strip().splitlines()[-1])
    print(f'{n} mechanism checks and {len(scripts)} runnable demos passed')


if __name__ == '__main__':
    main()
