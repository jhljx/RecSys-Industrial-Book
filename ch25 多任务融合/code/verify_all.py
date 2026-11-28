"""第25章方法专属检查：断言足以区分常见的错误或退化实现。"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import cast

from common import make_data, normalized_ranks, split_train_adapt_test
from ch01_manual.fusion import rank_fusion, value_fusion
from ch02_search.search import objective
from ch03_learned_ltr.learned_ltr import DynamicFusion, ranking_loss
from ch04_intel.intel import IntEL, intel_loss
from ch05_umre.umre import UMRE, clenshaw_curtis, update_reward_weights, user_uauc
from ch05_umre.umre import MonotoneIntegral
from ch06_emer.emer import EMER, advantage_weights, pairwise_losses, prior_targets
from ch07_harmonrank.harmonrank import HarmonRank, auc_rank_loss, soft_rank


def has_nonzero_grad(parameter: torch.Tensor) -> bool:
    grad = parameter.grad
    return grad is not None and bool(grad.abs().sum() > 0)


def check_data():
    data = make_data(16)
    assert data.pxtr.shape == (16, 8, 3)
    assert all(len(torch.unique(row)) == len(row) for row in data.item_id)
    assert (~data.exposed).any() and data.exposed.any()
    assert torch.equal(data.feedback[~data.exposed], torch.zeros_like(data.feedback[~data.exposed]))
    assert torch.all(data.history_time < data.request_time[:, None])
    assert data.pxtr.grad_fn is None
    assert torch.all((data.exposure_propensity > 0) & (data.exposure_propensity <= 1))
    assert torch.all(data.exposure_propensity[:, 0] > data.exposure_propensity[:, -1])
    train, adapt, test = split_train_adapt_test(make_data())
    assert train.request_time.max() < adapt.request_time.min() < test.request_time.min()
    for user in data.user_id.unique():
        rows = data.user_id == user
        assert torch.equal(data.user_dense[rows], data.user_dense[rows][0].expand(int(rows.sum()), -1))
    # 同一用户的历史在其正反馈后更新；历史 Session 必须携带原事件上下文。
    full = make_data(64)
    assert torch.any(full.history_item[:32] != full.history_item[32:64])
    assert not torch.allclose(full.session_context[0, -1], full.context[0])
    assert torch.allclose(data.intent.sum(-1), torch.ones(16))


def check_manual_search():
    data = make_data(8)
    weight = [0.5, 0.3, 0.2]
    assert torch.allclose(value_fusion(data.pxtr, weight), (data.pxtr*torch.tensor(weight)).sum(-1))
    assert torch.allclose(rank_fusion(data.pxtr, weight), rank_fusion(data.pxtr.square(), weight))
    assert torch.allclose(normalized_ranks(data.pxtr), normalized_ranks(data.pxtr.square()))
    assert not torch.allclose(objective(data.pxtr, data.relevance, data.exposed, [1, 0, 0]),
                              objective(data.pxtr, data.relevance, data.exposed, [0, 1, 0]))
    try:
        value_fusion(data.pxtr, mode="multiply", alpha=[-1]*3, beta=[0]*3, gamma=[0.5]*3)
    except ValueError:
        pass
    else:
        raise AssertionError("乘法非法底数未拦截")


def check_learned_ltr():
    data = make_data(8)
    model = DynamicFusion()
    score, weights = model(data)
    assert score.shape == data.relevance.shape
    assert torch.allclose(weights.sum(-1), torch.ones_like(score))
    assert float(weights.var(dim=1).mean()) > 1e-8
    ranking_loss(score, data.relevance, data.exposed, "pairwise").backward()
    assert has_nonzero_grad(cast(nn.Linear, model.gate[0]).weight)
    for kind in ("pointwise", "pairwise", "listwise"):
        assert torch.isfinite(ranking_loss(score, data.relevance, data.exposed, kind))


def check_intel():
    data = make_data(8)
    model = IntEL()
    score, weights, intent = model(data)
    assert torch.allclose(weights.sum(-1), torch.ones_like(score), atol=1e-6)
    assert float((weights[:, 0] - weights[:, 1]).abs().max()) > 1e-5, "意图 Query 重复不能使物品权重退化为全局权重"
    assert torch.allclose(intent.sum(-1), torch.ones(len(intent)), atol=1e-6)
    # EA Pointwise 恒等式：加权基础 MSE - 模型间模糊性 = 融合 MSE。
    base = (weights*(data.pxtr - data.relevance.unsqueeze(-1)/2).square()).sum(-1)
    ambiguity = (weights*(data.pxtr - score.unsqueeze(-1)).square()).sum(-1)
    assert torch.allclose(base-ambiguity, (score-data.relevance/2).square(), atol=1e-6)
    intel_loss((score, weights, intent), data).backward()
    assert has_nonzero_grad(model.context_encoder.weight)
    assert has_nonzero_grad(cast(torch.Tensor, model.session_gru.weight_ih_l0))
    assert has_nonzero_grad(cast(torch.Tensor, model.item_gru.weight_ih_l0))
    attention_weight = model.score_attention.in_proj_weight
    assert attention_weight is not None and has_nonzero_grad(attention_weight)


def check_umre():
    nodes, weights = clenshaw_curtis(32)
    assert torch.allclose(weights.sum(), torch.tensor(2.), atol=1e-6)
    assert torch.allclose((weights*nodes.square()).sum(), torch.tensor(2/3), atol=1e-5)
    data = make_data(8)
    model = UMRE()
    context = torch.cat((data.user_dense[:, None].expand_as(data.item_dense), data.item_dense), -1)
    low = model.transforms[0](torch.full((8, 8), 0.2), context)
    high = model.transforms[0](torch.full((8, 8), 0.8), context)
    assert torch.all(high > low), "同一上下文内积分变换必须严格单调"
    score, transformed, fusion_weights = model(data)
    assert torch.allclose(score, (transformed*fusion_weights).sum(-1), atol=1e-6)
    score.mean().backward()
    transform = cast(MonotoneIntegral, model.transforms[0])
    assert has_nonzero_grad(transform.integrand.input.weight)
    next_weight = update_reward_weights(torch.tensor([0.5, 0.5]), torch.tensor([0.6, 0.6]), torch.tensor([0.5, 0.7]))
    assert next_weight[0] > next_weight[1] and torch.allclose(next_weight.sum(), torch.tensor(1.))
    # 两个请求属于同一用户时，应跨请求汇总正负例，而非先平均请求 AUC。
    synthetic_score = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    synthetic_label = torch.tensor([[[1.], [0.]], [[0.], [1.]]])
    assert torch.allclose(user_uauc(synthetic_score, synthetic_label,
                                   torch.ones(2, 2, dtype=torch.bool), torch.tensor([7, 7])),
                          torch.tensor([0.75]))


def check_emer():
    data = make_data(8)
    model = EMER()
    score = model(data)
    assert score.shape == (8, 8)
    prior = prior_targets(data)
    assert torch.allclose(prior[..., 2], data.pxtr[..., 2]/data.watchtime)
    total, posterior, prior_losses = pairwise_losses(score, data)
    assert torch.isfinite(total) and len(prior_losses) == 3
    # 更改未曝光反馈不能影响后验；先验仍覆盖未曝光候选。
    modified = data.take(slice(None))
    modified.feedback = data.feedback.clone()
    modified.feedback[~data.exposed] = 1
    changed_posterior = pairwise_losses(score, modified)[1]
    assert torch.allclose(posterior, changed_posterior)
    score2 = score.detach().clone().requires_grad_()
    pairwise_losses(score2, data)[0].backward()
    assert score2.grad is not None and score2.grad[~data.exposed].abs().sum() > 0, "先验必须监督未曝光候选"
    assert advantage_weights(torch.tensor([0.8]), torch.tensor([0.4]))[0] > 1
    # 列表 Transformer 必须允许其他候选影响某候选得分。
    altered = data.take(slice(None))
    altered.item_dense = data.item_dense.clone()
    altered.item_dense[:, 1] += 3
    assert not torch.allclose(score[:, 0], model(altered)[:, 0])


def check_harmonrank():
    assert torch.allclose(soft_rank(torch.tensor([0., 0.])), torch.tensor([1.5, 1.5]))
    assert torch.allclose(soft_rank(torch.tensor([0., 10.])), torch.tensor([1., 2.]))
    assert torch.allclose(soft_rank(torch.tensor([0., 0.4])), torch.tensor([1.3, 1.7]), atol=1e-6)
    x = torch.tensor([0., 0.3, 0.4], requires_grad=True)
    (soft_rank(x)*torch.tensor([1., 0., 0.])).sum().backward()
    assert has_nonzero_grad(x)
    # 未排序输入且同时含单元素块与合并块，检查 backward 的局部 Jacobian。
    projection_input = torch.tensor([0.2, 3., 0.], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(soft_rank, (projection_input,))
    hard = soft_rank(torch.tensor([0.1, 10., 3.]))
    auc = (hard[1] - 1)/2
    assert torch.allclose(auc, torch.tensor(1.))
    data = make_data(8)
    model = HarmonRank()
    score, parts, relation, gate = model(data, return_parts=True)
    assert torch.allclose(score, parts[0] + parts[1] + parts[2])
    assert relation.shape[-2:] == (3, 3) and torch.allclose(relation.sum(-1), torch.ones_like(gate))
    assert torch.all((gate > 0) & (gate < 1))
    auc_rank_loss(score, data.feedback, data.exposed).backward()
    assert has_nonzero_grad(model.q_relation.weight)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_num_threads(1)
    checks = (check_data, check_manual_search, check_learned_ltr, check_intel,
              check_umre, check_emer, check_harmonrank)
    for check in checks:
        check()
        print(check.__name__, "通过")
    print(f"{len(checks)} 组方法专属检查通过")
