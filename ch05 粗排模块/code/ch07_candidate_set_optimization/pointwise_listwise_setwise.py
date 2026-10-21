"""
Point-wise / List-wise / Set-wise 粗排建模演进对比(第 5 章 §5.7 粗排候选集集合优化)

对应正文概念
------------
正文在 §5.7 末尾给出粗排建模从「物品」走向「集合」的演进路线:

1. Point-wise:对样本 (u, i) 用 L_point = CrossEntropy(y, ŷ);
   隐含假设是「每个样本之间都是彼此独立的,因此可以分开进行 loss 计算与建模」。
   但真实场景中「一个内容进入候选集后会影响其他内容的曝光机会」。
2. List-wise:候选集 L = {i_1, i_2, ..., i_n},优化目标从 ∏_i P(i|u) 变为 P(L|u),
   正文列出的典型方法是 ListNet、ListMLE、LambdaRank、Soft-NDCG。
3. Set-wise:候选集本质上是集合而不是序列 —— {i_1,i_2,i_3} 与 {i_3,i_2,i_1} 是同一个
   候选集,所以目标进一步变为 P(Value|Set),并要求模型满足置换不变性:
       f({x_1, x_2, ..., x_n}) = f({x_{π(1)}, x_{π(2)}, ..., x_{π(n)}})
   正文点名的方法是 Deep Sets 与 Set Transformer(细节留到粗排模型一章)。
4. 粗排的评价指标是集合级的:Recall@K = |Relevant ∩ CandidateSet| / |Relevant|,
   以及多样性、聚集度等。

本实现做了什么
--------------
1. [MOCK] 合成一批请求,每个请求有 n 个候选。真值刻意带集合级耦合:
       r_i    = σ(w · x_i)                        单体相关性
       disc_i = 1 / (1 + ρ (m_{c_i} - 1))          同簇挤占折扣(m_c = 该簇在本请求的条数)
       p_i    = r_i · disc_i,   y_i ~ Bernoulli(p_i)
   也就是说标签依赖「同一个集合里还有谁」,这正是 Point-wise 独立性假设失效的地方。
2. 用同一份数据训练 4 个模型并横评:
     A. Point-wise MLP + BCE          (正文的 L_point)
     B. ListNet(top-one 概率交叉熵)  —— List-wise 损失,打分函数仍是逐点的
     C. ListMLE(Plackett-Luce 似然)  —— List-wise 损失
     D. Set-wise Deep Sets(φ/ρ 结构,逐点特征拼接集合池化)+ BCE
3. 指标:样本级 AUC、列表内 NDCG@5(对 p_i 真值排序)、Recall@5、Top-5 真实价值之和、
   Top-5 里的同簇重复条数(聚集度)。
4. 数值验证 Set-wise 的置换不变性:对同一集合做随机排列,打分向量逐元素一致;
   并打印「复制一条内容进入集合」时两种模型打分的变化 —— Point-wise 完全不变,
   Set-wise 会自动给重复内容降权。

关键公式
--------
    L_point  = -Σ_i [ y_i log ŷ_i + (1-y_i) log(1-ŷ_i) ]
    ListNet  : P_s(i) = exp(s_i)/Σ_j exp(s_j),  L = -Σ_i P_y(i) log P_s(i)
    ListMLE  : L = -Σ_{k=1}^{n} log [ exp(s_{π(k)}) / Σ_{j≥k} exp(s_{π(j)}) ]
    Deep Sets: f(S) = ρ( Σ_{x∈S} φ(x) ),  score_i = ψ([ φ(x_i), pool(S) ])
    NDCG@K   = DCG@K / IDCG@K,  DCG@K = Σ_{k=1}^{K} p_{(k)} / log2(k+1)
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn

N_TRAIN = 1600          # [MOCK] 训练请求数
N_VALID = 400           # [MOCK] 验证请求数
N_CAND = 20             # 每个请求的候选数 |L|
D_DENSE = 8             # 稠密特征维度
N_CLUSTER = 5           # 内容簇数(同簇 = 高度相似内容)
RHO = 1.2               # 同簇挤占强度 ρ
TOPK = 5                # 评估用的截断长度
EPOCHS = 14
LR = 0.02


def make_data(rng: np.random.Generator, n_req: int):
    """[MOCK] 生成请求级数据。返回 (x, cluster_onehot, p_true, y)。"""
    x = rng.normal(0.0, 1.0, size=(n_req, N_CAND, D_DENSE)).astype(np.float32)
    cid = rng.integers(0, N_CLUSTER, size=(n_req, N_CAND))
    w = rng.normal(0.0, 0.8, size=(D_DENSE,)).astype(np.float32)
    r = 1.0 / (1.0 + np.exp(-(x @ w - 0.4)))                 # 单体相关性 σ(w·x)
    onehot = np.zeros((n_req, N_CAND, N_CLUSTER), dtype=np.float32)
    rows = np.arange(n_req)[:, None]
    onehot[rows, np.arange(N_CAND)[None, :], cid] = 1.0
    m = onehot.sum(axis=1)                                   # 每簇在本请求的条数
    m_i = m[rows, cid]                                       # 每个候选所属簇的条数
    disc = 1.0 / (1.0 + RHO * (m_i - 1.0))                   # 同簇挤占折扣
    p = (r * disc).astype(np.float32)
    y = (rng.random(p.shape) < p).astype(np.float32)
    return x, onehot, p, y


class PointModel(nn.Module):
    """A. Point-wise 打分器:score_i = MLP(x_i),与集合里其他候选无关。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_DENSE + N_CLUSTER, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class SetModel(nn.Module):
    """D. Set-wise Deep Sets:score_i = ψ([φ(x_i), mean_j φ(x_j), 同簇计数])。

    池化用 mean(对称函数),因此 f 对输入顺序严格不敏感,满足
    f({x_1,...,x_n}) = f({x_{π(1)},...,x_{π(n)}})。
    """

    def __init__(self) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(D_DENSE + N_CLUSTER, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.psi = nn.Sequential(
            nn.Linear(32 + 32 + N_CLUSTER, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.phi(feat)                                   # φ(x_i)
        pool = h.mean(dim=-2, keepdim=True).expand_as(h)     # Σ/|S| 对称池化
        # 集合级统计量:本集合内各簇的条数(置换不变)
        onehot = feat[..., D_DENSE:]
        cnt = onehot.sum(dim=-2, keepdim=True).expand(*onehot.shape)
        return self.psi(torch.cat([h, pool, cnt], dim=-1)).squeeze(-1)


def listnet_loss(s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ListNet:两个 top-one 概率分布的交叉熵。"""
    p_y = torch.softmax(y * 3.0, dim=-1)                     # 标签侧 top-one 概率
    log_p_s = torch.log_softmax(s, dim=-1)
    return -(p_y * log_p_s).sum(dim=-1).mean()


def listmle_loss(s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ListMLE:按标签给出的排列 π 计算 Plackett-Luce 负对数似然。"""
    order = torch.argsort(y + 1e-3 * torch.rand_like(y), dim=-1, descending=True)
    s_sorted = torch.gather(s, -1, order)
    # logcumsumexp 从尾部往前累积,得到 log Σ_{j≥k} exp(s_{π(j)})
    tail = torch.logcumsumexp(s_sorted.flip(-1), dim=-1).flip(-1)
    return (tail - s_sorted).sum(dim=-1).mean()


def ndcg_at_k(score: np.ndarray, p_true: np.ndarray, k: int) -> float:
    idx = np.argsort(-score, axis=-1)[:, :k]
    gain = np.take_along_axis(p_true, idx, axis=-1)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gain * disc).sum(axis=-1)
    ideal = np.sort(p_true, axis=-1)[:, ::-1][:, :k]
    idcg = (ideal * disc).sum(axis=-1)
    return float(np.mean(dcg / np.maximum(idcg, 1e-9)))


def recall_at_k(score: np.ndarray, p_true: np.ndarray, k: int) -> float:
    """Relevant = 真值 p 最高的 k 个,CandidateSet = 模型 top-k。"""
    pred = np.argsort(-score, axis=-1)[:, :k]
    rel = np.argsort(-p_true, axis=-1)[:, :k]
    hit = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(pred, rel)]
    return float(np.mean(hit)) / k


def topk_value(score: np.ndarray, p_true: np.ndarray, k: int) -> float:
    idx = np.argsort(-score, axis=-1)[:, :k]
    return float(np.take_along_axis(p_true, idx, axis=-1).sum(axis=-1).mean())


def topk_dup(score: np.ndarray, onehot: np.ndarray, k: int) -> float:
    """Top-K 内的同簇重复条数:Σ_c max(m_c - 1, 0),衡量聚集度。"""
    idx = np.argsort(-score, axis=-1)[:, :k]
    sel = np.take_along_axis(onehot, idx[:, :, None].repeat(onehot.shape[-1], -1), 1)
    cnt = sel.sum(axis=1)
    return float(np.maximum(cnt - 1.0, 0.0).sum(axis=-1).mean())


def auc(score: np.ndarray, y: np.ndarray) -> float:
    s, lab = score.ravel(), y.ravel()
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = lab.sum()
    n_neg = len(lab) - n_pos
    return float((ranks[lab > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train(model: nn.Module, feat: torch.Tensor, y: torch.Tensor, loss_name: str):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        for i in range(0, feat.shape[0], 200):
            xb, yb = feat[i:i + 200], y[i:i + 200]
            s = model(xb)
            if loss_name == "bce":
                loss = bce(s, yb)
            elif loss_name == "listnet":
                loss = listnet_loss(s, yb)
            else:
                loss = listmle_loss(s, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    xa, oa, pa, ya = make_data(rng, N_TRAIN + N_VALID)
    feat_all = np.concatenate([xa, oa], axis=-1)
    ftr = torch.from_numpy(feat_all[:N_TRAIN])
    ytr = torch.from_numpy(ya[:N_TRAIN])
    fva = torch.from_numpy(feat_all[N_TRAIN:])
    p_va, y_va, o_va = pa[N_TRAIN:], ya[N_TRAIN:], oa[N_TRAIN:]

    print("=" * 108)
    print("[MOCK] 数据生成:标签带集合级耦合  p_i = σ(w·x_i) / (1 + ρ(m_c - 1)),  ρ = "
          f"{RHO},  |L| = {N_CAND}")
    print("=" * 108)
    m_all = (oa[:, :, :].sum(axis=1))
    rows = np.arange(oa.shape[0])[:, None]
    cid = oa.argmax(axis=-1)
    m_i = m_all[rows, cid]
    print(f"{'同簇条数 m_c':>14}{'样本占比':>12}{'平均折扣 1/(1+ρ(m-1))':>24}"
          f"{'平均真值 p':>14}{'平均点击率 y':>14}")
    for m in range(1, 8):
        mask = m_i[N_TRAIN:] == m
        if mask.sum() < 20:
            continue
        print(f"{m:>14}{mask.mean():>12.4f}{1.0 / (1.0 + RHO * (m - 1)):>24.4f}"
              f"{p_va[mask].mean():>14.4f}{y_va[mask].mean():>14.4f}")
    print("同一个簇在集合里出现得越多,每条内容的真实价值被摊薄得越厉害 —— "
          "Point-wise 的独立性假设在此失效。")

    configs = [
        ("A Point-wise + BCE", PointModel, "bce", "否"),
        ("B ListNet(List-wise)", PointModel, "listnet", "否"),
        ("C ListMLE(List-wise)", PointModel, "listmle", "否"),
        ("D Set-wise Deep Sets", SetModel, "bce", "是"),
    ]
    res = {}
    models = {}
    for name, cls, loss_name, invariant in configs:
        torch.manual_seed(42)
        model = train(cls(), ftr, ytr, loss_name)
        model.eval()
        with torch.no_grad():
            s = model(fva).numpy()
        models[name] = model
        res[name] = {
            "置换不变": invariant,
            "AUC": auc(s, y_va),
            f"NDCG@{TOPK}": ndcg_at_k(s, p_va, TOPK),
            f"Recall@{TOPK}": recall_at_k(s, p_va, TOPK),
            f"Top{TOPK}真实价值": topk_value(s, p_va, TOPK),
            f"Top{TOPK}同簇重复": topk_dup(s, o_va, TOPK),
        }

    print("\n" + "=" * 108)
    print(f"四种建模方式横评(验证集 {N_VALID} 个请求,Relevant = 真值 p 最高的 "
          f"{TOPK} 条)")
    print("=" * 108)
    cols = ["置换不变", "AUC", f"NDCG@{TOPK}", f"Recall@{TOPK}",
            f"Top{TOPK}真实价值", f"Top{TOPK}同簇重复"]
    print(f"{'模型':>24}" + "".join(f"{c:>16}" for c in cols))
    for name, m in res.items():
        cells = "".join(
            f"{m[c]:>16}" if c == "置换不变" else f"{m[c]:>16.4f}" for c in cols)
        print(f"{name:>24}{cells}")
    ora = np.argsort(-p_va, axis=-1)[:, :TOPK]
    best_val = float(np.take_along_axis(p_va, ora, axis=-1).sum(axis=-1).mean())
    print(f"{'(真值上界 Oracle)':>24}{'-':>16}{'-':>16}{1.0:>16.4f}{1.0:>16.4f}"
          f"{best_val:>16.4f}{topk_dup(p_va, o_va, TOPK):>16.4f}")

    a = res["A Point-wise + BCE"]
    d = res["D Set-wise Deep Sets"]
    print(f"\nPoint-wise 的 Top{TOPK} 真实价值 {a[f'Top{TOPK}真实价值']:.4f}、同簇重复 "
          f"{a[f'Top{TOPK}同簇重复']:.4f};Set-wise 为 {d[f'Top{TOPK}真实价值']:.4f} / "
          f"{d[f'Top{TOPK}同簇重复']:.4f}。")
    print("List-wise 损失只改变「如何比较同一列表内的候选」,打分函数仍是逐点的,"
          "因此拿不到集合上下文;")
    print("Set-wise 把集合统计量喂进打分函数,才真正建模 P(Value|Set)。")

    print("\n" + "=" * 108)
    print("置换不变性数值验证:f({x_1,...,x_n}) = f({x_{π(1)},...,x_{π(n)}})")
    print("=" * 108)
    sample = fva[:1]
    with torch.no_grad():
        base = models["D Set-wise Deep Sets"](sample)[0]
    print(f"{'随机排列 π':>14}{'打分向量最大逐元素偏差':>26}{'Top-5 集合是否一致':>22}")
    for t in range(3):
        perm = torch.randperm(N_CAND, generator=torch.Generator().manual_seed(100 + t))
        with torch.no_grad():
            sp = models["D Set-wise Deep Sets"](sample[:, perm])[0]
        restored = torch.empty_like(sp)
        restored[perm] = sp
        same = (set(torch.topk(base, TOPK).indices.tolist())
                == set(torch.topk(restored, TOPK).indices.tolist()))
        print(f"{t + 1:>14}{float((restored - base).abs().max()):>26.2e}"
              f"{('是' if same else '否'):>22}")
    print("偏差为浮点误差量级,说明 mean 池化带来的对称性让集合而非序列成为输入单位 ——")
    print("正文所说「{i_1,i_2,i_3} 与 {i_3,i_2,i_1} 是同一个候选集」。")

    print("\n" + "=" * 108)
    print("「往集合里再塞一条同簇内容」时两种模型的打分变化(取验证集第 1 个请求的 Top-1)")
    print("=" * 108)
    probe = fva[:1].clone()
    target = probe[0, 0].clone()
    t_cid = int(target[D_DENSE:].argmax())
    # 集合大小固定为 N_CAND:每一步把一个「其他簇」的候选替换成 target 的副本,
    # 于是 target 所属簇的条数 m_c 单调上升,而 Point-wise 的输入 x_0 丝毫未变。
    others = [j for j in range(1, N_CAND)
              if int(probe[0, j, D_DENSE:].argmax()) != t_cid]
    print(f"{'替换次数':>10}{'集合内同簇条数 m_c':>22}{'Point-wise 打分':>18}"
          f"{'Set-wise 打分':>18}{'真值折扣':>12}")
    for extra in range(0, 5):
        cur = probe.clone()
        for j in others[:extra]:
            cur[0, j] = target
        with torch.no_grad():
            sp_pt = float(models["A Point-wise + BCE"](cur)[0, 0])
            sp_st = float(models["D Set-wise Deep Sets"](cur)[0, 0])
        m_c = float(cur[0, :, D_DENSE:].sum(0)[t_cid])
        print(f"{extra:>10}{int(m_c):>22}{sp_pt:>18.4f}{sp_st:>18.4f}"
              f"{1.0 / (1.0 + RHO * (m_c - 1)):>12.4f}")
    print("Point-wise 打分完全不随集合变化(独立性假设);"
          "Set-wise 打分随同簇条数下降,与真值折扣同向 ——")
    print("这就是 Point-wise -> List-wise -> Set-wise 演进要解决的核心问题。")
