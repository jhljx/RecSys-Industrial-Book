"""对应正文第 7 章 §7.4「序列评估与排序」中的**序列评估模型**(List Evaluator)。

本实现做了什么
----------------
正文 §7.4 要求序列评估「采用监督学习范式对整个序列的价值进行建模」,并给出逐物品损失
    L = (1/B) sum_b sum_j || y_hat_bj - value_bj ||^2 ,  value_ij = p(item_i) * f(x_i)
本脚本用 torch 实现三个最小可运行的评估器,输入是**整条候选序列**,输出每个位置的价值:

1. PointwiseMLP(基线,对应正文批评的「单点打分」):
   每个位置独立过 MLP,序列内物品互不可见 → 无法建模互相影响;
2. DLCM 风格(Deep Listwise Context Model,Ai et al., SIGIR 2018,
   《Learning a Deep Listwise Context Model for Ranking Refinement》):
   用 GRU 从后往前扫描初始排序序列得到「列表上下文」s_n,
   再用局部排序函数 phi(o_i, s_n) = V^T tanh(W [o_i ; s_n]) 给每个位置打分;
3. PRM 风格(Personalized Re-ranking Model,Pei et al., RecSys 2019,
   《Personalized Re-ranking for Recommendation》):
   输入 = 物品特征 + 可学习位置编码,过多头自注意力 Transformer 编码器,
   每个位置用共享 FFN 输出分数,因此序列内任意两个位置可以互相影响。

训练标签与正文完全对齐:value_ij = p(item_i) * f(x_i),
p(item_i) = p0 (i=1) 或 p0 * prod_{j<i} p_slide(x_j) (i>1),损失为逐物品 MSE。
评估时序列价值 = sum_i y_hat_i(正文 value = sum_i value_i)。

与正文的差异:正文 §7.4 只给出「序列评估 = 监督学习建模整条序列价值」的范式与上面的
逐物品损失,并没有点名具体网络结构。本脚本的 DLCM / PRM 两种结构取自公开论文,
训练目标与标签定义严格按正文公式,属于「正文范式的两种具体实例」。

输出的数值表包含:逐物品 MSE、序列价值 MAE、候选序列排序的 Spearman 相关系数、
top-1 命中率、相对最优序列的价值损失(regret)、以及「位置互换敏感性」
(把同一个集合的两种排列喂进去,看评估器能否区分 —— pointwise 恒等于 0)。

关键公式(§7.4)
---------------
    p(item_i) = p0,                        i = 1
              = p0 * prod_{j=1..i-1} p_slide(x_j),  i > 1
    value_i   = p(item_i) * f(x_i),  value = sum_i value_i
    L         = (1/B) sum_b sum_j || y_hat_bj - value_bj ||^2
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

D = 8            # 物品特征维度
LIST_LEN = 10    # 序列长度 N
P0 = 0.95        # 正文 p0


# ---------------------------------------------------------------------------
# [MOCK] 合成上下滑环境:真实的 f(x) 与 p_slide(x) 由固定随机权重给出
# ---------------------------------------------------------------------------
class SlideEnv:
    """[MOCK] 教学用上下滑消费环境,提供 ground-truth 的 f 与 p_slide。"""

    def __init__(self, rng):
        self.w_f = rng.normal(0, 1.0, D)
        self.w_s = rng.normal(0, 0.8, D)
        self.b_s = 0.6

    def f(self, x):
        """期望观看时长(秒):线性 + 饱和非线性。"""
        z = x @ self.w_f
        return 20.0 + 25.0 * np.tanh(z)

    def p_slide(self, x):
        """看完该物品后继续下滑的概率。"""
        z = x @ self.w_s + self.b_s
        return 1.0 / (1.0 + np.exp(-z))

    def per_item_value(self, seq):
        """返回 (value_i 数组, 序列总价值),严格按正文公式累乘。"""
        p = P0
        vals = []
        for i, x in enumerate(seq):
            if i > 0:
                p = p * self.p_slide(seq[i - 1])
            vals.append(p * self.f(x))
        vals = np.asarray(vals)
        return vals, float(vals.sum())


def sample_dataset(env, n_lists, pool_size, rng):
    """生成 (序列特征, 逐位置 value 标签)。"""
    xs, ys = [], []
    for _ in range(n_lists):
        pool = rng.normal(0, 1.0, (pool_size, D))
        seq = pool[rng.permutation(pool_size)[:LIST_LEN]]
        vals, _ = env.per_item_value(seq)
        xs.append(seq)
        ys.append(vals)
    return (torch.tensor(np.asarray(xs), dtype=torch.float32),
            torch.tensor(np.asarray(ys), dtype=torch.float32))


# ---------------------------------------------------------------------------
# 评估器 1:Pointwise MLP(序列内物品互不可见)
# ---------------------------------------------------------------------------
class PointwiseMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):                     # x: (B, N, D)
        return self.net(x).squeeze(-1)        # (B, N)


# ---------------------------------------------------------------------------
# 评估器 2:DLCM 风格 —— GRU 编码列表上下文 + 局部排序函数
# ---------------------------------------------------------------------------
class DLCMEvaluator(nn.Module):
    """phi(o_i, s_n) = V^T tanh(W [o_i ; s_n] + b)。

    按 DLCM 原文,GRU 从排序靠后的物品往靠前的物品扫描,最后一个隐状态 s_n
    即「整条列表的上下文」,o_i 为第 i 个位置的隐状态。
    """

    def __init__(self, hidden=32):
        super().__init__()
        self.proj = nn.Linear(D, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.score = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))

    def forward(self, x):
        h = torch.tanh(self.proj(x))
        rev = torch.flip(h, dims=[1])          # 从后往前
        out_rev, s_n = self.gru(rev)
        o = torch.flip(out_rev, dims=[1])      # 还原成原始位置顺序
        ctx = s_n[-1].unsqueeze(1).expand_as(o)
        return self.score(torch.cat([o, ctx], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# 评估器 3:PRM 风格 —— 位置编码 + 多头自注意力
# ---------------------------------------------------------------------------
class PRMEvaluator(nn.Module):
    def __init__(self, hidden=32, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(D, hidden)
        self.pos = nn.Parameter(torch.zeros(LIST_LEN, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden,
            dropout=0.0, batch_first=True, activation="relu")
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.proj(x) + self.pos.unsqueeze(0)
        return self.out(self.enc(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# 训练:逐物品 MSE,正文公式 L_ij = ||y_hat_ij - value_ij||^2
# ---------------------------------------------------------------------------
def train(model, xs, ys, epochs, lr, batch=64, y_scale=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = xs.shape[0]
    hist = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            pred = model(xs[idx])
            loss = ((pred - ys[idx] / y_scale) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if ep % max(1, epochs // 4) == 0 or ep == 1:
            hist.append((ep, total / n))
    return hist


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else 0.0


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    rng = np.random.RandomState(42)
    env = SlideEnv(rng)

    n_train, n_test = 900, 300
    xs_tr, ys_tr = sample_dataset(env, n_train, 24, rng)
    xs_te, ys_te = sample_dataset(env, n_test, 24, rng)
    y_scale = float(ys_tr.mean())      # 标签量纲 ~20 秒,缩放后更好训

    print("== 序列评估模型训练(逐物品 MSE,标签 = p(item_i) * f(x_i))==")
    print(f"训练序列 {n_train} 条,测试 {n_test} 条,序列长度 N={LIST_LEN},"
          f"特征维度 d={D},p0={P0}")
    print(f"标签缩放系数 y_scale = 训练集平均 value = {y_scale:.4f}")

    models = {
        "PointwiseMLP": (PointwiseMLP(), 40, 0.012),
        "DLCM(GRU)   ": (DLCMEvaluator(), 40, 0.012),
        "PRM(SelfAttn)": (PRMEvaluator(), 40, 0.012),
    }
    trained = {}
    for name, (model, epochs, lr) in models.items():
        hist = train(model, xs_tr, ys_tr, epochs, lr, y_scale=y_scale)
        trained[name] = model
        print(f"  {name} 训练损失: "
              + " -> ".join(f"ep{e}={l:.4f}" for e, l in hist))

    # ---------------- 测试集精度 ----------------
    print("\n== (1) 测试集精度 ==")
    print(f"{'评估器':>14} {'逐物品RMSE(秒)':>16} {'序列价值MAE(秒)':>17} "
          f"{'序列价值MAPE':>13}")
    # f(x)=20+25 tanh(z) 可能为负,故 MAPE 分母取 |真实序列价值|
    true_list_val = ys_te.sum(dim=1)
    for name, model in trained.items():
        with torch.no_grad():
            pred = model(xs_te) * y_scale
        rmse = float(torch.sqrt(((pred - ys_te) ** 2).mean()))
        pv = pred.sum(dim=1)
        mae = float((pv - true_list_val).abs().mean())
        mape = float(((pv - true_list_val).abs()
                      / true_list_val.abs()).mean()) * 100
        print(f"{name:>14} {rmse:>16.4f} {mae:>17.4f} {mape:>12.2f}%")

    # ---------------- 候选序列选序能力 ----------------
    print("\n== (2) 选序能力:同一候选池生成 32 条排列,看谁能挑出真实价值最高的 ==")
    n_trial, n_cand = 60, 32
    stats = {name: {"rho": [], "hit": 0, "regret": []} for name in trained}
    stats["单点分朴素加和"] = {"rho": [], "hit": 0, "regret": []}
    for _ in range(n_trial):
        pool = rng.normal(0, 1.0, (24, D))
        cands = np.asarray([pool[rng.permutation(24)[:LIST_LEN]]
                            for _ in range(n_cand)])
        gt = np.asarray([env.per_item_value(c)[1] for c in cands])
        best = float(gt.max())
        ct = torch.tensor(cands, dtype=torch.float32)
        for name, model in trained.items():
            with torch.no_grad():
                pv = (model(ct) * y_scale).sum(dim=1).numpy()
            stats[name]["rho"].append(spearman(list(pv), list(gt)))
            pick = int(np.argmax(pv))
            stats[name]["hit"] += int(pick == int(np.argmax(gt)))
            stats[name]["regret"].append((best - gt[pick]) / best * 100)
        naive = np.asarray([sum(env.f(x) for x in c) for c in cands])
        s = stats["单点分朴素加和"]
        s["rho"].append(spearman(list(naive), list(gt)))
        pick = int(np.argmax(naive))
        s["hit"] += int(pick == int(np.argmax(gt)))
        s["regret"].append((best - gt[pick]) / best * 100)

    print(f"{'评估器':>16} {'Spearman':>10} {'top-1命中':>10} {'价值regret':>11}")
    for name, s in stats.items():
        print(f"{name:>16} {np.mean(s['rho']):>10.4f} "
              f"{100.0 * s['hit'] / n_trial:>9.1f}% "
              f"{np.mean(s['regret']):>10.2f}%")

    # ---------------- 位置互换敏感性 ----------------
    print("\n== (3) 位置敏感性:同一物品集合的不同排列,预测价值的标准差 ==")
    print("(pointwise 评估器对排列完全不敏感 → 标准差必为 0,这正是正文说的"
          "「单点打分无法做序列级优化」)")
    pool = rng.normal(0, 1.0, (LIST_LEN, D))
    perms = np.asarray([pool[rng.permutation(LIST_LEN)] for _ in range(64)])
    pt = torch.tensor(perms, dtype=torch.float32)
    gt_perm = np.asarray([env.per_item_value(p)[1] for p in perms])
    print(f"{'评估器':>16} {'预测价值均值':>14} {'预测价值std':>13} "
          f"{'极差/均值':>11}")
    print(f"{'真实环境':>16} {gt_perm.mean():>14.4f} {gt_perm.std():>13.4f} "
          f"{(gt_perm.max() - gt_perm.min()) / gt_perm.mean():>10.2%}")
    for name, model in trained.items():
        with torch.no_grad():
            pv = (model(pt) * y_scale).sum(dim=1).numpy()
        rng_ratio = (pv.max() - pv.min()) / pv.mean()
        print(f"{name:>16} {pv.mean():>14.4f} {pv.std():>13.4f} "
              f"{rng_ratio:>10.2%}")

    print("\n[结论] 对应正文 §7.4:")
    print("  - 逐物品标签 value_ij = p(item_i) f(x_i) 把位置偏置与前缀依赖都写进监督信号;")
    print("  - Pointwise 评估器预测的序列价值与排列无关,无法在候选序列之间做选择;")
    print("  - DLCM 的 GRU 上下文与 PRM 的自注意力都能看到整条序列,")
    print("    因此 Spearman / top-1 命中率明显优于单点口径;")
    print("  - 把序列生成看作 Actor、序列评估看作 Critic,本脚本训练的就是 Critic。")


if __name__ == "__main__":
    main()
