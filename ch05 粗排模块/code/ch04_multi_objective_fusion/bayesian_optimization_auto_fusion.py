"""
自动化融合与贝叶斯优化(第 5 章 §5.4.4 自动化融合与贝叶斯优化)

对应正文概念
------------
正文 §5.4.4 指出线性/乘法/值序融合都属于「人工设计排序公式(Hand-Crafted Fusion)」,
工程师要手调 score = w_1 p_ctr + w_2 p_cvr + w_3 p_evtr + ... 里的 n 个权重;一旦因子
变多(10 个因子 × 每个 20 档 = 20^10 组合),人工调参不可行。于是把融合参数寻优看成
黑盒优化问题:

    w   = (w_1, w_2, ..., w_n)
    score = f(w, pxtr)
    Target(w) = AUC_click + AUC_follow + ...   (或 λ_1 AUC_click + λ_2 AUC_follow + ...)
    w*  = argmax_w Target(w)

目标函数没有解析式、不可求梯度、每次评估很贵,因此用贝叶斯优化。正文给出的两个核心
组件:
  - 代理模型(Surrogate Model):高斯过程  f(x) ~ GP(μ_0(x), K(x, x')),
    后验 f(x)|D ~ N(μ_n(x), σ_n²(x)),μ_n 是预测收益、σ_n 是不确定性;
  - 采集函数(Acquisition Function):在 Exploration 与 Exploitation 之间平衡,
    f_n* = max_{i<=n} f(x_i),EI(x) = E[max(f(x) - f_n*, 0)],x_{n+1} = argmax_x EI(x)。
    经典实现即「GP + EI」。
正文还提到高维时 GP 复杂度上涨,工业界常用 TPE(建模 P(x|y) 而不是 P(y|x),Optuna
默认算法)替代 GP。

本实现做了什么
--------------
1. 手写高斯过程回归(RBF 核 + 噪声项 + Cholesky 求解,只用 numpy),给出 μ_n(x)、σ_n(x)。
2. 手写三种采集函数:EI(闭式解)、UCB(μ + κσ)、PI,全部用 argmax 选下一个采样点。
3. 手写 TPE:按 γ 分位把历史切成 l(x)/g(x) 两个高斯核密度,采样后按 l(x)/g(x) 排序。
4. 黑盒目标函数 Target(w) = AUC_click + AUC_follow,在合成离线样本上真实计算 AUC
   (秩和公式),完全不依赖外部库的优化器。
5. 五种寻参策略(GP-EI / GP-UCB / TPE / 随机搜索 / 网格搜索)同预算对比,打印
   「评估次数 -> 当前最优 Target」收敛曲线数值表 + 最终最优权重表。

关键公式
--------
    f(x) | D ~ N(μ_n(x), σ_n²(x))
    μ_n(x) = k(x,X)[K + σ_n² I]^{-1} y        σ_n²(x) = k(x,x) - k(x,X)[K+σ_n²I]^{-1}k(X,x)
    EI(x)  = (μ - f*)Φ(z) + σφ(z),  z = (μ - f*)/σ
    UCB(x) = μ(x) + κ σ(x)
    TPE:   argmax_x l(x) / g(x)

[MOCK] `offline_dataset` 生成的 pxtr 与 click/follow 标签都是教学用合成数据,
       `black_box_target` 模拟「下发线上 A/B 并回收 reward」这一昂贵评估。
"""

from __future__ import annotations

import math
import random

import numpy as np

OBJECTIVES = ("pctr", "pcvr", "pltr")
BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]   # 三个融合权重的搜索空间 A


# ----------------------------------------------------------------------------
# [MOCK] 离线样本与黑盒目标函数
# ----------------------------------------------------------------------------
def offline_dataset(n: int, rng: np.random.Generator):
    """[MOCK] 合成离线评估样本:三路 pxtr + click / follow 两个用户反馈标签。

    click  主要由 p_ctr 驱动、p_ltr 有少量贡献;
    follow 主要由 p_ltr 驱动、p_cvr 有中等贡献。
    因此 AUC_click 与 AUC_follow 的最优权重方向不同,Target 的最优解是两者的折中。
    """
    pctr = np.clip(rng.beta(3.0, 7.0, n), 1e-4, 1 - 1e-4)          # 均值 ~0.30
    pcvr = np.clip(rng.beta(1.5, 36.0, n), 1e-4, 1 - 1e-4)         # 均值 ~0.04
    pltr = np.clip(rng.beta(5.0, 5.0, n), 1e-4, 1 - 1e-4)          # 均值 ~0.50
    pxtr = np.stack([pctr, pcvr, pltr], axis=1)

    z_click = 3.0 * pctr + 0.6 * pltr - 1.4 + rng.normal(0, 0.35, n)
    z_follow = 2.6 * pltr + 8.0 * pcvr - 1.8 + rng.normal(0, 0.35, n)
    click = (1.0 / (1.0 + np.exp(-3.0 * z_click)) > rng.random(n)).astype(np.float64)
    follow = (1.0 / (1.0 + np.exp(-3.0 * z_follow)) > rng.random(n)).astype(np.float64)
    return pxtr, click, follow


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """秩和公式 AUC = (Σ_{正样本} rank - n_pos(n_pos+1)/2) / (n_pos · n_neg)。"""
    n_pos = float(label.sum())
    n_neg = float(len(label) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(score, kind="stable")
    r = np.empty(len(score), dtype=np.float64)
    r[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # 同分并列取平均秩
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (r[label > 0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class BlackBoxTarget:
    """[MOCK] 昂贵黑盒评估:给定权重 w,算 Target(w) = AUC_click + AUC_follow。"""

    def __init__(self, pxtr, click, follow):
        self.pxtr, self.click, self.follow = pxtr, click, follow
        self.n_eval = 0

    def __call__(self, w) -> float:
        self.n_eval += 1
        s = self.pxtr @ np.asarray(w, dtype=np.float64)
        return auc(s, self.click) + auc(s, self.follow)


# ----------------------------------------------------------------------------
# 代理模型:高斯过程回归(RBF 核)
# ----------------------------------------------------------------------------
class GaussianProcess:
    """f(x) ~ GP(μ_0, K(x, x')),RBF 核 k(x,x') = σ_f² exp(-||x-x'||² / (2 l²))。"""

    def __init__(self, length_scale: float = 0.35, sigma_f: float = 1.0,
                 sigma_n: float = 1e-3):
        self.l, self.sf, self.sn = length_scale, sigma_f, sigma_n

    def _k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return self.sf ** 2 * np.exp(-0.5 * d2 / self.l ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.X = X
        self.mu0 = float(y.mean())
        self.ystd = float(y.std()) or 1.0
        yc = (y - self.mu0) / self.ystd
        K = self._k(X, X) + self.sn ** 2 * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yc))
        return self

    def predict(self, Xs: np.ndarray):
        """返回后验均值 μ_n(x) 与标准差 σ_n(x)(已还原到原始目标尺度)。"""
        Ks = self._k(self.X, Xs)
        mu = Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)
        var = self.sf ** 2 - (v ** 2).sum(0)
        sd = np.sqrt(np.clip(var, 1e-12, None))
        return mu * self.ystd + self.mu0, sd * self.ystd


# ----------------------------------------------------------------------------
# 采集函数
# ----------------------------------------------------------------------------
_SQRT2 = math.sqrt(2.0)


def _phi(z):          # 标准正态 pdf
    return np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)


def _Phi(z):          # 标准正态 cdf
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / _SQRT2))


def acq_ei(mu, sd, f_best, xi: float = 0.01):
    """EI(x) = E[max(f(x) - f_n*, 0)] = (μ - f*)Φ(z) + σφ(z)。"""
    imp = mu - f_best - xi
    z = imp / np.maximum(sd, 1e-12)
    return imp * _Phi(z) + sd * _phi(z)


def acq_ucb(mu, sd, f_best, kappa: float = 2.0):
    """UCB(x) = μ(x) + κ σ(x)。"""
    return mu + kappa * sd


def acq_pi(mu, sd, f_best, xi: float = 0.01):
    """PI(x) = P(f(x) > f_n*) = Φ((μ - f*)/σ)。"""
    return _Phi((mu - f_best - xi) / np.maximum(sd, 1e-12))


# ----------------------------------------------------------------------------
# 五种寻参策略
# ----------------------------------------------------------------------------
def sample_space(m: int, rng: np.random.Generator) -> np.ndarray:
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    return lo + (hi - lo) * rng.random((m, len(BOUNDS)))


def bayes_opt(target, n_init: int, n_total: int, acq, rng, n_cand: int = 1500):
    """GP + 采集函数的贝叶斯优化主循环。返回每次评估后的 best-so-far 曲线。"""
    X = sample_space(n_init, rng)
    y = np.array([target(x) for x in X])
    curve = list(np.maximum.accumulate(y))
    while len(y) < n_total:
        gp = GaussianProcess().fit(X, y)
        cand = sample_space(n_cand, rng)
        mu, sd = gp.predict(cand)
        x_next = cand[int(np.argmax(acq(mu, sd, y.max())))]
        X = np.vstack([X, x_next])
        y = np.append(y, target(x_next))
        curve.append(max(curve[-1], y[-1]))
    return np.array(curve), X[int(np.argmax(y))], float(y.max())


def tpe_opt(target, n_init: int, n_total: int, rng, gamma: float = 0.25,
            n_cand: int = 500, bw: float = 0.15):
    """TPE:按 γ 分位切分历史,用高斯核密度建 l(x)/g(x),取 argmax l(x)/g(x)。"""
    X = sample_space(n_init, rng)
    y = np.array([target(x) for x in X])
    curve = list(np.maximum.accumulate(y))
    while len(y) < n_total:
        order = np.argsort(-y)
        n_good = max(1, int(math.ceil(gamma * len(y))))
        good, bad = X[order[:n_good]], X[order[n_good:]]
        if len(bad) == 0:
            bad = X
        # 从 l(x) 采样:以 good 点为中心加高斯扰动
        centers = good[rng.integers(0, len(good), n_cand)]
        cand = np.clip(centers + rng.normal(0, bw, centers.shape), 0.0, 1.0)

        def kde(ref):
            d2 = ((cand[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
            return np.exp(-0.5 * d2 / bw ** 2).mean(1) + 1e-12

        x_next = cand[int(np.argmax(kde(good) / kde(bad)))]
        X = np.vstack([X, x_next])
        y = np.append(y, target(x_next))
        curve.append(max(curve[-1], y[-1]))
    return np.array(curve), X[int(np.argmax(y))], float(y.max())


def random_search(target, n_total: int, rng):
    X = sample_space(n_total, rng)
    y = np.array([target(x) for x in X])
    return np.maximum.accumulate(y), X[int(np.argmax(y))], float(y.max())


def grid_search(target, n_total: int):
    """正文提到的「每个权重取若干候选档位」式网格搜索。3 维 × 4 档 = 64 个点。"""
    g = np.linspace(0.0, 1.0, 4)
    pts = [np.array([a, b, c]) for a in g for b in g for c in g][:n_total]
    y = np.array([target(p) for p in pts])
    return np.maximum.accumulate(y), pts[int(np.argmax(y))], float(y.max())


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    N_SAMPLE, N_INIT, N_TOTAL = 4000, 8, 48
    pxtr, click, follow = offline_dataset(N_SAMPLE, np.random.default_rng(42))

    def fresh():
        return BlackBoxTarget(pxtr, click, follow)

    print("=" * 102)
    print(f"黑盒目标 Target(w) = AUC_click + AUC_follow,样本数 {N_SAMPLE},"
          f"搜索空间 w ∈ [0,1]^3,预算 {N_TOTAL} 次评估")
    print("=" * 102)

    # 人工基线:正文 §5.4.1 常用的手调权重
    base = fresh()
    manual = {"手调 (0.5,0.3,0.2)": (0.5, 0.3, 0.2),
              "手调 (0.34,0.33,0.33)": (0.34, 0.33, 0.33),
              "只用 p_ctr (1,0,0)": (1.0, 0.0, 0.0),
              "只用 p_ltr (0,0,1)": (0.0, 0.0, 1.0)}
    print(f"{'人工权重基线':>24}{'AUC_click':>12}{'AUC_follow':>12}{'Target':>10}")
    for name, w in manual.items():
        s = pxtr @ np.array(w)
        a1, a2 = auc(s, click), auc(s, follow)
        print(f"{name:>24}{a1:>12.4f}{a2:>12.4f}{a1 + a2:>10.4f}")

    # 六种自动化寻参(同预算 48 次评估,同一套初始点随机流)
    runs = {}
    runs["GP + EI"] = bayes_opt(fresh(), N_INIT, N_TOTAL, acq_ei, np.random.default_rng(7))
    runs["GP + UCB(κ=2)"] = bayes_opt(fresh(), N_INIT, N_TOTAL, acq_ucb, np.random.default_rng(7))
    runs["GP + PI"] = bayes_opt(fresh(), N_INIT, N_TOTAL, acq_pi, np.random.default_rng(7))
    runs["TPE(γ=0.25)"] = tpe_opt(fresh(), N_INIT, N_TOTAL, np.random.default_rng(7))
    runs["随机搜索"] = random_search(fresh(), N_TOTAL, np.random.default_rng(7))
    runs["网格搜索(4档/维)"] = grid_search(fresh(), N_TOTAL)

    print("\n" + "=" * 102)
    print("收敛曲线:第 n 次评估后的 best-so-far Target(越早越高越好)")
    print("=" * 102)
    checkpoints = [1, 4, 8, 12, 16, 24, 32, 40, 48]
    print(f"{'策略':>20}" + "".join(f"{f'n={c}':>9}" for c in checkpoints))
    for name, (curve, _, _) in runs.items():
        print(f"{name:>20}" + "".join(f"{curve[c - 1]:>9.4f}" for c in checkpoints))

    print("\n" + "=" * 102)
    print("最终结果:最优权重与达到「随机搜索终点水平」所需的评估次数")
    print("=" * 102)
    rand_final = runs["随机搜索"][0][-1]
    print(f"{'策略':>20}{'w1(p_ctr)':>11}{'w2(p_cvr)':>11}{'w3(p_ltr)':>11}"
          f"{'归一化方向 w/Σw':>26}{'最优Target':>12}{'vs手调Δ':>10}{'追上随机需评估':>15}")
    s0 = pxtr @ np.array([0.5, 0.3, 0.2])
    manual_target = auc(s0, click) + auc(s0, follow)
    for name, (curve, w_best, y_best) in runs.items():
        hit = next((i + 1 for i, v in enumerate(curve) if v >= rand_final - 1e-9), None)
        hit_s = str(hit) if hit else ">48"
        nw = np.asarray(w_best) / max(float(np.sum(w_best)), 1e-12)
        direction = f"({nw[0]:.2f},{nw[1]:.2f},{nw[2]:.2f})"
        print(f"{name:>20}{w_best[0]:>11.4f}{w_best[1]:>11.4f}{w_best[2]:>11.4f}"
              f"{direction:>26}{y_best:>12.4f}{y_best - manual_target:>+10.4f}{hit_s:>15}")
    print(f"\n随机搜索 48 次评估的终点 Target = {rand_final:.4f};"
          f"手调 (0.5,0.3,0.2) 的 Target = {manual_target:.4f}")
    print("注意 AUC 只依赖排序,Target(w) 对 w 的整体缩放不变,所以各策略的最优解应比较"
          "「归一化方向」而不是绝对数值。")

    # 代理模型质量:GP 后验在留出点上的校准情况
    print("\n" + "=" * 102)
    print("代理模型质量:用 48 个观测点拟合 GP,在 300 个留出点上检查 μ_n 与 σ_n")
    print("=" * 102)
    rng2 = np.random.default_rng(123)
    tgt = fresh()
    X_obs = sample_space(N_TOTAL, np.random.default_rng(7))
    y_obs = np.array([tgt(x) for x in X_obs])
    gp = GaussianProcess().fit(X_obs, y_obs)
    X_val = sample_space(300, rng2)
    mu, sd = gp.predict(X_val)
    y_val = np.array([tgt(x) for x in X_val])
    err = np.abs(mu - y_val)
    inside = float(np.mean(err <= 1.96 * sd))
    print(f"{'指标':>28}{'数值':>12}")
    print(f"{'后验均值 MAE':>28}{err.mean():>12.5f}")
    print(f"{'后验均值 RMSE':>28}{math.sqrt((err ** 2).mean()):>12.5f}")
    print(f"{'平均后验标准差 σ_n':>28}{sd.mean():>12.5f}")
    print(f"{'真值落在 μ±1.96σ 的比例':>28}{inside:>12.3f}")
    print(f"{'Pearson 相关(μ_n vs 真值)':>28}"
          f"{float(np.corrcoef(mu, y_val)[0, 1]):>12.4f}")
    print("\n结论:GP 后验既给出预测收益 μ_n,也给出不确定性 σ_n;EI/UCB 正是靠 σ_n 做探索,")
    print("因此在相同评估预算下比随机搜索、网格搜索更早逼近最优融合权重。")
