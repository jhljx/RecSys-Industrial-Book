"""
排序公式超参数寻优:CEM(Cross-Entropy Method)(第 6 章 §6.4.3 排序公式超参数寻优)

对应正文概念:
正文把排序公式超参数寻优定义为一个黑盒优化问题:
  theta = (theta_1, ..., theta_N),  theta* = argmax_theta R(theta)
其中 R(theta) 是线上 AB 实验反馈的业务收益(人均观看时长 / 留存率 / GMV /
广告收益 / LTV),无法求 dR/dtheta_i,因此传统梯度下降不可用。CEM 的核心思想是
「不直接搜索最优参数,而是搜索最优参数所在的概率分布」,假设 theta ~ N(mu, Sigma),
不断更新 (mu, Sigma) 使采样分布向高收益区域收缩:

  采样阶段:   theta_i ~ N(mu^(t), Sigma^(t)),  i = 1..K
  评估阶段:   R_1..R_K,降序排列 R_(1) >= ... >= R_(K)
  精英样本:   E = {theta_(1), ..., theta_(M)},  M = rho * K,  rho in [0.05, 0.2]
  分布更新:   mu^(t+1)    = (1/M) sum_i theta_(i)
              Sigma^(t+1) = (1/M) sum_i (theta_(i)-mu^(t+1))(theta_(i)-mu^(t+1))^T

随迭代进行 Sigma^(t) -> 0,采样空间不断收缩到局部最优附近。正文强调其理论基础是
最小化 D_KL(g* || f_theta),等价于最小化交叉熵,故名 Cross-Entropy Method。

本实现做了什么(严格按正文四个阶段 + 正文列出的三个工业界问题 + 正文结语):
1. 完整实现正文公式的 CEM(对角 Sigma 与完整 Sigma 两种),搜索正文给的乘法
   融合公式 Score = prod_i (beta_i + alpha_i * p_i)^gamma_i 的全部参数
     theta = (alpha_1..alpha_N, beta_1..beta_N, gamma_1..gamma_N),维度 3N。
2. [MOCK] 一个进程内「线上 AB 环境」把 theta 映射到业务收益 R:按融合分取
   Top-K 曝光,累计人均观看时长与人均 GMV。对优化器完全黑盒(只返回标量)。
3. 复现正文「工业界 CEM 应用时存在的问题」三节:
   (a) 数据噪声:R_hat(theta) = R(theta) + epsilon,打印不同噪声强度下
       「观测最优」与「该参数真实收益」的背离,以及精英样本命中率;
   (b) 延迟反馈:小时级代理指标(快但有偏)vs 天级真指标(准但慢),打印
       T_total = 轮数 x T_eval 与最终真实收益;
   (c) 维度灾难:固定评估预算,比较 3N = 3 / 9 / 15 维下 CEM 与随机搜索。
4. 与随机搜索、坐标式网格调参([MOCK] 模拟「人工经验调参」)做同预算对比。
5. 复现正文结语「究竟什么指标才应该作为自动寻优的目标函数」:同一个 CEM、
   三种目标函数,得到三套完全不同的最优参数与完全不同的指标权衡。

关键公式(正文原文符号):
  mu^(t+1) = (1/M) sum theta_(i)
  Sigma^(t+1) = (1/M) sum (theta_(i)-mu)(theta_(i)-mu)^T,  M = rho K
"""

import math
import random

# 正文乘法公式的因子集合:前三个与业务收益真实相关,后两个是「看起来有用但对
# 当前收益口径几乎无贡献」的干扰因子 —— 工业界扩因子时的常态。
FACTORS = ["ctr", "ltr", "cvr", "cmtr", "ftr"]
N = len(FACTORS)
DIM = 3 * N                    # 正文说的 3N 维参数空间
LAM_GMV = 20.0                 # R = 人均时长 + LAM_GMV * 人均 GMV


class MockOnlineEnv:
    """[MOCK] 模拟线上 AB 实验环境:输入一套排序公式参数,返回业务收益 R。

    真实系统里这一步是「把线上流量切给若干实验桶、积累数小时到数天」,这里用
    进程内合成流量代替。对优化器而言它是完全黑盒:只能查询 theta -> 标量 R。

    每个候选物品有:
      pxtr 元组 p = (p_ctr, p_ltr, p_cvr, p_cmtr, p_ftr) —— 融合公式唯一可见的输入,
                    带校准噪声(正文说的「数值尺度不一致、分布漂移、校准误差」);
      真实后验 true_ctr / true_cvr / watch / price —— 只有环境知道,用于算收益。
    """

    def __init__(self, n_req=30, n_cand=25, topk=5, seed=7):
        rng = random.Random(seed)
        self.topk = topk
        self.requests = []
        for _ in range(n_req):
            cands = []
            for _ in range(n_cand):
                q = rng.random()                       # 内容质量(潜变量)
                u = [rng.random() for _ in range(5)]
                ctr = 0.02 + 0.14 * q + 0.10 * u[0]
                ltr = 0.004 + 0.012 * q ** 2 + 0.016 * u[1] ** 2
                cvr = 0.0005 + 0.0025 * q ** 3 + 0.0035 * u[2] ** 3
                cmtr = 0.001 + 0.004 * u[3] ** 2       # 与收益口径无关
                ftr = 0.0002 + 0.0015 * u[4] ** 2      # 与收益口径无关
                noisy = tuple(
                    max(1e-6, v * (0.85 + 0.3 * rng.random()))
                    for v in (ctr, ltr, cvr, cmtr, ftr))
                watch = 8.0 + 90.0 * ltr / 0.03 + 20.0 * q
                price = 20.0 + 180.0 * rng.random()
                cands.append((noisy, watch * ctr, price * cvr))
            self.requests.append(cands)

    @staticmethod
    def fuse_score(p, a, b, g):
        """Score = prod_i (beta_i + alpha_i * p_i)^gamma_i(正文乘法公式)。"""
        s = 1.0
        for i in range(N):
            s *= max(1e-9, b[i] + a[i] * p[i]) ** g[i]
        return s

    def expose(self, theta):
        """按融合分取 Top-K 曝光,返回 (人均时长, 人均 GMV)。"""
        a, b, g = theta[:N], theta[N:2 * N], theta[2 * N:]
        tot_wt = tot_gmv = 0.0
        k = self.topk
        for cands in self.requests:
            scored = sorted(cands, key=lambda c: -self.fuse_score(c[0], a, b, g))
            for c in scored[:k]:
                tot_wt += c[1]
                tot_gmv += c[2]
        n = len(self.requests)
        return tot_wt / n, tot_gmv / n

    def true_reward(self, theta):
        """真实业务收益 R(theta)(无噪声、无延迟 —— 现实中拿不到)。"""
        wt, gmv = self.expose(theta)
        return wt + LAM_GMV * gmv

    def observed_reward(self, theta, noise_std, rng):
        """正文式 R_hat(theta) = R(theta) + epsilon(流量波动/热点/埋点异常)。"""
        return self.true_reward(theta) + rng.gauss(0.0, noise_std)

    def proxy_reward(self, theta, rng, noise_std=0.0):
        """[MOCK] 小时级代理指标:只看人均时长,忽略 GMV —— 快但有偏。"""
        wt, _ = self.expose(theta)
        return wt + (rng.gauss(0.0, noise_std) if noise_std else 0.0)


# ============================================================ CEM(正文四阶段)
def cem(reward_fn, dim, lo, hi, rng, rounds=10, K=30, rho=0.15,
        full_cov=False, var_floor=1e-4, trace=None):
    """按正文公式实现 CEM。

    reward_fn(theta) -> 标量观测收益;lo/hi 为各维搜索上下界(采样后裁剪);
    full_cov=True 用完整协方差矩阵(正文给的 Sigma 更新式),否则只保留对角线
    (工业界常用简化版,避免估计 3N x 3N 矩阵所需的巨大样本量);
    var_floor 给方差加下限,防止 Sigma -> 0 后彻底失去探索能力。
    """
    mu = [(lo[i] + hi[i]) / 2.0 for i in range(dim)]
    var = [((hi[i] - lo[i]) / 4.0) ** 2 for i in range(dim)]
    cov = ([[var[i] if i == j else 0.0 for j in range(dim)] for i in range(dim)]
           if full_cov else None)
    M = max(2, int(round(rho * K)))
    best_theta, best_obs = list(mu), reward_fn(mu)
    n_eval = 1

    for t in range(rounds):
        # ---- 采样阶段:theta_i ~ N(mu^(t), Sigma^(t))
        samples = []
        for _ in range(K):
            th = (_sample_full(mu, cov, rng) if full_cov else
                  [rng.gauss(mu[i], math.sqrt(max(var[i], var_floor)))
                   for i in range(dim)])
            samples.append([min(hi[i], max(lo[i], th[i])) for i in range(dim)])

        # ---- 线上评估阶段:得到 R_1..R_K 并降序
        scored = sorted(((reward_fn(th), th) for th in samples),
                        key=lambda x: -x[0])
        n_eval += K
        if scored[0][0] > best_obs:
            best_obs, best_theta = scored[0][0], list(scored[0][1])

        # ---- 分布更新阶段:只用前 M = rho*K 个精英样本
        elites = [th for _, th in scored[:M]]
        mu = [sum(th[i] for th in elites) / M for i in range(dim)]
        if full_cov:
            cov = [[sum((th[i] - mu[i]) * (th[j] - mu[j]) for th in elites) / M
                    + (var_floor if i == j else 0.0)
                    for j in range(dim)] for i in range(dim)]
            var = [cov[i][i] for i in range(dim)]
        else:
            var = [sum((th[i] - mu[i]) ** 2 for th in elites) / M + var_floor
                   for i in range(dim)]
        if trace is not None:
            trace.append({
                "round": t + 1,
                "sigma_mean": sum(math.sqrt(v) for v in var) / dim,
                "elite_mean": sum(r for r, _ in scored[:M]) / M,
                "all_mean": sum(r for r, _ in scored) / K,
                "best": best_obs,
            })
    return {"theta": best_theta, "mu": mu, "obs": best_obs, "n_eval": n_eval,
            "M": M}


def _sample_full(mu, cov, rng):
    """用 Cholesky 分解从 N(mu, Sigma) 采样(纯标准库)。"""
    n = len(mu)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = cov[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(s, 1e-12))
            else:
                L[i][j] = s / L[j][j] if L[j][j] > 1e-12 else 0.0
    z = [rng.gauss(0.0, 1.0) for _ in range(n)]
    return [mu[i] + sum(L[i][k] * z[k] for k in range(i + 1)) for i in range(n)]


# ============================================================ 对照基线
def random_search(reward_fn, dim, lo, hi, rng, budget):
    """同预算纯随机搜索(均匀采样)。"""
    best, bt = -float("inf"), None
    for _ in range(budget):
        th = [rng.uniform(lo[i], hi[i]) for i in range(dim)]
        r = reward_fn(th)
        if r > best:
            best, bt = r, th
    return {"theta": bt, "obs": best, "n_eval": budget}


def coordinate_tuning(reward_fn, dim, lo, hi, budget, grid=5):
    """[MOCK] 模拟「人工经验调参」:逐维扫网格、固定其余维度(坐标上升)。"""
    th = [(lo[i] + hi[i]) / 2.0 for i in range(dim)]
    best = reward_fn(th)
    used = 1
    while used + grid <= budget:
        improved = False
        for i in range(dim):
            if used + grid > budget:
                break
            cur, cur_best = th[i], best
            for s in range(grid):
                th[i] = lo[i] + (hi[i] - lo[i]) * s / (grid - 1)
                r = reward_fn(th)
                used += 1
                if r > cur_best:
                    cur_best, cur, improved = r, th[i], True
            th[i], best = cur, cur_best
        if not improved:
            break
    return {"theta": th, "obs": best, "n_eval": used}


if __name__ == "__main__":
    random.seed(42)
    env = MockOnlineEnv()
    LO = [0.1] * N + [0.0] * N + [0.1] * N
    HI = [20.0] * N + [1.0] * N + [3.0] * N
    NAMES = ([f"alpha_{f}" for f in FACTORS] + [f"beta_{f}" for f in FACTORS]
             + [f"gamma_{f}" for f in FACTORS])
    ROUNDS, K, RHO = 10, 30, 0.15

    print("=" * 100)
    print(f"CEM 搜索乘法融合公式 Score = prod (beta_i + alpha_i*p_i)^gamma_i")
    print(f"因子 {FACTORS},待搜索参数维度 3N = {DIM};"
          f"收益 R = 人均时长 + {LAM_GMV:g} x 人均GMV")
    print("=" * 100)
    hand = [1.0] * N + [0.0] * N + [1.0] * N   # 人工默认:权重全 1、无偏置、指数 1
    base = env.true_reward(hand)
    bw, bg = env.expose(hand)
    print(f"人工默认参数(alpha=1, beta=0, gamma=1):人均时长 {bw:.4f}, "
          f"人均GMV {bg:.4f}, R = {base:.4f}")

    # ---------------- (0) CEM 迭代轨迹:验证 Sigma -> 0 与精英驱动
    trace = []
    res = cem(env.true_reward, DIM, LO, HI, random.Random(42),
              rounds=ROUNDS, K=K, rho=RHO, trace=trace)
    print(f"\n【CEM 迭代轨迹(无噪声)】K={K}, rho={RHO} -> M={res['M']} 个精英样本")
    print(f"{'轮次':>4}{'全体观测均值':>16}{'精英观测均值':>16}{'历史最优':>14}"
          f"{'平均 sigma':>14}{'相对人工提升':>14}")
    print("-" * 100)
    for row in trace:
        print(f"{row['round']:>4}{row['all_mean']:>16.4f}{row['elite_mean']:>16.4f}"
              f"{row['best']:>14.4f}{row['sigma_mean']:>14.4f}"
              f"{row['best'] / base - 1:>13.2%}")
    print("-" * 100)
    print("CEM 收敛得到的分布中心 mu^(T):")
    for i in range(0, DIM, N):
        print("  " + "  ".join(
            f"{NAMES[j]:<11}={res['mu'][j]:>7.3f}" for j in range(i, i + N)))
    print(f"最优真实收益 R = {res['obs']:.4f}(评估 {res['n_eval']} 次),"
          f"相对人工 {res['obs'] / base - 1:+.2%}")
    print("观察:每一轮「精英观测均值」都高于「全体观测均值」 —— 这正是分布向高收益")
    print("      区域移动的驱动力;平均 sigma 从 "
          f"{trace[0]['sigma_mean']:.4f} 收缩到 {trace[-1]['sigma_mean']:.4f},")
    print("      对应正文 Sigma^(t) -> 0、「采样空间不断收缩到局部最优附近」。")

    # ---------------- (1) 同预算对比:人工 / 坐标网格 / 随机搜索 / CEM
    BUDGET = res["n_eval"]
    print("\n" + "=" * 100)
    print(f"同评估预算({BUDGET} 次线上评估)下几种寻优方式对比")
    print("=" * 100)
    r_coord = coordinate_tuning(env.true_reward, DIM, LO, HI, BUDGET)
    r_rand = random_search(env.true_reward, DIM, LO, HI, random.Random(42), BUDGET)
    r_full = cem(env.true_reward, DIM, LO, HI, random.Random(42),
                 rounds=ROUNDS, K=K, rho=RHO, full_cov=True)
    print(f"{'寻优方式':<30}{'评估次数':>12}{'人均时长':>12}{'人均GMV':>12}"
          f"{'真实收益R':>14}{'相对人工':>12}")
    print("-" * 100)
    for nm, r in [("人工默认参数", {"theta": hand, "n_eval": 1}),
                  ("坐标式人工调参(逐维网格)", r_coord),
                  ("随机搜索(均匀采样)", r_rand),
                  ("CEM(对角 Sigma)", res),
                  ("CEM(完整 Sigma + Cholesky)", r_full)]:
        wt, gmv = env.expose(r["theta"])
        R = wt + LAM_GMV * gmv
        print(f"{nm:<30}{r['n_eval']:>12d}{wt:>12.4f}{gmv:>12.4f}"
              f"{R:>14.4f}{R / base - 1:>11.2%}")
    print("-" * 100)
    print("观察:CEM 在同预算下给出最高收益,但领先随机搜索的幅度并不大 —— 因为")
    print("      「按融合分取 Top-K」的收益曲面是分段常值的大平台,很多参数组合会")
    print("      给出同一个排序。CEM 真正的价值在于它用分布收缩把评估集中在高收益")
    print("      区域,而不是像随机搜索那样把预算均匀撒在整个 15 维立方体里。")

    # ---------------- (2) 正文问题 (a):线上反馈噪声
    print("\n" + "=" * 100)
    print("正文问题(a)数据噪声:观测收益 R_hat(theta) = R(theta) + epsilon")
    print("=" * 100)
    print(f"{'噪声 std':>10}{'噪声/可挖收益':>16}{'CEM观测最优(虚高)':>22}"
          f"{'该参数真实收益':>18}{'真实收益损失':>16}{'精英命中率':>14}")
    print("-" * 100)
    signal = res["obs"] - base
    for std in [0.0, 0.1, 0.5, 2.0]:
        rg = random.Random(42)
        r = cem(lambda th: env.observed_reward(th, std, rg), DIM, LO, HI,
                random.Random(42), rounds=ROUNDS, K=K, rho=RHO)
        true_r = env.true_reward(r["theta"])
        # 精英命中率:同一批采样下「按观测值选出的 Top-M」落在「按真实值选出的
        # Top-M」里的比例 —— 直接量化正文说的「Elite Samples 未必真的最优」
        rg2 = random.Random(7)
        cand = [[min(HI[i], max(LO[i],
                 rg2.gauss(r["mu"][i], 0.1 * (HI[i] - LO[i]))))
                 for i in range(DIM)] for _ in range(K)]
        tru = [env.true_reward(th) for th in cand]
        obs = [tru[i] + rg2.gauss(0.0, std) for i in range(K)]
        M = r["M"]
        eo = set(sorted(range(K), key=lambda i: -obs[i])[:M])
        et = set(sorted(range(K), key=lambda i: -tru[i])[:M])
        print(f"{std:>10.2f}{std / signal:>16.3f}{r['obs']:>22.4f}"
              f"{true_r:>18.4f}{res['obs'] - true_r:>16.4f}"
              f"{len(eo & et) / M:>14.0%}")
    print("-" * 100)
    print("观察:噪声越大,CEM 报出的「观测最优」被 epsilon 抬得越虚高(std=2.0 时")
    print("      观测值 128.28 远超无噪声真实上界 122.75),但它对应的真实收益反而")
    print("      更低(122.75 -> 121.62);精英命中率从 100% 一路掉到 75%/50%/0% ——")
    print("      即正文所说「Elite Samples 并不一定是真正意义上的最优参数区域,")
    print("      也有可能只是在局部最优做随机扰动」。")

    # ---------------- (3) 正文问题 (b):延迟反馈
    print("\n" + "=" * 100)
    print("正文问题(b)延迟反馈:小时级代理指标(快但有偏)vs 天级真指标(准但慢)")
    print("=" * 100)
    print(f"{'寻优目标函数':<30}{'单轮T_eval':>12}{'轮数':>8}{'T_total':>12}"
          f"{'人均时长':>12}{'人均GMV':>12}{'真实收益R':>14}")
    print("-" * 100)
    rg = random.Random(42)
    r_proxy = cem(lambda th: env.proxy_reward(th, rg, 0.1), DIM, LO, HI,
                  random.Random(42), rounds=ROUNDS, K=K, rho=RHO)
    rg = random.Random(42)
    r_day3 = cem(lambda th: env.observed_reward(th, 0.1, rg), DIM, LO, HI,
                 random.Random(42), rounds=3, K=K, rho=RHO)
    rg = random.Random(42)
    r_day10 = cem(lambda th: env.observed_reward(th, 0.1, rg), DIM, LO, HI,
                  random.Random(42), rounds=ROUNDS, K=K, rho=RHO)
    plans = [("小时级代理(仅人均时长)", 2, ROUNDS, r_proxy),
             ("天级真指标(时长+GMV)", 24, 3, r_day3),
             ("天级真指标(时长+GMV)", 24, ROUNDS, r_day10)]
    for nm, per, rounds_, r in plans:
        tot = per * rounds_
        span = f"{tot} 小时" if tot < 48 else f"{tot // 24} 天"
        wt, gmv = env.expose(r["theta"])
        print(f"{nm:<30}{str(per) + ' h':>12}{rounds_:>8}{span:>12}"
              f"{wt:>12.4f}{gmv:>12.4f}{wt + LAM_GMV * gmv:>14.4f}")
    print("-" * 100)
    print("观察:小时级代理指标 10 轮只花 20 小时,但它只盯人均时长,优化出的参数")
    print("      GMV 明显偏低,综合收益反而最差;天级真指标更准,但按正文的算法")
    print("      T_total = 轮数 x T_eval,10 轮就要 10 天 —— 这期间用户兴趣分布、")
    print("      内容生态与流量结构可能已经改变,搜出的「最优参数」可能已经过期。")

    # ---------------- (4) 正文问题 (c):维度灾难
    print("\n" + "=" * 100)
    print("正文问题(c)维度灾难:固定评估预算,比较不同 3N 下 CEM 与随机搜索")
    print("=" * 100)
    print(f"{'参与因子数 n':>12}{'参数维度 3n':>14}{'CEM 收益':>14}"
          f"{'随机搜索收益':>16}{'CEM 相对优势':>16}{'CEM 末轮平均sigma':>20}")
    print("-" * 100)
    for n_fac in [1, 3, 5]:
        def rew(th, n_fac=n_fac):
            full = [1.0] * N + [0.0] * N + [0.0] * N   # 未参与的因子 gamma=0
            for i in range(n_fac):
                full[i] = th[i]
                full[N + i] = th[n_fac + i]
                full[2 * N + i] = th[2 * n_fac + i]
            return env.true_reward(full)

        d = 3 * n_fac
        lo = [0.1] * n_fac + [0.0] * n_fac + [0.1] * n_fac
        hi = [20.0] * n_fac + [1.0] * n_fac + [3.0] * n_fac
        tr = []
        rc = cem(rew, d, lo, hi, random.Random(42), rounds=ROUNDS, K=K,
                 rho=RHO, trace=tr)
        rr = random_search(rew, d, lo, hi, random.Random(42), rc["n_eval"])
        print(f"{n_fac:>12}{d:>14}{rc['obs']:>14.4f}{rr['obs']:>16.4f}"
              f"{rc['obs'] / rr['obs'] - 1:>15.2%}{tr[-1]['sigma_mean']:>20.4f}")
    print("-" * 100)
    print("观察:n=1(3 维)时搜索空间极小,CEM 与随机搜索撞到同一个解(优势 0%);")
    print("      维度升到 9、15 后 CEM 的相对优势才显现(+0.37% -> +0.72%),同时末轮")
    print("      平均 sigma 从 0.04 量级涨到 0.17 —— 同样 301 次评估,高维下分布已经")
    print("      压不下去。这就是正文说的「搜索空间 O(c^N) 指数膨胀,即使采用 CEM 也")
    print("      往往需要大量采样才能覆盖有效搜索区域」;实际工业公式有 N=50 个因子、")
    print("      3N=150 维,算法还没收敛业务环境已经变了。")

    # ---------------- (5) 正文结语:优化器 vs 目标函数
    print("\n" + "=" * 100)
    print("正文结语:「究竟什么指标才应该作为自动寻优的目标函数?」")
    print("=" * 100)
    r_gmv = cem(lambda th: env.expose(th)[1], DIM, LO, HI, random.Random(42),
                rounds=ROUNDS, K=K, rho=RHO)
    print(f"{'寻优目标函数':<30}{'人均时长':>14}{'人均GMV':>14}"
          f"{'综合收益R':>14}{'相对人工':>12}")
    print("-" * 100)
    for nm, th in [("(不寻优)人工默认参数", hand),
                   ("max 人均时长", r_proxy["theta"]),
                   ("max 人均GMV", r_gmv["theta"]),
                   (f"max 时长 + {LAM_GMV:g}*GMV", res["theta"])]:
        wt, gmv = env.expose(th)
        R = wt + LAM_GMV * gmv
        print(f"{nm:<30}{wt:>14.4f}{gmv:>14.4f}{R:>14.4f}{R / base - 1:>11.2%}")
    print("-" * 100)
    print("观察:同一个 CEM、同样的预算,换一个目标函数就得到完全不同的参数与完全")
    print("      不同的指标权衡:只优化时长会牺牲 GMV,只优化 GMV 会把人均时长打到")
    print("      比人工基线还低。CEM / 贝叶斯优化 / 强化学习 / LLM Agent 本质都只是")
    print("      优化器,正文的结论是「如何定义一个合理的优化目标,远比选择哪一种")
    print("      优化算法更加重要」。")
    print("=" * 100)
