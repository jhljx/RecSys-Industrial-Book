"""
多业务分数校准：IR / SIR / Bayesian-SIR（第 8 章 §8.2.1 分数校准方法）

对应正文概念
------------
正文指出混排前处理必须做业务分校准，否则不同业务模型的高估/低估会直接导致
流量错配。给出的三级方案为：
  1. IR（Isotonic Regression，保序回归，fielding1974statistical）：分桶统计真实后验
     r_k = Σ_{i∈B_k} y_i / |B_k|，用 PAVA（de2010isotone）合并违反单调的相邻桶；
  2. SIR（Smoothed Isotonic Regression，deng2020calibrating）：在 IR 的分段常数映射上
     做线性插值，消除桶边界跳变；
  3. Bayesian-SIR：对小样本桶做 Beta-Binomial 共轭贝叶斯平滑，抑制高方差。

关键公式
--------
    单调约束:     f(p_i) <= f(p_j),  ∀ p_i <= p_j
    桶后验:       r_k = Σ_{i∈B_k} y_i / |B_k|
    PAVA 合并:    r_new = (|B_k| r_k + |B_{k+1}| r_{k+1}) / (|B_k| + |B_{k+1}|)
    SIR 插值:     f(p) = r_k + (p - p_k)/(p_{k+1} - p_k) · (r_{k+1} - r_k)
    贝叶斯平滑:   CTR|Data ~ Beta(α + Click, β + PV - Click)
                  E[CTR] = (Click + α) / (PV + α + β)

本实现做了什么
--------------
在合成的「模型系统性高估」日志上依次拟合 IR / SIR / Bayesian-SIR，
打印校准前后的 COPC（= Σpred / Σlabel，正文称 PCOC）与 ECE（期望校准误差），
并复现正文表 8.1 的小样本分桶案例（PV=50, Click=20, 经验 CTR=0.40）说明
为什么需要贝叶斯平滑。

注：正文称 SIR 由「阿里妈妈团队在 2018 年提出」，而引用文献 deng2020calibrating
为 ECML-PKDD 2020 的 "Calibrating User Response Predictions in Online Advertising"。
本实现以正文给出的公式为准。
"""

import math
import random
import statistics


# ------------------------------------------------------------------ 指标
def copc(preds: list[float], labels: list[float]) -> float:
    """COPC / PCOC = Σ预估 / Σ真实，等于 1 表示无偏。"""
    s = sum(labels)
    return float("inf") if s == 0 else sum(preds) / s


def ece(preds: list[float], labels: list[float], n_bin: int = 20) -> float:
    """Expected Calibration Error = Σ_b (n_b/N)·|acc_b - conf_b|。"""
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bin)]
    for p, y in zip(preds, labels):
        bins[min(n_bin - 1, int(p * n_bin))].append((p, y))
    n = len(preds)
    return sum(len(b) / n * abs(sum(y for _, y in b) / len(b) - sum(p for p, _ in b) / len(b))
               for b in bins if b)


# ------------------------------------------------------------------ 校准器
class IsotonicCalibrator:
    """保序回归：等频分桶 + PAVA，映射为分段常数函数 f(p) = c_k。

    prior_strength > 0 时启用分层贝叶斯平滑（Bayesian-SIR）：每个细桶向
    「同区间的粗桶经验 CTR」收缩，即 Beta 先验的均值取局部估计而非全局常数，
    这样既能缩减稀疏桶的方差，又不会给整条校准曲线引入全局偏置。
        r_k = (Click_k + κ·m_k) / (PV_k + κ),  m_k = 所属粗桶的经验 CTR
    """

    def __init__(self, n_bucket: int = 20, prior_strength: float = 0.0,
                 coarse_ratio: int = 5):
        self.n_bucket = n_bucket
        self.prior_strength = prior_strength
        self.coarse_ratio = coarse_ratio
        self.blocks: list[dict] = []

    def fit(self, preds: list[float], labels: list[float]) -> "IsotonicCalibrator":
        pairs = sorted(zip(preds, labels))
        size = max(1, len(pairs) // self.n_bucket)
        raw: list[dict] = []
        for s in range(0, len(pairs), size):
            chunk = pairs[s:s + size]
            if not chunk:
                continue
            pv, click = len(chunk), sum(y for _, y in chunk)
            raw.append({"lo": chunk[0][0], "hi": chunk[-1][0],
                        "mid": sum(p for p, _ in chunk) / pv,
                        "pv": pv, "click": click, "emp": click / pv, "r": click / pv})
        # 分层先验：把相邻 coarse_ratio 个细桶聚成一个粗桶，粗桶经验 CTR 作为先验均值
        if self.prior_strength > 0:
            kappa, cr = self.prior_strength, max(1, self.coarse_ratio)
            for s in range(0, len(raw), cr):
                grp = raw[s:s + cr]
                pv = sum(b["pv"] for b in grp)
                m = sum(b["click"] for b in grp) / pv
                for b in grp:
                    b["r"] = (b["click"] + kappa * m) / (b["pv"] + kappa)
        self.n_raw_block = len(raw)
        merged: list[dict] = []
        for b in raw:
            merged.append(b)
            while len(merged) >= 2 and merged[-2]["r"] > merged[-1]["r"]:
                b2, b1 = merged.pop(), merged.pop()
                pv, click = b1["pv"] + b2["pv"], b1["click"] + b2["click"]
                merged.append({
                    "lo": b1["lo"], "hi": b2["hi"],
                    "mid": (b1["mid"] * b1["pv"] + b2["mid"] * b2["pv"]) / pv,
                    "pv": pv, "click": click, "emp": click / pv,
                    "r": (b1["pv"] * b1["r"] + b2["pv"] * b2["r"]) / pv,
                })
        self.blocks = merged
        return self

    def predict(self, p: float) -> float:
        for b in self.blocks:
            if p <= b["hi"]:
                return b["r"]
        return self.blocks[-1]["r"]


class SmoothedIsotonicCalibrator(IsotonicCalibrator):
    """SIR：在 IR 的单调锚点 (p_k, r_k) 之间做线性插值，得到连续分段线性校准函数。"""

    def predict(self, p: float) -> float:
        bs = self.blocks
        if p <= bs[0]["mid"]:
            return bs[0]["r"]
        if p >= bs[-1]["mid"]:
            return bs[-1]["r"]
        for k in range(len(bs) - 1):
            if bs[k]["mid"] <= p <= bs[k + 1]["mid"]:
                w = (p - bs[k]["mid"]) / max(1e-12, bs[k + 1]["mid"] - bs[k]["mid"])
                return bs[k]["r"] + w * (bs[k + 1]["r"] - bs[k]["r"])
        return bs[-1]["r"]


# ------------------------------------------------------------------ 数据
def make_logs(n: int, bias: float):
    """合成日志：真实后验 t ~ Beta(2,5)，模型预估 p = clip(bias·t + 噪声)。"""
    preds, labels, truths = [], [], []
    for _ in range(n):
        t = random.betavariate(2.0, 5.0)
        p = max(1e-4, min(0.9999, bias * t + random.gauss(0.0, 0.03)))
        preds.append(p)
        truths.append(t)
        labels.append(1.0 if random.random() < t else 0.0)
    return preds, labels, truths


def main() -> None:
    random.seed(42)
    print("=" * 76)
    print("多业务分数校准：IR / SIR / Bayesian-SIR（§8.2.1）")
    print("=" * 76)

    # ---- 1) 大样本：IR / SIR 校准效果
    tr_p, tr_y, _ = make_logs(40000, bias=1.45)      # 训练集：广告 CTR 模型高估 45%
    te_p, te_y, _ = make_logs(20000, bias=1.45)
    print(f"\n【1】大样本场景：训练 {len(tr_p)} / 测试 {len(te_p)}，模型高估系数 bias=1.45")

    ir = IsotonicCalibrator(n_bucket=20).fit(tr_p, tr_y)
    sir = SmoothedIsotonicCalibrator(n_bucket=20).fit(tr_p, tr_y)

    rows = [("校准前(原始pxtr)", te_p),
            ("IR  保序回归", [ir.predict(p) for p in te_p]),
            ("SIR 平滑保序回归", [sir.predict(p) for p in te_p])]

    print(f"\n{'方法':<20}{'COPC':>10}{'ECE':>10}{'|COPC-1|':>12}")
    print("-" * 52)
    for name, pr in rows:
        c = copc(pr, te_y)
        print(f"{name:<20}{c:>10.4f}{ece(pr, te_y):>10.5f}{abs(c - 1):>12.4f}")

    print(f"\nIR 分桶：等频 {ir.n_raw_block} 桶 → PAVA 合并后 {len(ir.blocks)} 桶（单调）")
    print(f"{'区间':<22}{'PV':>7}{'Click':>7}{'经验CTR':>10}{'IR值':>10}")
    for b in ir.blocks[:6]:
        print(f"[{b['lo']:.4f},{b['hi']:.4f}]{'':<4}{b['pv']:>7d}{int(b['click']):>7d}"
              f"{b['emp']:>10.4f}{b['r']:>10.4f}")

    # ---- 2) 桶边界跳变：IR 阶梯 vs SIR 连续
    print("\n【2】桶边界跳变对比（相邻两个预估分几乎相同，但落在不同桶）：")
    edge = ir.blocks[len(ir.blocks) // 2]["hi"]
    print(f"{'p':>10}{'IR f(p)':>12}{'SIR f(p)':>12}")
    for d in (-0.004, -0.001, 0.001, 0.004):
        p = edge + d
        print(f"{p:>10.4f}{ir.predict(p):>12.4f}{sir.predict(p):>12.4f}")
    jump_ir = abs(ir.predict(edge + 0.001) - ir.predict(edge - 0.001))
    jump_sir = abs(sir.predict(edge + 0.001) - sir.predict(edge - 0.001))
    print(f"边界 ±0.001 跳变幅度：IR={jump_ir:.5f}  SIR={jump_sir:.5f}")

    # ---- 3) 小样本：Bayesian-SIR 的方差缩减价值
    #     正文强调贝叶斯平滑是针对「分桶样本稀疏 → 后验统计高方差」的方差缩减技术，
    #     因此只有在小样本分桶下它才优于纯 SIR；大样本时先验会被数据淹没。
    print("\n【3】小样本场景：SIR vs Bayesian-SIR（2500 条日志切 50 桶，每桶约 50 条）")
    print("     Bayesian-SIR 采用分层先验：细桶向所属粗桶（5 个细桶一组）的经验 CTR 收缩，κ=60")
    grid = [i / 40 for i in range(1, 40)]
    n_trial = 15
    acc = {"SIR": [0.0, 0.0], "Bayesian-SIR": [0.0, 0.0]}
    curves: dict[str, list[list[float]]] = {"SIR": [], "Bayesian-SIR": []}
    for _ in range(n_trial):
        s_p, s_y, _ = make_logs(2500, bias=1.45)      # 稀疏日志
        q_p, q_y, _ = make_logs(20000, bias=1.45)
        for name, cal in (("SIR", SmoothedIsotonicCalibrator(50)),
                          ("Bayesian-SIR", SmoothedIsotonicCalibrator(50, 60.0, 5))):
            cal.fit(s_p, s_y)
            out = [cal.predict(p) for p in q_p]
            acc[name][0] += abs(copc(out, q_y) - 1.0)
            acc[name][1] += ece(out, q_y)
            curves[name].append([cal.predict(g) for g in grid])
    print(f"  {'方法':<16}{'平均|COPC-1|':>14}{'平均ECE':>12}{'曲线方差':>12}"
          f"  （{n_trial} 次重复实验）")
    for name, (a, b) in acc.items():
        cs = curves[name]
        var = statistics.fmean([statistics.pvariance([c[j] for c in cs])
                                for j in range(len(grid))])
        print(f"  {name:<16}{a / n_trial:>14.4f}{b / n_trial:>12.5f}{var:>12.6f}")
    print("  「曲线方差」= 不同随机日志拟合出的校准函数在同一 p 处的方差均值，越小越稳。")

    # ---- 4) 正文表 8.1 小样本案例：为什么要贝叶斯平滑
    print("\n【4】正文表 8.1 小样本分桶案例（Beta 先验 α=30, β=170，先验均值 0.15）：")
    alpha, beta = 30.0, 170.0
    print(f"{'pCTR分桶':<14}{'PV':>8}{'Click':>8}{'经验CTR':>10}{'平滑后CTR':>12}{'收缩量':>10}")
    for lo, hi, pv, ck in [(0.1, 0.2, 100000, 12000), (0.2, 0.3, 80000, 18000),
                           (0.3, 0.4, 50, 20)]:
        emp = ck / pv
        sm = (ck + alpha) / (pv + alpha + beta)
        print(f"[{lo},{hi}]{'':<6}{pv:>8d}{ck:>8d}{emp:>10.4f}{sm:>12.4f}{emp - sm:>+10.4f}")
    print("可见大样本桶几乎不变，小样本桶 0.4000 被拉回 0.2000，噪声被有效抑制。")

    # ---- 5) 多业务：校准让不同业务的分数进入同一可信空间
    print("\n【5】多业务校准效果（各业务 bias 不同，SIR 校准后 COPC 全部回到 1 附近）：")
    print(f"{'业务':<10}{'bias':>7}{'COPC前':>10}{'COPC后':>10}{'ECE前':>10}{'ECE后':>10}")
    for biz, bias in [("广告", 1.45), ("直播", 0.80), ("电商", 1.15), ("短视频", 1.02)]:
        p_tr, y_tr, _ = make_logs(30000, bias)
        p_te, y_te, _ = make_logs(15000, bias)
        c = SmoothedIsotonicCalibrator(20).fit(p_tr, y_tr)
        out = [c.predict(p) for p in p_te]
        print(f"{biz:<10}{bias:>7.2f}{copc(p_te, y_te):>10.4f}{copc(out, y_te):>10.4f}"
              f"{ece(p_te, y_te):>10.5f}{ece(out, y_te):>10.5f}")


if __name__ == "__main__":
    main()
