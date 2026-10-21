"""
跨业务分数归一化方法对比（第 8 章 §8.2.2 分数归一化）

对应正文概念
------------
各业务竞价分经 SIR 校准后量纲仍不可比（广告 eCPM 数百到数千，推荐 pCTR 在 [0,1]），
因此需要归一化把它们映射到统一数值空间后再参与跨业务流量竞争。
正文列出四种工业界常见方法，并强调「归一化不能替代校准」以及
「归一化 + 线性融合」会放大预估偏差的风险。

关键公式
--------
    Min-Max:       x' = (x - x_min) / (x_max - x_min)
    Log Min-Max:   x' = (log x - log x_min) / (log x_max - log x_min)
    Z-Score:       x' = (x - μ) / σ
    高斯 CDF:      x' = Φ((x - μ) / σ)

本实现做了什么
--------------
1. 构造三个量纲/分布差异极大的业务（广告 eCPM 长尾、直播打赏、短视频 pVTR），
   用四种方法归一化，打印归一化后的均值/标准差/分位数以及跨业务「赢得首位的占比」，
   直观展示哪种方法能让业务公平竞争；
2. 验证四种方法都保持业务内严格保序（Spearman 排序相关 = 1）；
3. 复现正文的风险提示：给广告业务注入 +20% 预估偏差，对比
   「有校准 + 归一化」与「无校准 + 归一化 + 线性融合」下广告流量占比的放大效应；
4. 展示正文提到的分布塑形（Score Shaping）：Top-K 置 1 的做法如何削弱归一化影响。
"""

import math
import random
import statistics


# --------------------------------------------------------------- 归一化
def minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.5] * len(xs) if hi - lo < 1e-12 else [(x - lo) / (hi - lo) for x in xs]


def log_minmax(xs: list[float]) -> list[float]:
    ls = [math.log(max(x, 1e-9)) for x in xs]
    return minmax(ls)


def zscore(xs: list[float]) -> list[float]:
    mu = statistics.fmean(xs)
    sd = statistics.pstdev(xs) or 1e-9
    return [(x - mu) / sd for x in xs]


def _phi(z: float) -> float:
    """标准正态 CDF：Φ(z) = 0.5·(1 + erf(z/√2))。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def gauss_cdf(xs: list[float]) -> list[float]:
    return [_phi(z) for z in zscore(xs)]


NORMALIZERS = {
    "Min-Max": minmax,
    "LogMin-Max": log_minmax,
    "Z-Score": zscore,
    "Gauss-CDF": gauss_cdf,
}


# --------------------------------------------------------------- 工具
def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def quantiles(xs: list[float], qs=(0.1, 0.5, 0.9)) -> list[float]:
    s = sorted(xs)
    return [s[min(len(s) - 1, int(q * len(s)))] for q in qs]


def gen_biz_scores(n: int):
    """三业务竞价分：广告 eCPM（对数正态长尾）、直播打赏（Gamma 状）、短视频 pVTR（Beta）。"""
    ad = [math.exp(random.gauss(6.0, 1.1)) for _ in range(n)]          # ~数百到上万
    live = [sum(random.expovariate(1 / 8.0) for _ in range(3)) for _ in range(n)]  # ~几十
    video = [random.betavariate(2.0, 4.0) for _ in range(n)]           # [0,1]
    return {"广告eCPM": ad, "直播打赏": live, "短视频pVTR": video}


def main() -> None:
    random.seed(42)
    print("=" * 80)
    print("跨业务分数归一化方法对比（§8.2.2）")
    print("=" * 80)

    N = 5000
    raw = gen_biz_scores(N)

    print("\n【1】归一化前：三业务竞价分完全不可比")
    print(f"{'业务':<14}{'min':>12}{'p50':>12}{'mean':>12}{'max':>12}{'std':>12}")
    for biz, xs in raw.items():
        print(f"{biz:<14}{min(xs):>12.4f}{quantiles(xs)[1]:>12.4f}"
              f"{statistics.fmean(xs):>12.4f}{max(xs):>12.4f}{statistics.pstdev(xs):>12.4f}")

    print("\n【2】四种归一化后的分布（p10 / p50 / p90 / mean / std）")
    normed: dict[str, dict[str, list[float]]] = {}
    for name, fn in NORMALIZERS.items():
        normed[name] = {biz: fn(xs) for biz, xs in raw.items()}
        print(f"\n  -- {name} --")
        print(f"  {'业务':<14}{'p10':>10}{'p50':>10}{'p90':>10}{'mean':>10}{'std':>10}")
        for biz, v in normed[name].items():
            q = quantiles(v)
            print(f"  {biz:<14}{q[0]:>10.4f}{q[1]:>10.4f}{q[2]:>10.4f}"
                  f"{statistics.fmean(v):>10.4f}{statistics.pstdev(v):>10.4f}")

    print("\n【3】保序性校验：归一化后业务内 Spearman 排序相关系数（应恒为 1.0）")
    print(f"  {'方法':<14}" + "".join(f"{b:>14}" for b in raw))
    for name in NORMALIZERS:
        line = "".join(f"{spearman(raw[b], normed[name][b]):>14.4f}" for b in raw)
        print(f"  {name:<14}{line}")

    print("\n【4】公平竞争：逐条请求让三业务各出一个候选争夺首位，统计胜出占比")
    print("  （不归一化时广告凭量纲碾压，归一化后三业务按真实分位竞争）")
    def win_rates(get):
        cnt = {b: 0 for b in raw}
        for i in range(N):
            best = max(raw, key=lambda b: get(b, i))
            cnt[best] += 1
        return {b: c / N for b, c in cnt.items()}
    print(f"  {'方案':<14}" + "".join(f"{b:>14}" for b in raw))
    r = win_rates(lambda b, i: raw[b][i])
    print(f"  {'不归一化':<14}" + "".join(f"{r[b]:>14.4f}" for b in raw))
    for name in NORMALIZERS:
        r = win_rates(lambda b, i, n=name: normed[n][b][i])
        print(f"  {name:<14}" + "".join(f"{r[b]:>14.4f}" for b in raw))

    print("\n【5】正文风险提示：归一化不能替代校准")
    print("  归一化只对齐量纲，不纠正预估偏差。对 Min-Max / Z-Score / Gauss-CDF 而言，")
    print("  纯乘性缩放 x→c·x 甚至是完全尺度不变的，所以靠归一化根本看不出模型高估。")
    print("  下面给广告模型注入「高分段额外抬升 1.0σ」的失准，比较两种链路的真实收益：")
    print("    A 先校准再归一化  B 不校准直接归一化（都用 Z-Score + 跨业务取最大值）")
    ad = raw["广告eCPM"]
    thr, sd = sorted(ad)[int(0.5 * N)], statistics.pstdev(ad)
    # true_val 是真实价值，失准只污染「排序用分数」，不改变真实价值
    mis_ad = [x + 1.0 * sd if x > thr else x for x in ad]
    true_val = {b: NORMALIZERS["Z-Score"](xs) for b, xs in raw.items()}
    for tag, ad_score in (("A 先校准再归一化", ad), ("B 未校准直接归一化", mis_ad)):
        score = {b: NORMALIZERS["Z-Score"](xs) for b, xs in raw.items()}
        score["广告eCPM"] = NORMALIZERS["Z-Score"](ad_score)
        gain, wins = 0.0, 0
        for i in range(N):
            pick = max(score, key=lambda b: score[b][i])   # 按（可能失准的）分数选业务
            gain += true_val[pick][i]                      # 但结算按真实价值
            wins += pick == "广告eCPM"
        print(f"    {tag:<22} 广告胜出率={wins / N:.4f}  实际兑现价值={gain / N:+.4f}")
    print("  未被校准吸收的高估让广告抢走更多流量，而真实兑现价值反而下降 —— 流量错配。")

    print("\n【6】分布塑形（Score Shaping）：Top-K 内容分数统一置 1")
    k = int(0.05 * N)
    ad_sorted = sorted(range(N), key=lambda i: -raw["广告eCPM"][i])
    shaped = minmax(raw["广告eCPM"])
    topk = set(ad_sorted[:k])
    shaped = [1.0 if i in topk else v * 0.8 for i, v in enumerate(shaped)]
    plain = minmax(raw["广告eCPM"])
    print(f"  原始 Min-Max：   p50={quantiles(plain)[1]:.4f}  p90={quantiles(plain)[2]:.4f}"
          f"  Top5%均值={statistics.fmean([plain[i] for i in topk]):.4f}")
    print(f"  塑形后：         p50={quantiles(shaped)[1]:.4f}  p90={quantiles(shaped)[2]:.4f}"
          f"  Top5%均值={statistics.fmean([shaped[i] for i in topk]):.4f}")
    print("  Top-K 被钉在 1.0，长尾被整体压低，业务想重点透出的内容曝光更稳定。")


if __name__ == "__main__":
    main()
