"""
粗排截断机制:截断长度 K 对下游漏斗与算力的影响(第 5 章 §5.6.3 粗排截断机制)

对应正文概念
------------
正文设定:召回阶段返回 5000 个候选,粗排阶段输出 300 ~ 500 个候选,因此需要执行

    TopK(CandidateSet)

例如 K = 300。正文指出截断的核心目的是「控制精排计算成本,提高后续模型吞吐,
降低推荐系统整体的延迟」,并强调「由于精排模型通常远比粗排复杂,因此合理的截断策略
对于整体推荐系统性能具有重要影响」。

本实现做了什么
--------------
1. [MOCK] 一个三段漏斗:召回 5000 -> 粗排打分(带噪)-> TopK 截断 -> 精排(视作真值)
   -> 最终 Top10 曝光。以「不截断时的理论最优 Top10」为参照,度量不同 K 的:
      Recall@K = |Relevant ∩ CandidateSet| / |Relevant|(Relevant = 真值 Top10)
      最终 Top10 的真值收益率与收益损失
      精排算力 FLOPs 与精排环节延迟估计,以及是否满足延迟预算
2. 扫描 K ∈ {50, 100, 200, 300, 500, 1000, 2000, 5000},打印收益-成本表,
   解释正文取 300~500 的原因。
3. 边际收益分析 ΔRecall / ΔFLOPs,展示收益随 K 递减。
4. 一致性强弱(粗排噪声 σ)与所需截断长度的折算表,对应 §5.3 链路一致性。
5. 自适应截断:请求的「难度」不同(头部真值分布有宽有窄),用可观测的粗排分头部区分度
   margin 作为代理,给难请求更大的 K、给简单请求更小的 K,在同等平均算力下对比固定 K。

关键公式
--------
    CandidateSet = TopK(score),  K = 300(正文示例)
    Recall@K = |Relevant ∩ CandidateSet| / |Relevant|
    FLOPs(K) = K · c_rank,  Latency(K) ≈ L_fix + K · c_rank_ms / parallelism
    自适应:K_t = clip( K_base · (1 + λ (1 − margin_t / margin_ref)), K_min, K_max )
"""

from __future__ import annotations

import random

N_RECALL = 5000          # 正文:召回返回 5000 个候选
N_FINAL = 10             # 最终曝光条数
N_REQ = 200              # [MOCK] 仿真请求数
C_RANK_MS = 0.012        # [MOCK] 精排单条打分耗时(ms)
C_RANK_FLOPS = 2.4e6     # [MOCK] 精排单条 FLOPs
L_FIX_MS = 18.0          # [MOCK] 固定开销(特征抽取 + RPC)
PARALLELISM = 8          # [MOCK] 精排打分并行度
LAT_BUDGET_MS = 25.0     # [MOCK] 精排环节延迟预算
SIGMA = 0.016            # [MOCK] 粗排相对精排的打分噪声(越大=链路一致性越差)
KS = (50, 100, 200, 300, 500, 1000, 2000, 5000)


class Request:
    """[MOCK] 一个请求:true 为精排真值分,pre = true + N(0, σ) 为粗排估计分。

    不同请求的真值离散度 tau 不同:tau 小 => 候选彼此接近 => 同样噪声下更难区分
    (「难请求」),需要更长的截断才能保住 Relevant。
    """

    def __init__(self, rng: random.Random, sigma: float = SIGMA, tau: float | None = None):
        self.tau = rng.choice([0.008, 0.014, 0.022, 0.034]) if tau is None else tau
        self.true = [max(1e-5, rng.gauss(0.055, self.tau)) for _ in range(N_RECALL)]
        pre = [t + rng.gauss(0.0, sigma) for t in self.true]
        self.pre_order = sorted(range(N_RECALL), key=lambda i: -pre[i])
        self.pre_sorted = sorted(pre, reverse=True)
        true_order = sorted(range(N_RECALL), key=lambda i: -self.true[i])
        self.relevant = set(true_order[:N_FINAL])
        self.opt_gain = sum(self.true[i] for i in true_order[:N_FINAL])
        # Relevant 物品在粗排序中的位置,用于 O(1) 计算任意 K 的 Recall@K
        pos = {idx: r for r, idx in enumerate(self.pre_order)}
        self.relevant_pos = sorted(pos[i] for i in self.relevant)
        # 可观测的头部区分度:粗排分 Top1 与 Top(4×N_FINAL) 的差
        self.margin = self.pre_sorted[0] - self.pre_sorted[4 * N_FINAL - 1]

    def recall_at(self, k: int) -> float:
        return sum(1 for p in self.relevant_pos if p < k) / N_FINAL

    def gain_at(self, k: int) -> float:
        kept = self.pre_order[:k]
        best = sorted((self.true[i] for i in kept), reverse=True)[:N_FINAL]
        return sum(best) / self.opt_gain


def cost(k: float) -> tuple[float, float]:
    return k * C_RANK_FLOPS, L_FIX_MS + k * C_RANK_MS / PARALLELISM


def evaluate_fixed(reqs: list[Request], k: int) -> dict[str, float]:
    n = len(reqs)
    fl, lat = cost(k)
    return {"recall": sum(r.recall_at(k) for r in reqs) / n,
            "gain": sum(r.gain_at(k) for r in reqs) / n,
            "flops": fl, "lat": lat}


def evaluate_adaptive(reqs: list[Request], k_base: int, lam: float,
                      k_min: int, k_max: int) -> dict[str, float]:
    margins = [r.margin for r in reqs]
    ref = sorted(margins)[len(margins) // 2]
    rec = gain = 0.0
    ks: list[int] = []
    for r in reqs:
        k = int(k_base * (1.0 + lam * (1.0 - r.margin / ref)))
        k = max(k_min, min(k_max, k))
        ks.append(k)
        rec += r.recall_at(k)
        gain += r.gain_at(k)
    n = len(reqs)
    k_avg = sum(ks) / n
    fl, lat = cost(k_avg)
    return {"recall": rec / n, "gain": gain / n, "flops": fl, "lat": lat,
            "k_avg": k_avg, "k_min": min(ks), "k_max": max(ks)}


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)
    reqs = [Request(rng) for _ in range(N_REQ)]

    print("=" * 108)
    print("截断机制仿真([MOCK] 漏斗):召回 5000 -> 粗排打分 -> TopK 截断 -> 精排 -> Top10")
    print("=" * 108)
    print(f"请求数 {N_REQ},精排单条 {C_RANK_MS} ms / {C_RANK_FLOPS:.1e} FLOPs,"
          f"并行度 {PARALLELISM},固定开销 {L_FIX_MS} ms,延迟预算 {LAT_BUDGET_MS} ms")
    print(f"粗排噪声 σ = {SIGMA};Relevant = 精排真值 Top{N_FINAL},"
          f"Recall@K = |Relevant ∩ TopK(score)| / {N_FINAL}")

    res = {k: evaluate_fixed(reqs, k) for k in KS}
    print(f"\n{'K':>7}{'Recall@K':>11}{'Top10收益率':>13}{'收益损失':>10}"
          f"{'精排FLOPs':>13}{'精排延迟(ms)':>14}{'预算内':>8}")
    for k in KS:
        r = res[k]
        ok = "满足" if r["lat"] <= LAT_BUDGET_MS else "超限"
        print(f"{k:>7}{r['recall']:>11.4f}{r['gain']:>13.4f}{1 - r['gain']:>10.4f}"
              f"{r['flops']:>13.2e}{r['lat']:>14.3f}{ok:>8}")

    feasible = [k for k in KS if res[k]["lat"] <= LAT_BUDGET_MS]
    print(f"\n延迟预算 {LAT_BUDGET_MS} ms 内可行的 K:{feasible};K=5000(不截断)"
          f"延迟 {res[5000]['lat']:.1f} ms 已超限。")
    print(f"正文工业取值 300~500:Recall@300={res[300]['recall']:.4f}、"
          f"Recall@500={res[500]['recall']:.4f};相比不截断收益率仅差 "
          f"{res[5000]['gain'] - res[300]['gain']:.4f} / "
          f"{res[5000]['gain'] - res[500]['gain']:.4f},"
          f"FLOPs 却只有 {res[300]['flops'] / res[5000]['flops']:.1%} / "
          f"{res[500]['flops'] / res[5000]['flops']:.1%}。")

    print("\n" + "=" * 108)
    print("边际收益分析:每多花 1e9 FLOPs 能换到多少 Recall")
    print("=" * 108)
    print(f"{'K 区间':>16}{'ΔRecall':>11}{'Δ收益率':>11}{'ΔFLOPs':>13}"
          f"{'ΔRecall/1e9FLOPs':>20}")
    for a, b in zip(KS[:-1], KS[1:]):
        d_rec = res[b]["recall"] - res[a]["recall"]
        d_gain = res[b]["gain"] - res[a]["gain"]
        d_fl = res[b]["flops"] - res[a]["flops"]
        print(f"{f'{a} -> {b}':>16}{d_rec:>11.4f}{d_gain:>11.4f}{d_fl:>13.2e}"
              f"{d_rec / (d_fl / 1e9):>20.4f}")
    print("边际收益单调递减:K 从 50 加到 300 性价比最高,500 之后基本是纯算力支出。")

    print("\n" + "=" * 108)
    print("链路一致性(§5.3)与截断长度(§5.6.3)的折算:σ 越大,同样 Recall 需要越长的 K")
    print("=" * 108)
    sub_ks = (100, 300, 500, 1000)
    print(f"{'σ(粗排噪声)':>14}" + "".join(f"{'K=' + str(k):>11}" for k in sub_ks)
          + f"{'达到0.95所需K':>16}")
    for sigma in (0.004, 0.008, 0.016, 0.032):
        r2 = random.Random(777)
        noisy = [Request(r2, sigma=sigma, tau=0.022) for _ in range(60)]
        row = "".join(f"{evaluate_fixed(noisy, k)['recall']:>11.4f}" for k in sub_ks)
        need = next((k for k in KS if evaluate_fixed(noisy, k)["recall"] >= 0.95), None)
        print(f"{sigma:>14.3f}{row}{(need if need else '>5000'):>16}")
    print("σ 从 0.004 翻到 0.032,为守住 Recall@K≥0.95 所需的截断长度成倍放大 ——")
    print("一致性越差,就必须用更长的候选集(更多精排算力)来兜底。")

    print("\n" + "=" * 108)
    print("自适应截断:同等平均算力下,按请求难度(头部区分度 margin)分配 K")
    print("=" * 108)
    print(f"{'真值离散度 tau':>16}{'请求数':>8}{'平均 margin':>14}"
          f"{'Recall@300':>12}{'需要更大K?':>12}")
    for tau in (0.008, 0.014, 0.022, 0.034):
        grp = [r for r in reqs if r.tau == tau]
        if not grp:
            continue
        m = sum(r.margin for r in grp) / len(grp)
        rc = sum(r.recall_at(300) for r in grp) / len(grp)
        print(f"{tau:>16.3f}{len(grp):>8}{m:>14.5f}{rc:>12.4f}"
              f"{('是' if rc < 0.95 else '否'):>12}")
    print("tau 越小 => margin 越小、固定 K=300 下 Recall 越低 => margin 是难度的可观测代理。")

    ad = evaluate_adaptive(reqs, k_base=300, lam=1.6, k_min=120, k_max=1200)
    f300 = res[300]
    print(f"\n{'策略':>26}{'平均K':>10}{'K范围':>14}{'Recall@K':>11}"
          f"{'Top10收益率':>13}{'平均FLOPs':>13}{'延迟(ms)':>11}")
    print(f"{'固定 K=300':>26}{300:>10}{'300~300':>14}{f300['recall']:>11.4f}"
          f"{f300['gain']:>13.4f}{f300['flops']:>13.2e}{f300['lat']:>11.3f}")
    krange = f"{ad['k_min']}~{ad['k_max']}"
    print(f"{'自适应 K(margin 驱动)':>26}{ad['k_avg']:>10.1f}{krange:>14}"
          f"{ad['recall']:>11.4f}{ad['gain']:>13.4f}{ad['flops']:>13.2e}{ad['lat']:>11.3f}")
    print(f"\n自适应在平均 K={ad['k_avg']:.1f}(与固定 300 同量级、FLOPs "
          f"{ad['flops'] / f300['flops']:.2f}×)下把 Recall 从 {f300['recall']:.4f} "
          f"提到 {ad['recall']:.4f},收益率 {f300['gain']:.4f} -> {ad['gain']:.4f};")
    print("即把算力从「头部区分度高的简单请求」挪给「候选彼此接近的难请求」。")
    print(f"参考上界:不截断(K=5000)Recall={res[5000]['recall']:.4f}、"
          f"收益率={res[5000]['gain']:.4f},但 FLOPs 是自适应方案的 "
          f"{res[5000]['flops'] / ad['flops']:.1f} 倍,延迟 {res[5000]['lat']:.1f} ms 超预算。")
