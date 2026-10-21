"""对应正文第 7 章 §7.5「多样性打散」三种算法(MMR / DPP / SSD)的统一对比。

本实现做了什么
----------------
在**同一个**合成候选集、同一目标序列长度上,把 §7.5.1 的 MMR、§7.5.2 的 DPP 快速贪心
(Cholesky 增量)、§7.5.3 的 SSD(one-step MGS)跑在一起,输出可直接横向比较的数值表:
1. 每个算法在各自权衡系数(lambda / theta / gamma)上扫描,找出「相关性损失最接近 5%」
   的那一档,形成 **等相关性损失下的多样性对比**(避免拿不同强度的参数做不公平比较);
2. 指标口径统一:
   - 相关性损失 = (纯精排 Top-K 的平均融合分 - 本算法的平均融合分) / 纯精排平均融合分
   - ILD      = 平均两两 (1 - cos) ,集合内多样性
   - 类目熵 H = -sum_c p_c log p_c
   - 类目覆盖 = 选中集合出现的类目数
   - 相邻同类 = 序列里相邻两位同类目的对数(正文 §7.6 关心的「连续出现」)
   - log det(S_Y) = DPP 的纯多样性项(行列式体积的对数)
3. 耗时随候选规模 N 的增长曲线:复现正文的复杂度结论
   MMR O(K^2 N)、DPP 快速贪心 O(K^2 N)、SSD O(N T d)。

关键公式(§7.5,三式并列)
--------------------------
    MMR: argmax_i [ lambda * r_i - (1-lambda) * max_{j in S} S_ij ]
    DPP: argmax_i log(d_i^2),  L' = Diag(e^{alpha r}) S Diag(e^{alpha r}),
         alpha = theta / (2(1-theta))
    SSD: argmax_j [ r_j + gamma * ||v_j|| * V ],  v_j <- v_j - (<v_j,v_p>/||v_p||^2) v_p
"""

import math
import random
import time

import numpy as np

CATEGORIES = ["影视", "美食", "萌宠", "游戏", "音乐", "知识"]


# ---------------------------------------------------------------------------
# [MOCK] 合成候选集:同类目物品围绕同一中心,高分物品集中在两个类目
# ---------------------------------------------------------------------------
def build_candidates(n_items, dim, rng):
    centers = {}
    for c in CATEGORIES:
        v = np.array([rng.gauss(0, 1) for _ in range(dim)])
        centers[c] = v / np.linalg.norm(v)
    cats, embs, rels = [], [], []
    for i in range(n_items):
        cat = CATEGORIES[i % len(CATEGORIES)]
        v = centers[cat] + 0.30 * np.array([rng.gauss(0, 1) for _ in range(dim)])
        embs.append(v / np.linalg.norm(v))
        cats.append(cat)
        bonus = 0.25 if cat in ("影视", "美食") else 0.0
        rels.append(min(1.0, max(0.05, 0.5 + 0.3 * rng.gauss(0, 1) + bonus)))
    return cats, np.array(embs), np.array(rels)


def cos_matrix(embs):
    f = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return f @ f.T


# ---------------------------------------------------------------------------
# 三种算法
# ---------------------------------------------------------------------------
def mmr(rels, cos_sim, k, lam):
    n = len(rels)
    sel = [int(np.argmax(rels))]
    rest = [i for i in range(n) if i != sel[0]]
    while len(sel) < k and rest:
        best, best_s = None, -float("inf")
        for i in rest:
            max_sim = max(cos_sim[i][j] for j in sel)
            s = lam * rels[i] - (1.0 - lam) * max_sim
            if s > best_s:
                best, best_s = i, s
        sel.append(best)
        rest.remove(best)
    return sel


def dpp(rels, embs, k, theta):
    n = embs.shape[0]
    S = (1.0 + cos_matrix(embs)) / 2.0
    alpha = 0.0 if theta >= 1.0 else theta / (2.0 * (1.0 - theta))
    w = np.exp(alpha * rels)
    L = w[:, None] * S * w[None, :]
    c = [[] for _ in range(n)]
    d2 = L.diagonal().copy()
    sel = []
    while len(sel) < k:
        mask = np.ones(n, dtype=bool)
        mask[sel] = False
        valid = np.where(mask)[0]
        if valid.size == 0:
            break
        j = int(valid[int(np.argmax(np.maximum(d2[valid], 1e-300)))])
        sel.append(j)
        if len(sel) == k:
            break
        d_j = math.sqrt(max(d2[j], 1e-300))
        cj = np.array(c[j]) if c[j] else None
        for i in range(n):
            if i in sel:
                continue
            dot = float(cj @ np.array(c[i])) if cj is not None and c[i] else 0.0
            e_i = (L[j, i] - dot) / d_j
            c[i].append(e_i)
            d2[i] -= e_i ** 2
    return sel


def ssd(rels, embs, k, gamma):
    emb = np.array(embs, dtype=float, copy=True)
    n = emb.shape[0]
    i_t = int(np.argmax(rels))
    sel = [i_t]
    V = float(np.linalg.norm(emb[i_t]))
    for _ in range(1, k):
        pivot = emb[i_t].copy()
        denom = float(pivot @ pivot)
        if denom > 1e-300:
            for j in range(n):
                if j not in sel:
                    emb[j] = emb[j] - (float(emb[j] @ pivot) / denom) * pivot
        cand = [j for j in range(n) if j not in sel]
        if not cand:
            break
        scores = [rels[j] + float(np.linalg.norm(emb[j])) * gamma * V for j in cand]
        i_t = cand[int(np.argmax(scores))]
        sel.append(i_t)
        V = float(np.linalg.norm(emb[i_t])) * V
    return sel


# ---------------------------------------------------------------------------
# 统一指标
# ---------------------------------------------------------------------------
def logdet(sub):
    if sub.shape[0] == 0:
        return 0.0
    low = np.linalg.cholesky(sub + 1e-10 * np.eye(sub.shape[0]))
    return 2.0 * float(np.sum(np.log(np.diag(low))))


def metrics(cats, embs, rels, S, picked):
    m = len(picked)
    sims = cos_matrix(embs)[np.ix_(picked, picked)]
    ild = sum(1.0 - sims[x, y] for x in range(m) for y in range(x + 1, m))
    ild /= m * (m - 1) / 2
    sc = [cats[i] for i in picked]
    counts = {c: sc.count(c) for c in set(sc)}
    ent = -sum((v / m) * math.log(v / m) for v in counts.values())
    return {
        "rel": float(np.mean(rels[picked])),
        "ild": ild,
        "n_cat": len(counts),
        "entropy": ent,
        "adj_same": sum(1 for t in range(m - 1) if sc[t] == sc[t + 1]),
        "logdet_S": logdet(S[np.ix_(picked, picked)]),
    }


def main():
    random.seed(42)
    np.random.seed(42)

    n_items, dim, k = 60, 16, 12
    cats, embs, rels = build_candidates(n_items, dim, random.Random(42))
    S = (1.0 + cos_matrix(embs)) / 2.0
    cs = cos_matrix(embs)

    top_rel = list(np.argsort(-rels)[:k])
    base = metrics(cats, embs, rels, S, top_rel)
    print(f"== 统一候选集:N={n_items}, d={dim}, K={k},类目数={len(CATEGORIES)} ==")
    print("基线 = 纯精排 Top-K(不打散)")
    print(f"  avg_rel={base['rel']:.4f}  ILD={base['ild']:.4f}  "
          f"类目数={base['n_cat']}  类目熵={base['entropy']:.4f}  "
          f"相邻同类={base['adj_same']}  log det S_Y={base['logdet_S']:.4f}")

    # ---------------- (1) 各算法完整参数扫描 ----------------
    sweeps = {
        "MMR": [("lambda", lam, lambda lam=lam: mmr(rels, cs, k, lam))
                for lam in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0)],
        "DPP": [("theta", th, lambda th=th: dpp(rels, embs, k, th))
                for th in (0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.0)],
        "SSD": [("gamma", g, lambda g=g: ssd(rels, embs, k, g))
                for g in (0.0, 1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 50.0)],
    }

    records = {}
    for algo, entries in sweeps.items():
        print(f"\n== (1.{list(sweeps).index(algo) + 1}) {algo} 参数扫描 ==")
        print(f"{entries[0][0]:>8} {'avg_rel':>8} {'rel损失':>8} {'ILD':>7} "
              f"{'类目数':>6} {'类目熵':>7} {'相邻同类':>8} {'log det S_Y':>12} "
              f"{'耗时ms':>8}")
        rows = []
        for pname, pval, fn in entries:
            t0 = time.perf_counter()
            picked = fn()
            dt = (time.perf_counter() - t0) * 1000
            m = metrics(cats, embs, rels, S, picked)
            loss = (base["rel"] - m["rel"]) / base["rel"] * 100
            rows.append((pval, loss, m, dt, picked))
            print(f"{pval:>8.2f} {m['rel']:>8.4f} {loss:>7.2f}% {m['ild']:>7.4f} "
                  f"{m['n_cat']:>6d} {m['entropy']:>7.4f} {m['adj_same']:>8d} "
                  f"{m['logdet_S']:>12.4f} {dt:>8.2f}")
        records[algo] = rows

    # ---------------- (2) 等相关性损失(≈5%)下的横向对比 ----------------
    print("\n== (2) 等相关性损失对比(每个算法取 rel 损失最接近 5% 的那一档)==")
    target = 5.0
    print(f"{'算法':>6} {'参数':>14} {'rel损失':>8} {'ILD':>7} {'类目数':>6} "
          f"{'类目熵':>7} {'相邻同类':>8} {'log det S_Y':>12} {'耗时ms':>8}")
    chosen = {}
    for algo, rows in records.items():
        pname = sweeps[algo][0][0]
        pval, loss, m, dt, picked = min(rows, key=lambda r: abs(r[1] - target))
        chosen[algo] = picked
        print(f"{algo:>6} {pname + '=' + format(pval, '.2f'):>14} {loss:>7.2f}% "
              f"{m['ild']:>7.4f} {m['n_cat']:>6d} {m['entropy']:>7.4f} "
              f"{m['adj_same']:>8d} {m['logdet_S']:>12.4f} {dt:>8.2f}")

    print("\n该档位下三者选出的类目序列:")
    print(f"  {'精排基线':>8}: " + "->".join(cats[i] for i in top_rel))
    for algo, picked in chosen.items():
        print(f"  {algo:>8}: " + "->".join(cats[i] for i in picked))

    print("\n选中集合的重叠情况(Jaccard):")
    names = list(chosen)
    print(f"{'':>8} " + " ".join(f"{n:>8}" for n in names + ["基线"]))
    for a in names:
        cells = []
        for b in names + ["基线"]:
            sb = set(top_rel) if b == "基线" else set(chosen[b])
            sa = set(chosen[a])
            cells.append(len(sa & sb) / len(sa | sb))
        print(f"{a:>8} " + " ".join(f"{c:>8.3f}" for c in cells))

    # ---------------- (3) 耗时随 N 增长 ----------------
    print("\n== (3) 耗时随候选规模 N 增长(K=12,单位 ms)==")
    print(f"{'N':>7} {'MMR':>9} {'DPP':>9} {'SSD':>9}  复杂度: MMR/DPP O(K^2 N), SSD O(NTd)")
    for n_probe in (60, 120, 240, 480, 960):
        c_p, e_p, r_p = build_candidates(n_probe, dim, random.Random(7))
        cs_p = cos_matrix(e_p)
        t0 = time.perf_counter(); mmr(r_p, cs_p, k, 0.8); t_m = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); dpp(r_p, e_p, k, 0.9); t_d = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); ssd(r_p, e_p, k, 5.0); t_s = (time.perf_counter() - t0) * 1000
        print(f"{n_probe:>7} {t_m:>9.2f} {t_d:>9.2f} {t_s:>9.2f}")

    print("\n== (4) 三者定位小结(对应正文 §7.5)==")
    print("  MMR:只看「与已选集合的最大相似度」,实现最简单,但对集合整体结构不敏感;")
    print("  DPP:用 det(L_Y)=Vol^2 度量集合整体多样性,理论最完备,")
    print("       Cholesky 增量把贪心推断降到 O(K^2 N);")
    print("  SSD:用 one-step MGS 的残差模长近似「新增体积」,单步 O(d),")
    print("       整体 O(NTd),最适合长序列信息流场景。")


if __name__ == "__main__":
    main()
