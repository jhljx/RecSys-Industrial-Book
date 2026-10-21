"""对应正文第 7 章 §7.5.3「SSD 算法」(Sliding Spectrum Decomposition,滑动谱分解)。

本实现做了什么
----------------
1. Gram-Schmidt / MGS / one-step MGS 三者的对照实验:
   在一个人为构造的高条件数矩阵上比较正交性误差 max|E^T E - I|,
   复现正文「GS 在浮点下正交性显著退化,MGS 把误差限制在局部」的结论;
2. 复刻正文 code_ssd 的 `ssd_without_sliding_window`:贪心选择,
   每轮对所有未选候选做一步 MGS 投影扣除,按 r_j + gamma * ||v_j|| * V 选下一个,
   并把体积更新为 V <- ||v_{i_t}|| * V。为避免正文代码就地改写入参,
   本实现对 embeddings 做了一份拷贝(数值逻辑与正文完全相同);
3. 实现带滑动窗口的变体:正文把序列划分成 L = T - W + 1 个窗口并堆叠成
   张量 X ∈ R^{L x W x d};本脚本把 one-step MGS 的基限制在最近 W-1 个已选物品上,
   即「窗口内正交化」,并与全序列版对照——这正是正文强调的
   「窗口内 DPP 丢弃窗外历史 vs SSD 纳入完整序列」的差异;
4. 打印 gamma 扫描表与「序列体积谱」:对每个窗口的 W x d 矩阵算奇异值,
   给出 sum sigma(窗口奇异值之和,即正文目标函数的多样性项)与 log Vol。

关键公式(§7.5.3)
-----------------
    X = sum_{sigma_ijk > 0} sigma_ijk u_i^(1) ⊗ u_j^(2) ⊗ u_k^(3)      (高阶 SVD)
    max_{i_1..i_T} sum_t r_{i_t} + gamma * sum_{sigma_ijk>0} sigma_ijk
    one-step MGS:  v_j <- v_j - (<v_j, v_{i_{t-1}}> / ||v_{i_{t-1}}||^2) v_{i_{t-1}}
    贪心打分:      score_j = r_j + gamma * ||v_j|| * V ,  V <- ||v_{i_t}|| * V

复杂度(正文结论):one-step MGS 单次 O(d),整体 O(N T d),优于 DPP 贪心的 O(N^2 d)。
"""

import math
import random
import time

import numpy as np

CATEGORIES = ["影视", "美食", "萌宠", "游戏", "音乐", "知识"]


# ---------------------------------------------------------------------------
# (1) GS / MGS / one-step MGS 正交性对照
# ---------------------------------------------------------------------------
def classical_gs(vectors):
    """正文公式 GS_iter:u_m = v_m - sum_{p<m} (v_m^T e_p) e_p。"""
    basis = []
    for v in vectors:
        u = v.copy()
        for e in basis:                       # 一次性扣除所有历史投影
            u = u - float(v @ e) * e          # 注意用的是原始 v,这是 GS 不稳定的根源
        n = np.linalg.norm(u)
        basis.append(u / n if n > 1e-14 else u)
    return np.array(basis)


def modified_gs(vectors):
    """MGS:逐步正交化,每生成一个 e_{k-1} 立刻更新所有后续向量。"""
    work = [v.copy() for v in vectors]
    basis = []
    for k in range(len(work)):
        n = np.linalg.norm(work[k])
        e = work[k] / n if n > 1e-14 else work[k]
        basis.append(e)
        for j in range(k + 1, len(work)):     # 立即作用于剩余向量,误差不向后放大
            work[j] = work[j] - float(work[j] @ e) * e
    return np.array(basis)


def one_step_mgs(vectors):
    """SSD 的 one-step MGS:只对「最新加入的那个向量」做一次投影扣除。"""
    work = [v.copy() for v in vectors]
    basis = []
    for k in range(len(work)):
        n = np.linalg.norm(work[k])
        e = work[k] / n if n > 1e-14 else work[k]
        basis.append(e)
        latest = work[k]
        denom = float(latest @ latest)
        if denom <= 1e-300:
            continue
        for j in range(k + 1, len(work)):
            work[j] = work[j] - (float(work[j] @ latest) / denom) * latest
    return np.array(basis)


def orthogonality_error(basis):
    g = basis @ basis.T
    return float(np.max(np.abs(g - np.eye(g.shape[0]))))


def demo_orthogonalization():
    print("== (1) GS / MGS / one-step MGS 的正交性与耗时 ==")
    rng = np.random.RandomState(42)
    print(f"{'矩阵':>22} {'条件数':>12} {'GS误差':>12} {'MGS误差':>12} "
          f"{'1step误差':>12}")
    cases = []
    # 良态矩阵
    cases.append(("良态随机 (24x24)", rng.randn(24, 24)))
    # 病态矩阵:奇异值按几何级数衰减,条件数可控(避免数值秩亏导致三者都饱和)
    k, dk, cond_target = 16, 16, 1e10
    q1, _ = np.linalg.qr(rng.randn(k, k))
    q2, _ = np.linalg.qr(rng.randn(dk, dk))
    sv = np.array([cond_target ** (-i / (k - 1)) for i in range(k)])
    cases.append((f"病态 cond=1e10 ({k}x{dk})", q1 @ np.diag(sv) @ q2[:k].T))
    # 近线性相关:每个向量都是前一个的微小扰动
    m, d = 16, 16
    base = rng.randn(1, d)
    rows = [base[0]]
    for _ in range(m - 1):
        rows.append(rows[-1] + 1e-6 * rng.randn(d))
    cases.append((f"近线性相关 ({m}x{d})", np.array(rows)))

    for name, mat in cases:
        cond = np.linalg.cond(mat)
        e_gs = orthogonality_error(classical_gs(mat))
        e_mgs = orthogonality_error(modified_gs(mat))
        e_one = orthogonality_error(one_step_mgs(mat))
        print(f"{name:>22} {cond:>12.3e} {e_gs:>12.3e} {e_mgs:>12.3e} "
              f"{e_one:>12.3e}")
    print("→ 病态/近线性相关时 GS 的正交性误差比 MGS 大若干数量级;")
    print("  one-step MGS 只扣除最新向量的分量,本就不追求全局正交(误差最大),")
    print("  它换来的是 O(d) 的单步开销与「局部时序多样性」的语义。")


# ---------------------------------------------------------------------------
# [MOCK] 合成候选集
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


# ---------------------------------------------------------------------------
# (2) 正文 code_ssd:无滑动窗口版本
# ---------------------------------------------------------------------------
def ssd_without_sliding_window(r, embeddings, T, gamma):
    """复刻正文 code_ssd。embeddings 内部做拷贝,不污染调用方数据。"""
    emb = np.array(embeddings, dtype=float, copy=True)
    n, _ = emb.shape
    selected = []
    i_t = int(np.argmax(r))
    selected.append(i_t)
    V = float(np.linalg.norm(emb[i_t]))

    for _ in range(1, T):
        pivot = emb[i_t].copy()
        denom = float(pivot @ pivot)
        if denom > 1e-300:
            for j in range(n):
                if j not in selected:
                    emb[j] = emb[j] - (float(emb[j] @ pivot) / denom) * pivot
        candidates = [j for j in range(n) if j not in selected]
        if not candidates:
            break
        scores = [r[j] + float(np.linalg.norm(emb[j])) * gamma * V for j in candidates]
        i_t = candidates[int(np.argmax(scores))]
        selected.append(i_t)
        V = float(np.linalg.norm(emb[i_t])) * V
    return selected


# ---------------------------------------------------------------------------
# (3) 滑动窗口变体:one-step MGS 的基只保留最近 W-1 个已选物品
# ---------------------------------------------------------------------------
def ssd_with_sliding_window(r, embeddings, T, gamma, window):
    """窗口内正交化:对候选 j,只扣除其在最近 window-1 个已选物品上的分量。

    正文把序列划成 L = T - W + 1 个窗口并堆成张量 X ∈ R^{L x W x d};
    贪心实现上等价于「只让窗口内的已选物品参与正交化」,窗外历史不再压制候选。
    """
    raw = np.array(embeddings, dtype=float, copy=True)
    n, _ = raw.shape
    selected = [int(np.argmax(r))]
    V = float(np.linalg.norm(raw[selected[0]]))

    for _ in range(1, T):
        recent = selected[-(window - 1):] if window > 1 else []
        residual = raw.copy()
        for p in recent:                       # 逐个扣除窗口内已选物品的分量
            pivot = residual[p].copy()
            denom = float(pivot @ pivot)
            if denom <= 1e-300:
                continue
            for j in range(n):
                if j not in selected:
                    residual[j] = residual[j] - (float(residual[j] @ pivot) / denom) * pivot
        candidates = [j for j in range(n) if j not in selected]
        if not candidates:
            break
        scores = [r[j] + float(np.linalg.norm(residual[j])) * gamma * V
                  for j in candidates]
        nxt = candidates[int(np.argmax(scores))]
        selected.append(nxt)
        V = float(np.linalg.norm(residual[nxt])) * V
    return selected


# ---------------------------------------------------------------------------
# 指标:窗口奇异值之和(正文目标函数的多样性项)+ 常规多样性指标
# ---------------------------------------------------------------------------
def window_spectrum(embeddings, picked, window):
    """按 §7.5.3 把序列切成 L = T - W + 1 个窗口,返回每个窗口的奇异值之和。"""
    seq = embeddings[picked]
    T = len(picked)
    sums = []
    for start in range(max(1, T - window + 1)):
        block = seq[start:start + window]
        sv = np.linalg.svd(block, compute_uv=False)
        sums.append(float(np.sum(sv)))
    return sums


def seq_metrics(cats, embeddings, r, picked, window):
    m = len(picked)
    sel = embeddings[picked]
    sims = sel @ sel.T
    ild = sum(1.0 - sims[x, y] for x in range(m) for y in range(x + 1, m))
    ild /= m * (m - 1) / 2
    sc = [cats[i] for i in picked]
    counts = {c: sc.count(c) for c in set(sc)}
    ent = -sum((v / m) * math.log(v / m) for v in counts.values())
    adj = sum(1 for k in range(m - 1) if sc[k] == sc[k + 1])
    spec = window_spectrum(embeddings, picked, window)
    # 窗口内平均 log 体积(用奇异值乘积),反映正文「序列体积」
    logvols = []
    for start in range(max(1, m - window + 1)):
        sv = np.linalg.svd(embeddings[picked][start:start + window],
                           compute_uv=False)
        logvols.append(float(np.sum(np.log(np.maximum(sv, 1e-12)))))
    return {
        "rel": float(np.mean(r[picked])),
        "ild": ild,
        "n_cat": len(counts),
        "entropy": ent,
        "adj_same": adj,
        "sum_sigma": float(np.mean(spec)),
        "log_vol": float(np.mean(logvols)),
    }


def main():
    random.seed(42)
    np.random.seed(42)

    demo_orthogonalization()

    rng = random.Random(42)
    n_items, dim, T, window = 60, 16, 12, 5
    cats, embs, rels = build_candidates(n_items, dim, rng)

    print(f"\n== (2) SSD 贪心选择(正文 code_ssd),N={n_items}, d={dim}, "
          f"T={T}, 窗口 W={window} ==")
    top_rel = list(np.argsort(-rels)[:T])
    base = seq_metrics(cats, embs, rels, top_rel, window)
    print(f"{'策略':>18} {'avg_rel':>8} {'rel损失':>8} {'ILD':>7} "
          f"{'类目数':>6} {'类目熵':>7} {'相邻同类':>8} {'sum_sigma':>10} {'log_vol':>9}")
    print(f"{'纯相关性 Top-T':>18} {base['rel']:>8.4f} {0.0:>7.2f}% "
          f"{base['ild']:>7.4f} {base['n_cat']:>6d} {base['entropy']:>7.4f} "
          f"{base['adj_same']:>8d} {base['sum_sigma']:>10.4f} {base['log_vol']:>9.4f}")
    for gamma in (0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        picked = ssd_without_sliding_window(rels, embs, T, gamma)
        m = seq_metrics(cats, embs, rels, picked, window)
        loss = (base["rel"] - m["rel"]) / base["rel"] * 100
        print(f"{'SSD gamma=' + format(gamma, '.1f'):>18} {m['rel']:>8.4f} "
              f"{loss:>7.2f}% {m['ild']:>7.4f} {m['n_cat']:>6d} "
              f"{m['entropy']:>7.4f} {m['adj_same']:>8d} {m['sum_sigma']:>10.4f} "
              f"{m['log_vol']:>9.4f}")

    print("\n== (3) 全序列 one-step MGS vs 滑动窗口内正交化 ==")
    gamma = 10.0
    full = ssd_without_sliding_window(rels, embs, T, gamma)
    print(f"{'变体':>20} {'avg_rel':>8} {'ILD':>7} {'类目熵':>7} "
          f"{'相邻同类':>8} {'sum_sigma':>10}")
    mf = seq_metrics(cats, embs, rels, full, window)
    print(f"{'全序列(W=T)':>20} {mf['rel']:>8.4f} {mf['ild']:>7.4f} "
          f"{mf['entropy']:>7.4f} {mf['adj_same']:>8d} {mf['sum_sigma']:>10.4f}")
    for w in (2, 3, 5, 8):
        picked = ssd_with_sliding_window(rels, embs, T, gamma, w)
        m = seq_metrics(cats, embs, rels, picked, window)
        print(f"{'滑动窗口 W=' + str(w):>20} {m['rel']:>8.4f} {m['ild']:>7.4f} "
              f"{m['entropy']:>7.4f} {m['adj_same']:>8d} {m['sum_sigma']:>10.4f}")
    print("→ 全序列版把所有已选物品的分量都扣掉,候选残差模长压得最狠,"
          "因此多样性项占主导、相关性损失最大;")
    print("  窗口越小,被压制的历史越少,相关性保留更多,但窗口外的长程重复不再受约束。")

    print(f"\n== (4) 选中序列的类目分布(gamma={gamma:.1f})==")
    print(f"  {'纯相关性':>12}: " + "->".join(cats[i] for i in top_rel))
    print(f"  {'SSD 全序列':>12}: " + "->".join(cats[i] for i in full))
    print(f"  {'SSD W=3':>12}: "
          + "->".join(cats[i] for i in ssd_with_sliding_window(rels, embs, T, gamma, 3)))

    print("\n== (5) 逐窗口奇异谱(SSD 全序列,W=5)==")
    seq = embs[full]
    print(f"{'窗口':>6} " + " ".join(f"{'s' + str(i + 1):>8}" for i in range(window))
          + f" {'sum':>9}")
    for start in range(T - window + 1):
        sv = np.linalg.svd(seq[start:start + window], compute_uv=False)
        print(f"{start:>6} " + " ".join(f"{v:>8.4f}" for v in sv)
              + f" {np.sum(sv):>9.4f}")

    print("\n== (6) 复杂度验证:SSD 耗时随 N 近似线性 ==")
    print(f"{'N':>7} {'T':>4} {'耗时(ms)':>10} {'ms/N':>9}")
    for n_probe in (200, 400, 800, 1600):
        _, e_p, r_p = build_candidates(n_probe, dim, random.Random(7))
        t0 = time.perf_counter()
        ssd_without_sliding_window(r_p, e_p, T, gamma)
        dt = (time.perf_counter() - t0) * 1000
        print(f"{n_probe:>7} {T:>4} {dt:>10.2f} {dt / n_probe:>9.5f}")
    print("→ 单次 one-step MGS 为 O(d),整体 O(N T d),ms/N 基本不随 N 增长。")


if __name__ == "__main__":
    main()
