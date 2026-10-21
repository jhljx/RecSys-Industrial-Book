"""对应正文第 7 章 §7.5.2「DPP 算法」(Determinantal Point Process,行列式点过程)。

本实现做了什么
----------------
1. 按正文公式构造核矩阵:B_i = r_i f_i,||f_i||_2 = 1,
       S_ij = (1 + <f_i, f_j>) / 2        (把余弦从 [-1,1] 压到 [0,1] 保半正定)
       L  = Diag(r) · S · Diag(r)
       L' = Diag(exp(alpha r)) · S · Diag(exp(alpha r)),alpha = theta / (2(1-theta))
   并数值验证 L 的半正定性(最小特征值 >= 0)与正文公式 log_det_L 的分解恒等式;
2. 实现两种贪心 MAP 推断,二者选出的集合必须完全一致:
   (a) 朴素版:每轮对每个候选显式构造子矩阵 L_{Y∪{i}} 并做一次 O(k^3) 的
       Cholesky 分解求 log det,复杂度 O(K^2 N k^3);
   (b) 快速版:复刻正文 code_dpp 的增量递推(Chen et al., NeurIPS 2018),
       维护 c_i 与 d_i^2,每轮只做 O(N k) 的更新:
           e_i    = (L_ji - <c_j, c_i>) / d_j
           c_i'   = [c_i, e_i]
           d_i'^2 = d_i^2 - e_i^2
           j      = argmax_i log(d_i^2)
   打印两者耗时与加速比;
3. 几何解释(正文图 7.6):det(L_Y) = Vol^2({B_i}_{i in Y}),
   打印选中集合的 log det(L_Y)、log det(S_Y)(纯多样性项)与超平行体体积;
4. theta 扫描:theta=1 退化为纯相关性(alpha 置 0 时 L' 的相关性权重消失,
   正文代码对 theta==1 令 alpha=0),theta 越小多样性权重越大。
   打印平均相关性 / 相关性损失 / log det(S_Y) / ILD / 类目熵。

关键公式(§7.5.2)
-----------------
    P(Y) ∝ det(L_Y) = Vol^2({B_i}_{i in Y})
    log det L = sum_i log(r_i^2) + log det(S)
    log det L' = lambda sum_i log(r_i^2) + (1-lambda) log det(S)
    j = argmax_{i in Z\\Y} [log det(L_{Y∪{i}}) - log det(L_Y)] = argmax_i log(d_i^2)

复杂度(正文结论):sum_{k=1..K} O(k(N-k)) = O(K^2 N),选满 N 个时为 O(N^3)。
"""

import math
import random
import time

import numpy as np

CATEGORIES = ["影视", "美食", "萌宠", "游戏", "音乐", "知识"]


# ---------------------------------------------------------------------------
# [MOCK] 合成候选集:同类目物品 embedding 围绕同一类目中心,因此天然高相似
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
# 核矩阵构造
# ---------------------------------------------------------------------------
def similarity_matrix(embeddings):
    """S_ij = (1 + <f_i, f_j>) / 2,f 已 L2 归一化。"""
    f = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (1.0 + f @ f.T) / 2.0


def kernel_matrix(embeddings, r, theta):
    """L' = Diag(exp(alpha r)) · S · Diag(exp(alpha r)),alpha = theta/(2(1-theta))。"""
    S = similarity_matrix(embeddings)
    alpha = 0.0 if theta >= 1.0 else theta / (2.0 * (1.0 - theta))
    w = np.exp(alpha * r)
    return w[:, None] * S * w[None, :], S, alpha


# ---------------------------------------------------------------------------
# (a) 朴素贪心:每个候选显式算 log det(L_{Y∪{i}})
# ---------------------------------------------------------------------------
def greedy_map_naive(L, K):
    n = L.shape[0]
    selected = []
    for _ in range(K):
        best_j, best_gain = None, -float("inf")
        base = logdet_cholesky(L[np.ix_(selected, selected)]) if selected else 0.0
        for i in range(n):
            if i in selected:
                continue
            idx = selected + [i]
            gain = logdet_cholesky(L[np.ix_(idx, idx)]) - base
            if gain > best_gain:
                best_j, best_gain = i, gain
        selected.append(best_j)
    return selected


def logdet_cholesky(sub):
    """用 Cholesky 分解算 log det,比直接求行列式数值更稳。"""
    if sub.shape[0] == 0:
        return 0.0
    jitter = 1e-10 * np.eye(sub.shape[0])
    low = np.linalg.cholesky(sub + jitter)
    return 2.0 * float(np.sum(np.log(np.diag(low))))


# ---------------------------------------------------------------------------
# (b) 快速贪心:正文 code_dpp 的 Cholesky 增量递推
# ---------------------------------------------------------------------------
def fast_greedy_map_inference(item_embeddings, r_u, theta, K):
    """复刻正文 code_dpp。返回选中物品下标列表(长度 K)。"""
    n = item_embeddings.shape[0]
    if K <= 0 or K > n:
        raise ValueError(f"K 必须为正整数且不超过物品总数 {n}")

    L, _, _ = kernel_matrix(item_embeddings, r_u, theta)

    c = [[] for _ in range(n)]          # c[i]:物品 i 的辅助向量,长度 = 已选个数
    d_squared = L.diagonal().copy()     # d_i^2 初始为 L 的对角元
    selected = []

    while len(selected) < K:
        mask = np.ones(n, dtype=bool)
        mask[selected] = False
        valid = np.where(mask)[0]
        if valid.size == 0:
            break
        # argmax log(d_i^2) 等价于 argmax d_i^2(log 单调),这里按正文写法保留 log
        safe = np.maximum(d_squared[valid], 1e-300)
        j = int(valid[int(np.argmax(np.log(safe)))])
        selected.append(j)
        if len(selected) == K:
            break

        d_j = math.sqrt(max(d_squared[j], 1e-300))
        cj = np.array(c[j]) if c[j] else None
        for i in range(n):
            if i in selected:
                continue
            dot = float(cj @ np.array(c[i])) if cj is not None and c[i] else 0.0
            e_i = (L[j, i] - dot) / d_j
            c[i].append(e_i)
            d_squared[i] -= e_i ** 2
    return selected


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def set_metrics(cats, embeddings, r, S, picked):
    m = len(picked)
    sel_emb = embeddings[picked]
    sims = sel_emb @ sel_emb.T
    ild = 0.0
    for x in range(m):
        for y in range(x + 1, m):
            ild += 1.0 - sims[x, y]
    ild /= m * (m - 1) / 2
    sel_cats = [cats[i] for i in picked]
    counts = {c: sel_cats.count(c) for c in set(sel_cats)}
    ent = -sum((v / m) * math.log(v / m) for v in counts.values())
    return {
        "rel": float(np.mean(r[picked])),
        "ild": ild,
        "n_cat": len(counts),
        "entropy": ent,
        "logdet_S": logdet_cholesky(S[np.ix_(picked, picked)]),
    }


def main():
    random.seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    n_items, dim, top_k = 60, 16, 10
    cats, embs, rels = build_candidates(n_items, dim, rng)

    # ---------------- (1) 核矩阵性质验证 ----------------
    print("== (1) 核矩阵构造与性质验证 ==")
    S = similarity_matrix(embs)
    L_plain = rels[:, None] * S * rels[None, :]          # L = Diag(r) S Diag(r)
    eig_S = np.linalg.eigvalsh(S)
    eig_L = np.linalg.eigvalsh(L_plain)
    print(f"候选集 N={n_items},emb 维度 d={dim},目标长度 K={top_k}")
    print(f"S 取值范围           : [{S.min():.4f}, {S.max():.4f}]  (已压到 [0,1])")
    print(f"S 最小特征值         : {eig_S.min():+.3e}   (>= 0 → 半正定)")
    print(f"L 最小特征值         : {eig_L.min():+.3e}   (>= 0 → 半正定)")
    sub = list(range(8))
    lhs = logdet_cholesky(L_plain[np.ix_(sub, sub)])
    rhs = float(np.sum(np.log(rels[sub] ** 2))) + logdet_cholesky(S[np.ix_(sub, sub)])
    print("正文公式 log det L = sum_i log(r_i^2) + log det S 验证(取前 8 个物品子阵):")
    print(f"  左边 log det(L_Y) = {lhs:+.6f}   右边 = {rhs:+.6f}   "
          f"|差| = {abs(lhs - rhs):.3e}")

    # ---------------- (2) 两种贪心的一致性与加速比 ----------------
    print("\n== (2) 贪心 MAP 推断:朴素实现 vs Cholesky 增量实现 ==")
    theta = 0.7
    L, _, alpha = kernel_matrix(embs, rels, theta)
    print(f"theta={theta}  →  alpha = theta/(2(1-theta)) = {alpha:.4f}")

    t0 = time.perf_counter()
    naive = greedy_map_naive(L, top_k)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = fast_greedy_map_inference(embs, rels, theta, top_k)
    t_fast = time.perf_counter() - t0

    print(f"朴素贪心选中: {naive}")
    print(f"快速贪心选中: {fast}")
    print(f"两者是否一致: {naive == fast}")
    print(f"朴素耗时 {t_naive * 1000:.2f} ms,快速耗时 {t_fast * 1000:.2f} ms,"
          f"加速比 {t_naive / t_fast:.2f}x")
    print("复杂度:朴素每轮要对 N-k 个候选各做一次 O(k^3) Cholesky;"
          "快速版每轮只做 O(N k) 的 c/d 递推 → 总 O(K^2 N)")

    print("\n不同规模下的耗时对比(单位 ms):")
    print(f"{'N':>6} {'K':>4} {'朴素':>10} {'增量':>10} {'加速比':>8} {'集合一致':>8}")
    for n_probe, k_probe in ((40, 8), (80, 10), (120, 12), (200, 15)):
        cat_p, emb_p, rel_p = build_candidates(n_probe, dim, random.Random(7))
        L_p, _, _ = kernel_matrix(emb_p, rel_p, theta)
        t0 = time.perf_counter()
        a_p = greedy_map_naive(L_p, k_probe)
        t_a = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        b_p = fast_greedy_map_inference(emb_p, rel_p, theta, k_probe)
        t_b = (time.perf_counter() - t0) * 1000
        print(f"{n_probe:>6} {k_probe:>4} {t_a:>10.2f} {t_b:>10.2f} "
              f"{t_a / t_b:>7.2f}x {str(a_p == b_p):>8}")

    # 增量 d_i^2 与真实边际增益的一致性(论证递推正确)
    print("\n增量量 log(d_i^2) 与真实边际增益 log det(L_{Y∪{i}}) - log det(L_Y) 对照"
          "(已选前 3 个,取前 5 个未选候选):")
    Y = fast[:3]
    base = logdet_cholesky(L[np.ix_(Y, Y)])
    # 重放递推拿到第 4 轮的 d_i^2
    c = [[] for _ in range(n_items)]
    d2 = L.diagonal().copy()
    for step, j in enumerate(Y):
        d_j = math.sqrt(d2[j])
        cj = np.array(c[j]) if c[j] else None
        for i in range(n_items):
            if i in Y[:step + 1]:
                continue
            dot = float(cj @ np.array(c[i])) if cj is not None and c[i] else 0.0
            e_i = (L[j, i] - dot) / d_j
            c[i].append(e_i)
            d2[i] -= e_i ** 2
    probe = [i for i in range(n_items) if i not in Y][:5]
    print(f"{'item':>5} {'log(d_i^2)':>12} {'真实边际增益':>14} {'|差|':>10}")
    for i in probe:
        idx = Y + [i]
        true_gain = logdet_cholesky(L[np.ix_(idx, idx)]) - base
        print(f"{i:>5} {math.log(d2[i]):>12.6f} {true_gain:>14.6f} "
              f"{abs(math.log(d2[i]) - true_gain):>10.2e}")

    # ---------------- (3) 几何解释:体积 ----------------
    print("\n== (3) 几何解释 det(L_Y) = Vol^2({B_i}) ==")
    print(f"{'集合':>14} {'log det(L_Y)':>13} {'Vol':>12} {'log det(S_Y)':>13}")
    top_rel = list(np.argsort(-rels)[:top_k])
    for name, picked in (("纯相关性 Top-K", top_rel), ("DPP 贪心", fast)):
        ld = logdet_cholesky(L[np.ix_(picked, picked)])
        vol = math.exp(0.5 * ld)
        print(f"{name:>14} {ld:>13.4f} {vol:>12.4e} "
              f"{logdet_cholesky(S[np.ix_(picked, picked)]):>13.4f}")
    print("→ DPP 选出的集合张成的超平行体体积更大,即物品之间更「排斥」。")

    # ---------------- (4) theta 扫描 ----------------
    print("\n== (4) theta 扫描:相关性-多样性权衡 ==")
    base_rel = float(np.mean(rels[top_rel]))
    print(f"{'theta':>6} {'alpha':>8} {'avg_rel':>8} {'rel损失':>8} "
          f"{'log det S_Y':>12} {'ILD':>7} {'类目数':>6} {'类目熵':>7}")
    for theta in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1.0):
        picked = fast_greedy_map_inference(embs, rels, theta, top_k)
        _, _, a = kernel_matrix(embs, rels, theta)
        m = set_metrics(cats, embs, rels, S, picked)
        loss = (base_rel - m["rel"]) / base_rel * 100
        print(f"{theta:>6.2f} {a:>8.3f} {m['rel']:>8.4f} {loss:>7.2f}% "
              f"{m['logdet_S']:>12.4f} {m['ild']:>7.4f} {m['n_cat']:>6d} "
              f"{m['entropy']:>7.4f}")
    print("注:theta=1 时正文代码令 alpha=0,此时 L'=S,相关性被完全抹掉,"
          "结果反而是「纯多样性」;真正的纯相关性需令 alpha→∞(见 theta=0.99)。")

    print("\n== (5) 选中序列的类目分布 ==")
    for name, picked in (("纯相关性 Top-K", top_rel),
                         ("DPP theta=0.7", fast)):
        print(f"  {name:>14}: " + "->".join(cats[i] for i in picked))


if __name__ == "__main__":
    main()
