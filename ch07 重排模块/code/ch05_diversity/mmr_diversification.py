"""对应正文第 7 章 §7.5.1「MMR 算法」(Maximal Marginal Relevance,最大边际相关性)。

本实现做了什么
----------------
1. 严格按正文公式 (mmr_equation) 实现 MMR 贪心选择:首轮只按相关性选,之后每轮
   对 R \\ S 中每个候选计算 MMR 分,取最大者加入 S,直到 |S| = M;
2. 复刻正文 code_mmr_ir / code_mmr_ir_main 的「信息检索」用法:对正文给出的 7 篇
   中文文档(其中第 2/4/7 篇分别与第 1/3/6 篇高度相似)用 lambda=0.6 选 3 篇。
   正文代码用 sklearn 的 TfidfVectorizer,本仓库不允许该依赖,这里手写
   「字符 1-gram + 2-gram」TF-IDF 向量化 + 余弦相似度,逻辑与正文一致;
3. 复刻正文给出的「推荐系统重排」用法:Sim_1 直接取精排/重排融合分 r_i,
   Sim_2 取 item embedding 的余弦相似度;
4. 做 lambda 从 0.0 到 1.0 的扫描,打印相关性-多样性权衡的数值表
   (平均相关性 / 相关性损失 / 最大成对相似度 / ILD / 类目数 / 类目熵 / 相邻同类目对数),
   直观展示 lambda=1 退化为纯精排序、lambda=0 退化为纯去冗余。

关键公式(§7.5.1,公式 mmr_equation)
-------------------------------------
    MMR = argmax_{D_i in R \\ S} [ lambda * Sim_1(D_i, Q)
                                   - (1 - lambda) * max_{D_j in S} Sim_2(D_i, D_j) ]

说明:正文公式里的括号把 (1-lambda) 项写在了 lambda 的括号内部,按其文字描述
「lambda 用于平衡边际相关性与相关性创新度」以及正文代码
`mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim`,
本脚本采用上式(与正文代码一致)。

评价指标
--------
    ILD  = 2 / (M(M-1)) * sum_{i<j} (1 - cos(f_i, f_j))      集合内平均两两距离
    类目熵 H = - sum_c p_c log p_c,p_c 为所选集合中类目 c 的占比
"""

import math
import random


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def l2_normalize(v):
    n = norm(v)
    return [x / n for x in v] if n > 0 else list(v)


# ---------------------------------------------------------------------------
# MMR 主体:正文公式 mmr_equation
# ---------------------------------------------------------------------------
def mmr_select(relevance, sim_matrix, n_selected, lambda_param):
    """返回按选择顺序排列的下标列表。

    relevance[i]      : Sim_1(D_i, Q)
    sim_matrix[i][j]  : Sim_2(D_i, D_j)
    """
    n = len(relevance)
    n_selected = min(n_selected, n)
    remaining = list(range(n))
    selected = []

    # 第一次选择:S 为空,无法算多样性项,仅按相关性
    first = max(remaining, key=lambda i: relevance[i])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_selected and remaining:
        best_idx, best_score = None, -float("inf")
        for i in remaining:
            max_sim = max(sim_matrix[i][j] for j in selected)
            score = lambda_param * relevance[i] - (1.0 - lambda_param) * max_sim
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


# ---------------------------------------------------------------------------
# 手写 TF-IDF(字符 1-gram + 2-gram),替代正文的 sklearn TfidfVectorizer
# ---------------------------------------------------------------------------
STOP_CHARS = set("的是在，。、了和与对也及其中可能会")


def tokenize(text):
    chars = [c for c in text if c.strip() and c not in STOP_CHARS]
    grams = list(chars)
    grams += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return grams


def tfidf_matrix(texts):
    docs = [tokenize(t) for t in texts]
    vocab = sorted({g for d in docs for g in d})
    index = {g: k for k, g in enumerate(vocab)}
    n_doc = len(docs)
    df = [0] * len(vocab)
    for d in docs:
        for g in set(d):
            df[index[g]] += 1
    mat = []
    for d in docs:
        row = [0.0] * len(vocab)
        for g in d:
            row[index[g]] += 1.0
        for k, v in enumerate(row):
            if v > 0.0:
                # 平滑 idf,与 sklearn 的 smooth_idf 口径一致
                row[k] = v * (math.log((1 + n_doc) / (1 + df[k])) + 1.0)
        mat.append(l2_normalize(row))
    return mat


def demo_information_retrieval():
    """复刻正文 code_mmr_ir_main 的样例。"""
    query = "人工智能 医疗 应用"
    documents = [
        "人工智能可用于医学影像识别，辅助医生检测肿瘤。",
        "医学影像识别是人工智能在医疗中的重要应用，尤其擅长肺癌筛查。",   # 与第1篇高度相似
        "AI技术能分析电子病历，预测患者的疾病风险。",
        "电子病历分析可帮助医生快速判断患者病情，AI在其中起到关键作用。",  # 与第3篇高度相似
        "人工智能驱动的药物研发能缩短新药上市时间。",
        "AI在手术机器人中的应用提高了微创手术的精度。",
        "手术机器人结合AI技术后，患者术后恢复时间显著缩短。",            # 与第6篇高度相似
    ]
    mat = tfidf_matrix([query] + documents)
    qv, dvs = mat[0], mat[1:]
    relevance = [cosine(v, qv) for v in dvs]
    sim = [[cosine(a, b) for b in dvs] for a in dvs]

    print("== (1) 信息检索场景(正文 code_mmr_ir)==")
    print("文档与查询的相关性 Sim_1 以及与其他文档的最大相似度:")
    print(f"{'doc':>4} {'Sim_1(D,Q)':>11} {'max_j Sim_2':>12}  首句")
    for i, d in enumerate(documents):
        others = [sim[i][j] for j in range(len(documents)) if j != i]
        print(f"{i:>4} {relevance[i]:>11.4f} {max(others):>12.4f}  {d[:16]}")

    for lam in (1.0, 0.6):
        picked = mmr_select(relevance, sim, n_selected=3, lambda_param=lam)
        tag = "纯相关性(lambda=1.0)" if lam == 1.0 else "MMR(lambda=0.6)"
        print(f"\n{tag} 选出的 3 篇:")
        for rank, i in enumerate(picked, 1):
            print(f"  {rank}. doc{i}  rel={relevance[i]:.4f}  {documents[i]}")
        pairs = [sim[a][b] for x, a in enumerate(picked) for b in picked[x + 1:]]
        print(f"  选中集合内最大两两相似度 = {max(pairs):.4f}")


# ---------------------------------------------------------------------------
# 推荐场景:Sim_1 = 融合分,Sim_2 = item embedding 余弦
# ---------------------------------------------------------------------------
CATEGORIES = ["影视", "美食", "萌宠", "游戏", "音乐", "知识"]


def build_candidates(n_items, dim, rng):
    """[MOCK] 合成候选集:同一类目的 embedding 围绕同一个类目中心,因此天然高相似。"""
    centers = {c: l2_normalize([rng.gauss(0, 1) for _ in range(dim)]) for c in CATEGORIES}
    items = []
    for i in range(n_items):
        cat = CATEGORIES[i % len(CATEGORIES)]
        emb = l2_normalize([centers[cat][k] + 0.35 * rng.gauss(0, 1) for k in range(dim)])
        # 让高分物品集中在少数类目,放大「不打散就同质化」的现象
        bonus = 0.25 if cat in ("影视", "美食") else 0.0
        rel = min(1.0, max(0.0, 0.5 + 0.3 * rng.gauss(0, 1) + bonus))
        items.append({"id": f"item_{i:02d}", "cat": cat, "emb": emb, "rel": rel})
    return items


def list_metrics(items, picked):
    sel = [items[i] for i in picked]
    m = len(sel)
    rel = sum(s["rel"] for s in sel) / m
    pair_sims, ild = [], 0.0
    for x in range(m):
        for y in range(x + 1, m):
            s = cosine(sel[x]["emb"], sel[y]["emb"])
            pair_sims.append(s)
            ild += 1.0 - s
    npair = m * (m - 1) / 2
    ild /= npair
    cats = [s["cat"] for s in sel]
    counts = {c: cats.count(c) for c in set(cats)}
    ent = -sum((v / m) * math.log(v / m) for v in counts.values())
    adj = sum(1 for k in range(m - 1) if cats[k] == cats[k + 1])
    return {
        "rel": rel,
        "max_sim": max(pair_sims),
        "ild": ild,
        "n_cat": len(counts),
        "entropy": ent,
        "adj_same": adj,
    }


def demo_recommendation(items, top_m):
    sim = [[(cosine(a["emb"], b["emb"])) for b in items] for a in items]
    relevance = [it["rel"] for it in items]

    print("\n== (2) 推荐重排场景:lambda 扫描 ==")
    print(f"候选集 N={len(items)},目标序列长度 M={top_m},"
          f"Sim_1 = 融合分 r_i,Sim_2 = cos(f_i, f_j)")
    base = mmr_select(relevance, sim, top_m, 1.0)
    base_rel = list_metrics(items, base)["rel"]
    header = (f"{'lambda':>7} {'avg_rel':>8} {'rel损失':>8} {'max_sim':>8} "
              f"{'ILD':>7} {'类目数':>6} {'类目熵':>7} {'相邻同类':>8}")
    print(header)
    for lam in [i / 10 for i in range(11)]:
        picked = mmr_select(relevance, sim, top_m, lam)
        m = list_metrics(items, picked)
        loss = (base_rel - m["rel"]) / base_rel * 100
        print(f"{lam:>7.1f} {m['rel']:>8.4f} {loss:>7.2f}% {m['max_sim']:>8.4f} "
              f"{m['ild']:>7.4f} {m['n_cat']:>6d} {m['entropy']:>7.4f} {m['adj_same']:>8d}")

    print("\n两个极端的序列(类目序列):")
    for lam in (1.0, 0.7, 0.0):
        picked = mmr_select(relevance, sim, top_m, lam)
        cats = "->".join(items[i]["cat"] for i in picked)
        print(f"  lambda={lam:.1f}: {cats}")

    print("\n结论(对应正文 §7.5.1):")
    print("  - lambda=1.0 时 MMR 退化为按精排分排序,相邻同类目对数最多(同质化);")
    print("  - lambda 下降时以少量相关性损失换取 ILD / 类目熵上升,是典型的贪心权衡;")
    print("  - lambda=0.0 时完全不看相关性,只做去冗余,相关性损失最大。")


def main():
    random.seed(42)
    demo_information_retrieval()
    rng = random.Random(42)
    items = build_candidates(n_items=36, dim=16, rng=rng)
    demo_recommendation(items, top_m=10)


if __name__ == "__main__":
    main()
