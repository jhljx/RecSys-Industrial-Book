"""对应正文第 9 章 §9.2「生成式推荐架构」中生成式推荐模块的内部结构。

正文把生成式推荐模块拆成三个核心组成部分（图「生成式推荐模块内部结构」）：
    语义ID生成(Semantic ID Construction) -> 生成式推荐模型(Generative Recommendation Model)
    -> 物品ID检索(Item Retrieval)
而在整条链路上，正文还强调「大多数工业系统仍然会保留重排(Re-ranking)模块作为最后一道
决策环节」，因为生成式模型难以兼顾广告收益、多样性约束、生态治理等复杂业务目标。

本实现用纯标准库在进程内串起这 5 个环节，并逐级统计候选规模与真实耗时：
    1) 语义ID化：对 item embedding 做残差量化(RQ)，得到 L 层语义ID
           e_i -> (s_i^1, s_i^2, ..., s_i^L)
       正文四层 codebook、每层 512 的组合空间规模为 512^4 ~ 6.87e10。
    2) 序列建模：用户行为序列展开为语义 token 序列
           [s_1^1..s_1^L, s_2^1..s_2^L, ...]
    3) 自回归生成：按 Score(SID) = sum_{i=1}^{L} log P(s_i | s_{<i}, u) 做 beam search。
    4) 物品ID检索：SID -> item，多对一去重、无效 SID 丢弃并从其他路径补充。
    5) 过滤重排：去重/已看过滤 + 业务重排分 = alpha*生成分 + beta*质量分 + gamma*商业分。

[MOCK] 本文件中的 `MockItemContentModel`（内容理解打 embedding）、`MockSeqGenModel`
（用共现统计代替 Transformer 的 next-token 分布）、`MockItemStore`（倒排索引/物品属性
存储）都是教学用的进程内假组件，不起真实服务、不读取任何真实数据。
"""

import math
import random
import time
from collections import Counter, defaultdict

L_LAYERS = 3          # 语义ID层数
CODEBOOK_SIZE = 8     # 每层码本大小 K
EMB_DIM = 8
N_ITEMS = 240
N_CATEGORIES = 6


# ----------------------------------------------------------------------------
# [MOCK] 内容理解模型：真实系统里是多模态大模型，这里只用类别中心 + 噪声合成 embedding
# ----------------------------------------------------------------------------
class MockItemContentModel:
    def __init__(self, n_items, n_cat, dim):
        self.centers = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n_cat)]
        self.items = {}
        for i in range(n_items):
            cat = i % n_cat
            c = self.centers[cat]
            self.items[i] = {
                "cat": cat,
                "emb": [c[d] + random.gauss(0, 0.45) for d in range(dim)],
                "quality": round(random.uniform(0.2, 1.0), 3),
                "biz": round(random.uniform(0.0, 1.0), 3),   # 商业化价值（广告/电商）
            }

    def embedding(self, item_id):
        return self.items[item_id]["emb"]


# ----------------------------------------------------------------------------
# 1) 语义ID生成：残差量化（这里用逐层 KMeans，对应 §9.3.3 RQ-KMeans）
# ----------------------------------------------------------------------------
def _l2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _kmeans(vectors, k, iters=12):
    centers = [list(v) for v in random.sample(vectors, k)]
    assign = [0] * len(vectors)
    for _ in range(iters):
        changed = False
        for i, v in enumerate(vectors):
            best = min(range(k), key=lambda j: _l2(v, centers[j]))
            if best != assign[i]:
                assign[i], changed = best, True
        buckets = defaultdict(list)
        for i, a in enumerate(assign):
            buckets[a].append(vectors[i])
        for j in range(k):
            if buckets[j]:
                centers[j] = [sum(col) / len(buckets[j]) for col in zip(*buckets[j])]
        if not changed:
            break
    return centers, assign


def build_semantic_ids(content_model, n_layers=L_LAYERS, k=CODEBOOK_SIZE):
    ids = sorted(content_model.items)
    residual = [list(content_model.embedding(i)) for i in ids]
    codes = [[] for _ in ids]
    codebooks = []
    for _ in range(n_layers):
        centers, assign = _kmeans(residual, k)
        codebooks.append(centers)
        for idx, a in enumerate(assign):
            codes[idx].append(a)
            residual[idx] = [x - c for x, c in zip(residual[idx], centers[a])]
    return {ids[i]: tuple(codes[i]) for i in range(len(ids))}, codebooks


# ----------------------------------------------------------------------------
# [MOCK] 生成式推荐模型：真实系统是 Transformer Decoder-only，这里用
#        (上文 token, 下一 token) 的平滑共现频率近似 P(s_t | s_<t, u)
# ----------------------------------------------------------------------------
class MockSeqGenModel:
    def __init__(self, vocab_per_layer, n_layers):
        self.K = vocab_per_layer
        self.L = n_layers
        self.counts = [defaultdict(Counter) for _ in range(n_layers)]

    def fit(self, token_sequences):
        """token_sequences: 每条是展开后的语义 token 序列 [(layer, code), ...]"""
        for seq in token_sequences:
            for t in range(1, len(seq)):
                layer, code = seq[t]
                ctx = seq[t - 1]                     # 1 阶上下文，教学简化
                self.counts[layer][ctx][code] += 1

    def log_probs(self, ctx, layer):
        """返回第 layer 层码本上的 log P(code | ctx)，Laplace 平滑。"""
        cnt = self.counts[layer][ctx]
        total = sum(cnt.values()) + self.K
        return [math.log((cnt[c] + 1) / total) for c in range(self.K)]


# ----------------------------------------------------------------------------
# [MOCK] 物品存储 / 倒排索引
# ----------------------------------------------------------------------------
class MockItemStore:
    def __init__(self, sid_map, content_model):
        self.sid_to_items = defaultdict(list)
        for item, sid in sid_map.items():
            self.sid_to_items[sid].append(item)
        self.meta = content_model.items

    def lookup(self, sid):
        return self.sid_to_items.get(sid, [])


# ----------------------------------------------------------------------------
# 3) beam search：Score(SID) = sum_i log P(s_i | s_<i, u)
# ----------------------------------------------------------------------------
def beam_search(model, start_ctx, n_layers, beam=12):
    beams = [([], 0.0, start_ctx)]
    for layer in range(n_layers):
        nxt = []
        for codes, score, ctx in beams:
            lps = model.log_probs(ctx, layer)
            for c in range(model.K):
                nxt.append((codes + [c], score + lps[c], (layer, c)))
        nxt.sort(key=lambda x: -x[1])
        beams = nxt[:beam]
    return [(tuple(c), s) for c, s, _ in beams]


def main():
    random.seed(42)
    t_all = time.perf_counter()
    stages = []

    # ---- 阶段 0：内容理解（离线） ----
    t0 = time.perf_counter()
    content = MockItemContentModel(N_ITEMS, N_CATEGORIES, EMB_DIM)
    t_content = (time.perf_counter() - t0) * 1e3

    # ---- 阶段 1：语义ID生成（离线） ----
    t0 = time.perf_counter()
    sid_map, codebooks = build_semantic_ids(content)
    t_sid = (time.perf_counter() - t0) * 1e3
    uniq_sid = len(set(sid_map.values()))
    stages.append(("1 语义ID生成(离线)", N_ITEMS, uniq_sid, t_sid))

    # ---- 构造用户行为序列（同类别内游走，制造可学习结构） ----
    sequences, user_hist = [], {}
    for u in range(300):
        cat = random.randrange(N_CATEGORIES)
        pool = [i for i in sid_map if content.items[i]["cat"] == cat]
        hist = random.sample(pool, 6)
        user_hist[u] = hist
        toks = []
        for it in hist:
            for layer, code in enumerate(sid_map[it]):
                toks.append((layer, code))
        sequences.append(toks)

    # ---- 阶段 2：序列建模（训练） ----
    t0 = time.perf_counter()
    model = MockSeqGenModel(CODEBOOK_SIZE, L_LAYERS)
    model.fit(sequences)
    t_fit = (time.perf_counter() - t0) * 1e3

    # ---- 在线链路：以 user 0 为例 ----
    uid = 0
    hist = user_hist[uid]
    ctx_tokens = []
    for it in hist[:-1]:
        for layer, code in enumerate(sid_map[it]):
            ctx_tokens.append((layer, code))
    stages.append(("2 序列建模(token化)", len(hist) - 1, len(ctx_tokens), t_fit))

    # ---- 阶段 3：自回归生成 ----
    t0 = time.perf_counter()
    beams = beam_search(model, ctx_tokens[-1], L_LAYERS, beam=24)
    t_gen = (time.perf_counter() - t0) * 1e3
    stages.append(("3 自回归生成(beam=24)", CODEBOOK_SIZE ** L_LAYERS, len(beams), t_gen))

    # ---- 阶段 4：物品ID检索（多对一去重 + 无效SID丢弃） ----
    t0 = time.perf_counter()
    store = MockItemStore(sid_map, content)
    cand, invalid, seen = [], 0, set()
    for sid, score in beams:
        items = store.lookup(sid)
        if not items:
            invalid += 1
            continue
        for it in items:
            if it in seen:        # 多个语义ID映射到同一物品 -> 仅保留得分最高者
                continue
            seen.add(it)
            cand.append((it, score))
    t_ret = (time.perf_counter() - t0) * 1e3
    stages.append(("4 物品ID检索", len(beams), len(cand), t_ret))

    # ---- 阶段 5：过滤 + 重排 ----
    t0 = time.perf_counter()
    watched = set(hist)
    filtered = [(i, s) for i, s in cand if i not in watched]
    alpha, beta, gamma = 1.0, 2.0, 0.5
    reranked = sorted(
        (
            (
                i,
                alpha * s + beta * content.items[i]["quality"] + gamma * content.items[i]["biz"],
                s,
            )
            for i, s in filtered
        ),
        key=lambda x: -x[1],
    )[:10]
    t_rr = (time.perf_counter() - t0) * 1e3
    stages.append(("5 过滤+重排", len(cand), len(reranked), t_rr))

    print("=" * 78)
    print("§9.2 生成式推荐模块链路仿真（语义ID化 -> 序列建模 -> 生成 -> 检索 -> 重排）")
    print("=" * 78)
    print(f"内容理解([MOCK] MLLM) 耗时 {t_content:.2f} ms；物品数 {N_ITEMS}，类别 {N_CATEGORIES}")
    print(f"语义ID配置：L={L_LAYERS} 层，每层 K={CODEBOOK_SIZE}，"
          f"理论组合空间 {CODEBOOK_SIZE}^{L_LAYERS}={CODEBOOK_SIZE ** L_LAYERS}")
    print(f"正文四层 512 码本的组合空间：512^4 = {512 ** 4:.4g}")
    print()
    print(f"{'阶段':<24}{'输入规模':>10}{'输出规模':>10}{'耗时(ms)':>12}")
    print("-" * 78)
    for name, nin, nout, ms in stages:
        print(f"{name:<24}{nin:>10}{nout:>10}{ms:>12.3f}")
    print("-" * 78)
    print(f"{'合计':<24}{'':>10}{'':>10}{sum(s[3] for s in stages):>12.3f}")

    print()
    print(f"beam 候选中无效语义ID(Invalid SID) {invalid}/{len(beams)} "
          f"= {invalid / len(beams):.1%}（被直接丢弃，由其他路径补充）")
    collide = Counter(len(v) for v in store.sid_to_items.values())
    print("语义ID冲突分布（一个SID命中的物品数 -> SID个数）：",
          dict(sorted(collide.items())))

    print()
    print("Top-10 重排结果：")
    print(f"{'rank':>5}{'item':>7}{'cat':>5}{'生成分':>10}{'质量':>8}{'商业':>8}{'重排分':>10}")
    for r, (it, final, gen) in enumerate(reranked, 1):
        m = content.items[it]
        print(f"{r:>5}{it:>7}{m['cat']:>5}{gen:>10.3f}{m['quality']:>8.3f}"
              f"{m['biz']:>8.3f}{final:>10.3f}")

    cats = Counter(content.items[i]["cat"] for i, _, _ in reranked)
    print()
    print(f"Top-10 类别分布 {dict(sorted(cats.items()))}；用户历史类别 "
          f"{content.items[hist[0]]['cat']}")
    print(f"链路总墙钟耗时 {(time.perf_counter() - t_all) * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
