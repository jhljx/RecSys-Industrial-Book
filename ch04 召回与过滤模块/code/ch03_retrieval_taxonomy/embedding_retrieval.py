"""
对应正文第 4 章 §4.3.3「向量召回(Embedding-based Retrieval)」。

正文口径:向量召回把用户与物品映射到同一向量空间,用向量相似度衡量匹配程度;
离线用海量行为数据训练 Embedding 模型(DSSM / YouTube DNN / 双塔 / MIND / ...),
线上用近似最近邻搜索(ANN)快速检索与用户向量最接近的物品集合。

本脚本实现的是「向量召回链路」而不是模型细节(双塔模型本身见第 10、11 章):
  [离线] 行为日志 -> 双塔模型(用户塔 / 物品塔,In-Batch Sampled Softmax 训练)
                  -> 导出全量物品向量 -> 建 ANN 索引
  [在线] 用户塔前向得到用户向量 -> ANN 检索 TopK -> 召回候选

为了说明正文「线上为什么必须用 ANN 而不是暴力检索」,脚本自己实现了两种索引,
不依赖任何第三方向量检索库:
  1) BruteForceIndex :全量内积扫描,结果精确,作为 Recall 的 ground truth;
  2) IVFIndex        :倒排文件(Inverted File)。k-means 把物品空间切成 nlist 个
                       聚类,查询时只扫描离查询向量最近的 nprobe 个桶;
  3) HNSWIndex       :分层可导航小世界图的简化版(单层 kNN 图 + 随机长边),
                       贪心路径搜索 + 优先队列 beam search。

关键公式:
  匹配函数(余弦相似度,向量已 L2 归一化,等价于内积)
      s(u, i) = <e_u, e_i> / (||e_u|| · ||e_i||)
  In-Batch Sampled Softmax 损失(温度 τ)
      L = - (1/B) Σ_b log [ exp(s(u_b, i_b)/τ) / Σ_{b'} exp(s(u_b, i_{b'})/τ) ]
  ANN 召回率(以暴力检索 TopK 为基准)
      Recall@K = |TopK_ann ∩ TopK_brute| / K
  IVF 的扫描量 ≈ N · nprobe / nlist,这正是 ANN 的加速来源。

[MOCK] 合成行为数据:用户/物品各带一个隐含兴趣主题,点击概率由主题匹配度决定,
       不读取任何真实数据集。
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn

N_USERS = 600
N_ITEMS = 4000
N_TOPICS = 8
DIM = 16
TOP_K = 50
N_QUERY = 200


# ============================================================ 合成行为数据
def gen_logs(rng: random.Random):
    """用户与物品各有一个主题分布;点击由主题匹配概率采样。"""
    user_topic = [rng.randrange(N_TOPICS) for _ in range(N_USERS)]
    item_topic = [rng.randrange(N_TOPICS) for _ in range(N_ITEMS)]
    item_pop = [rng.random() ** 2 for _ in range(N_ITEMS)]  # 长尾热度

    by_topic = {t: [i for i in range(N_ITEMS) if item_topic[i] == t] for t in range(N_TOPICS)}
    logs = []
    for u in range(N_USERS):
        t = user_topic[u]
        n_click = rng.randint(12, 30)
        for _ in range(n_click):
            if rng.random() < 0.8:                     # 80% 命中自己主题
                pool = by_topic[t]
            else:                                      # 20% 跨主题探索
                pool = by_topic[rng.randrange(N_TOPICS)]
            # 热度加权采样:模拟工业日志的头部效应
            cand = [pool[rng.randrange(len(pool))] for _ in range(4)]
            i = max(cand, key=lambda x: item_pop[x])
            logs.append((u, i))
    return logs, user_topic, item_topic


# ============================================================ 双塔模型
class TwoTower(nn.Module):
    """最小双塔:用户塔 / 物品塔各是 Embedding + 一层 MLP,输出 L2 归一化向量。"""

    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.u_emb = nn.Embedding(n_users, dim)
        self.i_emb = nn.Embedding(n_items, dim)
        self.u_mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.i_mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        nn.init.normal_(self.u_emb.weight, std=0.1)
        nn.init.normal_(self.i_emb.weight, std=0.1)

    def user_vec(self, u):
        return torch.nn.functional.normalize(self.u_mlp(self.u_emb(u)), dim=-1)

    def item_vec(self, i):
        return torch.nn.functional.normalize(self.i_mlp(self.i_emb(i)), dim=-1)


def train_two_tower(logs, epochs: int = 6, batch: int = 256, tau: float = 0.1):
    model = TwoTower(N_USERS, N_ITEMS, DIM)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    pairs = torch.tensor(logs, dtype=torch.long)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(pairs))
        tot, nb = 0.0, 0
        for s in range(0, len(pairs) - batch + 1, batch):
            b = pairs[perm[s:s + batch]]
            uv, iv = model.user_vec(b[:, 0]), model.item_vec(b[:, 1])
            logits = uv @ iv.t() / tau                      # In-Batch 采样 softmax
            label = torch.arange(len(b))
            loss = torch.nn.functional.cross_entropy(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        hist.append(tot / max(nb, 1))
    return model, hist


# ============================================================ 索引 1:暴力检索
class BruteForceIndex:
    """全量内积扫描。结果精确,耗时 O(N·d),线上亿级物品库下不可接受。"""

    name = "BruteForce"

    def __init__(self, vecs: np.ndarray):
        self.vecs = vecs
        self.scanned = 0

    def search(self, q: np.ndarray, k: int):
        self.scanned += len(self.vecs)
        s = self.vecs @ q
        idx = np.argpartition(-s, k)[:k]
        idx = idx[np.argsort(-s[idx])]
        return list(idx)


# ============================================================ 索引 2:IVF
class IVFIndex:
    """倒排文件索引:k-means 聚类 + 只扫 nprobe 个最近桶。

    build : 对物品向量做 n_iter 轮 k-means,得到 nlist 个质心与倒排桶;
    search: 先算 q 与 nlist 个质心的内积,取最近 nprobe 个桶,仅在桶内精排。
    期望扫描量 ≈ N · nprobe / nlist。
    """

    def __init__(self, vecs: np.ndarray, nlist: int, nprobe: int, n_iter: int = 8,
                 seed: int = 42):
        self.vecs = vecs
        self.nlist, self.nprobe = nlist, nprobe
        self.name = f"IVF(nlist={nlist},nprobe={nprobe})"
        rs = np.random.RandomState(seed)
        cent = vecs[rs.choice(len(vecs), nlist, replace=False)].copy()
        for _ in range(n_iter):
            assign = np.argmax(vecs @ cent.T, axis=1)
            for c in range(nlist):
                m = assign == c
                if m.any():
                    v = vecs[m].sum(axis=0)
                    cent[c] = v / (np.linalg.norm(v) + 1e-9)
        self.cent = cent
        self.assign = np.argmax(vecs @ cent.T, axis=1)
        self.buckets = [np.where(self.assign == c)[0] for c in range(nlist)]
        self.scanned = 0

    def search(self, q: np.ndarray, k: int):
        cs = self.cent @ q
        probe = np.argpartition(-cs, self.nprobe - 1)[:self.nprobe]
        cand = np.concatenate([self.buckets[c] for c in probe if len(self.buckets[c])])
        self.scanned += len(cand)
        s = self.vecs[cand] @ q
        kk = min(k, len(cand))
        idx = np.argpartition(-s, kk - 1)[:kk]
        idx = idx[np.argsort(-s[idx])]
        return list(cand[idx])


# ============================================================ 索引 3:HNSW(简化)
class HNSWIndex:
    """HNSW 简化版:kNN 近邻图 + 随机长边,贪心 beam search。

    build : 对每个节点取 M 个最近邻作为短边,再随机加 M_long 条长边(小世界性质),
            保证图连通、可在 O(log N) 量级跳数内到达目标区域;
    search: 从若干随机入口出发,维护大小为 ef 的候选队列,不断扩展未访问邻居,
            直到候选队列中再无比当前 TopK 更优的节点。
    """

    def __init__(self, vecs: np.ndarray, M: int = 16, M_long: int = 4, ef: int = 100,
                 seed: int = 42):
        self.vecs = vecs
        self.ef = ef
        self.name = f"HNSW(M={M},ef={ef})"
        n = len(vecs)
        rs = np.random.RandomState(seed)
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, -2.0)
        nn_idx = np.argpartition(-sim, M, axis=1)[:, :M]
        self.graph = [set(nn_idx[v].tolist()) for v in range(n)]
        for v in range(n):                                   # 随机长边 + 无向化
            for w in rs.randint(0, n, M_long):
                if w != v:
                    self.graph[v].add(int(w))
                    self.graph[int(w)].add(v)
        self.graph = [np.fromiter(g, dtype=np.int64) for g in self.graph]
        self.entries = rs.choice(n, 8, replace=False).tolist()
        self.scanned = 0

    def search(self, q: np.ndarray, k: int):
        visited = set(self.entries)
        self.scanned += len(self.entries)
        top = sorted(((float(self.vecs[e] @ q), e) for e in self.entries),
                     reverse=True)[:self.ef]
        frontier = [v for _, v in top]
        while frontier:
            nxt = []
            for v in frontier:
                nbrs = [int(w) for w in self.graph[v] if w not in visited]
                if not nbrs:
                    continue
                visited.update(nbrs)
                self.scanned += len(nbrs)
                s = self.vecs[nbrs] @ q
                nxt.extend(zip(s.tolist(), nbrs))
            if not nxt:
                break
            worst = top[-1][0] if len(top) >= self.ef else -2.0
            top = sorted(top + nxt, reverse=True)[:self.ef]
            # 只继续扩展「挤进了 ef 队列」的新节点,保证搜索收敛
            keep = {v for _, v in top}
            frontier = [v for s, v in nxt if v in keep and s > worst]
        return [v for _, v in top[:k]]


# ============================================================ 评估
def recall_at_k(ann_res, brute_res, k: int) -> float:
    return len(set(ann_res[:k]) & set(brute_res[:k])) / k


def bench(index, queries, k, brute_ref=None):
    t0 = time.perf_counter()
    res = [index.search(q, k) for q in queries]
    ms = (time.perf_counter() - t0) * 1000 / len(queries)
    rec = 1.0 if brute_ref is None else float(
        np.mean([recall_at_k(r, b, k) for r, b in zip(res, brute_ref)]))
    return res, ms, rec, index.scanned / len(queries)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)   # 小模型下单线程更快,也便于复现 ANN 耗时对比
    rng = random.Random(42)

    logs, user_topic, item_topic = gen_logs(rng)
    print("=" * 74)
    print("§4.3.3 向量召回:双塔训练 -> 导出物品向量 -> ANN 检索")
    print("=" * 74)
    print(f"合成日志: 用户 {N_USERS} / 物品 {N_ITEMS} / 正样本 {len(logs)} 条, "
          f"隐含主题 {N_TOPICS} 个, Embedding 维度 {DIM}")

    model, hist = train_two_tower(logs)
    print("\n[离线] 双塔 In-Batch Sampled Softmax 训练 (τ=0.1)")
    print("  epoch |  loss")
    for ep, l in enumerate(hist, 1):
        print(f"  {ep:5d} | {l:6.4f}")

    with torch.no_grad():
        item_vecs = model.item_vec(torch.arange(N_ITEMS)).numpy().astype(np.float32)
        user_vecs = model.user_vec(torch.arange(N_USERS)).numpy().astype(np.float32)

    # 表征质量:同主题物品的平均相似度 vs 跨主题
    it = np.array(item_topic)
    same = cross = 0.0
    cnt_s = cnt_c = 0
    for _ in range(4000):
        a, b = rng.randrange(N_ITEMS), rng.randrange(N_ITEMS)
        s = float(item_vecs[a] @ item_vecs[b])
        if it[a] == it[b]:
            same += s
            cnt_s += 1
        else:
            cross += s
            cnt_c += 1
    print(f"\n[表征检查] 同主题物品平均余弦 {same / cnt_s:+.4f} | "
          f"跨主题平均余弦 {cross / cnt_c:+.4f}  (差值越大表示向量空间越有判别力)")

    queries = [user_vecs[rng.randrange(N_USERS)] for _ in range(N_QUERY)]

    brute = BruteForceIndex(item_vecs)
    brute_res, brute_ms, _, brute_scan = bench(brute, queries, TOP_K)

    rows = [("BruteForce(精确)", brute_ms, 1.0, brute_scan)]
    for nlist, nprobe in [(64, 1), (64, 4), (64, 8), (64, 16)]:
        idx = IVFIndex(item_vecs, nlist, nprobe)
        _, ms, rec, scan = bench(idx, queries, TOP_K, brute_res)
        rows.append((idx.name, ms, rec, scan))
    for M, ef in [(8, 50), (16, 100)]:
        idx = HNSWIndex(item_vecs, M=M, ef=ef)
        _, ms, rec, scan = bench(idx, queries, TOP_K, brute_res)
        rows.append((idx.name, ms, rec, scan))

    print(f"\n[在线] {N_QUERY} 次查询, TopK={TOP_K}, 精确检索为 ground truth")
    print(f"{'索引':<24}{'Recall@50':>11}{'平均耗时/ms':>14}{'平均扫描向量数':>16}{'加速比':>9}")
    print("-" * 74)
    for name, ms, rec, scan in rows:
        print(f"{name:<24}{rec:>11.4f}{ms:>14.3f}{scan:>16.1f}{brute_ms / ms:>9.2f}x")

    print("\n结论(正文 §4.3.3 的工程含义):")
    print("  1) IVF 的 nprobe 就是「精度-算力」旋钮:nprobe 越大召回率越高、扫描量越大;")
    print("  2) 扫描量 ≈ N·nprobe/nlist,与实测列一致,这是 ANN 相比暴力检索的加速来源;")
    print("  3) 本脚本只有 4000 个物品,且暴力检索是一次 numpy 矩阵乘(C 实现),")
    print("     而 HNSW 的图遍历在 Python 层逐跳进行,因此墙上耗时反而更慢;")
    print("     ANN 的真实收益要看「平均扫描向量数」这一列,量级上亿时才体现为耗时优势;")
    print("  4) 工业界 U2I 向量召回在线只跑一次用户塔前向 + 一次 ANN 查询,")
    print("     物品塔向量全部离线预计算并灌入索引,这样才能把单路召回耗时压到十毫秒级。")


if __name__ == "__main__":
    main()
