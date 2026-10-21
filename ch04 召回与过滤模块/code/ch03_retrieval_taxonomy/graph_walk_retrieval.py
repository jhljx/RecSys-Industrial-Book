"""
对应正文第 4 章 §4.3.4「图召回(Graph-based Retrieval)」—— 随机游走与图扩散。

正文口径:推荐系统天然是一张大规模用户-物品二分图,行为构成边。图召回能挖掘
二阶、三阶乃至更高阶关联(用户A→物品X→用户B→物品Y,即使 A 从未碰过 Y);这类
多节点依次连接形成的轨迹称为「路径(Path)」,异构图中带不同节点类型时称为
「元路径(Meta-path)」。正文列举的工业界图召回技术包括 DeepWalk、Node2Vec、
GraphSAGE、PinSage、LightGCN、HGT 等。

本脚本实现 3 种无需复杂框架即可落地的图召回方案,并在同一份留一法测试集上对比:
  1) 二跳路径扩展(U2I2I):score(j) = Σ_{i∈I_u} Σ_{v∈U_i∩U_j} 1 ,
     并按节点度做热度惩罚,这就是 Item-CF 的图语言表述(只用一阶共现);
  2) 图扩散 / 个性化 PageRank(Personalized PageRank, PPR):
         p^(t+1) = (1-α) · P^T p^(t) + α · e_u
     其中 P 为按度归一化的转移矩阵,α 为重启概率,e_u 是以用户 u 为种子的 one-hot。
     收敛后的 p 在物品节点上的质量即为候选分数,天然融合任意阶路径;
  3) DeepWalk / Node2Vec:在图上采样随机游走序列(Node2Vec 的二阶偏置参数 p 控制
     回退、q 控制 DFS/BFS 倾向),把游走序列当句子做 Skip-gram 负采样训练:
         L = - log σ(z_c · z_o) - Σ_{n∈Neg} log σ(-z_c · z_n)
     训练完用「用户向量 · 物品向量」检索 TopK。DeepWalk 等价于 p=q=1。

评估指标(留一法:每个用户留 1 条正样本做测试集)
  HitRate@K = (1/|U|) Σ_u 1[ 测试物品 ∈ TopK(u) ]
  Recall@K  本脚本每人只留 1 条,故 Recall@K == HitRate@K
  覆盖率     = 所有用户 TopK 并集占物品库的比例(衡量是否只会推头部热门)

[MOCK] 合成二分图:用户与物品分属若干兴趣社区,社区内连边概率高、跨社区低,
       另外注入少量高热度物品构造长尾分布;不读取任何真实数据集。
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

N_USERS = 300
N_ITEMS = 1200
N_COMM = 6
TOP_K = 50
EMB_DIM = 32
WALKS_PER_NODE = 4
WALK_LEN = 12
WINDOW = 3
N_NEG = 5


# ============================================================ 合成二分图
def gen_bipartite(rng: random.Random):
    u_comm = [rng.randrange(N_COMM) for _ in range(N_USERS)]
    i_comm = [rng.randrange(N_COMM) for _ in range(N_ITEMS)]
    by_comm = {c: [i for i in range(N_ITEMS) if i_comm[i] == c] for c in range(N_COMM)}
    hot = set(rng.sample(range(N_ITEMS), 40))      # 高热度物品:跨社区都会被点

    edges = []
    for u in range(N_USERS):
        deg = rng.randint(10, 26)
        picked = set()
        while len(picked) < deg:
            if rng.random() < 0.15:
                j = rng.choice(sorted(hot))
            elif rng.random() < 0.85:
                j = rng.choice(by_comm[u_comm[u]])
            else:
                j = rng.choice(by_comm[rng.randrange(N_COMM)])
            picked.add(j)
        edges.extend((u, j) for j in picked)
    return edges, u_comm, i_comm


def leave_one_out(edges, rng: random.Random):
    """每个用户随机留 1 条边作为测试正样本,其余用于建图。"""
    by_u = defaultdict(list)
    for u, i in edges:
        by_u[u].append(i)
    train, test = [], {}
    for u, items in by_u.items():
        held = items[rng.randrange(len(items))]
        test[u] = held
        train.extend((u, i) for i in items if i != held)
    return train, test


# ============================================================ 方案 1:二跳路径扩展
def two_hop_retrieve(u, top_k, user_items, item_users, penalty=0.5):
    """U2I2I:u -> 历史物品 i -> 共同交互用户 v -> v 的其他物品 j。

    score(j) = Σ_{i∈I_u} Σ_{v∈U_i∩U_j} 1 / (|I_v| · |U_j|^penalty)
    其中 1/|I_v| 抑制重度用户带来的噪声,|U_j|^penalty 抑制热门物品。
    """
    seen = user_items[u]
    score = defaultdict(float)
    for i in seen:
        for v in item_users[i]:
            if v == u:
                continue
            w = 1.0 / len(user_items[v])
            for j in user_items[v]:
                if j not in seen:
                    score[j] += w
    out = sorted(((j, s / (len(item_users[j]) ** penalty)) for j, s in score.items()),
                 key=lambda x: -x[1])
    return [int(j) for j, _ in out[:top_k]]


# ============================================================ 方案 2:图扩散 PPR
class PPRDiffusion:
    """个性化 PageRank 图扩散:p = (1-α) P^T p + α e_u,幂迭代到固定步数。

    二分图上一步转移 = 用户->物品;两步 = 用户->物品->用户;因此迭代 2T 步即
    融合了 1..2T 阶所有路径的权重,阶数越高衰减越强(系数 (1-α)^t)。
    """

    def __init__(self, train, alpha: float = 0.15, n_iter: int = 12):
        self.alpha, self.n_iter = alpha, n_iter
        self.ui = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
        for u, i in train:
            self.ui[u, i] = 1.0
        du = self.ui.sum(axis=1, keepdims=True) + 1e-9
        di = self.ui.sum(axis=0, keepdims=True) + 1e-9
        self.P_ui = self.ui / du          # 用户 -> 物品 转移概率
        self.P_iu = (self.ui / di).T      # 物品 -> 用户 转移概率

    def retrieve(self, u, top_k):
        pu = np.zeros(N_USERS, dtype=np.float32)
        pu[u] = 1.0
        seed = pu.copy()
        pi = np.zeros(N_ITEMS, dtype=np.float32)
        for _ in range(self.n_iter):
            pi = (1 - self.alpha) * (pu @ self.P_ui)
            pu = (1 - self.alpha) * (pi @ self.P_iu) + self.alpha * seed
        pi = pi.copy()
        pi[self.ui[u] > 0] = -1.0         # 过滤已交互
        idx = np.argpartition(-pi, top_k)[:top_k]
        return [int(x) for x in idx[np.argsort(-pi[idx])]]


# ============================================================ 方案 3:Node2Vec
def node2vec_walks(adj, n_nodes, rng: random.Random, p: float = 1.0, q: float = 1.0):
    """二阶偏置随机游走。返回节点 ID 序列列表(用户节点 = u,物品节点 = N_USERS + i)。

    从节点 t 走到 v 后,下一步候选 x 的未归一化概率:
        1/p  若 x == t (回到上一步)
        1    若 x 与 t 相邻(同一「圈子」内)
        1/q  否则(向外探索)
    DeepWalk 即 p = q = 1 的特例。
    """
    walks = []
    nbr_set = [set(a) for a in adj]
    order = list(range(n_nodes))
    for _ in range(WALKS_PER_NODE):
        rng.shuffle(order)
        for start in order:
            if not adj[start]:
                continue
            walk = [start, rng.choice(adj[start])]
            while len(walk) < WALK_LEN:
                t, v = walk[-2], walk[-1]
                cands = adj[v]
                if not cands:
                    break
                if p == 1.0 and q == 1.0:
                    walk.append(rng.choice(cands))
                    continue
                w = [1.0 / p if x == t else (1.0 if x in nbr_set[t] else 1.0 / q)
                     for x in cands]
                tot = sum(w)
                r = rng.random() * tot
                acc = 0.0
                nxt = cands[-1]
                for x, wx in zip(cands, w):
                    acc += wx
                    if r <= acc:
                        nxt = x
                        break
                walk.append(nxt)
            walks.append(walk)
    return walks


def train_skipgram(walks, n_nodes, rng: random.Random, epochs: int = 2):
    """Skip-gram 负采样(SGNS),按 degree^0.75 构造负采样分布。"""
    pairs = []
    for w in walks:
        for a in range(len(w)):
            for b in range(max(0, a - WINDOW), min(len(w), a + WINDOW + 1)):
                if a != b:
                    pairs.append((w[a], w[b]))
    freq = np.bincount(np.array([n for w in walks for n in w]), minlength=n_nodes)
    prob = (freq + 1.0) ** 0.75
    prob = prob / prob.sum()
    neg_table = torch.tensor(prob, dtype=torch.float)

    emb_c = nn.Embedding(n_nodes, EMB_DIM)
    emb_o = nn.Embedding(n_nodes, EMB_DIM)
    nn.init.normal_(emb_c.weight, std=0.1)
    nn.init.zeros_(emb_o.weight)
    opt = torch.optim.Adam([*emb_c.parameters(), *emb_o.parameters()], lr=0.02)

    data = torch.tensor(pairs, dtype=torch.long)
    batch = 4096
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(data))
        tot, nb = 0.0, 0
        for s in range(0, len(data) - batch + 1, batch):
            b = data[perm[s:s + batch]]
            zc, zo = emb_c(b[:, 0]), emb_o(b[:, 1])
            pos = torch.nn.functional.logsigmoid((zc * zo).sum(-1))
            neg_id = torch.multinomial(neg_table, len(b) * N_NEG, replacement=True)
            zn = emb_o(neg_id).view(len(b), N_NEG, EMB_DIM)
            neg = torch.nn.functional.logsigmoid(-(zc.unsqueeze(1) * zn).sum(-1)).sum(-1)
            loss = -(pos + neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        losses.append(tot / max(nb, 1))
    return emb_c.weight.detach().numpy(), losses


def emb_retrieve(vecs, u, top_k, user_items):
    uv = vecs[u]
    iv = vecs[N_USERS:]
    s = iv @ uv
    s[list(user_items[u])] = -1e9
    idx = np.argpartition(-s, top_k)[:top_k]
    return [int(x) for x in idx[np.argsort(-s[idx])]]


# ============================================================ 评估
def evaluate(name, topk_by_user, test, i_comm, u_comm):
    hits = sum(1 for u, lst in topk_by_user.items() if test[u] in lst)
    hr = hits / len(test)
    cov = len({j for lst in topk_by_user.values() for j in lst}) / N_ITEMS
    same = tot = 0
    for u, lst in topk_by_user.items():
        for j in lst:
            tot += 1
            same += int(i_comm[j] == u_comm[u])
    return name, hr, cov, same / max(tot, 1)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)   # 小模型下单线程更快,也便于复现
    rng = random.Random(42)

    edges, u_comm, i_comm = gen_bipartite(rng)
    train, test = leave_one_out(edges, rng)
    user_items = defaultdict(set)
    item_users = defaultdict(set)
    for u, i in train:
        user_items[u].add(i)
        item_users[i].add(u)

    print("=" * 78)
    print("§4.3.4 图召回:二跳路径扩展 / 图扩散(PPR) / 随机游走(DeepWalk·Node2Vec)")
    print("=" * 78)
    print(f"二分图: 用户 {N_USERS} 个 / 物品 {N_ITEMS} 个 / 训练边 {len(train)} 条 / "
          f"测试边 {len(test)} 条(留一法), 兴趣社区 {N_COMM} 个")
    print(f"平均用户度 {len(train) / N_USERS:.1f}, 平均物品度 "
          f"{len(train) / len({i for _, i in train}):.2f}")

    # 1) 二跳路径
    t2 = {u: two_hop_retrieve(u, TOP_K, user_items, item_users)
          for u in range(N_USERS)}

    # 2) PPR 图扩散(不同重启概率 α)
    ppr_rows = []
    for alpha in (0.05, 0.15, 0.40):
        d = PPRDiffusion(train, alpha=alpha)
        res = {u: d.retrieve(u, TOP_K) for u in range(N_USERS)}
        ppr_rows.append((f"PPR图扩散(α={alpha})", res))

    # 3) 随机游走 + SGNS
    n_nodes = N_USERS + N_ITEMS
    adj = [[] for _ in range(n_nodes)]
    for u, i in train:
        adj[u].append(N_USERS + i)
        adj[N_USERS + i].append(u)

    walk_rows = []
    for name, (p, q) in [("DeepWalk(p=q=1)", (1.0, 1.0)),
                         ("Node2Vec(p=1,q=0.5)", (1.0, 0.5)),
                         ("Node2Vec(p=4,q=1)", (4.0, 1.0))]:
        walks = node2vec_walks(adj, n_nodes, random.Random(7), p=p, q=q)
        vecs, losses = train_skipgram(walks, n_nodes, rng)
        res = {u: emb_retrieve(vecs, u, TOP_K, user_items) for u in range(N_USERS)}
        walk_rows.append((name, res, len(walks), losses[-1]))

    print(f"\n[随机游走统计] 每节点 {WALKS_PER_NODE} 条游走 × 长度 {WALK_LEN}, "
          f"窗口 {WINDOW}, 负样本 {N_NEG}, 维度 {EMB_DIM}")
    print(f"{'游走策略':<24}{'游走条数':>10}{'SGNS末轮loss':>16}")
    print("-" * 52)
    for name, _, nw, l in walk_rows:
        print(f"{name:<24}{nw:>10d}{l:>16.4f}")

    rows = [("二跳路径U2I2I", t2)] + ppr_rows + [(n, r) for n, r, _, _ in walk_rows]
    print(f"\n[召回效果对比] TopK={TOP_K}, 留一法测试集")
    print(f"{'图召回方案':<24}{'HitRate@50':>12}{'物品覆盖率':>12}{'同社区占比':>12}")
    print("-" * 64)
    for name, res in rows:
        _, hr, cov, same = evaluate(name, res, test, i_comm, u_comm)
        print(f"{name:<24}{hr:>12.4f}{cov:>12.4f}{same:>12.4f}")

    # 高阶路径的直观演示:A 从未交互 Y,但通过 A->X->B->Y 被召回
    d = PPRDiffusion(train, alpha=0.15)
    u0 = 0
    cand = d.retrieve(u0, 5)
    print(f"\n[高阶路径示例] 用户 u{u0} 历史交互 {len(user_items[u0])} 个物品, "
          f"PPR TopK 前 5 个候选:")
    for j in cand:
        bridges = [v for v in item_users[j] if user_items[v] & user_items[u0]][:3]
        shared = sum(len(user_items[v] & user_items[u0]) for v in item_users[j])
        print(f"  item{j:<5d} 自身度={len(item_users[j]):<3d} "
              f"桥梁用户示例={bridges} 与u0共同交互总数={shared} "
              f"(u0 从未直接交互该物品: {j not in user_items[u0]})")

    print("\n结论(正文 §4.3.4 的工程含义):")
    print("  1) 二跳路径只用一阶共现,等价于 Item-CF 的图语言表述,受热门物品影响大;")
    print("  2) 图扩散 PPR 通过幂迭代天然融合 1..2T 阶路径,α 越小高阶信息权重越大,")
    print("     但过小会让分布趋近全局 PageRank 而丢失个性化;")
    print("  3) 随机游走 + Skip-gram(DeepWalk/Node2Vec)把图结构压缩成稠密向量,")
    print("     线上可直接复用向量召回的 ANN 基础设施,这是工业界最常见的落地形态;")
    print("  4) Node2Vec 的 p/q 控制游走倾向:q<1 偏 DFS(更远的社区),p>1 抑制回退。")


if __name__ == "__main__":
    main()
