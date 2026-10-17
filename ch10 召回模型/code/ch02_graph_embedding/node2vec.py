"""
Node2Vec 图嵌入召回（第10章 §10.2.1 「Node2Vec召回」）

核心思想：
    在 DeepWalk 均匀随机游走的基础上引入「带偏置的二阶随机游走」（Biased Random Walk）。
    游走时不仅看当前节点 v，还要看上一跳节点 t，对候选下一跳 x 赋非归一化权重：
        pi_vx = alpha_pq(t, x) * w_vx,   P(x | v, t) = pi_vx / sum_{z in N(v)} pi_vz
        alpha_pq(t, x) = 1/p   若 d(t, x) = 0（x 就是上一跳 t，即「返回」）
                       = 1     若 d(t, x) = 1（x 与 t 直接相邻，即「同一局部邻域」）
                       = 1/q   若 d(t, x) = 2（x 离 t 更远，即「向外探索」）
    p 为返回参数（Return Parameter），越小越容易回头、越强化局部结构；q 为进出参数
    （In-Out Parameter），q > 1 偏 BFS（局部同质邻域），q < 1 偏 DFS（高阶/跨社区结构）。
    游走序列再喂给 Skip-Gram + 负采样训练 Embedding，线上用 ANN 做 Top-K 近邻召回。
论文出处：Grover A., Leskovec J. "node2vec: Scalable Feature Learning for Networks". KDD 2016.
仅使用 Python 标准库（math / random / collections / typing），便于原理演示。
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

Graph = Dict[str, Dict[str, float]]


# ── 1. 构图：用户行为序列 → 带权物品图 ──────────────────────────────────────────
def build_item_graph(sessions: List[List[str]]) -> Graph:
    """由用户 Session 行为序列构建无向带权物品图：边权为共现次数 w_ij = Count(i -> j)。"""
    graph: Graph = defaultdict(dict)
    for seq in sessions:
        for a, b in zip(seq, seq[1:]):
            if a == b:
                continue
            graph[a][b] = graph[a].get(b, 0.0) + 1.0
            graph[b][a] = graph[b].get(a, 0.0) + 1.0
    return dict(graph)


# ── 2. 别名采样表（Alias Table）：O(N) 预处理，O(1) 采样 ─────────────────────────
class AliasTable:
    """别名采样表，把任意离散分布转成两张查找表，单次采样复杂度 O(1)。"""
    def __init__(self, weights: List[float]) -> None:
        n = len(weights)
        total = sum(weights) or 1.0
        scaled = [w * n / total for w in weights]  # 归一化到均值为 1
        self.prob: List[float] = [0.0] * n
        self.alias: List[int] = [0] * n
        small, large = [], []
        for i, s in enumerate(scaled):
            (small if s < 1.0 else large).append(i)
        while small and large:
            si, li = small.pop(), large.pop()
            self.prob[si] = scaled[si]
            self.alias[si] = li
            scaled[li] -= 1.0 - scaled[si]
            (small if scaled[li] < 1.0 else large).append(li)
        for i in large + small:  # 残留项概率视为 1
            self.prob[i], self.alias[i] = 1.0, i

    def sample(self) -> int:
        i = random.randrange(len(self.prob))
        return i if random.random() < self.prob[i] else self.alias[i]


# ── 3. Node2Vec 模型 ─────────────────────────────────────────────────────────
class Node2Vec:
    """Node2Vec：带偏置二阶随机游走 + Skip-Gram 负采样。"""
    def __init__(self, graph: Graph, dim: int = 16, p: float = 1.0,
                 q: float = 1.0, lr: float = 0.05) -> None:
        self.graph = graph
        self.p, self.q, self.dim, self.lr = p, q, dim, lr
        self.nodes: List[str] = sorted(graph.keys())
        self.node2id = {n: i for i, n in enumerate(self.nodes)}
        # 邻居列表与边权列表固定顺序，便于与 alias 下标对应
        self.nbrs: Dict[str, List[str]] = {v: sorted(graph[v]) for v in self.nodes}
        self._node_alias: Dict[str, AliasTable] = {
            v: AliasTable([graph[v][x] for x in self.nbrs[v]])
            for v in self.nodes if self.nbrs[v]
        }
        self._edge_alias: Dict[Tuple[str, str], AliasTable] = {}  # (t, v) → alias 表
        scale = 1.0 / math.sqrt(dim)
        n = len(self.nodes)
        self.emb_in = [[random.uniform(-scale, scale) for _ in range(dim)] for _ in range(n)]
        self.emb_out = [[0.0] * dim for _ in range(n)]

    # ---- 3.1 带偏置的二阶游走 ----
    def _build_edge_alias(self, prev: str, cur: str) -> AliasTable:
        """按 pi_vx = alpha_pq(t,x) * w_vx 计算非归一化权重，并构建 alias 表（带缓存）。"""
        key = (prev, cur)
        if key in self._edge_alias:
            return self._edge_alias[key]
        prev_nbrs, weights = self.graph[prev], []
        for x in self.nbrs[cur]:
            w = self.graph[cur][x]
            if x == prev:            # d(t, x) = 0 → 返回上一跳
                weights.append(w / self.p)
            elif x in prev_nbrs:     # d(t, x) = 1 → 停留在 t 的邻域（BFS 式）
                weights.append(w)
            else:                    # d(t, x) = 2 → 走向更远的节点（DFS 式）
                weights.append(w / self.q)
        self._edge_alias[key] = AliasTable(weights)
        return self._edge_alias[key]

    def walk(self, start: str, walk_len: int) -> List[str]:
        """从 start 出发做一次带偏置的二阶随机游走。"""
        path = [start]
        while len(path) < walk_len:
            cur = path[-1]
            if not self.nbrs.get(cur):
                break
            if len(path) == 1:  # 第一步没有上一跳，退化为按边权的一阶采样
                idx = self._node_alias[cur].sample()
            else:
                idx = self._build_edge_alias(path[-2], cur).sample()
            path.append(self.nbrs[cur][idx])
        return path

    def generate_walks(self, num_walks: int = 5, walk_len: int = 10) -> List[List[str]]:
        """每个节点出发 num_walks 次游走，得到训练语料。"""
        walks, nodes = [], list(self.nodes)
        for _ in range(num_walks):
            random.shuffle(nodes)
            walks.extend(self.walk(v, walk_len) for v in nodes)
        return walks

    # ---- 3.2 Skip-Gram + 负采样 ----
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

    def _sgd_step(self, center: int, ctx: int, label: float) -> float:
        """单个 (center, ctx, label) 样本的梯度更新，返回 logistic 损失。"""
        u, v = self.emb_in[center], self.emb_out[ctx]
        score = sum(a * b for a, b in zip(u, v))
        pred = self._sigmoid(score)
        grad = pred - label  # dL/dscore
        for k in range(self.dim):
            uk, vk = u[k], v[k]
            self.emb_out[ctx][k] = vk - self.lr * grad * uk
            self.emb_in[center][k] = uk - self.lr * grad * vk
        return -math.log(pred + 1e-10) if label > 0.5 else -math.log(1.0 - pred + 1e-10)

    def train(self, walks: List[List[str]], epochs: int = 3, window: int = 2,
              neg_k: int = 3, verbose: bool = True) -> None:
        """Skip-Gram 负采样训练，按轮打印平均损失。"""
        n = len(self.nodes)
        for ep in range(1, epochs + 1):
            total, cnt = 0.0, 0
            for path in walks:
                ids = [self.node2id[v] for v in path]
                for i, center in enumerate(ids):
                    ctxs = ids[max(0, i - window):i] + ids[i + 1:i + window + 1]
                    for ctx in ctxs:
                        total += self._sgd_step(center, ctx, 1.0)
                        cnt += 1
                        for _ in range(neg_k):  # 负样本：全局均匀采样（简化版）
                            neg = random.randrange(n)
                            if neg == ctx:
                                continue
                            total += self._sgd_step(center, neg, 0.0)
                            cnt += 1
            if verbose:
                print(f"    epoch {ep}/{epochs}  平均损失 = {total / max(cnt, 1):.4f}")

    # ---- 3.3 召回 ----
    def embedding(self, node: str) -> Optional[List[float]]:
        """取节点的 Embedding（emb_in 为节点表示，emb_out 为上下文表示）。"""
        idx = self.node2id.get(node)
        return self.emb_in[idx] if idx is not None else None

    def recommend(self, node: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """以 node 为 Trigger，用 Embedding 余弦相似度检索 Top-K 近邻物品。"""
        src = self.embedding(node)
        if src is None:
            return []
        scores = [(v, cosine_sim(src, self.emb_in[i]))
                  for i, v in enumerate(self.nodes) if v != node]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# ── 4. 演示 ─────────────────────────────────────────────────────────────────
def make_sessions(groups: Dict[str, List[str]], bridges: List[Tuple[str, str]],
                  num: int = 60) -> List[List[str]]:
    """构造合成行为序列：组内高频共现，少量跨组桥接行为。"""
    names = list(groups)
    sessions = []
    for _ in range(num):  # 每个 Session 只在同一个类目内部产生连续点击
        items = groups[random.choice(names)]
        sessions.append([random.choice(items) for _ in range(4)])
    for a, b in bridges:  # 桥接边，使不同社区之间存在少量通路
        sessions.extend([[a, b]] * 2)
    return sessions


def same_group_ratio(walks: List[List[str]], item2group: Dict[str, str]) -> float:
    """统计游走序列中「与起点同社区」的节点占比，用于观察 BFS / DFS 倾向。"""
    hit = tot = 0
    for path in walks:
        g0 = item2group[path[0]]
        for v in path[1:]:
            tot += 1
            hit += int(item2group[v] == g0)
    return hit / max(tot, 1)


if __name__ == "__main__":
    random.seed(42)
    groups = {
        "数码": [f"i{j}" for j in range(1, 9)],
        "服饰": [f"i{j}" for j in range(9, 17)],
        "食品": [f"i{j}" for j in range(17, 25)],
    }
    item2group = {it: g for g, its in groups.items() for it in its}
    sessions = make_sessions(groups, bridges=[("i8", "i9"), ("i16", "i17"), ("i24", "i1")])
    graph = build_item_graph(sessions)
    print(f"物品图：{len(graph)} 个节点，{sum(len(v) for v in graph.values()) // 2} 条无向边\n")

    # (p, q) 的两种典型取值：q > 1 偏 BFS（局部同质），q < 1 偏 DFS（跨社区探索）
    configs = [("BFS 式 (p=1, q=4)", 1.0, 4.0), ("DFS 式 (p=1, q=0.25)", 1.0, 0.25)]
    trigger = "i8"  # 桥接物品，位于「数码」与「服饰」两个社区之间
    for name, p, q in configs:
        random.seed(42)  # 固定种子，仅 (p, q) 不同，便于对比
        model = Node2Vec(graph, dim=16, p=p, q=q, lr=0.05)
        walks = model.generate_walks(num_walks=5, walk_len=10)
        print(f"[{name}] 游走 {len(walks)} 条，同社区节点占比 = "
              f"{same_group_ratio(walks, item2group):.3f}")
        model.train(walks, epochs=3, window=2, neg_k=3)

        ranking = model.recommend(trigger, top_k=len(graph))
        same = [s for it, s in ranking if item2group[it] == item2group[trigger]]
        cross = [(r, it, s) for r, (it, s) in enumerate(ranking, 1)
                 if item2group[it] != item2group[trigger]]
        avg_same, avg_cross = sum(same) / len(same), sum(s for _, _, s in cross) / len(cross)
        print(f"    Trigger={trigger} 的 Top-5 召回："
              + "  ".join(f"{it}({item2group[it]}):{s:.3f}" for it, s in ranking[:5]))
        print(f"    最高分跨类物品 {cross[0][1]}({item2group[cross[0][1]]}) 排名第 {cross[0][0]}；"
              f"同类均值 {avg_same:.3f} / 跨类均值 {avg_cross:.3f}，"
              f"类间差距 = {avg_same - avg_cross:.3f}")

    print("\n结论：q 较大（BFS 式）时游走几乎不出起点社区，Embedding 把同类物品挤在一起，同类"
          "\n      与跨类的相似度差距很大；q 较小（DFS 式）时游走频繁跨社区，类间差距明显缩小，"
          "\n      跨类物品更容易被召回，因此更适合兴趣扩散、长尾发现类任务。")
