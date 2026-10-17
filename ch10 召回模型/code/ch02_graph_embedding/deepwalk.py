"""
DeepWalk 图嵌入召回（第10章 §10.2.1）

流程：
1. 在用户-物品二部图上做随机游走生成序列（Random Walk）
2. 用 Skip-Gram（Word2Vec 思想）训练 Embedding
3. 线上用 Item Embedding 做最近邻召回

此处用纯 Python 实现，不依赖 gensim/networkx，便于原理演示。
"""

import random
import math
from collections import defaultdict


# ── 1. 构建图 ─────────────────────────────────────────────────────────────────

def build_graph(user_items: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    构建无向二部图（用户节点 ↔ 物品节点）。
    返回邻接表 {node: [neighbor, ...]}。
    """
    graph: dict[str, list[str]] = defaultdict(list)
    for user, items in user_items.items():
        uid = f"u:{user}"
        for item in items:
            iid = f"i:{item}"
            graph[uid].append(iid)
            graph[iid].append(uid)
    return dict(graph)


# ── 2. 随机游走 ───────────────────────────────────────────────────────────────

def random_walk(graph: dict[str, list[str]], start: str, walk_len: int) -> list[str]:
    """从 start 节点出发进行随机游走，返回节点序列。"""
    walk = [start]
    for _ in range(walk_len - 1):
        neighbors = graph.get(walk[-1], [])
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk


def generate_walks(
    graph: dict[str, list[str]],
    num_walks: int = 10,
    walk_len: int = 20,
) -> list[list[str]]:
    """对图中所有节点各做 num_walks 次随机游走。"""
    nodes = list(graph.keys())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(graph, node, walk_len))
    return walks


# ── 3. Skip-Gram 训练（负采样简化版）────────────────────────────────────────

class SkipGram:
    def __init__(self, vocab: list[str], dim: int = 32, lr: float = 0.025):
        self.vocab = vocab
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.dim = dim
        self.lr = lr
        n = len(vocab)
        scale = 1.0 / math.sqrt(dim)
        self.emb_in  = [[random.uniform(-scale, scale) for _ in range(dim)] for _ in range(n)]
        self.emb_out = [[0.0] * dim for _ in range(n)]

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30, min(30, x))))

    def _update(self, center_id: int, ctx_id: int, label: float) -> None:
        u = self.emb_in[center_id]
        v = self.emb_out[ctx_id]
        score = sum(a * b for a, b in zip(u, v))
        grad = (self._sigmoid(score) - label)
        for k in range(self.dim):
            self.emb_out[ctx_id][k] -= self.lr * grad * u[k]
            self.emb_in[center_id][k] -= self.lr * grad * v[k]

    def train(self, walks: list[list[str]], window: int = 3, neg_k: int = 5) -> None:
        """Skip-Gram + 负采样训练。"""
        n = len(self.vocab)
        for walk in walks:
            ids = [self.word2id[w] for w in walk if w in self.word2id]
            for i, center in enumerate(ids):
                ctx_range = ids[max(0, i - window): i] + ids[i + 1: i + window + 1]
                for ctx in ctx_range:
                    # 正样本
                    self._update(center, ctx, 1.0)
                    # 负样本
                    for _ in range(neg_k):
                        neg = random.randint(0, n - 1)
                        while neg == ctx:
                            neg = random.randint(0, n - 1)
                        self._update(center, neg, 0.0)

    def embedding(self, node: str) -> list[float] | None:
        idx = self.word2id.get(node)
        return self.emb_in[idx] if idx is not None else None


# ── 4. 召回 ──────────────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def recall_deepwalk(
    user_id: str,
    model: SkipGram,
    item_ids: list[str],
    seen: set[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """用用户 Embedding 检索最相似物品。"""
    u_emb = model.embedding(f"u:{user_id}")
    if u_emb is None:
        return []
    scores = []
    for item in item_ids:
        if item in seen:
            continue
        i_emb = model.embedding(f"i:{item}")
        if i_emb is not None:
            scores.append((item, cosine_sim(u_emb, i_emb)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    user_items = {
        "u1": ["i1", "i2", "i3", "i4"],
        "u2": ["i1", "i2", "i5"],
        "u3": ["i3", "i4", "i6"],
        "u4": ["i2", "i4", "i7"],
        "u5": ["i5", "i6", "i7"],
    }
    graph = build_graph(user_items)
    walks = generate_walks(graph, num_walks=20, walk_len=10)

    vocab = list(graph.keys())
    model = SkipGram(vocab, dim=16, lr=0.05)
    model.train(walks, window=2, neg_k=3)

    all_items = [f"i{j}" for j in range(1, 8)]
    seen = {f"i{j}" for j in [1, 2, 3, 4]}
    result = recall_deepwalk("u1", model, all_items, seen, top_k=3)
    print("DeepWalk 召回结果（u1）：")
    for item, s in result:
        print(f"  {item}: {s:.4f}")
