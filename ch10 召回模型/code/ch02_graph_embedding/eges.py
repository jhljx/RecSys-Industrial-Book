"""
EGES 图嵌入召回（第10章 §10.2.1 「EGES召回」）

核心思想：
    DeepWalk / Node2Vec 只用图结构学习 Item ID Embedding，新物品缺少游走路径就学不到好的
    表示。EGES（Enhanced Graph Embedding with Side Information）为每个物品额外引入 n 种
    Side Information（品类、品牌、店铺……），每种 Side Info 维护独立的 Embedding 矩阵 W_s，
    并用「每个物品私有且可学习」的权重矩阵 A ∈ R^{|V|x(n+1)} 做加权融合（s=0 表示 Item ID）：
        H_v = sum_{s=0..n} exp(a_v^s) * W_s(v) / sum_{s=0..n} exp(a_v^s),  a_v^s = A_{vs}
    exp 保证权重为正、分母做归一化，本质是 Softmax 加权平均（Weighted Average Layer，
    思想类似 Attention）。融合后的 H_v 代替原始 Item Embedding 输入 Skip-Gram，用负采样
    联合优化上下文 Embedding Z_u、各类 Side Info Embedding W_s 与权重参数 a_v^s。
    由于 Side Info Embedding 在物品间共享，冷启动物品（无交互但有属性）也能得到合理表示。
论文出处：Wang J. et al. "Billion-scale Commodity Embedding for E-commerce Recommendation
    in Alibaba"（EGES）, KDD 2018. PyTorch 实现，自动求导。
"""

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

Graph = Dict[str, Dict[str, float]]


# ── 1. 构图与采样工具 ────────────────────────────────────────────────────────
def build_item_graph(sessions: List[List[str]]) -> Graph:
    """由 Session 行为序列构建带权有向物品图，边权 w_ij = Count(i -> j)。"""
    graph: Graph = defaultdict(dict)
    for seq in sessions:
        for a, b in zip(seq, seq[1:]):
            if a != b:
                graph[a][b] = graph[a].get(b, 0.0) + 1.0
    return dict(graph)


class AliasTable:
    """别名采样表：O(N) 预处理、O(1) 采样，工业界负采样与随机游走的常用实现。"""
    def __init__(self, weights: List[float]) -> None:
        n = len(weights)
        total = sum(weights) or 1.0
        scaled = [w * n / total for w in weights]
        self.prob, self.alias = [0.0] * n, [0] * n
        small, large = [], []
        for i, s in enumerate(scaled):
            (small if s < 1.0 else large).append(i)
        while small and large:
            si, li = small.pop(), large.pop()
            self.prob[si], self.alias[si] = scaled[si], li
            scaled[li] -= 1.0 - scaled[si]
            (small if scaled[li] < 1.0 else large).append(li)
        for i in large + small:
            self.prob[i], self.alias[i] = 1.0, i

    def sample(self) -> int:
        i = random.randrange(len(self.prob))
        return i if random.random() < self.prob[i] else self.alias[i]


def random_walk(graph: Graph, start: str, walk_len: int) -> List[str]:
    """按边权在带权有向图上做随机游走（轮盘赌采样），生成节点访问序列。"""
    path = [start]
    while len(path) < walk_len:
        nbrs = graph.get(path[-1])
        if not nbrs:
            break
        r, acc = random.random() * sum(nbrs.values()), 0.0
        for it, w in nbrs.items():
            acc += w
            if acc >= r:
                path.append(it)
                break
    return path


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb + 1e-9)


# ── 2. EGES 模型 ────────────────────────────────────────────────────────────
class EGES(nn.Module):
    """EGES：Side Info Embedding + 物品私有可学习权重的加权融合 + Weighted Skip-Gram。
    使用 PyTorch autograd 自动求导。"""
    def __init__(self, side_info: Dict[str, Dict[str, str]], dim: int = 16,
                 fields: Sequence[str] = ("item_id", "cate", "brand", "shop")) -> None:
        super().__init__()
        self.side_info = side_info
        self.fields = list(fields)  # fields[0] = "item_id" 对应公式中的 s = 0
        self.dim = dim
        self.items = sorted(side_info)
        self.item2id = {v: i for i, v in enumerate(self.items)}

        # 每种 Side Info 一张独立的 Embedding 矩阵 W_s
        self.value2id: Dict[str, Dict[str, int]] = {}
        self.W = nn.ModuleDict()
        for f in self.fields:
            vals = sorted({self._value(v, f) for v in self.items})
            self.value2id[f] = {val: i for i, val in enumerate(vals)}
            emb = nn.Embedding(len(vals), dim)
            nn.init.uniform_(emb.weight, -1.0 / math.sqrt(dim), 1.0 / math.sqrt(dim))
            self.W[f] = emb

        # A ∈ R^{|V|x(n+1)}：每个物品「私有」的可学习融合权重，初始化为 0 等价于均匀平均
        self.A = nn.Embedding(len(self.items), len(self.fields))
        nn.init.zeros_(self.A.weight)
        self.Z = nn.Embedding(len(self.items), dim)  # 上下文 Embedding Z_u
        nn.init.zeros_(self.Z.weight)
        self._neg: Optional[AliasTable] = None

    def _value(self, item: str, field: str) -> str:
        """取物品在某个字段上的取值；item_id 字段的取值就是物品自身。"""
        return item if field == "item_id" else self.side_info[item][field]

    # ---- 2.1 Weighted Average Layer ----
    def fuse(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Softmax 加权融合 H_v = Σ_s β_s W_s(v)，β_s = exp(a_v^s)/Σ_j exp(a_v^j)；
        返回 (H_v, β, 各 Side Info Embedding)。idx 为物品下标 tensor。"""
        a = self.A(idx)                       # (B, n_fields)
        betas = torch.softmax(a, dim=-1)      # (B, n_fields)
        vecs = torch.stack(
            [self.W[f](torch.tensor([self.value2id[f][self._value(it, f)] for it in _items_of(idx, self.items)],
                                    dtype=torch.long)) for f in self.fields], dim=1
        )                                     # (B, n_fields, dim)
        emb = (betas.unsqueeze(-1) * vecs).sum(dim=1)   # (B, dim)
        return emb, betas, vecs

    def fuse_item(self, item: str) -> Tuple[List[float], List[float], List[List[float]]]:
        """单个物品的融合结果（纯 Python 返回，便于观察与检索）。"""
        idx = torch.tensor([self.item2id[item]], dtype=torch.long)
        with torch.no_grad():
            emb, betas, vecs = self.fuse(idx)
        return (emb[0].tolist(), betas[0].tolist(),
                [[vecs[0, s, d].item() for d in range(self.dim)] for s in range(len(self.fields))])

    def attention(self, item: str) -> Dict[str, float]:
        """返回该物品各类 Side Info 的归一化融合权重，便于观察学到的重要性。"""
        _, betas, _ = self.fuse_item(item)
        return {f: b for f, b in zip(self.fields, betas)}

    # ---- 2.2 Weighted Skip-Gram + 负采样 ----
    def score(self, center_idx: torch.Tensor, ctx_idx: torch.Tensor) -> torch.Tensor:
        """σ(H_v · Z_u)，Skip-Gram 的正样本/负样本得分。"""
        emb, _, _ = self.fuse(center_idx)   # (B, dim)
        z = self.Z(ctx_idx)                 # (B, dim)
        return (emb * z).sum(dim=-1)

    def _build_negative_table(self, walks: List[List[str]]) -> None:
        """按节点出现频率的 3/4 次幂构建负采样 alias 表。"""
        cnt = Counter(v for path in walks for v in path)
        self._neg = AliasTable([math.pow(cnt.get(v, 0) + 1e-3, 0.75) for v in self.items])

    def train_model(self, walks: List[List[str]], epochs: int = 3, window: int = 2,
                    neg_k: int = 3, lr: float = 0.05, verbose: bool = True) -> None:
        """Weighted Skip-Gram 训练：联合优化 Item/Side Info Embedding 与融合权重。
        梯度由 PyTorch autograd 自动完成。"""
        self._build_negative_table(walks)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            total, num = 0.0, 0
            optimizer.zero_grad()
            loss_sum = None

            for path in walks:
                for i, center in enumerate(path):
                    if center not in self.item2id:
                        continue
                    ctxs = path[max(0, i - window):i] + path[i + 1:i + window + 1]
                    for ctx in ctxs:
                        c_idx = torch.tensor([self.item2id[center]], dtype=torch.long)
                        p_idx = torch.tensor([self.item2id[ctx]], dtype=torch.long)
                        logit = self.score(c_idx, p_idx)
                        l = -torch.log(torch.sigmoid(logit) + 1e-10)
                        loss_sum = l if loss_sum is None else loss_sum + l
                        total += l.item()
                        num += 1
                        for _ in range(neg_k):  # 负采样：频率 3/4 次幂分布
                            neg = self._neg.sample() if self._neg else 0
                            if self.items[neg] == ctx:
                                continue
                            n_idx = torch.tensor([neg], dtype=torch.long)
                            logit_n = self.score(c_idx, n_idx)
                            l_n = -torch.log(1.0 - torch.sigmoid(logit_n) + 1e-10)
                            loss_sum = loss_sum + l_n
                            total += l_n.item()
                            num += 1

            if loss_sum is not None:
                loss_sum.backward()
                optimizer.step()
            if verbose:
                print(f"  epoch {ep}/{epochs}  平均损失 = {total / max(num, 1):.4f}")

    # ---- 2.3 召回 ----
    def embedding(self, item: str) -> List[float]:
        """物品最终 Embedding，即融合后的 H_v。"""
        return self.fuse_item(item)[0]

    def cold_start_embedding(self, values: Dict[str, str]) -> List[float]:
        """冷启动物品 Embedding：无交互时 Item ID Embedding 不可靠，退化为只用
        Side Info Embedding，并采用全量物品的平均融合权重。"""
        side = [f for f in self.fields if f != "item_id"]
        avg = {f: sum(self.attention(v)[f] for v in self.items) for f in side}
        tot = sum(avg.values()) or 1.0
        emb = [0.0] * self.dim
        with torch.no_grad():
            for f in side:
                idx = torch.tensor([self.value2id[f][values[f]]], dtype=torch.long)
                vec = self.W[f](idx)[0]
                for k in range(self.dim):
                    emb[k] += avg[f] / tot * vec[k].item()
        return emb

    def recommend(self, query, top_k: int = 3,
                  exclude: Sequence[str] = ()) -> List[Tuple[str, float]]:
        """Trigger 可以是物品 ID，也可以是一个 Embedding 向量；返回 Top-K 近邻物品。"""
        src = self.embedding(query) if isinstance(query, str) else list(query)
        skip = set(exclude) | ({query} if isinstance(query, str) else set())
        scores = [(v, cosine_sim(src, self.embedding(v)))
                  for v in self.items if v not in skip]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _items_of(idx: torch.Tensor, items: List[str]) -> List[str]:
    """把物品下标 tensor 转回物品名列表（供 fuse 内部查 side info 取值）。"""
    return [items[i] for i in idx.tolist()]


# ── 3. 演示 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    # 24 个热门物品分属 3 个品类，每个品类 2 个品牌；另有 1 个冷启动物品（无任何交互）
    cates = ["数码", "服饰", "食品"]
    side_info: Dict[str, Dict[str, str]] = {}
    groups: Dict[str, List[str]] = {c: [] for c in cates}
    for idx in range(24):
        item, c = f"i{idx + 1}", cates[idx // 8]
        groups[c].append(item)
        side_info[item] = {"cate": c, "brand": f"{c}-b{idx % 2}", "shop": f"shop{idx % 4}"}
    cold_item = "i_new"
    side_info[cold_item] = {"cate": "数码", "brand": "数码-b0", "shop": "shop0"}
    sessions = []
    for _ in range(60):  # 同品类内部的连续点击行为（冷启动物品不出现在任何 Session 中）
        pool = groups[random.choice(cates)]
        sessions.append([random.choice(pool) for _ in range(4)])
    graph = build_item_graph(sessions)
    walks = [random_walk(graph, v, 10) for _ in range(5) for v in sorted(graph)]
    print(f"物品图 {len(graph)} 个节点；随机游走 {len(walks)} 条；{cold_item} 不在图中\n")
    model = EGES(side_info, dim=16)
    print("训练前 i1 的融合权重（A 初始化为 0，等价于平均池化）：",
          {k: round(v, 2) for k, v in model.attention("i1").items()})
    model.train_model(walks, epochs=3, window=2, neg_k=3, lr=0.05)

    print("\n训练后各物品私有的融合权重（每个物品一组，彼此不同，说明权重是可学习的）：")
    for item in ["i1", "i2", "i12", "i20"]:
        w = model.attention(item)
        print(f"  {item}（{side_info[item]['cate']}）: "
              + "  ".join(f"{f}={w[f]:.3f}" for f in model.fields))

    print("\n热门物品 i3（数码）的 Top-3 召回：")
    for it, s in model.recommend("i3", top_k=3, exclude=[cold_item]):
        print(f"  {it}（{side_info[it]['cate']}）: {s:.4f}")

    print(f"\n冷启动物品 {cold_item}（无交互，仅有 {side_info[cold_item]}）的 Top-3 召回：")
    cold_emb = model.cold_start_embedding(side_info[cold_item])
    cold_recall = model.recommend(cold_emb, top_k=3, exclude=[cold_item])
    for it, s in cold_recall:
        print(f"  {it}（{side_info[it]['cate']}，{side_info[it]['brand']}）: {s:.4f}")
    hit = sum(1 for it, _ in cold_recall if side_info[it]["cate"] == "数码")
    print(f"  命中同品类物品 {hit}/3 → Side Info Embedding 在物品间共享，无交互物品也能召回")
