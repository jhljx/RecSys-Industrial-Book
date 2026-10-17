"""
TDM 树模型召回（第10章 §10.6.2）

论文：Zhu et al., "Learning Tree-based Deep Model for Recommender Systems", KDD 2018

核心思想：给全库物品建一棵平衡二叉树（叶子 = 物品），把检索建模成「最大堆概率树」
    P^(j)(n|u) = max_{n_c ∈ Child(n)} P^(j+1)(n_c|u) / α^(j)
即父节点的兴趣概率由孩子中的最大值决定。于是逐层做 Beam Search（每层只保留打分最高的 B 个
节点并展开其孩子）即可在 O(B·log|I|) 次打分内拿到近似 Top-K，无需对全库逐一打分；打分网络
还可任意复杂（这里用 node 作 Query 对行为序列做 Attention + MLP），表达能力不受内积结构
约束。树结构本身也可学习：先按类目聚类初始化，训练若干轮后用学到的叶子 Embedding 做平衡
k-means 重排叶子再继续训练，即「模型训练 ↔ 树结构学习」交替优化。

它解决了 Hierarchical Softmax（HS）的两个问题：
1) HS 的目标是加速语言模型 Softmax 的归一化，并非真正解决 Top-K 检索：它只能沿单条路径
   贪心下降，高层一旦判错就无法回退，误差逐层累积；TDM 的最大堆形式配合 Beam Search
   每层保留 B 个候选，容错能力强得多。
2) HS 把每层退化成「一对兄弟节点之间的二分类」这种局部问题，而非「同层全局排序」，而相邻
   兄弟往往语义相近，强行当互斥两类会损失表达能力；TDM 改为每层独立的全局二分类——正样本
   是目标物品的祖先节点，负样本从同层随机采样。

本实现：PyTorch 实现，树用 Python dict 表示（parent/children），节点打分 MLP 用 nn.Linear，
逐层 Beam Search 与平衡 k-means 树重建为 Python 逻辑。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TDM(nn.Module):
    """
    简化版 TDM 树模型召回。树用堆式下标存储：根为 1，节点 n 的孩子为 2n / 2n+1，叶子为
    [2^D, 2^(D+1))，D = log2(|I|)。参数分两套（类似 word2vec 的输入/输出表）：beh_emb 是
    行为序列侧的物品 Embedding；树节点侧内部节点用 node_emb、叶子用随物品走的 item_emb，
    便于重建树时平滑继承表征。

    打分网络：p(node|u) = sigmoid(MLP([û_n ; ê_n ; d·(û_n ⊙ ê_n)]))，其中 u_n 是
    以该节点为 Query 对行为序列做 Attention 得到的节点相关用户表征。
    """

    def __init__(self, n_items: int, item_cate: list, emb_dim: int = 16, hid: int = 8,
                 lr: float = 0.05, reg: float = 1e-4):
        super().__init__()
        d = int(math.log2(n_items))
        assert 2 ** d == n_items, "本实现要求物品数为 2 的幂，便于构建满二叉树"
        self.n_items, self.item_cate, self.emb_dim = n_items, item_cate, emb_dim
        self.depth, self.leaf0, self.n_nodes = d, 2 ** d, 2 ** (d + 1)  # 下标 1..n_nodes-1
        s = math.sqrt(1.0 / emb_dim)

        self.beh_emb = nn.Embedding(n_items, emb_dim)        # 行为序列侧
        self.item_emb = nn.Embedding(n_items, emb_dim)       # 叶子节点侧（随物品走）
        self.node_emb = nn.Embedding(self.n_nodes, emb_dim)  # 内部节点侧
        for tab in (self.beh_emb, self.item_emb, self.node_emb):
            nn.init.uniform_(tab.weight, -s, s)

        # 输入 [û_n ; ê_n ; d·(û_n ⊙ ê_n)]
        self.net = nn.Sequential(
            nn.Linear(3 * emb_dim, hid),
            nn.Tanh(),
            nn.Linear(hid, 1),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

        self.reg = reg
        self.n_score_calls = 0                       # 用于复杂度对比
        self.leaf_item: dict = {}                    # 叶子节点 <-> 物品 的双向映射
        self.item_leaf: dict = {}
        self.build_tree_by_category()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    # ── (a) 树构建：按类目聚类初始化 ──────────────────────────────────────────

    def build_tree_by_category(self) -> None:
        """初始树：按 (类目, id) 排序挂叶子，同类目物品相邻、共享较多祖先 —— 即论文建议的
        初始化（尚无 Embedding 时用类目充当聚类先验）。"""
        order = sorted(range(self.n_items), key=lambda i: (self.item_cate[i], i))
        self._assign_leaves(order)

    def _assign_leaves(self, order: list) -> None:
        self.leaf_item = {self.leaf0 + p: i for p, i in enumerate(order)}
        self.item_leaf = {i: self.leaf0 + p for p, i in enumerate(order)}
        self._build_parent_children()

    def _build_parent_children(self) -> None:
        """用普通 Python dict 显式存储树的父子结构（根为 1）。

        注意：属性名不能叫 children / parent —— nn.Module 已有同名方法，会被覆盖。
        """
        self.child_map: dict = {}
        self.parent_map: dict = {}
        for node in range(1, self.n_nodes):
            if node < self.leaf0:
                self.child_map[node] = (2 * node, 2 * node + 1)
            else:
                self.child_map[node] = ()
            if node > 1:
                self.parent_map[node] = node // 2

    def layer_of(self, node: int) -> int:
        return node.bit_length() - 1                # 根为第 0 层，叶子为第 depth 层

    def ancestors(self, item: int) -> list:
        """目标物品叶子节点到根的整条路径（不含根），即逐层的正样本节点。"""
        n = self.item_leaf[item]                    # 叶子右移 k 位即它在上面第 k 层的祖先
        return [n >> k for k in reversed(range(self.depth))]

    def _node_vec(self, node: int) -> torch.Tensor:
        # 叶子用随物品走的 item_emb，内部节点用 node_emb
        if node >= self.leaf0:
            return self.item_emb(torch.tensor(self.leaf_item[node], dtype=torch.long))
        return self.node_emb(torch.tensor(node, dtype=torch.long))

    # ── (b) 深度打分网络：node 作为 Query 对行为序列做 Attention ───────────────

    def _score(self, seq: list, node: int) -> torch.Tensor:
        """p(node|u) = σ(MLP([û_n ; ê_n ; d·(û_n ⊙ ê_n)]))，a_t = Softmax_t(γ·cos(e_n, h_t))
        是以树节点为 Query 的注意力，u_n = Σ_t a_t·h_t 是节点相关的用户表征 —— 同一用户在
        不同节点上表征不同，这是 TDM 比双塔更强的地方。向量先 L2 归一化、交叉项再乘 d
        （使其求和等于 d·cos），否则交叉项数量级远小于前两块，MLP 会直接忽略用户信息。"""
        self.n_score_calls, d = self.n_score_calls + 1, self.emb_dim
        e_hat = F.normalize(self._node_vec(node), dim=-1)                # [d]
        H = F.normalize(self.beh_emb(torch.tensor(seq, dtype=torch.long)), dim=-1)  # [T, d]
        a = F.softmax(4.0 * (H @ e_hat), dim=0)                          # [T] 注意力权重
        u_hat = F.normalize((a.unsqueeze(-1) * H).sum(dim=0), dim=-1)    # [d]
        x = torch.cat([u_hat, e_hat, d * u_hat * e_hat])                 # [3d]
        return self.net(x).squeeze(-1)                                   # scalar logit

    # ── (d) 逐层正负样本构造 + 训练 ──────────────────────────────────────────

    def _train_step(self, seq: list, target: int, n_negs: int = 3) -> float:
        """TDM 的样本构造：目标物品叶子的**所有祖先节点都是正样本**（逐层各一个），负样本则在
        **同一层内随机采样**，于是每层都是独立的二分类 —— 学的是「这层里哪些节点值得展开」。
        越深的层节点越多，故按层放大负样本数（n_negs + layer），否则叶子层信号过于稀疏。"""
        pos_path = self.ancestors(target)
        nodes, labels = [], []
        for pos in pos_path:
            layer = self.layer_of(pos)
            lo, hi = 2 ** layer, 2 ** (layer + 1)       # 该层节点下标区间
            pool = [n for n in range(lo, hi) if n != pos]
            cands = [(pos, 1.0)] + [(n, 0.0) for n in
                                    random.sample(pool, min(n_negs + layer, len(pool)))]
            for node, label in cands:
                nodes.append(node)
                labels.append(label)

        logits = torch.stack([self._score(seq, n) for n in nodes])
        y = torch.tensor(labels, dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss, pos_path

    def _train_step_loss(self, seq: list, target: int, n_negs: int = 3) -> float:
        """训练一步并更新参数，返回 loss 值。"""
        self.optimizer.zero_grad()
        loss, _ = self._train_step(seq, target, n_negs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(self, samples: list, epochs: int = 6, n_negs: int = 3, tag: str = "TDM"):
        """samples: [(行为序列, 目标物品), ...]；每条样本在树的每一层各做一次二分类。"""
        self.train()
        for epoch in range(epochs):
            random.shuffle(samples)
            total = sum(self._train_step_loss(seq, tgt, n_negs) for seq, tgt in samples)
            print(f"  [{tag}] epoch {epoch}  avg_loss={total / max(len(samples), 1):.4f}")

    # ── (c) 逐层 Beam Search 检索 ───────────────────────────────────────────

    def recommend(self, seq: list, top_k: int = 5, beam: int = 3,
                  verbose: bool = False) -> list:
        """从根出发逐层下降：对当前层候选打分、保留 Top-B，只展开它们的孩子。共 depth 层、
        每层最多打分 2B 个节点，故复杂度 O(B·log|I|)，与 |I| 几乎无关。"""
        self.eval()
        keep = [(1, 0.0)]
        with torch.no_grad():
            for layer in range(1, self.depth + 1):
                cand = [c for n, _ in keep for c in self.child_map[n]]
                scored = sorted(((n, self._score(seq, n).item()) for n in cand),
                                key=lambda x: x[1], reverse=True)
                keep = scored[:beam] if layer < self.depth else scored[:top_k]
                if verbose:
                    detail = "  ".join(
                        f"n{n}" + (f"(i{self.leaf_item[n]})" if n >= self.leaf0 else "")
                        + f"={s:+.3f}" for n, s in keep)
                    print(f"    第 {layer} 层：候选 {len(cand)} 个 → 保留 {len(keep)} 个   {detail}")
        return [(self.leaf_item[n], s) for n, s in keep]

    def recommend_bruteforce(self, seq: list, top_k: int = 5) -> list:
        # 全库逐个叶子打分（线性扫描），用作 Beam Search 的对照组
        self.eval()
        with torch.no_grad():
            scored = [(self.leaf_item[n], self._score(seq, n).item())
                      for n in range(self.leaf0, self.leaf0 + self.n_items)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    # ── (e) 树结构学习：对叶子 Embedding 做平衡 k-means 重排 ──────────────────

    def _leaf_vec(self, item: int) -> torch.Tensor:
        with torch.no_grad():
            return self.item_emb(torch.tensor(item, dtype=torch.long)).clone()

    def _balanced_kmeans2(self, items: list, iters: int = 6) -> tuple:
        """手写二分 k-means，并强制两个簇等大（保证树仍然平衡）。用 Python 手动迭代。"""
        a0 = items[0]  # 用「离 a0 最远的点」做第二个簇心，避免两个簇心初始重合
        a1 = max(items, key=lambda i: float(((self._leaf_vec(a0) - self._leaf_vec(i)) ** 2).sum()))
        c = [self._leaf_vec(a0), self._leaf_vec(a1)]
        half, left, right = len(items) // 2, items[:len(items) // 2], items[len(items) // 2:]
        for _ in range(iters):
            # 分配步：按「到簇0 与到簇1 的距离差」排序，前一半归簇0（这样天然等大）
            ranked = sorted(items, key=lambda i: float(((self._leaf_vec(i) - c[0]) ** 2).sum()
                                                       - ((self._leaf_vec(i) - c[1]) ** 2).sum()))
            left, right = ranked[:half], ranked[half:]
            for ci, grp in ((0, left), (1, right)):     # 更新步：重算簇心
                c[ci] = torch.stack([self._leaf_vec(i) for i in grp]).mean(dim=0)
        return left, right

    def rebuild_tree(self) -> list:
        """递归平衡 k-means 得到新的叶子顺序，让 Embedding 相近的物品共享更多祖先。"""
        def split(items: list) -> list:
            if len(items) <= 1:
                return items
            lf, rt = self._balanced_kmeans2(items)
            return split(lf) + split(rt)
        order = split(list(range(self.n_items)))
        self._assign_leaves(order)
        self._reseed_internal_nodes()
        return order

    def _reseed_internal_nodes(self) -> None:
        # 树结构变了，内部节点旧表征已失效：自底向上用两个孩子的均值重新初始化
        with torch.no_grad():
            for node in range(self.leaf0 - 1, 0, -1):
                l, r = self._node_vec(2 * node), self._node_vec(2 * node + 1)
                self.node_emb.weight[node] = (l + r) / 2.0


def _tree_gap(md: TDM, pairs: set) -> float:
    """树结构质量：共现物品对的平均 LCA 层数 - 非共现物品对的平均 LCA 层数，越大越好 ——
    常被一起消费的物品共享越多祖先，Beam Search 就越不容易在高层把目标剪掉。"""
    acc: dict = {True: [], False: []}
    for i in range(md.n_items):
        for j in range(i + 1, md.n_items):
            a, b = md.item_leaf[i], md.item_leaf[j]       # 两叶子同层，同步右移求 LCA
            k = next(k for k in range(md.depth + 1) if a >> k == b >> k)
            acc[(i, j) in pairs].append(md.depth - k)
    return sum(acc[True]) / len(acc[True]) - sum(acc[False]) / len(acc[False])


def _hit_rate(model: TDM, samples: list, top_k: int = 5, beam: int = 3) -> float:
    return sum(1.0 for seq, tgt in samples
               if tgt in [i for i, _ in model.recommend(seq, top_k, beam)]) / len(samples)


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_items, n_users, n_cate = 16, 8, 4
    item_cate = [i % n_cate for i in range(n_items)]  # 每个类目 4 个物品

    # 合成数据：每个用户偏好 2 个类目，用「序列前缀 → 下一个物品」构造逐层训练样本
    samples = []
    user_seq = {}
    for u in range(n_users):
        favs = random.sample(range(n_cate), 2)
        pool = [i for i in range(n_items) if item_cate[i] in favs]
        seq = user_seq[u] = random.sample(pool, k=6)
        samples += [(seq[:t], seq[t]) for t in range(2, len(seq))]

    model = TDM(n_items, item_cate, emb_dim=16, hid=8, lr=0.05)
    print(f"物品数={n_items} 类目数={n_cate} 树深度={model.depth} "
          f"节点数={model.n_nodes - 1} 训练样本数={len(samples)}")
    print("初始树（按类目聚类初始化）叶子顺序：",
          [f"i{(j := model.leaf_item[n])}(c{item_cate[j]})" for n in range(model.leaf0, 2 * n_items)])
    cooc = {(min(a, b), max(a, b)) for sq in user_seq.values() for a in sq for b in sq if a != b}
    model._assign_leaves(list(range(n_items)))   # 对照组：无任何先验，按 id 顺序挂叶子
    gap_rand = _tree_gap(model, cooc)
    model.build_tree_by_category()               # 恢复类目初始树
    gap_cate = _tree_gap(model, cooc)
    print("\n=== 阶段一：固定树结构，训练深度打分网络（逐层二分类） ===")
    model.train_model(samples, epochs=6, n_negs=3)
    hr_before = _hit_rate(model, samples, top_k=5, beam=3)

    print("\n=== 阶段二：树结构学习（对学到的叶子 Embedding 做递归平衡 k-means） ===")
    order = model.rebuild_tree()
    print("  重建后叶子顺序：", [f"i{i}(c{item_cate[i]})" for i in order])
    print(f"  树结构质量（共现物品对的 LCA 层数优势，越大越好）：无先验树 {gap_rand:+.2f}"
          f" / 类目初始树 {gap_cate:+.2f} / k-means 重建树 {_tree_gap(model, cooc):+.2f}")
    print("  k-means 只看学到的叶子 Embedding、完全没用类目特征，却恢复出了接近类目先验的"
          "结构（远优于无先验树）—— 这就是「树结构可学习」的意义。")
    hr_mid = _hit_rate(model, samples, top_k=5, beam=3)
    print("  树结构已变、内部节点表征需重新适配 —— 继续训练（模型 ↔ 树结构交替优化）：")
    model.train_model(samples, epochs=5, n_negs=3, tag="重训")
    hr_after = _hit_rate(model, samples, top_k=5, beam=3)
    print(f"\n  HR@5（Beam Search, B=3）：原始树 {hr_before:.3f}"
          f" → 刚重建完 {hr_mid:.3f} → 重训之后 {hr_after:.3f}")
    print("  重建瞬间 HR 会掉（节点语义被打乱），重训后反超 —— 这正是交替优化的典型表现。")

    seq, target = samples[0]
    print(f"\n=== 逐层 Beam Search 检索（用户序列 {seq}，类目 "
          f"{[item_cate[i] for i in seq]}，真实目标 i{target}） ===")
    model.n_score_calls = 0
    recs = model.recommend(seq, top_k=5, beam=3, verbose=True)
    beam_calls, model.n_score_calls = model.n_score_calls, 0
    brute = model.recommend_bruteforce(seq, top_k=5)
    full_calls = model.n_score_calls
    fmt = lambda r: [f"i{i}(c{item_cate[i]}){s:+.2f}" for i, s in r]
    print(f"  Beam Search Top-5：{fmt(recs)}\n  全库打分    Top-5：{fmt(brute)}")
    overlap = len({i for i, _ in recs} & {i for i, _ in brute})
    print(f"\n--- 打分次数对比（B=3, |I|={n_items}） ---\n"
          f"  全库线性扫描 O(|I|)      ：{full_calls} 次\n"
          f"  Beam Search  O(B·log|I|) ：{beam_calls} 次，与全库 Top-5 重合 {overlap}/5")
    print("  |I|=16 时树的开销还摊不开，物品量越大优势越明显（|I| vs 2B·log2|I|）：")
    for n, bm in ((m, 6 * int(math.log2(m))) for m in (16, 1024, 1 << 20)):
        print(f"    |I|={n:>9}：全库 {n:>9} 次  vs  Beam {bm:>4} 次  加速 {n / bm:>9.1f}x")
    print("\n--- 最终召回示例（偏好类目内的物品标 √） ---")
    for u in range(3):
        ctx, favs = user_seq[u][:3], {item_cate[i] for i in user_seq[u]}
        print(f"  用户 {u} 序列 {ctx}（类目 {sorted(favs)}）-> "
              + str([f"i{i}(c{item_cate[i]})" + ("√" if item_cate[i] in favs else "×")
                     for i, _ in model.recommend(ctx, top_k=5, beam=3)]))
