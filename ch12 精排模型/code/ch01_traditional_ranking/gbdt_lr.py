"""
GBDT + LR 精排模型（第12章 §12.3.2）

经典两阶段组合：
  1. 训练 GBDT，将样本映射到叶节点 ID；
  2. 将叶节点 ID 做 One-Hot 编码，作为 LR 的输入特征，再训练 LR。

正文说明（PDF 第 5–7 页）：
  - GBDT 通过多棵决策树逐步组合预测，F_T(x)=F_0(x)+η Σ h_t(x)；
  - 叶节点编码将 GBDT 自动学习到的非线性划分转为新特征，再由 LR 拟合；
  - 这体现"先提取特征、再建模"的思路，是深度学习前工业推荐的主流方案。

实现说明：
  - GBDT 用纯 Python + numpy-free 方式实现（不依赖 sklearn/xgboost），
    以便直接 python3 gbdt_lr.py 运行，无需额外依赖。
  - LR 沿用本目录的 PyTorch 实现。
  - 合成数据与 lr.py 一致，方便对比 AUC。

依赖：PyTorch（仅 LR 部分）。运行：python3 gbdt_lr.py
"""

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


# ── 决策树（GBDT 的基学习器）────────────────────────────────────────────────

class _Node:
    __slots__ = ("feat", "thresh", "left", "right", "value")

    def __init__(self):
        self.feat: Optional[int] = None
        self.thresh: Optional[float] = None
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.value: float = 0.0  # 叶节点预测值


class DecisionTree:
    """回归决策树，拟合伪残差。最大深度 max_depth，叶节点最小样本 min_leaf。"""

    def __init__(self, max_depth: int = 3, min_leaf: int = 5):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root: Optional[_Node] = None
        self.n_leaves: int = 0
        self._leaf_id: dict[int, int] = {}  # node id → leaf index

    def _best_split(self, X: list[list[float]], r: list[float],
                    idxs: list[int]):
        best_gain, best_feat, best_thresh = -1.0, -1, 0.0
        n = len(idxs)
        mean_r = sum(r[i] for i in idxs) / n
        ss_total = sum((r[i] - mean_r) ** 2 for i in idxs)
        n_feats = len(X[0])
        for f in range(n_feats):
            vals = sorted({X[i][f] for i in idxs})
            if len(vals) < 2:
                continue
            thresholds = [(vals[k] + vals[k + 1]) / 2 for k in range(len(vals) - 1)]
            for thresh in thresholds:
                left = [i for i in idxs if X[i][f] <= thresh]
                right = [i for i in idxs if X[i][f] > thresh]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                ml = sum(r[i] for i in left) / len(left)
                mr = sum(r[i] for i in right) / len(right)
                ss = (sum((r[i] - ml) ** 2 for i in left) +
                      sum((r[i] - mr) ** 2 for i in right))
                gain = ss_total - ss
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, f, thresh
        return best_feat, best_thresh

    def _build(self, X, r, idxs, depth) -> _Node:
        node = _Node()
        node.value = sum(r[i] for i in idxs) / len(idxs)
        if depth >= self.max_depth or len(idxs) <= self.min_leaf:
            self.n_leaves += 1
            self._leaf_id[id(node)] = self.n_leaves - 1
            return node
        f, thresh = self._best_split(X, r, idxs)
        if f < 0:
            self.n_leaves += 1
            self._leaf_id[id(node)] = self.n_leaves - 1
            return node
        left_idx = [i for i in idxs if X[i][f] <= thresh]
        right_idx = [i for i in idxs if X[i][f] > thresh]
        node.feat = f
        node.thresh = thresh
        node.left = self._build(X, r, left_idx, depth + 1)
        node.right = self._build(X, r, right_idx, depth + 1)
        return node

    def fit(self, X: list[list[float]], residuals: list[float]) -> None:
        self.n_leaves = 0
        self._leaf_id = {}
        self.root = self._build(X, residuals, list(range(len(X))), 0)

    def predict(self, x: list[float]) -> float:
        node = self.root
        while node.feat is not None:
            node = node.left if x[node.feat] <= node.thresh else node.right
        return node.value

    def leaf_index(self, x: list[float]) -> int:
        """返回样本落入的叶节点 index（用于 GBDT+LR 编码）。"""
        node = self.root
        while node.feat is not None:
            node = node.left if x[node.feat] <= node.thresh else node.right
        return self._leaf_id[id(node)]


# ── GBDT 分类器 ──────────────────────────────────────────────────────────────

class GBDT:
    """
    梯度提升分类器（logistic loss）。
    F_T(x) = F_0(x) + η Σ_{t=1}^T h_t(x)
    伪残差 r_i = y_i − σ(F_{t−1}(x_i))（对数损失的负梯度）
    """

    def __init__(self, n_trees: int = 20, max_depth: int = 3,
                 lr: float = 0.1, min_leaf: int = 5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = lr
        self.min_leaf = min_leaf
        self.trees: list[DecisionTree] = []
        self.f0: float = 0.0          # 初始常数预测（log-odds of mean y）
        self._leaf_offsets: list[int] = []  # 各棵树叶节点的全局偏移

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        p_bar = sum(y) / len(y)
        self.f0 = math.log(p_bar / (1 - p_bar + 1e-9) + 1e-9)
        F = [self.f0] * len(X)
        self.trees = []
        self._leaf_offsets = []
        offset = 0
        for _ in range(self.n_trees):
            residuals = [y[i] - _sigmoid(F[i]) for i in range(len(X))]
            tree = DecisionTree(max_depth=self.max_depth, min_leaf=self.min_leaf)
            tree.fit(X, residuals)
            self._leaf_offsets.append(offset)
            offset += tree.n_leaves
            for i in range(len(X)):
                F[i] += self.lr * tree.predict(X[i])
            self.trees.append(tree)
        self._total_leaves = offset

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [_sigmoid(self.f0 + self.lr * sum(t.predict(x) for t in self.trees))
                for x in X]

    def encode(self, X: list[list[float]]) -> list[list[float]]:
        """
        GBDT+LR 核心：将每棵树的叶节点 ID 展开为 One-Hot 向量，拼接后作为 LR 输入。
        输出维度 = Σ tree.n_leaves。
        """
        n = len(X)
        result = [[0.0] * self._total_leaves for _ in range(n)]
        for x, row in zip(X, result):
            for tree, off in zip(self.trees, self._leaf_offsets):
                leaf = tree.leaf_index(x)
                row[off + leaf] = 1.0
        return result


# ── LR（PyTorch）─────────────────────────────────────────────────────────────

class LogisticRegression(nn.Module):
    def __init__(self, in_dim: int, reg: float = 1e-3):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.reg = reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def fit(self, x: torch.Tensor, y: torch.Tensor,
            epochs: int = 80, lr: float = 0.05) -> None:
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg)
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self.forward(x), y.float())
            loss.backward()
            opt.step()

    def predict_proba(self, x: torch.Tensor) -> list[float]:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).tolist()


# ── 合成数据（与 lr.py 相同）──────────────────────────────────────────────────

def make_data(n: int = 400, seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    feats = []
    labels = []
    true_w = [0.3, 0.5, 0.2, 0.6, 0.4, 0.3, 0.5, 0.7]
    for _ in range(n):
        x = [float(random.random() < p) for p in [0.5, 0.4, 0.6, 0.3,
                                                   0.3, 0.5, 0.4, 0.35]]
        logit = sum(w * v for w, v in zip(true_w, x)) - 1.4
        p = _sigmoid(logit + 0.3 * (random.gauss(0, 1)))
        feats.append(x)
        labels.append(1 if p > 0.5 else 0)
    return feats, labels


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    X_all, y_all = make_data(400)
    X_tr, y_tr = X_all[:300], y_all[:300]
    X_te, y_te = X_all[300:], y_all[300:]

    # ── 阶段 1：训练 GBDT ──────────────────────────────────────────────────────
    gbdt = GBDT(n_trees=10, max_depth=4, lr=0.1, min_leaf=8)
    gbdt.fit(X_tr, y_tr)

    p_gbdt_te = gbdt.predict_proba(X_te)
    auc_gbdt = _auc(p_gbdt_te, y_te)
    print(f"GBDT 独立精排  AUC={auc_gbdt:.4f}")
    print(f"  总叶节点数（LR 输入维度）: {gbdt._total_leaves}")

    # ── 阶段 2：叶节点编码 → LR ───────────────────────────────────────────────
    # 正文说明：叶节点对应样本经过特征划分后所处区域，编码后交给 LR 学习
    X_tr_enc = gbdt.encode(X_tr)
    X_te_enc = gbdt.encode(X_te)

    x_tr_t = torch.tensor(X_tr_enc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    x_te_t = torch.tensor(X_te_enc, dtype=torch.float32)

    lr = LogisticRegression(in_dim=gbdt._total_leaves, reg=1e-2)
    lr.fit(x_tr_t, y_tr_t, epochs=120)

    p_gbdt_lr_te = lr.predict_proba(x_te_t)
    auc_gbdt_lr = _auc(p_gbdt_lr_te, y_te)
    print(f"GBDT+LR 精排   AUC={auc_gbdt_lr:.4f}")

    # ── 排序演示 ──────────────────────────────────────────────────────────────
    # 取测试集前 20 条作为一次请求的候选集合
    cand_enc = torch.tensor(X_te_enc[:20], dtype=torch.float32)
    with torch.no_grad():
        scores = torch.sigmoid(lr(cand_enc)).tolist()
    top5 = sorted(enumerate(scores), key=lambda t: -t[1])[:5]
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, score) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  score={score:.4f}  label={y_te[idx]}")

    # ── 组合优势说明 ───────────────────────────────────────────────────────────
    print(f"\n结论：")
    print(f"  GBDT 独立：AUC={auc_gbdt:.4f}（树结构学非线性划分）")
    print(f"  GBDT+LR ：AUC={auc_gbdt_lr:.4f}（叶节点特征 + LR 高效参数学习）")
    print("  正文指出：该组合减少了对人工构造交叉特征的依赖，")
    print("  并兼顾了 GBDT 的非线性建模能力与 LR 的高效部署优势。")
