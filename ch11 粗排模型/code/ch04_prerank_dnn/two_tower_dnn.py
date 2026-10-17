"""
双塔DNN模型（第11章 §11.4.1）

核心思想（严格对应书中 §11.4.1 的口径）：在传统双塔结构基础上**引入用户-物品交叉特征**，
通过**轻量级DNN**增强特征交互能力。模型保留用户塔、物品塔的独立表示学习，但顶层不再只做一次
向量内积，而是把 h_u = f_u(x_u)、h_i = f_i(x_i)、h_cross = f_c(x_{u,i}) 拼接成
h = [h_u, h_i, h_cross]，再经轻量级全连接网络输出 pxtr：
        ŷ = f_dnn(h) = σ(W2·ReLU(W1·h + b1) + b2)
相比纯双塔的 s(u,i) = h_u^T·h_i，顶部 DNN 能学到 h_u 与 h_i 之间的高阶非线性关系。

交叉特征 x_{u,i} 的来源（书中列举的工业做法）：(1) 用户对该物品类目的历史点击率（leave-one-out
平滑统计，避免标签泄漏）；(2) 用户最偏好类目与当前物品类目的匹配指示；(3) SIM（Search-based
Interest Model）检索特征——先从用户历史行为 H_u 中检索出与当前候选物品最相关的若干行为
H_u^i = SIM(H_u, i)，再与物品属性做相似度交互，避免把完整长行为序列塞进粗排模型。

效率代价与适用阶段（与 §11.1 纯双塔 DSSM 的权衡）：
    * 纯双塔 DSSM 的物品向量与用户无关，可**离线预计算**并灌入 ANN / 向量检索引擎，线上一次
      用户塔前向 + 内积即可扫完百万级候选（O(N) 内积，可向量化、可 ANN 剪枝）；
    * 双塔DNN 的交叉特征 x_{u,i} 与顶层 DNN 输入都依赖 (u,i) 组合，物品侧**不可能**再被压成一个
      与用户无关的向量，「打分函数可分解为内积」这一 ANN 前提被破坏，必须对每个候选**逐一**跑顶层
      DNN，复杂度 O(N·|DNN|)。因此它只适用于候选集已缩小到**千级（甚至几百）**的粗排阶段，
      召回阶段仍必须用纯双塔 + ANN。效率-效果谱系：双塔 < 双塔DNN < COLD/精排DNN；
    * 工程上仍保留双塔底座的可预计算性：h_i 离线算好缓存（本实现用 _item_cache 模拟），线上只
      增量计算交叉特征 + 顶部几层小矩阵乘，这也是本模型相比 COLD 的成本优势。

PyTorch 实现（autograd 自动微分），可直接 `python3 two_tower_dnn.py` 运行。
"""
import math
import random
import collections
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 工具函数（用于数据生成与 AUC 统计，不依赖 PyTorch）────────────────────────

def _dot_py(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _sigmoid_py(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

def _cosine_py(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x * x for x in a) + 1e-9)
    nb = math.sqrt(sum(x * x for x in b) + 1e-9)
    return _dot_py(a, b) / (na * nb)

def _auc(pairs: List[Tuple[float, int]]) -> float:
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else (0.5 if p == n else 0.0)
               for p in pos for n in neg) / (len(pos) * len(neg))


def _make_mlp(dims: List[int]) -> nn.Sequential:
    """构建 ReLU 隐层 + 线性输出层的 MLP。"""
    layers = []
    for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TwoTowerDNN(nn.Module):
    """
    双塔DNN粗排模型：双塔底座（可预计算） + 用户-物品交叉特征 + 顶部轻量DNN。

    与「带交叉特征的多塔模型」(§11.2.4) 的区别：多塔模型把交叉特征塔当成与双塔并列的独立分支、
    参与完整表示学习；本模型只在双塔**顶层 Embedding** 上加一个很窄的交互网络，结构更紧凑、
    在线成本更低，所以更适合粗排的大候选集场景。

    梯度：PyTorch autograd 自动微分，无手写梯度。
    """

    def __init__(self, n_users: int, n_items: int, n_cates: int,
                 item_cate: List[int], cross_store: Dict[Tuple[int, int], List[float]],
                 emb_dim: int = 16, cate_dim: int = 4, tower_out: int = 8,
                 cross_in: int = 4, cross_out: int = 4, hidden: int = 16,
                 lr: float = 0.05, reg: float = 1e-5, aux_w: float = 0.5):
        super().__init__()
        self.item_cate = item_cate
        self.cross_store = cross_store
        self.aux_w = aux_w
        self.tower_out = tower_out

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.cate_emb = nn.Embedding(n_cates, cate_dim)

        nn.init.uniform_(self.user_emb.weight, -1/math.sqrt(emb_dim), 1/math.sqrt(emb_dim))
        nn.init.uniform_(self.item_emb.weight, -1/math.sqrt(emb_dim), 1/math.sqrt(emb_dim))
        nn.init.uniform_(self.cate_emb.weight, -1/math.sqrt(cate_dim), 1/math.sqrt(cate_dim))

        self.user_tower = _make_mlp([emb_dim, emb_dim, tower_out])
        self.item_tower = _make_mlp([emb_dim + cate_dim, emb_dim, tower_out])
        self.cross_net  = _make_mlp([cross_in, cross_in * 2, cross_out])
        self.top_dnn    = _make_mlp([2 * tower_out + cross_out, hidden, 1])

        self._item_cache: Dict[int, torch.Tensor] = {}
        self.n_dot_scored = 0
        self.n_dnn_scored = 0

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    def encode_user(self, uid: int) -> torch.Tensor:
        """用户塔 h_u = f_u(x_u)，线上实时计算一次。"""
        x = self.user_emb(torch.tensor(uid))
        return self.user_tower(x)

    def encode_item(self, iid: int) -> torch.Tensor:
        """物品塔 h_i = f_i(x_i)，输入为 [物品ID Embedding, 类目 Embedding]，可离线预计算。"""
        x = torch.cat([self.item_emb(torch.tensor(iid)),
                       self.cate_emb(torch.tensor(self.item_cate[iid]))])
        return self.item_tower(x)

    def precompute_items(self, items: List[int]) -> None:
        """模拟离线刷库：把 h_i 全部算好缓存，线上阶段一只需做内积。"""
        with torch.no_grad():
            self._item_cache = {iid: self.encode_item(iid) for iid in items}

    def _item_vec(self, iid: int) -> torch.Tensor:
        if iid in self._item_cache:
            return self._item_cache[iid]
        with torch.no_grad():
            return self.encode_item(iid)

    def dot_score(self, uid: int, iid: int,
                  u_vec: Optional[torch.Tensor] = None) -> float:
        """阶段一：纯双塔内积 s(u,i) = h_u^T·h_i（可预计算 + 可 ANN，极廉价）。"""
        with torch.no_grad():
            hu = u_vec if u_vec is not None else self.encode_user(uid)
            self.n_dot_scored += 1
            return float(torch.dot(hu, self._item_vec(iid)).item())

    def forward(self, uid: int, iid: int,
                cross_feat: torch.Tensor) -> torch.Tensor:
        """阶段二：ŷ = f_dnn([h_u, h_i, h_cross])，返回 logit（未过 Sigmoid）。"""
        hu = self.encode_user(uid)
        hi = self.encode_item(iid)
        hc = self.cross_net(cross_feat)
        h  = torch.cat([hu, hi, hc])
        self.n_dnn_scored += 1
        return self.top_dnn(h)

    def score(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            cf = torch.tensor(self.cross_store[(uid, iid)], dtype=torch.float32)
            return float(torch.sigmoid(self.forward(uid, iid, cf)).item())

    def train_model(self, samples: List[Tuple[int, int, int]], epochs: int = 6) -> None:
        data = list(samples)
        for epoch in range(epochs):
            random.shuffle(data)
            total_loss = 0.0
            for u, i, y in data:
                self.optimizer.zero_grad()
                cf = torch.tensor(self.cross_store[(u, i)], dtype=torch.float32)
                logit = self.forward(u, i, cf).squeeze()
                loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(float(y)))
                # 辅助损失：保留纯双塔内积头，使 h_u/h_i 自身可用于阶段一粗筛
                hu = self.encode_user(u)
                hi_vec = self.encode_item(i)
                dot_logit = torch.dot(hu, hi_vec)
                loss = loss + self.aux_w * F.binary_cross_entropy_with_logits(
                    dot_logit, torch.tensor(float(y)))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"  epoch {epoch}  loss={total_loss / max(1, len(data)):.4f}")

    def pre_rank(self, uid: int, candidates: List[int], top_k: int = 10,
                 stage1_n: Optional[int] = None) -> List[Tuple[int, float]]:
        """粗排打分并返回 Top-K。
            stage1_n=None  → 单阶段：对全部候选逐个跑顶部轻量DNN（最准、最贵）；
            stage1_n=N     → 两阶段：先用纯双塔内积粗筛 Top-N，再只对这 N 个跑 DNN 精算。
        """
        with torch.no_grad():
            u_vec = self.encode_user(uid)
            pool = list(candidates)
            if stage1_n is not None and stage1_n < len(pool):
                rough = [(iid, self.dot_score(uid, iid, u_vec)) for iid in pool]
                rough.sort(key=lambda x: x[1], reverse=True)
                pool = [iid for iid, _ in rough[:stage1_n]]
            scored = [(iid, self.score(uid, iid)) for iid in pool]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def reset_counters(self) -> None:
        self.n_dot_scored = self.n_dnn_scored = 0


# ── 合成数据：构造标签 + 可观测的用户-物品交叉特征 ──
def make_dataset(n_users: int, n_items: int, n_cates: int, dim: int = 4, sim_top_m: int = 3):
    """
    生成点击标签与交叉特征：物品隐向量围绕类目中心生成（使「类目」成为有效信号）；
    点击概率 σ(⟨z_u, z_i⟩/2)；交叉特征全部用 leave-one-out 统计量，避免标签泄漏。
    """
    cate_center = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n_cates)]
    item_cate = [random.randrange(n_cates) for _ in range(n_items)]
    lat_u = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n_users)]
    lat_i = [[cate_center[item_cate[i]][k] + random.gauss(0, 0.6) for k in range(dim)]
             for i in range(n_items)]
    item_attr = [[lat_i[i][k] + random.gauss(0, 0.3) for k in range(dim)] for i in range(n_items)]

    label = {(u, i): (1 if random.random() < _sigmoid_py(_dot_py(lat_u[u], lat_i[i]) / 2.0) else 0)
             for u in range(n_users) for i in range(n_items)}
    samples = [(u, i, label[(u, i)]) for u in range(n_users) for i in range(n_items)]

    cross_store: Dict[Tuple[int, int], List[float]] = {}
    for u in range(n_users):
        hist = [i for i in range(n_items) if label[(u, i)] == 1]
        for i in range(n_items):
            h_loo = [j for j in hist if j != i]
            c = item_cate[i]
            imp = sum(1 for j in range(n_items) if j != i and item_cate[j] == c)
            f1 = (sum(1 for j in h_loo if item_cate[j] == c) + 1.0) / (imp + 5.0)
            cnt: Dict[int, int] = {}
            for j in h_loo:
                cnt[item_cate[j]] = cnt.get(item_cate[j], 0) + 1
            f2 = 1.0 if cnt and c == max(cnt, key=lambda k: cnt[k]) else 0.0
            picked = sorted((_cosine_py(item_attr[i], item_attr[j]) for j in h_loo),
                            reverse=True)[:sim_top_m]
            f3 = picked[0] if picked else 0.0
            f4 = sum(picked) / len(picked) if picked else 0.0
            cross_store[(u, i)] = [f1, f2, f3, f4]

    for k in range(4):
        vals = [v[k] for v in cross_store.values()]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) + 1e-6
        for v in cross_store.values():
            v[k] = (v[k] - mu) / sd
    return samples, item_cate, cross_store


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, n_cates, emb_dim, epochs = 8, 24, 4, 16, 6
    samples, item_cate, cross_store = make_dataset(n_users, n_items, n_cates)
    model = TwoTowerDNN(n_users, n_items, n_cates, item_cate, cross_store, emb_dim=emb_dim, lr=0.05)
    print(f"双塔DNN粗排（§11.4.1）：用户数={n_users} 物品数={n_items} 类目数={n_cates} "
          f"样本数={len(samples)} 正样本={sum(y for _, _, y in samples)}")
    print(f"  顶部DNN输入维度 = h_u(8) + h_i(8) + h_cross(4) = {2 * 8 + 4}")
    print("\n[1] 训练（主损失：顶部DNN LogLoss；辅助损失：双塔内积头，供阶段一粗筛使用）")
    model.train_model(samples, epochs=epochs)
    print("\n[2] 两个打分头的离线 AUC 对比")
    model.precompute_items(list(range(n_items)))
    with torch.no_grad():
        auc_dot = _auc([(float(torch.dot(model.encode_user(u),
                                         model._item_vec(i)).item()), y)
                        for u, i, y in samples])
        auc_dnn = _auc([(model.score(u, i), y) for u, i, y in samples])
    print(f"  纯双塔内积头 AUC = {auc_dot:.4f}   （可预计算 + ANN，表达能力弱）")
    print(f"  双塔DNN 打分头 AUC = {auc_dnn:.4f}   （带交叉特征 + 顶部DNN，必须逐候选计算）")
    print("\n[3] 工程折中：单阶段全量 DNN  vs  两阶段（内积粗筛 Top-N → DNN 精算）")
    uid, stage1_n, top_k = 0, 8, 5
    cands = list(range(n_items))
    model.reset_counters()
    full = model.pre_rank(uid, cands, top_k=top_k)
    dot_a, dnn_a = model.n_dot_scored, model.n_dnn_scored
    model.reset_counters()
    two = model.pre_rank(uid, cands, top_k=top_k, stage1_n=stage1_n)
    dot_b, dnn_b = model.n_dot_scored, model.n_dnn_scored
    print(f"  单阶段 : 内积打分 {dot_a:>3d} 次, 轻量DNN打分 {dnn_a:>3d} 次")
    print(f"  两阶段 : 内积打分 {dot_b:>3d} 次（物品向量已离线预计算）, 轻量DNN打分 {dnn_b:>3d} 次")
    print(f"  昂贵的 DNN 打分次数 {dnn_a} → {dnn_b}，下降 {(1 - dnn_b / dnn_a) * 100:.1f}%")
    inter = len({i for i, _ in full} & {i for i, _ in two})
    print(f"  两阶段 Top-{top_k} 与单阶段 Top-{top_k} 重合 {inter}/{top_k}（粗筛漏召回的代价）")
    print(f"  用户 {uid} 单阶段粗排 Top-{top_k}：" +
          "  ".join(f"item{i}(c{item_cate[i]}):{s:.3f}" for i, s in full))
    print(f"  用户 {uid} 两阶段粗排 Top-{top_k}：" +
          "  ".join(f"item{i}(c{item_cate[i]}):{s:.3f}" for i, s in two))
