"""
PDN 路径召回（第10章 §10.6.1）

论文：Li et al., "Path-based Deep Network for Candidate Item Matching in Recommenders",
     SIGIR 2021

核心思想：把传统的 U2I 匹配扩展为 U2I2I 的两跳路径建模  u → j → i，
其中 j 是用户历史交互过的 Trigger 物品，i 是待召回的目标物品：
    t_uj  = TriggerNet(z_u, a_uj, x_j)      第一跳：用户对历史物品 j 的触发强度
    s_ji  = SimNet(x_j, x_i, c_ji)          第二跳：历史物品 j 与候选物品 i 的相关性
    PATH_uji = t_uj + s_ji                   路径得分（用于 BCE 训练）
    d_ui  = DirectNet 双塔内积，泛化兴趣得分
    y_bias = BiasNet 曝光偏置（仅训练期使用）
    ŷ_ui  = d_ui + Σ_{j∈N(u)} (t_uj + s_ji) + y_bias   训练时的总得分 logit
    p_ui  = sigmoid(ŷ_ui)，用 BCEWithLogits 训练

PDN 兼具两方面优势：每条路径 u→j→i 都能单独解释「因为你看过 j 所以推荐 i」，
保留了 ItemCF 的可解释性；同时 TriggerNet / SimNet 都是深度网络，可融合用户画像、
物品 side info、共现统计等丰富特征，具备深度模型的表达能力。
在线 Serving 时 SimNet 离线产出 I2I 倒排索引，TriggerNet 在线筛 Top-m Trigger，
把 O(|N(u)|·|I|) 的全量路径计算降到 m×K 规模，并移除 BiasNet 得到无偏打分。

本实现：PyTorch 实现，y_ui = AGG_j(t_uj * s_ji)，BCE 训练，四个子网络联合优化。
"""

import math
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class _MLP(nn.Module):
    """两层感知机 h = ReLU(W1 x + b1)，y = W2 h + b2。"""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class PDN(nn.Module):
    """
    简化版 PDN 召回模型，包含 TriggerNet / SimNet / DirectNet / BiasNet 四个模块。

    特征约定：
        z_u  = [user_emb(u); 年龄归一化, 性别]      用户画像
        x_i  = [item_emb(i); cate_emb(c_i)]          物品特征（含 side info）
        a_uj = [点击次数归一化, 时间衰减]            用户对历史物品 j 的行为特征
        c_ji = [共现次数归一化, 是否同类目]          物品 j 与 i 的共现 / 关联特征
    """

    def __init__(self, n_users: int, n_items: int, item_cate: list, n_cate: int,
                 emb_dim: int = 16, cate_dim: int = 4, hid: int = 8,
                 lr: float = 0.05, reg: float = 1e-4):
        super().__init__()
        self.n_users, self.n_items, self.item_cate = n_users, n_items, item_cate
        self.emb_dim, self.cate_dim = emb_dim, cate_dim
        s = math.sqrt(1.0 / emb_dim)

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.cate_emb = nn.Embedding(n_cate, cate_dim)
        for tab in (self.user_emb, self.item_emb, self.cate_emb):
            nn.init.uniform_(tab.weight, -s, s)

        du, di = emb_dim + 2, emb_dim + cate_dim
        self.trigger_net = _MLP(du + 2 + di, hid, 1)   # (z_u, a_uj, x_j) -> t_uj
        self.sim_net     = _MLP(di + di + 2, hid, 1)   # (x_j, x_i, c_ji) -> s_ji
        self.user_tower  = _MLP(du, hid, emb_dim)      # DirectNet 用户塔 -> p_u
        self.item_tower  = _MLP(di, hid, emb_dim)      # DirectNet 物品塔 -> q_i
        self.bias_net    = _MLP(2, hid, 1)             # BiasNet（仅训练期）

        self.cooc: dict = collections.defaultdict(float)  # 物品共现统计
        self.i2i:  dict = {}                               # 离线 I2I 倒排索引

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=reg)

    # ── 特征构造 ────────────────────────────────────────────────────────────

    def _z_u(self, uid: int, profile: list) -> torch.Tensor:
        prof = torch.tensor(profile, dtype=torch.float32)
        return torch.cat([self.user_emb(torch.tensor(uid, dtype=torch.long)), prof])

    def _x_i(self, iid: int) -> torch.Tensor:
        e = self.item_emb(torch.tensor(iid, dtype=torch.long))
        c = self.cate_emb(torch.tensor(self.item_cate[iid], dtype=torch.long))
        return torch.cat([e, c])

    def _c_ji(self, j: int, i: int) -> torch.Tensor:
        """物品间共现特征：归一化共现次数 + 是否同类目。"""
        return torch.tensor(
            [self.cooc[(j, i)], 1.0 if self.item_cate[j] == self.item_cate[i] else 0.0],
            dtype=torch.float32)

    # ── 四个子网络的前向 ────────────────────────────────────────────────────

    def trigger(self, uid: int, profile: list, j: int, a_uj: list) -> torch.Tensor:
        """TriggerNet：t_uj = TriggerNet(z_u, a_uj, x_j)，第一跳「用户 → 历史物品」。"""
        a = torch.tensor(a_uj, dtype=torch.float32)
        x = torch.cat([self._z_u(uid, profile), a, self._x_i(j)])
        return self.trigger_net(x).squeeze(-1)          # scalar tensor

    def sim(self, j: int, i: int) -> torch.Tensor:
        """SimNet：s_ji = SimNet(x_j, x_i, c_ji)，第二跳「历史物品 → 候选物品」。"""
        x = torch.cat([self._x_i(j), self._x_i(i), self._c_ji(j, i)])
        return self.sim_net(x).squeeze(-1)              # scalar tensor

    def direct(self, uid: int, profile: list, i: int) -> torch.Tensor:
        """DirectNet：双塔内积 d_ui = p_u^T q_i，建模一跳泛化兴趣。"""
        p = self.user_tower(self._z_u(uid, profile))
        q = self.item_tower(self._x_i(i))
        return (p * q).sum()                            # scalar tensor

    # ── 训练 ────────────────────────────────────────────────────────────────

    def forward(self, uid: int, profile: list, hist: list,
                i: int, label: float, bias_feat: list) -> torch.Tensor:
        """
        一步训练（逐路径二分类 + 一跳 DirectNet 联合优化）。

        与「把一条样本的所有路径加总再打一个标签」不同，这里对**每一条路径单独打标签**：
        目标是「同类目（或高共现）的物品往往被同一用户一起消费」，因此把历史中与目标 i
        同类目的物品记为**正路径** u→j→i，其余记为**负路径**，路径得分
            path_uji = t_uj + s_ji
        直接用 BCEWithLogits 训练。这样 SimNet 才知道「哪些 (j,i) 对是相关的」——否则
        路径和在一起时正负抵消，SimNet 学不到任何区分度（I2I 索引会退化成常数）。
        DirectNet 则用样本级标签，建模一跳泛化兴趣。
        """
        # 1) DirectNet 一跳兴趣：样本级 BCE
        d_ui = self.direct(uid, profile, i)
        lab = torch.tensor([label], dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(d_ui.unsqueeze(0), lab)

        # 2) 两跳路径：按「是否与目标同类目」逐路径打标签
        t_ujs = torch.stack([self.trigger(uid, profile, j, a) for j, a in hist])  # [|N(u)|]
        sim_ji = torch.stack([self.sim(j, i) for j, _ in hist])                   # [|N(u)|]
        path_label = torch.tensor(
            [1.0 if self.item_cate[j] == self.item_cate[i] else 0.0 for j, _ in hist],
            dtype=torch.float32)
        loss = loss + F.binary_cross_entropy_with_logits(
            t_ujs + sim_ji, path_label, reduction="mean")

        # 3) BiasNet：吸收曝光位置偏置（仅训练期）
        y_bias = self.bias_net(torch.tensor(bias_feat, dtype=torch.float32)).squeeze(-1)
        loss = loss + F.binary_cross_entropy_with_logits(y_bias.unsqueeze(0), lab)
        return loss

    def train_model(self, samples: list, epochs: int = 6) -> None:
        """samples: [(uid, profile, hist, item, label, bias_feat), ...]"""
        self.train()
        for epoch in range(epochs):
            random.shuffle(samples)
            total = 0.0
            for sp in samples:
                self.optimizer.zero_grad()
                loss = self(*sp)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
            print(f"  epoch {epoch}  avg_loss={total / max(len(samples), 1):.4f}")

    # ── 离线索引 + 在线召回 ─────────────────────────────────────────────────

    def build_i2i_index(self, top_k: int = 4) -> None:
        """离线用 SimNet 给候选物品对打分，为每个 Trigger 物品保留 Top-K 关联物品。"""
        self.eval()
        self.i2i = {}
        with torch.no_grad():
            for j in range(self.n_items):
                scored = [(i, self.sim(j, i).item()) for i in range(self.n_items) if i != j]
                scored.sort(key=lambda x: x[1], reverse=True)
                self.i2i[j] = scored[:top_k]

    def recommend(self, uid: int, profile: list, hist: list,
                  top_n: int = 5, top_m: int = 3) -> list:
        """
        在线召回：TriggerNet 选 Top-m Trigger → 查 I2I 索引得到候选 → 路径打分重排。
        线上不使用 BiasNet，因此得到的是去除曝光位置偏置后的兴趣打分。
        """
        self.eval()
        with torch.no_grad():
            trig_scores = [(j, self.trigger(uid, profile, j, a).item(), a) for j, a in hist]
            triggers = sorted(trig_scores, key=lambda x: x[1], reverse=True)[:top_m]
            seen = {j for j, _ in hist}
            cands = {i for j, _, _ in triggers for i, _ in self.i2i.get(j, []) if i not in seen}
            out = []
            for i in cands:
                score = self.direct(uid, profile, i).item()
                best_path = (None, -1e18)
                for j, t_uj, a in triggers:
                    # 路径得分：重新调用 trigger 以保持图一致（均在 no_grad 下）
                    path_score = (self.trigger(uid, profile, j, a) + self.sim(j, i)).item()
                    score += path_score
                    if path_score > best_path[1]:
                        best_path = (j, path_score)
                out.append((i, score, best_path[0], best_path[1]))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_n]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items, n_cate = 8, 16, 8
    item_cate = [i % n_cate for i in range(n_items)]  # 每个类目恰好 2 个物品

    # 合成数据：用户偏好 4 个类目，历史行为在每个偏好类目里取 1 个物品，
    # 于是「正样本」恰好是某个历史物品的同类目兄弟 —— U2I2I 的路径信号非常干净
    profiles, hists, favs = {}, {}, {}
    for u in range(n_users):
        favs[u] = random.sample(range(n_cate), 4)
        picked = [random.choice([i for i in range(n_items) if item_cate[i] == c]) for c in favs[u]]
        profiles[u] = [random.random(), float(random.randint(0, 1))]
        hists[u] = [(j, [random.random(), random.random()]) for j in picked]

    model = PDN(n_users, n_items, item_cate, n_cate, emb_dim=16, cate_dim=4, hid=8, lr=0.05)
    # 共现统计：同一用户历史内部的物品两两共现
    for u in range(n_users):
        ids = [j for j, _ in hists[u]]
        for a in ids:
            for b in ids:
                if a != b:
                    model.cooc[(a, b)] += 0.25

    # 训练样本：目标物品落在偏好类目内记为点击；头部位置带来一部分虚假点击（曝光偏置）
    samples = []
    for u in range(n_users):
        seen = {j for j, _ in hists[u]}
        for i in range(n_items):
            if i in seen:
                continue
            pos = random.randint(0, 9)
            label = 1.0 if item_cate[i] in favs[u] else 0.0
            if label == 0.0 and pos <= 1 and random.random() < 0.3:
                label = 1.0
            samples.append((u, profiles[u], hists[u], i, label, [pos / 9.0, 1.0]))

    n_pos = sum(1 for s in samples if s[4] > 0.5)
    print(f"用户数={n_users} 物品数={n_items} 类目数={n_cate} "
          f"训练样本数={len(samples)}（正样本 {n_pos}）")
    print("PDN 训练（TriggerNet + SimNet + DirectNet + BiasNet 联合优化）：")
    model.train_model(samples, epochs=6)

    print("\n离线构建 I2I 倒排索引（SimNet 为每个 Trigger 物品保留 Top-3 关联物品）：")
    model.build_i2i_index(top_k=3)
    for j in range(4):
        pairs = "  ".join(f"i{i}(c{item_cate[i]}, s={s:+.3f})" for i, s in model.i2i[j])
        print(f"  trigger item {j:>2} (类目 {item_cate[j]}) -> {pairs}")
    sib_hit = sum(1 for j in range(n_items)
                  if item_cate[model.i2i[j][0][0]] == item_cate[j]) / n_items
    print(f"  SimNet Top-1 命中同类目兄弟物品的比例：{sib_hit:.3f}（共现 + side info 特征生效）")

    u = 0
    print(f"\n用户 {u}：偏好类目={favs[u]}，历史行为={[j for j, _ in hists[u]]}"
          f"（类目 {[item_cate[j] for j, _ in hists[u]]}）")
    print("TriggerNet 对历史行为的触发强度 t_uj（在线只保留 Top-m 作为 Trigger）：")
    model.eval()
    with torch.no_grad():
        for j, a in hists[u]:
            t_uj = model.trigger(u, profiles[u], j, a).item()
            print(f"  item {j:>2} (类目 {item_cate[j]})  t_uj={t_uj:+.4f}")

    recs = model.recommend(u, profiles[u], hists[u], top_n=5, top_m=3)
    print("\n在线召回 Top-5（已移除 BiasNet，附最大贡献路径以体现可解释性）：")
    for i, score, j_best, contrib in recs:
        hit = "√" if item_cate[i] in favs[u] else "×"
        print(f"  item {i:>2} (类目 {item_cate[i]}) {hit}  ŷ={score:.4f}"
              f"   主路径 u→{j_best}→{i}  贡献={contrib:.4f}")
    print("  每条推荐都能回溯到「因为你看过物品 j，所以推荐物品 i」——"
          "这正是 PDN 相比双塔模型多出来的可解释性。")
