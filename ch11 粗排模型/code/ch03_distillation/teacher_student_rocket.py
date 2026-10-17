"""
知识蒸馏粗排（第11章 §11.3）

本文件实现两种粗排蒸馏范式：
1. Teacher-Student：精排模型（教师）→ 粗排双塔（学生），KL 散度对齐 logit
2. Rocket Launching（火箭发射）：共享 Booster 层，轻量 Light Net 作推理路径

PyTorch autograd 实现，无手写梯度。
"""

import math
import random
import collections
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─── 通用 MLP ────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


# ─── 1. Teacher-Student 蒸馏 ─────────────────────────────────────────────────

class TeacherModel(nn.Module):
    """大模型（精排教师）：深层 MLP + 交叉特征。"""

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32, output_dim: int = 32):
        super().__init__()
        scale = math.sqrt(1.0 / emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.net = SimpleMLP([emb_dim * 2, 64, 32, output_dim])
        nn.init.uniform_(self.user_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

    def forward(self, uid: torch.Tensor, iid: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(uid), self.item_emb(iid)], dim=-1)
        out = self.net(x)
        return out.norm(dim=-1)  # 模拟打分

    def soft_labels(self, uid: int, candidates: List[int], temperature: float = 1.0) -> torch.Tensor:
        """对候选集输出 soft label（用于学生蒸馏）。"""
        with torch.no_grad():
            u = torch.tensor([uid]).expand(len(candidates))
            c = torch.tensor(candidates)
            scores = self(u, c) / temperature
            return F.softmax(scores, dim=0)


class StudentDSSM(nn.Module):
    """轻量双塔学生模型：蒸馏教师的 soft label。"""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        output_dim: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_items = n_items
        self.temperature = temperature
        scale = math.sqrt(1.0 / emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_mlp = SimpleMLP([emb_dim, 32, output_dim])
        self.item_mlp = SimpleMLP([emb_dim, 32, output_dim])
        nn.init.uniform_(self.user_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

    def encode_user(self, uid: torch.Tensor) -> torch.Tensor:
        return self.user_mlp(self.user_emb(uid))

    def encode_item(self, iid: torch.Tensor) -> torch.Tensor:
        return self.item_mlp(self.item_emb(iid))

    def score(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            ue = self.encode_user(torch.tensor([uid]))
            ie = self.encode_item(torch.tensor([iid]))
            return F.cosine_similarity(ue, ie).item()

    def distill_step(
        self,
        uid: int,
        candidates: List[int],
        teacher_probs: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        KL 蒸馏 + BPR 硬标签联合训练。
        loss = (1-α)*bpr_loss + α*F.kl_div(log_p_S, p_T, reduction='batchmean')
        """
        u = torch.tensor([uid]).expand(len(candidates))
        c = torch.tensor(candidates)
        ue = self.encode_user(u)                          # [N, D]
        ie = self.encode_item(c)                          # [N, D]
        sims = F.cosine_similarity(ue, ie, dim=1)         # [N]

        # KL 散度蒸馏
        log_p_S = F.log_softmax(sims / self.temperature, dim=0).unsqueeze(0)
        p_T = teacher_probs.unsqueeze(0)
        kl_loss = F.kl_div(log_p_S, p_T, reduction='batchmean')

        # BPR 硬标签：取教师打分最高的为正样本
        pos_idx = teacher_probs.argmax().item()
        neg_idx = random.choice([j for j in range(len(candidates)) if j != pos_idx])
        bpr_loss = -F.logsigmoid(sims[pos_idx] - sims[neg_idx])

        return (1 - alpha) * bpr_loss + alpha * kl_loss

    def train_distill(
        self,
        user_seqs: Dict[int, List[int]],
        teacher: TeacherModel,
        epochs: int = 6,
        lr: float = 0.01,
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0.0
            for uid, seq in user_seqs.items():
                if not seq:
                    continue
                seen = set(seq)
                pool = [i for i in range(self.n_items) if i not in seen]
                if len(pool) < 2:
                    continue
                candidates = random.sample(pool, k=min(8, len(pool)))
                teacher_probs = teacher.soft_labels(uid, candidates)

                optimizer.zero_grad()
                loss = self.distill_step(uid, candidates, teacher_probs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 2 == 0:
                print(f"  [Student] epoch {epoch:3d}  avg_loss={total_loss / max(len(user_seqs), 1):.4f}")

    def pre_rank(
        self, uid: int, candidates: List[int], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        with torch.no_grad():
            u = torch.tensor([uid]).expand(len(candidates))
            c = torch.tensor(candidates)
            ue = self.encode_user(u)
            ie = self.encode_item(c)
            sims = F.cosine_similarity(ue, ie, dim=1)
            scores = [(candidates[j], sims[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─── 2. Rocket Launching（共享 Booster 层）────────────────────────────────────

class RocketLaunching(nn.Module):
    """
    Rocket Launching 蒸馏范式：
    - Heavy Net（精排路径）= Booster 层 + Heavy Head
    - Light Net（粗排路径）= Booster 层（共享） + Light Head（轻量头）
    训练时让 Light Head 做 hint loss 对齐 Heavy Head 中间层（.detach() 实现 stop-gradient）。
    推理只走 Light Net（Booster + Light Head）。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 32,
        booster_dim: int = 32,
        output_dim: int = 16,
    ):
        super().__init__()
        self.n_items = n_items
        scale = math.sqrt(1.0 / emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        # 共享 Booster
        self.booster = SimpleMLP([emb_dim, booster_dim])
        # Heavy Net Head
        self.heavy_head = SimpleMLP([booster_dim, 64, output_dim])
        # Light Net Head（轻量：单隐层，仍远小于 Heavy Head）
        self.light_head = SimpleMLP([booster_dim, booster_dim, output_dim])
        nn.init.uniform_(self.user_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

    def _booster_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.booster(x)

    def heavy_encode(self, uid: torch.Tensor, iid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ub = self._booster_out(self.user_emb(uid))
        ib = self._booster_out(self.item_emb(iid))
        return self.heavy_head(ub), self.heavy_head(ib)

    def light_encode(self, uid: torch.Tensor, iid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ub = self._booster_out(self.user_emb(uid))
        ib = self._booster_out(self.item_emb(iid))
        return self.light_head(ub), self.light_head(ib)

    def forward(
        self,
        uid: torch.Tensor,
        pos_iid: torch.Tensor,
        neg_iid: torch.Tensor,
    ) -> torch.Tensor:
        # Heavy Net BPR loss
        ue_h, pe_h = self.heavy_encode(uid, pos_iid)
        ue_h_neg, ne_h = self.heavy_encode(uid, neg_iid)
        heavy_loss = -F.logsigmoid(
            F.cosine_similarity(ue_h, pe_h, dim=1) - F.cosine_similarity(ue_h_neg, ne_h, dim=1)
        ).mean()

        # Light Net hint loss：Light head 对齐 Heavy head 输出（L2 归一化后做 MSE）
        # 归一化把向量投到单位球，MSE ∈ [0, 4]，与 BPR loss 量级匹配
        # .detach() 实现 stop-gradient（hint target 不回传梯度给 Heavy head）
        ub = self._booster_out(self.user_emb(uid))
        ue_l = F.normalize(self.light_head(ub), dim=1)
        hint_loss = F.mse_loss(ue_l, F.normalize(ue_h.detach(), dim=1))

        return heavy_loss + 0.5 * hint_loss

    def fit(
        self,
        user_seqs: Dict[int, List[int]],
        epochs: int = 6,
        lr: float = 0.01,
    ) -> None:
        """训练入口（命名为 fit 避免遮蔽 nn.Module.train()）。"""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        user_pos: Dict[int, set] = collections.defaultdict(set)
        for uid, seq in user_seqs.items():
            for i in seq:
                user_pos[uid].add(i)
        samples = [
            (uid, seq[t - 1], seq[t])
            for uid, seq in user_seqs.items()
            if len(seq) >= 2
            for t in range(1, len(seq))
        ]
        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss = 0.0
            for uid_s, pos_id, _ in samples:
                neg_id = random.randint(0, self.n_items - 1)
                while neg_id in user_pos[uid_s]:
                    neg_id = random.randint(0, self.n_items - 1)
                uid_t = torch.tensor([uid_s])
                pos_t = torch.tensor([pos_id])
                neg_t = torch.tensor([neg_id])
                optimizer.zero_grad()
                loss = self(uid_t, pos_t, neg_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            if epoch % 2 == 0:
                print(f"  [Rocket] epoch {epoch:3d}  avg_loss={total_loss / max(len(samples), 1):.4f}")

    def light_pre_rank(
        self, uid: int, candidates: List[int], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """推理只走 Light Net（Booster + Light Head）。"""
        with torch.no_grad():
            ub = self._booster_out(self.user_emb(torch.tensor([uid])))
            ue = self.light_head(ub)  # [1, D]
            cand_t = torch.tensor(candidates)
            ib = self._booster_out(self.item_emb(cand_t))
            ie = self.light_head(ib)  # [N, D]
            sims = F.cosine_similarity(ue, ie, dim=1)
            scores = [(candidates[j], sims[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─── 快速验证 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    n_users, n_items = 8, 20
    user_seqs = {u: random.sample(range(n_items), k=random.randint(5, 10)) for u in range(n_users)}

    print("=== Teacher-Student 蒸馏 ===")
    teacher = TeacherModel(n_users, n_items, emb_dim=16, output_dim=16)
    student = StudentDSSM(n_users, n_items, emb_dim=16, output_dim=16, temperature=1.0)
    student.train_distill(user_seqs, teacher, epochs=6, lr=0.05)
    uid, seq = 0, user_seqs[0]
    cands = [i for i in range(n_items) if i not in set(seq)]
    for iid, s in student.pre_rank(uid, cands, top_k=5):
        print(f"  item {iid}: {s:.4f}")

    print("\n=== Rocket Launching ===")
    rocket = RocketLaunching(n_users, n_items, emb_dim=16, booster_dim=16, output_dim=16)
    rocket.fit(user_seqs, epochs=6, lr=0.05)
    for iid, s in rocket.light_pre_rank(uid, cands, top_k=5):
        print(f"  item {iid}: {s:.4f}")
