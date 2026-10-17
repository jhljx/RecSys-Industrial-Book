"""
PFD 粗排蒸馏模型（第11章 §11.3.3）

PFD = Privileged Feature Distillation（特权特征蒸馏）
论文出处：Xu et al., "Privileged Features Distillation at Taobao Recommendations", KDD 2020。

核心思想（对应书中 §11.3.3）：
    训练阶段可以拿到一批「在线推理阶段不可得 / 代价过高」的特权特征 x_privileged
    （例如详情页停留时长、是否查看评论、是否咨询客服等用户后验行为，
      以及用户-商品的复杂实时交叉统计特征）。
    PFD 让教师模型同时吃「在线特征 + 特权特征」，学生模型只吃在线特征，
    通过蒸馏把特权信息迁移给学生，在不增加任何在线推理开销的前提下提升粗排效果。
    目标函数（书中公式）：
        L = (1-λ)·L_S(y, f_S(x)) + λ·L_KD(f_T(x, x_priv), f_S(x)) + L_T(y, f_T(x, x_priv))

本实现同时覆盖书中强调的两项工程机制：
    (a) 参数共享（Parameter Sharing）：教师与学生共享 Embedding 层，仅高层预测分支独立；
    (b) 渐进式 Warm-up（Synchronous Training + Warm-up）：训练初期 λ=0 只学各自监督损失，
        待教师收敛后 λ 随 epoch 线性上升到 λ_max，避免早期教师给出不稳定监督信号。

实现方式：PyTorch autograd，nn.Embedding 由教师与学生两个分支共享，
梯度阻断用 .detach()（教师只被真实标签 L_T 优化）。
可直接 `python3 pfd.py` 运行。
"""
import math
import random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── 工具：AUC ─────────────────────────────────────────────────────────────────

def _auc(pairs: List[Tuple[float, int]]) -> float:
    """AUC：正负样本对中被正确排序的比例。"""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return 0.5
    hit = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return hit / (len(pos) * len(neg))


class Branch(nn.Module):
    """教师 / 学生的高层预测分支（共享 Embedding 之外的独立参数）。"""

    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]


class PFD(nn.Module):
    """
    PFD 特权特征蒸馏粗排模型。

    教师 f_T(x, x_priv)、学生 f_S(x) 共享同一对 nn.Embedding(user_emb / item_emb)，
    仅高层 Branch 独立。推理只用学生分支 f_S(x)。
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 16,
        priv_dim: int = 3,
        lr: float = 0.05,
        lam_max: float = 0.8,
        warmup_epochs: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.lr = lr
        self.lam_max = lam_max
        self.warmup_epochs = warmup_epochs
        self.temperature = temperature

        # (a) 参数共享：教师与学生共用这层 Embedding
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        scale = math.sqrt(1.0 / emb_dim)
        nn.init.uniform_(self.user_emb.weight, -scale, scale)
        nn.init.uniform_(self.item_emb.weight, -scale, scale)

        # 教师吃 [在线 emb, 特权特征]，学生只吃 [在线 emb]
        self.teacher_branch = Branch(emb_dim * 2 + priv_dim)
        self.student_branch = Branch(emb_dim * 2)

    # ── 在线特征（用户 / 物品 Embedding 拼接）──────────────────────────────
    def online_emb(self, uid: torch.Tensor, iid: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.user_emb(uid), self.item_emb(iid)], dim=-1)  # [B, 2D]

    def teacher_logit(self, uid: torch.Tensor, iid: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.online_emb(uid, iid), priv], dim=-1)
        return self.teacher_branch(x)

    def student_logit(self, uid: torch.Tensor, iid: torch.Tensor) -> torch.Tensor:
        return self.student_branch(self.online_emb(uid, iid))

    # ── (b) warm-up：λ 随 epoch 线性上升 ────────────────────────────────────
    def lam_at(self, epoch: int, epochs: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        span = max(epochs - self.warmup_epochs, 1)
        return self.lam_max * min(1.0, (epoch - self.warmup_epochs + 1) / span)

    # ── KD 损失：温度缩放后对教师软标签做 KL ────────────────────────────────
    def _kd_loss(self, s_T: torch.Tensor, s_S: torch.Tensor) -> torch.Tensor:
        # 把标量 logit 扩展成二分类 logits [B, 2] = [0, s]，得到合法概率分布
        T = self.temperature
        logits_T = torch.stack([torch.zeros_like(s_T), s_T / T], dim=-1)  # [B, 2]
        logits_S = torch.stack([torch.zeros_like(s_S), s_S / T], dim=-1)
        p_T = F.softmax(logits_T, dim=-1)
        log_p_S = F.log_softmax(logits_S, dim=-1)
        # .detach()：教师不被 KD 梯度更新，只被真实标签 L_T 优化（梯度阻断）
        return F.kl_div(log_p_S, p_T.detach(), reduction='batchmean') * (T * T)

    def forward(
        self,
        uid: torch.Tensor,
        iid: torch.Tensor,
        y: torch.Tensor,
        priv: torch.Tensor,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        L = (1-λ)·L_S + λ·L_KD + L_T
        Returns: (total, L_S, L_KD, L_T)
        """
        s_T = self.teacher_logit(uid, iid, priv)
        s_S = self.student_logit(uid, iid)

        loss_T = F.binary_cross_entropy_with_logits(s_T, y)
        loss_S = F.binary_cross_entropy_with_logits(s_S, y)
        loss_KD = self._kd_loss(s_T, s_S)

        total = (1 - lam) * loss_S + lam * loss_KD + loss_T
        return total, loss_S, loss_KD, loss_T

    # ── 训练：Synchronous Training ──────────────────────────────────────────
    def train_pfd(
        self,
        samples: List[Tuple[int, int, int, List[float]]],
        epochs: int = 6,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        data = list(samples)
        for epoch in range(epochs):
            lam = self.lam_at(epoch, epochs)
            random.shuffle(data)
            acc = [0.0, 0.0, 0.0, 0.0]
            nb = 0
            for start in range(0, len(data), batch_size):
                batch = data[start: start + batch_size]
                uids = torch.tensor([s[0] for s in batch])
                iids = torch.tensor([s[1] for s in batch])
                ys = torch.tensor([s[2] for s in batch], dtype=torch.float)
                priv = torch.tensor([s[3] for s in batch], dtype=torch.float)

                optimizer.zero_grad()
                total, l_s, l_kd, l_t = self(uids, iids, ys, priv, lam)
                total.backward()
                optimizer.step()
                for k, v in enumerate((total, l_s, l_kd, l_t)):
                    acc[k] += v.item()
                nb += 1
            nb = max(nb, 1)
            if verbose and epoch % 2 == 0:
                print(f"  epoch {epoch:2d}  λ={lam:.3f}  "
                      f"L={acc[0] / nb:.4f} (L_S={acc[1] / nb:.4f}  "
                      f"L_KD={acc[2] / nb:.4f}  L_T={acc[3] / nb:.4f})")

    # ── 推理接口（只用学生分支，零特权特征、零额外在线开销）────────────────
    def score(self, uid: int, iid: int) -> float:
        with torch.no_grad():
            s = self.student_logit(torch.tensor([uid]), torch.tensor([iid]))
            return torch.sigmoid(s).item()

    def teacher_score(self, uid: int, iid: int, priv: List[float]) -> float:
        with torch.no_grad():
            s = self.teacher_logit(
                torch.tensor([uid]), torch.tensor([iid]),
                torch.tensor([priv], dtype=torch.float),
            )
            return torch.sigmoid(s).item()

    def pre_rank(
        self, uid: int, candidates: List[int], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        with torch.no_grad():
            u = torch.tensor([uid]).expand(len(candidates))
            c = torch.tensor(candidates)
            s = torch.sigmoid(self.student_logit(u, c))
            scores = [(candidates[j], s[j].item()) for j in range(len(candidates))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def make_samples(n_users: int, n_items: int, lat_dim: int = 4):
    """
    构造带特权特征的合成样本：在线特征即用户/物品 Embedding；
    特权特征 = [详情页停留时长, 是否查看评论, 是否咨询客服]，均为点击之后才可得的后验行为。
    """
    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def _sigmoid(x):
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

    lat_u = [[random.gauss(0, 1) for _ in range(lat_dim)] for _ in range(n_users)]
    lat_i = [[random.gauss(0, 1) for _ in range(lat_dim)] for _ in range(n_items)]
    samples = []
    for u in range(n_users):
        for i in range(n_items):
            s = _dot(lat_u[u], lat_i[i]) / 2.0
            y = 1 if random.random() < _sigmoid(s) else 0
            dwell = max(0.0, s + random.gauss(0, 0.3)) if y == 1 else max(0.0, s - 1.0 + random.gauss(0, 0.3))
            comment = 1.0 if (y == 1 and random.random() < 0.8) or random.random() < 0.15 else 0.0
            consult = 1.0 if (y == 1 and random.random() < 0.6) else 0.0
            samples.append((u, i, y, [dwell, comment, consult]))
    return samples


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    n_users, n_items, emb_dim, epochs = 8, 24, 16, 6
    all_samples = make_samples(n_users, n_items)

    # 划分训练 / 测试集：蒸馏的收益体现在「泛化」而非训练集拟合上
    random.shuffle(all_samples)
    n_test = len(all_samples) // 4
    test_samples = all_samples[:n_test]
    train_samples = all_samples[n_test:]
    print(f"PFD 特权特征蒸馏：n_users={n_users}  n_items={n_items}  "
          f"训练 {len(train_samples)} 条 / 测试 {len(test_samples)} 条")

    print("\n[1] PFD 同步训练（Warm-up 2 轮后渐进引入蒸馏损失）")
    pfd = PFD(n_users, n_items, emb_dim=emb_dim, priv_dim=3, lr=0.05,
              lam_max=0.8, warmup_epochs=2)
    pfd.train_pfd(train_samples, epochs=epochs)

    print("\n[2] 消融：关闭蒸馏（λ≡0，学生只用在线特征+硬标签）")
    torch.manual_seed(42)
    base = PFD(n_users, n_items, emb_dim=emb_dim, priv_dim=3, lr=0.05,
               lam_max=0.0, warmup_epochs=0)
    base.train_pfd(train_samples, epochs=epochs, verbose=False)

    def auc_of(scorer):
        return _auc([(scorer(u, i, p), y) for u, i, y, p in test_samples])

    t_auc = auc_of(lambda u, i, p: pfd.teacher_score(u, i, p))
    s_auc = auc_of(lambda u, i, p: pfd.score(u, i))
    b_auc = auc_of(lambda u, i, p: base.score(u, i))
    b_t_auc = auc_of(lambda u, i, p: base.teacher_score(u, i, p))
    print("\n[3] 测试集 AUC 对比")
    print(f"  教师 f_T(x, x_priv)       ：{t_auc:.4f}   ← 吃特权特征，线上不可用")
    print(f"  PFD 学生 f_S(x)           ：{s_auc:.4f}   ← 线上实际部署的粗排模型")
    print(f"  无蒸馏学生 f_S(x), λ=0    ：{b_auc:.4f}")

    # 与教师的排序一致性：学生 Top-5 与教师 Top-5 的平均重合度
    priv_of = {(u, i): p for u, i, _, p in all_samples}

    def agree_with_teacher(student: "PFD", k: int = 5) -> float:
        tot = 0.0
        for u in range(n_users):
            t_top = sorted(range(n_items), key=lambda i: -pfd.teacher_score(u, i, priv_of[(u, i)]))[:k]
            s_top = sorted(range(n_items), key=lambda i: -student.score(u, i))[:k]
            tot += len(set(t_top) & set(s_top)) / k
        return tot / n_users

    print("\n[4] 学生与教师的排序一致性（Top-5 平均重合度）")
    print(f"  PFD 学生：{agree_with_teacher(pfd):.3f}    无蒸馏学生：{agree_with_teacher(base):.3f}")
    uid = 0
    cands = [i for i in range(n_items) if i not in {i2 for u, i2, y, _ in train_samples if u == uid and y == 1}]
    print(f"\n[5] 用户 {uid} 粗排 Top-5（学生模型，零额外在线开销）")
    for iid, s in pfd.pre_rank(uid, cands, top_k=5):
        print(f"  item {iid:2d}: {s:.4f}")
