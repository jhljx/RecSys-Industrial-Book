"""
特征交互实验的公共部分（第15章 §15.5）

这里只做三件事：生成一份"交互确实有用"的合成 CTR 数据、跑一个简单的训练循环、
算 AUC 和参数量。各模型脚本只关心自己的网络结构。

数据里埋了一个重要的交互项：用户 ID 与物品 ID 的配对决定点击概率
（同一批 user 对某类 item 天生偏好更高），这是纯加性 MLP 学得慢、显式交互
结构学得快的原因。
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn

# field 名称 -> 词表大小
FIELD_SIZES = {
    "user": 200,
    "item": 400,
    "category": 8,
    "author": 40,
    "gender": 2,
    "age_band": 5,
    "city_tier": 4,
    "hour_bucket": 6,
    "device": 3,
    "interest_match": 2,
}
FIELD_NAMES = list(FIELD_SIZES)
N_FIELDS = len(FIELD_NAMES)


def make_dataset(n: int, seed: int, chunk: int = 8000):
    """返回 (sparse features [n, F] int64, labels [n] float)。分批生成避免大内存。"""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_FIELDS), dtype=np.int64)
    y = np.zeros(n, dtype=np.float32)
    # 用户与物品各自的隐向量 + 类目表
    user_pref = rng.normal(0, 1, size=(FIELD_SIZES["user"], 6))
    user_ui = rng.normal(0, 1, size=(FIELD_SIZES["user"], 4))
    item_vec = rng.normal(0, 1, size=(FIELD_SIZES["item"], 6))
    item_ui = rng.normal(0, 1, size=(FIELD_SIZES["item"], 4))
    item_cat = rng.integers(0, FIELD_SIZES["category"], size=FIELD_SIZES["item"])
    item_author = rng.integers(0, FIELD_SIZES["author"], size=FIELD_SIZES["item"])

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        b = end - start
        users = rng.integers(0, FIELD_SIZES["user"], b)
        items = rng.integers(0, FIELD_SIZES["item"], b)
        cats = item_cat[items]
        authors = item_author[items]
        genders = rng.integers(0, 2, b)
        ages = rng.integers(0, 5, b)
        cities = rng.integers(0, 4, b)
        hours = rng.integers(0, 6, b)
        devices = rng.integers(0, 3, b)

        # 交互项：用户偏好与物品向量的点积；超过阈值才显著正向
        match = np.einsum("ij,ij->i", user_pref[users], item_vec[items])
        interest = (match > 0.6).astype(np.int64)
        # 纯 pair 交互：只在 user-item 隐向量同向时产生，加性 MLP 无法表达
        ui = np.einsum("ij,ij->i", user_ui[users], item_ui[items])
        ui_strong = (ui > 1.2).astype(np.int64)

        logit = (
            -1.0
            + 0.3 * match
            + 0.9 * interest                 # 阈值门控
            + 1.4 * ui_strong                # 真正的 user×item 二阶交互
            + 0.5 * (cats == (ages % 8))     # 年龄与类目的交叉
            + 0.25 * (hours >= 4)
            - 0.3 * (cities == 3)
            + rng.normal(0, 0.2, b)
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        label = (rng.random(b) < p).astype(np.float32)

        X[start:end] = np.stack(
            [users, items, cats, authors, genders, ages, cities, hours, devices, interest], axis=1
        )
        y[start:end] = label
    return X, y


def to_tensor(X: np.ndarray, y: np.ndarray, device: str = "cpu"):
    return torch.from_numpy(X).long().to(device), torch.from_numpy(y).float().to(device)


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit(
    model: nn.Module,
    Xtr, ytr, Xva, yva,
    steps: int = 600,
    batch: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-5,
    log_every: int = 100,
    name: str = "",
):
    """训练并返回 (best_val_auc, history)。模型 forward 接收 [B, F] 的 long 张量。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    history = []
    best_auc = 0.0
    rng = np.random.default_rng(0)
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        xb, yb = Xtr[idx], ytr[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xva).detach().cpu().numpy()
                train_logits = model(Xtr[:4096]).detach().cpu().numpy()
            model.train()
            train_auc = auc_score(train_logits, ytr[:4096].numpy())
            val_auc = auc_score(val_logits, yva.numpy())
            best_auc = max(best_auc, val_auc)
            history.append((step, float(loss), train_auc, val_auc))
            if name:
                print(f"  [{name}] step {step:4d}  loss={float(loss):.4f}  "
                      f"train_auc={train_auc:.4f}  val_auc={val_auc:.4f}")
    return best_auc, history


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class MLP(nn.Module):
    """通用小 MLP。out_dim=None 时只输出最后一层隐层表示（用于双塔式拼接）。"""

    def __init__(self, in_dim: int, hidden=(64, 32), out_dim: int | None = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        if out_dim is not None:
            layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def embedding_table(dim: int, seed: int = 0) -> nn.ModuleDict:
    """为每个 field 建一个 Embedding，统一维度 dim。"""
    torch.manual_seed(seed)
    return nn.ModuleDict({name: nn.Embedding(size, dim) for name, size in FIELD_SIZES.items()})


def stack_embeddings(table: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
    """[B, F] -> [B, F, d]。"""
    return torch.stack([table[name](x[:, i]) for i, name in enumerate(FIELD_NAMES)], dim=1)


class BaselineMLP(nn.Module):
    """纯 Embedding + Concat + MLP，作为所有交互模型的对照。"""

    def __init__(self, dim: int = 4, hidden=(32,), seed: int = 0, dropout: float = 0.2):
        super().__init__()
        self.emb = embedding_table(dim, seed)
        self.mlp = MLP(dim * N_FIELDS, hidden, out_dim=1, dropout=dropout)

    def forward(self, x, return_repr: bool = False):
        e = stack_embeddings(self.emb, x).flatten(1)
        logits = self.mlp(e).squeeze(-1)
        return (logits, e) if return_repr else logits


def run_experiment(model_factory, name: str, steps: int = 600, lr: float = 2e-3, seed: int = 0):
    """统一入口：生成数据、训练、打印对比。"""
    torch.manual_seed(seed)
    Xtr, ytr = make_dataset(40_000, seed=1)
    Xva, yva = make_dataset(8_000, seed=2)
    Xtr, ytr = to_tensor(Xtr, ytr)
    Xva, yva = to_tensor(Xva, yva)

    print(f"[{name}] 训练 {steps} 步，batch=512，lr={lr}")
    model = model_factory()
    n_params = count_params(model)
    best_auc, history = fit(model, Xtr, ytr, Xva, yva, steps=steps, lr=lr, name=name)
    print(f"  [{name}] 参数量={n_params:,}  最佳 val AUC={best_auc:.4f}")
    return model, best_auc, history


def print_comparison(results: list[tuple[str, float, int]]) -> None:
    print(f"\n{'模型':<22}{'参数量':>12}{'val AUC':>12}")
    print("-" * 46)
    for name, auc, params in results:
        print(f"{name:<22}{params:>12,}{auc:>12.4f}")
    best = max(results, key=lambda r: r[1])
    print(f"\n最佳：{best[0]}（AUC {best[1]:.4f}）—— 注意这是小数据快速实验，"
          "数值只用于横向对照，不代表线上收益。")
