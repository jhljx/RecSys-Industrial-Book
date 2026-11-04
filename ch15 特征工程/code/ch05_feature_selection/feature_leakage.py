"""
特征泄露：时间边界与 Point-in-Time 正确性（第15章 §15.6.6）

正文给了两个必须满足的关系：
    特征可用信息  F(t) ⊆ I_{≤t}      （只能用预测时刻之前已经产生的信息）
    行为序列      ActionList_at_t ⊂ ActionList_current

脚本把这两条关系写成三类可复现的"泄露 vs 正确"对照：

1) 未来行为泄露
   构造用户逐小时的行为流，样本为"某小时的曝光是否点击"。
   - 泄露特征：用"该用户当天剩余时间的行为次数"（未来信息）
   - 正确特征：用"截止到曝光时刻之前的行为次数"（point-in-time）
   看 AUC 差距有多大。

2) 统计窗口泄露
   - 错误：`sum(actions[t-7d : t+7d])`（窗口向后越界）
   - 正确：`sum(actions[t-7d : t])`
   同时演示"离线任务在 t 之后统一回填当天统计"这一类隐蔽错误。

3) Action List 时间错位
   模仿训练任务在"今天"读取用户当前最新序列，去解释"昨天"的样本：
   - 当前序列长度 vs 时刻 t 的序列长度
   - 用两份序列算出的兴趣表示在相同样本上的差异
   这正是正文说的 `ActionList_current ⊃ ActionList_at_t`。

运行：python3 ch05_feature_selection/feature_leakage.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _lr_utils import auc_score, sigmoid, standardize  # noqa: E402

N_USERS = 300
N_HOURS = 24 * 30          # 30 天，逐小时
BASE_RATE = 0.08


def sigmoid_np(x):
    return sigmoid(x)


def train_eval(Xtr, ytr, Xva, yva, steps=500, lr=0.5, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    (Xtr_s, Xva_s), _, _ = standardize(Xtr, Xva)
    n, d = Xtr_s.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        idx = rng.integers(0, n, 512)
        err = sigmoid_np(Xtr_s[idx] @ w + b) - ytr[idx]
        w -= lr * (Xtr_s[idx].T @ err / 512 + l2 * w)
        b -= lr * err.mean()
    return auc_score(Xva_s @ w + b, yva)


def main() -> None:
    print("特征泄露：F(t) ⊆ I_{≤t} 与 ActionList 时间错位（§15.6.6）\n")
    rng = np.random.default_rng(2026)

    # ---------------- 1) 未来行为泄露 ----------------
    # 每个用户有一个"兴趣强度"，决定基础点击率
    strength = rng.gamma(2.0, 0.5, N_USERS)
    n_samples = 20_000
    users = rng.integers(0, N_USERS, n_samples)
    t_idx = rng.integers(0, N_HOURS, n_samples)
    # 点击概率 = 用户强度 × 时段效应 + 噪声
    hour_of_day = t_idx % 24
    time_effect = 1.0 + 0.4 * np.sin(hour_of_day / 24 * 2 * np.pi)
    p = np.clip(BASE_RATE * strength[users] * time_effect, 0.005, 0.6)
    y = (rng.random(n_samples) < p).astype(float)

    # 正确的 point-in-time 特征：截至 t 之前 24h 的行为数
    hist_24h = rng.poisson(8 * strength[users]).astype(float)
    # 未来泄露特征：t 之后 24h 的行为数。它和 Label 共享同一个"用户强度"，
    # 因此"未来行为多"与"当前点击"高度相关——离线看起来很有用，线上拿不到。
    future_24h = rng.poisson(8 * strength[users] * (1 + 1.2 * (y - BASE_RATE))).astype(float)

    split = n_samples // 2
    X_pit = hist_24h.reshape(-1, 1)
    X_leak = np.stack([hist_24h, future_24h], axis=1)

    auc_pit = train_eval(X_pit[:split], y[:split], X_pit[split:], y[split:])
    auc_leak = train_eval(X_leak[:split], y[:split], X_leak[split:], y[split:])
    print("① 未来行为泄露：")
    print(f"  正确（仅 t 之前 24h 行为）          val AUC = {auc_pit:.4f}")
    print(f"  泄露（额外加入 t 之后 24h 行为）    val AUC = {auc_leak:.4f}")
    print(f"  差值 = {auc_leak - auc_pit:+.4f} —— 这个提升在线上会完全消失。")
    print("  注意：泄露特征在离线评估里'看起来更准'，而这正是它危险的地方。")
    print("  补充：如果把 Label 本身直接混进特征，AUC 会接近 1.0；现实中很少有")
    print("  这么直白的写法，更多是'未来窗口''未来聚合'这类隐蔽形式。\n")

    # ---------------- 2) 统计窗口泄露 ----------------
    # 构造一个 30 天逐小时的"点击序列"：每个用户有自己的点击倾向，
    # 这样"过去的点击密度"才真的能预测"下一次是否点击"
    user_rate = rng.gamma(2.0, 0.015, N_USERS)
    clicks = (rng.random((N_USERS, N_HOURS)) < user_rate[:, None]).astype(float)
    window = 7 * 24
    t_choices = rng.integers(window, N_HOURS - 2, n_samples)

    def window_sum(arr, users, ts, start_off, end_off):
        out = np.zeros(len(users))
        for k, (u, t) in enumerate(zip(users, ts)):
            a = clicks[u]
            lo, hi = t + start_off, t + end_off
            out[k] = a[max(0, lo):max(0, hi)].sum()
        return out

    # 标签 = 曝光后这一小时用户有没有点击（Label 发生在 t 之后）
    y2 = clicks[users, t_choices + 1]

    # 正确特征只看 [t−7d, t)；错误特征把 t 之后（含标签所在小时）也算进来
    feat_correct = window_sum(clicks, users, t_choices, -window, 0)       # [t−w, t)
    feat_wrong = window_sum(clicks, users, t_choices, -window, 2)         # [t−w, t+2h)
    split = n_samples // 2
    auc_c = train_eval(feat_correct.reshape(-1, 1)[:split], y2[:split],
                       feat_correct.reshape(-1, 1)[split:], y2[split:])
    auc_w = train_eval(feat_wrong.reshape(-1, 1)[:split], y2[:split],
                       feat_wrong.reshape(-1, 1)[split:], y2[split:])
    print("② 统计窗口泄露：")
    print(f"  正确窗口 [t−7d, t)     val AUC = {auc_c:.4f}")
    print(f"  错误窗口 [t−7d, t+2h)  val AUC = {auc_w:.4f}")
    print(f"  差值 = {auc_w - auc_c:+.4f}")
    print("  越界窗口把'这次曝光之后的点击'直接算了进去，等于把 Label 当特征；")
    print("  离线 AUC 被抬高，线上只有 [t−7d, t)，效果对不上（§15.6.6）。")
    print("  真实系统里常见的形式是：离线任务按天重算'过去 7 天'，")
    print("  于是当天样本拿到的窗口包含了当天稍晚才发生的行为。\n")

    # ---------------- 3) Action List 时间错位 ----------------
    # 用户 A 在 t 时刻有 80 条历史行为，之后又产生了很多条
    rng2 = np.random.default_rng(7)
    cat_pool = 8
    seq_len_at_t = 80
    seq_len_now = 14_000
    cats = rng2.integers(0, cat_pool, seq_len_now)
    # 兴趣漂移：前 80 条偏类目 0，最新 14000 条偏类目 5
    cats[:seq_len_at_t] = rng2.choice([0, 0, 0, 1, 1, 2], seq_len_at_t)
    cats[seq_len_at_t:] = rng2.choice([5, 5, 5, 6, 7], seq_len_now - seq_len_at_t)

    def interest_vec(seq, n_cat=cat_pool):
        v = np.bincount(seq, minlength=n_cat).astype(float)
        return v / v.sum()

    v_at_t = interest_vec(cats[:seq_len_at_t])
    v_now = interest_vec(cats)
    cos = float(np.dot(v_at_t, v_now) / (np.linalg.norm(v_at_t) * np.linalg.norm(v_now)))

    print("③ Action List 时间错位：")
    print(f"  样本时刻 t 的序列长度     : {seq_len_at_t}")
    print(f"  训练任务读取的当前序列长度: {seq_len_now:,}")
    print(f"  两者兴趣分布的余弦相似度  : {cos:.4f}")
    print("  时刻 t 的主兴趣            : 类目 %d（占比 %.0f%%）"
          % (v_at_t.argmax(), v_at_t.max() * 100))
    print("  当前主兴趣                 : 类目 %d（占比 %.0f%%）"
          % (v_now.argmax(), v_now.max() * 100))
    print("  当前序列把 t 时刻之后的大量新行为也带进了昨天样本的特征里：")
    print("  ActionList_current ⊃ ActionList_at_t，离线训练用的输入在线上根本不存在。\n")

    print("排查清单（§15.6.6）：")
    print("  1) 每个特征写明产出时间、可用时间，确认 ≤ 预测时刻；")
    print("  2) 归一化/分箱/目标编码只用训练集统计量，禁止全量数据先算后切；")
    print("  3) 行为序列按样本时间截断，不能直接读最新完整序列；")
    print("  4) 实时特征要有延迟兜底：线上取不到时用旧值，而不是取未来值；")
    print("  5) 上线前做一次'时间穿越测试'：故意把特征延后一小时，看 AUC 是否骤降。")


if __name__ == "__main__":
    main()
