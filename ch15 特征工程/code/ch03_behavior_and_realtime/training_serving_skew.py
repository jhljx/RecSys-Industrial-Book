"""
训练-服务偏差（Training-Serving Skew）与特征穿越（第15章 §15.4、§15.6.6）

正文给出两个关键事实：
- 训练用离线特征、推理用实时特征，若口径不同就产生 Training-Serving Skew；
- 行为序列若在训练时读取"当前最新"的 ActionList，就会看到
  ActionList_current ⊃ ActionList_at t，即特征穿越。

本脚本做两件事：
1) 模拟离线特征与在线特征在（延迟、缺失默认值、归一化方式）上的差异，
   量化离线 AUC 与线上真实 AUC 的落差；
2) 模拟"按样本时刻截断"与"直接读最新 ActionList"两种序列构造方式，
   展示穿越如何虚高离线指标。

运行：python3 ch03_behavior_and_realtime/training_serving_skew.py
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

N_USERS = 400
N_DAYS = 14
SWITCH_DAY = 7


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def auc(scores, labels) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos = np.asarray(labels).sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    return (ranks[np.asarray(labels) == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def simulate_users(rng: random.Random):
    """每个用户有一个前半段兴趣和后半段兴趣，用于制造'近期行为'信号。"""
    users = []
    for u in range(N_USERS):
        c1, c2 = rng.randrange(4), rng.randrange(4)
        while c2 == c1:
            c2 = rng.randrange(4)
        users.append({"c1": c1, "c2": c2, "days": []})
    # 每个用户每天产生若干带 (day, category) 的行为
    for u in users:
        for d in range(N_DAYS):
            n = rng.randint(0, 4)
            for _ in range(n):
                cat = u["c1"] if d < SWITCH_DAY else u["c2"]
                if rng.random() < 0.2:
                    cat = rng.randrange(4)
                u["days"].append((d * 24 + rng.randrange(24), cat))
    return users


def build_samples(users, rng):
    """
    每个样本 = 某用户某天的一条曝光。
    真实标签由"用户当前兴趣 == 候选物品类目"决定（兴趣在第 7 天切换）。
    特征 = 该用户截至样本时刻的行为类目分布。
    """
    samples = []
    for u, user in enumerate(users):
        for day in range(1, N_DAYS):
            cand_cat = rng.randrange(4)
            true_cat = user["c1"] if day < SWITCH_DAY else user["c2"]
            logit = -1.2 + 2.2 * (cand_cat == true_cat) + 0.3 * (user["c1"] == user["c2"])
            label = 1 if rng.random() < sigmoid(logit) else 0
            samples.append({"u": u, "day": day, "cand": cand_cat, "y": label})
    return samples


def features_until(users, sample, cutoff_hour: int | None = None) -> np.ndarray:
    """按时间截断构造 4 维类目计数特征；cutoff_hour=None 表示用全量（穿越）。"""
    user = users[sample["u"]]
    if cutoff_hour is None:
        events = user["days"]
    else:
        events = [e for e in user["days"] if e[0] < cutoff_hour]
    counts = np.zeros(4)
    for _, cat in events:
        counts[cat] += 1
    return counts


def main() -> None:
    rng = random.Random(42)
    users = simulate_users(rng)
    samples = build_samples(users, rng)
    labels = np.array([s["y"] for s in samples])
    print(f"样本 {len(samples)} 条，正样本率 {labels.mean():.3f}，"
          f"第 {SWITCH_DAY} 天兴趣切换")

    # ---------- 1) 序列构造：截断 vs 穿越 ----------
    feats_clean = np.stack([features_until(users, s, s["day"] * 24) for s in samples])
    feats_leak = np.stack([features_until(users, s, None) for s in samples])
    # 简单线性打分：候选类目对应的计数 - 其他类目均值
    def score(F, S):
        return np.array([F[i, s["cand"]] - (F[i].sum() - F[i, s["cand"]]) / 3
                         for i, s in enumerate(S)])

    print("\n一、特征穿越的影响（行为序列未按样本时刻截断）")
    print(f"  正确构造（截断到样本时刻）AUC = {auc(score(feats_clean, samples), labels):.4f}")
    print(f"  穿越构造（读取全量/最新序列）AUC = {auc(score(feats_leak, samples), labels):.4f}")
    print("  穿越版本的标签可以由'未来行为'解释，离线指标被系统性抬高；")
    print("  这部分增益线上拿不到，是典型的 Training-Serving Skew（§15.6.6）。")

    # ---------- 2) 离线 vs 在线口径差异 ----------
    print("\n二、离线/在线口径差异的三种来源")
    print("  (a) 延迟：在线特征落后 5~60 分钟")
    print("  (b) 缺失默认值：离线填 0，在线填全局均值")
    print("  (c) 归一化：离线用全量 min-max，在线用滑动窗口 min-max")

    # 构造四种在线版本，并与"完美离线"对比
    def build_online(F, s, lag_hours, missing_rate, norm_mode):
        i = samples.index(s)
        cut = s["day"] * 24 - lag_hours
        online = features_until(users, s, cut)
        # 缺失模拟
        if rng.random() < missing_rate:
            online = np.zeros(4) if norm_mode == "offline" else np.full(4, F[i].mean())
        return online

    rows = []
    for name, lag, miss, norm in (
        ("完全一致", 0, 0.0, "offline"),
        ("仅延迟 60 分钟", 1, 0.0, "offline"),
        ("仅缺失 10%（填 0）", 0, 0.10, "offline"),
        ("仅缺失 10%（填均值）", 0, 0.10, "online"),
        ("延迟 + 缺失 + 归一化", 1, 0.10, "online"),
    ):
        F = np.stack([build_online(feats_clean, s, lag, miss, norm) for s in samples])
        rows.append([name, auc(score(F, samples), labels)])

    print(f"\n{'在线特征版本':<26} {'线上真实 AUC':>12}")
    print("-" * 42)
    for name, a in rows:
        print(f"{name:<26} {a:>12.4f}")
    print("\n  单独一项差异影响不大，但'延迟 + 缺失 + 归一化'叠加后落差被放大；")
    print("  工程上要做的是逐字段对账（同一 user、同一请求，离线重算 vs 在线快照）。")

    # ---------- 3) 行为序列长度与穿越程度 ----------
    print("\n三、ActionList 长度与穿越程度（样本在重放时读到多少未来行为）")
    print(f"{'读取长度':>8} {'平均包含未来行为数':>18} {'AUC':>8}")
    print("-" * 40)
    for L in (10, 50, 200, 800):
        Fs = []
        for s in samples:
            user = users[s["u"]]
            past = [e for e in user["days"] if e[0] < s["day"] * 24]
            latest = user["days"][-L:]           # 直接读最新 L 条（穿越）
            if past:
                latest = [e for e in latest if e[0] >= past[-1][0] - 24] or past[-1:]
            n_future = sum(1 for e in latest if e[0] >= s["day"] * 24)
            Fs.append((latest, n_future))
        feats = []
        futures = []
        for (latest, nf) in Fs:
            c = np.zeros(4)
            for _, cat in latest:
                c[cat] += 1
            feats.append(c)
            futures.append(nf)
        feats = np.stack(feats)
        print(f"{L:>8} {np.mean(futures):>18.2f} {auc(score(feats, samples), labels):>8.4f}")
    print("\n  序列越长，'最新列表'里混入的未来行为越多，离线指标越虚高。")
    print("  正确做法是 Point-in-Time 截断：F(t) ⊆ I_{≤t}（§15.6.6）。")


if __name__ == "__main__":
    main()
