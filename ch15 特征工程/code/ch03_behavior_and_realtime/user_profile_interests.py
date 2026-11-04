"""
长短期兴趣画像与多兴趣表示（第15章 §15.3）

正文指出：
- 用户同时具有长期兴趣与短期兴趣，二者需要分开建模；
- 一个用户往往有多个互不相关的兴趣方向，可用多兴趣表示
  E_u = {e_u^1, ..., e_u^K}。

本脚本在合成行为上构造：
1) 长期（30d）与短期（3d）类目分布，用 KL 散度量化"兴趣漂移"；
2) 用简单的 spherical k-means 从用户行为 item 的类目向量中抽出 K 个兴趣中心；
3) 比较"全历史平均向量"与"多兴趣向量（取与候选最相似的那个）"在召回时的差异。

运行：python3 ch03_behavior_and_realtime/user_profile_interests.py
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import entropy, make_events, print_table  # noqa: E402

N_CAT = 8
NOW = 30.0


def category_dist(events: list[dict], uid: int, start: float, end: float) -> np.ndarray:
    counts = np.zeros(N_CAT)
    for e in events:
        if e["user_id"] != uid or e["action"] not in ("click", "watch"):
            continue
        if start <= e["day"] < end:
            counts[e["category"]] += 1
    return counts


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q)，内部加平滑避免 log0。"""
    p = p + 1e-9
    q = q + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def interest_centers(vectors: np.ndarray, k: int, rng: random.Random, iters: int = 30) -> np.ndarray:
    """spherical k-means：归一化后用余弦相似度分配，中心再归一化。"""
    n = len(vectors)
    if n <= k:
        return vectors
    normalize = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    V = normalize(vectors)
    idx = rng.sample(range(n), k)
    C = V[idx].copy()
    for _ in range(iters):
        sim = V @ C.T
        assign = sim.argmax(1)
        newC = np.zeros_like(C)
        for j in range(k):
            members = V[assign == j]
            if len(members) == 0:
                newC[j] = V[rng.randrange(n)]
            else:
                newC[j] = members.mean(0)
        C = normalize(newC)
    return C


def main() -> None:
    events = make_events(80, 12_000, 400, N_CAT, 30, random.Random(11))
    # 给 item 一个 8 维的类目 one-hot 作为"内容向量"
    rng = random.Random(42)

    # ---- 1) 长短期漂移 ----
    rows = []
    drifts = []
    for uid in range(10):
        long_counts = category_dist(events, uid, 0, NOW)
        short_counts = category_dist(events, uid, NOW - 3, NOW)
        if long_counts.sum() == 0:
            continue
        p_long = long_counts / long_counts.sum()
        p_short = short_counts / short_counts.sum() if short_counts.sum() else p_long
        d = kl(p_short, p_long)
        drifts.append(d)
        main_long = int(p_long.argmax())
        main_short = int(p_short.argmax())
        # 兜底 max(0, ·)：浮点误差可能让接近 0 的熵显示成 -0.00
        h_long = max(0.0, entropy(list(p_long)))
        h_short = max(0.0, entropy(list(p_short)))
        rows.append([uid, main_long, main_short, f"{h_long:.2f}",
                     f"{h_short:.2f}", f"{d:.3f}",
                     "漂移" if main_long != main_short else "稳定"])
    print("一、长期(30d) vs 短期(3d) 的类目兴趣")
    print_table(["uid", "长期主类目", "短期主类目", "H_long", "H_short", "KL(短||长)", "状态"], rows)
    print(f"  平均 KL = {np.mean(drifts):.3f}；KL 越大说明近期兴趣偏离长期画像越远，")
    print("  这类用户应更多依赖短期序列特征（§15.3）。")

    # ---- 2) 多兴趣中心 ----
    print("\n二、多兴趣表示（K=3 的 spherical k-means）")
    rows = []
    for uid in range(5):
        cats = []
        for e in events:
            if e["user_id"] == uid and e["action"] in ("click", "watch"):
                v = np.zeros(N_CAT)
                v[e["category"]] = 1.0
                cats.append(v)
        if len(cats) < 5:
            continue
        noise = np.random.default_rng(uid)
        vectors = np.stack([c + noise.normal(0, 0.05, N_CAT) for c in cats])
        C = interest_centers(vectors, 3, random.Random(uid))
        # 每个中心的"主类目"= 分量最大者
        tops = [int(np.argmax(c)) for c in C]
        weights = [float(np.sum(np.argmax(vectors @ C.T, 1) == j)) for j in range(3)]
        wsum = sum(weights)
        rows.append([uid, len(cats), " / ".join(map(str, tops)),
                     " / ".join(f"{w / wsum:.2f}" for w in weights)])
    print_table(["uid", "行为数", "3 个兴趣中心的主类目", "中心占比"], rows)
    print("  一个用户被拆成 3 个兴趣向量后，召回时可以用'最匹配的那一个'参与打分，")
    print("  而不是被一个巨大的平均向量稀释掉小众兴趣。")

    # ---- 3) 单一向量 vs 多兴趣的召回对比 ----
    print("\n三、召回打分：平均向量 vs 多兴趣（取 max）")
    rng_np = np.random.default_rng(0)
    rows = []
    for uid in range(20):
        cats = []
        for e in events:
            if e["user_id"] == uid and e["action"] in ("click", "watch"):
                v = np.zeros(N_CAT)
                v[e["category"]] = 1.0
                cats.append(v)
        if len(cats) < 10:
            continue
        V = np.stack(cats)
        avg = V.mean(0)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        C = interest_centers(V, 3, random.Random(uid))
        # 100 个候选 item，随机内容向量
        cand = rng_np.normal(size=(100, N_CAT))
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        s_avg = cand @ avg
        s_multi = (cand @ C.T).max(1)
        # 用"top-10 候选的类目集中度"衡量召回多样性
        def concentration(scores):
            top = np.argsort(-scores)[:10]
            return entropy(list(np.bincount(top % N_CAT, minlength=N_CAT) + 1e-9))
        rows.append([uid, f"{s_avg.mean():+.3f}", f"{s_multi.mean():+.3f}",
                     f"{s_avg.max():+.3f}", f"{s_multi.max():+.3f}",
                     f"{concentration(s_avg):.2f}", f"{concentration(s_multi):.2f}"])
        if len(rows) >= 8:
            break
    print_table(["uid", "avg均值", "multi均值", "avg最大", "multi最大", "avg多样性", "multi多样性"], rows)
    print("  multi 的最大分通常更高（小众兴趣不被平均掉），Top-10 的类目熵也更分散。")
    print("  这就是正文'多兴趣表示 e_u^k'在召回/排序阶段的实际作用（§15.3）。")


if __name__ == "__main__":
    main()
