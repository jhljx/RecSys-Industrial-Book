"""
UserCF 召回 —— 基于用户的协同过滤（第10章 §10.1.2）

核心思想：计算用户间相似度，找到 Top-K 相似用户，
再把相似用户喜欢但目标用户未见过的物品推荐出来。
"""

import math
from collections import defaultdict


def build_user_similarity(
    user_items: dict[str, list[str]], top_k: int = 20
) -> dict[str, list[tuple[str, float]]]:
    """
    离线构建用户相似度表。

    相似度公式（余弦）：
        S(A,B) = |N(A) ∩ N(B)| / sqrt(|N(A)| * |N(B)|)
    """
    # 构建物品→用户倒排表，方便枚举共同交互用户
    item_users: dict[str, set[str]] = defaultdict(set)
    user_item_count: dict[str, int] = {}
    for user, items in user_items.items():
        user_item_count[user] = len(items)
        for item in items:
            item_users[item].add(user)

    # 统计用户对共现次数
    co_occur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, users in item_users.items():
        users_list = list(users)
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                a, b = users_list[i], users_list[j]
                co_occur[a][b] += 1
                co_occur[b][a] += 1

    # 余弦相似度
    user_sim: dict[str, list[tuple[str, float]]] = {}
    for a, neighbors in co_occur.items():
        sims = []
        for b, cnt in neighbors.items():
            sim = cnt / math.sqrt(user_item_count[a] * user_item_count[b])
            sims.append((b, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        user_sim[a] = sims[:top_k]
    return user_sim


def recall_usercf(
    user_id: str,
    user_items: dict[str, list[str]],
    user_sim: dict[str, list[tuple[str, float]]],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    在线召回：利用相似用户的行为为目标用户评分。

    得分：p(A, i) = Σ_{B ∈ N(A)} S(A,B) * R(B,i)
    """
    history = set(user_items.get(user_id, []))
    score: dict[str, float] = defaultdict(float)
    for sim_user, sim_val in user_sim.get(user_id, []):
        for item in user_items.get(sim_user, []):
            if item not in history:
                score[item] += sim_val
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user_items = {
        "u1": ["i1", "i2", "i3"],
        "u2": ["i1", "i2", "i4"],
        "u3": ["i2", "i3", "i5"],
        "u4": ["i1", "i3", "i5"],
        "u5": ["i4", "i5"],
    }

    user_sim = build_user_similarity(user_items, top_k=5)
    print("用户相似度（Top-3）：")
    for u in ["u1", "u2"]:
        print(f"  {u}: {user_sim.get(u, [])[:3]}")

    result = recall_usercf("u1", user_items, user_sim, top_n=5)
    print("\nu1 的 UserCF 召回结果：")
    for item, s in result:
        print(f"  {item}: {s:.4f}")
