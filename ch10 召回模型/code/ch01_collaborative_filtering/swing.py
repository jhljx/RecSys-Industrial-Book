"""
Swing 召回（第10章 §10.1.4）

Swing 在 ItemCF 余弦相似度基础上增加了用户对重叠的惩罚项：
若两个用户 A、B 本身就有大量共同交互物品，则对 (i,j) 共现的贡献应打折扣，
从而避免"同一圈子"用户反复强化热门物品对之间的虚假相似度。

Swing 相似度：
    S(i, j) = Σ_{u,v ∈ N(i)∩N(j)} 1 / (α + |N(u) ∩ N(v)|)
"""

from collections import defaultdict


def build_swing_similarity(
    user_items: dict[str, list[str]],
    alpha: float = 5.0,
    top_k: int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """
    离线计算 Swing 物品相似度。

    Args:
        user_items: {user_id: [item_id, ...]}
        alpha: 平滑系数，防止分母为零
        top_k: 每个物品保留的 Top-K 邻居
    """
    # 物品→用户集合
    item_users: dict[str, set[str]] = defaultdict(set)
    # 用户→物品集合（方便计算用户间交集）
    user_item_set: dict[str, set[str]] = {u: set(items) for u, items in user_items.items()}

    for user, items in user_items.items():
        for item in items:
            item_users[item].add(user)

    item_sim: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    items = list(item_users.keys())

    for idx, i in enumerate(items):
        for j in items[idx + 1 :]:
            common_users = item_users[i] & item_users[j]
            if not common_users:
                continue
            score = 0.0
            common_list = list(common_users)
            for a in range(len(common_list)):
                for b in range(a + 1, len(common_list)):
                    u, v = common_list[a], common_list[b]
                    uv_common = len(user_item_set[u] & user_item_set[v])
                    score += 1.0 / (alpha + uv_common)
            item_sim[i][j] += score
            item_sim[j][i] += score

    result: dict[str, list[tuple[str, float]]] = {}
    for item, neighbors in item_sim.items():
        ranked = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        result[item] = ranked[:top_k]
    return result


def recall_swing(
    user_id: str,
    user_items: dict[str, list[str]],
    item_sim: dict[str, list[tuple[str, float]]],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """在线召回：与 ItemCF 召回逻辑相同，相似度表换成 Swing 版本。"""
    history = set(user_items.get(user_id, []))
    score: dict[str, float] = defaultdict(float)
    for j in history:
        for i, sim in item_sim.get(j, []):
            if i not in history:
                score[i] += sim
    return sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    user_items = {
        "u1": ["i1", "i2", "i3"],
        "u2": ["i1", "i2", "i4"],
        "u3": ["i2", "i3", "i5"],
        "u4": ["i1", "i3", "i5"],
        "u5": ["i4", "i5"],
    }
    sim = build_swing_similarity(user_items, alpha=5.0, top_k=5)
    print("Swing 相似度（Top-3）：")
    for item in ["i1", "i2"]:
        print(f"  {item}: {sim.get(item, [])[:3]}")

    result = recall_swing("u1", user_items, sim, top_n=5)
    print("\nu1 的 Swing 召回结果：")
    for item, s in result:
        print(f"  {item}: {s:.4f}")
