"""
ItemCF 召回 —— 基于物品的协同过滤（第10章 §10.1.1）

核心思想：统计物品间共现用户数，用余弦相似度衡量物品相似性；
召回时查询用户交互物品的相似物品列表并累加得分。
"""

import math
from collections import defaultdict


def build_item_similarity(user_items: dict[str, list[str]], top_k: int = 20) -> dict[str, list[tuple[str, float]]]:
    """
    离线计算物品间相似度。

    Args:
        user_items: {user_id: [item_id, ...]}，每个用户的交互物品列表
        top_k: 每个物品保留的最相似邻居数

    Returns:
        {item_i: [(item_j, sim), ...]} 按相似度降序排列
    """
    # 统计物品交互用户数 N(i) 以及共现次数 |N(i) ∩ N(j)|
    item_user_count: dict[str, int] = defaultdict(int)   # N(i)
    co_occur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, items in user_items.items():
        for i in items:
            item_user_count[i] += 1
        # 枚举用户行为物品对
        for i in items:
            for j in items:
                if i != j:
                    co_occur[i][j] += 1

    # 余弦相似度：|N(i) ∩ N(j)| / sqrt(N(i) * N(j))
    item_sim: dict[str, list[tuple[str, float]]] = {}
    for i, neighbors in co_occur.items():
        sims = []
        for j, cnt in neighbors.items():
            sim = cnt / math.sqrt(item_user_count[i] * item_user_count[j])
            sims.append((j, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        item_sim[i] = sims[:top_k]
    return item_sim


def recall_itemcf(
    user_id: str,
    user_items: dict[str, list[str]],
    item_sim: dict[str, list[tuple[str, float]]],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    在线召回：根据用户历史行为和物品相似度表打分并排序。

    得分：p(u, i) = Σ_{j ∈ N(u)} S(i, j)
    """
    history = set(user_items.get(user_id, []))
    score: dict[str, float] = defaultdict(float)
    for j in history:
        for i, sim in item_sim.get(j, []):
            if i not in history:
                score[i] += sim
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

    item_sim = build_item_similarity(user_items, top_k=5)
    print("物品相似度（Top-3）：")
    for item in ["i1", "i2", "i3"]:
        print(f"  {item}: {item_sim.get(item, [])[:3]}")

    result = recall_itemcf("u1", user_items, item_sim, top_n=5)
    print("\nu1 的 ItemCF 召回结果：")
    for item, score in result:
        print(f"  {item}: {score:.4f}")
