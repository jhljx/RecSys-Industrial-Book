"""
第15章 公共工具：合成数据的生成与通用小函数

本文件不实现任何模型，只提供各脚本共享的合成数据构造器，避免每个文件重复写
一套"造数据"的代码。所有数据均为进程内随机生成，不读取外部文件，也不联网。

约定：
- 固定随机种子，输出可复现；
- 只依赖 Python 3.12 标准库（interaction/ 子目录下的 PyTorch 脚本除外）；
- 所有打印都以"表格 + 数值"的方式呈现，不依赖 matplotlib。
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# 一个带类别偏向的简化 Softmax 采样器
# ---------------------------------------------------------------------------


def softmax_sample(logits: list[float], rng: random.Random) -> int:
    """按 softmax(logits) 采样一个下标。用于合成'用户兴趣'行为。"""
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    z = sum(exps)
    r = rng.random() * z
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r <= acc:
            return i
    return len(logits) - 1


# ---------------------------------------------------------------------------
# 合成用户与物品
# ---------------------------------------------------------------------------


def make_catalog(
    n_items: int = 400,
    n_categories: int = 8,
    rng: random.Random | None = None,
) -> list[dict]:
    """构造一个简单的内容池：每个 item 有类目、作者、物理时长、质量分。"""
    rng = rng or random.Random(42)
    items = []
    for i in range(n_items):
        items.append(
            {
                "item_id": i,
                "category": rng.randrange(n_categories),
                "author": rng.randrange(max(8, n_items // 20)),
                "duration": round(rng.lognormvariate(3.4, 0.7), 1),  # 秒
                "quality": round(rng.random(), 4),
                # 长尾：少数热门 item 拥有更高的亲和度
                "popularity": round(rng.paretovariate(1.6), 4),
            }
        )
    return items


def make_users(
    n_users: int = 200,
    n_categories: int = 8,
    rng: random.Random | None = None,
) -> list[dict]:
    """构造用户画像：每个用户有一个主兴趣类目与活跃度。"""
    rng = rng or random.Random(7)
    users = []
    for u in range(n_users):
        main = rng.randrange(n_categories)
        logits = [0.4] * n_categories
        logits[main] += 1.6
        users.append(
            {
                "user_id": u,
                "main_category": main,
                "intensity": round(rng.uniform(0.2, 1.0), 4),
                "age_band": rng.randrange(5),
                "gender": rng.randrange(2),
                "city_tier": rng.choice([1, 2, 3, 3, 4]),
            }
        )
    return users


# ---------------------------------------------------------------------------
# 行为事件流：带时间戳、行为类型、观看时长
# ---------------------------------------------------------------------------

ACTION_TYPES = ("exposure", "click", "watch", "like", "comment", "share", "follow")


def make_events(
    n_users: int = 200,
    n_events: int = 20_000,
    n_items: int = 400,
    n_categories: int = 8,
    days: int = 30,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    生成带时间戳的行为事件。

    每个事件形如：
        {"day": float, "user_id": int, "item_id": int, "action": str,
         "value": float, "category": int}

    其中 value 对 watch 事件是观看秒数，对 like/comment 等是 1。
    行为发生的概率依赖用户主兴趣类目，因此后续统计类特征确实"有信号"。
    """
    rng = rng or random.Random(2024)
    users = make_users(n_users, n_categories, rng)
    items = make_catalog(n_items, n_categories, rng)
    # 按主类目建立 item 池，便于按兴趣采样
    pools: dict[int, list[int]] = defaultdict(list)
    for it in items:
        pools[it["category"]].append(it["item_id"])

    events = []
    for _ in range(n_events):
        user = users[rng.randrange(n_users)]
        day = rng.uniform(0, days)
        # 近期活跃度更高：类似真实业务的用户活跃衰减
        if rng.random() > math.exp(-day / (days * 0.6)):
            day = rng.uniform(0, days)
        cat_logits = [0.25] * n_categories
        cat_logits[user["main_category"]] += 1.8 * user["intensity"]
        cat = softmax_sample(cat_logits, rng)
        item_id = pools[cat][rng.randrange(len(pools[cat]))]
        item = items[item_id]

        # 曝光 -> 点击 -> 观看 -> 互动 的简化漏斗
        click_p = 0.08 + 0.35 * user["intensity"] * (cat == user["main_category"])
        action = "exposure"
        value = 0.0
        if rng.random() < click_p:
            action = "click"
            base = item["duration"] * (0.25 + 0.65 * user["intensity"])
            value = round(min(item["duration"], max(0.3, rng.gauss(base, 4.0))), 2)
            if rng.random() < 0.25 + 0.3 * user["intensity"]:
                action = "watch"
            if rng.random() < 0.04 + 0.05 * item["quality"] * user["intensity"]:
                action = rng.choice(("like", "comment", "share", "follow"))
                value = 1.0
        events.append(
            {
                "day": round(day, 4),
                "user_id": user["user_id"],
                "item_id": item_id,
                "action": action,
                "value": value,
                "category": cat,
            }
        )
    events.sort(key=lambda e: e["day"])
    return events


# ---------------------------------------------------------------------------
# 用户行为序列切片
# ---------------------------------------------------------------------------


def user_action_lists(events: list[dict]) -> dict[int, list[dict]]:
    """把事件流按用户聚合成按时间升序的行为列表（用于序列特征）。"""
    seq: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e["action"] == "exposure":
            continue
        seq[e["user_id"]].append(e)
    for lst in seq.values():
        lst.sort(key=lambda x: x["day"])
    return dict(seq)


# ---------------------------------------------------------------------------
# 打印工具
# ---------------------------------------------------------------------------


def print_table(headers: list[str], rows: list[list]) -> None:
    """打印一个左对齐的文本表。"""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def gini(values: list[float]) -> float:
    """基尼系数，用于衡量分布集中度（0=完全均匀）。"""
    xs = sorted(v for v in values if v >= 0)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = 0.0
    total = sum(xs)
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2 * cum) / (n * total) - (n + 1) / n


def entropy(probabilities: list[float]) -> float:
    """香农熵（自然对数），概率列表内部会被归一化。"""
    s = sum(probabilities)
    if s <= 0:
        return 0.0
    ps = [p / s for p in probabilities if p > 0]
    return -sum(p * math.log(p) for p in ps)
