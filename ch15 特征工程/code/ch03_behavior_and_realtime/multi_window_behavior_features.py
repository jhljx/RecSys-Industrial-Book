"""
多时间窗口行为统计、时间衰减与 F/R/I 特征（第15章 §15.3、§15.1.4）

对应正文的两段内容：
- §15.1.4：Feature_1d / Feature_7d / Feature_30d，以及时间衰减 w(Δt)=exp(-λΔt)；
- §15.3：用频率（Frequency）、时效（Recency）、强度（Intensity）三个维度刻画行为。

本脚本在合成事件流上构造 1d/7d/30d 三个窗口的点击/观看/平均时长特征，并比较
"朴素计数"与"指数衰减加权"在同一批用户上的排序差异，说明窗口与衰减各自解决
什么问题。[MOCK] 事件流由 common.make_events 生成。

运行：python3 ch03_behavior_and_realtime/multi_window_behavior_features.py
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import make_events, print_table  # noqa: E402

NOW = 30.0  # 第 30 天作为"当前时刻"
WINDOWS = (1, 7, 30)


def window_features(events: list[dict], now: float = NOW) -> dict[int, dict[str, float]]:
    """为每个用户计算多窗口的 click/watch/avg_duration。"""
    feats: dict[int, dict[str, float]] = defaultdict(dict)
    for W in WINDOWS:
        clicks: dict[int, int] = defaultdict(int)
        watches: dict[int, int] = defaultdict(int)
        watch_secs: dict[int, float] = defaultdict(float)
        for e in events:
            if e["day"] < now - W:
                continue
            if e["action"] in ("click", "watch"):
                clicks[e["user_id"]] += 1
            if e["action"] == "watch":
                watches[e["user_id"]] += 1
                watch_secs[e["user_id"]] += e["value"]
        for uid in set(clicks) | set(watches):
            feats[uid][f"click_{W}d"] = C = clicks[uid]
            feats[uid][f"watch_{W}d"] = watches[uid]
            feats[uid][f"avg_dur_{W}d"] = watch_secs[uid] / watches[uid] if watches[uid] else 0.0
    return feats


def decay_features(events: list[dict], now: float = NOW) -> dict[int, dict[str, float]]:
    """指数衰减统计：w = exp(-λΔt)，Δt 以天计。"""
    out: dict[int, dict[str, float]] = defaultdict(lambda: {"w_click": 0.0, "w_watch": 0.0, "w_sec": 0.0})
    for e in events:
        dt = now - e["day"]
        if dt < 0:
            continue
        w = math.exp(-0.1 * dt)  # λ=0.1/天，半衰期约 6.9 天
        if e["action"] in ("click", "watch"):
            out[e["user_id"]]["w_click"] += w
        if e["action"] == "watch":
            out[e["user_id"]]["w_watch"] += w
            out[e["user_id"]]["w_sec"] += w * e["value"]
    for v in out.values():
        v["w_avg_dur"] = v["w_sec"] / v["w_watch"] if v["w_watch"] > 0 else 0.0
    return out


def main() -> None:
    events = make_events(200, 20_000, 400, 8, 30, random.Random(2024))
    raw = window_features(events)
    dec = decay_features(events)

    print("一、多窗口特征示例（前 8 个用户）")
    rows = []
    for uid in range(8):
        f = raw.get(uid, {})
        rows.append([
            uid,
            f.get("click_1d", 0), f.get("click_7d", 0), f.get("click_30d", 0),
            f.get("watch_7d", 0),
            f"{f.get('avg_dur_7d', 0):.1f}",
            f"{dec.get(uid, {}).get('w_click', 0):.2f}",
        ])
    print_table(["uid", "click_1d", "click_7d", "click_30d", "watch_7d", "avg_dur_7d(s)", "decay_click(λ=.1)"], rows)

    # ---- 窗口的作用：短期 vs 长期兴趣排序不同 ----
    print("\n二、'短期活跃 vs 长期活跃'的排序差异（全部 200 用户）")
    scores_1d = {u: raw.get(u, {}).get("click_1d", 0) for u in range(200)}
    scores_30d = {u: raw.get(u, {}).get("click_30d", 0) for u in range(200)}
    top1 = sorted(scores_1d, key=lambda u: -scores_1d[u])[:10]
    top30 = sorted(scores_30d, key=lambda u: -scores_30d[u])[:10]
    overlap = len(set(top1) & set(top30))
    print(f"  1d Top-10 与 30d Top-10 的交集：{overlap} / 10")
    print(f"  1d Top-10：{top1}")
    print(f" 30d Top-10：{top30}")
    print("  两个榜单不同，说明短期与长期统计captured的是不同尺度的兴趣，必须同时保留。")

    # ---- 衰减的作用：把"很久没来但历史很长"的用户排下去 ----
    print("\n三、计数 vs 衰减：同一用户的排名变化")
    counts = {u: raw.get(u, {}).get("click_30d", 0) for u in range(200)}
    decays = {u: dec.get(u, {}).get("w_click", 0) for u in range(200)}
    rank_c = {u: r for r, u in enumerate(sorted(counts, key=lambda x: -counts[x]), 1)}
    rank_d = {u: r for r, u in enumerate(sorted(decays, key=lambda x: -decays[x]), 1)}
    moved = sorted(range(200), key=lambda u: rank_c[u] - rank_d[u], reverse=True)[:5]
    rows = []
    for u in moved:
        rows.append([u, counts[u], f"{decays[u]:.2f}", rank_c[u], rank_d[u], rank_c[u] - rank_d[u]])
    print_table(["uid", "click_30d", "weighted", "计数排名", "衰减排名", "名次上升"], rows)
    print("  衰减把'历史多但近期不活跃'的用户往下压，这正是 w(Δt)=exp(-λΔt) 的目的。")

    # ---- F/R/I 三维 ----
    print("\n四、F / R / I 三维画像（前 8 个用户）")
    last_click = {}
    for e in events:
        if e["action"] in ("click", "watch"):
            last_click[e["user_id"]] = e["day"]
    rows = []
    for uid in range(8):
        f = raw.get(uid, {})
        freq = f.get("click_7d", 0) / 7.0
        rec = NOW - last_click.get(uid, 0.0)
        inten = f.get("avg_dur_7d", 0.0)
        rows.append([uid, f"{freq:.2f}", f"{rec:.2f}", f"{inten:.1f}",
                     "高频" if freq > 0.5 else "低频",
                     "近期活跃" if rec < 3 else "久未活跃",
                     "深度" if inten > 20 else "浅层"])
    print_table(["uid", "F(次/天)", "R(天)", "I(秒)", "频率", "时效", "强度"], rows)
    print("\n同一组数字可以组合出不同策略：高频+近期+深度=核心用户；低频+近=新兴趣；")
    print("高频+久未=流失预警。单一'点击次数'无法区分这些状态（§15.3）。")


if __name__ == "__main__":
    main()
