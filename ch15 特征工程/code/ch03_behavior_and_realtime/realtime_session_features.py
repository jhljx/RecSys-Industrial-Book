"""
离线 / 准实时 / 实时特征与 Session 特征（第15章 §15.4）

正文把特征按更新频率分成三类：离线（天级）、准实时（分钟/小时级）、实时
（秒级），并强调 Session 特征补充长期画像。本脚本模拟一条行为流，
分别产出三类特征，观察它们对"用户刚刚改变兴趣"的反应速度。

[MOCK] 行为流与特征写入均为进程内模拟，不依赖 Flink/Redis。

运行：python3 ch03_behavior_and_realtime/realtime_session_features.py
"""

from __future__ import annotations

import math
import random
from collections import deque

N_CAT = 4
CAT_NAME = ["体育", "游戏", "美食", "音乐"]
MINUTES = 18 * 60  # 模拟 18 小时，按分钟推进
SWITCH_MIN = 300   # 第 300 分钟兴趣从"体育"切到"游戏"


def top(counts: list[int]) -> str:
    s = sum(counts)
    if s == 0:
        return "无"
    i = max(range(len(counts)), key=lambda k: counts[k])
    return f"{CAT_NAME[i]}({counts[i] / s:.0%})"


def entropy(counts: list[int]) -> float:
    s = sum(counts)
    if s == 0:
        return 0.0
    return -sum((c / s) * math.log(c / s) for c in counts if c) + 0.0


def main() -> None:
    rng = random.Random(42)
    # 每分钟以 0.15 的概率发生一次观看行为（约每小时 9 条）
    stream = []
    for t in range(MINUTES):
        if rng.random() > 0.15:
            continue
        cat = 0 if t < SWITCH_MIN else 1
        if t >= SWITCH_MIN and rng.random() < 0.15:
            cat = 0  # 切换后仍有旧兴趣残留
        watch = max(1.0, rng.gauss(25 if cat == 1 else 18, 6))
        stream.append((t, cat, watch))

    # ---- 三类特征 ----
    offline_counts = [0] * N_CAT        # 每天 0 点批量刷新
    near_counts = [0] * N_CAT           # 每小时刷新
    realtime_counts = [0] * N_CAT       # 每分钟刷新
    session = deque(maxlen=10)
    offline_last, near_last = -1, -1

    checkpoints = [SWITCH_MIN - 5, SWITCH_MIN, SWITCH_MIN + 5,
                   SWITCH_MIN + 20, SWITCH_MIN + 60, SWITCH_MIN + 180]
    captured: dict[int, tuple] = {}

    for t, cat, watch in stream:
        realtime_counts[cat] += 1
        session.append(cat)

        if t // 60 != near_last:
            near_counts = realtime_counts[:]
            near_last = t // 60
        if t // (24 * 60) != offline_last:
            offline_counts = realtime_counts[:]
            offline_last = t // (24 * 60)

        for cp in checkpoints:
            if cp not in captured and t >= cp:
                captured[cp] = (t, cat, offline_counts[:], near_counts[:],
                                realtime_counts[:], list(session))

    print("一、兴趣切换（第 300 分钟：体育 -> 游戏）后三类特征的反应")
    header = f"{'时刻':>5}  {'真实行为':>6}  {'离线(日级)':>12}  {'准实时(小时)':>12}  {'实时(分钟)':>13}  {'Session Top':>11}"
    print(header)
    print("-" * len(header))
    printed_lines: list[str] = []
    # 事件是分钟级的，两个检查点可能落在同一条事件上；此时只打印一次，
    # 但保留后一个检查点的"刚切换"标记。
    last_t = None
    for cp in checkpoints:
        if cp not in captured:
            continue
        t, cat, off, near, rt, sess = captured[cp]
        if t == last_t:
            if cp == SWITCH_MIN + 5 and printed_lines:
                printed_lines[-1] += "  <-- 刚切换（与上一检查点同一事件）"
            continue
        sess_top = top([sess.count(i) for i in range(N_CAT)])
        flag = "  <-- 刚切换" if cp == SWITCH_MIN + 5 else ""
        printed_lines.append(
            f"{t:>5}  {CAT_NAME[cat]:>6}  {top(off):>12}  {top(near):>12}"
            f"  {top(rt):>13}  {sess_top:>11}{flag}")
        last_t = t
    for line in printed_lines:
        print(line)
    print("\n  离线特征在切换后数小时仍停留在旧兴趣；准实时约 1 小时跟上；")
    print("  实时与 Session 立刻反映变化。三类特征叠加才是正文的")
    print("  Long-term + Medium-term + Short-term + Real-time（§15.4）。")

    # ---- 多时间尺度汇总 ----
    t0 = SWITCH_MIN + 120
    print(f"\n二、多时间尺度特征汇总（t={t0} 分钟处回看）")
    for name, w in (("30 天（离线）", 30 * 24 * 60), ("1 小时（准实时）", 60), ("10 分钟（实时）", 10)):
        counts = [0] * N_CAT
        for t, cat, _ in stream:
            if t0 - w <= t <= t0:
                counts[cat] += 1
        print(f"  {name:14s} 类目计数={counts}  主类目={top(counts)}")
    print("  同一个用户在不同尺度上呈现不同主类目，这正是多尺度特征的价值。")

    # ---- Session 特征 ----
    last10 = [e for e in stream if t0 - 240 <= e[0] <= t0][-10:]
    cnt = [sum(1 for _, c, _ in last10 if c == i) for i in range(N_CAT)]
    print(f"\n三、Session 特征（t={t0} 之前最近 10 条）")
    print("  最近 10 条：", [CAT_NAME[c] for _, c, _ in last10])
    print(f"  类目计数={cnt}，熵={entropy(cnt):.3f}（越低越专注）")
    print(f"  平均观看时长={sum(w for _, _, w in last10) / max(1, len(last10)):.1f}s，"
          f"最近一次距当前={t0 - last10[-1][0] if last10 else 0} 分钟")
    print("  这类特征回答的是'用户现在在看什么'，而不是'用户一直喜欢什么'。")

    # ---- 更新延迟 ----
    print("\n四、更新延迟对 60 分钟窗口特征的影响（t=540）")
    t_now = 540
    actual = sum(1 for t, _, _ in stream if t_now - 60 <= t <= t_now)
    print(f"{'滞后':>6}  {'窗口内实际行为':>12}  {'特征可见行为':>12}  {'覆盖率':>8}")
    print("-" * 48)
    for lag in (0, 5, 15, 30, 60):
        visible = sum(1 for t, _, _ in stream if t_now - 60 <= t <= t_now - lag)
        print(f"{lag:>4}m  {actual:>12}  {visible:>12}  {visible / actual if actual else 0:>8.0%}")
    print("\n  滞后越长，窗口尾部被截掉的行为越多，特征越'旧'。")
    print("  实时特征的价值不是'更新快'本身，而是在可接受延迟内保持分布一致（§15.4）。")


if __name__ == "__main__":
    main()
