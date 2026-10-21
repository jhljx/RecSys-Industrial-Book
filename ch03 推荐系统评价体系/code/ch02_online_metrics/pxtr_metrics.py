"""对应正文第 3 章 §3.2「在线评价指标」中的 pxtr 指标体系（表 3.1）。

本实现做了什么
---------------
1. 用 `[MOCK]` 的曝光日志 + 行为日志（通过 req_id 关联，呼应 §3.3.3）合成一份
   线上流量，并**严格按正文表 3.1 给出的分母**计算全部 9 个 pxtr 指标；
2. 区分正文强调的两个概念：
   - pxtr 是**先验的模型预测结果**（模型为每次曝光输出的概率）；
   - CTR / CVR / Watch Time 等在线指标是**后验的业务观测结果**（真实流量的统计）。
   脚本把两者并排打印，并用 PCOC = 预测均值 / 实测值 衡量二者的匹配程度；
3. 复刻正文的多任务学习（MTL）口径：一个模型同时输出多个 pxtr 头。

关键公式（正文表 3.1，全部 x100%）
-----------------------------------
    CTR = 点击次数 / 曝光次数
    LTR = 点赞次数 / 曝光次数
    CMR = 评论次数 / 曝光次数
    WTR = 关注次数 / 曝光次数
    EVR = 有效播放次数 / 总播放次数        （播放时长达到阈值为有效）
    LVR = 长时播放次数 / 总播放次数        （播放完成为长播标准）
    VTR = 总播放时长 / (内容总时长 x 播放次数)
    CVR = 转化次数 / 点击次数              （注意分母是点击，不是曝光）
    GTR = 打赏次数 / 曝光次数或播放次数
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# [MOCK] 日志结构：曝光日志 + 行为日志，用 req_id 关联（§3.3.3 的日志体系）
# --------------------------------------------------------------------------
@dataclass
class ImpressionLog:
    """[MOCK] 曝光日志。正文 §3.3.3：记录用户ID、请求ID、推荐结果、排序位置、实验组等。"""

    req_id: str
    user_id: int
    item_id: int
    position: int
    duration_s: float                      # 内容总时长（秒），VTR 的分母要用
    exp_group: str
    pxtr: dict[str, float] = field(default_factory=dict)   # 模型先验预测


@dataclass
class ActionLog:
    """[MOCK] 行为日志。正文 §3.3.3：记录点击、播放、点赞、评论、分享、转化等反馈。"""

    req_id: str
    user_id: int
    clicked: bool = False
    played: bool = False
    watch_time_s: float = 0.0
    liked: bool = False
    commented: bool = False
    followed: bool = False
    converted: bool = False
    gratuity: bool = False


# --------------------------------------------------------------------------
# [MOCK] 多任务模型：一次前向输出多个 pxtr 头
# --------------------------------------------------------------------------
def mock_mtl_predict(user_id: int, item_id: int, duration_s: float) -> dict[str, float]:
    """[MOCK] 模拟正文 §3.2 的多任务学习模型：同一模型输出多个 pxtr。

    这里用确定性的伪随机（由 user_id/item_id 派生）代替真实网络前向，
    保证同一 (user, item) 的预测稳定可复现。
    """
    rng = random.Random(user_id * 1_000_003 + item_id)
    aff = rng.random()                      # 用户对该内容的"亲和度"隐变量
    short = 1.0 if duration_s <= 30 else 0.6   # 短内容更容易被看完
    return {
        "pCTR": min(0.95, 0.02 + 0.22 * aff),
        "pLTR": min(0.95, 0.004 + 0.045 * aff),
        "pCMR": min(0.95, 0.001 + 0.012 * aff),
        "pWTR": min(0.95, 0.0005 + 0.006 * aff),
        "pEVR": min(0.99, 0.35 + 0.45 * aff),
        "pLVR": min(0.99, (0.10 + 0.45 * aff) * short),
        "pVTR": min(0.99, (0.25 + 0.50 * aff) * short),
        "pCVR": min(0.95, 0.01 + 0.08 * aff),
        "pGTR": min(0.95, 0.0002 + 0.0035 * aff),
    }


def simulate_logs(n_users: int, items_per_user: int, exp_group: str = "base",
                  ctr_lift: float = 1.0, watch_lift: float = 1.0):
    """[MOCK] 生成一天的曝光日志与行为日志。

    ctr_lift / watch_lift 用于模拟实验组策略对点击率和播放时长的乘性影响。
    真实行为按「漏斗」顺序逐级采样：曝光 -> 点击 -> 播放 -> 互动 -> 转化 -> 打赏。
    """
    imps: list[ImpressionLog] = []
    acts: dict[str, ActionLog] = {}
    for u in range(n_users):
        for pos in range(items_per_user):
            item = random.randint(0, 50000)
            duration = random.choice([8.0, 15.0, 30.0, 60.0, 120.0, 300.0])
            req_id = f"req-{exp_group}-{u}-{pos}"
            pxtr = mock_mtl_predict(u, item, duration)
            imps.append(ImpressionLog(req_id, u, item, pos, duration, exp_group, pxtr))

            a = ActionLog(req_id, u)
            # 真实 CTR 与 pCTR 相关但不等于它（并叠加位置衰减与实验策略 lift）
            pos_decay = 1.0 / (1.0 + 0.06 * pos)
            p_click = min(0.95, pxtr["pCTR"] * 0.85 * pos_decay * ctr_lift)
            if random.random() < p_click:
                a.clicked = True
                a.played = True                      # 点击即进入播放
                # 播放完成度 ~ Beta 形状，由 pVTR 控制中心位置
                center = min(0.95, pxtr["pVTR"] * 0.9 * watch_lift)
                frac = max(0.0, min(1.0, random.gauss(center, 0.22)))
                a.watch_time_s = frac * duration
                a.liked = random.random() < pxtr["pLTR"] / max(p_click, 1e-9) * 0.9 * 0.25
                a.commented = random.random() < 0.05 * (1 + frac)
                a.followed = random.random() < 0.02 * frac
                a.converted = random.random() < pxtr["pCVR"] * 0.8
                a.gratuity = random.random() < 0.004 * frac
            acts[req_id] = a
    return imps, acts


# --------------------------------------------------------------------------
# 按正文表 3.1 计算后验 pxtr
# --------------------------------------------------------------------------
def compute_posterior_pxtr(imps: list[ImpressionLog], acts: dict[str, ActionLog],
                           effective_view_threshold_s: float = 5.0,
                           long_view_ratio: float = 0.95) -> dict[str, float]:
    """按表 3.1 的分母口径计算后验在线指标（返回百分数）。

    effective_view_threshold_s: 有效播放阈值（正文：播放时长达到阈值为有效标准）
    long_view_ratio:            长播标准（正文：以播放完成为长播标准，这里取完成度>=95%）
    """
    impressions = len(imps)
    clicks = plays = likes = comments = follows = conv = grat = 0
    eff_plays = long_plays = 0
    total_watch = 0.0
    duration_x_plays = 0.0

    for im in imps:
        a = acts[im.req_id]
        clicks += a.clicked
        likes += a.liked
        comments += a.commented
        follows += a.followed
        conv += a.converted
        grat += a.gratuity
        if a.played:
            plays += 1
            total_watch += a.watch_time_s
            duration_x_plays += im.duration_s
            if a.watch_time_s >= effective_view_threshold_s:
                eff_plays += 1
            if a.watch_time_s >= long_view_ratio * im.duration_s:
                long_plays += 1

    def pct(num: float, den: float) -> float:
        return 100.0 * num / den if den else 0.0

    return {
        "CTR": pct(clicks, impressions),
        "LTR": pct(likes, impressions),
        "CMR": pct(comments, impressions),
        "WTR": pct(follows, impressions),
        "EVR": pct(eff_plays, plays),
        "LVR": pct(long_plays, plays),
        "VTR": pct(total_watch, duration_x_plays),
        "CVR": pct(conv, clicks),
        "GTR": pct(grat, impressions),
        # 原始计数，便于核对分子分母
        "_impressions": impressions,
        "_clicks": clicks,
        "_plays": plays,
        "_eff_plays": eff_plays,
        "_long_plays": long_plays,
        "_total_watch_s": total_watch,
        "_duration_x_plays_s": duration_x_plays,
        "_conv": conv,
    }


def mean_prior_pxtr(imps: list[ImpressionLog], key: str) -> float:
    """模型先验预测的均值（百分数）。"""
    return 100.0 * sum(im.pxtr[key] for im in imps) / len(imps)


DENOMINATOR = {
    "CTR": "曝光次数", "LTR": "曝光次数", "CMR": "曝光次数", "WTR": "曝光次数",
    "EVR": "总播放次数", "LVR": "总播放次数", "VTR": "内容总时长x播放次数",
    "CVR": "点击次数", "GTR": "曝光次数",
}
NUMERATOR = {
    "CTR": "点击次数", "LTR": "点赞次数", "CMR": "评论次数", "WTR": "关注次数",
    "EVR": "有效播放次数", "LVR": "长时播放次数", "VTR": "总播放时长",
    "CVR": "转化次数", "GTR": "打赏次数",
}


def main() -> None:
    random.seed(42)

    print("=" * 84)
    print("§3.2 在线评价指标：pxtr 指标体系（正文表 3.1）")
    print("=" * 84)

    imps, acts = simulate_logs(n_users=3000, items_per_user=12)
    m = compute_posterior_pxtr(imps, acts)

    print(f"\n[MOCK 日志规模] 曝光日志 {m['_impressions']} 条，"
          f"行为日志 {len(acts)} 条（req_id 一一关联）")
    print(f"  点击 {m['_clicks']} 次，播放 {m['_plays']} 次，"
          f"有效播放 {m['_eff_plays']} 次（阈值 5s），"
          f"长播 {m['_long_plays']} 次（完成度>=95%）")
    print(f"  总播放时长 {m['_total_watch_s']/3600:.2f} 小时，"
          f"内容总时长x播放次数 = {m['_duration_x_plays_s']/3600:.2f} 小时，"
          f"转化 {m['_conv']} 次")

    print("\n[全部 9 个 pxtr 指标] 按表 3.1 的分子/分母口径计算")
    print(f"  {'指标':>5} | {'分子':>12} | {'分母':>18} | {'后验实测':>9} | "
          f"{'模型先验均值':>12} | {'PCOC':>6}")
    print("  " + "-" * 82)
    for k in ("CTR", "LTR", "CMR", "WTR", "EVR", "LVR", "VTR", "CVR", "GTR"):
        prior = mean_prior_pxtr(imps, "p" + k)
        post = m[k]
        ratio = prior / post if post > 0 else float("inf")
        print(f"  {k:>5} | {NUMERATOR[k]:>12} | {DENOMINATOR[k]:>18} | "
              f"{post:>8.3f}% | {prior:>11.3f}% | {ratio:>6.3f}")

    print("\n  >>> 正文强调的区分：pxtr 是模型输出的**先验预测概率**，")
    print("      而 CTR/CVR/Watch Time 等在线指标是真实流量产生的**后验统计结果**。")
    print("  >>> 右侧 PCOC 列（先验均值 / 后验实测）反映两者的匹配程度；")
    print("      EVR/LVR/VTR/CVR 的分母不是曝光，所以『先验均值 vs 后验实测』")
    print("      的口径差异天然更大——这也是工业界必须按表 3.1 对齐分母的原因。")
    print("      例如 CVR 的分母是点击次数而非曝光次数，若错按曝光计算会低估 "
          f"{m['CVR'] / (100.0*m['_conv']/m['_impressions']):.1f} 倍。")

    # ---------- 分母口径错误的后果 ----------
    print("\n[口径陷阱] CVR 分母用错会怎样：")
    cvr_right = 100.0 * m["_conv"] / m["_clicks"]
    cvr_wrong = 100.0 * m["_conv"] / m["_impressions"]
    print(f"  正确（分母=点击次数 {m['_clicks']}）: CVR = {cvr_right:.3f}%")
    print(f"  错误（分母=曝光次数 {m['_impressions']}）: CVR = {cvr_wrong:.3f}%")
    print(f"  两者相差 {cvr_right/cvr_wrong:.2f} 倍，恰好等于 1/CTR = "
          f"{100.0/m['CTR']:.2f}")

    print("\n[有效播放阈值的敏感性] EVR 对阈值的选择高度敏感：")
    print(f"  {'阈值(s)':>8} | {'有效播放次数':>12} | {'EVR':>8}")
    print("  " + "-" * 36)
    for th in (1.0, 3.0, 5.0, 10.0, 30.0):
        mm = compute_posterior_pxtr(imps, acts, effective_view_threshold_s=th)
        print(f"  {th:>8.1f} | {mm['_eff_plays']:>12d} | {mm['EVR']:>7.3f}%")
    print("  >>> 正文：EVR 通常以播放时长达到阈值为有效标准。阈值是产品定义，")
    print("      跨团队对比指标前必须先对齐阈值，否则数字完全不可比。")

    print("\n[长播标准的敏感性] LVR 以播放完成为标准，完成度门槛不同结果不同：")
    print(f"  {'完成度门槛':>10} | {'长播次数':>9} | {'LVR':>8}")
    print("  " + "-" * 34)
    for r in (0.60, 0.80, 0.90, 0.95, 1.00):
        mm = compute_posterior_pxtr(imps, acts, long_view_ratio=r)
        print(f"  {r:>10.2f} | {mm['_long_plays']:>9d} | {mm['LVR']:>7.3f}%")

    # ---------- 漏斗结构 ----------
    print("\n" + "-" * 84)
    print("[漏斗结构] 正文式：Impression -> Click -> Watch -> Interact -> Conversion -> Pay")
    print("-" * 84)
    interact = sum(1 for a in acts.values() if a.liked or a.commented or a.followed)
    stages = [
        ("Impression 曝光", m["_impressions"]),
        ("Click 点击", m["_clicks"]),
        ("Watch 有效播放", m["_eff_plays"]),
        ("Interact 互动", interact),
        ("Conversion 转化", m["_conv"]),
        ("Pay 打赏", sum(1 for a in acts.values() if a.gratuity)),
    ]
    print(f"  {'阶段':>16} | {'人次':>8} | {'占曝光':>8} | {'上一级转化率':>12}")
    print("  " + "-" * 56)
    for i, (name, cnt) in enumerate(stages):
        prev = stages[i - 1][1] if i > 0 else cnt
        step = 100.0 * cnt / prev if prev else 0.0
        print(f"  {name:>16} | {cnt:>8d} | {100.0*cnt/stages[0][1]:>7.3f}% | "
              f"{step:>11.3f}%")
    print("  >>> 正文口径：CTR 反映用户是否愿意进入内容；EVR/LVR/Watch Time 反映")
    print("      是否真正消费内容；LTR/CMR/WTR 反映参与程度；CVR 反映是否完成转化；")
    print("      GTR 进一步反映付费意愿与商业价值。各级 pxtr 对应漏斗的不同环节，")
    print("      这正是现代推荐模型采用多任务学习（MTL）同时预测多个目标的原因。")


if __name__ == "__main__":
    main()
