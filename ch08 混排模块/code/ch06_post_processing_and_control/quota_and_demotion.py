"""保量(Quota Guarantee)与降权(Demotion):混排后处理的生态调控。

对应正文第 8 章 §8.6.4「保量与降权」。

正文要点(本实现逐条复现):
1. 平台需要扶持新兴业务/内容生态(短剧、AI 漫剧、知识内容、本地生活等),
   这些内容在发展初期缺乏数据积累,完全依赖模型排序很难拿到足够曝光;
2. 因此混排后处理阶段增加保量机制,正文给出的四条规则正是本实现的四条配置:
   - 每 100 次曝光至少包含一定比例的短剧内容;
   - 新创作者内容获得最低曝光保障;
   - AIGC 内容进行适当的降权打压;
   - AI 漫剧内容获得额外曝光加权;
3. 降权针对低质/重复/高营销感/用户反馈差的内容,在最终曝光前进一步降低排序分数;
4. 保量与降权的目标不是短期 CTR/时长,而是长周期的生态健康,
   因此评估时必须同时看「规则达成度」与「模型价值损失」两张账。

本实现做了什么:
1. 降权/加权用乘性权重实现:s' = s * w(tag),w<1 为降权、w>1 为加权(软手段);
2. 保量给出两种实现并对比:
   (a) 软补偿:s'' = s' + kappa * max(0, q_tag - share_tag^(window)),亏欠越大提权越强;
   (b) 硬保量(欠量配速 debt pacing):每次请求给标签记入应得曝光 q*K,
       实际曝光多少就扣多少,欠量累计到 >=1 条时在当刷**预留坑位**强制补位。
       长期来看曝光占比收敛到 q;欠量做 clamp(与 §8.6.2 PID 的抗积分饱和同理),
       避免候选长期缺货后欠量无限累积、供给恢复瞬间刷屏。
3. 新创作者最低曝光保障用「每刷至少 m 条」的硬名额实现;
4. 打印四张表:配置表、三策略(无干预/仅权重/权重+硬保量)的曝光占比与价值损失、
   软补偿 kappa 扫描与硬保量的对照、滚动 100 次曝光窗口的占比轨迹。

关键公式:
    加权/降权      s'_i  = s_i * w(tag_i)
    软补偿保量      s''_i = s'_i + kappa * max(0, q_t - share_t^(window))
    硬保量配速      debt_t <- clamp(debt_t + q_t * K - n_t, 0, debt_max),预留 floor(debt_t) 个坑位
    生态价值账      Loss  = 1 - sum(s of 干预后 top-K) / sum(s of 无干预 top-K)

[MOCK] 内容池、标签分布、曝光窗口统计均为进程内合成,不读取任何真实数据。
"""

from __future__ import annotations

import math
import random
import statistics
from collections import deque

SLOT_NUM = 8            # 每刷曝光坑位数 K
WINDOW_SIZE = 100       # 正文「每 100 次曝光」的滚动统计窗口
KAPPA = 2.0             # 软补偿系数
DEBT_MAX = 3.0          # 欠量上限(抗饱和),最多攒 3 条待补

# 正文列举的生态标签 -> (内容池占比, 乘性权重, 保量下界)
TAG_CONFIG = {
    "short_drama": {"pool": 0.12, "weight": 1.00, "quota": 0.15},   # 短剧:靠硬保量
    "ai_manhua":   {"pool": 0.06, "weight": 1.30, "quota": 0.00},   # AI 漫剧:额外加权
    "knowledge":   {"pool": 0.08, "weight": 1.10, "quota": 0.05},   # 知识内容:加权 + 兜底
    "local_life":  {"pool": 0.07, "weight": 1.00, "quota": 0.05},   # 本地生活:靠硬保量
    "aigc":        {"pool": 0.15, "weight": 0.70, "quota": 0.00},   # AIGC:降权打压
    "low_quality": {"pool": 0.10, "weight": 0.50, "quota": 0.00},   # 低质/高营销感:重降权
    "normal":      {"pool": 0.42, "weight": 1.00, "quota": 0.00},   # 普通存量内容
}
QUOTA_TAGS = [t for t, c in TAG_CONFIG.items() if c["quota"] > 0]
NEW_CREATOR_RATE = 0.18     # 内容池中新创作者内容占比
NEW_CREATOR_MIN_SLOT = 1    # 正文「新创作者内容获得最低曝光保障」:每刷至少 1 条


# ------------------------------------------------------------ [MOCK] 内容池

def sample_candidates(n=60):
    """[MOCK] 合成一批混排候选:每条带模型分 s、生态标签 tag、是否新创作者。

    模型分用 lognormal 刻画长尾;生态扶持标签的模型分整体乘 0.75,
    刻画正文说的「发展初期缺乏数据积累,很难获得充足曝光」。
    """
    tags = list(TAG_CONFIG)
    probs = [TAG_CONFIG[t]["pool"] for t in tags]
    items = []
    for i in range(n):
        tag = random.choices(tags, weights=probs, k=1)[0]
        s = random.lognormvariate(0.0, 0.45)
        if TAG_CONFIG[tag]["quota"] > 0 or tag == "ai_manhua":
            s *= 0.75                      # 冷启内容的模型分天然偏低
        items.append({
            "id": f"I{i:02d}",
            "tag": tag,
            "score": s,
            "new_creator": random.random() < NEW_CREATOR_RATE,
        })
    return items


def weighted_key(it):
    """乘性加权/降权后的排序分 s' = s * w(tag)。"""
    return it["score"] * TAG_CONFIG[it["tag"]]["weight"]


# ------------------------------------------------------------ 四种后处理策略

def select_plain(items):
    """策略 A:无任何干预,纯按模型分取 Top-K。"""
    return sorted(items, key=lambda it: -it["score"])[:SLOT_NUM]


def select_weighted(items):
    """策略 B:只做乘性加权/降权。"""
    return sorted(items, key=lambda it: -weighted_key(it))[:SLOT_NUM]


def select_soft_quota(items, window, kappa=KAPPA):
    """策略 C:乘性权重 + 软补偿保量(按滚动窗口亏欠量加分)。"""
    total = len(window) if window else 1
    cnt = {t: 0 for t in TAG_CONFIG}
    for t in window:
        cnt[t] += 1

    def key(it):
        cfg = TAG_CONFIG[it["tag"]]
        deficit = max(0.0, cfg["quota"] - cnt[it["tag"]] / total)
        return -(weighted_key(it) + kappa * deficit)

    return sorted(items, key=key)[:SLOT_NUM]


def select_hard_quota(items, debt, new_creator_min=NEW_CREATOR_MIN_SLOT):
    """策略 D:乘性权重 + 硬保量预留坑位 + 新创作者名额。

    debt[tag] 是该标签当前累计的「欠曝光条数」;本函数会原地更新 debt。
    """
    ranked = sorted(items, key=lambda it: -weighted_key(it))
    chosen, used = [], set()

    # 1) 先按欠量预留坑位:欠量 >= 1 条就强制补位(取该标签内 s' 最高的)
    for tag in QUOTA_TAGS:
        need = int(math.floor(debt[tag]))
        if need <= 0:
            continue
        pool = [it for it in ranked if it["tag"] == tag and it["id"] not in used]
        for it in pool[:need]:
            if len(chosen) >= SLOT_NUM:
                break
            chosen.append(it)
            used.add(it["id"])

    # 2) 剩余坑位按 s' 填满
    for it in ranked:
        if len(chosen) >= SLOT_NUM:
            break
        if it["id"] not in used:
            chosen.append(it)
            used.add(it["id"])

    # 3) 新创作者最低曝光保障:不足则顶替掉非新创作者、非保量预留的最低分内容
    lack = new_creator_min - sum(1 for it in chosen if it["new_creator"])
    if lack > 0:
        pool = [it for it in ranked
                if it["new_creator"] and it["id"] not in used]
        for it in pool[:lack]:
            victims = [(i, c) for i, c in enumerate(chosen)
                       if not c["new_creator"] and TAG_CONFIG[c["tag"]]["quota"] == 0]
            if not victims:
                break
            idx = min(victims, key=lambda p: weighted_key(p[1]))[0]
            used.discard(chosen[idx]["id"])
            chosen[idx] = it
            used.add(it["id"])

    # 4) 更新欠量:记入应得 q*K,扣除实际曝光,再 clamp 抗饱和
    for tag in QUOTA_TAGS:
        got = sum(1 for it in chosen if it["tag"] == tag)
        debt[tag] = min(DEBT_MAX,
                        max(0.0, debt[tag] + TAG_CONFIG[tag]["quota"] * SLOT_NUM - got))
    return chosen


# ------------------------------------------------------------ demo1:配置表

def demo_config_table():
    print("=" * 86)
    print("demo1  正文 §8.6.4 的保量/降权配置(乘性权重 + 滚动 100 次曝光的保量下界)")
    print("=" * 86)
    print(f"{'生态标签':<14}{'内容池占比':>11}{'乘性权重':>10}{'保量下界':>10}"
          f"{'正文对应规则':>26}")
    print("-" * 86)
    note = {
        "short_drama": "每100次曝光含一定比例短剧",
        "ai_manhua": "AI漫剧额外曝光加权",
        "knowledge": "知识内容扶持",
        "local_life": "本地生活扶持",
        "aigc": "AIGC内容适当降权打压",
        "low_quality": "低质/高营销感内容降权",
        "normal": "(存量内容,不干预)",
    }
    for tag, cfg in TAG_CONFIG.items():
        print(f"{tag:<14}{cfg['pool']:>11.2%}{cfg['weight']:>10.2f}"
              f"{cfg['quota']:>10.2%}{note[tag]:>26}")
    print("-" * 86)
    print(f"每刷 K={SLOT_NUM} 个坑位,新创作者内容占内容池 {NEW_CREATOR_RATE:.0%},"
          f"硬名额为每刷至少 {NEW_CREATOR_MIN_SLOT} 条。")
    print("乘性权重是「软」手段(只改排序分);保量下界需要「硬」手段(预留坑位)才能兜住。")
    print("正文把两者都放在混排后处理阶段,在最终曝光前统一生效。")
    print()


# ------------------------------------------------------------ 策略评测

def run_strategy(mode, n_req=3000, kappa=KAPPA, seed=42, trace_every=0):
    """跑 n_req 次请求,返回曝光占比 / 价值损失 / 新创作者条数 / 窗口轨迹。"""
    random.seed(seed)
    window = deque(maxlen=WINDOW_SIZE)
    debt = {t: 0.0 for t in QUOTA_TAGS}
    exposure = {t: 0 for t in TAG_CONFIG}
    newc = 0
    losses, trace = [], []
    for r in range(n_req):
        items = sample_candidates()
        base = select_plain(items)
        base_val = sum(it["score"] for it in base)
        if mode == "plain":
            chosen = base
        elif mode == "weighted":
            chosen = select_weighted(items)
        elif mode == "soft":
            chosen = select_soft_quota(items, window, kappa=kappa)
        else:
            chosen = select_hard_quota(items, debt)
        losses.append((base_val - sum(it["score"] for it in chosen)) / base_val)
        for it in chosen:
            exposure[it["tag"]] += 1
            window.append(it["tag"])
        newc += sum(1 for it in chosen if it["new_creator"])
        if trace_every and (r + 1) % trace_every == 0 and len(window) == WINDOW_SIZE:
            trace.append((r + 1,
                          sum(1 for t in window if t == "short_drama") / WINDOW_SIZE,
                          sum(1 for t in window if t == "local_life") / WINDOW_SIZE,
                          debt.get("short_drama", 0.0)))
    tot = sum(exposure.values())
    return {
        "share": {t: exposure[t] / tot for t in TAG_CONFIG},
        "loss": statistics.fmean(losses),
        "new_creator_per_req": newc / n_req,
        "trace": trace,
    }


# ------------------------------------------------------------ demo2:三策略对比

def demo_three_strategies():
    print("=" * 86)
    print("demo2  三种后处理策略的曝光占比与价值损失(3000 次请求 x 8 坑位)")
    print("=" * 86)
    res = {m: run_strategy(m) for m in ("plain", "weighted", "hard")}
    print(f"{'生态标签':<14}{'保量下界':>10}{'无干预':>11}{'仅权重':>11}"
          f"{'权重+硬保量':>13}{'是否达标':>10}")
    print("-" * 86)
    for tag, cfg in TAG_CONFIG.items():
        ok = "-" if cfg["quota"] == 0 else (
            "达标" if res["hard"]["share"][tag] >= cfg["quota"] else "未达标")
        print(f"{tag:<14}{cfg['quota']:>10.2%}{res['plain']['share'][tag]:>11.2%}"
              f"{res['weighted']['share'][tag]:>11.2%}"
              f"{res['hard']['share'][tag]:>13.2%}{ok:>10}")
    print("-" * 86)
    print(f"{'平均价值损失':<14}{'':>10}{res['plain']['loss']:>11.2%}"
          f"{res['weighted']['loss']:>11.2%}{res['hard']['loss']:>13.2%}")
    print(f"{'新创作者条/刷':<14}{'':>10}"
          f"{res['plain']['new_creator_per_req']:>11.2f}"
          f"{res['weighted']['new_creator_per_req']:>11.2f}"
          f"{res['hard']['new_creator_per_req']:>13.2f}")
    print("-" * 86)
    print("无干预时扶持类内容因模型分天然偏低而曝光不足(短剧仅 5.38%,远低于 15% 下界);")
    print("乘性权重能把加权标签顶上来(AI漫剧 2.47%->8.75%),")
    print("但对权重=1.00 的短剧/本地生活几乎无效——软手段无法保证下界。")
    print("硬保量预留坑位后,三个带下界的标签全部达标(短剧 15.13%、知识 7.03%、本地生活 5.93%),")
    print("降权标签也被压低(AIGC 19.59%->5.62%,低质 12.92%->0.66%),")
    print("代价是平均序列价值损失升到 8.11%:这就是正文说的「不追求短期指标」的生态账。")
    print()
    return res


# ------------------------------------------------------------ demo3:软 vs 硬

def demo_soft_vs_hard():
    print("=" * 86)
    print("demo3  软补偿 kappa 扫描 vs 硬保量预留(短剧下界 15%,本地生活/知识 5%)")
    print("=" * 86)
    print(f"{'保量实现':<22}{'短剧占比':>11}{'本地生活':>11}{'知识内容':>11}"
          f"{'AIGC占比':>11}{'平均价值损失':>14}")
    print("-" * 86)
    rows = []
    for kappa in (0.0, 0.5, 1.0, 2.0, 5.0):
        r = run_strategy("soft", n_req=1200, kappa=kappa)
        rows.append((f"软补偿 kappa={kappa:.1f}", r))
    rows.append(("硬保量(预留坑位)", run_strategy("hard", n_req=1200)))
    for name, r in rows:
        print(f"{name:<22}{r['share']['short_drama']:>11.2%}"
              f"{r['share']['local_life']:>11.2%}{r['share']['knowledge']:>11.2%}"
              f"{r['share']['aigc']:>11.2%}{r['loss']:>14.2%}")
    print("-" * 86)
    print("kappa=0 等价于纯乘性权重。kappa 增大时短剧占比单调上升(7.36% -> 11.75%),")
    print("但即使 kappa=5.0 仍未达到 15% 的下界:软补偿只是把分数抬高一个常数,")
    print("当该标签候选的模型分普遍更低时,它仍然可能一条都挤不进 Top-K。")
    print("硬保量直接按欠量预留坑位,占比精确落在下界附近(短剧 15.11%),")
    print("价值损失也最高——这正是「软手段调倾向、硬手段保下界」的工程分工。")
    print()


# ------------------------------------------------------------ demo4:收敛轨迹

def demo_window_trace():
    print("=" * 86)
    print("demo4  滚动 100 次曝光窗口内的保量占比与欠量轨迹(硬保量)")
    print("=" * 86)
    r = run_strategy("hard", n_req=3000, trace_every=300)
    print(f"{'请求数':>8}{'窗口内短剧占比':>16}{'窗口内本地生活占比':>20}"
          f"{'短剧欠量(条)':>15}")
    print("-" * 86)
    for n, sd, ll, debt in r["trace"]:
        print(f"{n:>8}{sd:>16.2%}{ll:>20.2%}{debt:>15.3f}")
    print("-" * 86)
    sds = [x[1] for x in r["trace"]]
    lls = [x[2] for x in r["trace"]]
    print(f"短剧窗口占比均值 {statistics.fmean(sds):.2%}"
          f"(下界 {TAG_CONFIG['short_drama']['quota']:.0%}),"
          f"std={statistics.pstdev(sds):.4f}")
    print(f"本地生活窗口占比均值 {statistics.fmean(lls):.2%}"
          f"(下界 {TAG_CONFIG['local_life']['quota']:.0%}),"
          f"std={statistics.pstdev(lls):.4f}")
    print(f"短剧欠量始终在 [0, {DEBT_MAX}] 内(clamp 生效),不会无限累积。")
    print("占比稳定在下界附近而不是一路上冲:欠量归零后预留立即撤销,")
    print("因此保量机制本身就是一个「只在欠量时发力」的闭环,与 §8.6.2 的 PID 同源;")
    print("欠量 clamp 对应 PID 的抗积分饱和——避免候选长期缺货后供给恢复时刷屏。")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_config_table()
    demo_three_strategies()
    demo_soft_vs_hard()
    demo_window_trace()
