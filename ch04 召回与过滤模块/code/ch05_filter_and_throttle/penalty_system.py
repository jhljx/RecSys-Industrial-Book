"""
对应正文第 4 章 §4.5.3「负向判罚体系(Penalty System)」。

正文口径(逐条实现):
  1) 过滤模块本身不判断内容是否违规,只承担「策略执行器」的角色;违规识别由独立的
     内容安全系统、社区治理系统、反作弊系统完成。
  2) 这些系统产生的风险标签统一汇聚到负向判罚体系,正文列举的信号来源为:
     内容安全标签(涉黄/涉暴/涉政等)、用户举报信息、人工审核结论、账号风险等级、
     行为异常检测结果、AI 生成内容风险标签。
  3) 按风险等级采用不同处罚策略:
     高风险   -> 硬过滤 Hard Filtering:直接阻断进入任何推荐链路,曝光量限制为零;
     中风险   -> 软限流 Soft Throttling:保留部分流量 / 降低排序分数 / 限制分发范围;
     群体风险 -> 定向屏蔽 Targeted Blocking:对未成年人屏蔽成人内容、按地域限制等;
     低风险   -> 风险打标 Risk Tagging:不过滤,把风险标签透传给排序模型,
                 由排序模型综合内容质量、用户兴趣与风险等级共同决定曝光概率。

本脚本实现一个完整可跑的判罚体系:
  [上游风控系统(本脚本为 MOCK)] --风险事件--> [信号聚合 + 时间衰减] --> [风险分 & 等级]
      --> [判罚决策] --> {硬过滤 / 软限流 / 定向屏蔽 / 风险打标} --> [排序分修正]

关键公式:
  (a) 负向信号的时间衰减(正文强调风险信号并非永久有效,申诉/整改后应逐步恢复):
          s_c = Σ_e w(type_e) · conf_e · 0.5^((T - t_e) / H)
      其中 H 为半衰期(天),0.5^(Δt/H) 即指数半衰期衰减。
  (b) 风险分归一化到 [0, 1]:  risk_c = 1 - exp(-s_c / λ)
      s_c 越大越接近 1,且单条极端信号不会让分数溢出(饱和特性)。
  (c) 软限流的流量配额与排序分折扣:
          keep_ratio = (1 - risk)^γ            (γ=2,风险越高保留流量越少)
          score' = score · (1 - risk)^γ        (正文「降低排序分数」)
  (d) 风险打标透传给排序模型时,排序模型做的是一次重排权衡:
          final = score · (1 - β · risk)       (β 为排序侧对风险的敏感度)

[MOCK] 说明:上游的内容安全模型、举报流、人审队列、账号风险分、行为异常检测、
AIGC 检测器全部由本脚本用合成随机事件模拟,不接任何真实服务。
"""

import math
import random
from collections import defaultdict

HALF_LIFE_DAYS = 7.0        # 负向信号半衰期
LAMBDA = 2.0                # 风险分归一化尺度
GAMMA = 2.0                 # 软限流的流量折扣指数
BETA = 0.8                  # 排序侧对 risk tag 的敏感度
N_ITEMS = 400
N_USERS = 6
NOW = 30.0                  # 当前时刻(第 30 天)

# 正文列举的 6 类风险信号来源 -> (权重, 是否属于"硬风险"类型)
SIGNAL_SPEC = {
    "内容安全标签": (3.0, True),    # 涉黄/涉暴/涉政等
    "用户举报": (0.8, False),
    "人工审核结论": (4.0, True),
    "账号风险等级": (1.2, False),
    "行为异常检测": (1.0, False),
    "AIGC风险标签": (1.5, False),
}

# 定向屏蔽维度:(受众属性 -> 需要屏蔽的内容标记)
TARGETED_RULES = [
    ("age_minor", "adult_content", "未成年人屏蔽成人内容"),
    ("region_restricted", "region_limited", "特定地区限制展示"),
]


class RiskSignal:
    """[MOCK] 一条上游风控事件。type 为正文列举的信号来源之一。"""

    __slots__ = ("item", "type", "conf", "day", "note")

    def __init__(self, item, type_, conf, day, note=""):
        self.item = item
        self.type = type_
        self.conf = conf
        self.day = day
        self.note = note


class PenaltySystem:
    """负向判罚体系:统一接收风险信号 -> 聚合衰减 -> 定级 -> 输出处罚动作。"""

    LEVELS = ["无风险", "低风险", "中风险", "高风险"]

    def __init__(self, half_life=HALF_LIFE_DAYS, lam=LAMBDA):
        self.half_life = half_life
        self.lam = lam
        self.signals = defaultdict(list)

    # ---------------- 信号接入
    def ingest(self, sig: RiskSignal):
        self.signals[sig.item].append(sig)

    # ---------------- 聚合:公式 (a)
    def raw_score(self, item, now=NOW):
        s = 0.0
        for sig in self.signals.get(item, ()):
            w = SIGNAL_SPEC[sig.type][0]
            decay = 0.5 ** ((now - sig.day) / self.half_life)
            s += w * sig.conf * decay
        return s

    # ---------------- 归一化:公式 (b)
    def risk(self, item, now=NOW):
        return 1.0 - math.exp(-self.raw_score(item, now) / self.lam)

    def has_hard_signal(self, item, now=NOW):
        """人审确认 / 高置信度内容安全标签 -> 正文的高风险硬过滤入口。"""
        for sig in self.signals.get(item, ()):
            w, hard = SIGNAL_SPEC[sig.type]
            if hard and sig.conf >= 0.85 and (now - sig.day) <= 60:
                return True
        return False

    def level(self, item, now=NOW):
        r = self.risk(item, now)
        if self.has_hard_signal(item, now) or r >= 0.85:
            return "高风险"
        if r >= 0.45:
            return "中风险"
        if r > 0.05:
            return "低风险"
        return "无风险"

    # ---------------- 判罚决策
    def decide(self, item, item_meta, user, now=NOW):
        """返回 (动作, keep_ratio, score_mult, 说明)。

        keep_ratio: 允许进入链路的流量比例(硬过滤为 0,风险打标为 1);
        score_mult: 对排序分的乘性折扣(正文「降低排序分数」/「风险标签透传」)。
        """
        lv = self.level(item, now)
        r = self.risk(item, now)

        # 定向屏蔽优先于等级判定:仅对特定人群有风险
        for attr, mark, desc in TARGETED_RULES:
            if item_meta.get(mark) and user.get(attr):
                return "定向屏蔽", 0.0, 0.0, desc

        if lv == "高风险":
            return "硬过滤", 0.0, 0.0, "阻断进入任何推荐链路,曝光量为零"
        if lv == "中风险":
            keep = (1.0 - r) ** GAMMA
            return "软限流", keep, keep, f"保留流量 {keep:.1%} 并同比折扣排序分"
        if lv == "低风险":
            return "风险打标", 1.0, 1.0 - BETA * r, "标签透传排序模型,由排序决定曝光"
        return "放行", 1.0, 1.0, "无风险信号"


# ============================================================ 合成数据
def gen_items(rng):
    """[MOCK] 物品元信息:基础排序分 + 定向屏蔽相关标记。"""
    items = {}
    for i in range(N_ITEMS):
        items[i] = {
            "id": i,
            "score": round(rng.betavariate(2.0, 5.0), 4),   # 排序模型给的原始分
            "adult_content": rng.random() < 0.06,
            "region_limited": rng.random() < 0.05,
        }
    return items


def gen_signals(rng):
    """[MOCK] 模拟上游 6 个风控系统在过去 30 天产出的风险事件流。"""
    sigs = []
    rates = {              # 各信号源的事件密度(每物品期望条数)
        "内容安全标签": 0.08, "用户举报": 0.45, "人工审核结论": 0.04,
        "账号风险等级": 0.20, "行为异常检测": 0.18, "AIGC风险标签": 0.15,
    }
    for i in range(N_ITEMS):
        for t, rate in rates.items():
            n = 0
            # 泊松近似:逐次伯努利
            for _ in range(4):
                if rng.random() < rate / 4:
                    n += 1
            for _ in range(n):
                conf = rng.betavariate(2.5, 2.0) if t != "用户举报" \
                    else rng.betavariate(1.5, 3.0)
                day = round(rng.uniform(0.0, NOW), 1)
                sigs.append(RiskSignal(i, t, round(conf, 3), day))
    return sigs


def gen_users(rng):
    """[MOCK] 受众属性,用于定向屏蔽。"""
    users = []
    for u in range(N_USERS):
        users.append({
            "uid": u,
            "age_minor": u % 3 == 0,
            "region_restricted": u % 4 == 1,
        })
    return users


def main():
    random.seed(42)
    rng = random.Random(42)

    print("=" * 92)
    print("§4.5.3 负向判罚体系:风险信号汇聚 -> 衰减聚合 -> 风险定级 -> 四类处罚动作")
    print("=" * 92)

    items = gen_items(rng)
    users = gen_users(rng)
    sys_ = PenaltySystem()
    sigs = gen_signals(rng)
    for s in sigs:
        sys_.ingest(s)

    # ---------------- 1. 信号源统计
    cnt = defaultdict(int)
    conf_sum = defaultdict(float)
    for s in sigs:
        cnt[s.type] += 1
        conf_sum[s.type] += s.conf
    print(f"\n[信号接入] 过去 {NOW:.0f} 天共 {len(sigs)} 条风险事件,"
          f"覆盖 {len(sys_.signals)}/{N_ITEMS} 个物品"
          f"(上游风控系统均为 [MOCK])")
    print(f"{'信号来源(正文列举)':<18}{'权重 w':>8}{'硬风险':>8}"
          f"{'事件数':>8}{'平均置信度':>12}")
    print("-" * 56)
    for t, (w, hard) in SIGNAL_SPEC.items():
        print(f"{t:<18}{w:>8.1f}{('是' if hard else '否'):>8}"
              f"{cnt[t]:>8d}{conf_sum[t] / max(cnt[t], 1):>12.3f}")

    # ---------------- 2. 时间衰减:同一条信号随时间失效
    print("\n[负向信号时间衰减] s = Σ w·conf·0.5^(Δt/H),H = "
          f"{HALF_LIFE_DAYS:.0f} 天;risk = 1 - exp(-s/λ), λ = {LAMBDA:.1f}")
    print(f"{'信号年龄Δt/天':>14}{'衰减系数':>12}{'单条举报risk':>14}"
          f"{'单条人审risk':>14}{'定级(人审)':>14}")
    print("-" * 68)
    probe = PenaltySystem()
    for dt in [0, 1, 3, 7, 14, 21, 30]:
        probe.signals.clear()
        probe.ingest(RiskSignal(0, "用户举报", 0.9, NOW - dt))
        r_rep = probe.risk(0)
        probe.signals.clear()
        probe.ingest(RiskSignal(0, "人工审核结论", 0.95, NOW - dt))
        r_man = probe.risk(0)
        lv_man = probe.level(0)
        print(f"{dt:>14d}{0.5 ** (dt / HALF_LIFE_DAYS):>12.4f}"
              f"{r_rep:>14.4f}{r_man:>14.4f}{lv_man:>14}")
    print("  注:人审/高置信内容安全标签在 60 天内直接触发高风险硬过滤(正文:直接阻断),")
    print("     而举报这类弱信号只贡献连续风险分,靠累积与衰减自然进出中/低风险区间。")

    # ---------------- 3. 风险等级分布与处罚动作分布
    user0 = {"uid": -1, "age_minor": False, "region_restricted": False}
    lv_cnt = defaultdict(int)
    act_cnt = defaultdict(int)
    keep_sum = defaultdict(float)
    mult_sum = defaultdict(float)
    for i, meta in items.items():
        lv_cnt[sys_.level(i)] += 1
        act, keep, mult, _ = sys_.decide(i, meta, user0)
        act_cnt[act] += 1
        keep_sum[act] += keep
        mult_sum[act] += mult

    print("\n[风险定级分布] 普通受众(无定向屏蔽属性)下的判罚动作")
    print(f"{'风险等级':<10}{'物品数':>8}{'占比':>9}   |  "
          f"{'处罚动作':<10}{'物品数':>8}{'占比':>9}{'平均保留流量':>14}{'平均排序分乘子':>16}")
    print("-" * 104)
    pairs = [("高风险", "硬过滤"), ("中风险", "软限流"),
             ("低风险", "风险打标"), ("无风险", "放行")]
    for lv, act in pairs:
        n_lv, n_act = lv_cnt[lv], act_cnt[act]
        print(f"{lv:<10}{n_lv:>8d}{n_lv / N_ITEMS:>9.2%}   |  "
              f"{act:<10}{n_act:>8d}{n_act / N_ITEMS:>9.2%}"
              f"{keep_sum[act] / max(n_act, 1):>14.2%}"
              f"{mult_sum[act] / max(n_act, 1):>16.4f}")

    # ---------------- 4. 定向屏蔽:同一物品对不同受众的不同结论
    print("\n[定向屏蔽] 正文:仅对特定用户群体存在风险的内容做 Targeted Blocking")
    adult = next(i for i, m in items.items()
                 if m["adult_content"] and sys_.level(i) != "高风险")
    region = next(i for i, m in items.items()
                  if m["region_limited"] and not m["adult_content"]
                  and sys_.level(i) != "高风险")
    print(f"{'受众':<22}{'成人内容item' + str(adult):>20}"
          f"{'地域限制item' + str(region):>20}")
    print("-" * 62)
    for u in users:
        tags = []
        if u["age_minor"]:
            tags.append("未成年")
        if u["region_restricted"]:
            tags.append("受限地区")
        label = f"u{u['uid']}({'/'.join(tags) if tags else '普通'})"
        a1 = sys_.decide(adult, items[adult], u)[0]
        a2 = sys_.decide(region, items[region], u)[0]
        print(f"{label:<22}{a1:>20}{a2:>20}")

    # ---------------- 5. 风险打标 vs 硬过滤:对排序结果的影响
    print("\n[风险打标透传排序] final = score · (1 - β·risk), β = "
          f"{BETA:.1f};与「同样内容直接硬过滤」做对比")
    ranked = sorted(items.values(), key=lambda m: -m["score"])[:40]
    rows = []
    for m in ranked:
        r = sys_.risk(m["id"])
        act, keep, mult, _ = sys_.decide(m["id"], m, user0)
        rows.append((m["id"], m["score"], r, act, m["score"] * mult))
    rows.sort(key=lambda x: -x[4])
    print(f"{'item':>8}{'原始排序分':>12}{'风险分':>10}{'判罚动作':>12}"
          f"{'判罚后分数':>12}{'原排名':>8}{'新排名':>8}")
    print("-" * 72)
    orig_rank = {r[0]: k + 1 for k, r in
                 enumerate(sorted(rows, key=lambda x: -x[1]))}
    for k, (iid, s0, r, act, s1) in enumerate(rows[:12]):
        print(f"{iid:>8d}{s0:>12.4f}{r:>10.4f}{act:>12}{s1:>12.4f}"
              f"{orig_rank[iid]:>8d}{k + 1:>8d}")
    blocked = [r for r in rows if r[3] in ("硬过滤", "定向屏蔽")]
    soft = [r for r in rows if r[3] == "软限流"]
    tagged = [r for r in rows if r[3] == "风险打标"]
    print(f"  Top40 中:硬过滤 {len(blocked)} 个(直接出链路),软限流 {len(soft)} 个,"
          f"风险打标 {len(tagged)} 个(仅降分)")
    if tagged:
        avg_drop = sum(1 - x[4] / x[1] for x in tagged) / len(tagged)
        print(f"  风险打标组平均降分幅度 {avg_drop:.2%},"
              f"说明低风险内容仍保留被排序模型选中的机会")

    # ---------------- 6. 整改/申诉后的恢复过程(信号衰减的业务价值)
    print("\n[申诉与恢复] 一条 conf=0.80 的 AIGC 风险标签 + 两条举报后,随时间的自然恢复")
    probe2 = PenaltySystem()
    probe2.ingest(RiskSignal(1, "AIGC风险标签", 0.80, 0.0))
    probe2.ingest(RiskSignal(1, "用户举报", 0.70, 0.0))
    probe2.ingest(RiskSignal(1, "用户举报", 0.60, 1.0))
    print(f"{'距打标天数':>12}{'聚合分s':>12}{'风险分':>10}{'等级':>10}"
          f"{'保留流量':>12}{'排序分乘子':>14}")
    print("-" * 70)
    meta1 = {"adult_content": False, "region_limited": False}
    for d in [0, 2, 5, 7, 10, 14, 21, 28]:
        now = d
        s = probe2.raw_score(1, now)
        r = probe2.risk(1, now)
        lv = probe2.level(1, now)
        # 临时按 now 评估判罚动作
        act, keep, mult, _ = probe2.decide(1, meta1, user0, now=now)
        print(f"{d:>12d}{s:>12.4f}{r:>10.4f}{lv:>10}{keep:>12.2%}{mult:>14.4f}"
              f"   {act}")

    print("\n结论(正文 §4.5.3 的工程含义):")
    print("  1) 过滤模块只是策略执行器:风险标签由内容安全/社区治理/反作弊系统产出,")
    print("     判罚体系负责统一汇聚并翻译成可执行动作,这样策略才能集中演进;")
    print("  2) 四类动作对应四个风险档位 -- 硬过滤(曝光归零)、软限流(保留部分流量")
    print("     并降排序分)、定向屏蔽(按受众属性)、风险打标(透传排序模型决策);")
    print("  3) 负向信号必须带时间衰减,否则一次误判会永久压死一个创作者;")
    print("     半衰期 H 是治理强度的关键旋钮,实测 7 天半衰期下弱信号约 3 周回到无风险;")
    print("  4) 风险打标把「安全」与「体验」的权衡交给排序模型统一做,")
    print("     避免过滤层用硬阈值做出不可解释的误伤。")


if __name__ == "__main__":
    main()
