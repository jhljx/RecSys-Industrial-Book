"""
粗排 Quota 调控:固定配额 / 动态配额 / 闭环控制(第 5 章 §5.6.1 粗排Quota调控)

对应正文概念
------------
正文先用表 prerank_no_quota_example 说明:若完全按 TopK(score) 筛选,候选集可能
全部来自同一个池(5 个游戏主播 0.98/0.96/0.95/0.94/0.93),短期指标最优但带来内容同质化、
新主播无曝光、生态失衡等长期问题。于是引入 Quota 机制,表 prerank_quota_config_example
给出的配置是:游戏直播池 40%、娱乐直播池 30%、电商直播池 20%、新主播池 10%,
对应 100 个输出候选时:

    Quota_game = 40, Quota_entertainment = 30, Quota_ecommerce = 20, Quota_new = 10

正文 code_prerank_quota 伪代码的逻辑是:多路召回 -> 粗排模型预测 pctr/pcvr/pltr ->
融合 score = 0.5 pctr + 0.3 pcvr + 0.2 pltr -> 按业务规则切池 -> 池内按 score 排序 ->
各池取 TopQuota_i -> CandidateSet = ∪_i TopK_i。

正文接着指出固定 Quota 灵活性差:对重度游戏用户 P(Game|User) = 0.9 时仍给 40% 不合理,
因此引入动态 Quota:

    Quota_i = f(User, Context, BusinessGoal)

正文举例可调成 Quota_game = 50%、Quota_ecommerce = 10%;更进一步可用强化学习/在线优化
学习策略 π(s) -> Quota,根据实时业务指标动态调整流量结构。

本实现做了什么
--------------
1. 完整复刻 code_prerank_quota:[MOCK] retrieve() 多路召回 + 三个 [MOCK] pxtr 模型
   + 0.5/0.3/0.2 融合 + 切池 + 池内排序 + 固定 Quota 取 TopK,并与无约束 TopK(score)
   对比池分布、分数损失、生态指标。
2. 动态 Quota:Quota_i = f(User, Context, BusinessGoal) 的一个可解释实现 ——
   用户池偏好 P(pool|user)、业务目标权重与生态保底三者做凸组合后按最大余数法取整,
   打印重度游戏用户 / 电商意图用户 / 新用户三种画像下的配额与命中分数。
3. 闭环控制:把 π(s) -> Quota 写成一个带积分项的离散反馈控制律
       e_t = target_share - realized_share_t
       q_{t+1} = clip(q_t + K_p e_t + K_i Σ_{τ≤t} e_τ, q_min, q_max)
   然后归一化成整数配额。[MOCK] 环境给出「下游精排/用户对各池的实际消费倾向」,
   使开环固定配额无法达到曝光目标,闭环控制在若干轮内把误差压到阈值内。
   打印逐轮(轮次 × 池)的配额、实际曝光占比与误差,以及收敛轮次。

关键公式
--------
    score = 0.5 p_ctr + 0.3 p_cvr + 0.2 p_ltr
    CandidateSet = ∪_i TopK_i,  |TopK_i| = Quota_i
    Quota_i = f(User, Context, BusinessGoal)
    π(s) -> Quota:  q_{t+1} = clip(q_t + K_p e_t + K_i Σ e_τ, q_min, q_max)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

OUTPUT_SIZE = 100                       # 正文:粗排最终输出 100 个候选
FIXED_QUOTA = {"game": 40, "entertainment": 30, "ecommerce": 20, "new": 10}
FUSION_W = {"pctr": 0.5, "pcvr": 0.3, "pltr": 0.2}   # 正文 code_prerank_quota
POOLS = ("game", "entertainment", "ecommerce", "new")


@dataclass
class Item:
    """一个候选主播/直播间。"""
    iid: int
    category: str           # game / entertainment / ecommerce
    is_new_anchor: bool
    pctr: float = 0.0
    pcvr: float = 0.0
    pltr: float = 0.0
    score: float = 0.0


@dataclass
class MockModels:
    """[MOCK] 三个粗排 pxtr 模型:按类目/新主播身份给出带噪声的预测值,不做真实推理。"""
    rng: random.Random
    base: dict[str, tuple[float, float, float]] = field(default_factory=lambda: {
        # category -> (pctr 均值, pcvr 均值, pltr 均值)
        "game": (0.085, 0.020, 0.055),
        "entertainment": (0.060, 0.012, 0.070),
        "ecommerce": (0.045, 0.060, 0.020),
    })

    def predict(self, it: Item) -> None:
        mc, mv, ml = self.base[it.category]
        penalty = 0.72 if it.is_new_anchor else 1.0     # 新主播缺少后验,预测分天然偏低
        it.pctr = max(1e-4, self.rng.gauss(mc * penalty, mc * 0.35))
        it.pcvr = max(1e-5, self.rng.gauss(mv * penalty, mv * 0.40))
        it.pltr = max(1e-4, self.rng.gauss(ml * penalty, ml * 0.35))


def retrieve(n: int, rng: random.Random) -> list[Item]:
    """[MOCK] 多路召回:按类目/新主播比例生成候选,不访问任何真实索引。"""
    out = []
    for i in range(n):
        r = rng.random()
        cat = "game" if r < 0.55 else ("entertainment" if r < 0.82 else "ecommerce")
        out.append(Item(iid=i, category=cat, is_new_anchor=rng.random() < 0.18))
    return out


def pool_of(it: Item) -> str:
    """正文切池规则:new_anchor_pool 由 is_new_anchor 定义,与类目池并列。"""
    return "new" if it.is_new_anchor else it.category


def fuse(items: list[Item]) -> None:
    for it in items:
        it.score = (FUSION_W["pctr"] * it.pctr + FUSION_W["pcvr"] * it.pcvr
                    + FUSION_W["pltr"] * it.pltr)


def split_pools(items: list[Item]) -> dict[str, list[Item]]:
    """按业务规则切池,并在池内按 score 降序排序(正文伪代码第二段)。"""
    pools: dict[str, list[Item]] = {p: [] for p in POOLS}
    for it in items:
        pools[pool_of(it)].append(it)
    for p in pools:
        pools[p].sort(key=lambda x: -x.score)
    return pools


def apply_quota(pools: dict[str, list[Item]], quota: dict[str, int]) -> list[Item]:
    """CandidateSet = ∪_i TopK_i。"""
    result: list[Item] = []
    for p in POOLS:
        result.extend(pools[p][: quota.get(p, 0)])
    return result


def unconstrained_topk(items: list[Item], k: int) -> list[Item]:
    return sorted(items, key=lambda x: -x.score)[:k]


def share(sel: list[Item]) -> dict[str, float]:
    n = max(1, len(sel))
    cnt = {p: 0 for p in POOLS}
    for it in sel:
        cnt[pool_of(it)] += 1
    return {p: cnt[p] / n for p in POOLS}


def entropy(sh: dict[str, float]) -> float:
    h = -sum(v * math.log(v) for v in sh.values() if v > 0)
    return h / math.log(len(sh))


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """最大余数法:把连续份额转成整数配额,保证 Σ Quota_i = total。"""
    raw = {p: weights[p] * total for p in weights}
    base = {p: int(raw[p]) for p in raw}
    rest = total - sum(base.values())
    for p, _ in sorted(raw.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:rest]:
        base[p] += 1
    return base


def dynamic_quota(user_pref: dict[str, float], biz_goal: dict[str, float],
                  floor: dict[str, float], total: int,
                  w_user: float = 0.65, w_biz: float = 0.20,
                  w_floor: float = 0.15) -> dict[str, int]:
    """Quota_i = f(User, Context, BusinessGoal):三路份额的凸组合 + 最大余数取整。"""
    mix = {p: w_user * user_pref.get(p, 0.0) + w_biz * biz_goal.get(p, 0.0)
              + w_floor * floor.get(p, 0.0) for p in POOLS}
    tot = sum(mix.values())
    return largest_remainder({p: v / tot for p, v in mix.items()}, total)


class QuotaController:
    """π(s) -> Quota 的 PI 反馈控制律(离散时间,带积分项与饱和限幅)。"""

    def __init__(self, target: dict[str, float], kp: float = 0.45, ki: float = 0.30,
                 q_min: float = 0.03, q_max: float = 0.70):
        self.target = dict(target)
        self.kp, self.ki = kp, ki
        self.q_min, self.q_max = q_min, q_max
        self.q = dict(target)                        # 初始配额 = 目标曝光占比
        self.integral = {p: 0.0 for p in target}

    def act(self) -> dict[str, int]:
        tot = sum(self.q.values())
        return largest_remainder({p: v / tot for p, v in self.q.items()}, OUTPUT_SIZE)

    def update(self, realized: dict[str, float]) -> dict[str, float]:
        err = {p: self.target[p] - realized.get(p, 0.0) for p in self.target}
        for p in self.q:
            self.integral[p] += err[p]
            nq = self.q[p] + self.kp * err[p] + self.ki * self.integral[p]
            self.q[p] = min(self.q_max, max(self.q_min, nq))
        return err


def mock_downstream(quota: dict[str, int], pass_rate: dict[str, float],
                    rng: random.Random, requests: int = 200) -> dict[str, float]:
    """[MOCK] 下游环境:一轮内有 `requests` 次请求,各池候选以不同通过率进入最终曝光,
    导致开环配额与目标曝光结构失真(请求数越大,统计噪声越小)。"""
    exposed = {p: 0.0 for p in quota}
    for p, q in quota.items():
        trials = q * requests
        got = sum(1 for _ in range(trials) if rng.random() < pass_rate[p])
        exposed[p] = float(got)
    tot = max(1.0, sum(exposed.values()))
    return {p: v / tot for p, v in exposed.items()}


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)

    # ---- 0. 正文表 prerank_no_quota_example ----
    print("=" * 100)
    print("正文表 prerank_no_quota_example:无 Quota 约束时 Top5 全部来自游戏池")
    print("=" * 100)
    demo = [("A", "游戏", 0.98), ("B", "游戏", 0.96), ("C", "游戏", 0.95),
            ("D", "游戏", 0.94), ("E", "游戏", 0.93)]
    print(f"{'主播ID':>8}{'类别':>8}{'Score':>10}")
    for a, b, c in demo:
        print(f"{a:>8}{b:>8}{c:>10.2f}")
    print("短期业务指标最优,但内容同质化、新主播零曝光 -- 正文据此引入 Quota 机制。")

    # ---- 1. 复刻 code_prerank_quota:固定 Quota vs 无约束 TopK ----
    print("\n" + "=" * 100)
    print("复刻 code_prerank_quota:召回 5000 -> 粗排打分 -> 0.5/0.3/0.2 融合 -> 固定 Quota")
    print("=" * 100)
    cands = retrieve(5000, rng)
    models = MockModels(rng)
    for it in cands:
        models.predict(it)
    fuse(cands)
    pools = split_pools(cands)
    print(f"配额配置 {FIXED_QUOTA},Σ Quota_i = {sum(FIXED_QUOTA.values())} = 输出候选数")
    print(f"{'池':>16}{'池内候选数':>12}{'池内最高 score':>16}{'池内第 Quota 名 score':>22}")
    for p in POOLS:
        q = FIXED_QUOTA[p]
        top = pools[p][0].score if pools[p] else 0.0
        cut = pools[p][q - 1].score if len(pools[p]) >= q else float("nan")
        print(f"{p:>16}{len(pools[p]):>12}{top:>16.5f}{cut:>22.5f}")

    quota_set = apply_quota(pools, FIXED_QUOTA)
    free_set = unconstrained_topk(cands, OUTPUT_SIZE)
    print(f"\n{'策略':>18}{'候选数':>8}" + "".join(f"{p+'占比':>14}" for p in POOLS)
          + f"{'平均 score':>12}{'池熵':>8}{'新主播数':>10}")
    for name, sel in (("无约束 TopK(score)", free_set), ("固定 Quota ∪TopK_i", quota_set)):
        sh = share(sel)
        avg = sum(x.score for x in sel) / len(sel)
        newn = sum(1 for x in sel if x.is_new_anchor)
        print(f"{name:>18}{len(sel):>8}" + "".join(f"{sh[p]:>14.3f}" for p in POOLS)
              + f"{avg:>12.5f}{entropy(sh):>8.4f}{newn:>10}")
    a_free = sum(x.score for x in free_set) / len(free_set)
    a_quota = sum(x.score for x in quota_set) / len(quota_set)
    print(f"\nQuota 让平均融合分从 {a_free:.5f} 降到 {a_quota:.5f}"
          f"(相对损失 {(a_free - a_quota) / a_free:.2%}),"
          f"换来池熵 {entropy(share(free_set)):.4f} -> {entropy(share(quota_set)):.4f}"
          f" 与新主播曝光 {sum(1 for x in free_set if x.is_new_anchor)} -> "
          f"{sum(1 for x in quota_set if x.is_new_anchor)}。")

    # ---- 2. 动态 Quota ----
    print("\n" + "=" * 100)
    print("动态 Quota:Quota_i = f(User, Context, BusinessGoal)(凸组合 + 最大余数取整)")
    print("=" * 100)
    biz = {"game": 0.20, "entertainment": 0.20, "ecommerce": 0.50, "new": 0.10}
    floor = {"game": 0.25, "entertainment": 0.25, "ecommerce": 0.25, "new": 0.25}
    profiles = {
        "重度游戏用户 P(Game|U)=0.9": {"game": 0.90, "entertainment": 0.06,
                                 "ecommerce": 0.03, "new": 0.01},
        "电商意图用户": {"game": 0.15, "entertainment": 0.15,
                    "ecommerce": 0.65, "new": 0.05},
        "冷启新用户(兴趣未知)": {"game": 0.30, "entertainment": 0.30,
                          "ecommerce": 0.20, "new": 0.20},
    }
    print(f"{'用户画像':>24}" + "".join(f"{p:>14}" for p in POOLS)
          + f"{'平均 score':>12}{'池熵':>8}")
    print(f"{'固定 Quota(基线)':>24}"
          + "".join(f"{FIXED_QUOTA[p]:>14}" for p in POOLS)
          + f"{a_quota:>12.5f}{entropy(share(quota_set)):>8.4f}")
    for name, pref in profiles.items():
        dq = dynamic_quota(pref, biz, floor, OUTPUT_SIZE)
        sel = apply_quota(pools, dq)
        sh = share(sel)
        avg = sum(x.score for x in sel) / len(sel)
        print(f"{name:>24}" + "".join(f"{dq[p]:>14}" for p in POOLS)
              + f"{avg:>12.5f}{entropy(sh):>8.4f}")
    dq_game = dynamic_quota(profiles["重度游戏用户 P(Game|U)=0.9"], biz, floor, OUTPUT_SIZE)
    print(f"\n重度游戏用户的游戏配额由固定 40 提升到 {dq_game['game']},"
          f"电商配额由 20 降到 {dq_game['ecommerce']} -- 与正文「Quota_game=50%、"
          f"Quota_ecommerce=10%」的调控方向一致;")
    print("生态保底项 floor 保证了即使用户兴趣极度集中,各池仍有最低配额,不会被完全挤出。")

    # ---- 3. 闭环控制 π(s) -> Quota ----
    print("\n" + "=" * 100)
    print("闭环 Quota 调控 π(s)->Quota:PI 控制律把实际曝光占比拉回目标")
    print("=" * 100)
    target = {"game": 0.40, "entertainment": 0.30, "ecommerce": 0.20, "new": 0.10}
    # [MOCK] 下游通过率:电商与新主播在精排/用户侧更容易被淘汰,开环配额必然偏离目标
    pass_rate = {"game": 0.95, "entertainment": 0.80, "ecommerce": 0.45, "new": 0.35}
    print(f"目标曝光占比 {target}")
    print(f"[MOCK] 下游通过率 {pass_rate}\n")

    print("开环(固定配额,不反馈):")
    rng_open = random.Random(7)
    open_err = []
    for t in range(1, 7):
        real = mock_downstream(FIXED_QUOTA, pass_rate, rng_open)
        mae = sum(abs(target[p] - real[p]) for p in POOLS) / len(POOLS)
        open_err.append(mae)
        print(f"  轮 {t}: 实际占比 " + " ".join(f"{p}={real[p]:.3f}" for p in POOLS)
              + f" | MAE={mae:.4f}")

    print("\n闭环(PI 控制,K_p=0.45, K_i=0.30):")
    ctrl = QuotaController(target)
    rng_cl = random.Random(7)
    conv = None
    print(f"{'轮':>4}" + "".join(f"{p+' q/占比':>18}" for p in POOLS) + f"{'MAE':>9}")
    for t in range(1, 13):
        q = ctrl.act()
        real = mock_downstream(q, pass_rate, rng_cl)
        mae = sum(abs(target[p] - real[p]) for p in POOLS) / len(POOLS)
        if conv is None and mae <= 0.02:
            conv = t
        print(f"{t:>4}" + "".join(f"{q[p]:>9}/{real[p]:>8.3f}" for p in POOLS)
              + f"{mae:>9.4f}")
        ctrl.update(real)
    print(f"\n开环 MAE 均值 {sum(open_err)/len(open_err):.4f},始终无法收敛(电商/新主播常年欠曝光);")
    print(f"闭环首次把 MAE 压到 ≤0.02 的轮次 = {conv};稳态配额 "
          + " ".join(f"{p}={ctrl.act()[p]}" for p in POOLS))
    print("控制律通过放大高损耗池的配额来补偿下游损失,这正是 π(s)->Quota 在线优化的最小形态。")
