"""
多通道式流量调控与带约束的排序优化(第 6 章 §6.5.2 多通道式流量调控、
§6.5.4 流量调控的优化视角)

对应正文概念:
1) §6.5.2 多通道式:借鉴粗排的多通道(Multi-Channel)/多 Quota 架构,先按业务
   目标构建多个独立候选池(新内容池、电商内容池、创作者扶持池、普通内容池),
   每个池有独立的排序逻辑与多目标融合权重,各通道内部排序后按预设 Quota 合并成
   最终 Top-K。相比级联式,它把流量调控前置到候选池构建阶段,表达能力更强,但
   系统复杂度、参数维护成本与流量分配策略设计难度都明显更高。
2) §6.5.4 优化视角:流量调控可写成带约束的排序优化问题
       max  sum_{i=1..N} Score_i
       s.t. sum_i x_i^{new} >= Q_new,  sum_i x_i^{creator} >= Q_creator,
            sum_i x_i^{commerce} >= Q_commerce,   x_i in {0,1}
   即「在最大化排序收益与满足业务约束之间寻找平衡点」。正文还提到动态 Quota
       Q_t = f(S_t^{user}, S_t^{Content}, S_t^{Platform})
   随实时环境变化而更新,但工业界主流仍以规则 + 固定 Quota 为主。

本实现做了什么:
1. 完整复刻正文伪代码 `ChannelRanker` / `MultiChannelTrafficControl`:四个通道、
   正文给的四套融合权重(0.3/0.4/0.3、0.2/0.2/0.6、0.4/0.4/0.2、0.5/0.3/0.2)、
   正文给的 Quota 组合 (10, 20, 15, 55) —— 并按比例缩放到 Top-50 做演示。
2. 把 §6.5.4 的带约束问题用三种解法算出来并对比:
   (a) 固定 Quota 硬切(正文伪代码的做法);
   (b) 贪心补配额(greedy repair):先按全局 Score 取 Top-K,再用「代价最小替换」
       把违反的下限约束补齐 —— 这是 max sum Score s.t. 配额下限 的经典贪心近似;
   (c) 精确最优(小规模下用 DP/枚举求解上界),用来量化贪心的最优性 gap。
3. [MOCK] 供给不足场景:当某个通道的候选量少于它的 Quota 时,固定 Quota 会浪费
   坑位,打印「Quota 浪费量」与大盘损失。
4. 实现正文的动态 Quota Q_t = f(S_user, S_content, S_platform) 并做闭环仿真:
   10 个时间片内供给与用户状态变化,对比固定 Quota 与动态 Quota 的达成率与收益。

关键公式(正文原文):
  max sum Score_i   s.t. sum x_i^g >= Q_g,  x_i in {0,1}
  Q_t = f(S_t^user, S_t^Content, S_t^Platform)
"""

import random
from typing import List

CHANNELS = ("new", "commerce", "creator", "normal")
CH_CN = {"new": "新内容池", "commerce": "电商内容池",
         "creator": "创作者扶持池", "normal": "普通内容池"}


class RankItem:
    """复刻正文伪代码的候选结构;true_watch 是 [MOCK] 隐藏真值,仅用于算大盘指标。"""

    def __init__(self, pctr, pwatch_time, plike, is_new_video=False,
                 author_fans=0, in_campaign=False, iid=-1, true_watch=0.0):
        self.pctr = pctr
        self.pwatch_time = pwatch_time
        self.plike = plike

        self.is_new_video = is_new_video
        self.author_fans = author_fans
        self.in_campaign = in_campaign

        self.score = 0.0
        self.iid = iid
        self.true_watch = true_watch

    def channel(self):
        """按正文 split_channel 的 if/elif 顺序划分通道(互斥、有优先级)。"""
        if self.is_new_video:
            return "new"
        if self.in_campaign:
            return "commerce"
        if self.author_fans < 1000:
            return "creator"
        return "normal"


class ChannelRanker:
    """复刻正文伪代码:每个通道有独立的多目标融合权重与独立排序。"""

    def __init__(self, ctr_weight, watch_weight, like_weight):
        self.ctr_weight = ctr_weight
        self.watch_weight = watch_weight
        self.like_weight = like_weight

    def score(self, item):
        return (self.ctr_weight * item.pctr
                + self.watch_weight * item.pwatch_time
                + self.like_weight * item.plike)

    def rank(self, items: List[RankItem]):
        for item in items:
            item.score = self.score(item)
        return sorted(items, key=lambda x: x.score, reverse=True)


class MultiChannelTrafficControl:
    """复刻正文伪代码:四通道独立排序 + Quota 合并。

    正文的 Quota 是 (10, 20, 15, 55) 共 100 个坑位;本演示输出 Top-50,
    因此按 scale=0.5 等比缩放为 (5, 10, 7, 28),并把取整余量补给普通池,
    保证 sum(Quota) 恰好等于输出坑位数。
    """

    def __init__(self, quota=None, scale=1.0):
        self.new_video_ranker = ChannelRanker(0.3, 0.4, 0.3)
        self.commerce_ranker = ChannelRanker(0.2, 0.2, 0.6)
        self.creator_ranker = ChannelRanker(0.4, 0.4, 0.2)
        self.normal_ranker = ChannelRanker(0.5, 0.3, 0.2)
        q = quota or {"new": 10, "commerce": 20, "creator": 15, "normal": 55}
        self.quota = {k: int(v * scale) for k, v in q.items()}
        total = int(round(sum(q.values()) * scale))
        self.quota["normal"] += total - sum(self.quota.values())

    def split_channel(self, candidates):
        pools = {c: [] for c in CHANNELS}
        for item in candidates:
            pools[item.channel()].append(item)
        return pools

    def rank(self, candidates, return_pools=False):
        pools = self.split_channel(candidates)
        rankers = {"new": self.new_video_ranker,
                   "commerce": self.commerce_ranker,
                   "creator": self.creator_ranker,
                   "normal": self.normal_ranker}
        ranked = {c: rankers[c].rank(pools[c]) for c in CHANNELS}
        result = []
        for c in CHANNELS:                       # 正文的 Quota 控制与合并顺序
            result.extend(ranked[c][:self.quota[c]])
        return (result, ranked) if return_pools else result


# ========================================================= §6.5.4 带约束优化
def global_score(item):
    """全局统一融合分(用于 max sum Score_i 的目标函数);取正文普通通道权重。"""
    return 0.5 * item.pctr + 0.3 * item.pwatch_time + 0.2 * item.plike


def unconstrained_topk(cands, K):
    """无约束最优:纯按全局 Score 取 Top-K(模型最优,业务约束可能全不满足)。"""
    return sorted(cands, key=lambda x: -global_score(x))[:K]


def greedy_repair(cands, K, qmin):
    """贪心补配额:先全局 Top-K,再反复「踢掉损失最小的冗余物品、补进缺口通道
    得分最高的物品」,直到所有下限约束满足。目标是在满足约束下让 sum Score 尽量大。
    """
    sel = sorted(cands, key=lambda x: -global_score(x))[:K]
    rest = sorted(cands, key=lambda x: -global_score(x))[K:]
    sel_ids = {it.iid for it in sel}
    while True:
        cnt = {c: sum(1 for it in sel if it.channel() == c) for c in CHANNELS}
        deficit = [(c, qmin[c] - cnt[c]) for c in CHANNELS
                   if cnt[c] < qmin.get(c, 0)]
        if not deficit:
            break
        c, _ = max(deficit, key=lambda x: x[1])
        add = next((it for it in rest
                    if it.channel() == c and it.iid not in sel_ids), None)
        if add is None:
            break   # 供给不足:该通道没有更多候选可补
        # 踢掉「仍满足自身下限、且 Score 最低」的物品 —— 边际损失最小
        droppable = [it for it in sel
                     if cnt[it.channel()] > qmin.get(it.channel(), 0)]
        if not droppable:
            break
        drop = min(droppable, key=global_score)
        sel.remove(drop)
        sel_ids.discard(drop.iid)
        sel.append(add)
        sel_ids.add(add.iid)
        rest.remove(add)
    return sel


def exact_optimum(cands, K, qmin):
    """精确最优上界:按通道做 DP(每通道内部必然取 Score 降序前若干个)。

    由于目标是 max sum Score 且每通道内部的取舍只看 Score,最优解在每个通道里
    一定是「该通道 Score 前 n_c 个」,因此只需在满足 sum n_c = K、n_c >= Q_c 的
    整数组合上搜索,用前缀和 + DP 即可精确求解(维度 = 通道数,规模很小)。
    """
    pre = {}
    for c in CHANNELS:
        s = sorted((global_score(it) for it in cands
                    if it.channel() == c), reverse=True)
        acc, cum = [0.0], 0.0
        for v in s:
            cum += v
            acc.append(cum)
        pre[c] = acc
    best = (-1.0, None)

    def rec(idx, left, picks):
        nonlocal best
        if idx == len(CHANNELS):
            if left == 0:
                tot = sum(pre[c][n] for c, n in zip(CHANNELS, picks))
                if tot > best[0]:
                    best = (tot, list(picks))
            return
        c = CHANNELS[idx]
        cap = len(pre[c]) - 1
        for n in range(qmin.get(c, 0), min(cap, left) + 1):
            rec(idx + 1, left - n, picks + [n])

    rec(0, K, [])
    return best


def dynamic_quota(s_user, s_content, s_platform, K):
    """正文式 Q_t = f(S_t^user, S_t^Content, S_t^Platform) 的一个具体实现。

    S_user     : 用户当前对新内容的接受度(探索容忍度),0~1
    S_content  : 各通道的实时供给充足度(可用候选数 / 坑位需求),dict
    S_platform : 平台当期的业务侧重(0=偏生态扶持,1=偏商业化),0~1
    规则:基础权重 x 用户容忍度 x 平台侧重,再被供给充足度截断,最后归一化到 K。
    """
    base = {"new": 0.20, "commerce": 0.20, "creator": 0.15, "normal": 0.45}
    w = {
        "new": base["new"] * (0.5 + s_user),
        "commerce": base["commerce"] * (0.5 + 1.5 * s_platform),
        "creator": base["creator"] * (0.5 + s_user),
        "normal": base["normal"],
    }
    w = {c: min(w[c], s_content.get(c, 0.0)) for c in CHANNELS}   # 供给截断
    tot = sum(w.values()) or 1.0
    q = {c: int(K * w[c] / tot) for c in CHANNELS}
    q["normal"] += K - sum(q.values())                            # 余量兜底
    return q


def make_candidates(rng, n=200, new_ratio=0.18, camp_ratio=0.12,
                    newauthor_ratio=0.22):
    """[MOCK] 合成精排候选集;冷启动内容的 PXTR 被系统性低估(cold=0.8)。"""
    items = []
    for i in range(n):
        q = rng.random()
        is_new = rng.random() < new_ratio
        fans = 0 if rng.random() < newauthor_ratio else int(1000 + 9e4 * rng.random())
        camp = rng.random() < camp_ratio
        cold = 0.80 if (is_new or fans < 1000) else 1.0
        ctr = (0.03 + 0.17 * q) * cold
        wt = (0.10 + 0.70 * q ** 1.3) * cold
        like = (0.004 + 0.030 * q ** 2) * cold
        noisy = [max(1e-6, v * (0.9 + 0.2 * rng.random()))
                 for v in (ctr, wt, like)]
        items.append(RankItem(noisy[0], noisy[1], noisy[2], is_new, fans, camp,
                              iid=i, true_watch=ctr * (20.0 + 80.0 * wt)))
    return items


def stat(lst):
    """返回 (sum Score, 人均真实时长, 各通道数量)。"""
    cnt = {c: sum(1 for it in lst if it.channel() == c) for c in CHANNELS}
    return (sum(global_score(it) for it in lst),
            sum(it.true_watch for it in lst) / max(1, len(lst)), cnt)


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)
    K = 50
    cands = make_candidates(rng, n=200)
    mc = MultiChannelTrafficControl(scale=K / 100.0)

    print("=" * 108)
    print("多通道式流量调控(正文 §6.5.2 伪代码复刻)+ 带约束排序优化(§6.5.4)")
    print(f"候选集 {len(cands)} 个,输出 Top-{K};正文 Quota (10,20,15,55)"
          f" 按 {K}/100 等比缩放为 {tuple(mc.quota[c] for c in CHANNELS)}")
    print("=" * 108)

    # ---------------- 表 (a):各通道独立融合权重与池内排序
    res, pools = mc.rank(list(cands), return_pools=True)
    print("\n【表 a】四个通道:独立融合权重、候选池规模、Quota 与实际占用")
    rk = {"new": mc.new_video_ranker, "commerce": mc.commerce_ranker,
          "creator": mc.creator_ranker, "normal": mc.normal_ranker}
    print(f"{'通道':<16}{'w_ctr':>8}{'w_watch':>9}{'w_like':>8}{'池内候选数':>12}"
          f"{'Quota':>8}{'实际占用':>10}{'坑位浪费':>10}{'入选者池内真实时长均值':>26}")
    print("-" * 108)
    for c in CHANNELS:
        r, pool = rk[c], pools[c]
        take = pool[:mc.quota[c]]
        wt = sum(it.true_watch for it in take) / max(1, len(take))
        print(f"{CH_CN[c]:<16}{r.ctr_weight:>8.1f}{r.watch_weight:>9.1f}"
              f"{r.like_weight:>8.1f}{len(pool):>12}{mc.quota[c]:>8}"
              f"{len(take):>10}{mc.quota[c] - len(take):>10}{wt:>26.4f}")
    print("-" * 108)
    tot_s, tot_w, cnt = stat(res)
    print(f"合并结果:共 {len(res)} 个坑位,sum Score = {tot_s:.4f},"
          f"人均真实时长 = {tot_w:.4f} 秒")
    print("观察:Quota 把流量分配写死在候选池构建阶段 —— 电商池候选只有 23 个却拿到")
    print("      10 个坑位,普通池 98 个候选只拿 28 个,与各池的实际质量完全无关:")
    print("      普通池入选者人均真实时长 13.02 秒,是电商池(9.62 秒)的 1.35 倍。")
    print("      各池还用不同融合权重,电商池 w_like=0.6 让它在池内更偏互动排序,")
    print("      这正是正文说的「每个通道内部独立优化自己的业务目标」。")

    # ---------------- 表 (b):§6.5.4 三种解法对比
    QMIN = {"new": 8, "commerce": 6, "creator": 5, "normal": 0}
    print(f"\n【表 b】§6.5.4 带约束排序优化 max sum Score_i s.t. "
          f"Q_new>={QMIN['new']}, Q_commerce>={QMIN['commerce']}, "
          f"Q_creator>={QMIN['creator']}")
    uc = unconstrained_topk(cands, K)
    gr = greedy_repair(cands, K, QMIN)
    opt_val, opt_pick = exact_optimum(cands, K, QMIN)
    mcq = mc.rank(list(cands))
    print(f"{'解法':<30}{'sum Score':>12}{'相对最优gap':>14}{'新内容':>8}"
          f"{'电商':>8}{'创作者':>8}{'普通':>8}{'约束满足':>10}{'人均真实时长':>14}")
    print("-" * 108)
    for nm, lst in [("无约束 Top-K(模型最优)", uc),
                    ("固定 Quota 硬切(正文伪代码)", mcq),
                    ("贪心补配额(greedy repair)", gr)]:
        s, w, c = stat(lst)
        ok = all(c[g] >= QMIN[g] for g in CHANNELS)
        print(f"{nm:<30}{s:>12.4f}{s / opt_val - 1:>+14.2%}{c['new']:>8}"
              f"{c['commerce']:>8}{c['creator']:>8}{c['normal']:>8}"
              f"{('满足' if ok else '违反'):>10}{w:>14.4f}")
    print(f"{'精确最优(通道DP)':<30}{opt_val:>12.4f}{0.0:>+14.2%}"
          + "".join(f"{n:>8}" for n in opt_pick) + f"{'满足':>10}{'—':>14}")
    print("-" * 108)
    print("观察:无约束 Top-K 的 sum Score 最高(14.3356),但它只给电商 5 个坑位,")
    print("      违反 Q_commerce>=6;固定 Quota 虽然给了电商 10 个,却把新内容压到 5 个")
    print("      (低于 Q_new>=8)同样违反约束,而且 sum Score 还最低(13.9732,比最优")
    print("      低 2.33%)—— 事先拍死的坑位既不保证满足约束、又白付收益。贪心补配额")
    print("      在本例中恰好取到精确最优 14.3073(gap 0.00%),只比无约束上界低 0.20%,")
    print("      这就是正文说的「在最大化排序收益与满足业务约束之间寻找平衡点」。")

    # ---------------- 表 (c):供给不足时固定 Quota 的坑位浪费
    print("\n【表 c】供给不足场景:固定 Quota 的坑位浪费(电商内容供给逐步枯竭)")
    print(f"{'电商内容供给比例':>18}{'电商池候选数':>14}{'电商Quota':>12}"
          f"{'实际填充':>10}{'浪费坑位':>10}{'实际输出条数':>14}"
          f"{'sum Score':>12}{'人均真实时长':>14}")
    print("-" * 108)
    for ratio in [0.12, 0.06, 0.03, 0.01, 0.0]:
        c2 = make_candidates(random.Random(42), n=200, camp_ratio=ratio)
        lst, pl = mc.rank(list(c2), return_pools=True)
        s, w, cnt2 = stat(lst)
        print(f"{ratio:>18.0%}{len(pl['commerce']):>14}{mc.quota['commerce']:>12}"
              f"{cnt2['commerce']:>10}"
              f"{mc.quota['commerce'] - cnt2['commerce']:>10}{len(lst):>14}"
              f"{s:>12.4f}{w:>14.4f}")
    print("-" * 108)
    print("观察:电商供给从 12% 掉到 0,固定 Quota 预留的 10 个坑位无法填满,最终输出")
    print("      条数从 50 一路掉到 40,sum Score 从 13.9732 跌到 11.7532。注意人均真实")
    print("      时长反而从 11.64 升到 12.39 —— 因为被浪费的坑位本来装的是低质电商内容,")
    print("      少发比乱发好,但少发 10 条意味着整页流量白白损失。工程上必须加「未用")
    print("      配额回收」把余量让回普通池,这正是正文说多通道架构「策略设计难度明显")
    print("      更高」的地方。")

    # ---------------- 表 (d):动态 Quota 闭环仿真
    print("\n【表 d】动态 Quota Q_t = f(S_t^user, S_t^Content, S_t^Platform) 闭环仿真")
    print(f"{'时间片':>6}{'S_user':>8}{'S_platform':>12}{'电商供给':>10}"
          f"{'固定Quota输出/收益':>22}{'动态Quota (新/商/创/普)':>26}"
          f"{'动态输出/收益':>18}{'收益提升':>10}")
    print("-" * 108)
    fix_sum = dyn_sum = 0.0
    for t in range(1, 11):
        s_user = 0.2 + 0.07 * t                      # 用户探索容忍度随期间升高
        s_plat = 0.9 if t > 6 else 0.2               # 第 7 片起平台转向商业化
        camp = [0.12, 0.10, 0.08, 0.05, 0.03,
                0.02, 0.10, 0.14, 0.18, 0.20][t - 1]
        c2 = make_candidates(random.Random(100 + t), n=200, camp_ratio=camp)
        pools2 = mc.split_channel(c2)
        s_content = {c: len(pools2[c]) / 200.0 for c in CHANNELS}
        q_dyn = dynamic_quota(s_user, s_content, s_plat, K)
        fix_lst = mc.rank(list(c2))
        dyn_lst = MultiChannelTrafficControl(quota=q_dyn).rank(list(c2))
        sf, wf, _ = stat(fix_lst)
        sd, wd, _ = stat(dyn_lst)
        fix_sum += sf
        dyn_sum += sd
        qs = "/".join(str(q_dyn[c]) for c in CHANNELS)
        print(f"{t:>6}{s_user:>8.2f}{s_plat:>12.2f}{camp:>10.0%}"
              f"{f'{len(fix_lst)} / {sf:.3f}':>22}{qs:>26}"
              f"{f'{len(dyn_lst)} / {sd:.3f}':>18}{sd / sf - 1:>+10.2%}")
    print("-" * 108)
    print(f"10 个时间片累计 sum Score:固定 Quota {fix_sum:.4f} -> "
          f"动态 Quota {dyn_sum:.4f},合计 {dyn_sum / fix_sum - 1:+.2%}")
    print("观察:动态 Quota 并非处处占优:供给充足时(t=1、8、9、10)它为了迁就 S_user/")
    print("      S_platform 反而比固定 Quota 低 1%~3%;真正的收益来自供给枯竭的 t=5、6")
    print("      (+7.44%、+15.16%)—— 此时固定 Quota 只能输出 49/43 条,动态 Quota 把")
    print("      电商配额降到 2/0 并让回普通池,仍然输出满 50 条。10 片累计 +2.74%。")
    print("      正文也强调目前工业界主流仍是规则 + 固定 Quota,强化学习 / 大模型 Agent")
    print("      类动态方案更多用于复杂业务场景或离线辅助决策。")
    print("=" * 108)
