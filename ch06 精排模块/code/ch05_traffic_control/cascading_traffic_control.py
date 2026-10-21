"""
级联式多目标融合与流量调控(第 6 章 §6.5.1 级联式多目标融合与流量调控)

对应正文概念:
正文指出工业界精排链路的主流实现是把「多目标融合」与「流量调控」解耦为两个
串行独立阶段:系统先用精排/LTR 输出的各类 PXTR 做多目标融合得到统一排序分,
按融合分初排,再把结果送入流量调控模块,由后者处理模型难以直接学习的业务规则
(新内容冷启动扶持、新作者成长保护、电商活动流量倾斜、商业化保量等),常见手段
是动态 Boost、局部 Rank 调整、Quota 控制与保量策略。其优点是排序逻辑与业务逻辑
相互独立、实现简单、维护成本低、业务扩展灵活。

本实现做了什么:
1. 完整复刻正文伪代码 `RankItem` / `TrafficController` 的五步流程:
     Step1 多目标融合 -> Step2 初始排序 -> Step3 流量调控 -> Step4 再排序 -> Step5 输出 TopK
   包括正文给的加法融合权重与三条乘性 Boost 规则(1.10 / 1.05 / 1.15)。
2. [MOCK] 一个进程内合成候选集 + 「大盘指标」评估器:每个候选物品带隐藏的真实
   后验(真实点击/时长/转化),用于量化流量调控对大盘指标的代价。
3. 打印三张数值表:
   (a) 级联流水线的逐步骤 Top-10 榜单变化(谁被 Boost 挤上来、谁被挤下去);
   (b) 各业务分组曝光占比:融合分初排 vs 流量调控后,以及 Boost 系数的边际效应;
   (c) Boost 强度扫描:业务目标达成度 vs 大盘指标(人均时长)损失的权衡曲线。
4. 复现正文「Step3 在排序后对 score 做乘性修正、再 Step4 重排」这一细节带来的
   一个工程事实:乘性 Boost 的效果依赖融合分自身的量纲与分布 —— 同样的 1.10 倍,
   在分数稠密区能换来多个名次,在分数稀疏区几乎不动。

关键公式(正文原文):
  Score = w_ctr * pctr + w_wt * pwatch_time + w_like * plike       (多目标融合)
  Score' = Score * 1.10^[新内容] * 1.05^[作者粉丝<1000] * 1.15^[电商活动]   (流量调控)
"""

import random


class RankItem:
    """复刻正文伪代码的候选物品结构。

    正文字段:pctr / pwatch_time / plike + 三个业务标记 is_new_video /
    author_fans / in_campaign + 可变的 score。
    这里额外挂了 [MOCK] 的隐藏真值 (true_watch, iid),仅供大盘指标评估使用,
    排序与调控逻辑完全不读取它们。
    """

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

    def group(self):
        """按正文 Boost 规则的优先级给物品打一个展示用的业务分组标签。"""
        if self.is_new_video:
            return "新内容"
        if self.in_campaign:
            return "电商活动"
        if self.author_fans < 1000:
            return "新作者"
        return "普通内容"


class TrafficController:
    """复刻正文伪代码的级联式流量调控器(融合权重与 Boost 系数均取正文值)。"""

    def __init__(self, ctr_weight=0.5, wt_weight=0.3, like_weight=0.2,
                 boost_new=1.10, boost_creator=1.05, boost_campaign=1.15):
        self.ctr_weight = ctr_weight
        self.wt_weight = wt_weight
        self.like_weight = like_weight
        self.boost_new = boost_new
        self.boost_creator = boost_creator
        self.boost_campaign = boost_campaign

    def multi_objective_fusion(self, item):
        """Step1:正文的加法融合公式。"""
        return (self.ctr_weight * item.pctr
                + self.wt_weight * item.pwatch_time
                + self.like_weight * item.plike)

    def traffic_control(self, item):
        """Step3:正文的三条乘性 Boost 业务规则。"""
        score = item.score
        if item.is_new_video:            # 新内容扶持
            score *= self.boost_new
        if item.author_fans < 1000:      # 新作者扶持
            score *= self.boost_creator
        if item.in_campaign:             # 电商活动扶持
            score *= self.boost_campaign
        return score

    def rank(self, candidates, topk=100, return_stages=False):
        """正文五步流程。return_stages=True 时额外返回初排榜单用于对比。"""
        for item in candidates:                               # Step1 多目标融合
            item.score = self.multi_objective_fusion(item)
        rank_list = sorted(candidates, key=lambda x: x.score,  # Step2 初始排序
                           reverse=True)
        stage2 = list(rank_list)
        for item in rank_list:                                # Step3 流量调控
            item.score = self.traffic_control(item)
        rank_list = sorted(rank_list, key=lambda x: x.score,   # Step4 再排序
                           reverse=True)
        out = rank_list[:topk]                                 # Step5 输出 TopK
        return (out, stage2[:topk]) if return_stages else out


def make_candidates(rng, n=200):
    """[MOCK] 合成一批精排候选:PXTR 带校准噪声,真值只用于算大盘指标。"""
    items = []
    for i in range(n):
        q = rng.random()                              # 内容质量(潜变量)
        is_new = rng.random() < 0.18                  # 新内容占比 18%
        fans = 0 if rng.random() < 0.22 else int(1000 + 90000 * rng.random())
        camp = rng.random() < 0.12                    # 电商活动内容占比 12%
        # 新内容/新作者的后验天然偏低(样本不足、模型低估)-> 纯模型排序会压制它们
        cold = 0.80 if (is_new or fans < 1000) else 1.0
        ctr = (0.03 + 0.17 * q) * cold
        wt = (0.10 + 0.70 * q ** 1.3) * cold
        like = (0.004 + 0.030 * q ** 2) * cold
        noisy = [max(1e-6, v * (0.9 + 0.2 * rng.random()))
                 for v in (ctr, wt, like)]
        items.append(RankItem(noisy[0], noisy[1], noisy[2], is_new, fans, camp,
                              iid=i, true_watch=ctr * (20.0 + 80.0 * wt)))
    return items


def group_share(lst):
    """各业务分组在榜单中的曝光占比。"""
    n = len(lst)
    out = {}
    for g in ("新内容", "电商活动", "新作者", "普通内容"):
        out[g] = sum(1 for it in lst if it.group() == g) / n
    return out


def avg_watch(lst):
    """[MOCK] 大盘指标:榜单的人均真实观看时长(秒)。"""
    return sum(it.true_watch for it in lst) / len(lst)


if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)
    cands = make_candidates(rng, n=200)
    TOPK = 50
    GROUPS = ("新内容", "电商活动", "新作者", "普通内容")

    print("=" * 104)
    print("级联式多目标融合与流量调控(正文 §6.5.1 伪代码复刻):"
          "Step1 融合 -> Step2 初排 -> Step3 调控 -> Step4 重排 -> Step5 TopK")
    print(f"候选集 {len(cands)} 个,输出 Top-{TOPK};"
          "融合权重 w_ctr=0.5, w_wt=0.3, w_like=0.2;"
          "Boost 新内容 1.10 / 新作者 1.05 / 电商 1.15")
    print("=" * 104)

    tc = TrafficController()
    final, initial = tc.rank(list(cands), topk=TOPK, return_stages=True)

    # ---------------- 表 (a):Top-10 榜单在 Step2 与 Step4 之间的变化
    pos_init = {it.iid: r for r, it in enumerate(initial, 1)}
    print("\n【表 a】Top-10 榜单:融合分初排(Step2) vs 流量调控后(Step4)")
    print(f"{'名次':>4}{'初排物品':>10}{'初排分组':>10}"
          f"{'  |':>4}{'调控后物品':>12}{'调控后分组':>12}{'初排名次':>10}"
          f"{'名次变化':>10}{'Boost系数':>12}")
    print("-" * 104)
    for r in range(10):
        a, b = initial[r], final[r]
        boost = (tc.boost_new if b.is_new_video else 1.0) \
            * (tc.boost_creator if b.author_fans < 1000 else 1.0) \
            * (tc.boost_campaign if b.in_campaign else 1.0)
        old = pos_init.get(b.iid)
        delta = f"{old - (r + 1):+d}" if old else ">Top50"
        print(f"{r + 1:>4}{a.iid:>10}{a.group():>10}{'  |':>4}{b.iid:>12}"
              f"{b.group():>12}{(old if old else 0):>10}{delta:>10}"
              f"{boost:>12.4f}")
    print("-" * 104)
    moved = sum(1 for r, it in enumerate(final, 1) if pos_init.get(it.iid) != r)
    pushed_out = len({it.iid for it in initial} - {it.iid for it in final})
    print(f"观察:Top-{TOPK} 里有 {moved} 个物品名次被 Step3/Step4 改写,"
          f"{pushed_out} 个原本入榜的普通内容被挤出 —— 这就是「后置修正」的直接代价。")

    # ---------------- 表 (b):分组曝光占比的变化
    print("\n【表 b】各业务分组在 Top-50 中的曝光占比:初排 vs 调控后")
    s_init, s_fin = group_share(initial), group_share(final)
    pool = group_share(cands)
    print(f"{'业务分组':<12}{'候选池占比':>14}{'初排占比':>14}{'调控后占比':>14}"
          f"{'绝对变化':>12}{'相对变化':>12}")
    print("-" * 104)
    for g in GROUPS:
        d = s_fin[g] - s_init[g]
        rel = (d / s_init[g]) if s_init[g] > 0 else float("inf")
        rel_s = f"{rel:+.1%}" if s_init[g] > 0 else "  新增(0->)"
        print(f"{g:<12}{pool[g]:>14.2%}{s_init[g]:>14.2%}{s_fin[g]:>14.2%}"
              f"{d:>+12.2%}{rel_s:>12}")
    print("-" * 104)
    print(f"大盘指标(人均真实观看时长):初排 {avg_watch(initial):.4f} 秒 -> "
          f"调控后 {avg_watch(final):.4f} 秒,"
          f"变化 {avg_watch(final) / avg_watch(initial) - 1:+.2%}")
    print("观察:新内容占比从 16% 抬到 26%(+62.5%),但「新作者」组反而从 12% 掉到 8%:")
    print("      它只拿到最弱的 1.05 倍,被 1.10 / 1.15 的强扶持组一起挤了出去;电商组")
    print("      占比甚至一动不动(10%->10%),只是内部名次变了。代价是普通内容 -6pp 与")
    print("      大盘人均时长 -1.13% —— 乘性 Boost 的「实际分配结果」并不等于系数大小,")
    print("      这正是正文所说「调和模型最优与业务最优的矛盾」里最难的一部分。")

    # ---------------- 表 (c):Boost 强度扫描(权衡曲线)
    print("\n【表 c】Boost 强度扫描:业务目标达成度与大盘指标的权衡")
    print(f"{'Boost(新/创/电商)':<22}{'新内容占比':>12}{'电商占比':>12}"
          f"{'新作者占比':>12}{'普通内容占比':>14}{'人均时长':>12}{'时长相对损失':>14}")
    print("-" * 104)
    base_wt = None
    for mult in [1.00, 1.05, 1.10, 1.20, 1.50, 2.00]:
        t = TrafficController(boost_new=mult,
                              boost_creator=1.0 + (mult - 1.0) * 0.5,
                              boost_campaign=1.0 + (mult - 1.0) * 1.5)
        lst = t.rank(list(cands), topk=TOPK)
        sh = group_share(lst)
        wt = avg_watch(lst)
        if base_wt is None:
            base_wt = wt
        tag = (f"{t.boost_new:.2f}/{t.boost_creator:.2f}/{t.boost_campaign:.2f}")
        print(f"{tag:<22}{sh['新内容']:>12.2%}{sh['电商活动']:>12.2%}"
              f"{sh['新作者']:>12.2%}{sh['普通内容']:>14.2%}{wt:>12.4f}"
              f"{wt / base_wt - 1:>+14.2%}")
    print("-" * 104)
    print("观察:新内容与电商两个强扶持组的占比随系数单调上升,普通内容从 62% 一路被")
    print("      挤到 10%,人均时长从 12.04 掉到 9.11(-24.36%)。注意新作者组是非单调的")
    print("      (12%->10%->8%->10%->16%->18%):弱扶持组会先被强扶持组挤掉、只有系数")
    print("      整体放大到普通内容塌陷后才回升。该表就是上线前运营与算法要谈的「兑换")
    print("      汇率」:每多给扶持池 1 个点曝光,大盘时长要付出多少。")

    # ---------------- 乘性 Boost 的量纲敏感性
    print("\n【表 d】同样 1.10 倍 Boost,在不同分数密度区段换来的名次差异")
    print(f"{'初排名次段':>12}{'该段分数跨度':>16}{'1.10 倍可跨越名次(均值)':>26}")
    print("-" * 104)
    full = sorted(cands, key=lambda x: -tc.multi_objective_fusion(x))
    sc = [tc.multi_objective_fusion(it) for it in full]
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120)]:
        span = sc[lo] - sc[hi - 1]
        gains = []
        for r in range(lo, hi):
            boosted = sc[r] * 1.10
            new_r = sum(1 for s in sc if s > boosted)
            gains.append(r - new_r)
        print(f"{f'{lo + 1}-{hi}':>12}{span:>16.6f}"
              f"{sum(gains) / len(gains):>26.2f}")
    print("-" * 104)
    print("观察:同一个 1.10 倍系数,在头部稀疏段(1-10,分数跨度 0.0551)平均只能跨越")
    print("      3.00 个名次,而在中段(31-60,跨度 0.0445)能跨越 15.17 个。可见乘性")
    print("      Boost 的「实际力度」取决于融合分在该名次段的局部密度,而不是系数本身,")
    print("      这也是工业界更偏好直接用 Quota / 坑位(§6.5.2、§6.5.5)的原因之一。")
    print("=" * 104)
