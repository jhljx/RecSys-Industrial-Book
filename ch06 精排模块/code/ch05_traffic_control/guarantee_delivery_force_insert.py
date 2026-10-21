"""
精排保量与强插策略(第 6 章 §6.5.5 精排保量与强插策略;约束形式见 §6.5.4)

对应正文概念:
正文把保量(Guarantee Delivery)与 Quota 区分开:Quota 关注不同内容池之间的
流量分配比例,保量关注「某一类特定候选集合是否能够获得最低限度的曝光机会」,
写成带约束的排序优化问题:

    max  sum_{i in S} Score_i
    s.t. Exposure_target >= Q_target                    (正文公式)

其中 Exposure_target 是目标候选集合(白名单池:优质创作者 / 重点签约作者 /
官方活动内容 / 新业务形态)实际获得的曝光量,Q_target 是业务要求的最低保量。

正文给出的四类工程实现方式,本脚本逐一实现并量化对比:
  (1) 排序完成后强插(Force Insert):直接把白名单候选插入最终结果;
  (2) 在流量调控阶段预留固定坑位(Reserved Slots);
  (3) 把保量约束融入多目标融合,通过动态调整排序分数实现「更加平滑的流量控制」
      —— 本实现用一个拉格朗日式的分数加价 Score' = Score + lambda(乘性版本为
      Score * (1 + lambda)),lambda 由投放进度闭环反馈(pacing 控制);
  (0) 不做任何保量的纯模型最优基线,用来度量保量的收益代价。

本实现做了什么:
  表 a  单刷(一次请求 Top-K)四种实现的对比:保量达成、挤出了谁、sum Score 损失;
  表 b  强插位置(置顶 / 第 3 位 / 随机位 / 尾插)对曝光质量的影响 —— 正文说
        强插是「直接插入」,但插在哪里决定了代价;
  表 c  一天 24 个时间片的闭环 pacing 仿真:固定预留坑位 / 前置抢量(ASAP)/
        闭环动态 lambda 三种投放策略的保量完成率、分数损失与超投/欠投情况;
  表 d  保量目标 Q_target 从 0 抬到池子上限时的边际代价曲线(模型收益 vs 业务
        约束),对应正文「保量机制本质上是在模型最优与业务最优之间寻找平衡」。

关键公式:
    Exposure_target >= Q_target                              (保量约束)
    Score'_i = Score_i + lambda * 1[i in Target]             (加性加价)
    Score'_i = Score_i * (1 + lambda * 1[i in Target])       (乘性加价)
    lambda_{t+1} = clip(lambda_t + eta * (pace_t^need - pace_t^real), 0, lam_max)
    达成率   = Exposure_target / Q_target
    大盘代价 = sum Score(保量后) / sum Score(无约束 Top-K) - 1   (负值为损失)

[MOCK] 本脚本中的候选打分、白名单池、用户时长/点击反馈与 24 时间片流量曲线
       全部是进程内合成数据,不读取任何外部数据集,也不发起网络请求。
"""

import random

K = 50                      # 每次请求输出的坑位数
N_CAND = 200                # 精排候选集规模
TARGET_RATIO = 0.08         # 白名单(保量)内容在候选集中的占比
Q_TARGET = 6                # 单刷保量目标:白名单至少 6 条曝光


class Item:
    """[MOCK] 一个精排候选物品:融合分 + 隐藏的真实用户侧收益。"""

    __slots__ = ("iid", "score", "is_target", "watch", "ctr")

    def __init__(self, iid, score, is_target, watch, ctr):
        self.iid = iid
        self.score = score          # 多目标融合分(调控模块唯一可见的排序依据)
        self.is_target = is_target  # 是否属于保量白名单池
        self.watch = watch          # [MOCK] 真实观看时长(秒),线上才可观测
        self.ctr = ctr              # [MOCK] 真实点击率


def make_candidates(rng, n=N_CAND, target_ratio=TARGET_RATIO):
    """[MOCK] 合成候选集。白名单内容融合分整体偏低(这正是需要保量的原因),
    但真实时长并不差 —— 对应正文「不会因为短期数据不足而被排序模型完全压制」。"""
    items = []
    for i in range(n):
        is_t = rng.random() < target_ratio
        if is_t:
            score = max(1e-4, rng.gauss(0.21, 0.07))
            watch = max(0.0, rng.gauss(12.5, 4.0))
        else:
            score = max(1e-4, rng.gauss(0.26, 0.09))
            watch = max(0.0, rng.gauss(13.0, 5.0))
        items.append(Item(i, score, is_t, watch, min(0.95, score * 1.6)))
    return items


# ---------------------------------------------------------------- 四种实现

def plain_topk(cands, k=K):
    """基线 (0):无保量,纯按融合分取 Top-K(模型最优)。"""
    return sorted(cands, key=lambda x: -x.score)[:k]


def force_insert(cands, k=K, q=Q_TARGET, pos=0):
    """实现 (1) 强插:先出 Top-K,再把缺口数量的白名单内容插到 pos 位置,
    并从尾部挤掉同样数量的普通内容(总坑位数不变)。"""
    res = plain_topk(cands, k)
    have = sum(1 for x in res if x.is_target)
    need = max(0, q - have)
    if need == 0:
        return res
    chosen = set(x.iid for x in res)
    pool = [x for x in sorted(cands, key=lambda y: -y.score)
            if x.is_target and x.iid not in chosen][:need]
    out, removed = [], 0
    for x in reversed(res):
        if removed < len(pool) and not x.is_target:
            removed += 1
            continue
        out.append(x)
    out.reverse()
    p = min(max(pos, 0), len(out))
    return out[:p] + pool + out[p:]


def reserved_slots(cands, k=K, q=Q_TARGET):
    """实现 (2) 预留固定坑位:无条件留出 q 个坑位给白名单池(池内按分排序),
    剩余 k-q 个坑位再从非白名单内容中按融合分取。
    注意:坑位是「写死」的,即使白名单内容自然就能进 Top-K 也只给 q 个。"""
    srt = sorted(cands, key=lambda x: -x.score)
    tg = [x for x in srt if x.is_target][:q]
    rest = [x for x in srt if not x.is_target][:k - len(tg)]
    return sorted(tg + rest, key=lambda x: -x.score)


def score_boost(cands, k=K, lam=0.0, mode="add"):
    """实现 (3) 把保量约束融入融合分:对白名单内容加价后统一排 Top-K。
    mode='add': Score' = Score + lam;mode='mul': Score' = Score * (1 + lam)。"""
    def key(x):
        if not x.is_target:
            return -x.score
        return -(x.score + lam) if mode == "add" else -(x.score * (1.0 + lam))
    return sorted(cands, key=key)[:k]


def solve_lambda(cands, k=K, q=Q_TARGET, mode="add", lo=0.0, hi=2.0, iters=50):
    """二分出刚好满足 Exposure_target >= q 的最小 lambda(最小代价加价)。"""
    if sum(1 for x in plain_topk(cands, k) if x.is_target) >= q:
        return 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        n_t = sum(1 for x in score_boost(cands, k, mid, mode) if x.is_target)
        if n_t >= q:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------- 评估

def stat(lst, base_sum=None):
    """返回 (sum Score, 白名单曝光数, 人均真实时长, 人均真实 ctr, 大盘代价)。
    大盘代价为 sum/base - 1,负值表示相对无约束 Top-K 的收益损失。"""
    s = sum(x.score for x in lst)
    n_t = sum(1 for x in lst if x.is_target)
    watch = sum(x.watch for x in lst) / max(1, len(lst))
    ctr = sum(x.ctr for x in lst) / max(1, len(lst))
    cost = 0.0 if base_sum is None else s / base_sum - 1.0
    return s, n_t, watch, ctr, cost


def squeezed_out(base, lst):
    """被保量挤出最终结果的普通(非白名单)内容条数。"""
    keep = set(x.iid for x in lst)
    return sum(1 for x in base if not x.is_target and x.iid not in keep)


def position_weight(rank):
    """[MOCK] 位置衰减:越靠前曝光价值越高(用于强插位置的曝光质量评估)。"""
    return 1.0 / (1.0 + 0.08 * rank)


def weighted_exposure(lst):
    return sum(position_weight(i) for i, x in enumerate(lst) if x.is_target)


# ---------------------------------------------------------------- 闭环 pacing

def traffic_curve(t):
    """[MOCK] 一天 24 小时的请求量曲线(凌晨低、午晚高峰)。"""
    base = [0.3, 0.2, 0.15, 0.1, 0.1, 0.15, 0.4, 0.7, 0.9, 0.85, 0.8, 0.95,
            1.0, 0.9, 0.8, 0.8, 0.85, 0.95, 1.2, 1.4, 1.5, 1.3, 0.9, 0.5]
    return base[t % 24]


def run_pacing(strategy, daily_target, eta=0.30, lam_max=0.60, seed=7):
    """按 24 个时间片模拟整天投放,返回 (完成量, 保量后分数, 基线分数, 轨迹)。

    uniform: 每片按请求量占比均摊保量名额(固定预留坑位);
    asap   : 前置抢量,只要日目标未完成就把坑位开到最大;
    closed : 闭环动态 lambda,按「应完成进度 - 实际完成进度」调加价。
    """
    rng = random.Random(seed)
    total_req = sum(traffic_curve(t) for t in range(24))
    done, lam = 0, 0.0
    sum_score = base_score = 0.0
    rows = []
    for t in range(24):
        req = traffic_curve(t)
        n_req = max(1, int(round(req * 6)))          # 该时间片的请求条数
        share = req / total_req
        need_pace = sum(traffic_curve(j) for j in range(t + 1)) / total_req

        if strategy == "uniform":
            q_slot = int(round(daily_target * share / n_req))
        elif strategy == "asap":
            q_slot = K if done < daily_target else 0
        else:  # closed-loop lambda
            real_pace = done / daily_target
            lam = min(lam_max, max(0.0, lam + eta * (need_pace - real_pace)))
            q_slot = None

        got = 0
        for _ in range(n_req):
            c = make_candidates(random.Random(rng.randrange(10 ** 6)))
            base = plain_topk(c)
            base_score += sum(x.score for x in base)
            if q_slot is None:
                lst = score_boost(c, K, lam, "add")
            elif q_slot <= 0:
                lst = base
            else:
                lst = reserved_slots(c, K, min(q_slot, K))
            got += sum(1 for x in lst if x.is_target)
            sum_score += sum(x.score for x in lst)
        done += got
        rows.append((t, n_req, need_pace, done / daily_target, lam, got))
    return done, sum_score, base_score, rows


# ---------------------------------------------------------------- 演示

if __name__ == "__main__":
    random.seed(42)
    rng = random.Random(42)
    cands = make_candidates(rng)
    n_pool = sum(1 for x in cands if x.is_target)

    print("=" * 108)
    print("精排保量与强插策略(正文 §6.5.5):max sum Score  s.t. "
          "Exposure_target >= Q_target")
    print(f"候选集 {N_CAND} 个,其中白名单(保量)内容 {n_pool} 个;"
          f"输出 Top-{K},保量目标 Q_target = {Q_TARGET}")
    print("=" * 108)

    base = plain_topk(cands)
    b_sum, b_t, b_w, b_c, _ = stat(base)

    # ---------------- 表 a:四种实现方式对比
    print("\n【表 a】正文四类保量实现方式的单刷对比(大盘代价为负表示收益损失)")
    print(f"{'实现方式':<34}{'白名单曝光':>12}{'达成率':>10}{'sum Score':>12}"
          f"{'大盘分数代价':>14}{'挤出普通内容':>14}{'人均真实时长':>14}")
    print("-" * 108)
    lam_add = solve_lambda(cands, K, Q_TARGET, "add")
    lam_mul = solve_lambda(cands, K, Q_TARGET, "mul")
    variants = [
        ("(0) 无保量 Top-K(模型最优)", base),
        ("(1) 排序后强插(置顶)", force_insert(cands, K, Q_TARGET, 0)),
        ("(2) 预留固定坑位", reserved_slots(cands, K, Q_TARGET)),
        (f"(3a) 加性加价 lambda={lam_add:.4f}",
         score_boost(cands, K, lam_add, "add")),
        (f"(3b) 乘性加价 lambda={lam_mul:.4f}",
         score_boost(cands, K, lam_mul, "mul")),
    ]
    for name, lst in variants:
        s, n_t, w, c, cost = stat(lst, b_sum)
        print(f"{name:<34}{n_t:>12}{n_t / Q_TARGET:>10.0%}{s:>12.4f}"
              f"{cost:>+14.2%}{squeezed_out(base, lst):>14}{w:>14.4f}")
    print("-" * 108)
    print("观察:无保量时白名单只自然拿到 2 条曝光(达成率 33%),三类实现都把它抬到 6")
    print("      条、达成率 100%,并且代价完全一样(sum Score 18.8866 -> 18.4812,")
    print("      -2.15%,挤出 4 条普通内容)—— 因为在单刷、Q_target 远小于 K 的场景下,")
    print("      「补齐缺口 + 从尾部挤出」就是该约束下的精确最优解,三种工程写法只是")
    print("      同一个解的不同实现。有意思的是人均真实时长反而从 12.7047 升到")
    print("      12.7669:白名单内容融合分低(均值 0.21 vs 0.26)但真实时长不差,")
    print("      正文说的「不会因为短期数据不足而被排序模型完全压制」在这里成立。")
    print("      三者的真正差别在可控性:lambda 是连续旋钮(加性 0.1084 / 乘性")
    print("      0.4791),能按投放进度平滑调节(见表 c),强插与固定坑位只能硬切。")

    # ---------------- 表 b:强插位置的影响
    print("\n【表 b】强插位置的影响:正文只说「直接插入」,但插在哪里决定代价")
    print(f"{'强插位置':<24}{'白名单曝光':>12}{'加权曝光价值':>14}"
          f"{'头部3坑平均融合分':>20}{'整页 sum Score':>16}{'人均真实时长':>14}")
    print("-" * 108)
    for label, pos in [("置顶(pos=0)", 0), ("第 3 位(pos=2)", 2),
                       ("第 10 位(pos=9)", 9), ("第 25 位(pos=24)", 24),
                       ("尾插(pos=K)", K)]:
        lst = force_insert(cands, K, Q_TARGET, pos)
        s, n_t, w, c, _ = stat(lst, b_sum)
        head3 = sum(x.score for x in lst[:3]) / 3.0
        print(f"{label:<24}{n_t:>12}{weighted_exposure(lst):>14.4f}"
              f"{head3:>20.4f}{s:>16.4f}{w:>14.4f}")
    print("-" * 108)
    print("观察:整页 sum Score(18.4812)与人均真实时长(12.7669)在五行里完全相同 ——")
    print("      整页收益只取决于「挤掉了谁」,与插入位置无关。真正变化的是白名单拿到")
    print("      的加权曝光价值:置顶 4.0147,尾插只剩 1.2842,相差 3.1 倍;同时置顶把")
    print("      头部 3 坑的平均融合分从 0.4902 压到 0.2349(-52%),用户第一眼看到的")
    print("      就是扶持内容。注意 pos>=9 以后头部 3 坑已经不受影响(都是 0.4902),")
    print("      所以工业上常用「不进前 N 坑」的强插位置约束:保量条数照拿,体验损失")
    print("      只付在中后段坑位上。")

    # ---------------- 表 c:一天 24 片的闭环 pacing
    DAILY = 600
    print(f"\n【表 c】24 时间片闭环投放仿真(日保量目标 {DAILY} 条白名单曝光)")
    print(f"{'投放策略':<30}{'实际完成':>10}{'完成率':>10}"
          f"{'大盘分数代价':>14}{'超投/欠投':>12}   {'说明'}")
    print("-" * 108)
    traces = {}
    for name, key, note in [("固定预留坑位(uniform)", "uniform", "按流量占比均摊名额"),
                            ("前置抢量(ASAP)", "asap", "开局打满,早早撞线"),
                            ("闭环动态 lambda", "closed", "按进度差调加价")]:
        done, s_sum, b_sum2, rows = run_pacing(key, DAILY)
        traces[key] = (rows, done, s_sum / b_sum2 - 1.0)
        print(f"{name:<30}{done:>10}{done / DAILY:>10.1%}"
              f"{s_sum / b_sum2 - 1.0:>+14.2%}{done - DAILY:>+12}   {note}")
    print("-" * 108)
    print("【表 c-2】闭环策略的进度跟踪(应完成进度 vs 实际完成进度 vs lambda)")
    print(f"{'时间片':>6}{'请求数':>8}{'应完成进度':>12}{'实际完成进度':>14}"
          f"{'lambda':>10}{'本片白名单曝光':>16}")
    print("-" * 108)
    for t, n_req, need, real, lam, got in traces["closed"][0]:
        if t % 3 == 0 or t == 23:
            print(f"{t:>6}{n_req:>8}{need:>12.1%}{real:>14.1%}"
                  f"{lam:>10.4f}{got:>16}")
    print("-" * 108)
    print("观察:三种投放策略的日完成率都 >= 100%,但代价结构完全不同。ASAP 在凌晨低")
    print("      流量时段就把坑位开到最大,结果整天严重超投(710 条 / +110 / 118.3%),")
    print("      大盘分数代价 -3.78%,是三者里最差的 —— 保量「早早撞线」并不是优点,")
    print("      超投部分的分数损失全是白付。闭环 lambda 完成 617 条(+17)、代价")
    print("      -1.58%;按流量占比均摊名额的 uniform 在本仿真里恰好最优(602 条、")
    print("      -1.42%),因为这里的流量曲线是已知且平稳的。表 c-2 显示闭环的 lambda")
    print("      确实在跟随进度差:t=0..6 进度落后(1.0% vs 1.7%)时从 0.0051 抬到")
    print("      0.0519,t=12..18 进度超前(48.8% vs 37.3%)时又回落到 0.0267,这就是")
    print("      正文说的「通过动态调整排序分数实现更加平滑的流量控制」;它的真正优势")
    print("      在于流量曲线不可预知时不需要事先拍死每片名额。")

    # ---------------- 表 d:保量目标与大盘代价的权衡曲线
    print("\n【表 d】Q_target 边际代价曲线:模型最优 vs 业务最优的权衡"
          "(§6.5.4 / §6.5.5)")
    print(f"{'Q_target':>10}{'可满足':>8}{'最小 lambda':>14}{'白名单曝光':>12}"
          f"{'sum Score':>12}{'大盘分数代价':>14}{'本档新增代价':>14}"
          f"{'人均真实时长':>14}")
    print("-" * 108)
    prev_cost = 0.0
    q_list = sorted(set([0, 2, 4, 6, 8, 12, n_pool, n_pool + 4]))
    for q in q_list:
        if q > n_pool:
            print(f"{q:>10}{'不可行':>8}{'—':>14}{'—':>12}"
                  f"{'—':>12}{'—':>14}{'—':>14}{'—':>14}")
            continue
        lam = solve_lambda(cands, K, q, "add")
        lst = score_boost(cands, K, lam, "add")
        s, n_t, w, c, cost = stat(lst, b_sum)
        print(f"{q:>10}{'是':>8}{lam:>14.4f}{n_t:>12}{s:>12.4f}"
              f"{cost:>+14.2%}{(cost - prev_cost):>+14.2%}{w:>14.4f}")
        prev_cost = cost
    print("-" * 108)
    print(f"观察:候选集里白名单只有 {n_pool} 个,Q_target 取 {n_pool + 4} 时约束直接不可行,")
    print("      这是保量在工程上第一个要拦住的错误配置。可行区间内代价单调")
    print("      加速:Q=2 完全免费(0.00%,因为模型自己就给了 2 条),Q=6 为 -2.15%,")
    print("      Q=12 为 -6.75%,Q=16(整池全曝光)已经 -12.07%,「本档新增代价」从")
    print("      -1.03% 一路涨到 -5.31% —— 后续被挤掉的不再是尾部内容而是中部内容。")
    print("      正文强调保量「本质上是在模型最优与业务最优之间寻找平衡」,这张边际")
    print("      代价表就是和业务方谈保量目标时该拿出来的那张表。")
    print("=" * 108)
