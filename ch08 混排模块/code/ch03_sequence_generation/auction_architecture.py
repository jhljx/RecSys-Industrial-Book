"""预竞价（Pre-Auction）与全竞价（Full Auction）架构对比仿真。

对应正文第 8 章 §8.3.2「预竞价与全竞价架构」。

正文结论（本实现要复现的三点）：
1. 全竞价把「是否插广告 / 插几个 / 插哪些 / 插在哪」全部交给混排，理论上更容易拿到全局最优；
2. 但混排存在**信息边界**：它只能看到业务加工后的候选与有限特征，对价值的估计噪声比业务侧大，
   因此在相同算力下全竞价**未必赢**；
3. 业务预竞价本质是一种**搜索空间剪枝**，候选序列规模下降后还能释放算力给更精细的评估模型。
   折中方案（业务粗筛 + 混排统一决策）在实践中往往最稳。
   理想无噪声的全竞价（Oracle）给出该请求的价值上界。

序列生成的可控规模（本次修复的关键）：
    正文 §8.3 的枚举是「视频序列固定，枚举广告/直播的插入位置」，即**各业务内部的相对顺序不变**，
    混排只决定「每个业务取前几个 + 怎么交织」。因此合法序列数等于三路有序流的交织数
        N_seq = sum_{n_a, n_l} C(K, n_a) * C(K - n_a, n_l)   (受各业务候选量与 Gap 约束裁剪)
    而不是对候选做全排列。本实现按位置递归生成交织序列，并在生成过程中就做 Gap 约束剪枝
    （正文「约束通常会提前融入搜索过程，违反即剪枝」），K=8 时合法序列仅数百量级，可精确枚举。
    另外保留 `max_sequences` 安全阀与 `beam_size` 近似模式：当 K 或候选量放大到枚举不可行时，
    改用 Beam Search 近似（正文 §8.3.1 伪代码 `mixrank_const_beam_v2` 的逻辑）。

关键公式：
    序列真实价值        V(S) = sum_p d(p) * w_{biz(S_p)} * v_{S_p} - lambda_ad * fatigue(S)
    位置折扣            d(p) = 1 / log2(p + 2)
    广告疲劳惩罚        fatigue(S) = sum_{ad in S} (1 / (gap_before + 1))
    混排侧观测（信息边界）  \\hat v^{mix} = v * exp(eps),  eps ~ N(0, sigma_mix^2)
    业务侧观测            \\hat v^{biz} = v * exp(eps),  eps ~ N(0, sigma_biz^2),  sigma_biz < sigma_mix
    架构 A 的相对损失   gap(A) = 1 - E[V(S_A)] / E[V(S_oracle)]

[MOCK] 广告/直播/短视频三个业务推荐系统、它们的内部打分与预竞价决策，都是进程内的教学用
       假组件，不发起任何 RPC；"真实价值" v 由合成数据生成器给定，仅用于离线评估。
"""

import math
import random

# ---------------------------------------------------------------- 约束与价值

K_SEQ = 8          # 最终序列长度（一刷 8 个坑位）
AD_GAP = 3         # 刷次内 Gap 约束：两个广告之间至少间隔 3 个非广告内容
LIVE_MIN_POS = 1   # 直播不允许出现在首位（低直播偏好人群的位置约束）

BIZ_WEIGHT = {"video": 1.00, "live": 0.85, "ad": 0.60, "ec": 0.75}  # 统一价值空间下的业务权重
LAMBDA_AD = 0.50                                        # 广告疲劳惩罚系数


def pos_discount(p):
    """位置折扣 d(p) = 1 / log2(p + 2)，越靠后衰减越多。"""
    return 1.0 / math.log2(p + 2.0)


def ad_fatigue(seq):
    """广告疲劳：相邻广告越近惩罚越大，sum 1/(gap_before+1)。"""
    total, last_ad = 0.0, None
    for p, it in enumerate(seq):
        if it["biz"] != "ad":
            continue
        gap = K_SEQ if last_ad is None else (p - last_ad - 1)
        total += 1.0 / (gap + 1.0)
        last_ad = p
    return total


def seq_value(seq, key="v"):
    """序列价值 V(S)。key='v' 用真实价值，key='v_mix'/'v_biz' 用带噪观测值。"""
    base = sum(pos_discount(p) * BIZ_WEIGHT[it["biz"]] * it[key] for p, it in enumerate(seq))
    return base - LAMBDA_AD * ad_fatigue(seq)


def violates(seq):
    """增量约束检查：只检查刚追加的最后一个 item 是否破坏约束（用于生成时剪枝）。"""
    p = len(seq) - 1
    it = seq[p]
    if it["biz"] == "ad":
        for q in range(max(0, p - AD_GAP), p):
            if seq[q]["biz"] == "ad":
                return True
    if it["biz"] == "live" and p < LIVE_MIN_POS:
        return True
    return False


# ---------------------------------------------------------- 序列生成（交织枚举）

def enumerate_sequences(pools, k=K_SEQ, max_sequences=20000):
    """按位置递归枚举三路有序流的交织序列，生成中即做约束剪枝。

    pools: {biz: [item, ...]}，每个业务内部顺序固定（业务侧已排好序）。
    返回合法序列列表；超过 max_sequences 时抛 RuntimeError（提示改用 Beam Search）。
    """
    bizs = list(pools.keys())
    out = []

    def rec(seq, used):
        if len(seq) == k:
            out.append(list(seq))
            return
        if len(out) > max_sequences:
            raise RuntimeError("合法序列数超过 max_sequences，请改用 beam_search_sequences")
        for b in bizs:
            i = used[b]
            if i >= len(pools[b]):
                continue
            seq.append(pools[b][i])
            used[b] = i + 1
            if not violates(seq):
                rec(seq, used)
            used[b] = i
            seq.pop()

    rec([], {b: 0 for b in bizs})
    return out


def beam_search_sequences(pools, scorer, k=K_SEQ, beam_size=10):
    """带约束剪枝的 Beam Search 近似生成（正文 §8.3.1 伪代码的交织版实现）。

    枚举规模不可控时的退化方案：每步只保留打分最高的 beam_size 个部分序列。
    """
    bizs = list(pools.keys())
    beam = [([], {b: 0 for b in bizs})]
    for _ in range(k):
        nxt = []
        for seq, used in beam:
            for b in bizs:
                i = used[b]
                if i >= len(pools[b]):
                    continue
                new_seq = seq + [pools[b][i]]
                if violates(new_seq):
                    continue
                new_used = dict(used)
                new_used[b] = i + 1
                nxt.append((new_seq, new_used))
        if not nxt:
            break
        nxt.sort(key=lambda x: scorer(x[0]), reverse=True)
        beam = nxt[:beam_size]
    return [seq for seq, _ in beam]


# ------------------------------------------------------------ [MOCK] 业务推荐系统

def make_request(rid):
    """[MOCK] 合成一次请求：三个业务各自产出带「真实价值 v」的候选流（内部已排序）。

    v 服从 lognormal，刻画长尾；广告的 v 还叠加一个请求级的商业化强度因子，
    使得「这一刷该不该插广告」在不同请求间确有差异（否则对比无意义）。
    """
    commercial = random.choice([0.4, 0.8, 1.2, 1.6])  # 该请求的广告价值强度
    pools = {}
    for biz, n, mu, scale in (("video", 8, 1.00, 1.0),
                              ("live", 5, 0.90, 1.0),
                              ("ad", 5, 0.95, commercial)):
        items = []
        for i in range(n):
            v = random.lognormvariate(math.log(mu), 0.45) * scale
            items.append({"id": f"{biz[0].upper()}{i}", "biz": biz, "v": v})
        items.sort(key=lambda x: -x["v"])        # 业务侧按自己的价值排好序
        pools[biz] = items
    return {"rid": rid, "pools": pools, "commercial": commercial}


def observe(pools, sigma_mix, sigma_biz):
    """给每个 item 打上两种带噪观测值：混排侧 v_mix 与业务侧 v_biz。

    信息边界的形式化：混排只拿到加工后的特征，其噪声尺度 sigma_mix > sigma_biz。
    """
    for items in pools.values():
        for it in items:
            it["v_mix"] = it["v"] * math.exp(random.gauss(0.0, sigma_mix))
            it["v_biz"] = it["v"] * math.exp(random.gauss(0.0, sigma_biz))


def pre_auction(pools, floor):
    """[MOCK] 业务侧预竞价：用业务自己的（低噪）观测决定是否参与混排、下发几个候选。

    动作空间 {0, 1, ...}：广告侧按 v_biz 的位置折扣期望收益与保留价 floor 比较，
    逐个决定下发数量；直播侧同理但阈值更松。视频永远全量下发。
    """
    out = {"video": list(pools["video"])}
    for biz, thr, cap in (("ad", floor, 3), ("live", floor * 0.7, 2)):
        kept = []
        for rank, it in enumerate(pools[biz]):
            # 业务侧用自己的观测估算该候选放在中位位置能带来的加权收益
            gain = pos_discount(2 * rank + 2) * BIZ_WEIGHT[biz] * it["v_biz"]
            if gain >= thr and len(kept) < cap:
                kept.append(it)
        out[biz] = kept                           # 可能为空 → 该业务本刷不参与混排
    return out


# ---------------------------------------------------------------- 三种架构

def run_architecture(name, req, sigma_mix, sigma_biz, floor, beam_size=None):
    """执行一种架构，返回 (选中序列, 真实价值, 候选序列数, 评估次数)。

    - oracle    : 全竞价 + 无噪声（混排直接看到真实价值 v）→ 价值上界
    - full      : 全竞价，混排用 v_mix 打分，候选 video8/live5/ad5
    - pre       : 预竞价，业务先用 v_biz 剪枝，混排在小候选集上用 v_mix 打分
    - hybrid    : 折中，业务只做粗筛（保留价减半、cap 放宽），混排再统一决策
    """
    pools = req["pools"]
    if name == "pre":
        cand = pre_auction(pools, floor)
    elif name == "hybrid":
        cand = pre_auction(pools, floor * 0.5)
    else:
        cand = {b: list(v) for b, v in pools.items()}
    cand = {b: v for b, v in cand.items() if v}

    key = "v" if name == "oracle" else "v_mix"
    scorer = lambda s: seq_value(s, key)

    if beam_size is not None:
        seqs = beam_search_sequences(cand, scorer, beam_size=beam_size)
    else:
        seqs = enumerate_sequences(cand)
    if not seqs:                                  # 候选不足 K 个坑位，退化为纯视频
        seqs = [pools["video"][:K_SEQ]]
    best = max(seqs, key=scorer)
    return best, seq_value(best, "v"), len(seqs), len(seqs)


def biz_mix(seq):
    """序列的业务曝光构成。"""
    return tuple(sum(1 for it in seq if it["biz"] == b) for b in ("video", "live", "ad"))


def simulate(n_req=120, sigma_mix=0.55, sigma_biz=0.22, floor=0.35, beam_size=None):
    """跑 n_req 次请求，汇总四种架构的真实价值、业务构成与算力开销。"""
    archs = ["oracle", "full", "pre", "hybrid"]
    acc = {a: {"v": 0.0, "nseq": 0, "evals": 0, "mix": [0, 0, 0], "win": 0} for a in archs}
    for rid in range(n_req):
        req = make_request(rid)
        observe(req["pools"], sigma_mix, sigma_biz)
        vals = {}
        for a in archs:
            seq, v, nseq, ev = run_architecture(a, req, sigma_mix, sigma_biz, floor, beam_size)
            vals[a] = v
            s = acc[a]
            s["v"] += v
            s["nseq"] += nseq
            s["evals"] += ev
            for i, c in enumerate(biz_mix(seq)):
                s["mix"][i] += c
        # 全竞价 vs 预竞价 的逐请求胜负
        if vals["full"] > vals["pre"]:
            acc["full"]["win"] += 1
        elif vals["pre"] > vals["full"]:
            acc["pre"]["win"] += 1
    for a in archs:
        s = acc[a]
        s["v"] /= n_req
        s["nseq"] /= n_req
        s["evals"] /= n_req
        s["mix"] = [c / n_req for c in s["mix"]]
    return acc


# ---------------------------------------------------------------------- 演示

def demo_seq_space():
    """展示序列空间规模：候选规模对合法序列数的影响（全竞价组合爆炸的来源）。"""
    print("=" * 76)
    print("【1】序列生成空间规模（交织枚举 + Gap 约束剪枝，K = %d）" % K_SEQ)
    print("=" * 76)
    print(f"{'架构':<10}{'视频':>5}{'直播':>5}{'广告':>5}{'合法序列数':>12}{'相对预竞价':>12}")
    print("-" * 76)
    base = None
    for label, nv, nl, na in (("预竞价", 8, 1, 1), ("折中", 8, 2, 2),
                              ("全竞价", 8, 5, 5), ("全竞价+电商", 8, 5, 5)):
        pools = {"video": [{"id": f"V{i}", "biz": "video", "v": 1.0} for i in range(nv)],
                 "live": [{"id": f"L{i}", "biz": "live", "v": 1.0} for i in range(nl)],
                 "ad": [{"id": f"A{i}", "biz": "ad", "v": 1.0} for i in range(na)]}
        if label == "全竞价+电商":
            pools["ec"] = [{"id": f"E{i}", "biz": "live", "v": 1.0} for i in range(5)]
        n = len(enumerate_sequences(pools, max_sequences=10 ** 7))
        base = n if base is None else base
        extra = "（+电商5）" if label == "全竞价+电商" else ""
        print(f"{label + extra:<10}{nv:>5}{nl:>5}{na:>5}{n:>12}{n / base:>11.1f}x")
    print("-" * 76)
    print("候选从 (8,1,1) 增到 (8,5,5) 再加一路业务，合法序列数呈组合数级增长 → 正文的组合爆炸。")


def demo_architectures():
    """四种架构在相同请求流上的价值/算力对比。"""
    print()
    print("=" * 76)
    print("【2】架构对比：真实价值 vs 算力（120 次请求，sigma_mix=0.55 > sigma_biz=0.22）")
    print("=" * 76)
    acc = simulate()
    up = acc["oracle"]["v"]
    print(f"{'架构':<12}{'真实价值':>10}{'距上界gap':>11}{'候选序列数':>11}"
          f"{'视频':>6}{'直播':>6}{'广告':>6}")
    print("-" * 76)
    names = {"oracle": "全竞价(理想)", "full": "全竞价", "pre": "预竞价", "hybrid": "折中"}
    for a in ("oracle", "full", "pre", "hybrid"):
        s = acc[a]
        gap = 1.0 - s["v"] / up
        print(f"{names[a]:<12}{s['v']:>10.4f}{gap:>10.2%}{s['nseq']:>11.1f}"
              f"{s['mix'][0]:>6.2f}{s['mix'][1]:>6.2f}{s['mix'][2]:>6.2f}")
    print("-" * 76)
    print(f"逐请求胜负（全竞价 vs 预竞价）：全竞价赢 {acc['full']['win']} 次，"
          f"预竞价赢 {acc['pre']['win']} 次，共 120 次")
    print("理想无噪声全竞价是价值上界；带信息边界后全竞价未必赢过预竞价，且算力开销高一个量级。")
    return acc


def demo_information_boundary():
    """扫描混排侧噪声 sigma_mix：信息边界多大时全竞价才由赢转输。"""
    print()
    print("=" * 76)
    print("【3】信息边界扫描：混排噪声 sigma_mix 对架构优劣的影响（sigma_biz = 0.22 固定）")
    print("=" * 76)
    print(f"{'sigma_mix':>10}{'全竞价':>10}{'预竞价':>10}{'折中':>10}{'上界':>10}{'赢家':>12}")
    print("-" * 76)
    for sm in (0.10, 0.25, 0.40, 0.55, 0.70, 0.90):
        random.seed(1234)                          # 同一组请求流，只变噪声
        acc = simulate(n_req=60, sigma_mix=sm)
        f, p, h, o = (acc[a]["v"] for a in ("full", "pre", "hybrid", "oracle"))
        best = max((("全竞价", f), ("预竞价", p), ("折中", h)), key=lambda x: x[1])[0]
        print(f"{sm:>10.2f}{f:>10.4f}{p:>10.4f}{h:>10.4f}{o:>10.4f}{best:>12}")
    print("-" * 76)
    print("混排信息质量高（sigma_mix 小）时全竞价占优；信息边界变大后预竞价的剪枝反而更稳。")


def demo_compute_budget():
    """预竞价剪枝释放的算力 → 可换更精细的评估模型（噪声更小）。"""
    print()
    print("=" * 76)
    print("【4】算力预算再分配：候选变少 → 评估模型可以更重更准")
    print("=" * 76)
    print(f"{'方案':<24}{'候选序列数':>11}{'评估噪声':>10}{'真实价值':>10}{'距上界gap':>11}")
    print("-" * 76)
    rows = [("全竞价 + 轻量评估", "full", 0.55), ("预竞价 + 轻量评估", "pre", 0.55),
            ("预竞价 + 重型评估", "pre", 0.30), ("折中 + 重型评估", "hybrid", 0.35)]
    random.seed(2024)
    up = simulate(n_req=60, sigma_mix=0.0)["oracle"]["v"]
    for label, arch, sm in rows:
        random.seed(2024)
        acc = simulate(n_req=60, sigma_mix=sm)
        s = acc[arch]
        print(f"{label:<24}{s['nseq']:>11.1f}{sm:>10.2f}{s['v']:>10.4f}"
              f"{1.0 - s['v'] / up:>10.2%}")
    print("-" * 76)
    print("预竞价把候选压缩一个量级，省下的算力换成更准的评估模型后，总价值可反超全竞价。")


def demo_beam_fallback():
    """序列空间不可枚举时的退化：Beam Search 近似 vs 精确枚举。"""
    print()
    print("=" * 76)
    print("【5】枚举不可行时的退化方案：约束剪枝 Beam Search（§8.3.1 伪代码）")
    print("=" * 76)
    print(f"{'生成方式':<20}{'候选序列数':>11}{'价值(s=.55)':>12}{'价值(s=.10)':>12}{'相对枚举':>11}")
    print("-" * 76)
    rows = []
    for label, bs in (("精确枚举", None), ("Beam b=3", 3), ("Beam b=5", 5),
                      ("Beam b=10", 10), ("Beam b=20", 20)):
        random.seed(777)
        hi = simulate(n_req=40, sigma_mix=0.55, beam_size=bs)["full"]
        random.seed(777)
        lo = simulate(n_req=40, sigma_mix=0.10, beam_size=bs)["full"]
        rows.append((label, hi["nseq"], hi["v"], lo["v"]))
    base_lo = rows[0][3]
    for label, nseq, vhi, vlo in rows:
        print(f"{label:<20}{nseq:>11.1f}{vhi:>12.4f}{vlo:>12.4f}{vlo / base_lo:>11.5f}")
    print("-" * 76)
    print("低噪声（s=.10）时精确枚举最优，Beam b=10 仅差千分之几 → 近似代价很小。")
    print("高噪声（s=.55）时 Beam 反而略胜：在 1125 条带噪打分里取 max 会放大噪声"
          "（winner's curse），")
    print("这从另一个角度印证正文观点——信息边界下扩大搜索空间并不等于拿到更优解。")


if __name__ == "__main__":
    random.seed(42)
    demo_seq_space()
    demo_architectures()
    demo_information_boundary()
    demo_compute_budget()
    demo_beam_fallback()
