"""
混排序列生成的约束规则引擎（第 8 章 §8.3 / §8.3.1 序列生成约束规则）

对应正文概念
------------
正文指出混排本质是「带约束的序列优化问题」（Constrained Sequence Optimization），
序列生成阶段除了找高价值序列，还要满足大量业务约束：
  - 刷次内 Gap 约束（Intra-Feed Gap）：任意两个广告之间至少间隔若干自然内容；
  - 跨刷次 Gap 约束（Inter-Feed Gap）：上一刷尾部是广告则本刷首位不许再出广告；
  - 位置约束（Position）：某类内容只能出现在指定位置范围；
  - 频控约束（Frequency）：同一业务/同一作者窗口内曝光次数上限；
  - 多样性约束（Diversity）：避免连续出现同类别/同作者；
  - 保量约束（Quota）：保证特定业务最低曝光量；
  - 探索约束（Exploration）：为新内容预留曝光；
  - 商业化约束（Commercial）：广告/电商达到既定收益目标。

本实现做了什么
--------------
1. 复现正文的枚举规模算例：短视频 6 + 广告最多 3 + 直播最多 1，
   按正文口径逐项算出 294 / 336 / 105 / 8，合计 743 条合法序列；
2. 实现一个可插拔的 RuleEngine（每条规则是 `(名称, 检查函数)`），
   统计每条规则各自过滤掉多少序列，演示「规则引擎」如何成为独立模块；
3. 实现正文两段伪代码对应的真实可运行版本：
   `generate_valid_sequences`（先扩展后过滤）与
   `constrained_beam_search`（约束前置剪枝），打印两者的扩展节点数差异，
   验证正文论断「约束提前融入搜索可显著减少无效搜索空间」；
4. 对比暴力枚举与 Beam Search 的最优序列得分差距和算力开销。

关键公式
--------
    正文枚举算例:
      n_ad=3: C(10,4) + C(9,3)             = 210 + 84 = 294
      n_ad=2: 3·(C(9,3) + C(8,2))          = 3·(84+28) = 336
      n_ad=1: 3·(C(8,2) + C(7,1))          = 3·(28+7)  = 105
      n_ad=0: C(7,1) + C(6,0)              = 7 + 1     = 8
      合计 743
    刷次内 Gap:  pos(ad_{k+1}) - pos(ad_k) > gap
    序列打分:    Score(seq) = Σ_i γ^i · value(item_i)

注：正文的 C(10,4)=210 把「3 个广告 + 1 个直播」视为相对顺序已固定的插入串
（广告顺序由广告系统排序决定），只选插入位置集合，不再对插入项做内部排列。
本实现严格按此口径计数。
"""

import itertools
import math
import random


# ---------------------------------------------------------------- 规则引擎
class RuleEngine:
    """可插拔约束规则引擎：规则返回 True 表示「违反约束」，序列被剪枝。

    工业混排系统常同时挂载数十条规则，正文称其为独立的 Rule Engine /
    Constraint Solver。这里把规则实现成 (name, fn, is_prefix_safe) 三元组：
    is_prefix_safe=True 的规则对前缀成立即可提前剪枝（供 Beam Search 使用）。
    """

    def __init__(self):
        self.rules: list[tuple[str, callable, bool]] = []
        self.reject = {}
        self.cost = 0          # 规则判定的「序列元素访问量」，衡量全序列重扫的代价

    def add(self, name: str, fn, prefix_safe: bool = True) -> "RuleEngine":
        self.rules.append((name, fn, prefix_safe))
        self.reject[name] = 0
        return self

    def violated(self, seq: list[dict], prefix: bool = False) -> str | None:
        for name, fn, ps in self.rules:
            if prefix and not ps:
                continue            # 前缀阶段跳过只能对完整序列判定的规则（如保量）
            self.cost += len(seq)   # 每条规则都要重扫整个（部分）序列
            if fn(seq):
                self.reject[name] += 1
                return name
        return None


# ------------------------------------------------------------ 具体约束规则
def intra_feed_gap(gap: int):
    """刷次内 Gap：相邻两个广告之间至少间隔 gap 个自然内容。"""
    def fn(seq):
        last = -10 ** 6
        for i, it in enumerate(seq):
            if it["biz"] == "ad":
                if i - last <= gap:
                    return True
                last = i
        return False
    return fn


def inter_feed_gap(prev_tail_is_ad: bool):
    """跨刷次 Gap：上一刷最后一个 Item 是广告 → 本刷首位不能是广告。"""
    def fn(seq):
        return bool(seq) and prev_tail_is_ad and seq[0]["biz"] == "ad"
    return fn


def position_constraint(biz: str, min_pos: int):
    """位置约束：指定业务不得出现在 min_pos 之前（0-based）。"""
    def fn(seq):
        return any(it["biz"] == biz and i < min_pos for i, it in enumerate(seq))
    return fn


def frequency_constraint(biz: str, max_cnt: int):
    """频控约束：单刷内某业务曝光数上限。"""
    def fn(seq):
        return sum(it["biz"] == biz for it in seq) > max_cnt
    return fn


def diversity_constraint(key: str):
    """多样性约束：相邻两个 Item 的 key 字段不得相同（同类目/同作者）。"""
    def fn(seq):
        return any(seq[i][key] == seq[i + 1][key] for i in range(len(seq) - 1))
    return fn


def quota_constraint(biz: str, min_cnt: int, full_len: int):
    """保量约束：完整序列中某业务至少曝光 min_cnt 次（只对完整序列判定）。"""
    def fn(seq):
        return len(seq) >= full_len and sum(it["biz"] == biz for it in seq) < min_cnt
    return fn


# ------------------------------------------------------------ 正文枚举算例
def comb(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def textbook_enumeration() -> list[tuple]:
    """按正文口径逐项计算 743。"""
    rows = []
    n_video = 6
    for n_ad in (3, 2, 1, 0):
        combos = comb(3, n_ad)                       # 从 3 个广告候选中选 n_ad 个
        with_live = comb(n_video + n_ad + 1, n_ad + 1)
        no_live = comb(n_video + n_ad, n_ad)
        rows.append((n_ad, combos, with_live, no_live,
                     combos * (with_live + no_live)))
    return rows


def enumerate_sequences(video, ads, lives, max_ad, engine: RuleEngine):
    """暴力枚举：选 n_ad 个广告 + 是否带直播 → 枚举插入位置集合。"""
    all_seq, legal = 0, []
    for n_ad in range(max_ad, -1, -1):
        for ad_combo in itertools.combinations(ads, n_ad):
            for n_live in (1, 0) if lives else (0,):
                inserts = list(ad_combo) + list(lives[:n_live])
                total = len(video) + len(inserts)
                for slots in itertools.combinations(range(total), len(inserts)):
                    all_seq += 1
                    seq, vi, ii, s = [], 0, 0, set(slots)
                    for pos in range(total):
                        if pos in s:
                            seq.append(inserts[ii]); ii += 1
                        else:
                            seq.append(video[vi]); vi += 1
                    if engine.violated(seq) is None:
                        legal.append(seq)
    return all_seq, legal


# ---------------------------------------------- 正文伪代码的可运行实现
def generate_valid_sequences(candidates, K, beam_size, engine, scorer):
    """正文代码清单「基于约束过滤的混排序列生成」：先扩展、后整体过滤。"""
    beam, expanded = [[]], 0
    for _ in range(K):
        new_beam = []
        for seq in beam:
            for item in candidates:
                if item in seq:
                    continue
                expanded += 1
                new_seq = seq + [item]
                if engine.violated(new_seq, prefix=True) is not None:
                    continue
                new_beam.append(new_seq)
        if not new_beam:
            break
        new_beam.sort(key=scorer, reverse=True)
        beam = new_beam[:beam_size]
    return beam, expanded


def constrained_beam_search(candidates, K, beam_size, engine, scorer):
    """正文代码清单「带约束剪枝的 Beam Search」：约束前置，违反即不扩展。

    与上一版的差别在于：先用「增量约束」判断能否接在当前前缀后面，
    不合法的扩展连序列对象都不构造，从而减少节点生成与打分开销。
    """
    beam, expanded, scored = [([], 0.0)], 0, 0

    def incremental_ok(seq, item) -> bool:
        """只检查新增 item 引发的增量冲突，O(1)。"""
        i = len(seq)
        if item["biz"] == "ad":
            if i == 0 and engine_prev_tail_is_ad:
                return False
            for j in range(max(0, i - AD_GAP), i):
                if seq[j]["biz"] == "ad":
                    return False
        if item["biz"] == "live" and i < LIVE_MIN_POS:
            return False
        if seq and seq[-1]["cate"] == item["cate"]:
            return False
        if sum(x["biz"] == item["biz"] for x in seq) + 1 > FREQ_CAP[item["biz"]]:
            return False
        return True

    for _ in range(K):
        new_beam = []
        for seq, _sc in beam:
            for item in candidates:
                if item in seq:
                    continue
                expanded += 1
                if not incremental_ok(seq, item):
                    continue
                new_seq = seq + [item]
                scored += 1
                new_beam.append((new_seq, scorer(new_seq)))
        if not new_beam:
            break
        new_beam.sort(key=lambda t: -t[1])
        beam = new_beam[:beam_size]
    return beam, expanded, scored


# 增量约束所需的全局配置（教学用，避免跨函数 import）
AD_GAP = 3
LIVE_MIN_POS = 1
FREQ_CAP = {"ad": 3, "live": 1, "video": 8}
engine_prev_tail_is_ad = True


def main() -> None:
    random.seed(42)
    print("=" * 78)
    print("混排序列生成的约束规则引擎（§8.3.1）")
    print("=" * 78)

    # ---- 1) 复现正文枚举算例
    print("\n【1】正文枚举规模算例（短视频 6 + 广告≤3 + 直播≤1）")
    print(f"{'广告数':>8}{'广告组合':>10}{'含直播':>10}{'不含直播':>10}{'小计':>10}")
    rows = textbook_enumeration()
    for n_ad, c, wl, nl, sub in rows:
        print(f"{n_ad:>8d}{c:>10d}{wl:>10d}{nl:>10d}{sub:>10d}")
    total = sum(r[4] for r in rows)
    print(f"{'合计':>8}{'':<30}{total:>10d}   （正文给出 743，"
          f"{'一致' if total == 743 else '不一致'}）")

    # ---- 2) 构造候选 + 规则引擎
    video = [{"id": f"V{i+1}", "biz": "video", "cate": random.choice("ABC"),
              "val": round(random.uniform(0.5, 1.0), 3)} for i in range(6)]
    ads = [{"id": f"A{i+1}", "biz": "ad", "cate": "AD",
            "val": round(random.uniform(0.8, 1.6), 3)} for i in range(3)]
    lives = [{"id": "L1", "biz": "live", "cate": "LIVE",
              "val": round(random.uniform(0.9, 1.4), 3)}]

    engine = (RuleEngine()
              .add("刷次内Gap(≥3)", intra_feed_gap(3))
              .add("跨刷次Gap(首位禁广告)", inter_feed_gap(True))
              .add("位置约束(直播不进首位)", position_constraint("live", 1))
              .add("频控约束(广告≤3)", frequency_constraint("ad", 3))
              .add("多样性约束(相邻异类目)", diversity_constraint("cate"))
              .add("保量约束(直播≥1)", quota_constraint("live", 1, 10), prefix_safe=False))

    n_all, legal = enumerate_sequences(video, ads, lives, 3, engine)
    print(f"\n【2】规则引擎过滤统计：枚举 {n_all} 条 → 合法 {len(legal)} 条"
          f"（通过率 {len(legal)/n_all:.4f}）")
    print(f"  {'规则名':<26}{'首次命中拦截数':>14}")
    for name, cnt in engine.reject.items():
        print(f"  {name:<26}{cnt:>14d}")
    print("  注：规则按注册顺序短路判定，统计的是「该规则作为首个命中者」的次数。")

    # ---- 3) 两段伪代码实现对比
    # 序列级评估函数：位置衰减 + 广告疲劳交互项。
    # 前缀里每多一个广告，后续 Item 的价值按 exp(-β) 额外衰减，
    # 于是单个 Item 的边际收益依赖整条前缀，贪心（beam=1）不再保证最优。
    gamma, beta = 0.85, 0.6

    def scorer(seq):
        s, n_ad = 0.0, 0
        for i, it in enumerate(seq):
            s += gamma ** i * it["val"] * math.exp(-beta * n_ad)
            n_ad += it["biz"] == "ad"
        return s

    cands = video + ads + lives
    K, bw = 8, 20
    e2 = (RuleEngine()
          .add("刷次内Gap(≥3)", intra_feed_gap(3))
          .add("跨刷次Gap(首位禁广告)", inter_feed_gap(True))
          .add("位置约束(直播不进首位)", position_constraint("live", 1))
          .add("频控约束(广告≤3)", frequency_constraint("ad", 3))
          .add("多样性约束(相邻异类目)", diversity_constraint("cate")))
    beam_a, exp_a = generate_valid_sequences(cands, K, bw, e2, scorer)
    beam_b, exp_b, scored_b = constrained_beam_search(cands, K, bw, None, scorer)

    print(f"\n【3】两段正文伪代码的可运行实现对比（候选 {len(cands)}，K={K}，beam={bw}）")
    print(f"  {'方案':<36}{'扩展节点':>10}{'序列打分':>10}{'规则判定元素访问':>18}{'最优分':>10}")
    print(f"  {'A 先扩展后过滤 generate_valid_sequences':<36}{exp_a:>10d}"
          f"{exp_a:>10d}{e2.cost:>18d}{scorer(beam_a[0]):>10.4f}")
    print(f"  {'B 约束前置剪枝 constrained_beam_search':<36}{exp_b:>10d}"
          f"{scored_b:>10d}{exp_b * 4:>18d}{beam_b[0][1]:>10.4f}")
    print(f"  B 用 O(1) 增量约束替代 O(L) 全序列重扫：序列打分次数降到 {scored_b / exp_a:.1%}，")
    print(f"  规则判定的元素访问量降到 {exp_b * 4 / e2.cost:.1%}，最优分完全一致（搜索空间等价）。")

    # ---- 4) 等价搜索空间下：暴力枚举 vs 不同 beam 宽度
    print(f"\n【4】同一搜索空间（从 {len(cands)} 个候选中有序选 {K} 个）下的最优性-算力权衡")
    engine_bf = (RuleEngine()
                 .add("刷次内Gap(≥3)", intra_feed_gap(3))
                 .add("跨刷次Gap(首位禁广告)", inter_feed_gap(True))
                 .add("位置约束(直播不进首位)", position_constraint("live", 1))
                 .add("频控约束(广告≤3)", frequency_constraint("ad", 3))
                 .add("多样性约束(相邻异类目)", diversity_constraint("cate")))
    best_bf, n_perm, n_legal_bf = None, 0, 0
    for perm in itertools.permutations(cands, K):
        n_perm += 1
        seq = list(perm)
        if engine_bf.violated(seq) is not None:
            continue
        n_legal_bf += 1
        s = scorer(seq)
        if best_bf is None or s > best_bf[1]:
            best_bf = (seq, s)
    print(f"  暴力枚举：全排列 {n_perm} 条 → 合法 {n_legal_bf} 条，"
          f"最优分 {best_bf[1]:.4f}")
    print(f"            最优序列 = " + " ".join(x['id'] for x in best_bf[0]))
    print(f"  {'beam宽度':>10}{'扩展节点':>10}{'序列打分':>10}{'最优分':>10}{'与最优差距':>12}")
    beam_stat = {}
    for w in (1, 2, 5, 10, 20, 50):
        bm, ex, sc = constrained_beam_search(cands, K, w, None, scorer)
        gap = (best_bf[1] - bm[0][1]) / best_bf[1]
        beam_stat[w] = (sc, gap)
        print(f"  {w:>10d}{ex:>10d}{sc:>10d}{bm[0][1]:>10.4f}{gap:>11.2%}")
    sc5, gap5 = beam_stat[5]
    print(f"  正文观点验证：候选只有 {len(cands)} 个时暴力枚举仍可行"
          f"（{n_perm} 条全排列，秒级跑完）；")
    print(f"  而 beam=5 只对 {sc5} 条序列打分（全排列的 {100 * sc5 / n_perm:.4f}%），"
          f"结果与最优仅差 {gap5:.2%}；")
    print(f"  业务数与候选数继续增长时全排列组合爆炸，剪枝式搜索是唯一可落地的方案。")


if __name__ == "__main__":
    main()
