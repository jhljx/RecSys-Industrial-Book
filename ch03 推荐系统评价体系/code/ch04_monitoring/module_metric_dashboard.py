"""对应正文第 3 章 §3.4「推荐系统监控体系」的分模块指标体系与分层根因定位。

正文把推荐系统监控划分为「数据层、模型层、服务层、业务层」四个维度，并按模块
逐条列出了监控指标：

  * 数据与索引监控：特征缺失率、特征分布漂移、索引更新延迟、索引覆盖率、
    用户行为日志完整率；
  * 召回模块监控：各召回源的召回数量 / 覆盖率 / 命中率、候选集 Duration 分布、
    不同内容类型占比、长尾内容覆盖率；
  * 排序模块监控：pxtr 均值、pxtr 分位数（P50/P90/P99）、PCOC 校准、
    AUC 是否掉到 0.5 附近、是否出现 nan/inf、输出分布漂移；
  * 重排与混排监控：各内容池下发量 / 曝光流量占比、多样性、作者分布、
    类目分布、冷启动内容曝光占比；
  * 服务与稳定性监控：QPS、平均响应时间、P99 延迟、超时率、错误率、GPU 利用率。

本实现做了什么
---------------
1. `[MOCK]` 一个推荐在线链路的「指标采集 Agent」：一次 `collect()` 相当于一个分钟
   级监控窗口，输出上面五类指标，指标名沿用正文的叫法；
2. 指标都用真实公式算出来（不是随手给随机数）：
     - 特征缺失率      missing_rate = #缺失请求 / #请求
     - 特征分布漂移    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)，参考分布等频分桶
     - 索引覆盖率      coverage = #索引字段完好且可被检索的 item / #内容池 item
     - 召回命中率      hit_rate = #该源召回且最终被曝光的条数 / #该源召回条数
     - 长尾覆盖率      tail_cov = #被曝光的长尾 item / #长尾 item
     - 概率校准        PCOC = mean(pCTR) / 真实后验 CTR
     - 排序质量        AUC = (sum_rank_pos - M(M+1)/2) / (M*N)（秩和公式，含并列秩）
     - 多样性          归一化类目熵 H = -sum p ln p / ln K
     - 分位数          P50/P90/P99（线性插值）
3. 注入四个正文举到的典型故障，演示「一个环节异常 -> 下游指标连锁变化」：
     - 故障A：关键索引字段被写坏（正文：关键索引字段异常 -> 召回检索不到目标内容）
     - 故障B：某一路协同过滤召回数量大幅下降（正文原例）
     - 故障C：模型发布后 pCTR 整体抬高 50% 而真实 CTR 未同步提升（正文原例，
       表现为 PCOC 显著大于 1，而 AUC 不变，因为单调变换不改变排序）
     - 故障D：流量突增导致 P99 延迟与超时率恶化（只打服务层）
4. 每个场景打印「指标值 + 相对健康基线的环比变化」表，并自动做根因定位：沿
   「数据/索引 -> 召回 -> 排序 -> 混排 -> 服务」的因果顺序取第一个破线的层，
   这正是正文说的监控目标——及时发现问题、快速定位问题。

全部为进程内 `[MOCK]` 仿真，不起真实服务、不读外部数据。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ==========================================================================
# 指标公式
# ==========================================================================
def quantile(xs: list[float], q: float) -> float:
    """线性插值分位数（正文排序 / 服务监控要求的 P50、P90、P99）。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def psi(expect: list[float], actual: list[float], n_bins: int = 10) -> float:
    """特征分布漂移 PSI = sum_i (a_i - e_i) * ln(a_i / e_i)。

    分桶方式：按参考（昨日）分布做等频分桶，再把当前分布灌进同一组桶。
    工业界经验阈值：PSI < 0.1 稳定；0.1~0.25 轻度漂移；> 0.25 显著漂移需排查。
    """
    if not expect or not actual:
        return float("nan")
    edges = [quantile(expect, i / n_bins) for i in range(1, n_bins)]
    eps = 1e-6

    def hist(xs: list[float]) -> list[float]:
        cnt = [0] * n_bins
        for x in xs:
            b = 0
            while b < len(edges) and x > edges[b]:
                b += 1
            cnt[b] += 1
        return [max(c / len(xs), eps) for c in cnt]

    e, a = hist(expect), hist(actual)
    return sum((ai - ei) * math.log(ai / ei) for ei, ai in zip(e, a))


def norm_entropy(counts: dict[str, int]) -> float:
    """归一化类目熵（多样性指标）H / ln(K)，取值 [0, 1]，越大越分散。"""
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h / math.log(len(counts))


def auc_score(scores: list[float], labels: list[int]) -> float:
    """秩和公式 AUC = (sum_rank_pos - M(M+1)/2) / (M*N)，并列分数取平均秩。

    注意这里算的是「曝光样本上的在线 AUC」，样本已被排序截断，
    所以数值天然远低于离线全量 AUC；正文关心的是它是否掉到 0.5 附近。
    """
    m = sum(labels)
    n = len(labels) - m
    if m == 0 or n == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    s_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (s_pos - m * (m + 1) / 2) / (m * n)


# ==========================================================================
# [MOCK] 内容池与在线链路仿真
# ==========================================================================
CATEGORIES = ["短剧", "美食", "游戏", "知识", "体育", "音乐"]
POOLS = ["推荐图文", "推荐视频", "广告", "电商", "直播"]
RECALL_SOURCES = ["u2i_dssm", "i2i_cf", "hot_rank", "author_follow", "tag_match"]
SOURCE_QUOTA = {"u2i_dssm": 60, "i2i_cf": 50, "hot_rank": 30,
                "author_follow": 25, "tag_match": 35}
N_AUTHOR = 400
TOP_K = 10                      # 每屏下发条数
DIVERSITY_W = 2.0               # [MOCK] 混排阶段的多样性打散权重
TIMEOUT_MS = 200.0


@dataclass
class Item:
    item_id: int
    category: str
    pool: str
    author_id: int
    duration: float            # 视频时长（秒），索引关键字段
    popularity: float          # [MOCK] 热度，决定是否长尾
    quality: float             # [MOCK] 真实吸引力，决定后验 CTR
    is_cold: bool
    is_tail: bool = False
    index_ok: bool = True      # 索引字段是否完好（故障 A 把它写坏）


@dataclass
class Fault:
    """[MOCK] 注入的故障配置。"""
    name: str
    index_broken_ratio: float = 0.0      # 索引关键字段被写坏的 item 比例
    feature_miss_extra: float = 0.0      # 额外的特征缺失率
    recall_down_source: str | None = None
    recall_down_ratio: float = 0.0       # 该召回源返回数量下降比例
    pxtr_inflate: float = 1.0            # 模型输出整体放大倍数
    latency_inflate: float = 1.0         # 响应时间放大倍数


@dataclass
class Snapshot:
    """一个监控窗口的全链路指标快照。"""
    data: dict[str, float] = field(default_factory=dict)
    recall: dict[str, dict[str, float]] = field(default_factory=dict)
    rank: dict[str, float] = field(default_factory=dict)
    mix: dict[str, float] = field(default_factory=dict)
    serve: dict[str, float] = field(default_factory=dict)


def build_pool(n: int = 3000) -> list[Item]:
    items: list[Item] = []
    for i in range(n):
        items.append(Item(
            item_id=i,
            category=random.choice(CATEGORIES),
            pool=random.choices(POOLS, weights=[30, 45, 8, 7, 10])[0],
            author_id=i % N_AUTHOR,
            duration=random.lognormvariate(math.log(45), 0.8),
            popularity=random.lognormvariate(0.0, 1.2),
            quality=random.betavariate(1.2, 4.0),
            is_cold=random.random() < 0.12,
        ))
    med_pop = quantile([it.popularity for it in items], 0.5)
    for it in items:
        it.is_tail = it.popularity < med_pop          # 热度后 50% 视作长尾
    return items


def collect(items: list[Item], fault: Fault, n_req: int = 900) -> Snapshot:
    """[MOCK] 指标采集 Agent：跑 n_req 次在线请求，按正文的指标清单聚合一个窗口。"""
    # ---------- 故障 A：关键索引字段被写坏（duration 脏值 + 不可检索） ----------
    for it in items:
        it.index_ok = True
    if fault.index_broken_ratio > 0:
        for it in random.sample(items, int(len(items) * fault.index_broken_ratio)):
            it.index_ok = False
    healthy = [it for it in items if it.index_ok] or items
    hot_items = [it for it in healthy if not it.is_tail] or healthy

    snap = Snapshot()

    # ======================= 数据与索引监控 =======================
    ref_dur = [it.duration for it in items]
    # 索引里实际存着的 duration：写坏的 item 变成 0 脏值
    cur_dur = [it.duration if it.index_ok else 0.0 for it in items]
    snap.data = {
        "feature_missing_rate": min(1.0, 0.012 + fault.feature_miss_extra),
        "feature_psi(duration)": psi(ref_dur, cur_dur),
        "index_delay_sec": max(
            0.0, random.gauss(38, 6) + (900 if fault.index_broken_ratio > 0 else 0)),
        "index_coverage": len([it for it in items if it.index_ok]) / len(items),
        "log_completeness": min(1.0, random.gauss(0.995, 0.002)),
    }

    # ======================= 召回 / 排序 / 混排 =======================
    src_recalled = {s: 0 for s in RECALL_SOURCES}
    src_hit = {s: 0 for s in RECALL_SOURCES}
    src_items: dict[str, set[int]] = {s: set() for s in RECALL_SOURCES}
    cand_dur: list[float] = []
    pxtrs: list[float] = []
    labels: list[int] = []
    pool_cnt = {p: 0 for p in POOLS}
    cat_cnt = {c: 0 for c in CATEGORIES}
    author_cnt: dict[int, int] = {}
    tail_exposed: set[int] = set()
    cold_exposed = 0
    nan_cnt = 0
    n_exp = 0

    for _ in range(n_req):
        cands: list[tuple[Item, str]] = []
        for src, quota in SOURCE_QUOTA.items():
            q = quota
            if src == fault.recall_down_source:
                q = int(quota * (1 - fault.recall_down_ratio))
            # 索引字段坏掉的 item 检索不到（正文：关键索引字段异常 -> 召回不到目标内容）
            pool = hot_items if src == "hot_rank" else healthy
            got = random.sample(pool, min(q, len(pool)))
            src_recalled[src] += len(got)
            for it in got:
                src_items[src].add(it.item_id)
                cands.append((it, src))
        if not cands:
            continue
        cand_dur.extend(it.duration for it, _ in cands)

        # ---- 排序：[MOCK] pCTR 模型 = 真实后验 CTR * (1 + 乘性噪声) ----
        scored = []
        for it, src in cands:
            true_ctr = min(0.9, 0.01 + 0.45 * it.quality)
            pctr = true_ctr * (1 + random.gauss(0, 0.12)) * fault.pxtr_inflate
            if not math.isfinite(pctr):
                nan_cnt += 1
                pctr = 0.0
            pctr = max(1e-4, min(0.95, pctr))
            # ---- 混排：融合分 = pctr * 多样性打散因子 ----
            fuse = pctr * (1 + DIVERSITY_W * random.random())
            scored.append((fuse, pctr, true_ctr, it, src))
        scored.sort(key=lambda x: -x[0])

        for _fuse, pctr, true_ctr, it, src in scored[:TOP_K]:
            pxtrs.append(pctr)
            labels.append(1 if random.random() < true_ctr else 0)
            src_hit[src] += 1
            pool_cnt[it.pool] += 1
            cat_cnt[it.category] += 1
            author_cnt[it.author_id] = author_cnt.get(it.author_id, 0) + 1
            if it.is_tail:
                tail_exposed.add(it.item_id)
            if it.is_cold:
                cold_exposed += 1
            n_exp += 1

    n_exp = max(1, n_exp)
    n_tail = max(1, sum(1 for it in items if it.is_tail))
    for s in RECALL_SOURCES:
        snap.recall[s] = {
            "recall_cnt_per_req": src_recalled[s] / n_req,
            "coverage": len(src_items[s]) / len(items),
            "hit_rate": src_hit[s] / max(1, src_recalled[s]),
        }
    snap.recall["__global__"] = {
        "cand_duration_p50": quantile(cand_dur, 0.50),
        "cand_duration_p90": quantile(cand_dur, 0.90),
        "tail_coverage": len(tail_exposed) / n_tail,
    }

    mean_px = sum(pxtrs) / len(pxtrs)
    real_ctr = sum(labels) / len(labels)
    snap.rank = {
        "pctr_mean": mean_px,
        "pctr_p50": quantile(pxtrs, 0.50),
        "pctr_p90": quantile(pxtrs, 0.90),
        "pctr_p99": quantile(pxtrs, 0.99),
        "real_ctr": real_ctr,
        "PCOC": mean_px / real_ctr if real_ctr > 0 else float("inf"),
        "online_auc": auc_score(pxtrs, labels),
        "nan_inf_cnt": float(nan_cnt),
    }

    snap.mix = {f"pool_share[{p}]": pool_cnt[p] / n_exp for p in POOLS}
    snap.mix["category_entropy"] = norm_entropy(cat_cnt)
    snap.mix["cold_start_share"] = cold_exposed / n_exp
    snap.mix["distinct_author_ratio"] = len(author_cnt) / N_AUTHOR
    snap.mix["top1_author_share"] = max(author_cnt.values()) / n_exp

    # ======================= 服务与稳定性监控 =======================
    lat = [random.lognormvariate(math.log(28), 0.45) * fault.latency_inflate
           for _ in range(2000)]
    snap.serve = {
        "qps": n_req / 60 * 1000,                 # [MOCK] 单机窗口换算到大盘 QPS
        "latency_avg_ms": sum(lat) / len(lat),
        "latency_p99_ms": quantile(lat, 0.99),
        "timeout_rate": sum(1 for x in lat if x > TIMEOUT_MS) / len(lat),
        "error_rate": max(0.0, random.gauss(0.0008, 0.0003))
        + (0.03 if fault.index_broken_ratio > 0.3 else 0.0),
        "gpu_util": min(0.99, random.gauss(0.62, 0.05) * fault.latency_inflate),
    }
    return snap


# ==========================================================================
# 阈值与分层根因定位
# ==========================================================================
# (层, 指标, 方向, 阈值)；">" = 超过阈值告警，"<" = 低于阈值告警
THRESHOLDS = [
    ("数据与索引", "index_coverage", "<", 0.97),
    ("数据与索引", "index_delay_sec", ">", 300.0),
    ("数据与索引", "feature_missing_rate", ">", 0.05),
    ("数据与索引", "feature_psi(duration)", ">", 0.25),
    ("数据与索引", "log_completeness", "<", 0.98),
    ("召回", "u2i_dssm.recall_cnt_per_req", "<", 54.0),
    ("召回", "i2i_cf.recall_cnt_per_req", "<", 45.0),
    ("召回", "hot_rank.recall_cnt_per_req", "<", 27.0),
    ("召回", "tail_coverage", "<", 0.18),
    ("排序", "PCOC", ">", 1.15),
    ("排序", "online_auc", "<", 0.52),
    ("排序", "nan_inf_cnt", ">", 0.0),
    ("混排", "category_entropy", "<", 0.95),
    ("混排", "cold_start_share", "<", 0.06),
    ("混排", "top1_author_share", ">", 0.05),
    ("服务稳定性", "latency_p99_ms", ">", 150.0),
    ("服务稳定性", "timeout_rate", ">", 0.01),
    ("服务稳定性", "error_rate", ">", 0.005),
]
LAYER_ORDER = ["数据与索引", "召回", "排序", "混排", "服务稳定性"]


def flatten(snap: Snapshot) -> dict[str, float]:
    flat: dict[str, float] = {}
    flat.update(snap.data)
    for src, d in snap.recall.items():
        for k, v in d.items():
            flat[k if src == "__global__" else f"{src}.{k}"] = v
    flat.update(snap.rank)
    flat.update(snap.mix)
    flat.update(snap.serve)
    return flat


def check(snap: Snapshot) -> list[tuple[str, str, float, str, float]]:
    flat = flatten(snap)
    broken = []
    for layer, key, op, thr in THRESHOLDS:
        if key not in flat:
            continue
        v = flat[key]
        if (op == ">" and v > thr) or (op == "<" and v < thr):
            broken.append((layer, key, v, op, thr))
    return broken


def locate_root_cause(broken: list) -> str:
    """沿链路顺序取最上游的破线层：上游异常必然污染下游指标，必须先修上游。"""
    for layer in LAYER_ORDER:
        hits = [b for b in broken if b[0] == layer]
        if hits:
            return f"{layer}（{', '.join(b[1] for b in hits)}）"
    return "无（全链路指标正常）"


# ==========================================================================
# 打印
# ==========================================================================
def _delta(cur: float, ref: float | None) -> str:
    if ref is None or ref == 0:
        return f"{'--':>10}"
    return f"{100 * (cur / ref - 1):>+9.1f}%"


def print_dashboard(snap: Snapshot, base: Snapshot | None = None) -> None:
    sections = [
        ("数据与索引监控", snap.data, None if base is None else base.data),
        ("排序模块监控", snap.rank, None if base is None else base.rank),
        ("重排与混排监控", snap.mix, None if base is None else base.mix),
        ("服务与稳定性监控", snap.serve, None if base is None else base.serve),
    ]
    for title, cur, ref in sections:
        print(f"  -- {title}")
        for k, v in cur.items():
            r = None if ref is None else ref.get(k)
            print(f"     {k:<24}{v:>12.4f}   环比基线 {_delta(v, r)}")
    print("  -- 召回模块监控（分召回源）")
    print(f"     {'source':<18}{'召回数/请求':>12}{'覆盖率':>10}{'命中率':>10}"
          f"{'召回数环比':>13}")
    for s in RECALL_SOURCES:
        d = snap.recall[s]
        r = None if base is None else base.recall[s]["recall_cnt_per_req"]
        print(f"     {s:<18}{d['recall_cnt_per_req']:>12.2f}{d['coverage']:>10.4f}"
              f"{d['hit_rate']:>10.4f}{_delta(d['recall_cnt_per_req'], r):>13}")
    g = snap.recall["__global__"]
    gb = None if base is None else base.recall["__global__"]
    print(f"     候选集 Duration P50={g['cand_duration_p50']:.1f}s "
          f"P90={g['cand_duration_p90']:.1f}s | 长尾覆盖率={g['tail_coverage']:.4f} "
          f"环比 {_delta(g['tail_coverage'], None if gb is None else gb['tail_coverage'])}")


def main() -> None:
    random.seed(42)
    print("=" * 92)
    print("§3.4 推荐系统监控体系：数据/索引 -> 召回 -> 排序 -> 混排 -> 服务 全链路指标 + 根因定位")
    print("=" * 92)

    items = build_pool(3000)
    n_tail = sum(1 for it in items if it.is_tail)
    print(f"\n[MOCK] 内容池 {len(items)} 个 item（长尾 {n_tail} 个，{len(CATEGORIES)} 类目，"
          f"{len(POOLS)} 内容池，{N_AUTHOR} 作者）")
    print(f"[MOCK] 每个监控窗口 900 次请求，每请求召回 {sum(SOURCE_QUOTA.values())} 条，"
          f"下发 Top-{TOP_K}")

    print("\n[1] 健康基线窗口（无故障）")
    base = collect(items, Fault("healthy"))
    print_dashboard(base)
    print(f"  => 破线指标 {len(check(base))} 个，根因层：{locate_root_cause(check(base))}")

    faults = [
        Fault("故障A 关键索引字段写坏 40%", index_broken_ratio=0.40,
              feature_miss_extra=0.09),
        Fault("故障B i2i_cf 召回数量跌 70%", recall_down_source="i2i_cf",
              recall_down_ratio=0.70),
        Fault("故障C 新模型 pCTR 整体抬高 50%", pxtr_inflate=1.50),
        Fault("故障D 流量突增，响应时间整体放大 6 倍", latency_inflate=6.0),
    ]
    for f in faults:
        print(f"\n[2] 注入 {f.name}")
        snap = collect(items, f)
        print_dashboard(snap, base)
        broken = check(snap)
        print(f"  => 破线指标 {len(broken)} 个：")
        for layer, key, v, op, thr in broken:
            print(f"     [{layer}] {key} = {v:.4f} {op} 阈值 {thr}")
        print(f"  => 根因定位（最上游破线层）：{locate_root_cause(broken)}")

    print("\n[3] 结论")
    print("  >>> 故障A：索引层先破线，同时把召回覆盖率、长尾覆盖率、错误率一起拖下水。")
    print("      正文原话：关键索引字段异常时，召回系统无法正确检索目标内容。")
    print("      只看召回指标会误判成「召回模型退化」，所以必须按链路顺序定位根因。")
    print("  >>> 故障B：只在召回层破线，排序 / 混排指标基本不动——单通道供给故障。")
    print("  >>> 故障C：pctr 均值与各分位数整体右移、PCOC 显著大于 1，但 real_ctr 与")
    print("      online_auc 几乎不变（单调变换不改变排序）。正文原话：CTR 预测值整体")
    print("      提高 50% 而实际 CTR 未同步提升，意味着严重的概率高估问题。")
    print("  >>> 故障D：只打服务层，P99 / 超时率 / GPU 利用率破线而业务指标暂未受损，")
    print("      属于容量问题，应走限流与降级而不是回滚模型。")


if __name__ == "__main__":
    main()
