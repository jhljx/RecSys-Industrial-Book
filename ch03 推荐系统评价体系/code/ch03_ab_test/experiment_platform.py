"""对应正文第 3 章 §3.3.2「A/B实验平台」中列举的五大子系统。

正文原话：「一个典型实验平台包含：实验配置管理，流量分桶系统，指标计算系统，
实验分析系统，风险控制系统。为了提高流量利用率，许多公司还会采用分层实验
（Layered Experimentation）架构……不同层之间互不干扰，从而实现同一批用户流量
被多个实验同时复用。」

本实现做了什么
---------------
把这五个子系统各写成一个进程内 `[MOCK]` 组件，并串成一次完整的实验生命周期：
1. **实验配置管理（ConfigRegistry）**：实验以 `ExperimentConfig` 描述（所属层、
   桶区间、参数、核心/诊断指标与护栏阈值）。注册时做三类静态校验——
   (a) 层内**互斥**：同层实验的桶区间不得重叠；
   (b) 层内流量不得超过 100 个桶；
   (c) 参数必须落在该层声明的参数空间内（防止一个层里两个实验改同一个参数）。
2. **实验配置下发（ConfigDistributor）**：带版本号的配置灰度推送到多台 `[MOCK]`
   在线服务实例，模拟推送延迟与单实例推送失败，并监控「配置版本一致率」——
   版本不一致期间该实例仍按旧配置分流，是线上实验「脏流量」的常见来源。
3. **流量分桶系统（BucketService）**：md5(层 salt + uid) mod 100，层内互斥、
   层间正交，同一用户流量被多层实验同时复用。
4. **指标计算系统（MetricPipeline）**：按 (实验, 组) 聚合 `[MOCK]` 曝光/行为日志，
   算核心指标（CTR、人均时长）与诊断指标（hate 率、P99 延迟）。
5. **实验分析系统（AnalysisService）**：Welch t 检验 / 两比例 z 检验 + 95% 置信区间，
   给出显著性结论。
6. **风险控制系统（RiskControl）**：实时巡检护栏指标，触发阈值即自动「止损」
   （把实验流量拉回 0 并回滚配置版本）。

关键公式
---------
    bucket(uid, layer) = md5(f"{salt}:{uid}") mod 100
    Welch t = (x1-x2)/sqrt(s1^2/n1+s2^2/n2)
    两比例 z = (p1-p2)/sqrt(p_pool(1-p_pool)(1/n1+1/n2))
    配置版本一致率 = #{实例 version == 最新 version} / #实例
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field


# ==========================================================================
# 统计工具（纯标准库）
# ==========================================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_ttest(a: list[float], b: list[float]) -> dict:
    """双样本 Welch t 检验（方差不等），大样本下用正态近似算 p 与 CI。"""
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    p = 2 * (1 - norm_cdf(abs(t)))
    return {"mean_exp": m1, "mean_base": m2, "diff": m1 - m2, "t": t, "p": p,
            "ci_low": (m1 - m2) - 1.959964 * se, "ci_high": (m1 - m2) + 1.959964 * se,
            "lift_pct": 100.0 * (m1 / m2 - 1)}


def two_proportion_ztest(c1: int, n1: int, c2: int, n2: int) -> dict:
    """两比例 z 检验（CTR 这类比例指标）。"""
    p1, p2 = c1 / n1, c2 / n2
    pp = (c1 + c2) / (n1 + n2)
    se_pool = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    se_un = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = (p1 - p2) / se_pool if se_pool > 0 else 0.0
    return {"p1": p1, "p2": p2, "z": z, "p": 2 * (1 - norm_cdf(abs(z))),
            "ci_low": (p1 - p2) - 1.959964 * se_un,
            "ci_high": (p1 - p2) + 1.959964 * se_un,
            "lift_pct": 100.0 * (p1 / p2 - 1) if p2 > 0 else float("nan")}


# ==========================================================================
# 子系统 1：实验配置管理
# ==========================================================================
@dataclass
class ExperimentConfig:
    """一个实验的完整配置。buckets 为该实验在所属层占据的桶号集合。"""

    exp_id: str
    layer: str
    buckets: range                      # 层内互斥地占据这些桶
    params: dict                        # 下发给在线服务的策略参数
    core_metrics: tuple[str, ...]
    guardrails: dict                    # 指标名 -> 允许的最大相对恶化（负向，%）
    owner: str = "rec-algo"
    status: str = "pending"             # pending / running / stopped / rolled_back


@dataclass
class LayerSpec:
    """层定义：salt 决定层间正交；param_space 限定本层能改哪些参数。"""

    name: str
    salt: str
    param_space: tuple[str, ...]
    n_buckets: int = 100


class ConfigRegistry:
    """[MOCK] 实验配置管理：注册 + 三类静态校验（互斥 / 容量 / 参数空间）。"""

    def __init__(self, layers: list[LayerSpec]) -> None:
        self.layers = {ly.name: ly for ly in layers}
        self.experiments: dict[str, ExperimentConfig] = {}
        self.version = 0

    def _occupied(self, layer: str) -> set[int]:
        occ: set[int] = set()
        for e in self.experiments.values():
            if e.layer == layer and e.status == "running":
                occ |= set(e.buckets)
        return occ

    def register(self, cfg: ExperimentConfig) -> tuple[bool, str]:
        if cfg.layer not in self.layers:
            return False, f"层 {cfg.layer} 不存在"
        ly = self.layers[cfg.layer]
        bad = [k for k in cfg.params if k not in ly.param_space]
        if bad:
            return False, f"参数 {bad} 不在层 {cfg.layer} 的参数空间 {ly.param_space} 内"
        occ = self._occupied(cfg.layer)
        clash = sorted(occ & set(cfg.buckets))
        if clash:
            return False, (f"层内互斥冲突：桶 {clash[0]}~{clash[-1]} 已被同层实验占用"
                           f"（共 {len(clash)} 个桶）")
        if len(occ) + len(cfg.buckets) > ly.n_buckets:
            return False, f"层 {cfg.layer} 流量超额：已用 {len(occ)}% + 申请 {len(cfg.buckets)}%"
        cfg.status = "running"
        self.experiments[cfg.exp_id] = cfg
        self.version += 1
        return True, f"注册成功，占 {len(cfg.buckets)}% 流量，配置版本 -> v{self.version}"

    def stop(self, exp_id: str, reason: str) -> None:
        self.experiments[exp_id].status = "stopped"
        self.version += 1
        print(f"      [ConfigRegistry] 实验 {exp_id} 已置为 stopped（{reason}），"
              f"配置版本 -> v{self.version}")


# ==========================================================================
# 子系统 2：实验配置下发
# ==========================================================================
@dataclass
class ServiceInstance:
    """[MOCK] 一台在线推荐服务实例，持有自己的配置版本快照。"""

    host: str
    version: int = 0
    pushed_at_s: float = 0.0


class ConfigDistributor:
    """[MOCK] 配置灰度下发：分批推送 + 推送延迟 + 单实例失败。"""

    def __init__(self, n_instances: int = 20, fail_rate: float = 0.10) -> None:
        self.instances = [ServiceInstance(f"rec-online-{i:02d}") for i in range(n_instances)]
        self.fail_rate = fail_rate

    def push(self, version: int, batch_ratio: float = 0.25) -> list[tuple[float, float]]:
        """按 batch_ratio 分批推送，返回 [(时刻秒, 版本一致率)] 曲线。"""
        order = list(range(len(self.instances)))
        random.shuffle(order)
        batch = max(1, int(len(order) * batch_ratio))
        curve: list[tuple[float, float]] = []
        t = 0.0
        for start in range(0, len(order), batch):
            t += random.uniform(1.5, 3.0)                 # 批次间隔
            for idx in order[start:start + batch]:
                if random.random() >= self.fail_rate:     # 可能推送失败
                    self.instances[idx].version = version
                    self.instances[idx].pushed_at_s = t
            curve.append((t, self.consistency(version)))
        return curve

    def consistency(self, version: int) -> float:
        ok = sum(1 for s in self.instances if s.version == version)
        return ok / len(self.instances)

    def stale_hosts(self, version: int) -> list[str]:
        return [s.host for s in self.instances if s.version != version]


# ==========================================================================
# 子系统 3：流量分桶
# ==========================================================================
class BucketService:
    """[MOCK] 流量分桶系统：层内互斥、层间正交（不同 salt）。"""

    def __init__(self, registry: ConfigRegistry) -> None:
        self.reg = registry

    @staticmethod
    def bucket(uid: int, salt: str, n: int = 100) -> int:
        h = hashlib.md5(f"{salt}:{uid}".encode()).hexdigest()
        return int(h, 16) % n

    def assign(self, uid: int) -> dict[str, tuple[str, str]]:
        """返回 {层名: (exp_id, 'exp'|'base')}；未命中任何实验则落在该层 base。"""
        out: dict[str, tuple[str, str]] = {}
        for name, ly in self.reg.layers.items():
            b = self.bucket(uid, ly.salt, ly.n_buckets)
            hit = ("__base__", "base")
            for e in self.reg.experiments.values():
                if e.layer == name and e.status == "running" and b in e.buckets:
                    hit = (e.exp_id, "exp")
                    break
            out[name] = hit
        return out


# ==========================================================================
# 子系统 4：指标计算
# ==========================================================================
@dataclass
class GroupAgg:
    """一个 (实验, 组) 的指标累加器。"""

    imps: int = 0
    clicks: int = 0
    hates: int = 0
    users: set = field(default_factory=set)
    watch_by_user: dict = field(default_factory=lambda: defaultdict(float))
    latencies: list = field(default_factory=list)

    def ctr(self) -> float:
        return self.clicks / self.imps if self.imps else 0.0

    def hate_rate(self) -> float:
        return self.hates / self.imps if self.imps else 0.0

    def watch_samples(self) -> list[float]:
        return [self.watch_by_user[u] for u in sorted(self.users)]

    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[min(len(s) - 1, int(0.99 * len(s)))]


class MetricPipeline:
    """[MOCK] 指标计算系统：消费曝光/行为日志，按 (实验, 组) 聚合。"""

    def __init__(self) -> None:
        self.agg: dict[tuple[str, str], GroupAgg] = defaultdict(GroupAgg)

    def consume(self, exp_id: str, group: str, uid: int, imps: int, clicks: int,
                hates: int, watch_s: float, latency_ms: float) -> None:
        g = self.agg[(exp_id, group)]
        g.imps += imps
        g.clicks += clicks
        g.hates += hates
        g.users.add(uid)
        g.watch_by_user[uid] += watch_s
        g.latencies.append(latency_ms)


# ==========================================================================
# 子系统 5 / 6：实验分析 + 风险控制
# ==========================================================================
class AnalysisService:
    """[MOCK] 实验分析系统：显著性检验 + 置信区间 + 结论。"""

    @staticmethod
    def analyze(exp: GroupAgg, base: GroupAgg) -> dict:
        ctr = two_proportion_ztest(exp.clicks, exp.imps, base.clicks, base.imps)
        wt = welch_ttest(exp.watch_samples(), base.watch_samples())
        return {"ctr": ctr, "watch_time": wt}


class RiskControl:
    """[MOCK] 风险控制系统：护栏指标超阈值自动止损（kill switch）。"""

    def __init__(self, registry: ConfigRegistry, distributor: ConfigDistributor) -> None:
        self.reg = registry
        self.dist = distributor

    def inspect(self, exp_id: str, exp: GroupAgg, base: GroupAgg) -> list[str]:
        cfg = self.reg.experiments[exp_id]
        alarms: list[str] = []
        observed = {
            "hate_rate": (exp.hate_rate(), base.hate_rate(), "越低越好"),
            "p99_latency_ms": (exp.p99_latency(), base.p99_latency(), "越低越好"),
            "ctr": (exp.ctr(), base.ctr(), "越高越好"),
        }
        for name, limit_pct in cfg.guardrails.items():
            o, b, direction = observed[name]
            rel = 100.0 * (o / b - 1) if b else 0.0
            worse = rel if direction == "越高越好" else -rel
            if worse < limit_pct:
                alarms.append(f"{name}: {o:.4f} vs base {b:.4f}（相对 {rel:+.2f}%），"
                              f"突破护栏 {limit_pct:+.1f}%")
        if alarms:
            self.reg.stop(exp_id, reason="护栏指标恶化")
            self.dist.push(self.reg.version)
        return alarms


# ==========================================================================
# [MOCK] 线上流量
# ==========================================================================
def simulate_user_day(uid: int, assignment: dict[str, tuple[str, str]],
                      registry: ConfigRegistry) -> dict:
    """[MOCK] 一个用户一天的汇总行为；每层命中的实验按自身效应乘性叠加。"""
    rng = random.Random(uid * 104729 + 7)
    imps = max(3, int(rng.gauss(26, 9)))
    ctr, wt_per_imp, hate, lat = 0.072, 1.55, 0.0045, 38.0
    for layer, (exp_id, grp) in assignment.items():
        if grp != "exp":
            continue
        eff = registry.experiments[exp_id].params
        ctr *= eff.get("ctr_mult", 1.0)
        wt_per_imp *= eff.get("wt_mult", 1.0)
        hate *= eff.get("hate_mult", 1.0)
        lat *= eff.get("latency_mult", 1.0)
    clicks = sum(1 for _ in range(imps) if rng.random() < ctr)
    hates = sum(1 for _ in range(imps) if rng.random() < hate)
    watch = sum(rng.expovariate(1.0 / max(wt_per_imp, 1e-6)) for _ in range(clicks + 1))
    return {"imps": imps, "clicks": clicks, "hates": hates, "watch_s": watch,
            "latency_ms": lat * rng.lognormvariate(0.0, 0.35)}


def main() -> None:
    random.seed(42)
    print("=" * 86)
    print("§3.3.2 A/B实验平台：配置管理 / 配置下发 / 流量分桶 / 指标计算 / 实验分析 / 风险控制")
    print("=" * 86)

    layers = [
        LayerSpec("L1-召回层", "salt_recall_2024", ("recall_source", "ctr_mult", "wt_mult")),
        LayerSpec("L2-排序层", "salt_rank_2024",
                  ("model", "ctr_mult", "wt_mult", "latency_mult", "hate_mult")),
        LayerSpec("L3-重排层", "salt_rerank_2024", ("diversity_lambda", "ctr_mult", "hate_mult")),
    ]
    reg = ConfigRegistry(layers)
    dist = ConfigDistributor(n_instances=20, fail_rate=0.10)

    # ---------------- (1) 配置管理：注册与校验 ----------------
    print("\n[1] 实验配置管理：注册 + 静态校验（层内互斥 / 容量 / 参数空间）")
    todo = [
        ExperimentConfig("exp_recall_i2i", "L1-召回层", range(90, 100),
                         {"recall_source": "i2i_v3", "ctr_mult": 1.03, "wt_mult": 1.01},
                         ("ctr",), {"hate_rate": -5.0}),
        ExperimentConfig("exp_rank_mmoe", "L2-排序层", range(85, 95),
                         {"model": "mmoe_v2", "ctr_mult": 1.05, "wt_mult": 1.04,
                          "latency_mult": 1.10},
                         ("ctr", "watch_time"), {"hate_rate": -5.0, "p99_latency_ms": -20.0}),
        # 与 exp_rank_mmoe 的桶区间重叠 -> 应被互斥校验拒绝
        ExperimentConfig("exp_rank_dcn", "L2-排序层", range(92, 100),
                         {"model": "dcn_v2", "ctr_mult": 1.01},
                         ("ctr",), {"hate_rate": -5.0}),
        # 改了不属于本层的参数 -> 应被参数空间校验拒绝
        ExperimentConfig("exp_rerank_bad", "L3-重排层", range(90, 100),
                         {"model": "rerank_mmr", "ctr_mult": 1.0},
                         ("ctr",), {"hate_rate": -5.0}),
        ExperimentConfig("exp_rerank_mmr", "L3-重排层", range(90, 100),
                         {"diversity_lambda": 0.3, "ctr_mult": 0.995, "hate_mult": 0.80},
                         ("ctr",), {"hate_rate": -5.0}),
        # 激进策略：CTR 涨但 hate 率与 P99 延迟双双恶化 -> 应被风控止损
        ExperimentConfig("exp_rank_aggressive", "L2-排序层", range(75, 85),
                         {"model": "ctr_only_v9", "ctr_mult": 1.12, "wt_mult": 0.93,
                          "hate_mult": 1.60, "latency_mult": 1.45},
                         ("ctr",), {"hate_rate": -5.0, "p99_latency_ms": -20.0}),
    ]
    print(f"  {'实验':>20} | {'层':>10} | {'桶区间':>9} | 结果")
    print("  " + "-" * 82)
    for cfg in todo:
        ok, msg = reg.register(cfg)
        rng_s = f"[{cfg.buckets.start},{cfg.buckets.stop - 1}]"
        print(f"  {cfg.exp_id:>20} | {cfg.layer:>10} | {rng_s:>9} | "
              f"{'✓ ' if ok else '✗ '}{msg}")

    # ---------------- (2) 配置灰度下发 ----------------
    print(f"\n[2] 实验配置下发：当前配置版本 v{reg.version}，灰度推送到 20 台在线实例")
    curve = dist.push(reg.version, batch_ratio=0.25)
    print(f"  {'推送时刻(s)':>11} | {'版本一致率':>10} | 说明")
    print("  " + "-" * 56)
    for t, c in curve:
        note = "推送完成" if c == 1.0 else "部分实例仍按旧配置分流（脏流量）"
        print(f"  {t:>11.2f} | {100 * c:>9.1f}% | {note}")
    stale = dist.stale_hosts(reg.version)
    print(f"  最终一致率 {100 * dist.consistency(reg.version):.1f}%，"
          f"滞后实例 {len(stale)} 台：{stale if stale else '无'}")
    if stale:
        print("  >>> 配置下发未对齐期间，这些实例上的用户分组与指标归因都不可信，")
        print("      实验平台必须在「配置生效时间」之后才开始统计，否则实验组被旧配置污染。")

    # ---------------- (3) 流量分桶与流量复用 ----------------
    print("\n[3] 流量分桶系统：层内互斥 + 层间正交 => 同一批用户被多个实验复用")
    bs = BucketService(reg)
    n_users = 60_000
    hit_count = defaultdict(int)       # 命中实验个数 -> 用户数
    per_exp = defaultdict(int)
    assigns: dict[int, dict] = {}
    for u in range(n_users):
        a = bs.assign(u)
        assigns[u] = a
        k = sum(1 for _, (_, g) in a.items() if g == "exp")
        hit_count[k] += 1
        for _, (eid, g) in a.items():
            if g == "exp":
                per_exp[eid] += 1
    print(f"  {'同时命中的实验数':>16} | {'用户数':>8} | {'占比':>8}")
    print("  " + "-" * 40)
    for k in sorted(hit_count):
        print(f"  {k:>16d} | {hit_count[k]:>8d} | {100.0 * hit_count[k] / n_users:>7.2f}%")
    exp_total = sum(per_exp.values())
    layer_use = defaultdict(int)
    for cfg in reg.experiments.values():
        layer_use[cfg.layer] += len(cfg.buckets)
    print(f"  实验覆盖人次合计 = {exp_total}，物理用户 = {n_users}，"
          f"人均参与实验 {exp_total / n_users:.2f} 个")
    print("  分层后各层流量占用：" + "，".join(
        f"{lid} {pct}%" for lid, pct in sorted(layer_use.items())))
    print(f"  若不分层，这 {len(reg.experiments)} 个实验要在单层里互斥瓜分 "
          f"{sum(layer_use.values())}% 流量；分层后每层都能独立用满 100%，"
          f"这正是正文说的「同一批用户流量被多个实验同时复用」")
    print(f"  {'实验':>20} | {'用户数':>8} | {'实际占比':>8} | {'设计占比':>8}")
    print("  " + "-" * 56)
    for eid, cfg in reg.experiments.items():
        print(f"  {eid:>20} | {per_exp[eid]:>8d} | "
              f"{100.0 * per_exp[eid] / n_users:>7.2f}% | {len(cfg.buckets):>7d}%")

    # ---------------- (4) 指标计算 ----------------
    print("\n[4] 指标计算系统：消费一天的 [MOCK] 曝光/行为日志")
    mp = MetricPipeline()
    for u in range(n_users):
        a = assigns[u]
        day = simulate_user_day(u, a, reg)
        for layer, (eid, grp) in a.items():
            key_exp = eid if grp == "exp" else f"__base__::{layer}"
            mp.consume(key_exp, grp, u, day["imps"], day["clicks"], day["hates"],
                       day["watch_s"], day["latency_ms"])
    print(f"  {'统计单元':>26} | {'曝光':>9} | {'CTR':>7} | {'人均时长':>9} | "
          f"{'hate率':>7} | {'P99(ms)':>8}")
    print("  " + "-" * 86)
    for (key, grp), g in sorted(mp.agg.items()):
        wt = sum(g.watch_by_user.values()) / len(g.users)
        print(f"  {key[:26]:>26} | {g.imps:>9d} | {100 * g.ctr():>6.3f}% | "
              f"{wt:>9.3f} | {100 * g.hate_rate():>6.3f}% | {g.p99_latency():>8.2f}")

    # ---------------- (5) 实验分析 ----------------
    print("\n[5] 实验分析系统：各实验 vs 同层 base（Welch t / 两比例 z，95% CI）")
    print(f"  {'实验':>20} | {'指标':>10} | {'Base':>9} | {'Exp':>9} | "
          f"{'相对提升':>9} | {'p-value':>9} | {'95% CI(绝对)':>24} | 结论")
    print("  " + "-" * 122)
    results = {}
    for eid, cfg in reg.experiments.items():
        e = mp.agg[(eid, "exp")]
        b = mp.agg[(f"__base__::{cfg.layer}", "base")]
        r = AnalysisService.analyze(e, b)
        results[eid] = (e, b, r)
        for mname, key, scale, unit in (("CTR", "ctr", 100.0, "%"),
                                        ("人均时长", "watch_time", 1.0, "s")):
            d = r[key]
            lo, hi = d["ci_low"] * scale, d["ci_high"] * scale
            base_v = (d["p2"] if key == "ctr" else d["mean_base"]) * scale
            exp_v = (d["p1"] if key == "ctr" else d["mean_exp"]) * scale
            sig = "显著正向" if d["p"] < 0.05 and d["lift_pct"] > 0 else (
                "显著负向" if d["p"] < 0.05 else "不显著")
            print(f"  {eid:>20} | {mname:>10} | {base_v:>8.4f}{unit} | {exp_v:>8.4f}{unit} | "
                  f"{d['lift_pct']:>+8.2f}% | {d['p']:>9.2e} | "
                  f"[{lo:>+8.4f}, {hi:>+8.4f}]{unit:>2} | {sig}")

    # ---------------- (6) 风险控制 ----------------
    print("\n[6] 风险控制系统：巡检护栏指标，超阈值自动止损")
    for eid, (e, b, _) in results.items():
        alarms = RiskControl(reg, dist).inspect(eid, e, b)
        if alarms:
            print(f"  ✗ {eid}：触发 {len(alarms)} 条护栏告警")
            for a in alarms:
                print(f"      - {a}")
        else:
            print(f"  ✓ {eid}：护栏全部通过，允许继续放量")
    print("\n  实验最终状态：")
    for eid, cfg in reg.experiments.items():
        print(f"    {eid:>20} -> {cfg.status}")
    print(f"  配置版本已推进到 v{reg.version}，"
          f"止损后的一致率 {100 * dist.consistency(reg.version):.1f}%")
    print("\n  >>> 正文：实验平台由配置管理、流量分桶、指标计算、实验分析、风险控制五部分组成；")
    print("      exp_rank_aggressive 正是正文反复提醒的情形——核心指标 CTR 显著涨，")
    print("      但诊断指标（hate 率、P99 延迟）同时恶化，必须由风控系统拦住而非放量。")


if __name__ == "__main__":
    main()
