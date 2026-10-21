"""
混排模块端到端链路（第 8 章 §8.1 多业务混排问题，图「混排模块内部处理流程」）

对应正文概念
------------
正文把混排模块拆成四个核心环节：
    混排前处理（§8.2） → 混排序列生成与剪枝（§8.3） → 混排多业务排序（§8.4/§8.5） → 混排后处理（§8.6）
输入是推荐 Router 透传的多个业务排序列表（正文记作 {A1..A4}, ..., {D1..D4}），
输出是最终曝光给用户的一条融合序列。

本实现做了什么
--------------
用一个可运行的 Python 链路把四个环节串起来，每一环都打印中间产物，便于读者看到
「一条请求是怎么从 4 个业务列表变成 1 条曝光序列的」：
1. 前处理：保序回归（PAVA）校准各业务 pxtr + Min-Max 归一化，解决量纲不一致；
2. 序列生成：在固定的短视频骨架上枚举广告/直播插入位置，并用约束规则剪枝；
3. 序列排序：统一价值分 × 位置衰减因子 γ_i = e^{-λ(i-1)}，选 Top-1；
4. 后处理：Load 调控（广告占比目标区间）+ 运营强插 + 保量/降权。

关键公式
--------
    校准（IR/PAVA）:  r_k = Σ_{i∈B_k} y_i / |B_k|，相邻违序时合并
                     r_new = (|B_k| r_k + |B_{k+1}| r_{k+1}) / (|B_k| + |B_{k+1}|)
    归一化（Min-Max）: x' = (x - x_min) / (x_max - x_min)
    统一价值:          Value(item) = 兑换系数 × 业务原始价值
    序列价值:          Score_seq = Σ_{i=1..L} γ_i · Value(item_i),  γ_i = e^{-λ(i-1)}
    Load 调控:         score_final = score_raw + u_t

[MOCK] 本文件中的 BusinessRecSys（各业务推荐系统）、Router（推荐路由）、
       ExposureLogStream（曝光日志流）均为教学用假组件，不涉及任何真实 RPC / 存储。
"""

import math
import random

# ---------------------------------------------------------------- 业务常量
# 正文表「不同业务价值的兑换关系示例」：以「分钟时长」为统一价值单位，1 分钟 = 0.05 元
MINUTE_VALUE_RMB = 0.05
EXCHANGE_RATE = {
    "video": 1.0,    # 预计观看 x 分钟 → x 分钟时长价值
    "ad": 20.0,      # 广告收入 1 元 → 20 分钟
    "live": 10.0,    # 直播打赏 1 元 → 10 分钟
    "ecom": 2.0,     # 电商 GMV 1 元 → 2 分钟
}


# ============================================================ [MOCK] 业务侧
class BusinessRecSys:
    """[MOCK] 单个业务的推荐系统：内部已完成召回/排序，对外只吐出 Top-N 候选。"""

    def __init__(self, biz: str, n_items: int, raw_scale: float, bias: float):
        self.biz = biz
        self.n_items = n_items
        self.raw_scale = raw_scale   # 业务原始分量纲（广告 eCPM 可能上千，pCTR 只在 [0,1]）
        self.bias = bias             # 模型系统性高估/低估系数，>1 表示高估

    def serve(self) -> list[dict]:
        out = []
        for i in range(self.n_items):
            true_p = random.betavariate(2.0, 5.0)              # 真实后验概率
            pred_p = min(0.999, true_p * self.bias + random.gauss(0, 0.02))
            pred_p = max(1e-4, pred_p)
            out.append({
                "id": f"{self.biz[0].upper()}{i + 1}",
                "biz": self.biz,
                "true_p": true_p,
                "pxtr": pred_p,                                # 业务模型原始预估
                "raw_score": pred_p * self.raw_scale,          # 业务原始竞价分（量纲各异）
            })
        out.sort(key=lambda d: -d["raw_score"])
        return out


class Router:
    """[MOCK] 推荐 Router：并发请求各业务推荐系统，把多路结果透传给混排服务。"""

    def __init__(self, systems: list[BusinessRecSys]):
        self.systems = systems

    def fanout(self) -> dict[str, list[dict]]:
        return {s.biz: s.serve() for s in self.systems}


class ExposureLogStream:
    """[MOCK] 曝光日志流：滑动窗口统计各业务曝光占比，供 Load 调控读取。"""

    def __init__(self, window: int = 200):
        self.window, self.buf = window, []

    def report(self, seq: list[dict]) -> None:
        for it in seq:
            self.buf.append(it["biz"])
        self.buf = self.buf[-self.window:]

    def load_of(self, biz: str) -> float:
        return 0.0 if not self.buf else sum(b == biz for b in self.buf) / len(self.buf)


# ===================================================== 阶段 1：混排前处理
def pava_calibrate(pairs: list[tuple[float, float]], n_bucket: int = 10):
    """保序回归（PAVA）：按 pxtr 分桶统计真实后验，再合并违反单调的相邻桶。"""
    pairs = sorted(pairs, key=lambda t: t[0])
    size = max(1, len(pairs) // n_bucket)
    blocks = []
    for s in range(0, len(pairs), size):
        chunk = pairs[s:s + size]
        if not chunk:
            continue
        blocks.append({
            "lo": chunk[0][0], "hi": chunk[-1][0],
            "n": len(chunk), "r": sum(y for _, y in chunk) / len(chunk),
        })
    # PAVA：相邻违序则加权合并
    merged: list[dict] = []
    for b in blocks:
        merged.append(b)
        while len(merged) >= 2 and merged[-2]["r"] > merged[-1]["r"]:
            b2, b1 = merged.pop(), merged.pop()
            n = b1["n"] + b2["n"]
            merged.append({"lo": b1["lo"], "hi": b2["hi"], "n": n,
                           "r": (b1["n"] * b1["r"] + b2["n"] * b2["r"]) / n})
    return merged


def apply_calibration(table: list[dict], p: float) -> float:
    """查表 + 区间线性插值（SIR 的平滑思想），避免桶边界跳变。"""
    mids = [(b["lo"] + b["hi"]) / 2 for b in table]
    if p <= mids[0]:
        return table[0]["r"]
    if p >= mids[-1]:
        return table[-1]["r"]
    for k in range(len(mids) - 1):
        if mids[k] <= p <= mids[k + 1]:
            w = (p - mids[k]) / max(1e-12, mids[k + 1] - mids[k])
            return table[k]["r"] + w * (table[k + 1]["r"] - table[k]["r"])
    return table[-1]["r"]


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.5] * len(values) if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in values]


def pre_process(lists: dict[str, list[dict]], logs: dict[str, list[tuple[float, float]]]):
    """校准 + 归一化 + 统一价值折算，输出可跨业务比较的 Value(item)。"""
    report = {}
    for biz, items in lists.items():
        table = pava_calibrate(logs[biz])
        for it in items:
            it["calibrated"] = apply_calibration(table, it["pxtr"])
        before = sum(i["pxtr"] for i in items) / sum(i["true_p"] for i in items)
        after = sum(i["calibrated"] for i in items) / sum(i["true_p"] for i in items)
        norm = minmax([i["calibrated"] for i in items])
        for it, z in zip(items, norm):
            it["norm"] = z
            # 统一价值：归一化分 × 业务侧的原始业务量 × 兑换系数
            it["value"] = EXCHANGE_RATE[biz] * it["calibrated"] * BIZ_UNIT[biz]
        report[biz] = (before, after, len(table))
    return report


# 各业务「一次曝光对应的业务量」：短视频=预计分钟数；广告=eCPM 元；直播=打赏元；电商=GMV 元
BIZ_UNIT = {"video": 6.0, "ad": 0.5, "live": 1.2, "ecom": 4.0}


# ===================================================== 阶段 2：序列生成
def gen_sequences(video: list[dict], ads: list[dict], lives: list[dict],
                  max_ad: int, ad_gap: int, live_first_allowed: bool,
                  prev_tail_is_ad: bool) -> list[list[dict]]:
    """在短视频骨架上枚举广告/直播插入方案，同时执行约束剪枝。"""
    seqs = []

    def ok(seq: list[dict]) -> bool:
        if prev_tail_is_ad and seq[0]["biz"] == "ad":      # 跨刷次 Gap 约束
            return False
        if not live_first_allowed and seq[0]["biz"] == "live":
            return False
        last_ad = -10
        for i, it in enumerate(seq):
            if it["biz"] == "ad":
                if i - last_ad <= ad_gap:                   # 刷次内 Gap 约束
                    return False
                last_ad = i
        return True

    for n_ad in range(max_ad + 1):
        for n_live in (0, 1) if lives else (0,):
            inserts = ads[:n_ad] + lives[:n_live]
            total = len(video) + len(inserts)
            for mask in _combinations(range(total), len(inserts)):
                seq, vi, ii = [], 0, 0
                slots = set(mask)
                for pos in range(total):
                    if pos in slots:
                        seq.append(inserts[ii]); ii += 1
                    else:
                        seq.append(video[vi]); vi += 1
                if ok(seq):
                    seqs.append(seq)
    return seqs


def _combinations(pool, k):
    pool = list(pool)
    if k == 0:
        yield ()
        return
    for i in range(len(pool) - k + 1):
        for rest in _combinations(pool[i + 1:], k - 1):
            yield (pool[i],) + rest


# ===================================================== 阶段 3：序列排序
def seq_score(seq: list[dict], lam: float, mock_value: float, target_len: int) -> float:
    """Score_seq = Σ γ_i · Value(item_i)，不足长度用 Mock Item 补齐。"""
    s = 0.0
    for i in range(target_len):
        gamma = math.exp(-lam * i)
        s += gamma * (seq[i]["value"] if i < len(seq) else mock_value)
    return s


# ===================================================== 阶段 4：混排后处理
class LoadController:
    """PID Load 调控：u_t = Kp·e_t + Ki·Σe + Kd·(e_t - e_{t-1})，score_final = score_raw + u_t。"""

    def __init__(self, target: float, kp=1.2, ki=0.25, kd=0.3,
                 max_integral=1.0, max_adjust=0.4):
        self.target, self.kp, self.ki, self.kd = target, kp, ki, kd
        self.integral, self.last_err = 0.0, 0.0
        self.max_integral, self.max_adjust = max_integral, max_adjust

    def step(self, current: float) -> float:
        e = self.target - current
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral + e))
        u = self.kp * e + self.ki * self.integral + self.kd * (e - self.last_err)
        self.last_err = e
        return max(-self.max_adjust, min(self.max_adjust, u))


def post_process(seq: list[dict], ad_adjust: float, force_item: dict | None,
                 force_pos: int, demote_tags: set[str], quota_biz: str,
                 quota_min: int) -> list[dict]:
    """后处理：Load 调节量修分 → 降权 → 保量兜底 → 运营强插（优先级最高）。"""
    out = [dict(it) for it in seq]
    for it in out:
        if it["biz"] == "ad":
            it["value"] += ad_adjust * 5.0            # Load 调控作用于排序分
        if it.get("tag") in demote_tags:
            it["value"] *= 0.5                        # 降权策略
    # 保量：目标业务曝光数不足则把末位替换为该业务兜底内容
    have = sum(1 for it in out if it["biz"] == quota_biz)
    if have < quota_min:
        out[-1] = {"id": "QUOTA-1", "biz": quota_biz, "value": 1.0, "tag": "quota"}
    # 强插：最后阶段执行，优先级高于模型排序
    if force_item is not None:
        out.insert(min(force_pos, len(out)), force_item)
    return out


# ===================================================== 演示
def main() -> None:
    random.seed(42)
    print("=" * 74)
    print("混排模块端到端链路演示（§8.1 图 8.2 的四个环节）")
    print("=" * 74)

    router = Router([
        BusinessRecSys("video", 6, 1.0, 1.05),
        BusinessRecSys("ad", 3, 1800.0, 1.45),     # 广告系统明显高估
        BusinessRecSys("live", 2, 30.0, 0.80),     # 直播系统低估
        BusinessRecSys("ecom", 2, 120.0, 1.15),
    ])
    lists = router.fanout()

    print("\n【阶段 0】Router 透传的多业务候选（raw_score 量纲完全不可比）")
    for biz, items in lists.items():
        head = ", ".join(f"{i['id']}:{i['raw_score']:.2f}" for i in items[:3])
        print(f"  {biz:<6} n={len(items)}  raw_score 前三 = {head}")

    # 离线日志（用于拟合校准表）
    logs = {}
    for biz, sysm in zip(lists, router.systems):
        rows = []
        for _ in range(4000):
            t = random.betavariate(2.0, 5.0)
            p = max(1e-4, min(0.999, t * sysm.bias + random.gauss(0, 0.02)))
            rows.append((p, 1.0 if random.random() < t else 0.0))
        logs[biz] = rows

    rep = pre_process(lists, logs)
    print("\n【阶段 1】混排前处理：IR 校准 + Min-Max 归一化 + 统一价值折算")
    print(f"  {'业务':<8}{'PCOC(校准前)':>14}{'PCOC(校准后)':>14}{'IR桶数':>8}")
    for biz, (b, a, nb) in rep.items():
        print(f"  {biz:<8}{b:>14.4f}{a:>14.4f}{nb:>8d}")
    print("  统一价值分 Value(item)（已折算为分钟时长价值）:")
    for biz, items in lists.items():
        print(f"    {biz:<6} " + ", ".join(f"{i['id']}={i['value']:.2f}" for i in items))

    video, ads = lists["video"], lists["ad"]
    lives = lists["live"]
    seqs = gen_sequences(video, ads, lives, max_ad=3, ad_gap=3,
                         live_first_allowed=True, prev_tail_is_ad=True)
    print(f"\n【阶段 2】序列生成与剪枝：合法候选序列 {len(seqs)} 条"
          f"（约束：刷次内广告间隔≥3，上一刷尾部是广告故首位禁广告）")

    all_vals = [i["value"] for items in lists.values() for i in items]
    mock_v = sum(all_vals) / len(all_vals)
    target_len = max(len(s) for s in seqs)
    ranked = sorted(seqs, key=lambda s: -seq_score(s, 0.2, mock_v, target_len))
    best = ranked[0]
    print(f"\n【阶段 3】序列排序：λ=0.2 指数位置衰减，Mock Item 价值={mock_v:.3f}，"
          f"对齐长度={target_len}")
    for rank, s in enumerate(ranked[:3], 1):
        print(f"  Top{rank} score={seq_score(s, 0.2, mock_v, target_len):7.3f}  "
              + " → ".join(f"{i['id']}({i['biz'][:3]})" for i in s))

    stream = ExposureLogStream(window=120)
    for s in ranked[:20]:
        stream.report(s)
    ctrl = LoadController(target=0.12)
    cur = stream.load_of("ad")
    u = ctrl.step(cur)
    print(f"\n【阶段 4】混排后处理与流量调控")
    print(f"  滑动窗口广告 Load = {cur:.4f}，目标 = 0.1200，PID 调节量 u_t = {u:+.4f}")
    forced = {"id": "OP-LIVE", "biz": "live", "value": 9.9, "tag": "operation"}
    final = post_process(best, u, forced, force_pos=2,
                         demote_tags={"aigc"}, quota_biz="ecom", quota_min=1)
    print("  最终曝光序列：")
    for i, it in enumerate(final, 1):
        print(f"    pos{i:<2} {it['id']:<8} biz={it['biz']:<6} value={it['value']:.3f}"
              + ("   ← 运营强插" if it.get("tag") == "operation" else "")
              + ("   ← 保量兜底" if it.get("tag") == "quota" else ""))
    print(f"  序列长度 {len(best)} → {len(final)}，"
          f"广告占比 {sum(i['biz'] == 'ad' for i in final) / len(final):.3f}")


if __name__ == "__main__":
    main()
