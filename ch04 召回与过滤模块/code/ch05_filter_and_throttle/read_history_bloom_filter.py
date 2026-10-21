"""
对应正文第 4 章 §4.5.1「召回后过滤」中的「用户已消费内容过滤」(通常简称已读过滤)。

正文口径:召回后过滤阶段主要处理「用户已消费内容过滤、用户屏蔽内容过滤、黑名单作者
过滤以及地域限制过滤」,目的是避免「用户已经消费过的内容被反复推荐」(§4.5 开头列出的
第一个典型问题)。正文只给出策略定位,没有规定数据结构;本脚本给出工业界最常用的实现
--布隆过滤器(Bloom Filter),并用实验说明为什么用它而不是精确集合。

布隆过滤器原理:
  m 位 bit 数组 + k 个独立哈希函数。插入元素 x 时把 h_1(x)..h_k(x) 对应的 bit 置 1;
  查询时若 k 个 bit 全为 1 则判「可能已读」,否则判「一定未读」。
  因此它只会「误判未读为已读」(假阳性 False Positive),不会漏掉真正已读的内容
  --这正是已读过滤需要的单边错误方向:宁可少推一个没看过的,也不要重复推荐。

  假阳率理论值:  p = (1 - e^{-k·n/m})^k
  给定 n(元素数)与目标 p,最优参数:
      m = -n·ln p / (ln 2)^2 ,  k = (m/n)·ln 2
  每元素比特数 m/n = -log2(p) / ln 2 ≈ 1.44·log2(1/p),与 n 无关。

本脚本做了四件事:
  1) 纯 Python 实现 BloomFilter(bytearray 位图 + blake2b 双哈希派生 k 个哈希);
  2) 在 5 组 (m/n, k) 配置上实测假阳率,与理论值 (1-e^{-kn/m})^k 对照打印;
  3) 对比「精确 set 存 ID」与「布隆位图」的内存占用,说明亿级用户下的存储差异;
  4) 演示工业界的「按天分桶 + 过期淘汰」已读过滤器(保留最近 N 天已读),
     并在一条模拟召回候选流上打印过滤前后的候选数、重复推荐率。

依赖:仅 Python 3 标准库。[MOCK] 日志流与用户已读历史均为合成数据,不读取任何真实数据集。
"""

import hashlib
import math
import random


# ============================================================ 布隆过滤器实现
class BloomFilter:
    """标准布隆过滤器:m 位位图 + k 个哈希。

    k 个哈希由 blake2b 的一次摘要派生(Kirsch-Mitzenmacher 双哈希):
        h_i(x) = (h1(x) + i · h2(x)) mod m
    这样只做一次摘要就能得到 k 个下标,是工业实现的常用做法。
    """

    def __init__(self, m_bits: int, k: int):
        self.m = m_bits
        self.k = k
        self.bits = bytearray((m_bits + 7) // 8)
        self.n = 0

    def _idx(self, key: str):
        d = hashlib.blake2b(key.encode(), digest_size=16).digest()
        h1 = int.from_bytes(d[:8], "big")
        h2 = int.from_bytes(d[8:], "big") | 1  # 保证 h2 为奇数,避免步长退化
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, key: str):
        for p in self._idx(key):
            self.bits[p >> 3] |= 1 << (p & 7)
        self.n += 1

    def __contains__(self, key: str) -> bool:
        return all(self.bits[p >> 3] >> (p & 7) & 1 for p in self._idx(key))

    def fill_ratio(self) -> float:
        ones = sum(bin(b).count("1") for b in self.bits)
        return ones / self.m

    def bytes_used(self) -> int:
        return len(self.bits)


def theoretical_fpr(m: int, n: int, k: int) -> float:
    """p = (1 - e^{-k·n/m})^k"""
    return (1.0 - math.exp(-k * n / m)) ** k


def optimal_params(n: int, p: float):
    """m = -n·ln p / (ln2)^2 , k = (m/n)·ln2"""
    m = int(math.ceil(-n * math.log(p) / (math.log(2) ** 2)))
    k = max(1, int(round(m / n * math.log(2))))
    return m, k


# ============================================================ 按天分桶的已读过滤器
class DailyReadFilter:
    """[MOCK] 工业界已读服务的简化版:每天一个布隆过滤器,只保留最近 keep_days 天。

    真实系统里这些位图存在 Redis / 自研 KV 中并带 TTL,这里用进程内 dict 模拟,
    不起任何真实服务。查询时对保留窗口内的所有分桶做「或」操作。
    """

    def __init__(self, bits_per_day: int, k: int, keep_days: int):
        self.bits_per_day = bits_per_day
        self.k = k
        self.keep_days = keep_days
        self.buckets = {}

    def _bucket(self, day: int) -> BloomFilter:
        if day not in self.buckets:
            self.buckets[day] = BloomFilter(self.bits_per_day, self.k)
        return self.buckets[day]

    def mark_read(self, day: int, uid: int, item: int):
        self._bucket(day).add(f"{uid}:{item}")

    def is_read(self, day: int, uid: int, item: int) -> bool:
        key = f"{uid}:{item}"
        for d in range(day - self.keep_days + 1, day + 1):
            bf = self.buckets.get(d)
            if bf is not None and key in bf:
                return True
        return False

    def evict(self, day: int) -> int:
        """淘汰超出保留窗口的分桶,返回释放的字节数。"""
        dead = [d for d in self.buckets if d <= day - self.keep_days]
        freed = sum(self.buckets[d].bytes_used() for d in dead)
        for d in dead:
            del self.buckets[d]
        return freed


# ============================================================ 实验
def exp_fpr_table(rng):
    """在多组 (m/n, k) 上实测假阳率 vs 理论值。"""
    n = 20000                       # 已读元素数
    probe = 50000                   # 探测用的未插入元素数
    inserted = [f"u7:item{i}" for i in range(n)]
    probes = [f"u7:probe{i}" for i in range(probe)]

    rows = []
    for bits_per_elem, k in [(4, 3), (8, 4), (8, 6), (12, 8), (16, 11)]:
        m = bits_per_elem * n
        bf = BloomFilter(m, k)
        for key in inserted:
            bf.add(key)
        # 单边性校验:已插入元素必须全部判为已读
        recall = all(key in bf for key in inserted)
        fp = sum(1 for key in probes if key in bf)
        rows.append({
            "bpe": bits_per_elem, "k": k, "m": m,
            "fill": bf.fill_ratio(),
            "emp": fp / probe,
            "theory": theoretical_fpr(m, n, k),
            "no_fn": recall,
            "kb": bf.bytes_used() / 1024,
        })
    return n, probe, rows


def exp_memory_table():
    """精确 set 与布隆位图的内存对比(按 m/n 比特数折算)。"""
    rows = []
    for n, p in [(1_000, 0.01), (100_000, 0.01), (1_000_000, 0.001),
                 (10_000_000, 0.001)]:
        m, k = optimal_params(n, p)
        bloom_bytes = (m + 7) // 8
        # 精确方案:64 位整型 ID 存 Python set,实测单元素约 ~60B(含哈希表槽位开销);
        # 即便用紧凑的 C 数组也至少 8B/元素,这里两种都给出。
        rows.append({
            "n": n, "p": p, "m_over_n": m / n, "k": k,
            "bloom_mb": bloom_bytes / 1024 / 1024,
            "int64_mb": n * 8 / 1024 / 1024,
            "pyset_mb": n * 60 / 1024 / 1024,
        })
    return rows


def exp_online_filter(rng):
    """在模拟召回候选流上跑按天分桶的已读过滤。"""
    n_items = 200_000
    uid = 10086
    days = 10
    keep_days = 7
    # 每天真实消费 300 个物品,7 天窗口内约 2100 个已读元素
    bits_per_day = 300 * 12          # m/n = 12 bit/元素
    rf = DailyReadFilter(bits_per_day, k=8, keep_days=keep_days)

    truth = {}                       # day -> set(已读)
    stats = []
    for day in range(days):
        read_today = {rng.randrange(n_items) for _ in range(300)}
        truth[day] = read_today
        for it in read_today:
            rf.mark_read(day, uid, it)

        # 当天的召回候选:70% 从最近 7 天已读里重复出现(模拟热门反复召回)
        window = set()
        for d in range(max(0, day - keep_days + 1), day + 1):
            window |= truth[d]
        wl = sorted(window)
        cand = []
        for _ in range(3000):
            if wl and rng.random() < 0.7:
                cand.append(rng.choice(wl))
            else:
                cand.append(rng.randrange(n_items))
        cand = list(dict.fromkeys(cand))

        kept = [it for it in cand if not rf.is_read(day, uid, it)]
        # 真实已读(用于统计布隆的误伤)
        really_read = {it for it in cand if it in window}
        wrong_drop = sum(1 for it in cand
                         if it not in window and rf.is_read(day, uid, it))
        freed = rf.evict(day)
        stats.append({
            "day": day, "cand": len(cand), "kept": len(kept),
            "dup": len(really_read),
            "dup_rate_before": len(really_read) / len(cand),
            "dup_rate_after": sum(1 for it in kept if it in window) / max(len(kept), 1),
            "wrong_drop": wrong_drop,
            "buckets": len(rf.buckets),
            "freed_kb": freed / 1024,
            "mem_kb": sum(b.bytes_used() for b in rf.buckets.values()) / 1024,
        })
    return rf, stats


def main():
    random.seed(42)
    rng = random.Random(42)

    print("=" * 78)
    print("§4.5.1 已读过滤:布隆过滤器实现与假阳率实测")
    print("=" * 78)

    n, probe, rows = exp_fpr_table(rng)
    print(f"\n[实验 1] 插入 n={n} 个已读元素,用 {probe} 个未插入元素探测假阳性")
    print(f"{'m/n(bit)':>9}{'k':>4}{'位图KB':>9}{'bit占用率':>11}"
          f"{'实测假阳率':>12}{'理论假阳率':>12}{'相对偏差':>10}{'无假阴':>8}")
    print("-" * 76)
    for r in rows:
        dev = (r["emp"] - r["theory"]) / r["theory"] if r["theory"] > 0 else 0.0
        print(f"{r['bpe']:>9d}{r['k']:>4d}{r['kb']:>9.1f}{r['fill']:>11.3f}"
              f"{r['emp']:>12.5f}{r['theory']:>12.5f}{dev:>+10.1%}"
              f"{str(r['no_fn']):>8}")
    print("  说明:理论值 p=(1-e^{-kn/m})^k;「无假阴」为 True 说明所有真已读元素都被拦住,")
    print("       布隆过滤器的错误是单边的,这正是已读过滤可以接受的方向。")

    print("\n[实验 2] 最优参数下的存储对比  m=-n·ln p/(ln2)^2, k=(m/n)·ln2")
    print(f"{'已读元素数':>11}{'目标假阳率':>11}{'m/n(bit)':>10}{'k':>4}"
          f"{'布隆/MB':>11}{'int64数组/MB':>14}{'Python set/MB':>15}{'压缩比':>9}")
    print("-" * 86)
    for r in exp_memory_table():
        print(f"{r['n']:>11,d}{r['p']:>11.3f}{r['m_over_n']:>10.2f}{r['k']:>4d}"
              f"{r['bloom_mb']:>11.3f}{r['int64_mb']:>14.3f}{r['pyset_mb']:>15.3f}"
              f"{r['int64_mb'] / r['bloom_mb']:>8.1f}x")
    print("  说明:每元素比特数 m/n ≈ 1.44·log2(1/p),与 n 无关;亿级用户 × 千级已读时,")
    print("       这个常数级的压缩比直接决定已读服务能不能放进内存。")

    rf, stats = exp_online_filter(rng)
    print("\n[实验 3] [MOCK] 在线已读过滤:每天 300 条消费,保留 7 天分桶,"
          "候选中 70% 刻意取自已读窗口")
    print(f"{'天':>3}{'候选数':>8}{'真已读数':>9}{'过滤前重复率':>13}"
          f"{'过滤后重复率':>13}{'保留数':>8}{'误伤(假阳)':>11}{'分桶数':>7}{'位图KB':>8}")
    print("-" * 82)
    for s in stats:
        print(f"{s['day']:>3d}{s['cand']:>8d}{s['dup']:>9d}"
              f"{s['dup_rate_before']:>13.2%}{s['dup_rate_after']:>13.2%}"
              f"{s['kept']:>8d}{s['wrong_drop']:>11d}{s['buckets']:>7d}"
              f"{s['mem_kb']:>8.1f}")
    tot_cand = sum(s["cand"] for s in stats)
    tot_wrong = sum(s["wrong_drop"] for s in stats)
    print(f"  累计候选 {tot_cand} 个,过滤后重复推荐率全部降到 0.00%,"
          f"误伤合计 {tot_wrong} 个(实测误伤率 {tot_wrong / tot_cand:.4%});")
    print(f"  第 9 天结束时只剩 {len(rf.buckets)} 个分桶,过期分桶被淘汰,"
          "内存不随历史长度无限增长。")

    print("\n结论(正文 §4.5.1 的工程含义):")
    print("  1) 已读过滤要的是「绝不漏拦」,布隆过滤器的单边误差方向正好匹配:")
    print("     代价是少量未读内容被误判为已读,在亿级库存下这点损失可以接受;")
    print("  2) 工业实现必须带时间窗(按天/按小时分桶 + TTL 淘汰),否则位图越填越满,")
    print("     实测 bit 占用率一旦趋近 1,假阳率会急剧上升到不可用;")
    print("  3) 若业务不能容忍误伤(例如付费内容、重要通知),则需要精确集合或")
    print("     布隆做前置粗筛 + KV 二次确认的两级方案。")


if __name__ == "__main__":
    main()
