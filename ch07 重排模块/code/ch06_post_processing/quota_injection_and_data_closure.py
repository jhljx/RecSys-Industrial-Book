"""对应正文第 7 章 §7.6「重排后处理」中的**保量/强插策略 + 数据闭环 + 监控埋点**。

本实现做了什么
----------------
正文 §7.6 把后处理的职责分成四块,本脚本按原文顺序逐块实现(算法轻量,工程链路完整):

1. **保量策略(quota control)与强插策略(forced injection)**
   - 保量:对「商业内容 / 活动内容 / 运营内容」三类设定每刷最低条数与最高占比,
     不足则从对应池子抽取补入,超额则截断;
   - 强插:在固定坑位(如第 4、9 位,1-based)无条件插入指定内容,原内容整体后移;
   - 两者按「强插 → 保量 → 占比上限截断」的顺序执行,并统计对自然内容的挤出量。
2. **Redis 在线持久化** `[MOCK] RedisClient`:把重排中间信息(最终序列、各物品
   来源通路、强插/保量命中标记、融合分)以 `rerank:{req_id}` 为 key 写入,
   带 TTL;支持按 req_id 回溯查询(正文的「实时查询与策略回溯」)。
3. **Kafka 异步日志投递** `[MOCK] KafkaProducer`:把曝光级日志按分区异步 send,
   内部用队列缓冲 + flush 批量落盘,统计吞吐/批大小/丢弃数(模拟 buffer 满)。
4. **监控埋点体系**:按正文列的维度统计
   业务线曝光占比、强插比例、策略命中率、类别覆盖率、作者覆盖率,
   并做 A/B 两组(关闭 / 打开保量)的对比表。

关键公式
--------
- 类别覆盖率 = |{序列中出现的类别}| / |全部类别|
- 作者覆盖率 = |{序列中出现的作者}| / |序列长度|
- 业务线 b 的曝光占比 = (b 的曝光数) / (总曝光数)
- 保量缺口 = max(0, quota_min - 当前条数);占比上限截断量 = max(0, cnt - floor(L * ratio_max))
"""

import collections
import random
import time

BIZ_NATURAL = "natural"
BIZ_LINES = ("ad", "activity", "operation")     # 商业 / 活动 / 运营
CATS = ["游戏", "美食", "剧情", "知识", "体育", "音乐"]
AUTHORS = [f"u{i:02d}" for i in range(20)]
LIST_LEN = 12


class Doc:
    __slots__ = ("iid", "biz", "cat", "author", "fuse", "src", "tag")

    def __init__(self, iid, biz, cat, author, fuse, src):
        self.iid, self.biz, self.cat = iid, biz, cat
        self.author, self.fuse, self.src = author, fuse, src
        self.tag = ""        # 后处理打上的标记:forced / quota

    def brief(self):
        return f"{self.iid}:{self.biz[:3]}/{self.cat}"


# ---------------------------------------------------------------------------
# [MOCK] 基础设施
# ---------------------------------------------------------------------------
class RedisClient:
    """[MOCK] Redis 在线缓存:dict + 过期时间戳,只实现 setex / get / ttl / keys。"""

    def __init__(self):
        self._kv = {}
        self.n_set = 0
        self.n_get = 0
        self.n_miss = 0

    def setex(self, key, ttl, value):
        self._kv[key] = (time.time() + ttl, value)
        self.n_set += 1

    def get(self, key, now=None):
        self.n_get += 1
        rec = self._kv.get(key)
        now = now if now is not None else time.time()
        if rec is None or rec[0] < now:
            self.n_miss += 1
            return None
        return rec[1]

    def keys(self):
        return list(self._kv)


class KafkaProducer:
    """[MOCK] Kafka 生产者:环形缓冲 + 批量 flush,缓冲满则丢弃并计数。"""

    def __init__(self, topic, n_partition=3, buffer_max=64, batch_size=16):
        self.topic = topic
        self.n_partition = n_partition
        self.buffer_max = buffer_max
        self.batch_size = batch_size
        self._buf = collections.deque()
        self.sent = 0
        self.dropped = 0
        self.flushes = 0
        self.batch_sizes = []
        self.per_partition = collections.Counter()

    def send(self, key, payload):
        if len(self._buf) >= self.buffer_max:
            self.dropped += 1
            return False
        part = hash(key) % self.n_partition
        self._buf.append((part, payload))
        if len(self._buf) >= self.batch_size:
            self.flush()
        return True

    def flush(self):
        if not self._buf:
            return 0
        n = len(self._buf)
        while self._buf:
            part, _ = self._buf.popleft()
            self.per_partition[part] += 1
            self.sent += 1
        self.flushes += 1
        self.batch_sizes.append(n)
        return n


# ---------------------------------------------------------------------------
# 保量 + 强插
# ---------------------------------------------------------------------------
class QuotaPolicy:
    """每个业务线的保量下限 / 占比上限 / 兜底池。"""

    def __init__(self, quota_min, ratio_max):
        self.quota_min = quota_min      # dict biz -> 最少条数
        self.ratio_max = ratio_max      # dict biz -> 最大占比


def forced_injection(seq, forced_items, slots, list_len):
    """固定坑位强插:slots 为 1-based 位置,原内容整体后移。返回 (新序列, 挤出条数)。"""
    out = list(seq)
    for item, slot in zip(forced_items, slots):
        pos = min(max(slot - 1, 0), len(out))
        item.tag = "forced"
        out.insert(pos, item)
    squeezed = max(0, len(out) - list_len)
    return out[:list_len], squeezed


def apply_quota(seq, policy, backup_pools, list_len):
    """保量补齐 + 占比上限截断。返回 (新序列, 统计 dict)。"""
    out = list(seq)
    stat = {"filled": collections.Counter(), "trimmed": collections.Counter()}

    # (a) 下限:不足则从对应池子补,替换掉末尾分数最低的自然内容
    for biz, need in policy.quota_min.items():
        cnt = sum(1 for d in out if d.biz == biz)
        gap = max(0, need - cnt)
        pool = [d for d in backup_pools.get(biz, []) if d.iid not in {x.iid for x in out}]
        for k in range(min(gap, len(pool))):
            victims = [i for i, d in enumerate(out)
                       if d.biz == BIZ_NATURAL and d.tag != "forced"]
            if not victims:
                break
            worst = min(victims, key=lambda i: out[i].fuse)
            pool[k].tag = "quota"
            out[worst] = pool[k]
            stat["filled"][biz] += 1

    # (b) 上限:超过 floor(L * ratio_max) 的部分按融合分从低到高剔除
    for biz, ratio in policy.ratio_max.items():
        cap = int(list_len * ratio)
        idxs = [i for i, d in enumerate(out) if d.biz == biz]
        over = len(idxs) - cap
        if over <= 0:
            continue
        drop = sorted(idxs, key=lambda i: out[i].fuse)[:over]
        for i in sorted(drop, reverse=True):
            out.pop(i)
            stat["trimmed"][biz] += 1
    return out[:list_len], stat


# ---------------------------------------------------------------------------
# 监控指标
# ---------------------------------------------------------------------------
def monitor(seq, all_cats=CATS):
    biz_cnt = collections.Counter(d.biz for d in seq)
    n = max(1, len(seq))
    return {
        "len": len(seq),
        "biz_ratio": {b: biz_cnt[b] / n for b in (BIZ_NATURAL,) + BIZ_LINES},
        "forced_ratio": sum(1 for d in seq if d.tag == "forced") / n,
        "quota_hit": sum(1 for d in seq if d.tag == "quota") / n,
        "cat_cover": len({d.cat for d in seq}) / len(all_cats),
        "author_cover": len({d.author for d in seq}) / n,
        "fuse_sum": sum(d.fuse / (k + 1) for k, d in enumerate(seq)),
    }


def make_request(rng, req_id):
    """[MOCK] 重排模型输出 + 三类业务兜底池。"""
    natural = []
    for i in range(LIST_LEN + 6):
        natural.append(Doc(f"n{req_id:03d}{i:02d}", BIZ_NATURAL,
                           CATS[rng.randrange(len(CATS))],
                           AUTHORS[rng.randrange(6)],       # 自然内容作者偏集中
                           1.0 - i * 0.02 + rng.uniform(-0.01, 0.01),
                           src=rng.choice(["beam", "greedy", "gauss"])))
    natural.sort(key=lambda d: -d.fuse)
    pools = {}
    for b in BIZ_LINES:
        pools[b] = [Doc(f"{b[0]}{req_id:03d}{j:02d}", b,
                        CATS[rng.randrange(len(CATS))],
                        AUTHORS[6 + rng.randrange(14)],
                        0.5 + rng.uniform(0, 0.3), src="quota_pool")
                    for j in range(4)]
    return natural[:LIST_LEN], pools


def main():
    random.seed(42)
    rng = random.Random(42)

    policy = QuotaPolicy(
        quota_min={"ad": 2, "activity": 1, "operation": 1},
        ratio_max={"ad": 0.25, "activity": 0.17, "operation": 0.17},
    )
    redis = RedisClient()
    kafka = KafkaProducer("rerank_exposure_log", n_partition=3,
                          buffer_max=64, batch_size=16)

    print("== §7.6 重排后处理:强插 + 保量 + Redis 持久化 + Kafka 日志 + 监控 ==")
    print(f"序列长度 L={LIST_LEN};保量下限 {policy.quota_min};"
          f"占比上限 {policy.ratio_max}")
    print(f"强插坑位:第 4 位(运营)、第 9 位(活动)\n")

    # ---------------- 单次请求的逐步展示 ----------------
    base, pools = make_request(rng, 0)
    print("[step0 重排输出] " + " ".join(d.brief() for d in base))
    s0 = monitor(base)

    step1, squeezed = forced_injection(
        base, [pools["operation"][0], pools["activity"][0]], [4, 9], LIST_LEN)
    print("[step1 强插后]   " + " ".join(d.brief() for d in step1))
    print(f"           挤出自然内容 {squeezed} 条")
    s1 = monitor(step1)

    step2, qstat = apply_quota(step1, policy, pools, LIST_LEN)
    print("[step2 保量后]   " + " ".join(d.brief() for d in step2))
    print(f"           保量补入 {dict(qstat['filled'])},"
          f"超限剔除 {dict(qstat['trimmed'])}")
    s2 = monitor(step2)

    print(f"\n{'阶段':>10} {'长度':>5} {'自然占比':>9} {'广告占比':>9} "
          f"{'强插比例':>9} {'保量命中':>9} {'类别覆盖':>9} {'作者覆盖':>9} {'w=1/i分':>9}")
    print("-" * 92)
    for name, s in (("重排输出", s0), ("强插后", s1), ("保量后", s2)):
        print(f"{name:>10} {s['len']:>5} {s['biz_ratio'][BIZ_NATURAL]:>9.3f} "
              f"{s['biz_ratio']['ad']:>9.3f} {s['forced_ratio']:>9.3f} "
              f"{s['quota_hit']:>9.3f} {s['cat_cover']:>9.3f} "
              f"{s['author_cover']:>9.3f} {s['fuse_sum']:>9.4f}")

    # ---------------- 数据闭环:Redis + Kafka ----------------
    n_req = 60
    print(f"\n== {n_req} 次请求的 A/B 对比(A=关闭保量强插,B=打开)==")
    groups = {"A 对照(纯重排输出)": False, "B 实验(强插+保量)": True}
    agg = {g: collections.defaultdict(list) for g in groups}
    t0 = time.perf_counter()
    for r in range(1, n_req + 1):
        base, pools = make_request(rng, r)
        for gname, enabled in groups.items():
            if enabled:
                seq, _ = forced_injection(
                    [Doc(d.iid, d.biz, d.cat, d.author, d.fuse, d.src) for d in base],
                    [pools["operation"][0], pools["activity"][0]], [4, 9], LIST_LEN)
                seq, _ = apply_quota(seq, policy, pools, LIST_LEN)
            else:
                seq = base
            m = monitor(seq)
            a = agg[gname]
            for b in (BIZ_NATURAL,) + BIZ_LINES:
                a[f"ratio_{b}"].append(m["biz_ratio"][b])
            for k in ("forced_ratio", "quota_hit", "cat_cover",
                      "author_cover", "fuse_sum", "len"):
                a[k].append(m[k])
            if enabled:
                # 正文:关键中间信息写 Redis(实时查询与策略回溯)
                redis.setex(f"rerank:{r}", ttl=300, value={
                    "seq": [d.iid for d in seq],
                    "src": [d.src for d in seq],
                    "tag": [d.tag for d in seq],
                    "fuse": [round(d.fuse, 4) for d in seq],
                })
                # 正文:关键日志经 Kafka 异步投递,供离线训练/分析
                for pos, d in enumerate(seq, start=1):
                    kafka.send(f"req{r}", {"req": r, "pos": pos, "iid": d.iid,
                                           "biz": d.biz, "tag": d.tag})
    kafka.flush()
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"{'分组':>18} {'自然':>7} {'广告':>7} {'活动':>7} {'运营':>7} "
          f"{'强插比例':>9} {'保量命中':>9} {'类别覆盖':>9} {'作者覆盖':>9} {'w=1/i分':>9}")
    print("-" * 104)
    for gname in groups:
        a = agg[gname]
        m = lambda k: sum(a[k]) / len(a[k])            # noqa: E731
        print(f"{gname:>18} {m('ratio_natural'):>7.3f} {m('ratio_ad'):>7.3f} "
              f"{m('ratio_activity'):>7.3f} {m('ratio_operation'):>7.3f} "
              f"{m('forced_ratio'):>9.3f} {m('quota_hit'):>9.3f} "
              f"{m('cat_cover'):>9.3f} {m('author_cover'):>9.3f} "
              f"{m('fuse_sum'):>9.4f}")
    loss = 100.0 * (sum(agg["A 对照(纯重排输出)"]["fuse_sum"])
                    - sum(agg["B 实验(强插+保量)"]["fuse_sum"])) \
        / sum(agg["A 对照(纯重排输出)"]["fuse_sum"])
    print(f"-> B 组相对 A 组的 w=1/i 序列分损失:{loss:.2f}%"
          "(这是业务目标换来的自然内容体验代价)")

    print("\n== 数据闭环统计 ==")
    print(f"  [MOCK] Redis   : setex {redis.n_set} 次,当前 key 数 {len(redis.keys())}")
    probe = redis.get("rerank:7")
    print(f"           回溯 rerank:7 -> 序列 {probe['seq'][:4]}... "
          f"来源通路 {probe['src'][:4]}... 标记 {[t or '-' for t in probe['tag'][:4]]}...")
    expired = redis.get("rerank:7", now=time.time() + 600)
    print(f"           TTL=300s,模拟 600s 后再查:{expired}(过期 miss)")
    avg_batch = sum(kafka.batch_sizes) / max(1, len(kafka.batch_sizes))
    print(f"  [MOCK] Kafka   : topic={kafka.topic} 分区 {kafka.n_partition},"
          f"send 成功 {kafka.sent} 条,丢弃 {kafka.dropped} 条")
    print(f"           flush {kafka.flushes} 批,平均批大小 {avg_batch:.1f},"
          f"分区分布 {dict(sorted(kafka.per_partition.items()))}")
    print(f"  后处理 + 写入总耗时 {elapsed:.1f} ms"
          f"({elapsed / n_req:.2f} ms/请求,含两组)")

    print("\n[结论] 对应正文 §7.6:")
    print("  - 强插/保量作用在最终曝光结果上,对线上指标的影响比精排的流控更直接;")
    print("  - 执行顺序必须是「强插 → 保量补齐 → 占比上限截断」,否则上限会被强插突破;")
    print("  - 保量把自然内容占比从 1.000 压到表中数值,代价是 w=1/i 序列分下降,"
          "必须靠 A/B 量化;")
    print("  - Redis 存中间信息做策略回溯,Kafka 投日志做离线闭环,两者都要有"
          "过期/丢弃的降级口径。")


if __name__ == "__main__":
    main()
