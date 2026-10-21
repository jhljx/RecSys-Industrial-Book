"""对应正文第 7 章 §7.6「重排后处理」中的**规则化多样性打散与序列合法性修正**。

本实现做了什么
----------------
正文 §7.6 指出:后处理阶段「基于规则约束或轻量级贪心策略,在局部窗口内对排序结果进行
微调」,目标是「避免同一作者、同一内容类型或高度相似内容的连续出现」,并且要
「尽可能不破坏重排模型排序结果」。本脚本实现这条链路的四个算子:

1. `check_constraints`:序列合法性校验。三类约束
   - 相邻约束:相邻位置不得同作者 / 同类目;
   - 滑窗约束:任意长度 W 的窗口内,同一作者最多 1 个、同一类目最多 C 个;
   - 头部约束:前 H 位类目互不相同;
2. `sliding_window_fix`:**最小位移修正**。从前往后扫描,遇到违例位置就在其后面
   的候选里找第一个「插到这里合法」的物品,把它前移(其余保持相对顺序)。
   位移越小,对原排序的破坏越小 → 用 Kendall tau 距离量化破坏程度;
3. `local_window_swap`:只在半径 R 的局部窗口内做交换的变体(工业界更常见,
   因为位移被硬性限制住,排序破坏可控);
4. `backfill`:兜底补齐。若打散后序列长度不足(被约束卡住、或候选被过滤),
   从兜底池(热门/高质量池)按序补齐,保证坑位填满。

关键指标
--------
- Kendall tau 距离:两个排列中逆序对的比例,0 = 完全没动,1 = 完全反序;
- 平均位移 |原位置 - 新位置| 的均值;
- 排序分损失 = (原序列加权和 - 新序列加权和) / 原序列加权和,权重 w_i = 1/i(正文 §7.4);
- 违例数:修正前后的对比。
"""

import random

CATS = ["游戏", "美食", "剧情", "知识", "体育", "音乐"]
AUTHORS = [f"u{i:02d}" for i in range(14)]


class Item:
    """候选物品。[MOCK] 特征只保留后处理需要的 author / cat / score。"""

    __slots__ = ("iid", "author", "cat", "score")

    def __init__(self, iid, author, cat, score):
        self.iid, self.author, self.cat, self.score = iid, author, cat, score

    def __repr__(self):
        return f"{self.iid}({self.author},{self.cat},{self.score:.2f})"


# ---------------------------------------------------------------------------
# 约束定义与校验
# ---------------------------------------------------------------------------
class Constraints:
    def __init__(self, window=4, max_cat_in_window=2, head=3):
        self.window = window
        self.max_cat_in_window = max_cat_in_window
        self.head = head

    def ok_at(self, seq, pos, item):
        """把 item 放在 seq[pos](seq 为已定稿前缀,长度 == pos)是否合法。"""
        if pos > 0:
            prev = seq[pos - 1]
            if prev.author == item.author or prev.cat == item.cat:
                return False
        lo = max(0, pos - self.window + 1)
        win = seq[lo:pos]
        if any(w.author == item.author for w in win):
            return False
        if sum(1 for w in win if w.cat == item.cat) >= self.max_cat_in_window:
            return False
        if pos < self.head and any(seq[k].cat == item.cat for k in range(pos)):
            return False
        return True

    def violations(self, seq):
        """逐位置统计违例,返回 (总违例数, 各类违例计数 dict)。"""
        d = {"相邻同作者": 0, "相邻同类目": 0, "窗口内同作者": 0,
             "窗口内类目超限": 0, "头部类目重复": 0}
        for k, it in enumerate(seq):
            if k > 0:
                if seq[k - 1].author == it.author:
                    d["相邻同作者"] += 1
                if seq[k - 1].cat == it.cat:
                    d["相邻同类目"] += 1
            lo = max(0, k - self.window + 1)
            win = seq[lo:k]
            if any(w.author == it.author for w in win):
                d["窗口内同作者"] += 1
            if sum(1 for w in win if w.cat == it.cat) >= self.max_cat_in_window:
                d["窗口内类目超限"] += 1
            if k < self.head and any(seq[j].cat == it.cat for j in range(k)):
                d["头部类目重复"] += 1
        return sum(d.values()), d


# ---------------------------------------------------------------------------
# 算子 1:滑动窗口最小位移修正
# ---------------------------------------------------------------------------
def sliding_window_fix(cands, cons, target_len):
    """从前往后定坑:每个坑位取「最靠前的合法候选」,找不到则放宽(放原序第一个)。"""
    rest = list(cands)
    out, relaxed = [], 0
    while rest and len(out) < target_len:
        pick = None
        for j, it in enumerate(rest):
            if cons.ok_at(out, len(out), it):
                pick = j
                break
        if pick is None:              # 无合法候选 → 放宽约束,保证坑位不空
            pick = 0
            relaxed += 1
        out.append(rest.pop(pick))
    return out, relaxed


# ---------------------------------------------------------------------------
# 算子 2:局部窗口交换(位移上限 R)
# ---------------------------------------------------------------------------
def local_window_swap(cands, cons, target_len, radius=3):
    """只允许从当前位置往后 radius 个候选里挑,硬性限制位移。"""
    rest = list(cands)
    out, relaxed = [], 0
    while rest and len(out) < target_len:
        pick = None
        for j in range(min(radius + 1, len(rest))):
            if cons.ok_at(out, len(out), rest[j]):
                pick = j
                break
        if pick is None:
            pick = 0
            relaxed += 1
        out.append(rest.pop(pick))
    return out, relaxed


# ---------------------------------------------------------------------------
# 算子 3:兜底补齐
# ---------------------------------------------------------------------------
def backfill(seq, backup_pool, cons, target_len):
    """长度不足时从兜底池补齐,优先补合法的;补不够就无条件补(坑位不能空)。"""
    out = list(seq)
    used = {it.iid for it in out}
    forced = 0
    for it in backup_pool:
        if len(out) >= target_len:
            break
        if it.iid in used:
            continue
        if cons.ok_at(out, len(out), it):
            out.append(it)
            used.add(it.iid)
    for it in backup_pool:
        if len(out) >= target_len:
            break
        if it.iid not in used:
            out.append(it)
            used.add(it.iid)
            forced += 1
    return out, forced


# ---------------------------------------------------------------------------
# 排序破坏程度指标
# ---------------------------------------------------------------------------
def kendall_tau_distance(order_a, order_b):
    """归一化 Kendall tau 距离:逆序对数 / C(n,2)。"""
    pos = {x: i for i, x in enumerate(order_b)}
    common = [x for x in order_a if x in pos]
    n = len(common)
    if n < 2:
        return 0.0
    inv = sum(1 for i in range(n) for j in range(i + 1, n)
              if pos[common[i]] > pos[common[j]])
    return 2.0 * inv / (n * (n - 1))


def mean_shift(original, new):
    pos = {it.iid: i for i, it in enumerate(original)}
    return sum(abs(pos[it.iid] - i) for i, it in enumerate(new)) / max(1, len(new))


def weighted_sum(seq):
    """正文 §7.4 口径 w_i = 1/i 的序列分,用来衡量打散对排序分的损失。"""
    return sum(it.score / (k + 1) for k, it in enumerate(seq))


def ild_cat(seq):
    """类目维度的列表内差异度:相邻不同类目的比例。"""
    if len(seq) < 2:
        return 1.0
    return sum(1 for k in range(1, len(seq))
               if seq[k].cat != seq[k - 1].cat) / (len(seq) - 1)


def make_pool(rng, n, skew=True):
    """[MOCK] 合成候选池:精排分已降序;刻意制造作者/类目聚集(头部重复)。"""
    items = []
    for i in range(n):
        if skew and i % 5 < 3:
            author = AUTHORS[i // 5 % 3]        # 头部作者高度集中
            cat = CATS[i // 5 % 2]
        else:
            author = AUTHORS[rng.randrange(len(AUTHORS))]
            cat = CATS[rng.randrange(len(CATS))]
        items.append(Item(f"i{i:03d}", author, cat, 1.0 - i * 0.012
                          + rng.uniform(-0.004, 0.004)))
    items.sort(key=lambda it: -it.score)
    return items


def main():
    random.seed(42)
    rng = random.Random(42)

    cons = Constraints(window=4, max_cat_in_window=2, head=3)
    target = 12

    print("== §7.6 规则化多样性打散 + 序列合法性修正 ==")
    print(f"约束:相邻不同作者且不同类目;窗口 W={cons.window} 内同作者<=1、"
          f"同类目<={cons.max_cat_in_window};头部 {cons.head} 位类目互不相同")
    print(f"输出坑位 target={target}\n")

    # ---------------- 单次请求细节 ----------------
    pool = make_pool(rng, 30)
    raw = pool[:target]
    print("[原始精排序列] (按精排分降序直接截断)")
    print("  " + " ".join(f"{it.cat}/{it.author}" for it in raw))
    nv_raw, detail_raw = cons.violations(raw)
    print(f"  违例总数 {nv_raw}:" + ", ".join(f"{k}={v}" for k, v in detail_raw.items()))
    print(f"  w=1/i 加权分 {weighted_sum(raw):.4f},相邻异类目比例 {ild_cat(raw):.3f}")

    fixed, relaxed = sliding_window_fix(pool, cons, target)
    print("\n[滑窗最小位移修正]")
    print("  " + " ".join(f"{it.cat}/{it.author}" for it in fixed))
    nv_fix, detail_fix = cons.violations(fixed)
    print(f"  违例总数 {nv_fix},放宽次数 {relaxed}")
    print(f"  w=1/i 加权分 {weighted_sum(fixed):.4f},相邻异类目比例 {ild_cat(fixed):.3f}")

    swapped, relaxed_s = local_window_swap(pool, cons, target, radius=3)
    print("\n[局部窗口交换 R=3]")
    print("  " + " ".join(f"{it.cat}/{it.author}" for it in swapped))
    nv_sw, _ = cons.violations(swapped)
    print(f"  违例总数 {nv_sw},放宽次数 {relaxed_s}")

    # ---------------- 多次请求的统计表 ----------------
    print("\n== 100 次请求的统计:打散强度 vs 排序破坏 ==")
    print(f"{'策略':>22} {'违例数':>7} {'Kendall tau':>12} {'平均位移':>9} "
          f"{'w=1/i分损失':>12} {'相邻异类目':>10} {'放宽次数':>9}")
    print("-" * 90)
    strategies = [
        ("原始精排(不打散)", None),
        ("滑窗最小位移", ("fix", None)),
        ("局部交换 R=1", ("swap", 1)),
        ("局部交换 R=3", ("swap", 3)),
        ("局部交换 R=6", ("swap", 6)),
    ]
    acc = {name: {"nv": [], "tau": [], "shift": [], "loss": [], "ild": [],
                  "relax": []} for name, _ in strategies}
    for _ in range(100):
        pool = make_pool(rng, 30)
        raw = pool[:target]
        base_score = weighted_sum(raw)
        raw_ids = [it.iid for it in raw]
        for name, spec in strategies:
            if spec is None:
                seq, relaxed = raw, 0
            elif spec[0] == "fix":
                seq, relaxed = sliding_window_fix(pool, cons, target)
            else:
                seq, relaxed = local_window_swap(pool, cons, target, radius=spec[1])
            nv, _ = cons.violations(seq)
            a = acc[name]
            a["nv"].append(nv)
            a["tau"].append(kendall_tau_distance(raw_ids, [it.iid for it in seq]))
            a["shift"].append(mean_shift(pool, seq))
            a["loss"].append(100.0 * (base_score - weighted_sum(seq)) / base_score)
            a["ild"].append(ild_cat(seq))
            a["relax"].append(relaxed)
    for name, _ in strategies:
        a = acc[name]
        m = lambda k: sum(a[k]) / len(a[k])    # noqa: E731
        print(f"{name:>22} {m('nv'):>7.2f} {m('tau'):>12.4f} {m('shift'):>9.2f} "
              f"{m('loss'):>11.2f}% {m('ild'):>10.3f} {m('relax'):>9.2f}")

    # ---------------- 兜底补齐 ----------------
    print("\n== 兜底补齐(backfill):候选被约束卡住导致坑位不足时 ==")
    tiny = make_pool(rng, 7)                       # 候选只有 7 个,目标 12 个坑
    short, _ = sliding_window_fix(tiny, cons, target)
    backup = make_pool(random.Random(7), 10)
    for it in backup:
        it.iid = "b" + it.iid[1:]
    filled, forced = backfill(short, backup, cons, target)
    nv_f, _ = cons.violations(filled)
    print(f"  打散后长度 {len(short)} < target {target} → 需要补 {target - len(short)} 个")
    print(f"  兜底池大小 {len(backup)},无条件强补(违约束仍补)次数 {forced}")
    print(f"  补齐后长度 {len(filled)},违例数 {nv_f}")
    print("  补齐后序列:" + " ".join(
        ("*" if it.iid.startswith("b") else "") + f"{it.cat}/{it.author}"
        for it in filled) + "   (* = 兜底池物品)")

    print("\n[结论] 对应正文 §7.6:")
    print("  - 打散放在后处理、以「最小位移」方式做,是为了尽量不破坏重排模型的排序结果;")
    print("  - 位移半径 R 越大,违例清得越干净,但 Kendall tau / 排序分损失同步变大;")
    print("  - R=1 几乎不动排序,代价是残留违例;工业界通常取 R=2~4 的折中;")
    print("  - 任何约束都必须有「放宽 / 兜底补齐」兜底分支,否则坑位会空,直接掉曝光。")


if __name__ == "__main__":
    main()
