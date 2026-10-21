"""强插(Force Insertion)策略:运营规则在曝光前最后一步的确定性落地。

对应正文第 8 章 §8.6.3「强插策略」。

正文要点(本实现逐条复现):
1. 强插允许运营人员通过后台配置规则,在**特定用户群体、特定刷次、特定坑位**插入指定内容;
   正文给的三个例子分别对应本实现的三条规则:
     - 指定白名单创作者的视频强插至第 5 个位置;
     - 指定主播的直播间强插至用户**首次刷新**页面时的第 3 个位置;
     - 针对特定人群包插入活动广告内容;
2. 对影响力极大的运营活动(头部明星直播、大促),平台可能采用**全局强插**,对所有用户统一插入;
3. 强插发生在最终曝光前的**最后阶段**,优先级**高于**混排排序结果,
   因此能保证运营策略稳定执行;
4. 代价是挤占模型选出的坑位 -> 本实现量化这一代价(模型价值损失、各业务占比漂移)。

关键公式/口径:
  强插前 top1 序列 S = [s_1, ..., s_L](由 §8.4 序列排序给出)
  对每条命中的规则 (target_pos p, item x):把 x 放到位置 p,原 p 及其后元素整体后移一位,
  序列尾部截断回长度 L  ->  被挤出的是价值最低(位置最靠后)的那个模型元素。
  模型价值损失 Delta = sum_i gamma^(i-1) * v(s_i) - sum_i gamma^(i-1) * v(s'_i)
  其中 gamma^(i-1) 为 §8.4 的位置衰减因子(本实现 gamma = 0.85)。

说明:正文强调「强插优先级高于排序结果」,所以本实现中规则一旦命中就**无条件**执行,
不做价值比较;只在事后统计代价,供读者理解强插为什么必须受配额约束。

[MOCK] 运营后台配置、人群包服务、刷次计数器都是进程内的假组件,不访问任何外部服务。
"""

import random
import statistics

SEQ_LEN = 8
GAMMA = 0.85  # 位置衰减因子,与 §8.4 口径一致

# ----------------------------------------------------- [MOCK] 运营后台 + 人群包服务

WHITELIST_AUTHORS = {"author_7", "author_19"}      # 白名单创作者
CAMPAIGN_ANCHORS = {"anchor_3"}                    # 大型活动主播
TARGET_CROWD = "crowd_newuser_30d"                 # 特定人群包


def make_user(uid):
    """[MOCK] 合成一个用户:人群包标签 + 当前刷次(refresh index)。"""
    return {
        "uid": uid,
        "crowds": ({TARGET_CROWD} if random.random() < 0.25 else set()),
        "refresh": random.choice([1, 1, 2, 3, 4, 5]),  # 1 表示首刷
    }


def make_top1_sequence():
    """[MOCK] 合成一条混排 §8.4 已经选出的 top1 序列(长度 SEQ_LEN)。"""
    seq = []
    for i in range(SEQ_LEN):
        biz = random.choices(["video", "live", "ad"], weights=[0.78, 0.10, 0.12])[0]
        seq.append({
            "id": f"{biz[0].upper()}{i}",
            "biz": biz,
            # 模型价值随位置递减:top1 序列本身已按价值排好
            "value": round(max(0.05, random.gauss(1.0 - 0.05 * i, 0.12)), 4),
            "forced": False,
        })
    return seq


def make_operation_items():
    """[MOCK] 运营侧待强插的素材池。"""
    return {
        "whitelist_video": {"id": "OP_V", "biz": "video", "value": 0.42,
                            "author": "author_7", "forced": True},
        "campaign_live": {"id": "OP_L", "biz": "live", "value": 0.55,
                          "anchor": "anchor_3", "forced": True},
        "campaign_ad": {"id": "OP_A", "biz": "ad", "value": 0.30,
                        "forced": True},
        "global_star_live": {"id": "OP_STAR", "biz": "live", "value": 0.80,
                             "anchor": "anchor_star", "forced": True},
    }


# ----------------------------------------------------------------- 规则引擎

def build_rules(enable_global=False):
    """正文 §8.6.3 列出的三条规则(+ 可选的全局强插)。

    每条规则 = (名称, 命中条件 fn(user) -> bool, 目标坑位 position, 素材 key)。
    position 为 1-based,与正文「第 5 个位置 / 第 3 个位置」的说法一致。
    """
    rules = [
        ("白名单创作者视频 -> 第5坑",
         lambda u: True,  # 白名单由素材侧决定,对所有用户生效
         5, "whitelist_video"),
        ("活动主播直播间 -> 首刷第3坑",
         lambda u: u["refresh"] == 1,
         3, "campaign_live"),
        ("特定人群包 -> 活动广告第2坑",
         lambda u: TARGET_CROWD in u["crowds"],
         2, "campaign_ad"),
    ]
    if enable_global:
        rules.insert(0, ("全局强插:头部明星直播 -> 第1坑",
                         lambda u: True, 1, "global_star_live"))
    return rules


def apply_force_insertion(seq, user, rules, pool):
    """执行强插:命中规则无条件插入,尾部截断保持长度不变。

    返回 (新序列, 命中的规则名列表, 被挤出的元素列表)。
    """
    out = list(seq)
    hit, evicted = [], []
    for name, cond, pos, key in rules:
        if not cond(user):
            continue
        item = dict(pool[key])
        idx = min(pos - 1, len(out))     # 目标坑位超出长度就追加到末尾
        out.insert(idx, item)
        evicted.append(out.pop())        # 整体后移 -> 挤掉最后一个
        hit.append(name)
    return out, hit, evicted


def seq_value(seq):
    """位置衰减后的序列价值(§8.4 口径)。"""
    return sum(GAMMA ** i * it["value"] for i, it in enumerate(seq))


# --------------------------------------------------------------------- 演示

def demo_single_request():
    """demo1:单请求上逐条展示三类规则命中后的序列变化。"""
    print("=" * 84)
    print("demo1  正文 §8.6.3 三条强插规则在一次请求上的效果")
    print("=" * 84)
    pool = make_operation_items()
    user = {"uid": "u_demo", "crowds": {TARGET_CROWD}, "refresh": 1}
    seq = make_top1_sequence()
    rules = build_rules()

    print(f"用户: refresh={user['refresh']}(首刷)  人群包={sorted(user['crowds'])}")
    print(f"{'阶段':<30}{'序列(坑位1..8)':<46}{'序列价值':>10}")
    print("-" * 84)

    def fmt(s):
        return " ".join(("*" + it["id"]) if it["forced"] else it["id"] for it in s)

    print(f"{'混排 top1(强插前)':<30}{fmt(seq):<46}{seq_value(seq):>10.4f}")
    cur = seq
    for rule in rules:
        cur, hit, ev = apply_force_insertion(cur, user, [rule], pool)
        if hit:
            label = rule[0]
            print(f"{label:<30}{fmt(cur):<46}{seq_value(cur):>10.4f}")
    print("-" * 84)
    print(f"累计价值损失 {seq_value(seq) - seq_value(cur):+.4f}"
          f"(前缀 * 的是强插素材,3 个模型元素被挤出队尾)")
    print("强插按配置顺序逐条执行,后插入的规则可能把先插入的素材再往后推一位——")
    print("工程上因此需要约定规则优先级,正文的做法是把强插放在曝光前最后一步统一处理。")
    print()


def demo_cost_of_insertion(n_req=4000):
    """demo2:强插的系统性代价——模型价值损失与业务占比漂移。"""
    print("=" * 84)
    print("demo2  强插代价:模型价值损失 / 各业务曝光占比漂移(4000 次请求)")
    print("=" * 84)
    pool = make_operation_items()
    for enable_global, label in ((False, "三条定向规则"), (True, "再叠加全局强插")):
        rules = build_rules(enable_global)
        losses, before_cnt, after_cnt, n_hit = [], {}, {}, 0
        forced_exp = 0
        for i in range(n_req):
            user = make_user(f"u{i}")
            seq = make_top1_sequence()
            new, hit, ev = apply_force_insertion(seq, user, rules, pool)
            if hit:
                n_hit += 1
            losses.append(seq_value(seq) - seq_value(new))
            for it in seq:
                before_cnt[it["biz"]] = before_cnt.get(it["biz"], 0) + 1
            for it in new:
                after_cnt[it["biz"]] = after_cnt.get(it["biz"], 0) + 1
                if it["forced"]:
                    forced_exp += 1
        tot = n_req * SEQ_LEN
        print(f"--- {label}")
        print(f"{'指标':<24}{'数值':>12}")
        print(f"{'命中强插的请求占比':<24}{n_hit / n_req:>12.2%}")
        print(f"{'强插素材曝光占比':<24}{forced_exp / tot:>12.2%}")
        print(f"{'平均序列价值损失':<24}{statistics.fmean(losses):>12.4f}")
        print(f"{'最大序列价值损失':<24}{max(losses):>12.4f}")
        print(f"{'业务':<12}{'强插前占比':>12}{'强插后占比':>12}{'漂移':>10}")
        for biz in ("video", "live", "ad"):
            b = before_cnt.get(biz, 0) / tot
            a = after_cnt.get(biz, 0) / tot
            print(f"{biz:<12}{b:>12.2%}{a:>12.2%}{a - b:>+10.2%}")
        print()
    print("全局强插把一个高价值坑位无条件让给运营素材,直播占比被整体抬高、")
    print("视频占比被压低,平均价值损失显著上升:这正是正文说的「优先级高于排序结果」的代价。")
    print("因此生产环境必须给强插配额(每刷最多几条、整体曝光占比上限),")
    print("并与 §8.6.1 的 Load 调控联动,避免运营规则把业务占比推出安全区间。")
    print()


def demo_priority_vs_quota(n_req=4000):
    """demo3:强插 vs 配额约束——无条件强插 / 带每刷配额 / 带 Load 上界。"""
    print("=" * 84)
    print("demo3  强插配额:无条件 vs 每刷最多1条 vs 广告占比硬上界 15%")
    print("=" * 84)
    pool = make_operation_items()
    rules = build_rules(enable_global=True)
    print(f"{'策略':<26}{'强插曝光占比':>14}{'广告占比':>10}{'平均价值损失':>14}{'规则达成率':>12}")
    print("-" * 84)
    for mode in ("unconditional", "quota1", "ad_cap"):
        losses, forced_exp, ad_cnt, want, got = [], 0, 0, 0, 0
        for i in range(n_req):
            user = make_user(f"u{i}")
            seq = make_top1_sequence()
            fired = [r for r in rules if r[1](user)]
            want += len(fired)
            if mode == "quota1":
                fired = fired[:1]                      # 每刷最多执行 1 条
            elif mode == "ad_cap":
                # 广告硬上界:如果强插会让本序列广告数超过 15%*8≈1.2 条,则放弃广告类强插
                base_ad = sum(1 for it in seq if it["biz"] == "ad")
                fired = [r for r in fired
                         if not (pool[r[3]]["biz"] == "ad" and base_ad >= 1)]
            new, hit, ev = apply_force_insertion(seq, user, fired, pool)
            got += len(hit)
            losses.append(seq_value(seq) - seq_value(new))
            forced_exp += sum(1 for it in new if it["forced"])
            ad_cnt += sum(1 for it in new if it["biz"] == "ad")
        tot = n_req * SEQ_LEN
        name = {"unconditional": "无条件强插(正文语义)",
                "quota1": "每刷最多 1 条",
                "ad_cap": "广告占比硬上界 15%"}[mode]
        print(f"{name:<26}{forced_exp / tot:>14.2%}{ad_cnt / tot:>10.2%}"
              f"{statistics.fmean(losses):>14.4f}{got / want:>12.2%}")
    print("-" * 84)
    print("无条件强插规则达成率 100%,但代价最大(平均价值损失 0.3398);")
    print("每刷最多 1 条把损失压到 0.0086(约 1/40),代价是只有 38.96% 的运营规则被执行;")
    print("广告占比硬上界 15% 则在不牺牲非广告规则(达成率 93.77%)的前提下")
    print("把广告占比压回安全区间。工业界的取舍通常是:高优活动走无条件强插,")
    print("长尾运营规则走配额与排队,再由 Load 调控兜住业务占比底线。")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_single_request()
    demo_cost_of_insertion()
    demo_priority_vs_quota()
