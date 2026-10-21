"""对应正文第 7 章 §7.4「序列评估与排序」中的**序列排序与选择策略**(Critic 如何选序)。

本实现做了什么
----------------
正文 §7.4 指出:序列生成「由多种方法并行提供候选序列」,序列评估「相对固定」,
并从强化学习视角把生成器看作 Actor、评估器看作 Critic。本脚本把这条链路跑通:

1. **多路 Actor 并行生成候选序列**(对应 §7.3):
   - greedy    :按融合分 f(x) 直接贪心 Top-N(正文 code_greedy_gen_init 的无噪声版);
   - gauss     :融合分加高斯噪声后贪心,重复多次(正文 code_greedy_gaussian);
   - beam      :beam search,beam 内按「前缀序列价值」扩展(正文 code_beam_search_core);
   - swap      :在 greedy 序列上做局部两两交换扰动;
   - mmr_like  :相关性-多样性加权贪心(为选序池引入结构不同的序列)。
2. **一个 Critic 给所有候选序列打分**:评估器用正文 §7.4 第三种口径
       p(item_i) = p0 * prod_{j<i} p_slide(x_j),  value = sum_i p(item_i) f(x_i)
   其中 f / p_slide 由一个带噪声的「已学模型」给出(真实环境参数 + 扰动),
   以模拟 Critic 不完美的现实情况。
3. **对比 5 种选序策略**:
   - argmax         :直接取 Critic 分最高的序列;
   - soft_penalty   :Critic 分乘以 (1 - beta * 违例数),软惩罚而非硬过滤;
   - constrained    :先过硬约束(类目连续不超过 2、头部 3 位类目不重复),再 argmax;
   - softmax_sample :按 Critic 分做温度 softmax 采样(探索,对应 RL 的策略采样);
   - oracle/worst   :上界与下界参照。
   指标:真实序列价值、相对 oracle 的 regret%、约束违例率、被选中序列来自哪路 Actor。

关键公式(§7.4)
---------------
    value = sum_{i=1..N} w_i f(x_i),  w_i = 1/i 或 e^{-(i-1)}
    p(item_i) = p0 (i=1) | p0 * prod_{j<i} p_slide(x_j) (i>1)
    value_i = p(item_i) * f(x_i),  value = sum_i value_i
"""

import math
import random

import numpy as np

D = 6            # 物品特征维度
N_POOL = 30      # 候选池大小
LIST_LEN = 8     # 输出序列长度
P0 = 0.95
N_CAT = 5
CATS = ["游戏", "美食", "剧情", "知识", "体育"]


# ---------------------------------------------------------------------------
# [MOCK] 真实环境 + 不完美 Critic
# ---------------------------------------------------------------------------
class SlideEnv:
    """[MOCK] 上下滑环境:给出 ground-truth 的 f(x) 与 p_slide(x)。"""

    def __init__(self, rng, noise=0.0):
        self.w_f = rng.normal(0, 1.0, D)
        self.w_s = rng.normal(0, 0.8, D)
        self.b_s = 0.6
        if noise > 0:                      # Critic 版本:参数带估计误差
            self.w_f = self.w_f + rng.normal(0, noise, D)
            self.w_s = self.w_s + rng.normal(0, noise, D)

    def f(self, x):
        return 20.0 + 25.0 * math.tanh(float(x @ self.w_f))

    def p_slide(self, x):
        return 1.0 / (1.0 + math.exp(-float(x @ self.w_s) - self.b_s))

    def value(self, seq_idx, feats):
        """正文第三种口径的序列价值。"""
        p, total = P0, 0.0
        for k, i in enumerate(seq_idx):
            if k > 0:
                p *= self.p_slide(feats[seq_idx[k - 1]])
            total += p * self.f(feats[i])
        return total

    def weighted_sum(self, seq_idx, feats, decay="inv"):
        """正文第一种口径:位置衰减加权和。"""
        total = 0.0
        for k, i in enumerate(seq_idx, start=1):
            w = 1.0 / k if decay == "inv" else math.exp(-(k - 1))
            total += w * self.f(feats[i])
        return total


# ---------------------------------------------------------------------------
# 多路序列生成器(Actor)
# ---------------------------------------------------------------------------
def gen_greedy(scores):
    return [tuple(np.argsort(-scores)[:LIST_LEN].tolist())]


def gen_gaussian(scores, rng, n=12, sigma=0.25):
    """正文 code_greedy_gaussian:融合分加高斯噪声再贪心。"""
    out = []
    scale = float(np.std(scores)) + 1e-9
    for _ in range(n):
        noisy = scores + rng.normal(0, sigma * scale, scores.shape)
        out.append(tuple(np.argsort(-noisy)[:LIST_LEN].tolist()))
    return out


def gen_beam(scores, feats, critic, beam_w=4, top_k=12):
    """beam search:按 Critic 的前缀序列价值扩展,保留 beam_w 条。"""
    cand = list(np.argsort(-scores)[:top_k])
    beams = [((), 0.0)]
    for _ in range(LIST_LEN):
        nxt = []
        for seq, _ in beams:
            for i in cand:
                if i in seq:
                    continue
                new = seq + (int(i),)
                nxt.append((new, critic.value(new, feats)))
        nxt.sort(key=lambda t: -t[1])
        beams = nxt[:beam_w]
    return [seq for seq, _ in beams]


def gen_swap(scores, rng, n=12):
    base = list(np.argsort(-scores)[:LIST_LEN])
    out = []
    for _ in range(n):
        seq = list(base)
        for _ in range(rng.randint(1, 3)):
            a, b = rng.randint(0, LIST_LEN), rng.randint(0, LIST_LEN)
            seq[a], seq[b] = seq[b], seq[a]
        out.append(tuple(int(i) for i in seq))
    return out


def gen_mmr_like(scores, feats, lambdas=(0.9, 0.7, 0.5)):
    """相关性-多样性加权贪心,给选序池注入结构不同的序列。"""
    norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    rel = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    out = []
    for lam in lambdas:
        sel, rest = [], list(range(len(scores)))
        while len(sel) < LIST_LEN:
            best, best_s = None, -1e18
            for i in rest:
                div = max((sim[i, j] for j in sel), default=0.0)
                s = lam * rel[i] - (1 - lam) * div
                if s > best_s:
                    best, best_s = i, s
            sel.append(best)
            rest.remove(best)
        out.append(tuple(sel))
    return out


# ---------------------------------------------------------------------------
# 硬约束检查(§7.6 的前置校验在选序阶段也会用到)
# ---------------------------------------------------------------------------
def violates(seq, cats, max_run=2, head=3):
    run = 1
    for k in range(1, len(seq)):
        run = run + 1 if cats[seq[k]] == cats[seq[k - 1]] else 1
        if run > max_run:
            return True
    head_cats = [cats[i] for i in seq[:head]]
    return len(set(head_cats)) < len(head_cats)


# ---------------------------------------------------------------------------
# 选序策略
# ---------------------------------------------------------------------------
def n_violations(seq, cats, max_run=2, head=3):
    """违例计数:连续同类目超过 max_run 的次数 + 头部重复类目个数。"""
    cnt, run = 0, 1
    for k in range(1, len(seq)):
        run = run + 1 if cats[seq[k]] == cats[seq[k - 1]] else 1
        if run > max_run:
            cnt += 1
    head_cats = [cats[i] for i in seq[:head]]
    return cnt + (len(head_cats) - len(set(head_cats)))


def pick_argmax(cscore):
    return int(np.argmax(cscore))


def pick_soft_penalty(pool, cscore, cats, beta=0.06):
    """软惩罚:score * (1 - beta * 违例数),不做硬过滤。"""
    adj = np.array([c * (1.0 - beta * n_violations(s, cats))
                    for s, c in zip(pool, cscore)])
    return int(np.argmax(adj))


def pick_constrained(pool, cscore, cats):
    """硬过滤:只在完全不违例的序列里取 Critic 分最高者;全违例则退化为 argmax。"""
    order = np.argsort(-cscore)
    for i in order:
        if not violates(pool[i], cats):
            return int(i)
    return int(order[0])


def pick_softmax(cscore, rng, temp=0.05):
    z = (cscore - cscore.max()) / (temp * (abs(cscore.max()) + 1e-9))
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(cscore), p=p))


def main():
    random.seed(42)
    np.random.seed(42)

    rng = np.random.RandomState(42)
    env = SlideEnv(rng)
    critic = SlideEnv(np.random.RandomState(7), noise=0.0)
    critic.w_f, critic.w_s = env.w_f.copy(), env.w_s.copy()
    # Critic 带估计误差:正文说评估器是「学出来的」,不可能完全等于真实环境
    critic.w_f += np.random.RandomState(7).normal(0, 0.18, D)
    critic.w_s += np.random.RandomState(8).normal(0, 0.18, D)

    print("== 序列生成(多路 Actor)+ 序列评估(Critic)+ 选序策略对比 ==")
    print(f"候选池 N={N_POOL},输出序列长度 L={LIST_LEN},类目数 {N_CAT},p0={P0}")
    print("Critic = 真实环境参数 + N(0, 0.18^2) 扰动,模拟评估器的估计误差\n")

    # ---------- 单次请求的细节展示 ----------
    feats = rng.normal(0, 1.0, (N_POOL, D))
    cats = [CATS[rng.randint(N_CAT)] for _ in range(N_POOL)]
    scores = np.array([critic.f(x) for x in feats])     # 精排融合分(Critic 的单点分)

    pool, actors = [], []
    for name, seqs in (("greedy", gen_greedy(scores)),
                       ("gauss", gen_gaussian(scores, rng)),
                       ("beam", gen_beam(scores, feats, critic)),
                       ("swap", gen_swap(scores, rng)),
                       ("mmr_like", gen_mmr_like(scores, feats))):
        for s in seqs:
            pool.append(s)
            actors.append(name)
    seen, upool, uactors = set(), [], []
    for s, a in zip(pool, actors):
        if s not in seen:
            seen.add(s)
            upool.append(s)
            uactors.append(a)
    pool, actors = upool, uactors

    cscore = np.array([critic.value(s, feats) for s in pool])
    tscore = np.array([env.value(s, feats) for s in pool])

    print("[单次请求] 各路 Actor 产出的候选序列统计")
    print(f"{'Actor':>10} {'条数':>5} {'Critic分最高':>13} {'真实价值最高':>13} "
          f"{'违约束占比':>11}")
    for a in ("greedy", "gauss", "beam", "swap", "mmr_like"):
        idx = [i for i, x in enumerate(actors) if x == a]
        bad = sum(violates(pool[i], cats) for i in idx)
        print(f"{a:>10} {len(idx):>5} {cscore[idx].max():>13.3f} "
              f"{tscore[idx].max():>13.3f} {100.0 * bad / len(idx):>10.1f}%")
    print(f"{'合计':>10} {len(pool):>5} {cscore.max():>13.3f} "
          f"{tscore.max():>13.3f} "
          f"{100.0 * sum(violates(s, cats) for s in pool) / len(pool):>10.1f}%")

    best = pick_argmax(cscore)
    print(f"\n  Critic argmax 选中:{pool[best]}  来自 {actors[best]}")
    print(f"  类目序列:{'->'.join(cats[i] for i in pool[best])}")
    print(f"  Critic 分 {cscore[best]:.3f} / 真实价值 {tscore[best]:.3f} / "
          f"oracle {tscore.max():.3f}")

    # ---------- 多次请求的策略对比 ----------
    n_req = 120
    names = ["oracle(真实价值上界)", "argmax", "soft_penalty beta=0.06",
             "constrained(硬过滤)", "softmax_sample T=0.05", "worst(池内最差)",
             "greedy单路(无选序)"]
    agg = {k: {"val": [], "regret": [], "bad": 0, "from": {}} for k in names}
    for _ in range(n_req):
        feats = rng.normal(0, 1.0, (N_POOL, D))
        cats = [CATS[rng.randint(N_CAT)] for _ in range(N_POOL)]
        scores = np.array([critic.f(x) for x in feats])
        pool, actors = [], []
        for name, seqs in (("greedy", gen_greedy(scores)),
                           ("gauss", gen_gaussian(scores, rng)),
                           ("beam", gen_beam(scores, feats, critic)),
                           ("swap", gen_swap(scores, rng)),
                           ("mmr_like", gen_mmr_like(scores, feats))):
            for s in seqs:
                pool.append(s)
                actors.append(name)
        cscore = np.array([critic.value(s, feats) for s in pool])
        tscore = np.array([env.value(s, feats) for s in pool])
        oracle = float(tscore.max())

        picks = {
            "oracle(真实价值上界)": int(np.argmax(tscore)),
            "argmax": pick_argmax(cscore),
            "soft_penalty beta=0.06": pick_soft_penalty(pool, cscore, cats),
            "constrained(硬过滤)": pick_constrained(pool, cscore, cats),
            "softmax_sample T=0.05": pick_softmax(cscore, rng),
            "worst(池内最差)": int(np.argmin(tscore)),
            "greedy单路(无选序)": actors.index("greedy"),
        }
        for k, i in picks.items():
            agg[k]["val"].append(float(tscore[i]))
            agg[k]["regret"].append(100.0 * (oracle - tscore[i]) / abs(oracle))
            agg[k]["bad"] += int(violates(pool[i], cats))
            agg[k]["from"][actors[i]] = agg[k]["from"].get(actors[i], 0) + 1

    print(f"\n[{n_req} 次请求] 选序策略对比")
    print(f"{'策略':>22} {'真实序列价值':>13} {'regret%':>9} "
          f"{'约束违例率':>11} {'主要来源Actor':>16}")
    print("-" * 78)
    for k in names:
        s = agg[k]
        top_from = max(s["from"].items(), key=lambda t: t[1])
        print(f"{k:>22} {np.mean(s['val']):>13.3f} "
              f"{np.mean(s['regret']):>9.2f} "
              f"{100.0 * s['bad'] / n_req:>10.1f}% "
              f"{top_from[0] + '(%d%%)' % round(100 * top_from[1] / n_req):>16}")

    # ---------- 评估口径对选序的影响 ----------
    print("\n[口径对比] 换成正文第一种口径(位置衰减加权和)来选序,regret 如何变化")
    print(f"{'Critic 口径':>26} {'regret%':>9} {'top-1命中率':>12}")
    for label, fn in (("sum_i p(item_i) f(x_i)",
                       lambda s, f: critic.value(s, f)),
                      ("sum_i f(x_i)/i", lambda s, f: critic.weighted_sum(s, f, "inv")),
                      ("sum_i e^{-(i-1)} f(x_i)",
                       lambda s, f: critic.weighted_sum(s, f, "exp")),
                      ("sum_i f(x_i)(无位置权重)",
                       lambda s, f: sum(critic.f(f[i]) for i in s))):
        regs, hits = [], 0
        for _ in range(60):
            feats = rng.normal(0, 1.0, (N_POOL, D))
            scores = np.array([critic.f(x) for x in feats])
            pool = (gen_greedy(scores) + gen_gaussian(scores, rng)
                    + gen_swap(scores, rng) + gen_mmr_like(scores, feats))
            cs = np.array([fn(s, feats) for s in pool])
            ts = np.array([env.value(s, feats) for s in pool])
            pick = int(np.argmax(cs))
            regs.append(100.0 * (ts.max() - ts[pick]) / abs(ts.max()))
            hits += int(pick == int(np.argmax(ts)))
        print(f"{label:>26} {np.mean(regs):>9.2f} {100.0 * hits / 60:>11.1f}%")

    print("\n[结论] 对应正文 §7.4:")
    print("  - 多路 Actor 并行生成 + 单一 Critic 选序,比任何单路生成都好(见 greedy 单路一行);")
    print("  - Critic 有估计误差时,argmax 会放大误差,把硬约束放在选序里能稳定体验;")
    print("  - softmax 采样引入探索,代价是 regret 上升,线上通常只给小流量;")
    print("  - 评估口径决定选序上限:忽略位置偏置的加和口径 regret 最大。")


if __name__ == "__main__":
    main()
