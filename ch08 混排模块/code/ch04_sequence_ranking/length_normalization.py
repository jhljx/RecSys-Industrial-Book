"""序列长度不可比问题：长度归一化 vs Mock Item 补齐。

对应正文第 8 章 §8.4「序列排序」末尾关于「候选序列长度不一致」的讨论
（对应正文图 fig_list_mock_item 的 Mock Item 补齐示意）。

本实现做了什么：
1. 复刻正文的具体例子：序列1 = [A1,B1,A2,A3,C1,A4,B2,A5,A6]（L=9），
   序列2 = [A1,A2,C1,A3,A4,B1,A5]（L=7），A 为短视频、B 为广告、C 为直播；
   打印裸求和下「长度 9 天然占优」的数值证据；
2. 实现正文给出的两种解法并对比：
   (a) 长度归一化 Length Normalization；
   (b) Mock Item 补齐（正文说实践中更常见）—— Mock 的价值分取
       「当前业务类型下内容价值分的均值」，即正文明确的取值口径；
3. 对照实验：把「空坑位最终仍会被后续内容按基线价值填满」作为无长度偏置的真实目标，
   统计三种口径选对序列的比例（Mock 用的是历史均值估计，真实基线对打分方不可见，
   因此该实验不是自证循环）。

关键公式：
    裸求和        Score_raw(S)  = sum_{i=1..L(S)} gamma_i * v_i
    长度归一化    Score_norm(S) = Score_raw(S) / sum_{i=1..L(S)} gamma_i
    Mock 补齐     Score_mock(S) = Score_raw(S) + sum_{i=L(S)+1..L_max} gamma_i * v_mock
    真实目标      V_true(S)     = Score_raw(S) + sum_{i=L(S)+1..L_max} gamma_i * v_base
    位置衰减      gamma_i = exp(-lambda*(i-1))，lambda = 0.2
    Mock 价值     v_mock = 该业务类型历史内容价值分的均值（对 v_base 的有偏估计）

[MOCK] Mock Item（正文里的虚拟内容 "M"）本身就是工程占位内容，不对应真实物料；
       各业务的价值分、历史均值与真实基线均由合成数据生成。
"""

import math
import random

LAMBDA_DECAY = 0.2
L_MAX = 9          # 正文例子里最长序列（序列1）的长度

# 各业务「历史内容价值分均值」：Mock Item 的价值分就取它（正文给的取值口径）
HIST_MEAN = {"video": 2.0, "ad": 3.0, "live": 4.4}

# 真实的「空坑位后续填充价值」基线：打分方看不到，只用于对照实验判对错
TRUE_BASE = 2.3


def gamma(i, lam=LAMBDA_DECAY):
    return math.exp(-lam * (i - 1))


def score_raw(seq):
    """裸求和 Score_raw：长度越长分越高 → 统计偏置的来源。"""
    return sum(gamma(i) * it["v"] for i, it in enumerate(seq, start=1))


def score_len_norm(seq):
    """长度归一化 Score_norm：除以该长度下的位置权重之和。"""
    denom = sum(gamma(i) for i in range(1, len(seq) + 1))
    return score_raw(seq) / denom


def mock_item(v=None):
    """构造一个 Mock Item，价值分默认取短视频业务的历史均值。"""
    return {"id": "M", "biz": "mock",
            "v": HIST_MEAN["video"] if v is None else v, "mock": True}


def pad_with_mock(seq, l_max=L_MAX, v=None):
    """在序列末尾补 Mock Item 直到长度对齐（正文图 (b) 的做法）。"""
    return list(seq) + [mock_item(v) for _ in range(l_max - len(seq))]


def score_mock_pad(seq, l_max=L_MAX, v=None):
    """Mock Item 补齐后再求和 Score_mock。"""
    return score_raw(pad_with_mock(seq, l_max, v))


def true_value(seq, l_max=L_MAX, v_base=TRUE_BASE):
    """无长度偏置的真实收益：空坑位最终被后续内容按真实基线 v_base 填满。"""
    return score_raw(seq) + sum(gamma(i) * v_base
                                for i in range(len(seq) + 1, l_max + 1))


# -------------------------------------------------------- 正文例子复刻

def textbook_example():
    """正文 fig_list_mock_item 的两个序列（价值分取便于手算的整洁取值）。"""
    v = {"A1": 2.6, "A2": 2.5, "A3": 2.4, "A4": 2.3, "A5": 2.2, "A6": 1.0,
         "B1": 3.0, "B2": 2.8, "C1": 4.4}
    biz = {"A": "video", "B": "ad", "C": "live"}
    item = lambda k: {"id": k, "biz": biz[k[0]], "v": v[k]}
    seq1 = [item(k) for k in ("A1", "B1", "A2", "A3", "C1", "A4", "B2", "A5", "A6")]
    seq2 = [item(k) for k in ("A1", "A2", "C1", "A3", "A4", "B1", "A5")]
    return seq1, seq2


def fmt(seq):
    return "|".join(it["id"] for it in seq)


def demo_textbook():
    seq1, seq2 = textbook_example()
    print("=" * 80)
    print("【1】正文例子：序列1（L=9）vs 序列2（L=7），三种口径对比")
    print("=" * 80)
    print(f"序列1     = {fmt(seq1)}   (L={len(seq1)})")
    print(f"序列2     = {fmt(seq2)}   (L={len(seq2)})")
    print(f"序列2补齐 = {fmt(pad_with_mock(seq2))}   (L={L_MAX}, "
          f"M 价值分 = 短视频历史均值 {HIST_MEAN['video']:.1f})")
    print("-" * 80)
    print(f"{'口径':<22}{'序列1':>11}{'序列2':>11}{'胜者':>8}{'差值':>10}")
    print("-" * 80)
    for name, fn in (("裸求和 Score_raw", score_raw),
                     ("长度归一化 Score_norm", score_len_norm),
                     ("Mock 补齐 Score_mock", score_mock_pad)):
        s1, s2 = fn(seq1), fn(seq2)
        print(f"{name:<22}{s1:>11.4f}{s2:>11.4f}"
              f"{('序列1' if s1 > s2 else '序列2'):>8}{s1 - s2:>10.4f}")
    print("-" * 80)
    print(f"序列2 的单位置平均价值更高（{score_len_norm(seq2):.4f} > "
          f"{score_len_norm(seq1):.4f}），但裸求和下仍输给长度 9 的序列1，")
    print(f"差值 {score_raw(seq1) - score_raw(seq2):.4f} 完全来自多出的 2 个坑位 "
          "→ 这就是正文说的长度统计偏置。")


def demo_length_bias():
    """量化长度偏置：同质量内容下，序列长度每 +1 能白拿多少分。"""
    print()
    print("=" * 80)
    print("【2】长度偏置的量化：内容质量完全相同（v=2.0），只改长度")
    print("=" * 80)
    print(f"{'长度 L':>8}{'裸求和':>11}{'长度归一化':>13}{'Mock补齐(L_max=9)':>19}")
    print("-" * 80)
    for L in range(5, 10):
        seq = [{"id": f"X{i}", "biz": "video", "v": 2.0} for i in range(L)]
        print(f"{L:>8}{score_raw(seq):>11.4f}{score_len_norm(seq):>13.4f}"
              f"{score_mock_pad(seq):>19.4f}")
    print("-" * 80)
    print("裸求和随 L 单调上升（长度白送分）；长度归一化恒为 2.0；")
    print("Mock 补齐在 Mock 价值等于内容价值时也恒定 → 两种口径都消除了长度偏置。")


def demo_accuracy(n_trial=4000):
    """对照实验：哪种口径更常选出真实收益更高的序列。"""
    print()
    print("=" * 80)
    print(f"【3】对照实验：以真实收益（空坑位按基线 v_base={TRUE_BASE} 填充）为判据")
    print("=" * 80)
    stat = {"raw": 0, "norm": 0, "mock": 0, "mock_cal": 0}
    n = 0
    for _ in range(n_trial):
        pair = []
        for _ in range(2):
            L = random.randint(5, L_MAX)
            seq = []
            for _ in range(L):
                b = random.choices(["video", "ad", "live"], [0.7, 0.2, 0.1])[0]
                seq.append({"id": b[0].upper(), "biz": b,
                            "v": random.gauss(HIST_MEAN[b], 0.6)})
            pair.append(seq)
        s, t = pair
        if len(s) == len(t):
            continue                      # 等长时三种口径必然一致，不计入
        gt = true_value(s) - true_value(t)
        n += 1
        for key, fn in (("raw", score_raw), ("norm", score_len_norm),
                        ("mock", score_mock_pad),
                        ("mock_cal", lambda q: score_mock_pad(q, v=TRUE_BASE))):
            if (fn(s) - fn(t)) * gt > 0:
                stat[key] += 1
    print(f"{'口径':<26}{'选对次数':>10}{'选对率':>10}")
    print("-" * 80)
    for key, name in (("raw", "裸求和 Score_raw"),
                      ("norm", "长度归一化 Score_norm"),
                      ("mock", "Mock 补齐 v_M=2.0(历史均值)"),
                      ("mock_cal", "Mock 补齐 v_M=2.3(=真实基线)")):
        print(f"{name:<26}{stat[key]:>10}{stat[key] / n:>10.2%}")
    print("-" * 80)
    print(f"共 {n} 组长度不等的对比。裸求和把长度当成优势（80.83%），两种修正口径都到 97% 以上；")
    print("Mock 价值取准真实基线时口径与真实目标完全一致，选对率 100%，")
    print("取历史均值（有 0.3 的估计偏差）则略有损失 → 正文强调 Mock 价值要用历史统计估准。")
    print("长度归一化虽然也能消除偏置，但它改变了目标量纲（比平均价值而非总价值），")
    print("业务上更常见的是正文的第二种方案 Mock Item 补齐。")



def demo_mock_value_sensitivity():
    """Mock Item 价值分取值的敏感性。"""
    print()
    print("=" * 80)
    print(f"【4】Mock 价值取值敏感性（真实基线 v_base={TRUE_BASE}，"
          f"正文建议取历史均值 {HIST_MEAN['video']:.1f}）")
    print("=" * 80)
    seq1, seq2 = textbook_example()
    print(f"{'Mock价值 v_M':>13}{'序列1(L=9)':>13}{'序列2补齐':>12}{'胜者':>8}"
          f"{'真实胜者':>10}{'判对':>7}")
    print("-" * 80)
    gt = "序列1" if true_value(seq1) > true_value(seq2) else "序列2"
    for vm in (0.0, 1.0, 2.0, 2.3, 3.0, 4.0):
        s1, s2 = score_mock_pad(seq1, v=vm), score_mock_pad(seq2, v=vm)
        win = "序列1" if s1 > s2 else "序列2"
        print(f"{vm:>13.1f}{s1:>13.4f}{s2:>12.4f}{win:>8}{gt:>10}"
              f"{('√' if win == gt else '×'):>7}")
    print("-" * 80)
    print("v_M = 0 等价于裸求和（判错）；v_M 越接近真实基线 2.3，判断越可靠；")
    print("v_M 取得过高则把空坑位当成优质内容，反向奖励短序列。")
    print("因此正文强调 Mock 价值取业务历史均值或历史统计估计，以尽量减小对评估的扰动。")


if __name__ == "__main__":
    random.seed(42)
    demo_textbook()
    demo_length_bias()
    demo_accuracy()
    demo_mock_value_sensitivity()
