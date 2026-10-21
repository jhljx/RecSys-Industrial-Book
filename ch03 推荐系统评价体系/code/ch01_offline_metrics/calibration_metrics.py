"""对应正文第 3 章 §3.1.3「概率校准指标」。

本实现做了什么
---------------
1. 实现正文主指标 PCOC（Predicted Click over Observed Click），并复刻正文的两个算例：
   a. 预测总点击 500 次、实际 400 次 -> PCOC = 500/400 = 1.25（整体高估 25%）；
   b. 预测 [0.50, 0.25, 0.05] vs 真实 CTR [0.10, 0.05, 0.01]：排序完全正确、
      AUC 接近 1，但所有预测被系统性放大 5 倍，PCOC 远离 1。
   正文指出部分公司/论文把同类指标称为 COPC（Calibration Of Predicted CTR），
   本实现把 COPC 作为 PCOC 的同义实现一并给出（见 `copc`）。
2. 实现分桶校准曲线与 ECE / MCE（正文未给 ECE 公式，属于业界通用的校准度量补充，
   README 已注明来源：Guo et al., On Calibration of Modern Neural Networks,
   arXiv:1706.04599）。
3. 实现两种经典后验校准方法并对比校准前后的 PCOC / ECE / AUC：
   - Platt Scaling：在 logit 上拟合单变量逻辑回归 sigmoid(a*z + b)；
   - Isotonic Regression：PAV（Pool Adjacent Violators）算法求单调分段常数映射。
   两者都**保序或弱保序**，因此几乎不改变 AUC（Platt 严格保序，AUC 不变；
   Isotonic 会把相邻分数合并成同值，可能带来极小的 AUC 下降）。
4. 复刻 §3.1.3 的广告场景：eCPM = Bid x pCTR，演示 pCTR 系统性偏差
   如何改变广告排序结果与平台收益。

关键公式
---------
    PCOC = sum_i(y_hat_i) / sum_i(y_i)        （>1 高估，<1 低估，理想为 1）
    ECE  = sum_b (n_b / N) * | mean_pred_b - mean_label_b |
    eCPM = Bid x pCTR
Platt:     p = sigmoid(a * logit(p_raw) + b)，用 NLL 梯度下降拟合 (a, b)
Isotonic:  min sum w_i (y_i - f_i)^2  s.t. f 非递减，PAV 算法 O(n)
"""

from __future__ import annotations

import math
import random


# --------------------------------------------------------------------------
# PCOC / COPC
# --------------------------------------------------------------------------
def pcoc(preds: list[float], labels: list[int]) -> float:
    """PCOC = sum(预测CTR) / sum(真实点击标签)。等价于 预测点击数 / 实际点击数。"""
    obs = sum(labels)
    if obs == 0:
        return float("inf")
    return sum(preds) / obs


def copc(preds: list[float], labels: list[int]) -> float:
    """COPC（Calibration Of Predicted CTR）。

    正文：部分公司和论文将类似指标称为 COPC，具体定义可能存在细微差异，
    但本质上都用于衡量模型预测概率与真实概率之间的匹配程度。
    这里按与 PCOC 相同的口径实现（平均预估 CTR / 平均真实 CTR）。
    """
    n = len(labels)
    if n == 0 or sum(labels) == 0:
        return float("inf")
    return (sum(preds) / n) / (sum(labels) / n)


# --------------------------------------------------------------------------
# 分桶校准曲线 / ECE / MCE
# --------------------------------------------------------------------------
def calibration_buckets(preds: list[float], labels: list[int], n_bins: int = 10,
                        scheme: str = "equal_width"):
    """分桶校准曲线。

    scheme="equal_width"   等宽分桶：按 [0,1] 均分 n_bins 个区间（ECE 原始定义）
    scheme="equal_freq"    等频分桶：按预测分数分位数切分（CTR 预估分布极偏时更稳）
    返回 [(lo, hi, n_b, mean_pred, mean_label)]，空桶被丢弃。
    """
    pairs = sorted(zip(preds, labels))
    out = []
    if scheme == "equal_width":
        edges = [i / n_bins for i in range(n_bins + 1)]
    else:
        n = len(pairs)
        edges = [pairs[min(n - 1, round(i * n / n_bins))][0] for i in range(n_bins)]
        edges.append(float("inf"))
        edges[0] = -float("inf")
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        if scheme == "equal_width":
            sel = [(p, y) for p, y in pairs if (lo <= p < hi or (b == n_bins - 1 and p == hi))]
        else:
            sel = [(p, y) for p, y in pairs if lo <= p < hi]
        if not sel:
            continue
        mp = sum(p for p, _ in sel) / len(sel)
        ml = sum(y for _, y in sel) / len(sel)
        out.append((lo, hi, len(sel), mp, ml))
    return out


def ece(preds: list[float], labels: list[int], n_bins: int = 10,
        scheme: str = "equal_width") -> float:
    """ECE = sum_b (n_b / N) * |mean_pred_b - mean_label_b|，越小越校准。"""
    buckets = calibration_buckets(preds, labels, n_bins, scheme)
    n = len(labels)
    return sum(nb / n * abs(mp - ml) for _, _, nb, mp, ml in buckets)


def mce(preds: list[float], labels: list[int], n_bins: int = 10,
        scheme: str = "equal_width") -> float:
    """MCE（Maximum Calibration Error）：各桶偏差的最大值。"""
    buckets = calibration_buckets(preds, labels, n_bins, scheme)
    return max((abs(mp - ml) for _, _, _, mp, ml in buckets), default=0.0)


def print_calibration_table(preds: list[float], labels: list[int], n_bins: int = 10,
                            scheme: str = "equal_freq", indent: str = "  ") -> None:
    """打印分桶校准曲线（替代 matplotlib 的 reliability diagram，纯数值表）。"""
    buckets = calibration_buckets(preds, labels, n_bins, scheme)
    print(f"{indent}{'桶':>3} | {'样本数':>6} | {'平均预估':>9} | {'真实CTR':>9} | "
          f"{'偏差':>8} | {'桶内PCOC':>9}")
    print(indent + "-" * 62)
    for i, (_, _, nb, mp, ml) in enumerate(buckets):
        ratio = mp / ml if ml > 0 else float("inf")
        print(f"{indent}{i:>3d} | {nb:>6d} | {mp:>9.4f} | {ml:>9.4f} | "
              f"{mp-ml:>+8.4f} | {ratio:>9.3f}")


# --------------------------------------------------------------------------
# 校准方法 1：Platt Scaling
# --------------------------------------------------------------------------
def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(preds: list[float], labels: list[int], lr: float = 0.5,
              epochs: int = 600) -> tuple[float, float]:
    """Platt Scaling：在 logit 空间拟合 p = sigmoid(a*z + b)，z = logit(p_raw)。

    用 NLL 的解析梯度做全批量梯度下降（单变量逻辑回归，凸问题）：
        dNLL/da = mean[(sigmoid(a*z+b) - y) * z]
        dNLL/db = mean[ sigmoid(a*z+b) - y ]
    初始化 a=1, b=0 即恒等映射。
    """
    zs = [_logit(p) for p in preds]
    a, b = 1.0, 0.0
    n = len(zs)
    for _ in range(epochs):
        ga = gb = 0.0
        for z, y in zip(zs, labels):
            d = _sigmoid(a * z + b) - y
            ga += d * z
            gb += d
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def apply_platt(preds: list[float], a: float, b: float) -> list[float]:
    return [_sigmoid(a * _logit(p) + b) for p in preds]


# --------------------------------------------------------------------------
# 校准方法 2：Isotonic Regression（PAV 算法）
# --------------------------------------------------------------------------
def fit_isotonic(preds: list[float], labels: list[int]):
    """Isotonic Regression，Pool Adjacent Violators（PAV）算法。

    按 pred 升序扫描，维护 (sum_y, count) 块；若后一块均值低于前一块均值则合并，
    直到整体非递减。返回 (x_breaks, y_values)：x_breaks[i] 是第 i 块的最大 pred，
    y_values[i] 是该块的标签均值（即校准后概率）。复杂度 O(n log n)（含排序）。
    """
    pairs = sorted(zip(preds, labels))
    blocks: list[list[float]] = []   # [x_max, sum_y, count]
    for p, y in pairs:
        blocks.append([p, float(y), 1.0])
        while len(blocks) >= 2 and (blocks[-2][1] / blocks[-2][2]) >= (blocks[-1][1] / blocks[-1][2]):
            x2, s2, c2 = blocks.pop()
            blocks[-1][0] = x2            # 合并后块的右端点取较大的 pred
            blocks[-1][1] += s2
            blocks[-1][2] += c2
    x_breaks = [bk[0] for bk in blocks]
    y_values = [bk[1] / bk[2] for bk in blocks]
    return x_breaks, y_values


def apply_isotonic(preds: list[float], x_breaks: list[float], y_values: list[float]) -> list[float]:
    """分段常数查表：找第一个 x_breaks[i] >= p 的块，取其 y_values[i]。"""
    import bisect
    out = []
    for p in preds:
        i = bisect.bisect_left(x_breaks, p)
        out.append(y_values[min(i, len(y_values) - 1)])
    return out


# --------------------------------------------------------------------------
# AUC（与 auc_gauc.py 同算法，本文件自包含故重复一份）
# --------------------------------------------------------------------------
def auc_rank(labels: list[int], scores: list[float]) -> float:
    m, n = sum(labels), len(labels) - sum(labels)
    if m == 0 or n == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for y, r in zip(labels, ranks) if y == 1)
    return (rank_sum - m * (m + 1) / 2.0) / (m * n)


def log_loss(preds: list[float], labels: list[int], eps: float = 1e-9) -> float:
    total = 0.0
    for p, y in zip(preds, labels):
        p = min(1 - eps, max(eps, p))
        total -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return total / len(labels)


# --------------------------------------------------------------------------
# 合成数据：排序很好但概率系统性高估的模型
# --------------------------------------------------------------------------
def make_miscalibrated(n: int, blow_up: float):
    """造一个「排序能力强、概率高估」的 CTR 模型输出。

    真实 CTR: p_true = sigmoid(w . x)，label ~ Bernoulli(p_true)；
    模型输出: 在 logit 上乘以 blow_up（>1 则概率被系统性放大），
    保序变换不改变 AUC，只破坏校准。
    """
    preds, labels, truths = [], [], []
    for _ in range(n):
        z = random.gauss(-2.2, 1.1)         # 真实 logit，均值对应 ~10% CTR
        p_true = _sigmoid(z)
        y = 1 if random.random() < p_true else 0
        preds.append(_sigmoid(blow_up * z + 1.6))   # 系统性高估
        labels.append(y)
        truths.append(p_true)
    return preds, labels, truths


def main() -> None:
    random.seed(42)

    print("=" * 78)
    print("§3.1.3 概率校准指标：PCOC / COPC / ECE + Platt Scaling / Isotonic Regression")
    print("=" * 78)

    # ---------- (1) 正文算例 a：预测 500 次点击 vs 实际 400 次 ----------
    print("\n[正文算例 1] 模型预测总点击 500 次，实际点击 400 次")
    preds_a = [0.5] * 1000          # 1000 次曝光，每次预估 0.5 -> 预测点击 500
    labels_a = [1] * 400 + [0] * 600
    print(f"  预测点击数 = {sum(preds_a):.0f}，实际点击数 = {sum(labels_a)}")
    print(f"  PCOC = {pcoc(preds_a, labels_a):.4f}  -> 模型整体高估了 "
          f"{(pcoc(preds_a, labels_a)-1)*100:.0f}% 的点击概率")

    # ---------- (2) 正文算例 b：排序全对但放大 5 倍 ----------
    print("\n[正文算例 2] 预测 [0.50, 0.25, 0.05] vs 真实 CTR [0.10, 0.05, 0.01]")
    pred3 = [0.50, 0.25, 0.05]
    true3 = [0.10, 0.05, 0.01]
    print(f"  逐项比值 = {[round(p/t, 2) for p, t in zip(pred3, true3)]}"
          f"  -> 全部被系统性放大 5 倍")
    # 用大样本把真实 CTR 实例化成 0/1 标签，才能算 AUC 与 PCOC
    reps = 20000
    big_pred, big_label = [], []
    for p, t in zip(pred3, true3):
        for _ in range(reps):
            big_pred.append(p)
            big_label.append(1 if random.random() < t else 0)
    print(f"  按真实 CTR 采样 {len(big_label)} 条样本后：")
    print(f"    AUC  = {auc_rank(big_label, big_pred):.4f}   （排序顺序完全正确，远高于 0.5）")
    print(f"    PCOC = {pcoc(big_pred, big_label):.4f}   （远离 1，概率预估能力差）")
    print(f"    COPC = {copc(big_pred, big_label):.4f}   （同口径的另一种称法）")
    print("  >>> 正文结论：高 AUC 并不一定意味着良好的概率校准能力。")

    # ---------- (3) 分桶校准曲线 + ECE ----------
    print("\n" + "-" * 78)
    print("[分桶校准曲线] 一个排序好但概率高估的 CTR 模型（30000 条样本）")
    print("-" * 78)
    preds, labels, _ = make_miscalibrated(30000, blow_up=1.35)
    print(f"  整体：平均预估 = {sum(preds)/len(preds):.4f}，"
          f"真实 CTR = {sum(labels)/len(labels):.4f}")
    print(f"  AUC = {auc_rank(labels, preds):.4f}   PCOC = {pcoc(preds, labels):.4f}   "
          f"ECE = {ece(preds, labels, 10, 'equal_freq'):.4f}   "
          f"MCE = {mce(preds, labels, 10, 'equal_freq'):.4f}")
    print("\n  等频 10 分桶校准曲线（桶内 PCOC 应全部接近 1.0 才算校准良好）：")
    print_calibration_table(preds, labels, 10, "equal_freq")

    # ---------- (4) Platt Scaling / Isotonic Regression ----------
    print("\n" + "-" * 78)
    print("[后验校准] Platt Scaling vs Isotonic Regression")
    print("-" * 78)
    split = int(len(preds) * 0.6)
    tr_p, tr_y = preds[:split], labels[:split]
    te_p, te_y = preds[split:], labels[split:]
    print(f"  校准集 {len(tr_p)} 条 / 测试集 {len(te_p)} 条（校准器只在校准集上拟合）")

    a, b = fit_platt(tr_p, tr_y)
    platt_te = apply_platt(te_p, a, b)
    print(f"\n  Platt Scaling 拟合结果: a = {a:.4f}, b = {b:.4f}"
          f"  (a<1 压缩 logit 幅度，b<0 整体下调)")

    xb, yv = fit_isotonic(tr_p, tr_y)
    iso_te = apply_isotonic(te_p, xb, yv)
    print(f"  Isotonic Regression: PAV 把 {len(tr_p)} 个点合并成 {len(yv)} 个单调块，"
          f"输出值域 [{min(yv):.4f}, {max(yv):.4f}]")

    print(f"\n  {'方法':>22} | {'AUC':>7} | {'PCOC':>7} | {'ECE':>7} | {'MCE':>7} | {'LogLoss':>8}")
    print("  " + "-" * 72)
    for name, pp in (("原始输出（未校准）", te_p),
                     ("Platt Scaling", platt_te),
                     ("Isotonic Regression", iso_te)):
        print(f"  {name:>22} | {auc_rank(te_y, pp):>7.4f} | {pcoc(pp, te_y):>7.4f} | "
              f"{ece(pp, te_y, 10, 'equal_freq'):>7.4f} | "
              f"{mce(pp, te_y, 10, 'equal_freq'):>7.4f} | {log_loss(pp, te_y):>8.4f}")
    print("  >>> 两种校准方法都是（弱）保序变换，AUC 基本不变，但 PCOC 被拉回 1 附近、")
    print("      ECE/LogLoss 明显下降。这正好印证正文的分工：AUC 看排序，PCOC 看校准，")
    print("      二者可以独立优化。")

    print("\n  Platt 校准后的等频 10 分桶曲线：")
    print_calibration_table(platt_te, te_y, 10, "equal_freq")

    # ---------- (5) 广告场景 eCPM = Bid x pCTR ----------
    print("\n" + "-" * 78)
    print("[广告场景] eCPM = Bid x pCTR，pCTR 偏差如何改变排序与收益")
    print("-" * 78)
    # (广告, 出价 Bid(元/千次), 真实 CTR, pCTR 的系统性偏差倍数)
    ads = [
        ("广告A", 2.0, 0.080, 1.00),
        ("广告B", 8.0, 0.020, 1.00),
        ("广告C", 4.0, 0.045, 2.20),   # 模型对 C 严重高估
        ("广告D", 5.0, 0.040, 0.60),   # 模型对 D 严重低估
    ]
    print(f"  {'广告':>6} | {'Bid':>5} | {'真实CTR':>8} | {'pCTR':>7} | {'偏差':>6} | "
          f"{'真实eCPM':>9} | {'预估eCPM':>9}")
    print("  " + "-" * 72)
    rows = []
    for name, bid, ctr, bias in ads:
        pctr = min(1.0, ctr * bias)
        rows.append((name, bid, ctr, pctr, bias, bid * ctr * 1000, bid * pctr * 1000))
    for r in rows:
        print(f"  {r[0]:>6} | {r[1]:>5.1f} | {r[2]:>8.3f} | {r[3]:>7.3f} | "
              f"{r[4]:>6.2f} | {r[5]:>9.2f} | {r[6]:>9.2f}")
    ideal = sorted(rows, key=lambda r: -r[5])
    actual = sorted(rows, key=lambda r: -r[6])
    print(f"\n  按真实 eCPM 的理想排序: {[r[0] for r in ideal]}")
    print(f"  按预估 eCPM 的实际排序: {[r[0] for r in actual]}")
    print(f"  Top-1 理想收益 = {ideal[0][5]:.2f} 元/千次（{ideal[0][0]}）")
    print(f"  Top-1 实际收益 = {actual[0][5]:.2f} 元/千次（选中 {actual[0][0]}，"
          f"按它的真实 eCPM 结算）")
    loss = ideal[0][5] - actual[0][5]
    print(f"  >>> 收益损失 = {loss:.2f} 元/千次，相对损失 {loss/ideal[0][5]*100:.1f}%。")
    print("  >>> 正文结论：pCTR 的系统性偏差会直接影响广告排序结果与平台收益，")
    print("      因此工业界必须同时监控 AUC/GAUC（排序能力）与 PCOC（校准能力）。")


if __name__ == "__main__":
    main()
