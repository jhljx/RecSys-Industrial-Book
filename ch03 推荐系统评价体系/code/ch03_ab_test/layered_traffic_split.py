"""对应正文第 3 章 §3.3.1「分流机制」与 §3.3.2「A/B实验平台」的分层实验架构。

本实现做了什么
---------------
1. 实现正文给出的哈希分桶分流：`hash(user_id) mod 100`，落在 [0,94] 进 Base 组，
   否则进 Exp 组，形成 95% : 5% 的流量划分；
2. 实现正文 §3.3.2 的**分层实验（Layered Experimentation）**架构：
   顶层 Holdout 层 -> 业务 Combo 层 -> 普通实验层，层内互斥、层间正交，
   让同一批用户流量被多个实验同时复用；
3. 校验正文列出的三个关键假设：
   - 随机性（Randomization）：卡方检验校验分桶均匀性 + 校验组间协变量无系统性差异；
   - 一致性（Sticky Assignment）：同一用户在整个实验周期内始终同组；
   - 层间正交性：任意两层的分组交叉表应接近独立（卡方独立性检验）。

关键公式
---------
    bucket(uid, layer) = md5(f"{layer_salt}:{uid}") mod 100      （加盐保证层间独立）
    卡方均匀性统计量    chi2 = sum_i (O_i - E_i)^2 / E_i,  df = k - 1
    卡方独立性统计量    chi2 = sum_ij (O_ij - E_ij)^2 / E_ij, E_ij = R_i*C_j/N,
                        df = (r-1)(c-1)

注意：Python 内置 hash() 对 str 有随机化盐（PYTHONHASHSEED），跨进程不稳定，
生产环境必须用 md5/murmur 等稳定哈希。本脚本统一使用 md5。
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict


# --------------------------------------------------------------------------
# 稳定哈希分桶
# --------------------------------------------------------------------------
def stable_bucket(key: str, salt: str, n_buckets: int = 100) -> int:
    """md5 稳定哈希分桶。salt 为层标识，不同层用不同 salt 即可得到正交分流。"""
    h = hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(h, 16) % n_buckets


def assign_95_5(user_id: int, salt: str = "exp_ctr_v2") -> str:
    """正文示例：hash(user_id) mod 100 落在 [0,94] -> Base，否则 -> Exp（95%:5%）。"""
    b = stable_bucket(str(user_id), salt)
    return "Base" if 0 <= b <= 94 else "Exp"


# --------------------------------------------------------------------------
# 分层实验平台
# --------------------------------------------------------------------------
class Layer:
    """实验平台中的一层。层内各实验互斥瓜分 100 个桶，层与层之间通过 salt 正交。"""

    def __init__(self, name: str, salt: str, groups: dict[str, range]):
        self.name = name
        self.salt = salt
        self.groups = groups          # 组名 -> 桶区间
        covered = sum(len(r) for r in groups.values())
        assert covered == 100, f"层 {name} 的桶未覆盖满 100：{covered}"

    def assign(self, user_id: int) -> str:
        b = stable_bucket(str(user_id), self.salt)
        for g, rng in self.groups.items():
            if b in rng:
                return g
        raise RuntimeError("unreachable")


class ExperimentPlatform:
    """[MOCK] 极简实验平台：配置管理 + 流量分桶系统（正文 §3.3.2）。"""

    def __init__(self) -> None:
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def assign_all(self, user_id: int) -> dict[str, str]:
        return {ly.name: ly.assign(user_id) for ly in self.layers}


# --------------------------------------------------------------------------
# 统计校验
# --------------------------------------------------------------------------
def chi2_uniformity(counts: list[int]) -> tuple[float, int]:
    """卡方均匀性检验：H0 = 各桶频数相同。返回 (chi2, df)。"""
    n = sum(counts)
    k = len(counts)
    exp = n / k
    chi2 = sum((o - exp) ** 2 / exp for o in counts)
    return chi2, k - 1


def chi2_independence(table: dict[tuple[str, str], int],
                      rows: list[str], cols: list[str]) -> tuple[float, int]:
    """卡方独立性检验：H0 = 两层分组互相独立（正交）。返回 (chi2, df)。"""
    n = sum(table.values())
    row_sum = {r: sum(table.get((r, c), 0) for c in cols) for r in rows}
    col_sum = {c: sum(table.get((r, c), 0) for r in rows) for c in cols}
    chi2 = 0.0
    for r in rows:
        for c in cols:
            e = row_sum[r] * col_sum[c] / n
            if e > 0:
                chi2 += (table.get((r, c), 0) - e) ** 2 / e
    return chi2, (len(rows) - 1) * (len(cols) - 1)


def _lower_gamma_reg(s: float, x: float) -> float:
    """正则化下不完全 gamma 函数 P(s,x)，级数展开（x < s+1 收敛快）。"""
    if x <= 0:
        return 0.0
    term = 1.0 / s
    total = term
    for k in range(1, 400):
        term *= x / (s + k)
        total += term
        if term < total * 1e-15:
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))


def _upper_gamma_reg(s: float, x: float) -> float:
    """正则化上不完全 gamma Q(s,x)，连分式（x >= s+1 收敛快）。"""
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 400):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def chi2_sf(chi2: float, df: int) -> float:
    """卡方分布上尾概率 P(X > chi2)，即 p-value。"""
    s, x = df / 2.0, chi2 / 2.0
    if x < s + 1.0:
        return 1.0 - _lower_gamma_reg(s, x)
    return _upper_gamma_reg(s, x)


# --------------------------------------------------------------------------
# [MOCK] 用户画像：用于校验组间协变量平衡（随机性假设）
# --------------------------------------------------------------------------
def mock_user_profile(user_id: int) -> dict:
    """[MOCK] 用户画像：年龄、地域、新老用户、机型。由 user_id 确定性派生。"""
    rng = random.Random(user_id * 7919 + 13)
    return {
        "age": rng.gauss(28.0, 8.0),
        "region": rng.choice(["华北", "华东", "华南", "西部"]),
        "is_new": rng.random() < 0.18,
        "device": rng.choices(["iOS", "Android-高端", "Android-低端"],
                              weights=[0.3, 0.4, 0.3])[0],
    }


def main() -> None:
    random.seed(42)
    n_users = 200_000

    print("=" * 82)
    print("§3.3.1 分流机制 + §3.3.2 分层实验平台")
    print("=" * 82)

    # ---------- (1) 正文示例：95% : 5% ----------
    print("\n[1] 正文分流示例：hash(user_id) mod 100，[0,94]->Base，[95,99]->Exp")
    groups = Counter(assign_95_5(u) for u in range(n_users))
    tot = sum(groups.values())
    for g in ("Base", "Exp"):
        print(f"  {g:>5}: {groups[g]:>7d} 用户  实际占比 {100.0*groups[g]/tot:7.4f}%  "
              f"（设计占比 {95.0 if g=='Base' else 5.0:.1f}%）")

    # ---------- (2) 分桶均匀性校验 ----------
    print("\n[2] 流量均匀性校验（卡方拟合优度，H0 = 100 个桶均匀）")
    bucket_counts = Counter(stable_bucket(str(u), "exp_ctr_v2") for u in range(n_users))
    counts = [bucket_counts[i] for i in range(100)]
    chi2, df = chi2_uniformity(counts)
    p = chi2_sf(chi2, df)
    print(f"  每桶期望 {n_users/100:.0f} 条，实际 min={min(counts)} max={max(counts)} "
          f"mean={sum(counts)/100:.1f}")
    print(f"  chi2 = {chi2:.3f}, df = {df}, p-value = {p:.4f}  "
          f"-> {'均匀性通过（p>0.05，不拒绝 H0）' if p > 0.05 else '均匀性异常！'}")

    # 故意制造一个坏哈希做对照
    print("\n  [反例] 用 user_id mod 100 直接分桶（没有哈希打散）：")
    bad = Counter(u % 100 for u in range(n_users))
    bad_counts = [bad[i] for i in range(100)]
    c2, d2 = chi2_uniformity(bad_counts)
    print(f"    chi2 = {c2:.3f}（完美均匀），但 user_id 本身常带业务语义（注册顺序），")
    print("    低位桶可能全是老用户，会直接破坏随机性假设。必须哈希后再分桶。")

    # ---------- (3) 协变量平衡（随机性假设） ----------
    print("\n[3] 随机性假设校验：Base/Exp 组协变量是否存在系统性差异")
    agg: dict[str, dict] = {"Base": defaultdict(float), "Exp": defaultdict(float)}
    region_tbl: dict[tuple[str, str], int] = defaultdict(int)
    device_tbl: dict[tuple[str, str], int] = defaultdict(int)
    for u in range(n_users):
        g = assign_95_5(u)
        pf = mock_user_profile(u)
        agg[g]["n"] += 1
        agg[g]["age"] += pf["age"]
        agg[g]["new"] += pf["is_new"]
        region_tbl[(g, pf["region"])] += 1
        device_tbl[(g, pf["device"])] += 1
    print(f"  {'协变量':>12} | {'Base':>10} | {'Exp':>10} | {'绝对差':>8}")
    print("  " + "-" * 48)
    for name, key in (("平均年龄", "age"), ("新用户占比(%)", "new")):
        b = agg["Base"][key] / agg["Base"]["n"] * (100 if key == "new" else 1)
        e = agg["Exp"][key] / agg["Exp"]["n"] * (100 if key == "new" else 1)
        print(f"  {name:>12} | {b:>10.4f} | {e:>10.4f} | {abs(e-b):>8.4f}")
    for label, tbl, cols in (("地域", region_tbl, ["华北", "华东", "华南", "西部"]),
                             ("机型", device_tbl, ["iOS", "Android-高端", "Android-低端"])):
        c, d = chi2_independence(tbl, ["Base", "Exp"], cols)
        pv = chi2_sf(c, d)
        print(f"  {label}分布独立性：chi2={c:.3f}, df={d}, p={pv:.4f} -> "
              f"{'无系统性差异' if pv > 0.05 else '存在差异，需排查！'}")

    # ---------- (4) 一致性 Sticky Assignment ----------
    print("\n[4] 一致性假设（Sticky Assignment）：同一用户多次请求是否恒定同组")
    ok = True
    for u in random.sample(range(n_users), 2000):
        g0 = assign_95_5(u)
        if any(assign_95_5(u) != g0 for _ in range(20)):     # 模拟 20 次请求
            ok = False
            break
    print(f"  随机抽 2000 个用户、每人 20 次请求：{'全部恒定同组 ✓' if ok else '出现漂移 ✗'}")
    print("  >>> 实现要点：分流 key 必须是 user_id/device_id 这类稳定 ID；")
    print("      若用 request_id 分流，同一用户会在两组间反复横跳，策略影响相互污染。")
    rid_groups = Counter(
        "Base" if stable_bucket(f"req-{u}-{i}", "exp_ctr_v2") <= 94 else "Exp"
        for u in range(300) for i in range(20))
    switched = sum(1 for u in range(300)
                   if len({"Base" if stable_bucket(f"req-{u}-{i}", "exp_ctr_v2") <= 94
                           else "Exp" for i in range(20)}) > 1)
    print(f"  [反例] 改用 request_id 分流：300 个用户中有 {switched} 人"
          f"（{100.0*switched/300:.1f}%）在 20 次请求内切换过实验组。")

    # ---------- (5) 分层正交实验 ----------
    print("\n[5] 分层实验架构（正文 §3.3.2：顶层 Holdout / 业务 Combo / 普通实验层）")
    plat = ExperimentPlatform()
    plat.add_layer(Layer("L0-Holdout", "salt_holdout",
                         {"holdout": range(0, 1), "normal": range(1, 100)}))
    plat.add_layer(Layer("L1-召回Combo", "salt_recall_combo",
                         {"base": range(0, 80), "combo_A": range(80, 90),
                          "combo_B": range(90, 100)}))
    plat.add_layer(Layer("L2-排序实验", "salt_rank_exp",
                         {"base": range(0, 85), "exp_mmoe": range(85, 95),
                          "exp_feat": range(95, 100)}))
    plat.add_layer(Layer("L3-重排实验", "salt_rerank_exp",
                         {"base": range(0, 90), "exp_mmr": range(90, 100)}))

    layer_counts: dict[str, Counter] = {ly.name: Counter() for ly in plat.layers}
    cross: dict[tuple[str, str], int] = defaultdict(int)
    for u in range(n_users):
        a = plat.assign_all(u)
        for k, v in a.items():
            layer_counts[k][v] += 1
        cross[(a["L2-排序实验"], a["L3-重排实验"])] += 1

    print(f"  {'层':>14} | {'组':>10} | {'用户数':>8} | {'占比':>8} | {'设计':>7}")
    print("  " + "-" * 56)
    for ly in plat.layers:
        for g, rng in ly.groups.items():
            c = layer_counts[ly.name][g]
            print(f"  {ly.name:>14} | {g:>10} | {c:>8d} | "
                  f"{100.0*c/n_users:>7.3f}% | {len(rng):>6d}%")

    print("\n  层间正交性检验（L2 排序实验 x L3 重排实验 交叉表）：")
    rows = ["base", "exp_mmoe", "exp_feat"]
    cols = ["base", "exp_mmr"]
    print(f"    {'':>10} | " + " | ".join(f"{c:>9}" for c in cols))
    for r in rows:
        print(f"    {r:>10} | " + " | ".join(f"{cross[(r,c)]:>9d}" for c in cols))
    c, d = chi2_independence(cross, rows, cols)
    pv = chi2_sf(c, d)
    print(f"    chi2 = {c:.4f}, df = {d}, p = {pv:.4f} -> "
          f"{'两层正交 ✓（互不干扰，流量可复用）' if pv > 0.05 else '两层存在相关，salt 可能重复 ✗'}")

    print("\n  [反例] 两层使用相同 salt 会怎样：")
    same_a = Layer("X", "same_salt", {"base": range(0, 85), "exp": range(85, 100)})
    same_b = Layer("Y", "same_salt", {"base": range(0, 90), "exp": range(90, 100)})
    bad_cross: dict[tuple[str, str], int] = defaultdict(int)
    for u in range(n_users):
        bad_cross[(same_a.assign(u), same_b.assign(u))] += 1
    c, d = chi2_independence(bad_cross, ["base", "exp"], ["base", "exp"])
    print(f"    交叉表 = {dict(bad_cross)}")
    print(f"    chi2 = {c:.1f}, df = {d}, p = {chi2_sf(c, d):.3e} -> 严重相关，"
          f"Y 层的 exp 组被完全嵌套在 X 层 exp 组内，两个实验互相污染。")

    # ---------- (6) 长期 Holdout ----------
    hold = layer_counts["L0-Holdout"]["holdout"]
    print(f"\n[6] 长期 Holdout（正文 §3.3.4）：保留 {100.0*hold/n_users:.3f}% "
          f"（{hold} 用户）永不接收任何新策略，")
    print("    作为永久对照组观察系统长期演化趋势与跨实验横向比较。")


if __name__ == "__main__":
    main()
