"""对应正文第 9 章 §9.2「生成式推荐架构」中生成式推荐与传统级联式架构的对比。

正文给出了两种工业部署方式:
  (1) 并行候选生成架构(Parallel Candidate Generation Architecture):
      生成式推荐模块与传统召回并行产出候选, 最终汇合进入重排模块统一排序, 即
      User -> Generative Recommendation -> Rerank;
  (2) 流量隔离架构(Traffic Isolation Architecture): 划分一定比例实验流量, 由生成式
      模块直接产出 Item 集合再进入重排, 不再走"召回->粗排->精排"。
正文同时在 §9.8 讨论里强调: 级联架构的分层防护使单模块异常的影响可控, 而 One-Model
架构下模型异常会被放大到整条链路。

本实现用纯标准库做一个**算力/候选规模/可控性**的解析对比模型(不做真实服务):
  * 级联漏斗: 候选规模按 N_recall -> N_coarse -> N_fine -> N_rerank 逐级收缩,
    每级算力 = 该级候选数 x 单条打分 FLOPs, 总算力为各级求和;
  * 生成式: 算力 = T 步自回归 x 每步 Decoder FLOPs(与候选规模无关), 其中
    T = L (语义 ID 层数) x beam 扩展代价, 候选规模 = beam_size 条合法路径;
  * 可控性: 用"可插入业务约束的决策点个数 / 异常影响半径"量化正文的定性结论。

关键公式(本文件使用的解析式):
  级联总算力  C_cascade = sum_i N_i * f_i
  生成式总算力 C_gen     = L * B * f_dec + B * f_retrieval
  异常影响半径 blast(架构) = 单模块故障时受影响的最终曝光比例

[MOCK] 所有 FLOPs/延迟系数都是教学用的量级估计常数, 不代表任何真实系统的实测值。
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------- 架构参数
# [MOCK] 量级估计常数: 单条样本打分的 MFLOPs
CASCADE_STAGES = [
    # (阶段名, 入口候选数, 单条打分 MFLOPs, 可插入业务约束的决策点数)
    ("召回 Retrieval", 1_000_000, 0.002, 2),
    ("粗排 Coarse Rank", 6_000, 0.05, 2),
    ("精排 Fine Rank", 600, 2.0, 4),
    ("重排 Rerank", 60, 0.5, 5),
]

GEN_L = 4           # 语义 ID 层数, 即自回归步数
GEN_BEAM = 64       # beam 宽度, 也即生成式模块输出的候选条数
GEN_FLOPS_STEP = 9.0    # [MOCK] 每个 Decoder 前向步的 MFLOPs(含 beam 批处理)
GEN_FLOPS_RETRIEVAL = 0.004   # [MOCK] 语义 ID -> 物品 ID 的查表 MFLOPs


def cascade_cost() -> tuple[float, int, int]:
    """返回 (总 MFLOPs, 最终候选数, 决策点总数)。"""
    total = sum(n * f for _, n, f, _ in CASCADE_STAGES)
    hooks = sum(h for *_, h in CASCADE_STAGES)
    return total, CASCADE_STAGES[-1][1], hooks


def generative_cost() -> tuple[float, int, int]:
    """生成式模块: 算力只与 L、beam 有关, 与全库规模解耦。"""
    total = GEN_L * GEN_BEAM * GEN_FLOPS_STEP + GEN_BEAM * GEN_FLOPS_RETRIEVAL
    # 正文: 生成式模块缺少显式多样性控制, 业务约束只能塞进 Beam Search 打分与重排
    hooks = 1 + 5   # Beam Search 打分 1 个 + 保留的重排模块 5 个
    return total, GEN_BEAM, hooks


def blast_radius(arch: str) -> dict[str, float]:
    """单模块故障的影响半径。级联有逐层兜底, One-Model 无。"""
    if arch == "cascade":
        # 任一路召回挂掉, 其余召回仍能补齐候选 -> 影响被限制在该路的贡献占比
        return {"单路召回故障": 0.18, "粗排故障(降级为随机截断)": 0.35, "精排故障": 1.00}
    return {"单路召回故障": 1.00, "粗排故障(降级为随机截断)": 1.00, "精排故障": 1.00}


def deploy_modes() -> list[tuple[str, str, str]]:
    """正文 §9.2 给出的两种工业部署方式。"""
    return [
        ("并行候选生成",
         "User -> {传统召回, 生成式模块} -> 重排",
         "生成式模块当作打通召回到粗排的统一通道, 候选汇入重排统一决策"),
        ("流量隔离",
         "User -> 生成式模块 -> 重排",
         "划分 ~5% 实验流量, 生成式模块直接决定 Item 集合, 不走传统漏斗"),
    ]


def main() -> None:
    random.seed(42)

    c_flops, c_cand, c_hooks = cascade_cost()
    g_flops, g_cand, g_hooks = generative_cost()

    print("=" * 78)
    print("表 1  级联漏斗各级候选规模与算力(§9.2 传统多级漏斗)")
    print("-" * 78)
    print(f"{'阶段':<18}{'入口候选':>12}{'单条MFLOPs':>14}{'阶段MFLOPs':>14}{'占比':>9}")
    for name, n, f, _ in CASCADE_STAGES:
        stage = n * f
        print(f"{name:<18}{n:>12,}{f:>14.4f}{stage:>14.1f}{stage / c_flops:>8.1%}")
    print("-" * 78)
    print(f"{'级联合计':<18}{'':>12}{'':>14}{c_flops:>14.1f}{1.0:>8.1%}")

    print()
    print("=" * 78)
    print("表 2  生成式推荐模块算力(§9.2, 自回归 L 步 + 语义ID检索)")
    print("-" * 78)
    print(f"{'环节':<28}{'次数':>8}{'单次MFLOPs':>14}{'MFLOPs':>12}")
    print(f"{'Decoder 自回归前向':<28}{GEN_L * GEN_BEAM:>8}{GEN_FLOPS_STEP:>14.4f}"
          f"{GEN_L * GEN_BEAM * GEN_FLOPS_STEP:>12.1f}")
    print(f"{'语义ID->物品ID 查表':<28}{GEN_BEAM:>8}{GEN_FLOPS_RETRIEVAL:>14.4f}"
          f"{GEN_BEAM * GEN_FLOPS_RETRIEVAL:>12.3f}")
    print("-" * 78)
    print(f"{'生成式合计':<28}{'':>8}{'':>14}{g_flops:>12.1f}")

    print()
    print("=" * 78)
    print("表 3  两种架构的候选规模 / 算力 / 可控性对比")
    print("-" * 78)
    rows = [
        ("候选规模(最终进入决策)", f"{c_cand}", f"{g_cand}"),
        ("打分涉及的物品总量", f"{CASCADE_STAGES[0][1]:,}", f"{GEN_L * GEN_BEAM} 条路径"),
        ("总算力 MFLOPs", f"{c_flops:.1f}", f"{g_flops:.1f}"),
        ("算力是否随全库规模增长", "是(召回级线性)", "否(与 L、beam 有关)"),
        ("串行前向次数", f"{len(CASCADE_STAGES)}", f"{GEN_L}(逐 token 自回归)"),
        ("可插入业务约束的决策点", f"{c_hooks}", f"{g_hooks}"),
    ]
    print(f"{'维度':<26}{'级联漏斗':>22}{'生成式 One-Model':>26}")
    for k, a, b in rows:
        print(f"{k:<26}{a:>22}{b:>26}")
    print("-" * 78)
    print(f"算力比(生成式/级联) = {g_flops / c_flops:.3f}x; "
          f"候选规模比 = {g_cand / c_cand:.2f}x")

    print()
    print("=" * 78)
    print("表 4  异常影响半径(§9.8: 级联的分层防护 vs One-Model 的集中风险)")
    print("-" * 78)
    cb, gb = blast_radius("cascade"), blast_radius("generative")
    print(f"{'故障场景':<30}{'级联受影响曝光':>18}{'One-Model 受影响曝光':>22}")
    for key in cb:
        print(f"{key:<30}{cb[key]:>17.0%}{gb[key]:>21.0%}")
    print("-" * 78)
    print(f"平均影响半径: 级联 {sum(cb.values()) / len(cb):.0%}, "
          f"One-Model {sum(gb.values()) / len(gb):.0%}")

    print()
    print("=" * 78)
    print("表 5  §9.2 给出的两种工业部署方式")
    print("-" * 78)
    for name, chain, note in deploy_modes():
        print(f"[{name}] {chain}")
        print(f"    {note}")
    print("-" * 78)
    print("结论(与正文一致): 生成式模块把算力从'候选规模线性'转成'生成长度线性',")
    print("但可控决策点显著减少、异常影响半径放大, 因此工业系统仍保留重排兜底。")


if __name__ == "__main__":
    main()
