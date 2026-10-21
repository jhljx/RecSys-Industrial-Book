r"""对应正文第 9 章 §9.1「生成式推荐概述」: 为什么要先做语义离散化编码(Semantic Tokenization)。

正文要点:
  * 生成式推荐把推荐任务转化为序列生成任务: 给定用户历史行为序列、上下文与画像,
    由模型直接生成用户未来最可能感兴趣的内容标识;
  * 但推荐系统的 Item 规模常达百万甚至亿级, 直接把 Item ID 当生成目标会导致词表
    规模过大、训练与推理成本极高;
  * 因此主流做法是先把 Item 映射为一个或多个语义 ID:
        Item_A -> [s_1, s_2, s_3, s_4]
    训练阶段把用户行为序列转成语义 ID 序列 [s_1, s_2, s_3] -> [s_4], 用 NTP 目标训练;
  * 推理阶段自回归生成 [s_1, s_2] -> [s_3] -> [s_4], 生成完整语义 ID 后再查
    「语义 ID -> Item ID」映射表完成检索;
  * 正文的 4 层 × 512 码字例子: 组合空间 512^4 ≈ 6.87e10, 足以覆盖极大规模物品集。

本实现:
  1. 按正文公式逐项算出「Item ID 词表」与「语义 ID 词表」两种方案的词表规模、
     输出层参数量、softmax 浮点开销与组合空间, 打印对照表(纯解析计算, 无需训练);
  2. 用一个极小的真实模型实测两种输出头的参数量/前向耗时, 验证解析结论;
  3. 演示「行为序列 -> 语义 ID 序列 -> NTP 训练样本 -> 自回归生成」的完整数据形态,
     打印可读的 Token 流。
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn as nn

D_MODEL = 64                     # 演示用隐层维度(正文无具体取值)
L_LAYER = 4                      # 正文例子: 4 层 codebook
K_CODE = 512                     # 正文例子: 每层 512 个码字


def analytic_table():
    """解析对比: 词表规模 N vs. L*K, 输出层参数 = 词表 × d, softmax FLOPs 同量级。"""
    rows = []
    for n_item in (1_000_000, 10_000_000, 100_000_000, 1_000_000_000):
        v_item = n_item                              # 每个 Item 一个 Token
        v_sid = L_LAYER * K_CODE                     # 所有层码字共享/拼接后的词表
        rows.append((n_item, v_item, v_sid,
                     v_item * D_MODEL, v_sid * D_MODEL,
                     v_item / v_sid))
    return rows


class ItemIdHead(nn.Module):
    """[MOCK] 直接以 Item ID 为生成目标的输出头(教学规模: 只取 20 万物品)。"""

    def __init__(self, n_item):
        super().__init__()
        self.proj = nn.Linear(D_MODEL, n_item, bias=False)

    def forward(self, h):
        return self.proj(h)


class SemanticIdHead(nn.Module):
    """以语义 Token 为生成目标的输出头: 词表只有 L*K。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_MODEL, L_LAYER * K_CODE, bias=False)

    def forward(self, h):
        return self.proj(h)


def bench(head, h, n_rep=20):
    with torch.no_grad():
        head(h)                                      # warmup
        t0 = time.perf_counter()
        for _ in range(n_rep):
            head(h).log_softmax(-1)
        return (time.perf_counter() - t0) / n_rep * 1e3


def demo_sequence(gen, n_item=40, hist=3):
    """构造 Item -> 语义 ID 的映射并展示 NTP 样本与自回归生成过程。"""
    sid = torch.stack([torch.randint(0, 8, (n_item,), generator=gen)
                       for _ in range(L_LAYER)], 1)   # 演示用小码本 K=8
    seq = [int(torch.randint(0, n_item, (1,), generator=gen)) for _ in range(hist + 1)]
    return sid, seq


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    print("=" * 96)
    print("表 1  Item ID 词表 vs. 语义 ID 词表(正文例子: L=4 层, 每层 K=512 码字, "
          f"d={D_MODEL})")
    print("-" * 96)
    print(f"{'物品规模 N':>14}{'Item ID 词表':>14}{'语义 Token 词表':>16}"
          f"{'输出层参数(Item ID)':>22}{'输出层参数(SID)':>18}{'缩减倍数':>10}")
    for n, vi, vs, pi, ps, ratio in analytic_table():
        print(f"{n:>14,}{vi:>14,}{vs:>16,}{pi / 1e6:>19.1f}M"
              f"{ps / 1e6:>16.3f}M{ratio:>10.0f}x")
    print("-" * 96)
    print(f"语义 ID 的可表示组合空间 = K^L = {K_CODE}^{L_LAYER} = "
          f"{K_CODE ** L_LAYER:.6g} ≈ {K_CODE ** L_LAYER / 1e10:.2f}e10"
          f"(正文: 512^4 ≈ 6.87e10), 远大于亿级物品集")
    print("代价: 生成一个物品需要 L 次自回归前向, 而 Item ID 方案只需 1 次 —— "
          "这是本章后续\n      §9.7 讨论 MTP/DLM 等并行解码的根本原因。")

    print()
    print("表 2  实测输出头开销(受内存限制, Item ID 方案只取 20 万物品)")
    print("-" * 96)
    n_small = 200_000
    h = torch.randn(32, D_MODEL)
    head_i, head_s = ItemIdHead(n_small), SemanticIdHead()
    pi = sum(p.numel() for p in head_i.parameters())
    ps = sum(p.numel() for p in head_s.parameters())
    ti, ts = bench(head_i, h), bench(head_s, h)
    print(f"{'输出头':<30}{'词表':>12}{'参数量':>14}{'单次前向+softmax(ms)':>24}")
    print(f"{'Item ID 直接生成':<30}{n_small:>12,}{pi:>14,}{ti:>24.3f}")
    print(f"{'语义 Token 生成':<30}{L_LAYER * K_CODE:>12,}{ps:>14,}{ts:>24.3f}")
    print("-" * 96)
    print(f"参数量比 = {pi / ps:.1f}x, 实测耗时比 = {ti / ts:.1f}x"
          f"(仅 20 万物品即已如此, 亿级物品下 Item ID 方案的输出层根本无法放入显存)")

    print()
    print("表 3  数据形态: 行为序列 -> 语义 ID 序列 -> NTP 样本(演示用 L=4, K=8)")
    print("-" * 96)
    sid, seq = demo_sequence(gen)
    for i, it in enumerate(seq):
        code = [int(sid[it, l]) for l in range(L_LAYER)]
        tag = "历史" if i < len(seq) - 1 else "目标"
        print(f"  {tag} Item_{it:<4} -> [" + ", ".join(f"s{l + 1}={c}"
                                                       for l, c in enumerate(code))
              + "]")
    flat = []
    for it in seq[:-1]:
        flat += [f"({l + 1},{int(sid[it, l])})" for l in range(L_LAYER)]
    tgt = [f"({l + 1},{int(sid[seq[-1], l])})" for l in range(L_LAYER)]
    print()
    print("  NTP 输入(flatten 后的语义 Token 序列):")
    print("    " + " ".join(flat))
    print("  NTP 预测目标:")
    print("    " + " ".join(tgt))
    print()
    print("  自回归生成过程(正文 [s1,s2] -> [s3] -> [s4]):")
    ctx = list(flat)
    for step, t in enumerate(tgt):
        print(f"    第 {step + 1} 次前向: 输入 ...{' '.join(ctx[-3:])}  =>  生成 {t}")
        ctx.append(t)
    print(f"  得到完整语义 ID {''.join(tgt)} 后, 查映射表恢复 Item_{seq[-1]}")
    print("-" * 96)
    print("这正是正文概述的闭环: 语义离散化 -> NTP 训练 -> 自回归生成 -> SID 查表检索。")


if __name__ == "__main__":
    main()
