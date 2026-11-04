"""
Sparse 特征的参数与存储成本（第15章 §15.2.1、§15.2.2）

正文给出 Embedding 参数量公式：
    N_param = |U| × d
本脚本把用户 ID、物品 ID、作者 ID 等常见 Sparse 特征在不同词表规模/维度下的
参数量与 float32 显存逐一算出来，并对比四种工业压缩手段的节省比例：
    低维 Embedding / 频次截断 / 哈希桶 / int8 量化。

运行：python3 ch02_representation/sparse_dense_cost.py
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import print_table  # noqa: E402

GB = 1024 ** 3


def fancy_vocab_sizes() -> list[tuple[str, int]]:
    return [
        ("user_id", 100_000_000),
        ("device_id", 300_000_000),
        ("item_id", 10_000_000),
        ("author_id", 5_000_000),
        ("category_id", 10_000),
        ("tag_id", 1_000_000),
    ]


def main() -> None:
    """算参数表：词表 × 维度 → 参数量 → float32 显存。"""
    print("一、Sparse 特征的参数量与显存（float32，d=64）")
    rows = []
    total = 0
    for name, vocab in fancy_vocab_sizes():
        params = vocab * 64
        total += params
        rows.append([name, f"{vocab:,}", f"{params:,}", f"{params * 4 / GB:.2f} GB"])
    print_table(["特征", "词表", "参数量", "显存"], rows)
    print(f"合计：{total:,} 个参数，{total * 4 / GB:.2f} GB（仅 Embedding 表）")

    print("\n二、维度 d 的影响（只算 user_id，1 亿词表）")
    rows = []
    for d in (8, 16, 32, 64, 128, 256):
        rows.append([f"d={d}", f"{100_000_000 * d:,}", f"{100_000_000 * d * 4 / GB:.2f} GB"])
    print_table(["维度", "参数量", "显存"], rows)

    print("\n三、四种压缩手段的收益（以 user_id, 1e8 词表, d=64 为基准）")
    base = 100_000_000 * 64 * 4
    rows = [
        ["基线", "1.0x", f"{base / GB:.2f} GB", "无损（就是原表）"],
        ["低维 d=32", "0.5x", f"{base * 0.5 / GB:.2f} GB", "表达力下降，通常可接受"],
        ["频次截断（只保留 30% 活跃 ID）", "0.3x", f"{base * 0.3 / GB:.2f} GB", "长尾 ID 退化为默认向量"],
        ["哈希到 1,000 万桶", "0.1x", f"{base * 0.1 / GB:.2f} GB", "冲突引入噪声，热门 ID 更明显"],
        ["int8 量化", "0.25x", f"{base * 0.25 / GB:.2f} GB", "需反量化，精度略降"],
        ["d=32 + 哈希 + int8", "0.0125x", f"{base * 0.0125 / GB:.2f} GB", "工业常见组合"],
    ]
    print_table(["手段", "相对参数", "显存", "代价"], rows)

    # ---- Dense 特征为什么省：一条统计特征只有 1~几个维度 ----
    print("\n四、Dense 统计特征的规模（对比）")
    dense_fields = 300
    print_table(
        ["项目", "数量", "每维 4 字节"],
        [
            ["Dense 统计特征条数", f"{dense_fields}", "-"],
            ["每条维度（通常 1~3）", "1~3", "-"],
            ["总维度（按 2 维估算）", f"{dense_fields * 2}", f"{dense_fields * 2 * 4} B"],
        ],
    )
    print("结论：ID Embedding 用 25.6 GB 记忆个体，300 条 Dense 统计只占几 KB；")
    print("      两者不是替代关系，而是'记忆 + 泛化'的分工（§15.2.3）。")

    # ---- 长尾与冷启动的量化 ----
    print("\n五、长尾分布下的 Embedding 利用率")
    vocab = 100_000_000
    # 用 Pareto 模拟交互频次，统计覆盖 90% 交互所需的 ID 数
    import random
    rng = random.Random(42)
    freqs = [rng.paretovariate(1.2) for _ in range(200_000)]
    freqs.sort(reverse=True)
    total_freq = sum(freqs)
    acc, n90 = 0.0, 0
    for f in freqs:
        acc += f
        n90 += 1
        if acc >= 0.9 * total_freq:
            break
    print(f"  抽样 20 万 ID 的 Pareto 频次中，{n90 / 200_000:.1%} 的头部 ID 覆盖了 90% 的交互；")
    print(f"  其余 {1 - n90 / 200_000:.1%} 的 ID 共享不到 10% 的交互，其 Embedding 训练严重不足。")
    print("  这正是频次截断与哈希桶的动机：把参数留给有效样本多的 ID。")


if __name__ == "__main__":
    main()
