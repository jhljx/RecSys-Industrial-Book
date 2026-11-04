"""
交互模型统一对比（第15章 §15.5）

把 ch04 下各脚本里的模型放到同一份数据、同一套训练配置下跑一遍，
输出一张汇总表。各模型脚本本身已经自带"与 Baseline 对比"，这个脚本只是
把所有结果并排放，便于横向看"结构差异 → 效果差异"。

注意：这是一次小规模快速实验（4 万训练样本、隐层只有 32 维），
数值只用于演示相对关系，不代表线上收益。

运行：python3 ch04_feature_interaction/interaction_benchmark.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _train_utils import (  # noqa: E402
    N_FIELDS, BaselineMLP, count_params, print_comparison, run_experiment,
)
from afn import AFN  # noqa: E402
from autoint import AutoInt  # noqa: E402
from can import CAN  # noqa: E402
from dcn import DCN  # noqa: E402
from dcn_v2 import DCNV2  # noqa: E402
from fed import FED  # noqa: E402
from fibinet import FiBiNet  # noqa: E402
from fint import FINT  # noqa: E402
from gdcn import GDCN  # noqa: E402
from ipa import IPA  # noqa: E402

STEPS = 600


def main() -> None:
    print(f"交互模型统一对比：F={N_FIELDS}，训练 {STEPS} 步，batch=512\n")
    models = [
        ("Baseline MLP", BaselineMLP),
        ("DCN", lambda: DCN(n_cross=2)),
        ("DCN-V2", lambda: DCNV2(rank=None)),
        ("AutoInt", AutoInt),
        ("FiBiNet (All)", lambda: FiBiNet(mode="all")),
        ("AFN", AFN),
        ("FED", FED),
        ("CAN", CAN),
        ("CAN + Multi-order", lambda: CAN(multi_order=True)),
        ("FINT (2层)", lambda: FINT(n_layers=2)),
        ("GDCN-P", lambda: GDCN(mode="P")),
        ("IPA (projected/field/layer)", IPA),
    ]

    results = []
    t0 = time.time()
    for name, factory in models:
        t = time.time()
        _, auc, _ = run_experiment(factory, name, steps=STEPS)
        results.append((name, auc, count_params(factory())))
        print(f"  -> {name} 用时 {time.time() - t:.1f}s\n")

    print_comparison(results)
    print(f"\n全部实验总耗时 {time.time() - t0:.0f}s。")
    print("\n读数提示：")
    print("  - 差距只有千分之几，说明在这种低维小数据设定下，交互结构的差异")
    print("    很容易被优化噪声淹没；线上评估一定要用全量数据 + 多次重复。")
    print("  - 参数量增长远快于 AUC 增长，这正是正文讨论'表达能力与成本权衡'的原因。")
    print("  - 想观察某类交互的真正效果，应到对应脚本里看它自带的诊断输出")
    print("    （DCN 的交叉项、AutoInt 的注意力图、AFN 的阶数、GDCN 的门控值等）。")


if __name__ == "__main__":
    main()
