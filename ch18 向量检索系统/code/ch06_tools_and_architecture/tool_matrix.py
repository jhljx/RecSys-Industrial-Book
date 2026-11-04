"""
向量检索工具选型：把定性描述变成可打分的决策表（第18章 §18.6）

正文按「ANN 库 / 向量数据库」两条线介绍 Faiss、ScaNN、DiskANN、Milvus、
Qdrant、Weaviate、Pinecone、Elasticsearch。这些比较是定性的、且会随版本
快速变化，本脚本**不联网、不安装任何库**，只是把正文的判断结构化成一张
加权评分表，方便在真实选型时替换成自己的权重。

⚠️ 表中所有「得分」都是 [MOCK] 人工设定的相对值（1~5 分），
   **不是**任何基准测试的结果，也不能当作性能承诺。用途只有两个：
   1) 演示「先定约束、再加权打分」的选型方法；
   2) 让不同业务画像（小规模实验 / 十亿级 / RAG）算出不同结论。

运行：python3 ch06_tools_and_architecture/tool_matrix.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common import table  # noqa: E402

# 维度说明（全部为 [MOCK] 主观分，1 差 ~ 5 好）
DIMS = ["部署成本", "运维复杂度", "超大规模", "过滤/混合查询", "生态成熟度"]

# 每个工具在各维度的得分 + 一句定位
TOOLS: dict[str, tuple[list[int], str]] = {
    "Faiss": ([5, 2, 4, 1, 5],
              "ANN 算法库，不是数据库；索引丰富、性能强，但不带持久化/分布式"),
    "ScaNN": ([4, 3, 3, 1, 2],
              "Google 系 CPU 优化 + AVQ，内积检索精度好；生态较窄"),
    "DiskANN": ([3, 4, 5, 1, 2],
                "SSD 承载图索引，十亿~百亿级、内存受限场景的首选"),
    "Milvus": ([2, 3, 5, 4, 5],
               "分布式向量数据库，索引种类全，运维组件较多"),
    "Qdrant": ([4, 4, 3, 5, 4],
               "Rust 实现、默认 HNSW，Payload 过滤强，适合 RAG"),
    "Weaviate": ([3, 3, 3, 4, 3],
                 "知识图谱 + 向量检索融合，Schema 管理丰富"),
    "Pinecone": ([5, 5, 4, 3, 3],
                 "托管 SaaS，免运维；成本与数据出域是主要约束"),
    "Elasticsearch": ([3, 2, 3, 5, 5],
                      "全文 + 向量混合检索成熟，纯向量性能不是强项"),
}

# 三份业务画像：权重体现「哪个维度最重要」
PROFILES: dict[str, dict[str, float]] = {
    "小规模实验（<100 万，单机）": {
        "部署成本": 5, "运维复杂度": 4, "超大规模": 1,
        "过滤/混合查询": 2, "生态成熟度": 3,
    },
    "十亿级线上召回": {
        "部署成本": 1, "运维复杂度": 2, "超大规模": 5,
        "过滤/混合查询": 2, "生态成熟度": 4,
    },
    "RAG 知识库（带元数据过滤）": {
        "部署成本": 3, "运维复杂度": 4, "超大规模": 1,
        "过滤/混合查询": 5, "生态成熟度": 3,
    },
}


def score(tool: str, weights: dict[str, float]) -> float:
    """加权平均，归一化到 0~5。"""
    s, w = TOOLS[tool][0], [weights[d] for d in DIMS]
    return sum(a * b for a, b in zip(s, w)) / sum(w)


def rank(weights: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(((t, score(t, weights)) for t in TOOLS),
                  key=lambda x: -x[1])


def main() -> None:
    print("向量检索工具选型：加权打分（§18.6）\n")
    print("⚠️ 所有得分均为 [MOCK] 主观分，仅用于演示选型方法，不是性能结论。\n")

    print("① 评分表（1 差 ~ 5 好）：")
    rows = [[t] + [str(v) for v in TOOLS[t][0]] for t in TOOLS]
    table(["工具"] + DIMS, rows)

    print("\n② 各工具定位（正文的一句话摘要）：")
    for t, (_, note) in TOOLS.items():
        print(f"  · {t:<14}{note}")

    print("\n③ 三份业务画像下的排名（同一个库，权重不同，结论就不同）：")
    rows = []
    names = list(TOOLS)
    scores = {p: rank(w) for p, w in PROFILES.items()}
    for i in range(len(names)):
        row = [f"#{i + 1}"]
        for p in PROFILES:
            t, s = scores[p][i]
            row += [t, f"{s:.2f}"]
        rows.append(row)
    header = ["名次"]
    for p in PROFILES:
        header += [p, "得分"]
    table(header, rows)

    print("\n  三列的 Top-3 只有部分重合（Pinecone 与 Qdrant 在两个画像里都靠前），")
    print("  但第一名分别是 Pinecone / Milvus / Qdrant——「哪个向量库最好」")
    print("  这个问题本身没有意义：先写下约束（规模、能否上云、是否要元数据过滤、")
    print("  团队运维能力），再让权重替你选。这也是正文把工具分成")
    print("  「ANN 库」与「向量数据库」两层的原因：它们解决的不是同一个问题。")
    print("  另外注意「小规模实验」那列的第一名是托管 SaaS，说明这套权重把")
    print("  「少运维」看得比「零成本」重；真正做实验时，成本权重通常要调高。\n")

    print("④ 换维度的敏感性：固定「十亿级线上召回」画像，去掉运维维度")
    w = dict(PROFILES["十亿级线上召回"])
    before = [t for t, _ in rank(w)][:3]
    w["运维复杂度"] = 0
    after = [t for t, _ in rank(w)][:3]
    print(f"  原 Top-3：{', '.join(before)}")
    print(f"  去掉运维后 Top-3：{', '.join(after)}")
    print("  排名变化本身就提示了风险：如果某维度你没把握，看看结论会不会翻转。")
    print("  真实选型应该这样逐维做敏感性分析，而不是只算一次总分。\n")

    print("⑤ 一句话：这份表的作用不是给出答案，而是把「我觉得 X 更好」")
    print("   拆成「在哪个维度、以什么权重、赢了谁」——可复核，才可决策。")


if __name__ == "__main__":
    main()
