"""
对应正文第 4 章 §4.5.4「风控算法的发展」与 §4.5.5「过滤模块面临的挑战」。

正文口径(逐条对齐):
  阶段一「任务专用模型」:针对不同风险类型分别训练独立模型(色情识别、暴力识别、
    OCR 文本识别、人脸检测、Deepfake 检测等),常见骨干网络为 ResNet / MobileNet /
    EfficientNet / ViT。优点是单任务精度高;缺点是模型数量多、维护成本高,
    出现新的风险类型时需要重新构建数据集并训练新模型。
  阶段二「统一多模态建模」:以 CLIP / BLIP / Qwen-VL 为代表,同时理解图像、视频、
    语音、文本,并把不同模态映射到统一语义空间;不再为每种风险单独训练模型,
    而是用图文相似度判断色情内容、用跨模态语义理解发现隐晦违规内容。
  §4.5.5 的挑战:跨模态违规(画面合规但语音敏感)、新型风险演化快、误伤优质内容。

本脚本做了什么(全部用合成数据,不加载任何真实权重):
  1) 构造一个「统一语义空间」的生成过程:每个物品先有一个语义隐向量 z,
     风险概念 c_k 以不同方式注入不同模态 --
        涉黄 / 涉暴  -> 画面 + 文本都能看到
        涉政(OCR)    -> 只在文本
        Deepfake     -> 只在画面
        隐晦跨模态   -> 只在语音(画面与标题完全合规),即正文 §4.5.5 的典型难例
     然后用模态专属的随机线性映射 + 噪声生成三路「特征」(图像 / 文本 / 语音)。
  2) 阶段一:为前 4 类风险各训一个单模态逻辑回归(= 任务专用小模型的最小版本),
     每个模型只看自己那一路模态特征;第 5 类是上线后才出现的新风险,没有专用模型。
  3) 阶段二:用对比学习(InfoNCE,CLIP 的核心损失)把三路模态编码到同一个 D 维
     语义空间,然后用每类风险 8 条标注样本构造「风险原型」做少样本检索式判别:
        score_k(item) = max_{m∈{图,文,音}} cos(e_m(item), proto_{k,m})
     新风险类型无需重新训练主干,只需加一组原型 -- 对应正文「泛化能力与迭代效率」。
  4) 两段式/三段式审核链路(规则 + 模型):
        规则硬拦截(关键词命中,高精度低召回,成本最低)
          -> 专用小模型粗筛(高召回低阈值)
          -> 统一大模型精判(成本最高)
     打印各配置的精确率 / 召回率 / 相对算力成本,并单独看「隐晦跨模态」这类
     新型风险在各配置下的召回,解释正文为什么要走向统一建模。

关键公式:
  InfoNCE(对比对齐):  L = -1/B Σ_i log[ exp(e_i·g_i/τ) / Σ_j exp(e_i·g_j/τ) ]
  逻辑回归:            p = σ(w·x + b),  L = -[y log p + (1-y) log(1-p)]
  原型判别:            score_k = max_m cos(e_m, proto_{k,m})
  多风险合并(定标):   z_k = (score_k - μ_k^neg) / σ_k^neg,  s = max_k z_k
  级联成本:            Cost = Σ_stage (进入该阶段的样本数 × 单位成本)

[MOCK] 所有「模型」都是几十到上千参数的玩具模型,只用来复现正文描述的能力差异;
真实的 ResNet / ViT / CLIP / Qwen-VL 需要海量数据与算力,本脚本不下载任何权重。
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_SEM = 16          # 统一语义空间维度
D_RAW = 24          # 每路模态的原始特征维度
N_ITEMS = 2400      # 合成内容条数
TRAIN_RATIO = 0.7
NOISE = 0.2         # 模态观测噪声
TAU = 0.1           # InfoNCE 温度
UNI_STEPS = 1200    # 统一编码器对比训练步数
N_SHOT = 8          # 统一模型每类风险的标注样本数(少样本原型)

MODS = ["图像", "文本", "语音"]
RISKS = ["涉黄", "涉暴", "涉政(OCR)", "Deepfake", "隐晦跨模态"]
# 每类风险注入到哪些模态(正文 §4.5.5:画面合规但语音敏感 -> 第 5 类只进语音)
RISK_MODS = {0: [0, 1], 1: [0, 1], 2: [1], 3: [0], 4: [2]}
RISK_RATE = [0.06, 0.05, 0.04, 0.03, 0.05]
# 阶段一为前 4 类风险各建一个专用模型:色情/暴力/Deepfake 看画面,涉政看 OCR 文本;
# 第 5 类风险是「上线后才出现的新风险」,阶段一没有对应模型。
EXPERT_MOD = {0: 0, 1: 0, 2: 1, 3: 0}


# ============================================================ 合成数据
def gen_dataset(rng_np):
    """[MOCK] 生成三模态特征 + 多标签风险 + 关键词命中标记。"""
    concepts = rng_np.normal(size=(len(RISKS), D_SEM))
    concepts /= np.linalg.norm(concepts, axis=1, keepdims=True)
    # 模态专属映射:同一语义在不同模态下的「观测方式」不同
    proj = [rng_np.normal(size=(D_SEM, D_RAW)) / np.sqrt(D_SEM) for _ in MODS]

    labels = np.zeros((N_ITEMS, len(RISKS)), dtype=np.float32)
    for k, p in enumerate(RISK_RATE):
        labels[:, k] = (rng_np.random(N_ITEMS) < p).astype(np.float32)

    base = rng_np.normal(size=(N_ITEMS, D_SEM))
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    feats = []
    for m in range(len(MODS)):
        z = base.copy()
        for k in range(len(RISKS)):
            if m in RISK_MODS[k]:
                strength = 1.1 + 0.3 * rng_np.random((N_ITEMS, 1))
                z = z + labels[:, k:k + 1] * strength * concepts[k]
        x = z @ proj[m] + NOISE * rng_np.normal(size=(N_ITEMS, D_RAW))
        feats.append(x.astype(np.float32))

    # 规则层:关键词黑名单命中。文本可见的风险容易被命中,隐晦跨模态几乎命中不了
    kw = np.zeros(N_ITEMS, dtype=np.float32)
    for i in range(N_ITEMS):
        if labels[i, 0] or labels[i, 1] or labels[i, 2]:
            p = 0.45
        elif labels[i, 4]:
            p = 0.05
        elif labels[i, 3]:
            p = 0.08
        else:
            p = 0.02
        kw[i] = 1.0 if rng_np.random() < p else 0.0
    return feats, labels, kw


def auc(y, s):
    """基于秩的 AUC:AUC = (Σ_{正样本} rank - n_pos(n_pos+1)/2) / (n_pos·n_neg)。"""
    y = np.asarray(y)
    s = np.asarray(s)
    pos, neg = float(y.sum()), float((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def recall_at_fpr(y, s, fpr=0.05):
    """在给定误伤率(FPR)下的召回率 -- 正文 §4.5.5「误伤优质内容」的量化口径。"""
    y = np.asarray(y)
    s = np.asarray(s)
    neg = s[y == 0]
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr)
    return float((s[y == 1] >= thr).mean())


# ============================================================ 阶段一:任务专用模型
def train_expert(x_tr, y_tr, epochs=200):
    """单模态逻辑回归 = 任务专用小模型的最小版本(正文:每种风险一个独立模型)。"""
    model = nn.Linear(x_tr.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.08)
    pos_w = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum().item(), 1.0)])
    for _ in range(epochs):
        opt.zero_grad()
        logit = model(x_tr).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logit, y_tr, pos_weight=pos_w)
        loss.backward()
        opt.step()
    return model


# ============================================================ 阶段二:统一多模态模型
class UnifiedEncoder(nn.Module):
    """[MOCK] CLIP 式统一语义空间:每个模态一个投影头,用 InfoNCE 对齐同一条内容。"""

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(D_RAW, D_SEM) for _ in MODS])

    def forward(self, xs):
        return [F.normalize(h(x), dim=1) for h, x in zip(self.heads, xs)]


def infonce(a, b, tau=TAU):
    logits = a @ b.t() / tau
    target = torch.arange(len(a))
    return 0.5 * (F.cross_entropy(logits, target)
                  + F.cross_entropy(logits.t(), target))


def train_unified(xs_tr, steps=UNI_STEPS, bs=256, lr=0.05):
    """三路模态两两对齐,完全不使用风险标签(自监督,对应 CLIP 的训练范式)。"""
    model = UnifiedEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(xs_tr[0])
    last = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),))
        e = model([x[idx] for x in xs_tr])
        loss = (infonce(e[0], e[1]) + infonce(e[0], e[2]) + infonce(e[1], e[2])) / 3
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss)
    return model, last


def build_prototypes(emb_tr, y_tr, rng, n_shot=N_SHOT):
    """每类风险用 n_shot 条标注样本构造各模态原型(新风险上线只需补这一步)。"""
    protos = {}
    for k in range(len(RISKS)):
        pos = [i for i in range(len(y_tr)) if y_tr[i, k] > 0]
        rng.shuffle(pos)
        shot = pos[:n_shot]
        protos[k] = [e[shot].mean(0) / max(np.linalg.norm(e[shot].mean(0)), 1e-9)
                     for e in emb_tr]
    return protos


def proto_score(emb, protos, k):
    """score_k = max_m cos(e_m, proto_{k,m}) -- 统一语义空间里的检索式判别。"""
    sims = np.stack([e @ protos[k][m] for m, e in enumerate(emb)], axis=1)
    return sims.max(axis=1)


def merge_risks(score_tr, score_te, y_tr_any):
    """多风险分数合并:先用训练集负样本把每路分数定标,再取 max。

    不同风险头的分数量纲不同,直接取 max 会被量纲最大的那一路劫持,
    因此按 z_k = (s_k - μ_k^neg) / σ_k^neg 定标后再取最大值。
    """
    neg = score_tr[y_tr_any == 0]
    mu, sd = neg.mean(axis=0), neg.std(axis=0) + 1e-9
    return ((score_tr - mu) / sd).max(axis=1), ((score_te - mu) / sd).max(axis=1)


# ============================================================ 规则 + 模型级联
def cascade_eval(name, y_any, risk_new, kw, small, big, thr_small, thr_big,
                 use_rule, use_small, use_big,
                 cost_rule=1.0, cost_small=10.0, cost_big=200.0):
    """按「规则 -> 小模型 -> 大模型」逐级计算判定结果与算力成本。

    返回 (名称, 精确率, 召回率, 新型风险召回, 总成本, 大模型调用占比)。
    """
    n = len(y_any)
    cost = 0.0
    pred = np.zeros(n, dtype=bool)

    if use_rule:
        cost += cost_rule * n            # 规则层:全量都要过,但单位成本最低
        hit_rule = kw > 0
        pred |= hit_rule
        rest = ~hit_rule                 # 规则放过的才进入下游模型
    else:
        rest = np.ones(n, dtype=bool)

    if use_small:
        cost += cost_small * rest.sum()  # 专用小模型粗筛(低阈值,召回优先)
        rest = rest & (small >= thr_small)

    if use_big:
        cost += cost_big * rest.sum()    # 统一大模型精判
        pred |= rest & (big >= thr_big)
        big_calls = rest.sum()
    elif use_small:
        pred |= rest                     # 没有大模型时,小模型粗筛结果直接生效
        big_calls = 0
    else:
        big_calls = 0

    tp = float((pred & (y_any > 0)).sum())
    prec = tp / max(pred.sum(), 1)
    rec = tp / max(float((y_any > 0).sum()), 1)
    rec_new = (float((pred & (risk_new > 0)).sum())
               / max(float((risk_new > 0).sum()), 1))
    return name, prec, rec, rec_new, cost, big_calls / n


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    rng = random.Random(42)
    rng_np = np.random.default_rng(42)

    feats, labels, kw = gen_dataset(rng_np)
    n_tr = int(N_ITEMS * TRAIN_RATIO)
    xs_tr = [torch.from_numpy(f[:n_tr]) for f in feats]
    xs_te = [torch.from_numpy(f[n_tr:]) for f in feats]
    y_tr, y_te = labels[:n_tr], labels[n_tr:]
    kw_te = kw[n_tr:]

    print("=" * 92)
    print("§4.5.4 风控算法的发展:任务专用模型 -> 统一多模态建模,以及规则+模型级联链路")
    print("=" * 92)
    print(f"\n[MOCK] 合成内容 {N_ITEMS} 条(训练 {n_tr} / 测试 {N_ITEMS - n_tr}),"
          f"三路模态特征各 {D_RAW} 维,统一语义空间 {D_SEM} 维")
    print(f"{'风险类型':<14}{'信号出现的模态':<30}{'训练集占比':>10}{'测试集占比':>10}")
    print("-" * 68)
    for k, name in enumerate(RISKS):
        mods = "+".join(MODS[m] for m in RISK_MODS[k])
        tag = "" if k in EXPERT_MOD else "(新风险,阶段一无专用模型)"
        print(f"{name:<14}{mods + tag:<30}{y_tr[:, k].mean():>10.2%}"
              f"{y_te[:, k].mean():>10.2%}")

    # ---------------- 阶段一:每类风险一个单模态专用模型
    experts = {k: train_expert(xs_tr[m], torch.from_numpy(y_tr[:, k]))
               for k, m in EXPERT_MOD.items()}
    exp_tr = np.zeros((n_tr, len(EXPERT_MOD)), dtype=np.float32)
    exp_te = np.zeros((N_ITEMS - n_tr, len(EXPERT_MOD)), dtype=np.float32)
    for col, k in enumerate(sorted(EXPERT_MOD)):
        m = EXPERT_MOD[k]
        with torch.no_grad():
            exp_tr[:, col] = experts[k](xs_tr[m]).squeeze(1).numpy()
            exp_te[:, col] = experts[k](xs_te[m]).squeeze(1).numpy()

    # ---------------- 阶段二:统一多模态编码 + 少样本原型
    uni, last_loss = train_unified(xs_tr)
    with torch.no_grad():
        emb_tr = [e.numpy() for e in uni(xs_tr)]
        emb_te = [e.numpy() for e in uni(xs_te)]
    protos = build_prototypes(emb_tr, y_tr, rng)
    uni_tr = np.stack([proto_score(emb_tr, protos, k) for k in range(len(RISKS))], 1)
    uni_te = np.stack([proto_score(emb_te, protos, k) for k in range(len(RISKS))], 1)

    print(f"\n[统一语义空间训练] InfoNCE(τ={TAU}) 三路两两对齐 {UNI_STEPS} 步, "
          f"末步 loss={last_loss:.4f};每类风险仅用 {N_SHOT} 条标注样本建原型")
    print(f"{'风险类型':<14}{'专用模型AUC':>13}{'专用Recall@FPR5%':>18}"
          f"{'统一模型AUC':>13}{'统一Recall@FPR5%':>18}")
    print("-" * 78)
    for k, name in enumerate(RISKS):
        if k in EXPERT_MOD:
            col = sorted(EXPERT_MOD).index(k)
            s1 = exp_te[:, col]
        else:
            # 新风险没有专用模型,只能借用现有模型里表现最好的那个
            best = max(range(len(EXPERT_MOD)),
                       key=lambda c: auc(y_te[:, k], exp_te[:, c]))
            s1 = exp_te[:, best]
        print(f"{name:<14}{auc(y_te[:, k], s1):>13.4f}"
              f"{recall_at_fpr(y_te[:, k], s1):>18.2%}"
              f"{auc(y_te[:, k], uni_te[:, k]):>13.4f}"
              f"{recall_at_fpr(y_te[:, k], uni_te[:, k]):>18.2%}")
    print(f"  注:「隐晦跨模态」行的专用模型列是借用现有 {len(EXPERT_MOD)} 个模型里"
          f"表现最好的那个(画面/文本都看不到该风险),")
    print("     这正是正文说的「出现新的风险类型时需要重新构建数据集并训练新模型」。")

    # ---------------- 维护成本对比
    n_expert_params = sum(sum(p.numel() for p in m.parameters())
                          for m in experts.values())
    n_uni_params = sum(p.numel() for p in uni.parameters())
    print("\n[维护成本] 正文:专用模型方案「模型数量众多、维护成本较高」")
    print(f"{'方案':<22}{'模型个数':>10}{'主干参数量':>12}{'新增一类风险的边际成本':>26}")
    print("-" * 70)
    print(f"{'阶段一 任务专用模型':<22}{len(experts):>10}{n_expert_params:>12}"
          f"{'新建数据集+训练第N+1个模型':>26}")
    print(f"{'阶段二 统一多模态模型':<22}{1:>10}{n_uni_params:>12}"
          f"{f'仅补{N_SHOT}条样本做原型/Prompt':>26}")

    # ---------------- 规则 + 模型级联
    y_tr_any = (y_tr.sum(axis=1) > 0).astype(np.float32)
    y_te_any = (y_te.sum(axis=1) > 0).astype(np.float32)
    risk_new = y_te[:, len(RISKS) - 1]
    small_tr, small_te = merge_risks(exp_tr, exp_te, y_tr_any)
    big_tr, big_te = merge_risks(uni_tr, uni_te, y_tr_any)

    # 阈值全部在训练集上确定(线上不可能用测试集调阈值)
    thr_big = float(np.quantile(big_tr[y_tr_any == 0], 0.97))   # 精判:负样本 FPR 3%
    thr_small = float(np.quantile(small_tr[y_tr_any == 0], 0.60))  # 粗筛:召回优先

    print(f"\n[合并多风险分数] 训练集负样本定标后取 max;"
          f"合并分 AUC:专用模型 {auc(y_te_any, small_te):.4f} / "
          f"统一模型 {auc(y_te_any, big_te):.4f}")
    rows = [
        cascade_eval("A 仅规则(关键词)", y_te_any, risk_new, kw_te, small_te,
                     big_te, thr_small, thr_big, True, False, False),
        cascade_eval("B 规则->专用小模型", y_te_any, risk_new, kw_te, small_te,
                     big_te, float(np.quantile(small_tr[y_tr_any == 0], 0.97)),
                     thr_big, True, True, False),
        cascade_eval("C 全量过大模型", y_te_any, risk_new, kw_te, small_te,
                     big_te, thr_small, thr_big, False, False, True),
        cascade_eval("D 规则->大模型", y_te_any, risk_new, kw_te, small_te,
                     big_te, thr_small, thr_big, True, False, True),
        cascade_eval("E 规则->小模型->大模型", y_te_any, risk_new, kw_te, small_te,
                     big_te, thr_small, thr_big, True, True, True),
    ]
    base_cost = next(r[4] for r in rows if r[0].startswith("C"))
    print("\n[审核链路级联] 单位成本:规则 1 / 专用小模型 10 / 统一大模型 200")
    print(f"{'配置':<24}{'精确率':>9}{'召回率':>9}{'隐晦跨模态召回':>16}"
          f"{'相对成本':>10}{'大模型调用占比':>16}")
    print("-" * 86)
    for name, prec, rec, rec_new, cost, big_ratio in rows:
        print(f"{name:<24}{prec:>9.2%}{rec:>9.2%}{rec_new:>16.2%}"
              f"{cost / base_cost:>10.2f}{big_ratio:>16.2%}")
    print(f"  (测试集违规内容 {int(y_te_any.sum())} 条 / 共 {len(y_te_any)} 条,"
          f"其中隐晦跨模态 {int(risk_new.sum())} 条;")
    print("   大模型阈值 = 训练集负样本 97 分位,小模型粗筛阈值 = 60 分位,"
          "成本以配置 C 为 1.00)")

    print("\n结论(正文 §4.5.4 / §4.5.5 的工程含义):")
    print("  1) 任务专用模型在「自己能看到的模态」上精度很高(AUC 0.97+),但对未建模")
    print("     的新风险几乎等于随机 -- 每新增一类风险都要重新标数据、训模型、上线服务;")
    print("  2) 统一语义空间(CLIP 式对比对齐)只靠自监督对齐三路模态,不看风险标签,")
    print("     新风险只需少量样本建原型即可检索出来,这就是正文强调的泛化与迭代效率;")
    print("  3) 跨模态违规(画面合规·语音敏感)是单模态专用模型的结构性盲区,")
    print("     级联链路若用旧的小模型做第一道粗筛,会在粗筛阶段就把新型风险漏掉;")
    print("  4) 成本角度:规则层最便宜、精度最高但召回有限,大模型最贵,")
    print("     级联的意义是把昂贵的大模型调用压到少数疑似样本上;")
    print("  5) 阈值即「安全 vs 误伤」的旋钮,Recall@FPR 这类指标才是对齐业务的口径。")


if __name__ == "__main__":
    main()
