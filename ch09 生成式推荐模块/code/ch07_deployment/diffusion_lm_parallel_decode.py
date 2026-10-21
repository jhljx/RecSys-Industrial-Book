r"""对应正文第 9 章 §9.7 中的 Diffusion Language Model(DLM)并行生成。

正文要点与公式:
  * DLM 把离散 Token 序列的生成建模为逐步去噪(Denoising)过程: 训练阶段向真实
    Token 序列不断注入噪声得到不同噪声程度的中间状态; 推理阶段从随机噪声或
    完全 Mask 的序列开始, 经多步去噪恢复最终序列:
        x_T -> x_{T-1} -> ... -> x_1 -> x_0
    其中 x_0 为最终生成的 Token 序列;
  * 每一步去噪可同时预测多个位置的 Token, 因此天然具备并行生成能力, 不必像
    自回归那样严格按 Token 顺序生成, 显著减少串行依赖、提高生成速度;
  * 但离散 Token 上的扩散过程设计复杂, 推理仍需多个去噪步骤, 目前在生成质量与
    推理效率之间尚未达到最优平衡。

本实现(掩码离散扩散, 教学规模):
  1. 合成数据特意制造"位置之间必须协同"的结构: 待生成的 2 个物品中, 第 1 个物品的
     首层码字有两种等概率分支(模型无法从历史确定), 而物品内的第 2、3 层码字由首层
     唯一决定, 第 2 个物品的首层码字 = 第 1 个物品首层码字 + 1 (mod K);
  2. 训练: 按比例 t/T 随机 Mask 待生成段, 双向 Transformer 预测被 Mask 位置的原
     Token, 即学习单步去噪器 x_t -> x_0;
  3. 推理: 从全 Mask 出发, 每步并行预测所有 Mask 位置, 按置信度把最有把握的若干
     位置"定稿"(low-confidence remasking), S 步得到 x_0;
  4. 用"采样解码"度量协同能力: 合法 SID 率(生成三元组能在物品表中找到)与跨物品
     一致性率。S=1(完全并行)时各位置相互独立, 这两个指标显著变差; S 增大后已定稿
     Token 成为后续位置的条件, 指标回升到自回归水平 —— 即质量与效率的权衡。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

L_LAYER = 3
K_CODE = 10
N_ITEM = K_CODE                 # 物品由首层码字唯一决定, 共 K_CODE 个物品
SEQ_ITEM = 5                    # 历史物品数
GEN_ITEM = 2                    # 并行生成的物品数
GEN_LEN = L_LAYER * GEN_ITEM
V_SID = L_LAYER * K_CODE
MASK = V_SID                    # [MASK] 特殊 Token
VOCAB = V_SID + 1
D_MODEL = 40
T_STEP = 12
BRANCH = 5                      # 分支偏移: c_A ∈ {fav, (fav+BRANCH) % K}


def flat(layer, code):
    return layer * K_CODE + code


class Denoiser(nn.Module):
    """双向 Transformer 去噪器: 输入含 [MASK] 的序列, 预测每个位置的原 Token。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(64, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 80, batch_first=True,
                                           dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(D_MODEL, VOCAB)
        self.n_fwd = 0

    def forward(self, x):
        self.n_fwd += 1
        p = torch.arange(x.shape[1]).unsqueeze(0)
        return self.out(self.enc(self.emb(x) + self.pos(p)))


def build_items(gen):
    """物品 c 的语义 ID = (c, tab1[c], tab2[c]): 首层码字唯一决定后两层。"""
    tab1 = torch.randperm(K_CODE, generator=gen)
    tab2 = torch.randperm(K_CODE, generator=gen)
    return torch.stack([torch.arange(K_CODE), tab1, tab2], 1)


def item_tokens(sid, c):
    return [flat(l, int(sid[c, l])) for l in range(L_LAYER)]


def build_data(sid, gen, n_seq=320):
    rows, favs = [], []
    for _ in range(n_seq):
        fav = int(torch.randint(0, K_CODE, (1,), generator=gen))
        row = []
        for _ in range(SEQ_ITEM):                         # 历史: 全是 fav 物品
            row += item_tokens(sid, fav)
        b = int(torch.randint(0, 2, (1,), generator=gen)) # 等概率分支, 不可预测
        c_a = fav if b == 0 else (fav + BRANCH) % K_CODE
        c_b = (c_a + 1) % K_CODE                          # 第 2 个物品由第 1 个决定
        row += item_tokens(sid, c_a) + item_tokens(sid, c_b)
        rows.append(row)
        favs.append(fav)
    return torch.tensor(rows), torch.tensor(favs)


def train(data, epochs=110):
    torch.manual_seed(42)
    model = Denoiser()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    ctx = SEQ_ITEM * L_LAYER
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        t = torch.randint(1, T_STEP + 1, (data.shape[0], 1))
        m = torch.rand(data.shape[0], GEN_LEN) < t.float() / T_STEP
        m[:, 0] |= ~m.any(1)                              # 至少 Mask 一个位置
        tail = data[:, ctx:].clone()
        tail[m] = MASK
        logits = model(torch.cat([data[:, :ctx], tail], 1))[:, ctx:]
        loss = F.cross_entropy(logits[m], data[:, ctx:][m])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, float(loss)


@torch.no_grad()
def dlm_generate(model, ctx_tok, steps, sample=False, trace=False):
    """从全 Mask 出发做 steps 步并行去噪, 每步定稿置信度最高的若干位置。"""
    model.n_fwd = 0
    B, ctx = ctx_tok.shape[0], ctx_tok.shape[1]
    cur = torch.full((B, GEN_LEN), MASK, dtype=torch.long)
    frozen = torch.zeros(B, GEN_LEN, dtype=torch.bool)
    traj = []
    for s in range(steps):
        logits = model(torch.cat([ctx_tok, cur], 1))[:, ctx:, :V_SID]
        prob = logits.softmax(-1)
        if sample:                                        # 采样解码(DLM 常规做法)
            pick = torch.multinomial(prob.reshape(-1, V_SID), 1).reshape(B, GEN_LEN)
            conf = prob.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
        else:
            conf, pick = prob.max(-1)
        conf = conf.masked_fill(frozen, -1.0)
        n_left = GEN_LEN - int(frozen[0].sum())
        n_fix = max(1, -(-n_left // (steps - s)))         # 本步定稿个数
        idx = conf.topk(min(n_fix, n_left), dim=1).indices
        newf = frozen.clone()
        newf.scatter_(1, idx, True)
        cur = torch.where(newf & ~frozen, pick, cur)
        frozen = newf
        if trace:
            traj.append((s + 1, cur[0].tolist(), frozen[0].tolist()))
        if bool(frozen.all()):
            break
    if not bool(frozen.all()):                            # 收尾填满剩余位置
        logits = model(torch.cat([ctx_tok, cur], 1))[:, ctx:, :V_SID]
        cur = torch.where(frozen, cur, logits.argmax(-1))
    return cur, model.n_fwd, traj


@torch.no_grad()
def ar_generate(model, ctx_tok, sample=False):
    """自回归基线: 同一个模型, 每次只定稿最左边一个位置(严格串行, GEN_LEN 次前向)。"""
    model.n_fwd = 0
    ctx = ctx_tok.shape[1]
    cur = torch.full((ctx_tok.shape[0], GEN_LEN), MASK, dtype=torch.long)
    for j in range(GEN_LEN):
        logits = model(torch.cat([ctx_tok, cur], 1))[:, ctx + j, :V_SID]
        if sample:
            cur[:, j] = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
        else:
            cur[:, j] = logits.argmax(-1)
    return cur, model.n_fwd


def metrics(pred, sid, favs):
    """合法 SID 率、跨物品一致性率、分支合法率。"""
    table = {tuple(item_tokens(sid, c)): c for c in range(K_CODE)}
    legal, consist, branch = 0, 0, 0
    n = pred.shape[0] * GEN_ITEM
    for b in range(pred.shape[0]):
        cs = []
        for g in range(GEN_ITEM):
            tri = tuple(int(v) for v in pred[b, g * L_LAYER:(g + 1) * L_LAYER])
            if tri in table:
                legal += 1
                cs.append(table[tri])
            else:
                cs.append(None)
        if cs[0] is not None and cs[1] is not None and cs[1] == (cs[0] + 1) % K_CODE:
            consist += 1
        f = int(favs[b])
        if cs[0] in (f, (f + BRANCH) % K_CODE):
            branch += 1
    return legal / n, consist / pred.shape[0], branch / pred.shape[0]


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)
    sid = build_items(gen)
    data, favs = build_data(sid, gen)
    ctx = SEQ_ITEM * L_LAYER
    model, last_loss = train(data)

    ctx_tok, gold, fv = data[:256, :ctx], data[:256, ctx:], favs[:256]
    print("=" * 96)
    print(f"配置: 语义 Token 词表 {V_SID}(L={L_LAYER} × K={K_CODE}) + [MASK]; "
          f"条件长度 {ctx}; 并行生成 {GEN_LEN} 个 Token(= {GEN_ITEM} 个物品)")
    print(f"数据结构: 历史全是物品 fav -> 待生成物品 A 的首层码字 ∈ "
          f"{{fav, (fav+{BRANCH}) mod {K_CODE}}}(等概率, 不可预测), "
          f"物品 B 首层码字 = A + 1")
    print(f"训练: 掩码离散扩散, 最大噪声级别 T={T_STEP}, "
          f"末次 batch 去噪损失 = {last_loss:.4f}")

    print()
    print("表 1  贪心(argmax)解码: 去噪步数 S 对 Token 准确率的影响")
    print("-" * 96)
    print(f"{'生成方式':<30}{'前向次数':>10}{'串行步数':>10}"
          f"{'Token 准确率':>14}{'合法 SID 率':>14}{'跨物品一致性':>14}")
    pa, fa = ar_generate(model, ctx_tok)
    la, ca, _ = metrics(pa, sid, fv)
    print(f"{'自回归逐 Token':<30}{fa:>10}{GEN_LEN:>10}"
          f"{float((pa == gold).float().mean()):>14.4f}{la:>14.1%}{ca:>14.1%}")
    for s in (1, 2, 3, 6):
        p, f, _ = dlm_generate(model, ctx_tok, s)
        l, c, _ = metrics(p, sid, fv)
        print(f"{f'DLM 去噪 S={s}':<30}{f:>10}{s:>10}"
              f"{float((p == gold).float().mean()):>14.4f}{l:>14.1%}{c:>14.1%}")
    print("-" * 96)
    print("注: 分支等概率, 故 Token 准确率天花板本身只有约 0.55(历史确定部分 + 50% 分支命中),")
    print("    不宜横向解读; 真正的差别在\"合法 SID 率/跨物品一致性\": S=1 一次性并行")
    print("    生成时各位置独立决策, 会拼出表中不存在的 SID, 随 S 增大逐步恢复。")

    print()
    print("表 2  采样解码: 并行度与位置协同能力的权衡(这是 DLM 的真实约束)")
    print("-" * 96)
    print(f"{'生成方式':<30}{'前向次数':>10}{'串行步数':>10}"
          f"{'合法 SID 率':>14}{'跨物品一致性':>14}{'分支合法率':>12}")
    torch.manual_seed(7)
    pa, fa = ar_generate(model, ctx_tok, sample=True)
    la, ca, ba = metrics(pa, sid, fv)
    print(f"{'自回归逐 Token(采样)':<30}{fa:>10}{GEN_LEN:>10}"
          f"{la:>14.1%}{ca:>14.1%}{ba:>12.1%}")
    for s in (1, 2, 3, 6):
        torch.manual_seed(7)
        p, f, _ = dlm_generate(model, ctx_tok, s, sample=True)
        l, c, b = metrics(p, sid, fv)
        print(f"{f'DLM 去噪 S={s}(采样)':<30}{f:>10}{s:>10}"
              f"{l:>14.1%}{c:>14.1%}{b:>12.1%}")
    print("-" * 96)
    print("S=1: 6 个位置完全独立采样, 组合出的三元组常常不对应任何真实物品, 跨物品")
    print("     一致性接近随机 -> 一次前向最快, 但生成质量最差;")
    print("S 增大: 已定稿 Token 成为后续位置的条件, 合法率与一致性迅速回升到自回归水平,")
    print("     代价是串行步数增加。这就是正文所说的生成质量与推理效率的平衡问题。")

    print()
    print("表 3  一条序列的逐步去噪轨迹(S=3, '--' 表示该位置仍为 [MASK])")
    print("-" * 96)
    _, _, traj = dlm_generate(model, ctx_tok[:1], 3, trace=True)
    print(f"{'状态':>8}  " + "  ".join(f"{'pos' + str(j):>5}" for j in range(GEN_LEN)))
    for s, cur, fz in traj:
        cells = [f"{cur[j]:>5}" if fz[j] else f"{'--':>5}" for j in range(GEN_LEN)]
        print(f"{f'x_{3 - s}':>8}  " + "  ".join(cells))
    print(f"{'参考 x_0':>8}  " + "  ".join(f"{int(v):>5}" for v in gold[0])
          + "   <- 该样本的真值分支; 另一分支同样合法, 故不要求逐位相同")
    print("-" * 96)
    print("可见每一步都可同时定稿多个位置, 没有"
          "自回归那种\"必须先得到 s^1 才能算 s^2\"的串行约束。")


if __name__ == "__main__":
    main()
