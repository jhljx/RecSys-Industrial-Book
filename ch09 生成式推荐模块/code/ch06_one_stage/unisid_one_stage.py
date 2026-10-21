r"""对应正文第 9 章 §9.6.3「UniSID 算法」: 一阶段端到端语义 ID 生成。

正文要点与公式(本实现逐条复刻):
  * 输入序列 X_i = [x_instruction, x_image, x_text, x_attribute, x_sid, x_emb],
    经共享编码器 Z_i = MLLM(X_i), 取 SID/Emb 位置隐状态过两个独立投影头:
        z_i^SID = f_SID(Z_i^SID),  z_i^Emb = f_Emb(Z_i^Emb)
  * z_i^SID 切成 L 段, 每段直接取 argmax 维度作为 Token(不查码本、无最近邻):
        s_i^l = argmax(z_i^l),  s_i = (s_i^1, ..., s_i^L)
    因此语义 ID 只依赖维度间的大小顺序, 对数值缩放/平移具有鲁棒性。
  * 多粒度对比学习:
        L_sid = (1/L) Σ_l ( -(1/|P_l|) Σ_{p∈P_l} log
                 exp(sim(z_i^l, z_p^l)/τ) / Σ_{a∈A_l} exp(sim(z_i^l, z_a^l)/τ) )
    浅层正样本按粗粒度(大类)定义, 深层正样本按细粒度(子类)定义。
  * Embedding 分支标准对比学习:
        L_emb = -log exp(sim(z_i^Emb, z_j^Emb)/τ) / Σ_{k∈N_i} exp(sim(z_i^Emb, z_k^Emb)/τ)
  * Summary-based Reconstruction: 冻结 LLM 产出摘要 token, 由
        Z_i = [Z_i^SID; Z_i^Emb] -> h_i^rec = f_rec(Z_i) -> NTP 重建摘要
    对应 L_rec = Σ_t log p(s_{i,t}^sum | h_i^rec, s_{i,<t}^sum)
  * 总损失 L_total = L_sid + L_emb + λ L_rec

本实现:
  1. [MOCK] 用一个小 Transformer 编码器代替 MLLM, 多模态输入用随机投影的
     "图像 / 文本 / 属性" 三组合成向量 + 可学习的 SID / Emb 占位 Token;
  2. [MOCK] 冻结 LLM 的"摘要"用确定性哈希规则从属性生成一小段 token 序列;
  3. 打印: argmax 取 SID 的鲁棒性验证表(缩放/平移/噪声)、层次语义结构统计
     (浅层与大类的一致性 vs 深层与子类的一致性)、以及三项损失的消融表。
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

L_LAYER = 3          # SID 层数
K_CODE = 8           # 每层维度数 = 候选 Token 数
D_MODEL = 32
N_COARSE = 4         # 粗粒度大类数
N_FINE = 3           # 每个大类下的细粒度子类数
SUM_VOCAB = 12       # [MOCK] 摘要 token 词表
SUM_LEN = 4
N_ITEM = 180


class UniSID(nn.Module):
    """[MOCK] 统一编码器: 同一个 Transformer 同时产出 SID 与 Embedding。"""

    def __init__(self, d_in=20):
        super().__init__()
        self.proj = nn.ModuleDict({
            "image": nn.Linear(d_in, D_MODEL),
            "text": nn.Linear(d_in, D_MODEL),
            "attr": nn.Linear(d_in, D_MODEL),
        })
        # x_instruction / x_sid / x_emb 三个特殊占位 Token
        self.special = nn.Parameter(torch.randn(3, D_MODEL) * 0.1)
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 4 * D_MODEL, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.f_sid = nn.Linear(D_MODEL, L_LAYER * K_CODE)      # f_SID
        self.f_emb = nn.Linear(D_MODEL, D_MODEL)               # f_Emb
        self.f_rec = nn.Linear(2 * D_MODEL, D_MODEL)           # f_rec
        self.sum_emb = nn.Embedding(SUM_VOCAB + 1, D_MODEL)
        self.sum_out = nn.Linear(D_MODEL, SUM_VOCAB)

    def forward(self, img, txt, att):
        b = img.shape[0]
        toks = [self.special[0].expand(b, -1),                  # x_instruction
                self.proj["image"](img),                        # x_image
                self.proj["text"](txt),                          # x_text
                self.proj["attr"](att),                          # x_attribute
                self.special[1].expand(b, -1),                   # x_sid
                self.special[2].expand(b, -1)]                   # x_emb
        z = self.enc(torch.stack(toks, 1))                       # Z_i = MLLM(X_i)
        z_sid = self.f_sid(z[:, 4])                              # z_i^SID
        z_emb = self.f_emb(z[:, 5])                              # z_i^Emb
        h_rec = self.f_rec(torch.cat([z[:, 4], z[:, 5]], -1))    # h_i^rec
        return z_sid.view(b, L_LAYER, K_CODE), z_emb, h_rec

    def sid_of(self, z_sid):
        """s_i^l = argmax(z_i^l): 只看维度间大小顺序, 不查码本。"""
        return z_sid.argmax(-1)

    def recon_loss(self, h_rec, summ):
        """[MOCK] 冻结 LLM 的 NTP 重建: p(s_t^sum | h_rec, s_<t^sum)。"""
        b, t = summ.shape
        inp = torch.cat([torch.full((b, 1), SUM_VOCAB, dtype=torch.long),
                         summ[:, :-1]], 1)
        h = self.sum_emb(inp) + h_rec[:, None]
        logits = self.sum_out(torch.tanh(h))
        return F.cross_entropy(logits.reshape(b * t, -1), summ.reshape(-1))


def multi_granularity_contrastive(z_sid, coarse, fine, tau=0.2):
    """L_sid: 浅层正样本=同大类, 深层正样本=同子类(从粗到细的层次结构)。"""
    total = 0.0
    for l in range(L_LAYER):
        zl = F.normalize(z_sid[:, l], dim=-1)
        sim = zl @ zl.t() / tau
        if l == 0:
            pos = coarse[:, None] == coarse[None, :]              # 粗粒度
        elif l == L_LAYER - 1:
            pos = (coarse[:, None] == coarse[None, :]) & (fine[:, None] == fine[None, :])
        else:                                                      # 中间层: 过渡
            pos = (coarse[:, None] == coarse[None, :]) & \
                  ((fine[:, None] == fine[None, :]) | (torch.rand_like(sim) < 0.3))
        eye = torch.eye(len(zl), dtype=torch.bool)
        pos = pos & ~eye
        logden = torch.logsumexp(sim.masked_fill(eye, -1e9), dim=-1)
        num = (sim - logden[:, None]) * pos
        cnt = pos.sum(-1).clamp_min(1)
        total = total + (-(num.sum(-1) / cnt)).mean()
    return total / L_LAYER


def emb_contrastive(z_emb, view2, tau=0.2):
    """L_emb: 标准 InfoNCE, 正样本为同一物品的另一视图(属性 dropout 增强)。"""
    a = F.normalize(z_emb, dim=-1)
    b = F.normalize(view2, dim=-1)
    s = a @ b.t() / tau
    return F.cross_entropy(s, torch.arange(len(a)))


def make_summary(coarse, fine, gen):
    """[MOCK] 冻结 LLM 的摘要: 由大类/子类确定性映射到一段 token, 带少量噪声。"""
    table = torch.randint(0, SUM_VOCAB, (N_COARSE, N_FINE, SUM_LEN), generator=gen)
    return table[coarse, fine]


def nmi_like(tokens, labels, n_lab):
    """用"条件纯度"衡量 Token 与标签的一致性: Σ_t max_c p(c|t) * p(t)。"""
    pur = 0.0
    for t in tokens.unique():
        m = tokens == t
        cnt = torch.bincount(labels[m], minlength=n_lab).float()
        pur += float(cnt.max()) / len(tokens)
    return pur


def train(lam_rec=1.0, use_sid=True, use_emb=True, epochs=40, d_in=20, seed=42):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    # ------- 合成多模态物料 -------
    c_cent = torch.randn(N_COARSE, d_in, generator=gen) * 2.0
    f_cent = torch.randn(N_COARSE, N_FINE, d_in, generator=gen) * 0.8
    coarse = torch.randint(0, N_COARSE, (N_ITEM,), generator=gen)
    fine = torch.randint(0, N_FINE, (N_ITEM,), generator=gen)
    base = c_cent[coarse] + f_cent[coarse, fine]
    img = base + torch.randn(N_ITEM, d_in, generator=gen) * 0.5
    txt = base + torch.randn(N_ITEM, d_in, generator=gen) * 0.5
    att = base + torch.randn(N_ITEM, d_in, generator=gen) * 0.5
    att2 = base + torch.randn(N_ITEM, d_in, generator=gen) * 0.5   # 增强视图
    summ = make_summary(coarse, fine, gen)

    model = UniSID(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = []
    for ep in range(epochs):
        z_sid, z_emb, h_rec = model(img, txt, att)
        _, z_emb2, _ = model(img, txt, att2)
        ls = multi_granularity_contrastive(z_sid, coarse, fine) if use_sid \
            else torch.zeros(())
        le = emb_contrastive(z_emb, z_emb2) if use_emb else torch.zeros(())
        lr = model.recon_loss(h_rec, summ)
        loss = ls + le + lam_rec * lr                      # L_total
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            hist.append((ep + 1, float(ls), float(le), float(lr), float(loss)))
    with torch.no_grad():
        z_sid, z_emb, _ = model(img, txt, att)
        sid = model.sid_of(z_sid)
    return model, (img, txt, att), sid, coarse, fine, hist


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    model, data, sid, coarse, fine, hist = train()
    img, txt, att = data

    print("=" * 90)
    print(f"配置: 物品 {N_ITEM} 个, 大类 {N_COARSE} / 子类 {N_FINE}, "
          f"SID L={L_LAYER} K={K_CODE}, d_model={D_MODEL}")
    print()
    print("表 1  L_total = L_sid + L_emb + λ L_rec 的训练轨迹(λ=1.0)")
    print("-" * 90)
    print(f"{'epoch':>7}{'L_sid':>12}{'L_emb':>12}{'L_rec':>12}{'L_total':>12}")
    for ep, ls, le, lr, lo in hist:
        print(f"{ep:>7}{ls:>12.4f}{le:>12.4f}{lr:>12.4f}{lo:>12.4f}")

    # ---------- argmax 取 SID 的鲁棒性 ----------
    print()
    print("表 2  s^l = argmax(z^l) 的鲁棒性(正文: 只关注维度大小顺序, 不关心绝对数值)")
    print("-" * 90)
    print(f"{'扰动方式':<34}{'SID 完全一致比例':>20}{'逐层 Token 一致率':>20}")
    with torch.no_grad():
        z0, _, _ = model(img, txt, att)
        s0 = model.sid_of(z0)
        cases = [("原始输出(基准)", lambda z: z),
                 ("整体乘 10(缩放)", lambda z: z * 10.0),
                 ("整体除 10(缩放)", lambda z: z / 10.0),
                 ("每层加常数 +5(平移)", lambda z: z + 5.0),
                 ("加 N(0, 0.05²) 噪声", lambda z: z + torch.randn_like(z) * 0.05),
                 ("加 N(0, 0.5²) 噪声", lambda z: z + torch.randn_like(z) * 0.5)]
        for name, fn in cases:
            s = model.sid_of(fn(z0.clone()))
            same_all = float((s == s0).all(-1).float().mean())
            same_tok = float((s == s0).float().mean())
            print(f"{name:<34}{same_all:>20.4f}{same_tok:>20.4f}")
    print("-" * 90)
    print("可见缩放/平移完全不改变 argmax 结果; 只有足够大的噪声才会翻转 Token。")

    # ---------- 层次语义结构 ----------
    print()
    print("表 3  多粒度对比学习形成的层次结构(条件纯度: 越大说明该层 Token 越贴合该粒度)")
    print("-" * 90)
    print(f"{'SID 层':>8}{'对大类(粗粒度)纯度':>22}{'对(大类,子类)纯度':>22}"
          f"{'用到的 Token 数':>16}")
    joint = coarse * N_FINE + fine
    for l in range(L_LAYER):
        pc = nmi_like(sid[:, l], coarse, N_COARSE)
        pf = nmi_like(sid[:, l], joint, N_COARSE * N_FINE)
        print(f"{l + 1:>8}{pc:>22.4f}{pf:>22.4f}{len(sid[:, l].unique()):>16}")
    uniq = len({tuple(s.tolist()) for s in sid})
    print("-" * 90)
    print(f"不同 SID 数 = {uniq} / {N_ITEM} 个物品 (空间上限 K^L = {K_CODE ** L_LAYER})")

    # ---------- 消融 ----------
    print()
    print("表 4  损失项消融(各配置独立训练 40 epoch)")
    print("-" * 90)
    print(f"{'配置':<34}{'第1层对大类纯度':>18}{'第3层对子类纯度':>18}{'不同 SID 数':>14}")
    for name, kw in (("L_sid + L_emb + L_rec(完整)", {}),
                     ("去掉 L_sid", {"use_sid": False}),
                     ("去掉 L_emb", {"use_emb": False}),
                     ("去掉 L_rec(λ=0)", {"lam_rec": 0.0})):
        _, _, s, c, f, _ = train(**kw)
        j = c * N_FINE + f
        print(f"{name:<34}{nmi_like(s[:, 0], c, N_COARSE):>18.4f}"
              f"{nmi_like(s[:, L_LAYER - 1], j, N_COARSE * N_FINE):>18.4f}"
              f"{len({tuple(x.tolist()) for x in s}):>14}")
    print("-" * 90)
    print("正文强调 L_sid 负责层次解耦、L_emb 约束 Embedding 分支、L_rec 迫使 SID")
    print("编码更抽象的高层语义; 此处为教学规模复现, 数值仅用于说明三者的作用方向。")


if __name__ == "__main__":
    main()
