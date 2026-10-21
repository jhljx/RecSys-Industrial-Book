r"""对应正文第 9 章 §9.7「生成式推荐落地问题」中 MTP 与推测解码的推理加速。

正文公式:
  L_NTP = -log P(x_t | x_<t)
  L_MTP = -Σ_{i=0}^{K-1} log P(x_{t+i} | x_<t)
  自回归推理:  x_t -> x_{t+1} -> x_{t+2}            (生成长度 T 需 T 次前向)
  MTP 推理:    x_t -> (x_{t+1}, ..., x_{t+K})        (约 T/K 次前向)
  物品由多个语义 Token 组成 item_i = (s_i^1, s_i^2, s_i^3), 传统需 s^1->s^2->s^3,
  引入多 Token 预测头后 Decoder(h_t) 一次前向即可给出完整语义 ID。

本实现:
  1. 共享 Backbone(单层 GRU)上挂 K 个预测头, 同时用 NTP 与 MTP 目标训练, 对比
     L_NTP / L_MTP 以及各个偏移位置的 Token 准确率;
  2. 实现 Medusa 风格的推测解码: MTP 头做草稿(draft), 主模型一次前向并行验证,
     按"最长前缀匹配"接受, 打印接受长度分布、平均接受长度 tau 与实测加速比;
  3. 给出理论加速比与实测前向次数的对照表。
  4. [MOCK] 模型为教学规模, 前向次数而非墙钟时间才是加速比的主指标; 同时给出墙钟。
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

L_LAYER = 3          # 每个物品的语义 Token 数
K_CODE = 12          # 每层码字数
VOCAB = L_LAYER * K_CODE + 1
BOS = VOCAB - 1
D_MODEL = 48
SEQ_LEN = 8          # 历史物品数
K_MTP = 3            # 多 Token 预测头数(= L_LAYER, 一次给出完整 SID)
N_ITEM = 100


def flat(layer, code):
    return layer * K_CODE + code


class Backbone(nn.Module):
    """共享 Backbone + K 个预测头(第 0 头即标准 NTP 头)。"""

    def __init__(self, n_head=K_MTP):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.gru = nn.GRU(D_MODEL, D_MODEL, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(D_MODEL, VOCAB) for _ in range(n_head)])
        self.n_fwd = 0

    def hidden(self, x, h0=None):
        self.n_fwd += 1
        out, h = self.gru(self.emb(x), h0)
        return out, h

    def head_logits(self, hs):
        """返回 (B, T, n_head, V): 第 i 个头预测 x_{t+1+i}。"""
        return torch.stack([hd(hs) for hd in self.heads], dim=-2)


def build_data(gen, n_seq=256):
    sid = torch.stack([torch.randint(0, K_CODE, (N_ITEM,), generator=gen)
                       for _ in range(L_LAYER)], 1)
    rows = []
    for _ in range(n_seq):
        fav = int(torch.randint(0, K_CODE, (1,), generator=gen))
        pool = [i for i in range(N_ITEM) if int(sid[i, 0]) == fav] or list(range(N_ITEM))
        pick = [pool[int(torch.randint(0, len(pool), (1,), generator=gen))]
                for _ in range(min(4, len(pool)))]
        row = [BOS]
        for _ in range(SEQ_LEN):
            it = pick[int(torch.randint(0, len(pick), (1,), generator=gen))]
            row += [flat(l, int(sid[it, l])) for l in range(L_LAYER)]
        rows.append(row)
    return torch.tensor(rows), sid


def train(mode, data, epochs=120):
    """mode='ntp' 只用第 0 头; mode='mtp' 用全部 K 个头(L_MTP)。"""
    torch.manual_seed(42)
    model = Backbone()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    n_head = 1 if mode == "ntp" else K_MTP
    T = data.shape[1]
    for _ in range(epochs):
        hs, _ = model.hidden(data[:, :-1])
        logits = model.head_logits(hs)
        loss = 0.0
        for i in range(n_head):
            lo = logits[:, :T - 1 - i, i]
            tg = data[:, 1 + i:T]
            loss = loss + F.cross_entropy(lo.reshape(-1, VOCAB), tg.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def head_accuracy(model, data):
    T = data.shape[1]
    hs, _ = model.hidden(data[:, :-1])
    logits = model.head_logits(hs)
    acc = []
    for i in range(K_MTP):
        pred = logits[:, :T - 1 - i, i].argmax(-1)
        acc.append(float((pred == data[:, 1 + i:T]).float().mean()))
    return acc


@torch.no_grad()
def prefill(model, prefix):
    """预填充: 返回 (state, h0); 之后统计的前向次数不含这一次预填充。"""
    out, _ = model.hidden(prefix)
    s = out[:, -1:]                                    # (B, 1, D) 最后一步隐状态
    model.n_fwd = 0
    return s, s.transpose(0, 1).contiguous()


@torch.no_grad()
def autoregressive(model, prefix, n_token):
    """标准 NTP: 每次前向只产 1 个 Token, 共 n_token 次前向。"""
    s, h = prefill(model, prefix)
    gen = []
    for _ in range(n_token):
        nxt = model.head_logits(s)[:, -1, 0].argmax(-1, keepdim=True)
        gen.append(nxt)
        out, h = model.hidden(nxt, h)
        s = out[:, -1:]
    return torch.cat(gen, 1), model.n_fwd


@torch.no_grad()
def mtp_parallel(model, prefix, n_token, k=K_MTP):
    """MTP: 一次前向直出 k 个 Token, 共 ceil(n_token/k) 次前向。"""
    s, h = prefill(model, prefix)
    gen, n = [], 0
    while n < n_token:
        blk = model.head_logits(s)[:, -1, :k].argmax(-1)   # (B, k) 一次给 k 个
        gen.append(blk)
        n += k
        out, h = model.hidden(blk, h)
        s = out[:, -1:]
    return torch.cat(gen, 1)[:, :n_token], model.n_fwd


@torch.no_grad()
def speculative(model, prefix, n_token, k=K_MTP):
    """Medusa 风格推测解码: MTP 头出草稿, 主模型一次前向并行验证最长前缀。

    * 草稿由 K 个头在当前状态上一次给出, 不额外消耗 Backbone 前向;
    * 验证: 把草稿整体喂入 Backbone(1 次前向), 第 0 头在位置 j-1 的 argmax 即为
      "主模型认可的 x_{t+j}", 与 draft[j] 逐位比较, 首次不一致处截断;
    * 草稿位置 0 来自第 0 头, 与逐 Token 自回归完全相同, 故恒被接受;
      被拒绝的位置不强行纠正, 下一轮它会作为新草稿的位置 0 重新生成,
      因此最终输出与纯自回归贪心解码逐 Token 一致。
    """
    s, h = prefill(model, prefix)
    gen, accepted, n = [], [], 0
    while n < n_token:
        draft = model.head_logits(s)[:, -1, :k].argmax(-1)     # (B, k) 草稿
        vout, _ = model.hidden(draft, h)                        # 1 次验证前向
        main = model.head_logits(vout)[:, :, 0].argmax(-1)      # (B, k)
        n_acc = 1
        for j in range(1, k):
            if bool((draft[:, j] == main[:, j - 1]).all()):
                n_acc += 1
            else:
                break
        gen.append(draft[:, :n_acc])
        accepted.append(n_acc)
        n += n_acc
        s = vout[:, n_acc - 1:n_acc]                            # 接受到的最后一步状态
        h = s.transpose(0, 1).contiguous()
    return torch.cat(gen, 1)[:, :n_token], model.n_fwd, accepted


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)
    data, _ = build_data(gen)

    m_ntp = train("ntp", data)
    m_mtp = train("mtp", data)

    print("=" * 92)
    print(f"配置: 词表 V={VOCAB}(L={L_LAYER} 层 × K={K_CODE} 码字 + BOS), "
          f"d_model={D_MODEL}, 序列长度 {data.shape[1]}, K_MTP={K_MTP}")
    print()
    print("表 1  NTP 目标 vs. MTP 目标(训练集 Token 准确率)")
    print("-" * 92)
    a_n, a_m = head_accuracy(m_ntp, data), head_accuracy(m_mtp, data)
    print(f"{'预测偏移':>10}{'目标':>22}{'NTP 训练模型':>16}{'MTP 训练模型':>16}")
    for i in range(K_MTP):
        print(f"{f'x_(t+{i + 1})':>10}{f'第 {i} 个头':>22}{a_n[i]:>16.4f}{a_m[i]:>16.4f}")
    print("-" * 92)
    print("NTP 只训练第 0 头, 其余头是随机初始化故准确率接近 1/V ="
          f" {1 / VOCAB:.4f}; MTP 让一次前向即可给出完整语义 ID。")

    # -------- 推理前向次数与墙钟 --------
    prefix = data[:16, :1 + SEQ_LEN * L_LAYER - L_LAYER]
    N_GEN = 24                     # 生成 8 个物品 = 24 个语义 Token
    print()
    print(f"表 2  生成 {N_GEN} 个语义 Token(= {N_GEN // L_LAYER} 个物品)的推理开销")
    print("-" * 92)
    print(f"{'解码方式':<28}{'前向次数':>10}{'理论次数':>10}{'加速比(前向)':>14}"
          f"{'墙钟(ms)':>12}{'与 NTP 输出一致':>18}")

    t0 = time.perf_counter()
    g_ntp, f_ntp = autoregressive(m_mtp, prefix, N_GEN)
    t_ntp = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    g_mtp, f_mtp = mtp_parallel(m_mtp, prefix, N_GEN)
    t_mtp = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    g_spec, f_spec, acc_list = speculative(m_mtp, prefix, N_GEN)
    t_spec = (time.perf_counter() - t0) * 1e3

    rows = [("自回归 NTP(逐 Token)", f_ntp, N_GEN, t_ntp, "基准"),
            ("MTP 并行(一次出 3 Token)", f_mtp,
             -(-N_GEN // K_MTP), t_mtp,
             "是" if torch.equal(g_ntp, g_mtp) else "否"),
            ("推测解码(草稿+验证)", f_spec, None, t_spec,
             "是" if torch.equal(g_ntp, g_spec) else "否")]
    for name, nf, th, ms, same in rows:
        ths = "-" if th is None else str(th)
        print(f"{name:<28}{nf:>10}{ths:>10}{f_ntp / nf:>14.2f}x{ms:>12.2f}{same:>18}")
    print("-" * 92)
    tau = sum(acc_list) / len(acc_list)
    print(f"推测解码接受长度: {acc_list}  平均接受长度 tau = {tau:.2f} "
          f"(上限 K_MTP = {K_MTP})")
    print(f"理论加速比 ≈ tau / (1 个草稿前向 + 1 个验证前向 每轮产 tau+ Token) ="
          f" {f_ntp / f_spec:.2f}x(实测)")
    print("注: MTP 直出速度最快但输出不保证与逐 Token 自回归一致; 推测解码通过验证")
    print("    步骤保证输出一致性, 代价是额外的验证前向。")

    # -------- 不同 K 的理论/实测对照 --------
    print()
    print(f"表 3  不同多 Token 头数 K 的前向次数(生成 {N_GEN} 个 Token)")
    print("-" * 92)
    print(f"{'K':>4}{'MTP 前向次数':>14}{'理论 ceil(T/K)':>16}{'加速比':>10}"
          f"{'单次前向产出 Token':>20}")
    for k in (1, 2, 3):
        g, nf = mtp_parallel(m_mtp, prefix, N_GEN, k=k)
        print(f"{k:>4}{nf:>14}{-(-N_GEN // k):>16}{N_GEN / nf:>10.2f}x{k:>20}")
    print("-" * 92)
    print("正文: 将原本需要 T 次前向的生成过程缩减为约 T/K 次, 大幅降低推理延迟。")


if __name__ == "__main__":
    main()
