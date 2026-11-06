"""
MoE 精排模型（第12章 §12.4.3.7、§12.4.3.8）

用稀疏专家混合替换 Transformer 中的前馈网络（FFN）：

    MoE(x) = Σ_i g_i(x) · E_i(x)

  1. 稠密 MoE（所有专家都参与）：
       g_i(x) = softmax(W_r x)_i
     计算量随专家数线性增长，论文指出这不是稀疏化的初衷。

  2. Top-K 稀疏路由（§12.4.3.7）：
       r   = W_r x
       p   = softmax(r)
       𝒯_K = TopK(p, K)                    # 只激活 K 个专家
       MoE(x) = Σ_{i∈𝒯_K} g̃_i · E_i(x),  g̃_i = p_i / Σ_{j∈𝒯_K} p_j
     计算量只与 K 有关，与专家总数解耦。

  3. 共享专家（§12.4.3.8，DeepSeekMoE）：
       MoE(x) = Σ_{i=1}^{K_s} E_i^shared(x) + Σ_{j∈ℛ(x)} g_j · E_j^routed(x)
     共享专家对所有 token 都激活，负责捕获通用知识；
     路由专家负责捕获差异化知识，减少专家间的参数冗余。

  4. 负载均衡损失（§12.4.3.8）：
     专家级：L_ExpBal = α₁ · N · Σ_i f_i · P_i
             f_i = 分配到专家 i 的 token 占比，P_i = 专家 i 的平均路由概率
     设备级：L_DevBal = α₂ · D · Σ_d f'_d · P'_d
             （按设备对专家分组，避免单个设备成为瓶颈）

  5. 工程约束（§12.4.3.7 正文）：
     MoE 引入的通信开销与负载不均，往往是「稀疏化省下的算力」
     被抵消的主要原因，因此负载均衡损失不是可有可无的正则项。

本文件的对照实验直接检验上述两点：
  - 去掉负载均衡损失，观察专家利用率是否塌缩到少数专家；
  - 加上专家级 / 设备级均衡损失，观察利用率与设备分配是否变均匀。

本书依据：《第十二章-精排模型.pdf》§12.4.3.7–§12.4.3.8。
依赖：PyTorch。运行：python3 moe_ranking.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _auc(scores, labels) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg) / (len(pos) * len(neg))


# ── 路由层 ───────────────────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    """
    Top-K 稀疏路由：r = W_r x → softmax → 取 Top-K → 在 K 个专家内重新归一化。

    返回：
      topk_idx  : (B, K)  被选中的专家下标
      gates     : (B, K)  归一化后的门控权重
      probs     : (B, N)  完整路由分布（用于计算负载均衡的 P_i）
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        probs = F.softmax(self.gate(x), dim=-1)            # (B, N)
        topk_p, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        gates = topk_p / topk_p.sum(dim=-1, keepdim=True)  # 重新归一化
        return topk_idx, gates, probs


# ── MoE 前馈层 ───────────────────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """
    Top-K 路由专家 + 共享专家。

    参数：
      d_model     : 输入/输出维度
      d_ff        : 每个专家的隐藏维度
      n_routed    : 路由专家数（细粒度切分时取较大值，单个专家较窄）
      top_k       : 每个 token 激活的路由专家数
      n_shared    : 共享专家数（DeepSeekMoE 做法，对所有 token 激活）
      n_device    : 设备数，用于设备级负载均衡损失
    """

    def __init__(self, d_model: int, d_ff: int, n_routed: int = 8,
                 top_k: int = 2, n_shared: int = 1, n_device: int = 2):
        super().__init__()
        self.n_routed = n_routed
        self.n_shared = n_shared
        self.n_device = n_device

        self.router = TopKRouter(d_model, n_routed, top_k)
        self.routed_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Linear(d_ff, d_model))
            for _ in range(n_routed)])
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Linear(d_ff, d_model))
            for _ in range(n_shared)])
        # 残差投影：与 Transformer FFN 的位置对齐
        self.out_proj = nn.Linear(d_model, d_model)

    def _device_of(self, expert_idx: torch.Tensor) -> torch.Tensor:
        """把专家下标平均分到 n_device 个设备上。"""
        return expert_idx * self.n_device // self.n_routed

    def forward(self, x: torch.Tensor):
        """
        x : (B, d_model)
        返回: y (B, d_model), stats(dict) 供计算负载均衡损失与诊断
        """
        B, _ = x.shape
        topk_idx, gates, probs = self.router(x)            # (B,K), (B,K), (B,N)

        # 路由专家：逐专家处理被分配给它的 token（实现上是掩码 + 累加，
        # 与真实系统中按 token 重排（permutation）后分组计算等价）
        y = torch.zeros_like(x)
        for k in range(topk_idx.size(1)):
            idx_k = topk_idx[:, k]                         # (B,)
            gate_k = gates[:, k].unsqueeze(-1)             # (B,1)
            for e in range(self.n_routed):
                mask = (idx_k == e).unsqueeze(-1)          # (B,1)
                if not mask.any():
                    continue
                out_e = self.routed_experts[e](x)          # (B, d_model)
                y = y + torch.where(mask, gate_k * out_e,
                                    torch.zeros_like(out_e))

        # 共享专家：所有 token 都激活
        for e in self.shared_experts:
            y = y + e(x)

        y = self.out_proj(y)

        # ── 负载均衡统计 ─────────────────────────────────────────────────────
        # f_i：分配到专家 i 的 token 占比
        one_hot = F.one_hot(topk_idx, self.n_routed).float().sum(dim=1)  # (B,N)
        f = one_hot.mean(dim=0)                            # (N,)
        # P_i：专家 i 的平均路由概率
        P = probs.mean(dim=0)                              # (N,)
        # 设备级：把专家按设备分组聚合
        dev = self._device_of(torch.arange(self.n_routed))
        f_dev = torch.zeros(self.n_device, device=x.device)
        P_dev = torch.zeros(self.n_device, device=x.device)
        f_dev.index_add_(0, dev, f)
        P_dev.index_add_(0, dev, P)

        stats = dict(f=f, P=P, f_dev=f_dev, P_dev=P_dev,
                     top1_share=one_hot[:, topk_idx[:, 0]].mean().item())
        return y, stats

    # ── 负载均衡损失 ─────────────────────────────────────────────────────────
    def balance_loss(self, stats, alpha_exp: float = 0.01,
                     alpha_dev: float = 0.01) -> torch.Tensor:
        N = self.n_routed
        L_exp = N * (stats["f"] * stats["P"]).sum()
        D = self.n_device
        L_dev = D * (stats["f_dev"] * stats["P_dev"]).sum()
        return alpha_exp * L_exp + alpha_dev * L_dev


# ── 单个 MoE 层上的精排模型 ──────────────────────────────────────────────────

class MoERanker(nn.Module):
    """
    演示用的极简结构：嵌入 → 投影到 d_model → MoE 层 → 输出 logit。

    只用一层 MoE，是为了让路由行为成为唯一的观察对象；
    真实系统中的 MoE 位于 Transformer 的每个 FFN 位置。
    """

    def __init__(self, vocab: int, n_fields: int, d_model: int = 16,
                 d_ff: int = 16, n_routed: int = 8, top_k: int = 2,
                 n_shared: int = 1, n_device: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.n_fields = n_fields
        self.proj = nn.Linear(d_model * n_fields, d_model)
        self.moe = MoEFFN(d_model, d_ff, n_routed, top_k, n_shared, n_device)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, feat_ids: torch.Tensor):
        e = torch.cat([self.emb(feat_ids[:, i])
                       for i in range(self.n_fields)], dim=1)
        h = F.gelu(self.proj(e))                           # (B, d_model)
        y, stats = self.moe(h)
        h = self.norm(h + y)                               # 残差 + LayerNorm
        return self.head(h).squeeze(-1), stats

    def fit(self, feat_ids, y, use_balance: bool, epochs: int = 40,
            lr: float = 5e-3):
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(epochs):
            opt.zero_grad()
            logit, stats = self.forward(feat_ids)
            loss = F.binary_cross_entropy_with_logits(logit, y.float())
            if use_balance:
                loss = loss + self.moe.balance_loss(stats)
            loss.backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}")
        return loss.item()

    def predict_proba(self, feat_ids):
        with torch.no_grad():
            logit, _ = self.forward(feat_ids)
            return torch.sigmoid(logit)


# ── 合成数据 ─────────────────────────────────────────────────────────────────

def make_data(n: int = 3000, vocab: int = 16, n_fields: int = 6,
              seed: int = 42):
    """
    特征交互型标签：让不同样本落在特征空间的不同区域，
    从而为「不同专家负责不同区域」提供存在的理由。

    注意：真实系统中专家分工是被数据自然诱导出来的；
    本文件不人为构造「专家分工」，只观察路由是否出现塌缩。
    """
    torch.manual_seed(seed)
    random.seed(seed)

    feat = torch.randint(0, vocab, (n, n_fields)).float()
    logit = (0.35 * feat[:, 0] + 0.30 * feat[:, 1]
             - 0.25 * feat[:, 2]
             + 0.20 * (feat[:, 3] * feat[:, 4]).sin()
             + 0.15 * (feat[:, 5] > 8).float()
             - 1.2)
    y = (torch.sigmoid(logit / 3.0 + 0.4 * torch.randn(n)) > 0.5).long()
    return dict(feat=feat.long(), y=y)


# ── 主演示 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    VOCAB, N_FIELDS = 16, 6
    data = make_data(3000, vocab=VOCAB, n_fields=N_FIELDS)
    split = 2400
    tr = {k: v[:split] for k, v in data.items()}
    te = {k: v[split:] for k, v in data.items()}

    def build():
        torch.manual_seed(42)
        return MoERanker(VOCAB, N_FIELDS, d_model=16, d_ff=16,
                         n_routed=8, top_k=2, n_shared=1, n_device=2)

    def evaluate(model):
        p = model.predict_proba(te["feat"])
        return _auc(p.tolist(), te["y"].tolist())

    def show_util(model, tag):
        with torch.no_grad():
            _, stats = model.forward(te["feat"])
        f = stats["f"].tolist()
        fd = stats["f_dev"].tolist()
        print(f"  {tag}")
        print("    专家利用率 f_i: " +
              " ".join(f"E{i}={v:.3f}" for i, v in enumerate(f)))
        print(f"    最大/最小利用率比 = {max(f) / max(min(f), 1e-6):.2f}"
              f"   分布标准差 = {torch.tensor(f).std().item():.4f}")
        print("    设备 token 占比 f'_d: " +
              " ".join(f"Dev{d}={v:.3f}" for d, v in enumerate(fd)))

    # ── 对照组：无负载均衡损失 ───────────────────────────────────────────────
    print("=== 对照组：只优化主任务（无负载均衡损失）===")
    m_free = build()
    m_free.fit(tr["feat"], tr["y"], use_balance=False, epochs=40)
    auc_free = evaluate(m_free)
    show_util(m_free, "路由利用情况：")

    # ── 实验组：加入专家级 + 设备级负载均衡损失 ───────────────────────────────
    print("\n=== MoE：主任务 + 专家级 / 设备级负载均衡损失 ===")
    m_bal = build()
    m_bal.fit(tr["feat"], tr["y"], use_balance=True, epochs=40)
    auc_bal = evaluate(m_bal)
    show_util(m_bal, "路由利用情况：")

    # ── 对比结论 ─────────────────────────────────────────────────────────────
    print(f"\n对比（正文 §12.4.3.8）：")
    print(f"  无均衡损失 AUC={auc_free:.4f}")
    print(f"  有均衡损失 AUC={auc_bal:.4f}")
    print("  正文要点：Top-K 路由天然倾向于把 token 集中到少数「强者」专家，")
    print("  这会让其余专家参数闲置，并在分布式训练中造成设备负载倾斜；")
    print("  负载均衡损失是让稀疏化真正兑现「算力与专家数解耦」的必要条件。")

    # ── 共享专家的作用 ───────────────────────────────────────────────────────
    print("\n共享专家 vs 路由专家（正文 §12.4.3.8）：")
    print("  DeepSeekMoE 让共享专家对所有 token 激活，捕获通用知识；")
    print("  路由专家只对部分 token 激活，捕获差异化知识。")
    n_shared_p = sum(p.numel() for p in m_bal.moe.shared_experts.parameters())
    n_routed_p = sum(p.numel() for p in m_bal.moe.routed_experts.parameters())
    print(f"  共享专家参数={n_shared_p}  路由专家总参数={n_routed_p}"
          f"  每个 token 实际激活的路由专家={m_bal.moe.router.top_k} / "
          f"{m_bal.moe.n_routed}")

    # ── 精排推理演示 ─────────────────────────────────────────────────────────
    p = m_bal.predict_proba(te["feat"][:20]).tolist()
    top5 = sorted(enumerate(p), key=lambda t: -t[1])[:5]
    print("\n精排 Top-5（候选集 20 条）:")
    for rank_i, (idx, s) in enumerate(top5, 1):
        print(f"  #{rank_i}  item={idx:2d}  p_click={s:.4f}  "
              f"label={te['y'][idx].item()}")
