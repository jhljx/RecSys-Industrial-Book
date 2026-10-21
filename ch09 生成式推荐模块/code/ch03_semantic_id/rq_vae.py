"""对应正文第 9 章 §9.3.1「RQ-VAE」与 §9.3.2「RQ-VAE 的工程实践问题」。

实现内容:
  * Encoder f_theta: x -> z_0;
  * 逐层残差量化(RQ): s^l = argmin_{k in C^l} ||z_{l-1} - e_k||^2,
    z_l = z_{l-1} - e_{s^l}, 于是 z_0 ~= sum_{l=1}^{L} e_{s^l};
  * Decoder g_phi: x_hat = g_phi(sum_l e_{s^l});
  * 正文的整体训练目标 L = L_recon + lambda * sum_l ||z_{l-1} - e_{s^l}||^2,
    工程上把该量化项按 VQ-VAE 惯例拆成 codebook loss 与 commitment loss:
      ||sg[z_{l-1}] - e_{s^l}||^2 + beta * ||z_{l-1} - sg[e_{s^l}]||^2;
  * STE 梯度近似: z_STE = z_e + sg(z_q - z_e), 前向等于 z_q, 反向 dz_STE/dz_e = 1;
  * 码本坍缩诊断: p_k = n_k / sum_j n_j, Perplexity = exp(-sum_k p_k log p_k),
    均匀使用时 Perplexity = K, 远小于 K 则说明出现 Codebook Collapse。

本脚本在合成的「带簇结构」item embedding 上训练一个小规模 RQ-VAE, 打印
每层码本的使用率 / Perplexity / 逐层累计重构误差数值表。
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_items(n: int, d: int, n_cluster: int, gen: torch.Generator) -> torch.Tensor:
    """合成带簇结构的 item embedding(模拟内容大模型输出)。"""
    centers = torch.randn(n_cluster, d, generator=gen) * 2.0
    idx = torch.randint(0, n_cluster, (n,), generator=gen)
    return centers[idx] + 0.35 * torch.randn(n, d, generator=gen)


class ResidualQuantizer(nn.Module):
    """L 层残差量化器, 每层码本大小 K。"""

    def __init__(self, n_layer: int, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.n_layer, self.K, self.beta = n_layer, codebook_size, beta
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, dim) * 0.1) for _ in range(n_layer)]
        )

    def forward(self, z0: torch.Tensor):
        residual = z0
        quantized = torch.zeros_like(z0)
        codes, cb_loss, cm_loss = [], z0.new_zeros(()), z0.new_zeros(())
        layer_err = []
        for layer in range(self.n_layer):
            e = self.codebooks[layer]                       # (K, d)
            # s^l = argmin_k ||z_{l-1} - e_k||^2
            dist = (residual.pow(2).sum(1, keepdim=True)
                    - 2 * residual @ e.t()
                    + e.pow(2).sum(1).unsqueeze(0))
            s = dist.argmin(1)
            codes.append(s)
            e_s = e[s]
            # 正文量化项的两个工程拆分: codebook loss + commitment loss
            cb_loss = cb_loss + F.mse_loss(e_s, residual.detach())
            cm_loss = cm_loss + F.mse_loss(residual, e_s.detach())
            quantized = quantized + e_s
            residual = residual - e_s
            layer_err.append(residual.detach().pow(2).sum(1).mean().item())
        # STE: z_q 前向用量化值, 反向把梯度直通给 encoder 输出
        z_ste = z0 + (quantized - z0).detach()
        quant_loss = cb_loss + self.beta * cm_loss
        return z_ste, torch.stack(codes, 1), quant_loss, layer_err


class RQVAE(nn.Module):
    def __init__(self, in_dim: int, lat_dim: int, n_layer: int, K: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, lat_dim))
        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, 64), nn.GELU(), nn.Linear(64, in_dim))
        self.rq = ResidualQuantizer(n_layer, K, lat_dim)

    def forward(self, x: torch.Tensor, lam: float = 1.0):
        z0 = self.encoder(x)
        z_ste, codes, quant_loss, layer_err = self.rq(z0)
        x_hat = self.decoder(z_ste)
        recon = F.mse_loss(x_hat, x)
        return recon + lam * quant_loss, recon, codes, layer_err


def perplexity(codes_layer: torch.Tensor, K: int) -> tuple[float, float]:
    """返回 (Perplexity, 码本使用率)。"""
    n_k = torch.bincount(codes_layer, minlength=K).double()
    p = n_k / n_k.sum()
    nz = p[p > 0]
    return float(torch.exp(-(nz * nz.log()).sum())), float((n_k > 0).float().mean())


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    N, D, LAT, L, K = 3000, 48, 16, 3, 64
    x = make_items(N, D, n_cluster=40, gen=gen)
    model = RQVAE(D, LAT, L, K)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    print("=" * 72)
    print(f"RQ-VAE 训练: N={N} items, dim={D}, latent={LAT}, L={L} 层, K={K}/层")
    print(f"语义 ID 空间 = K^L = {K ** L:,}")
    print("-" * 72)
    print(f"{'step':>6}{'total':>11}{'recon':>11}{'层1 PPL':>10}{'层2 PPL':>10}{'层3 PPL':>10}")
    hist = []
    for step in range(301):
        perm = torch.randperm(N, generator=gen)[:512]
        loss, recon, codes, layer_err = model(x[perm])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 60 == 0:
            ppls = [perplexity(codes[:, l], K)[0] for l in range(L)]
            print(f"{step:>6}{loss.item():>11.4f}{recon.item():>11.4f}"
                  + "".join(f"{p:>10.1f}" for p in ppls))
            hist.append((step, recon.item(), ppls))

    with torch.no_grad():
        loss, recon, codes, layer_err = model(x)
        z0 = model.encoder(x)

    print()
    print("=" * 72)
    print("表 1  各层码本使用情况(Perplexity 上限 = K = %d)" % K)
    print("-" * 72)
    print(f"{'层 l':>5}{'被用码字数':>12}{'使用率':>10}{'Perplexity':>13}{'PPL/K':>9}")
    for l in range(L):
        ppl, usage = perplexity(codes[:, l], K)
        print(f"{l + 1:>5}{int(usage * K):>12}{usage:>10.1%}{ppl:>13.2f}{ppl / K:>9.1%}")

    print()
    print("表 2  逐层残差量化的累计逼近效果 z_0 ~= sum_l e_{s^l}")
    print("-" * 72)
    base = z0.pow(2).sum(1).mean().item()
    print(f"{'量化层数':>10}{'残差能量 E||z_l||^2':>22}{'相对初始':>12}")
    print(f"{0:>10}{base:>22.4f}{1.0:>12.1%}")
    for l, err in enumerate(layer_err, 1):
        print(f"{l:>10}{err:>22.4f}{err / base:>12.1%}")

    print()
    print("表 3  最终重构质量")
    print("-" * 72)
    with torch.no_grad():
        x_hat = model.decoder(model.rq(z0)[0])
        mse = F.mse_loss(x_hat, x).item()
        var = x.var().item()
    print(f"输入方差            = {var:.4f}")
    print(f"重构 MSE L_recon    = {mse:.4f}")
    print(f"相对重构误差 MSE/Var = {mse / var:.2%}")
    uniq = len({tuple(r.tolist()) for r in codes})
    print(f"唯一语义 ID 数       = {uniq} / {N} items (冲突率 {1 - uniq / N:.2%})")
    print("-" * 72)
    print("注: 本实现严格按正文 L = L_recon + lambda*sum_l||z_{l-1}-e_{s^l}||^2,")
    print("    并用 STE(z_e + sg(z_q - z_e)) 把 Decoder 梯度直通到 Encoder。")


if __name__ == "__main__":
    main()
