"""
奇异值分解（SVD）召回 —— 矩阵的最佳低秩逼近（第10章 §10.1.5）

核心思想：
    将用户-物品评分矩阵 R ∈ R^{|U|×|I|} 分解为 R = U Σ V^T，只保留最大的前 d 个
    奇异值得到低秩近似 R ≈ U_d Σ_d V_d^T。根据 Eckart-Young 定理，截断 SVD 就是
        min_{rank(X) ≤ d} ||R - X||_F^2
    的解析最优解，因此标准 SVD 无需梯度下降即可直接求得最优解。
    从表示学习角度看，U_d Σ_d^{1/2} 是用户的低维表示，V_d Σ_d^{1/2} 是物品的低维表示。

求解方法（纯 Python，不使用 numpy）：
    1) 构造对称矩阵 A = R R^T（取 |U|、|I| 中较小的一侧以降低计算量）；
    2) 用 Jacobi 旋转法把 A 对角化，得到特征值 λ_t 与特征向量 u_t；
    3) σ_t = sqrt(λ_t)，v_t = R^T u_t / σ_t，即得到截断 SVD 的三元组。

与后面的 MF（矩阵分解，§10.1.5 第二部分，见 mf.py）的对比：
    * SVD 要求输入矩阵「全部元素已知」，而推荐系统的 R 是极度稀疏的，缺失值是
      Unknown 而不是 0。强行用 0 填充会把「没看过」当成「不喜欢」，引入巨大偏差；
      用均值/行均值填充虽然缓解了偏差，但填充值本身会主导 Frobenius 损失。
    * SVD 需要对整个矩阵做分解，复杂度约 O(min(m,n)^2 · max(m,n))，在千万级
      用户/物品规模上完全不可行。
    * MF（Funk SVD）只在观测集合 Ω 上做优化：
          min Σ_{(u,i)∈Ω} (R_ui - p_u^T q_i)^2 + λ(||P||_F^2 + ||Q||_F^2)
      完全跳过缺失值，用 SGD 求解，因此才真正适配推荐系统的稀疏数据。
      本文件末尾会用数值对比展示「零填充 SVD」在观测位置上的重构误差为何偏大。
"""

import math
import random

Matrix = list[list[float]]


# ── 基础线性代数工具 ──────────────────────────────────────────────────────────
def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    """标准矩阵乘法：(p×q) × (q×r) -> (p×r)。"""
    bt = _transpose(b)
    return [[_dot(row, col) for col in bt] for row in a]


def _frobenius(a: Matrix, b: Matrix) -> float:
    """||A - B||_F。"""
    return math.sqrt(sum((a[i][j] - b[i][j]) ** 2 for i in range(len(a)) for j in range(len(a[0]))))


def jacobi_eigen(sym: Matrix, max_sweeps: int = 60, tol: float = 1e-12) -> tuple[list[float], Matrix]:
    """
    Jacobi 旋转法求实对称矩阵的全部特征值与特征向量。

    每次选取非对角元 A[p][q]，构造 Givens 旋转 G(p,q,θ) 使 A'[p][q] = 0：
        θ = (A_qq - A_pp) / (2 A_pq),   t = sign(θ) / (|θ| + sqrt(θ^2 + 1))
        c = 1 / sqrt(t^2 + 1),          s = t · c
    反复扫描直到非对角元的 Frobenius 范数小于 tol，此时 A -> diag(λ)，
    累乘的旋转矩阵 V 的第 t 列即为 λ_t 对应的特征向量。

    Returns:
        (eigenvalues, V)，V[i][t] 表示第 t 个特征向量的第 i 个分量。
    """
    n = len(sym)
    a = [row[:] for row in sym]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                # A <- G^T A G，分两步作用在列和行上
                for i in range(n):
                    aip, aiq = a[i][p], a[i][q]
                    a[i][p] = c * aip - s * aiq
                    a[i][q] = s * aip + c * aiq
                for j in range(n):
                    apj, aqj = a[p][j], a[q][j]
                    a[p][j] = c * apj - s * aqj
                    a[q][j] = s * apj + c * aqj
                # V <- V G
                for i in range(n):
                    vip, viq = v[i][p], v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq
    return [a[i][i] for i in range(n)], v


def truncated_svd(r: Matrix, d: int) -> tuple[Matrix, list[float], Matrix]:
    """
    截断 SVD：R ≈ U_d Σ_d V_d^T。

    Args:
        r: m×n 的稠密矩阵（缺失值必须已被填充）
        d: 保留的奇异值个数

    Returns:
        (U_d: m×d, sigma: 长度 d, V_d: n×d)
    """
    m, n = len(r), len(r[0])
    rt = _transpose(r)
    d = min(d, m, n)

    # 对较小的一侧做对角化：A = R R^T (m×m) 或 R^T R (n×n)
    left_side = m <= n
    a = _matmul(r, rt) if left_side else _matmul(rt, r)
    eigvals, eigvecs = jacobi_eigen(a)

    order = sorted(range(len(eigvals)), key=lambda t: -eigvals[t])[:d]
    sigma = [math.sqrt(max(eigvals[t], 0.0)) for t in order]

    if left_side:
        u = [[eigvecs[i][t] for t in order] for i in range(m)]
        # v_t = R^T u_t / σ_t
        v = [[0.0] * d for _ in range(n)]
        for k, t in enumerate(order):
            if sigma[k] < 1e-10:
                continue
            col = [_dot(rt[j], [eigvecs[i][t] for i in range(m)]) / sigma[k] for j in range(n)]
            for j in range(n):
                v[j][k] = col[j]
    else:
        v = [[eigvecs[j][t] for t in order] for j in range(n)]
        u = [[0.0] * d for _ in range(m)]
        for k, t in enumerate(order):
            if sigma[k] < 1e-10:
                continue
            col = [_dot(r[i], [eigvecs[j][t] for j in range(n)]) / sigma[k] for i in range(m)]
            for i in range(m):
                u[i][k] = col[i]
    return u, sigma, v


def reconstruct(u: Matrix, sigma: list[float], v: Matrix) -> Matrix:
    """重构低秩近似矩阵 R_hat = U_d Σ_d V_d^T。"""
    d = len(sigma)
    return [[sum(u[i][k] * sigma[k] * v[j][k] for k in range(d)) for j in range(len(v))] for i in range(len(u))]


def latent_factors(u: Matrix, sigma: list[float], v: Matrix) -> tuple[Matrix, Matrix]:
    """把奇异值均匀分到两侧：P = U_d Σ_d^{1/2}，Q = V_d Σ_d^{1/2}，于是 R ≈ P Q^T。"""
    root = [math.sqrt(s) for s in sigma]
    p = [[u[i][k] * root[k] for k in range(len(sigma))] for i in range(len(u))]
    q = [[v[j][k] * root[k] for k in range(len(sigma))] for j in range(len(v))]
    return p, q


def recommend(r_hat: Matrix, uid: int, observed: set[int], top_k: int = 5) -> list[tuple[int, float]]:
    """用重构矩阵的预测评分做 Top-K 推荐（过滤已交互物品）。"""
    scores = [(j, r_hat[uid][j]) for j in range(len(r_hat[0])) if j not in observed]
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def _rmse_on_observed(r_hat: Matrix, obs: dict[tuple[int, int], float]) -> float:
    """只在观测位置上计算 RMSE，用于公平比较不同填充策略。"""
    return math.sqrt(sum((r_hat[i][j] - val) ** 2 for (i, j), val in obs.items()) / len(obs))


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)

    n_users, n_items, rank_d = 8, 20, 4
    # 构造一个「真实低秩 + 噪声」的评分矩阵（评分 1~5），再随机挖掉 65% 的元素
    true_p = [[random.uniform(0.2, 1.0) for _ in range(3)] for _ in range(n_users)]
    true_q = [[random.uniform(0.2, 1.0) for _ in range(3)] for _ in range(n_items)]
    observed: dict[tuple[int, int], float] = {}
    for i in range(n_users):
        for j in range(n_items):
            if random.random() < 0.35:
                score = 1.0 + 4.0 * min(1.0, _dot(true_p[i], true_q[j]) / 2.2) + random.gauss(0, 0.2)
                observed[(i, j)] = max(1.0, min(5.0, score))
    print(f"评分矩阵规模 {n_users}×{n_items}，观测元素 {len(observed)} 个，稀疏度 "
          f"{1 - len(observed) / (n_users * n_items):.1%}")

    # 两种缺失值填充策略：零填充 vs 全局均值填充
    global_mean = sum(observed.values()) / len(observed)
    r_zero = [[observed.get((i, j), 0.0) for j in range(n_items)] for i in range(n_users)]
    r_mean = [[observed.get((i, j), global_mean) for j in range(n_items)] for i in range(n_users)]
    print(f"观测评分全局均值 = {global_mean:.4f}")

    print("\n【Jacobi 截断 SVD】（填充策略：全局均值）")
    u, sigma, v = truncated_svd(r_mean, rank_d)
    print(f"  U_d 形状 = {len(u)}×{len(u[0])}   V_d 形状 = {len(v)}×{len(v[0])}")
    print("  前 %d 个奇异值 σ = [%s]" % (rank_d, ", ".join(f"{s:.4f}" for s in sigma)))

    # 能量占比：Σσ_t^2 / ||R||_F^2
    total_energy = sum(r_mean[i][j] ** 2 for i in range(n_users) for j in range(n_items))
    kept = sum(s * s for s in sigma)
    print(f"  低秩能量占比 = {kept / total_energy:.4%}   ||R||_F = {math.sqrt(total_energy):.4f}")

    r_hat_mean = reconstruct(u, sigma, v)
    print(f"  rank-{rank_d} 近似的 Frobenius 误差 ||R - R_hat||_F = {_frobenius(r_mean, r_hat_mean):.4f}")

    # 验证正交性：U_d^T U_d ≈ I
    gram = _matmul(_transpose(u), u)
    max_off = max(abs(gram[a][b] - (1.0 if a == b else 0.0)) for a in range(rank_d) for b in range(rank_d))
    print(f"  正交性校验 max|U^T U - I| = {max_off:.2e}")

    # 低维表示：P = U_d Σ_d^{1/2}, Q = V_d Σ_d^{1/2}
    p_mat, q_mat = latent_factors(u, sigma, v)
    print(f"  用户低维表示 P 形状 = {len(p_mat)}×{len(p_mat[0])}，物品低维表示 Q 形状 = "
          f"{len(q_mat)}×{len(q_mat[0])}")
    print(f"  校验 p_0·q_0 = {_dot(p_mat[0], q_mat[0]):.4f} ≈ R_hat[0][0] = {r_hat_mean[0][0]:.4f}")

    print("\n【稀疏矩阵上的缺陷：缺失值填充策略直接决定结果】")
    u0, s0, v0 = truncated_svd(r_zero, rank_d)
    r_hat_zero = reconstruct(u0, s0, v0)
    print(f"  零填充   观测位置 RMSE = {_rmse_on_observed(r_hat_zero, observed):.4f}")
    print(f"  均值填充 观测位置 RMSE = {_rmse_on_observed(r_hat_mean, observed):.4f}")
    print("  说明：零填充把 Unknown 当成「不喜欢」，重构值被大量 0 拉低，观测位置误差明显更大；")
    print("        这正是推荐系统放弃标准 SVD、转向只在观测集合 Ω 上优化的 MF（Funk SVD）的原因。")

    print("\n【不同保留秩 d 对应的重构误差（均值填充）】")
    for dd in (1, 2, 4, 6, 8):
        uu, ss, vv = truncated_svd(r_mean, dd)
        rr = reconstruct(uu, ss, vv)
        print(f"  d={dd}:  ||R-R_hat||_F = {_frobenius(r_mean, rr):8.4f}   观测 RMSE = "
              f"{_rmse_on_observed(rr, observed):.4f}")

    print("\n【Top-K 推荐结果（基于 rank-4 均值填充重构矩阵）】")
    for uid in (0, 1):
        seen = {j for (i, j) in observed if i == uid}
        recs = recommend(r_hat_mean, uid, seen, top_k=5)
        pretty = "  ".join(f"item{j:02d}({sc:.3f})" for j, sc in recs)
        print(f"  用户 {uid}（已交互 {len(seen)} 个物品）Top-5：{pretty}")
    print(f"  注意：未观测位置的预测值大量塌缩到填充值 {global_mean:.3f} 附近，个性化区分度很弱，")
    print("        这是标准 SVD 用于稀疏召回场景的另一个固有缺陷。")
