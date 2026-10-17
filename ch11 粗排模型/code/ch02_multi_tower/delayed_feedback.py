"""
延迟反馈多塔粗排模型（第11章 §11.2.3 延迟反馈多塔模型）

论文出处：Chapelle, "Modeling Delayed Feedback in Display Advertising", KDD 2014
（DFM）；Ktena et al., "Addressing Delayed Feedback for Continuous Training with
Neural Networks in CTR Prediction", RecSys 2019（FNW / Fake Negative Weighting）。

核心思想：
点击等即时反馈在曝光后数秒内产生，而购买、支付等延迟反馈可能数天后才到达。
模型共享一个用户塔，物品侧分别建「即时反馈物品塔」与「延迟反馈物品塔」：
    s_click = <e_user, e_item^click>,  s_delay = <e_user, e_item^delay>
    L = L_click + λ·L_delay
核心问题是假负样本（Fake Negative）：曝光时延迟标签初始化为 0，但其中一部分
样本未来仍会转化，直接当负样本训练会系统性低估转化概率。
本文件实现两条数据流下的建模方式：
(a) 实时数据流（在线增量训练）：延迟时间分布建模（d ~ Exp(rate)，仍会转化的
    概率 P(d > t) = exp(-rate·t)）+ 假负样本修正的重要性加权 w = p/q，其中
    w_neg = 1 / (1 + p_pos·exp(-rate·t))，曝光时间越短权重越低（削弱错误监督），
    t 增大时 w → 1。
(b) 离线数据流（周期训练）：等待窗口 W 成熟后回填（Backfill）标签
    y_W = 1 if (t1 - t0) <= W else 0，标签完整性高但数据新鲜度下降。

本实现：纯 Python 标准库，反向传播采用近似梯度（只更新最后一层权重 + Embedding）。
"""

import math
import random


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _relu(x: float) -> float:
    return max(0.0, x)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class MLP:
    """简单 MLP：前向传播 + 最后一层近似梯度更新。"""

    def __init__(self, dims: list[int], lr: float, reg: float):
        self.lr, self.reg = lr, reg
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            s = math.sqrt(2.0 / in_d)
            self.weights.append([[random.gauss(0, s) for _ in range(out_d)] for _ in range(in_d)])
            self.biases.append([0.0] * out_d)
        self.input_dim, self.output_dim = dims[0], dims[-1]

    def forward(self, x: list[float]) -> tuple[list[float], list[list[float]]]:
        assert len(x) == self.input_dim, f"输入维度 {len(x)} != 期望 {self.input_dim}"
        acts, h = [x], x
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = [sum(h[j] * W[j][k] for j in range(len(h))) + b[k] for k in range(len(b))]
            if idx < len(self.weights) - 1:
                h = [_relu(v) for v in h]
            acts.append(h)
        return h, acts

    def update_last_layer(self, grad_out: list[float], acts: list[list[float]]) -> None:
        h_prev, W, b = acts[-2], self.weights[-1], self.biases[-1]
        for j in range(len(h_prev)):
            for k in range(len(grad_out)):
                W[j][k] -= self.lr * (grad_out[k] * h_prev[j] + self.reg * W[j][k])
        for k in range(len(grad_out)):
            b[k] -= self.lr * grad_out[k]


class DelayDistribution:
    """转化延迟分布建模（DFM 思想）：d ~ Exp(rate)，P(d > t) = exp(-rate·t) 为在途概率。"""

    def __init__(self, rate: float = 0.5):
        self.rate = rate

    def prob_still_pending(self, elapsed: float) -> float:
        """曝光后经过 elapsed 时间，转化「尚未到达但仍会发生」的概率。"""
        return math.exp(-self.rate * max(0.0, elapsed))

    def expected_delay(self) -> float:
        """指数分布期望延迟 E[d] = 1 / rate。"""
        return 1.0 / self.rate

    def fit(self, observed_delays: list[float]) -> None:
        """极大似然估计：rate_hat = n / Σ d_i。"""
        total = sum(observed_delays)
        if observed_delays and total > 1e-9:
            self.rate = len(observed_delays) / total


class DelayedFeedbackTower:
    """
    延迟反馈多塔模型：共享用户塔 + 即时反馈物品塔 + 延迟反馈物品塔。
    s_click = <e_user, e_item^click>,  s_delay = <e_user, e_item^delay>
    L = L_click + λ·L_delay（延迟分支支持重要性加权纠偏）
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 16,
                 hidden_dims: list[int] | None = None, output_dim: int = 16,
                 lr: float = 0.05, reg: float = 1e-4, lambda_delay: float = 1.0,
                 delay_rate: float = 0.5, prior_pos: float = 0.3):
        hidden_dims = hidden_dims or [32, 16]
        self.output_dim, self.lr = output_dim, lr
        self.lambda_delay = lambda_delay
        self.prior_pos = prior_pos            # 转化率先验 p_pos，用于重要性权重
        self.delay_dist = DelayDistribution(delay_rate)

        scale = math.sqrt(1.0 / emb_dim)
        self.user_emb = [[random.uniform(-scale, scale) for _ in range(emb_dim)] for _ in range(n_users)]
        self.item_emb = [[random.uniform(-scale, scale) for _ in range(emb_dim)] for _ in range(n_items)]
        dims = [emb_dim] + hidden_dims + [output_dim]
        self.user_tower = MLP(dims, lr, reg)         # 所有样本都更新
        self.click_tower = MLP(dims, lr, reg)        # 仅即时反馈信号更新
        self.delay_tower = MLP(dims, lr, reg)        # 仅延迟反馈信号更新

    def encode_user(self, uid: int) -> tuple[list[float], list[list[float]]]:
        return self.user_tower.forward(self.user_emb[uid])

    def encode_item_click(self, iid: int) -> tuple[list[float], list[list[float]]]:
        return self.click_tower.forward(self.item_emb[iid])

    def encode_item_delay(self, iid: int) -> tuple[list[float], list[list[float]]]:
        return self.delay_tower.forward(self.item_emb[iid])

    def predict_delay_prob(self, uid: int, iid: int) -> float:
        """延迟反馈（转化）概率 σ(s_delay)。"""
        return _sigmoid(_dot(self.encode_user(uid)[0], self.encode_item_delay(iid)[0]))

    # ── (a) 实时数据流：假负样本修正 + 重要性加权 ──────────────────────────
    def importance_weight(self, y_delay: float, elapsed: float) -> float:
        """
        重要性权重 w = p/q，纠正观测标签分布与真实分布的偏差。
        观测为负的样本中混入了比例 p_fake = p_pos·exp(-rate·t) 的假负样本，故
            w_neg = 1 / (1 + p_fake)
        t 很小（刚曝光）时 w_neg 最低，监督被削弱；t 增大后 w_neg → 1。
        观测为正的样本给 >1 的权重，补偿被延迟「吞掉」的那部分正样本。
        """
        if y_delay > 0.0:
            return 1.0 + self.delay_dist.prob_still_pending(elapsed)
        p_fake = self.prior_pos * self.delay_dist.prob_still_pending(elapsed)
        return 1.0 / (1.0 + p_fake)

    def _step(self, uid: int, iid: int, y_click: float, y_delay: float,
              elapsed: float, debias: bool) -> tuple[float, float]:
        """单条样本的一次联合更新，返回 (L_click, L_delay)。"""
        e_u, u_acts = self.encode_user(uid)
        e_c, c_acts = self.encode_item_click(iid)
        e_d, d_acts = self.encode_item_delay(iid)
        # 即时反馈分支
        p_c = _sigmoid(_dot(e_u, e_c))
        l_c = -(y_click * math.log(p_c + 1e-9) + (1 - y_click) * math.log(1 - p_c + 1e-9))
        g_c = p_c - y_click
        self.click_tower.update_last_layer([g_c * e_u[k] for k in range(self.output_dim)], c_acts)
        # 延迟反馈分支（可选重要性加权纠偏）
        w = self.importance_weight(y_delay, elapsed) if debias else 1.0
        p_d = _sigmoid(_dot(e_u, e_d))
        l_d = -(y_delay * math.log(p_d + 1e-9) + (1 - y_delay) * math.log(1 - p_d + 1e-9))
        g_d = (p_d - y_delay) * w
        self.delay_tower.update_last_layer([g_d * e_u[k] for k in range(self.output_dim)], d_acts)
        # 用户塔：两个任务的梯度共同更新 → 同时学习短期与长期兴趣
        d_u = [g_c * e_c[k] + self.lambda_delay * g_d * e_d[k] for k in range(self.output_dim)]
        self.user_tower.update_last_layer(d_u, u_acts)
        ue, ie = self.user_emb[uid], self.item_emb[iid]
        g_i = g_c + self.lambda_delay * g_d
        for j in range(len(ue)):
            ue[j] -= self.lr * d_u[j % self.output_dim] * 0.1
            ie[j] -= self.lr * g_i * e_u[j % self.output_dim] * 0.1
        return l_c, l_d * w

    def train_realtime(self, stream: list[tuple[int, int, float, float, float]],
                       epochs: int = 6, debias: bool = True, verbose: bool = True) -> None:
        """
        实时数据流训练（事件驱动）：stream 每条为
        (uid, iid, y_click, y_delay_observed, elapsed)。同一条曝光可能出现两次：
        第一次 y_delay=0（标签未成熟），延迟行为到达后再进入一次 y_delay=1，
        完成标签修正（Label Correction）。
        """
        tag = "纠偏" if debias else "不纠偏"
        for epoch in range(epochs):
            random.shuffle(stream)
            lc = ld = 0.0
            for uid, iid, y_c, y_d, el in stream:
                a, b = self._step(uid, iid, y_c, y_d, el, debias)
                lc, ld = lc + a, ld + b
            n = max(len(stream), 1)
            if verbose:
                print(f"  [{tag}] epoch {epoch:2d}  L_click={lc / n:.4f}  "
                      f"L_delay={ld / n:.4f}  L_total={(lc + self.lambda_delay * ld) / n:.4f}")

    # ── (b) 离线数据流：等待窗口 + 回填标签 ────────────────────────────────
    @staticmethod
    def backfill_labels(raw_logs: list[tuple[int, int, float, float, float]],
                        window: float) -> list[tuple[int, int, float, float, float]]:
        """
        离线周期训练的样本组织：等待窗口 W 成熟后回填（Backfill）延迟标签
            y_W = 1 if (t1 - t0) <= W else 0
        raw_logs 每条为 (uid, iid, y_click, true_delay, conv_happen)。
        返回与实时流同构的样本（elapsed 置为 window，标签已成熟）。
        """
        return [(u, i, y_c, 1.0 if (cv > 0.0 and d <= window) else 0.0, window)
                for u, i, y_c, d, cv in raw_logs]

    def train_offline(self, raw_logs: list[tuple[int, int, float, float, float]],
                      window: float = 3.0, epochs: int = 6) -> None:
        """离线周期训练：标签已回填成熟，无需重要性加权纠偏（debias=False）。"""
        data = self.backfill_labels(raw_logs, window)
        pos = sum(1 for d in data if d[3] > 0)
        print(f"  [离线 W={window}天] 回填后正样本 {pos}/{len(data)}（标签完整性高、新鲜度低）")
        self.train_realtime(data, epochs=epochs, debias=False)

    def pre_rank(self, uid: int, candidates: list[int], top_k: int = 20,
                 alpha_click: float = 1.0, alpha_delay: float = 0.5) -> list[tuple[int, float]]:
        """粗排：多塔分数融合 s_final = Σ α_i·s_i（两个物品塔均可离线预计算）。"""
        e_u, _ = self.encode_user(uid)
        scores = [(iid, alpha_click * _dot(e_u, self.encode_item_click(iid)[0])
                   + alpha_delay * _dot(e_u, self.encode_item_delay(iid)[0]))
                  for iid in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _make_logs(n_users: int, n_items: int, true_cvr: float, rate: float):
    """合成曝光日志：(uid, iid, y_click, true_delay, conv_happen)。"""
    logs = []
    for uid in range(n_users):
        for iid in range(n_items):
            if random.random() > 0.45:
                continue
            y_c = 1.0 if random.random() < 0.5 else 0.0
            conv = 1.0 if (y_c > 0 and random.random() < true_cvr) else 0.0
            logs.append((uid, iid, y_c,
                         random.expovariate(rate) if conv > 0 else 99.0, conv))
    return logs


# ── 快速验证 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    n_users, n_items, TRUE_CVR, RATE, OBS_T = 8, 24, 0.40, 0.5, 0.6
    raw_logs = _make_logs(n_users, n_items, TRUE_CVR, RATE)
    # 实时流：曝光时刻只能观测到 delay <= OBS_T 的转化，其余都是假负样本
    stream = [(u, i, yc, 1.0 if (cv > 0 and d <= OBS_T) else 0.0, OBS_T)
              for u, i, yc, d, cv in raw_logs]
    obs_pos = sum(1 for s in stream if s[3] > 0)
    real_pos = sum(1 for l in raw_logs if l[4] > 0)
    truth = real_pos / len(raw_logs)
    print(f"合成数据：曝光 {len(raw_logs)} 条，真实转化 {real_pos} 条，实时流观测到 "
          f"{obs_pos} 条（假负样本 {real_pos - obs_pos} 条）\n真实转化率 = {truth:.4f}，"
          f"实时流朴素观测转化率 = {obs_pos / len(raw_logs):.4f}\n")

    print("【实时数据流 A】不做纠偏（假负样本直接当负样本）：")
    m_naive = DelayedFeedbackTower(n_users, n_items, 16, lr=0.05,
                                   delay_rate=RATE, prior_pos=TRUE_CVR)
    m_naive.train_realtime(list(stream), epochs=5, debias=False)

    print("\n【实时数据流 B】重要性加权纠偏（FNW + 指数延迟分布）：")
    m_fix = DelayedFeedbackTower(n_users, n_items, 16, lr=0.05,
                                 delay_rate=RATE, prior_pos=TRUE_CVR)
    m_fix.delay_dist.fit([l[3] for l in raw_logs if l[4] > 0])
    print(f"  延迟分布 MLE 估计 rate={m_fix.delay_dist.rate:.4f}，"
          f"期望延迟 {m_fix.delay_dist.expected_delay():.4f} 天")
    m_fix.train_realtime(list(stream), epochs=5, debias=True)

    pairs = [(u, i) for u, i, *_ in raw_logs]
    avg_naive = sum(m_naive.predict_delay_prob(u, i) for u, i in pairs) / len(pairs)
    avg_fix = sum(m_fix.predict_delay_prob(u, i) for u, i in pairs) / len(pairs)
    print(f"\n正样本概率估计对比（真实转化率 {truth:.4f}）："
          f"\n  不做纠偏：平均预估 {avg_naive:.4f}，偏差 {avg_naive - truth:+.4f}"
          f"\n  做了纠偏：平均预估 {avg_fix:.4f}，偏差 {avg_fix - truth:+.4f}")

    print("\n【离线数据流】等待窗口 + 回填标签：")
    m_off = DelayedFeedbackTower(n_users, n_items, 16, lr=0.05, delay_rate=RATE)
    m_off.train_offline(raw_logs, window=3.0, epochs=5)

    uid = 0
    print(f"\n用户 {uid} 粗排 Top-5（s = 1.0·s_click + 0.5·s_delay）：")
    for iid, s in m_off.pre_rank(uid, list(range(n_items)), top_k=5):
        print(f"  item {iid:2d}: {s:.4f}")
