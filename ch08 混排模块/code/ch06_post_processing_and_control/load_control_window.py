"""基于请求窗口的 Load 调控闭环(多业务流量占比的稳态控制)。

对应正文第 8 章 §8.6.1「Load 调控」。

正文要点(本实现逐条复现):
1. 工业界为各业务设定**流量占比目标与安全波动区间**,正文给出的典型配置是
   广告 10%~15%、直播 5%~8%、普通内容兜底不低于 80%;
2. 通过**滑动窗口 / 固定请求窗口**实时统计各业务的实际曝光占比。正文的 C++ 工程实现是
   window_num 个窗口 × window_size 条请求的环形缓冲,窗口写满即轮转、丢弃最旧数据;
3. 用实际占比 y_t 与目标占比 r_t 的偏差动态修正排序分数:score_final = score_raw + u_t;
4. Load 调控位于推荐链路末端,本质是一道**熔断保护**:即便上游召回/排序/混排模型异常
   (正文举例「广告预估价值被异常高估」),也能把各业务流量约束回安全区间。

本实现做了什么:
- `RequestWindow`:正文 `std::vector<Window> windows_` 的 Python 等价实现;
- `CumulativeCounter`:对照组,全局累计统计(不开窗),用来说明为什么正文要求用窗口;
- `LoadController`:多业务调控器,按 e_t = r_t - y_t 产生分数调节量 u_t,
  含正文提到的**积分限幅** max_integral 与**调节量裁剪** max_adjust;
  调节量每个窗口刷新一次(而非每请求一次),对应正文「基于窗口统计的调控」;
- 普通内容不直接施加调节量,它是广告/直播被压回目标后的**残差兜底**,
  这与正文「普通内容兜底不低于 80%」的表述一致。

关键公式(取自正文):
    e_t = r_t - y_t                                 偏差
    u_t = clip(Kp*e_t + Ki*clip(sum e, ±I_max), ±u_max)
    score_final = score_raw + u_t
"""

import random
import statistics

# ---------------------------------------------------- 正文给定的业务流量约束配置

# 业务 -> (目标占比 r, 安全下界, 安全上界)
LOAD_CONFIG = {
    "ad":     (0.125, 0.10, 0.15),   # 正文:广告流量占比约束在 10%~15%
    "live":   (0.065, 0.05, 0.08),   # 正文:直播流量占比约束在 5%~8%
    "normal": (0.810, 0.80, 1.00),   # 正文:普通内容兜底不低于 80%
}
CONTROLLED = ("ad", "live")          # 普通内容为残差兜底,不直接加调节量

SEQ_LEN = 8          # 一次请求下发的坑位数
WINDOW_NUM = 10      # 正文 C++ 示例:window_num = 10
WINDOW_SIZE = 200    # 正文示例 window_size = 500,这里缩小以便单进程快速跑完
CTRL_PERIOD = 25     # 每 25 次请求按最新窗口统计刷新一次调节量

# [MOCK] 各业务原始分的分布(mu, sigma):普通内容分最高、广告最低,
# 因此「纯按模型分排序」时广告天然拿不到目标占比,必须靠调控抬升/压制。
SCORE_DIST = {"normal": (0.80, 0.10), "live": (0.62, 0.12), "ad": (0.58, 0.12)}


class RequestWindow:
    """固定请求窗口的环形缓冲,与正文 C++ 实现一一对应。

    - `report()` 累加当前窗口的 total / exposed;
    - 当前窗口写满 window_size 条就轮转到下一个窗口并清零(丢弃最旧数据);
    - `load()` 对所有窗口求和后算占比,即 exposed / total。
    """

    def __init__(self, window_num=WINDOW_NUM, window_size=WINDOW_SIZE):
        self.window_num = window_num
        self.window_size = window_size
        self.windows = [[0, 0] for _ in range(window_num)]  # [total, exposed]
        self.idx = 0
        self.cnt_in_win = 0

    def report(self, exposed_cnt, total_cnt=1):
        if self.cnt_in_win >= self.window_size:
            self.idx = (self.idx + 1) % self.window_num
            self.cnt_in_win = 0
            self.windows[self.idx] = [0, 0]
        w = self.windows[self.idx]
        w[0] += total_cnt
        w[1] += exposed_cnt
        self.cnt_in_win += total_cnt

    def load(self):
        total = sum(w[0] for w in self.windows)
        exposed = sum(w[1] for w in self.windows)
        return 0.0 if total == 0 else exposed / total

    def capacity(self):
        return self.window_num * self.window_size


class CumulativeCounter:
    """对照组:全局累计统计(不开窗)。"""

    def __init__(self):
        self.total = 0
        self.exposed = 0

    def report(self, exposed_cnt, total_cnt=1):
        self.total += total_cnt
        self.exposed += exposed_cnt

    def load(self):
        return 0.0 if self.total == 0 else self.exposed / self.total


class LoadController:
    """多业务 Load 调控器(比例 + 带限幅的积分;完整三项 PID 见 pid_controller.py)。"""

    def __init__(self, kp=1.0, ki=0.3, max_integral=5.0, max_adjust=0.5,
                 enabled=True, window_num=WINDOW_NUM, window_size=WINDOW_SIZE):
        self.kp, self.ki = kp, ki
        self.max_integral, self.max_adjust = max_integral, max_adjust
        self.enabled = enabled
        self.windows = {b: RequestWindow(window_num, window_size)
                        for b in LOAD_CONFIG}
        self.integral = {b: 0.0 for b in LOAD_CONFIG}
        self.u = {b: 0.0 for b in LOAD_CONFIG}

    def report(self, seq):
        """上报一次请求实际下发的序列(逐坑位记账)。"""
        for biz in LOAD_CONFIG:
            self.windows[biz].report(sum(1 for it in seq if it["biz"] == biz),
                                     len(seq))

    def load(self, biz):
        return self.windows[biz].load()

    def refresh(self):
        """窗口刷新时重算各业务调节量 u_t。"""
        if not self.enabled:
            return self.u
        for biz in CONTROLLED:
            r = LOAD_CONFIG[biz][0]
            e = r - self.load(biz)
            self.integral[biz] = max(-self.max_integral,
                                     min(self.max_integral,
                                         self.integral[biz] + e))
            u = self.kp * e + self.ki * self.integral[biz]
            self.u[biz] = max(-self.max_adjust, min(self.max_adjust, u))
        return self.u

    def in_safe_range(self, biz):
        _, lo, hi = LOAD_CONFIG[biz]
        return lo <= self.load(biz) <= hi


# ------------------------------------------------------ [MOCK] 上游混排打分

def make_candidates(n_per_biz=10, ad_score_bias=0.0):
    """[MOCK] 合成一次请求的候选池。

    `ad_score_bias` 模拟正文说的「广告预估价值被异常高估」故障:
    该偏置直接加在广告原始分上,>0 即为上游异常。
    """
    pool = []
    for i in range(n_per_biz):
        for biz, (mu, sigma) in SCORE_DIST.items():
            bias = ad_score_bias if biz == "ad" else 0.0
            pool.append({"id": f"{biz[0].upper()}{i}", "biz": biz,
                         "raw": random.gauss(mu, sigma) + bias})
    return pool


def serve(pool, adjust_map, length=SEQ_LEN):
    """score_final = score_raw + u_t,按最终分取 Top-L 作为下发序列。"""
    for it in pool:
        it["final"] = it["raw"] + adjust_map.get(it["biz"], 0.0)
    return sorted(pool, key=lambda x: x["final"], reverse=True)[:length]


def simulate(controller, n_req, ad_score_bias=0.0, fault_at=None,
             fault_bias=0.0, tail=600):
    """跑一段在线流量,返回最后 `tail` 个请求的窗口占比轨迹。"""
    hist = {b: [] for b in LOAD_CONFIG}
    adj = {b: 0.0 for b in LOAD_CONFIG}
    for t in range(n_req):
        if t % CTRL_PERIOD == 0:
            adj = dict(controller.refresh())
        bias = fault_bias if (fault_at is not None and t >= fault_at) \
            else ad_score_bias
        controller.report(serve(make_candidates(ad_score_bias=bias), adj))
        if t >= n_req - tail:
            for b in LOAD_CONFIG:
                hist[b].append(controller.load(b))
    return hist


# ---------------------------------------------------------------- 演示 1

def demo_window_vs_cumulative(n_req=1500, fault_at=500, fault_bias=0.35):
    """窗口统计 vs 全局累计:故障发生后谁能及时反映真实占比。"""
    win, cum = RequestWindow(), CumulativeCounter()
    print("=" * 86)
    print(f"【1】统计口径:请求窗口 vs 全局累计(第 {fault_at} 次请求起广告分被异常抬高)")
    print("=" * 86)
    print(f"窗口容量 = {WINDOW_NUM} x {WINDOW_SIZE} = {win.capacity()} 条曝光记录"
          f";不开调控,只比统计口径")
    print()
    print(f"{'请求数':>8}{'真实占比(近100请求)':>22}{'窗口统计':>12}{'全局累计':>12}"
          f"{'窗口-全局':>12}")
    print("-" * 86)
    recent = []
    for t in range(1, n_req + 1):
        bias = fault_bias if t > fault_at else 0.0
        seq = serve(make_candidates(ad_score_bias=bias), {})
        n_ad = sum(1 for it in seq if it["biz"] == "ad")
        win.report(n_ad, len(seq))
        cum.report(n_ad, len(seq))
        recent.append(n_ad / len(seq))
        if len(recent) > 100:
            recent.pop(0)
        if t % 250 == 0:
            print(f"{t:>8}{statistics.fmean(recent):>22.4f}{win.load():>12.4f}"
                  f"{cum.load():>12.4f}{win.load() - cum.load():>12.4f}")
    print("-" * 86)
    print("全局累计被故障前的历史数据稀释,始终低估当前真实占比;")
    print("请求窗口只看最近 2000 条曝光,能快速跟上分布漂移 -> 正文要求用窗口统计。")


# ---------------------------------------------------------------- 演示 2

def demo_circuit_breaker(n_req=3000, fault_at=1200, fault_bias=0.35):
    """熔断效果:上游广告分异常抬高时,开/关 Load 调控的占比对比。"""
    print()
    print("=" * 86)
    print(f"【2】熔断保护:第 {fault_at} 次请求起广告原始分被抬高 +{fault_bias}")
    print("=" * 86)
    rows = []
    for label, enabled in (("关闭 Load 调控", False), ("开启 Load 调控", True)):
        ctl = LoadController(enabled=enabled)
        hist = simulate(ctl, n_req, fault_at=fault_at, fault_bias=fault_bias)
        rows.append((label, ctl, hist))
    print(f"{'场景':<18}{'广告占比':>10}{'[min,max]':>18}{'直播占比':>10}"
          f"{'普通占比':>10}{'广告u_t':>10}{'广告判定':>11}")
    print("-" * 86)
    for label, ctl, hist in rows:
        ad = hist["ad"]
        rng = f"[{min(ad):.3f},{max(ad):.3f}]"
        ok = "在区间内" if ctl.in_safe_range("ad") else "越界!"
        print(f"{label:<18}{statistics.fmean(ad):>10.4f}{rng:>18}"
              f"{statistics.fmean(hist['live']):>10.4f}"
              f"{statistics.fmean(hist['normal']):>10.4f}"
              f"{ctl.u['ad']:>10.4f}{ok:>11}")
    print("-" * 86)
    print(f"目标 r_ad = {LOAD_CONFIG['ad'][0]:.3f},安全区间 "
          f"[{LOAD_CONFIG['ad'][1]:.2f}, {LOAD_CONFIG['ad'][2]:.2f}]"
          f"(统计取最后 600 个请求的窗口占比)")
    print("关闭调控时故障直接把广告占比推到 77%,用户体验被摧毁;")
    print("开启调控后积分项吃掉这个常值偏置(u_ad 变为负数),占比仍稳在目标附近。")


# ---------------------------------------------------------------- 演示 3

def demo_three_business(n_req=3000):
    """三业务同时调控:正文的 广告10~15% / 直播5~8% / 普通>=80% 能否同时满足。"""
    print()
    print("=" * 86)
    print("【3】三业务联合调控(正文配置:广告 10~15% / 直播 5~8% / 普通 >=80%)")
    print("=" * 86)
    off = LoadController(enabled=False)
    on = LoadController(enabled=True)
    h_off = simulate(off, n_req)
    h_on = simulate(on, n_req)
    print(f"{'业务':<10}{'目标 r':>9}{'安全区间':>14}{'无调控':>10}{'有调控':>10}"
          f"{'偏差 e_t':>10}{'u_t':>9}{'结论':>11}")
    print("-" * 86)
    for biz in ("ad", "live", "normal"):
        r, lo, hi = LOAD_CONFIG[biz]
        y_off = statistics.fmean(h_off[biz])
        y_on = statistics.fmean(h_on[biz])
        ok = "达标" if lo <= y_on <= hi else "未达标"
        print(f"{biz:<10}{r:>9.3f}{f'[{lo:.2f},{hi:.2f}]':>14}{y_off:>10.4f}"
              f"{y_on:>10.4f}{r - y_on:>10.4f}{on.u[biz]:>9.4f}{ok:>11}")
    print("-" * 86)
    print("纯按模型分排序时三业务占比完全由分数分布决定,与产品约束无关:")
    print("广告分最低所以只拿到 7.33%(低于 10% 下界),直播却超配到 13.54%;")
    print("加上 Load 调控后广告被抬升、直播被压低,三业务同时落入正文的安全区间。")


# ---------------------------------------------------------------- 演示 4

def demo_window_size(n_req=3000):
    """窗口长度的取舍:窗口越短响应越快但统计噪声越大。"""
    print()
    print("=" * 86)
    print("【4】窗口容量对调控质量的影响(目标 r_ad = 0.125)")
    print("=" * 86)
    print(f"{'窗口容量':>10}{'窗口数':>8}{'窗口长':>8}{'广告占比均值':>14}"
          f"{'占比标准差':>12}{'峰谷差':>10}")
    print("-" * 86)
    for num, size in ((5, 20), (10, 50), (10, 200), (10, 500)):
        random.seed(42)
        ctl = LoadController(window_num=num, window_size=size)
        hist = simulate(ctl, n_req)["ad"]
        print(f"{num * size:>10}{num:>8}{size:>8}{statistics.fmean(hist):>14.4f}"
              f"{statistics.pstdev(hist):>12.4f}"
              f"{max(hist) - min(hist):>10.4f}")
    print("-" * 86)
    print("窗口过短(100 条)时占比估计噪声大,调节量跟着抖,峰谷差明显变宽;")
    print("窗口过长(5000 条)时统计更平滑,但对突发故障的响应会被拖慢。")
    print("正文 C++ 示例取 10 x 500 = 5000 条,是在稳定性与响应速度之间折中。")


if __name__ == "__main__":
    random.seed(42)
    demo_window_vs_cumulative()
    demo_circuit_breaker()
    demo_three_business()
    demo_window_size()
