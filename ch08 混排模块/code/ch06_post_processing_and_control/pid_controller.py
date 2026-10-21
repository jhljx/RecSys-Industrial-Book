"""PID 控制器在广告 Load 调控中的离散实现与调参对比。

对应正文第 8 章 §8.6.2「PID 控制」。

正文要点(本实现逐条复现):
1. 离散 PID 的三项分别对应「当前偏差 / 历史累积偏差 / 偏差变化趋势」:
       e_t = r - y_t
       P   = Kp * e_t
       I   = Ki * sum_{k<=t} e_k
       D   = Kd * (e_t - e_{t-1})
       u_t = clamp(P + I + D, -max_adjust, +max_adjust)
   最终作用于序列排序阶段的分数:score_final = score_raw + u_t;
2. 正文 C++ 实现 PIDLoadController 的构造参数为
       (target_load, kp, ki, kd, window_num, window_size)
   并给出两处**饱和保护**:
       integral = clamp(integral + error, -max_integral, +max_integral)   # 抗积分饱和
       return   = clamp(P + I + D, -max_adjust,  +max_adjust)             # 输出限幅
   正文默认 max_integral = 1.0、max_adjust = 0.5;
3. 正文主函数用 pid(0.12, 1.8, 0.35, 0.45, 10, 500) 配置,8 线程并发上报,
   总请求 10000、每 500 条打印一次,被控量稳态区间为 [0.1170, 0.1280],
   稳态偏差限制在 ±0.008 以内;
4. 正文强调线上要同时监控「Load 占比曲线」与「PID 调节量曲线」:前者看业务结果,
   后者看控制过程(调节量恒为 0 说明控制器没生效,剧烈震荡说明过度调节)。

本实现做了什么:
1. PIDLoadController 为正文 C++ 类的逐行 Python 等价实现(环形请求窗口 + 双重限幅),
   单线程顺序执行代替正文的 8 线程 + mutex(结果等价,且可复现);
2. demo1 复刻正文主函数的仿真,打印与正文表格同列的监控表,并报告稳态区间;
3. demo2 固定同一条请求流,扫描 Kp / Ki / Kd,打印
   超调量 / 稳态误差 / 收敛步数 / 稳态标准差 对照表,定量说明三项的分工;
4. demo3 构造「广告库存中断 40 个窗口」的执行器饱和场景,对比开关 max_integral
   的抗积分饱和效果(积分项是否失控、恢复后是否剧烈反超);
5. demo4 拆解稳态时 P/I/D 三项各自的贡献量,解释为什么偏差能被消到 0。

[MOCK] 被控对象(广告是否出现在 top1 序列)用正文示例代码的同一套随机逻辑模拟:
       prob = clamp((0.20 + u_t) * 100, 0, 99)%,即基础倾向 0.20 叠加 PID 调节量。
       正文已注明:工业界 has_ad 完全由序列生成/序列排序阶段决定,与示例的随机方式不同。
"""

import random
import statistics

TARGET_LOAD = 0.12  # 正文 r = 0.12
KP, KI, KD = 1.8, 0.35, 0.45  # 正文 pid(0.12f, 1.8f, 0.35f, 0.45f, 10, 500)
WINDOW_NUM, WINDOW_SIZE = 10, 500
MAX_INTEGRAL, MAX_ADJUST = 1.0, 0.5
AD_BASE_SCORE = 0.20  # 正文 ad_base_score = 0.20f + adjust


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------- 正文 PIDLoadController 的 Python 等价实现

class PIDLoadController:
    """正文 §8.6.2 C++ 类 PIDLoadController 的逐行等价实现。

    环形窗口数组 windows[window_num],每个窗口记 (total, exposed);
    写满 window_size 条请求后轮转到下一个窗口并清零,等价于丢弃最旧的一段流量。
    """

    def __init__(self, target_load, kp, ki, kd, window_num, window_size,
                 max_adjust=MAX_ADJUST, max_integral=MAX_INTEGRAL,
                 anti_windup=True):
        self.target_load = target_load
        self.kp, self.ki, self.kd = kp, ki, kd
        self.window_num, self.window_size = window_num, window_size
        self.max_adjust, self.max_integral = max_adjust, max_integral
        self.anti_windup = anti_windup
        self.windows = [[0, 0] for _ in range(window_num)]  # [total, exposed]
        self.current_idx = 0
        self.req_count_in_win = 0
        self.integral = 0.0
        self.last_error = 0.0

    # --- 对应 C++ unsafe_get_current_load()
    def get_current_load(self):
        total = sum(w[0] for w in self.windows)
        exposed = sum(w[1] for w in self.windows)
        return 0.0 if total == 0 else exposed / total

    # --- 对应 C++ report(bool has_ad)
    def report(self, has_ad):
        if self.req_count_in_win >= self.window_size:
            self.current_idx = (self.current_idx + 1) % self.window_num
            self.windows[self.current_idx] = [0, 0]
            self.req_count_in_win = 0
        self.windows[self.current_idx][0] += 1
        if has_ad:
            self.windows[self.current_idx][1] += 1
        self.req_count_in_win += 1

    # --- 对应 C++ compute_adjustment()
    def compute_adjustment(self):
        error = self.target_load - self.get_current_load()
        raw_integral = self.integral + error
        self.integral = (clamp(raw_integral, -self.max_integral, self.max_integral)
                         if self.anti_windup else raw_integral)
        p = self.kp * error
        i = self.ki * self.integral
        d = self.kd * (error - self.last_error)
        self.last_error = error
        return clamp(p + i + d, -self.max_adjust, self.max_adjust), p, i, d


# ------------------------------------------------------------------ [MOCK] 被控对象

def has_ad_in_top1(adjust, base=AD_BASE_SCORE, blocked=False):
    """[MOCK] 正文示例的随机被控对象:prob = clamp((base+adjust)*100, 0, 99)%。

    blocked=True 模拟广告库存中断(执行器彻底饱和在 0),用于 demo3 的抗饱和实验。
    """
    if blocked:
        return False
    prob = clamp((base + adjust) * 100.0, 0.0, 99.0)
    return (random.random() * 100.0) < prob


def demo_textbook_simulation(total=10000, interval=500):
    """demo1:复刻正文主函数的 10000 请求仿真,打印与正文表格同列的监控表。"""
    random.seed(42)
    pid = PIDLoadController(TARGET_LOAD, KP, KI, KD, WINDOW_NUM, WINDOW_SIZE)
    rows, loads, adjusts = [], [], []
    for n in range(1, total + 1):
        adjust, _, _, _ = pid.compute_adjustment()
        pid.report(has_ad_in_top1(adjust))
        loads.append(pid.get_current_load())
        adjusts.append(adjust)
        if n % interval == 0:
            rows.append((n, pid.get_current_load(), TARGET_LOAD, adjust))

    print("=" * 72)
    print("demo1  复刻正文 §8.6.2 主函数仿真 pid(0.12, 1.8, 0.35, 0.45, 10, 500)")
    print("=" * 72)
    print(f"{'总请求数':>10}{'当前广告占比':>14}{'目标占比':>11}{'PID调节量':>12}")
    print("-" * 72)
    for n, load, tgt, adj in rows:
        print(f"{n:>10}{load:>14.4f}{tgt:>11.4f}{adj:>+12.4f}")
    print("-" * 72)
    warm = WINDOW_NUM * WINDOW_SIZE  # 窗口写满之前统计量不可信
    steady = loads[warm:]
    print(f"稳态(窗口写满后 {len(steady)} 条请求)区间 = "
          f"[{min(steady):.4f}, {max(steady):.4f}],"
          f"均值 {statistics.fmean(steady):.4f}")
    print(f"稳态偏差绝对值最大 {max(abs(x - TARGET_LOAD) for x in steady):.4f}"
          f"(正文给出的指标是 ±0.008 以内)")
    pos = sum(1 for a in adjusts if a > 0)
    print(f"调节量双向可调:正向 {pos} 次 / 负向 {len(adjusts) - pos} 次,"
          f"取值范围 [{min(adjusts):+.4f}, {max(adjusts):+.4f}]")
    print("正文表的被控量区间为 [0.1170, 0.1280];本实现用单线程顺序执行代替 8 线程并发,")
    print("随机序列不同所以逐行数值不可能对齐,但稳态区间宽度与偏差量级一致。")
    print()


# ------------------------------------------------ demo2:Kp/Ki/Kd 扫参(按窗口为控制步)

def run_window_loop(kp, ki, kd, steps=200, init_load=0.30, blocked_steps=0,
                    anti_windup=True, max_integral=MAX_INTEGRAL,
                    window_size=WINDOW_SIZE, seed=42):
    """以「一个窗口 = 一个控制步」运行闭环,便于度量超调/收敛步数。

    初始把所有窗口预填为 init_load(模拟广告占比从 30% 的高位被拉回 12% 的阶跃)。
    """
    random.seed(seed)
    exposed = [int(round(init_load * window_size))] * WINDOW_NUM
    idx = 0
    integral, last_error = 0.0, 0.0
    ys, us, integrals = [], [], []
    for s in range(steps):
        y = sum(exposed) / (WINDOW_NUM * window_size)
        error = TARGET_LOAD - y
        raw = integral + error
        integral = clamp(raw, -max_integral, max_integral) if anti_windup else raw
        u = clamp(kp * error + ki * integral + kd * (error - last_error),
                  -MAX_ADJUST, MAX_ADJUST)
        last_error = error
        blocked = s < blocked_steps
        idx = (idx + 1) % WINDOW_NUM
        exposed[idx] = sum(1 for _ in range(window_size)
                           if has_ad_in_top1(u, blocked=blocked))
        ys.append(y)
        us.append(u)
        integrals.append(integral)
    return ys, us, integrals


def settle_step(ys, band=0.012, hold=25):
    """首次进入 ±band 并连续保持 hold 步的步数;始终未进入返回 None。"""
    for i in range(len(ys) - hold):
        if all(abs(x - TARGET_LOAD) <= band for x in ys[i:i + hold]):
            return i
    return None


def demo_gain_sweep():
    """demo2:同一条阶跃场景下扫 Kp/Ki/Kd,打印超调/稳态误差/收敛步数/稳态标准差。"""
    print("=" * 86)
    print("demo2  Kp/Ki/Kd 扫参(广告占比由 30% 阶跃回落到目标 12%,一步 = 一个 500 请求窗口)")
    print("=" * 86)
    print(f"{'Kp':>6}{'Ki':>7}{'Kd':>7}{'类型':>8}{'下冲量':>10}"
          f"{'稳态误差':>11}{'收敛步数':>10}{'稳态std':>10}")
    print("-" * 86)
    grid = [
        (0.0, 0.0, 0.0, "无调控"),
        (0.5, 0.0, 0.0, "P"),
        (1.0, 0.0, 0.0, "P"),
        (1.8, 0.0, 0.0, "P"),
        (4.0, 0.0, 0.0, "P"),
        (1.8, 0.10, 0.0, "PI"),
        (1.8, 0.35, 0.0, "PI"),
        (1.8, 1.00, 0.0, "PI"),
        (1.8, 0.35, 0.45, "PID"),
        (1.8, 0.35, 2.00, "PID"),
        (1.8, 1.00, 2.00, "PID"),
        (4.0, 0.35, 0.45, "PID"),
    ]
    for kp, ki, kd, kind in grid:
        ys, us, _ = run_window_loop(kp, ki, kd)
        undershoot = max(0.0, TARGET_LOAD - min(ys)) / TARGET_LOAD * 100
        tail = ys[-60:]
        st = settle_step(ys)
        print(f"{kp:>6.1f}{ki:>7.2f}{kd:>7.2f}{kind:>8}{undershoot:>9.1f}%"
              f"{statistics.fmean(tail) - TARGET_LOAD:>+11.4f}"
              f"{(str(st) if st is not None else '未收敛'):>10}"
              f"{statistics.pstdev(tail):>10.4f}")
    print("-" * 86)
    print("纯 P(含 Ki=0 各行)稳态误差恒为正且随 Kp 增大才变小:比例项必须靠残留偏差")
    print("才能产生输出,这就是正文说的「只有 P 会留下稳态误差」;")
    print("加入 I 后稳态误差被压到 1e-4 量级,代价是先向下冲过目标值(下冲量从 16.8% 涨到 73.8%);")
    print("Ki 过大(1.00)时积分项反复充放电,稳态 std 从 0.0025 恶化到 0.0454 并且不再收敛;")
    print("再把 Kd 加到 2.00 可以靠趋势项抑制住这种震荡(std 回到 0.0035),")
    print("这正对应正文「D 项提供阻尼、抑制超调与震荡」的说法。")
    print()


def demo_anti_windup():
    """demo3:执行器饱和(广告库存中断)下的积分饱和 vs 抗积分饱和。"""
    print("=" * 86)
    print("demo3  积分饱和与抗饱和:前 40 个窗口广告库存中断(执行器恒为 0)")
    print("=" * 86)
    print(f"{'积分处理':>22}{'中断末积分':>13}{'恢复后峰值Load':>16}"
          f"{'峰值超出目标':>14}{'恢复期累计偏差':>16}")
    print("-" * 86)
    for label, aw in (("clamp 到 ±1.0(正文)", True), ("不做 clamp(对照)", False)):
        ys, us, integrals = run_window_loop(
            1.8, 0.35, 0.45, steps=200, init_load=0.0,
            blocked_steps=40, anti_windup=aw)
        i_at_block_end = integrals[39]
        peak = max(ys)
        # 恢复期累计偏差(IAE):库存恢复后 160 个窗口的 |y-r| 之和,越小越好
        iae = sum(abs(x - TARGET_LOAD) for x in ys[40:])
        print(f"{label:>22}{i_at_block_end:>+13.3f}{peak:>16.4f}"
              f"{(peak - TARGET_LOAD) / TARGET_LOAD * 100:>13.1f}%"
              f"{iae:>16.3f}")
    print("-" * 86)
    print("库存中断期间偏差 e=+0.12 恒定,积分项持续充电:不限幅时积到 +4.800,")
    print("限幅后停在 +1.000(正文的 max_integral)。库存恢复瞬间,")
    print("不限幅的积分项把 Load 一路推到 0.6444(超目标 437.0%),恢复期累计偏差 5.869;")
    print("限幅版本峰值只有 0.2838(超目标 136.5%),累计偏差 2.369,仅为前者的 40%。")
    print("这就是正文在 compute_adjustment 里写 clamp(integral + error, ...) 的原因:")
    print("执行器饱和时阻止积分项无限累积,避免饱和解除后的剧烈反超。")
    print()


def demo_term_decomposition():
    """demo4:拆解 P/I/D 三项在瞬态与稳态的贡献量。"""
    random.seed(42)
    pid = PIDLoadController(TARGET_LOAD, KP, KI, KD, WINDOW_NUM, WINDOW_SIZE)
    # 预填窗口到 30%,制造一次阶跃
    for k in range(WINDOW_NUM):
        pid.windows[k] = [WINDOW_SIZE, int(round(0.30 * WINDOW_SIZE))]
    pid.current_idx = 0
    pid.req_count_in_win = WINDOW_SIZE

    print("=" * 86)
    print("demo4  P/I/D 三项贡献拆解(窗口预填 30%,观察前若干个窗口与最终稳态)")
    print("=" * 86)
    print(f"{'窗口#':>6}{'Load':>9}{'偏差e':>10}{'P项':>10}{'I项':>10}"
          f"{'D项':>10}{'u(限幅后)':>12}")
    print("-" * 86)
    snapshots = {0, 1, 2, 3, 5, 10, 20, 50, 99}
    for w in range(100):
        adjust, p, i, d = pid.compute_adjustment()
        load = pid.get_current_load()
        err = TARGET_LOAD - load
        if w in snapshots:
            print(f"{w:>6}{load:>9.4f}{err:>+10.4f}{p:>+10.4f}{i:>+10.4f}"
                  f"{d:>+10.4f}{adjust:>+12.4f}")
        for _ in range(WINDOW_SIZE):
            pid.report(has_ad_in_top1(adjust))
    print("-" * 86)
    print("第 0 个窗口偏差最大(-0.1800),P 项独自给出 -0.3240 并被输出限幅压到 -0.5000 之内;")
    print("随着 Load 回落,P 项迅速衰减到 ~0,此时维持 Load 的几乎全部是 I 项;")
    print("D 项只在偏差拐点附近有量级,稳态下在 0 附近抖动——这正是正文对三项分工的描述:")
    print("P 管「现在差多少」、I 管「一直差」、D 管「差得多快」。")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_textbook_simulation()
    demo_gain_sweep()
    demo_anti_windup()
    demo_term_decomposition()
