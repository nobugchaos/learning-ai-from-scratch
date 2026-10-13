"""两节机械臂：运动学、逆运动学和关节控制，只用 numpy。

用法：
    python arm.py          # 跑完全部实验，生成 arm_result.png 和 arm_reach.gif

具身智能的第一步不是"学习"，而是让身体能动：知道关节角度时手在哪（正运动学），
想让手到某处时关节该转多少（逆运动学），以及电机该出多大力才能让关节转到位（控制）。
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
L1, L2 = 1.0, 0.7      # 两节臂长（米）
M1, M2 = 1.0, 0.6      # 质量集中在每节末端（千克），手里还算上一个小物体
G = 9.81


# ---------- ① 正运动学：关节角 -> 手的位置 ----------
def forward(q):
    """返回肩、肘、手三个点的坐标。角度从水平向右量起，逆时针为正。"""
    q1, q2 = q
    elbow = np.array([L1 * np.cos(q1), L1 * np.sin(q1)])
    hand = elbow + [L2 * np.cos(q1 + q2), L2 * np.sin(q1 + q2)]
    return np.array([[0.0, 0.0], elbow, hand])


def jacobian(q):
    """雅可比矩阵：关节角各转一点点，手会往哪个方向移动多少。"""
    q1, q2 = q
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


# ---------- ② 逆运动学：手的位置 -> 关节角 ----------
def inverse_analytic(target, elbow_up=True):
    """用余弦定理直接解。同一个点一般有两组解：肘朝上、肘朝下。"""
    x, y = target
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return None  # 够不着
    q2 = np.arccos(c2) * (-1 if elbow_up else 1)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def inverse_numeric(target, q0, damping=0.05, tol=1e-6, max_iter=200):
    """数值解：每次用雅可比把"手差多少"换算成"关节该转多少"，一点点逼近。

    这就是复杂机器人（6 个、7 个关节，没有现成公式）实际在用的方法。
    damping 是阻尼最小二乘：手臂接近伸直时雅可比几乎不可逆，加阻尼防止关节乱转。
    """
    q, errors = np.array(q0, float), []
    for _ in range(max_iter):
        err = target - forward(q)[-1]
        errors.append(np.linalg.norm(err))
        if errors[-1] < tol:
            break
        J = jacobian(q)
        q += J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(2), err)
    return q, errors


# ---------- ③ 动力学：电机出力 -> 关节怎么转 ----------
def dynamics(q, dq, tau, friction=0.0):
    """拉格朗日方程 M(q)·q̈ + C(q, q̇)·q̇ + G(q) = τ，解出角加速度 q̈。"""
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    M = np.array([[(M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2, M2 * L2 ** 2 + M2 * L1 * L2 * c2],
                  [M2 * L2 ** 2 + M2 * L1 * L2 * c2, M2 * L2 ** 2]])
    h = M2 * L1 * L2 * s2
    coriolis = np.array([-h * (2 * dq[0] * dq[1] + dq[1] ** 2), h * dq[0] ** 2])  # 科里奥利力和离心力
    return np.linalg.solve(M, tau - coriolis - gravity(q) - friction * dq)


def gravity(q):
    """重力对两个关节产生的力矩。"""
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return np.array([(M1 + M2) * G * L1 * c1 + M2 * G * L2 * c12, M2 * G * L2 * c12])


def energy(q, dq):
    """总机械能 = 动能 + 势能，用来检验动力学写得对不对。"""
    pts = forward(q)
    q1d, q12d = dq[0], dq[0] + dq[1]
    v1 = L1 * q1d
    v2_sq = v1 ** 2 + (L2 * q12d) ** 2 + 2 * L1 * L2 * q1d * q12d * np.cos(q[1])
    return 0.5 * M1 * v1 ** 2 + 0.5 * M2 * v2_sq + M1 * G * pts[1, 1] + M2 * G * pts[2, 1]


def rk4_step(q, dq, tau, dt, friction=0.0):
    """四阶龙格-库塔积分：比 v += a·dt 精确得多。"""
    def f(state):
        return np.concatenate([state[2:], dynamics(state[:2], state[2:], tau, friction)])
    s = np.concatenate([q, dq])
    k1 = f(s)
    k2 = f(s + dt / 2 * k1)
    k3 = f(s + dt / 2 * k2)
    k4 = f(s + dt * k3)
    s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s[:2], s[2:]


def simulate(q_target, controller, q0, seconds=3.0, dt=0.002):
    """用控制器驱动手臂，从 q0 转向 q_target，返回每一步的关节角。"""
    q, dq, history = np.array(q0, float), np.zeros(2), []
    for _ in range(int(seconds / dt)):
        tau = controller(q, dq, q_target)
        tau = np.clip(tau, -40, 40)           # 电机力矩有上限
        q, dq = rk4_step(q, dq, tau, dt, friction=0.1)
        history.append(q.copy())
    return np.array(history)


KP, KD = np.array([60.0, 30.0]), np.array([12.0, 5.0])


def pd(q, dq, q_target):
    """PD 控制：差得越多推得越用力（P），转得越快刹得越狠（D）。"""
    return KP * (q_target - q) - KD * dq


def pd_gravity(q, dq, q_target):
    """PD + 重力补偿：先算出"托住手臂需要多大力矩"，再在此基础上做 PD。"""
    return pd(q, dq, q_target) + gravity(q)


# ---------- 实验 ----------
def main():
    t0 = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 实验 1：正运动学 + 工作空间
    rng = np.random.default_rng(0)
    qs = rng.uniform([-np.pi, -np.pi], [np.pi, np.pi], size=(4000, 2))
    hands = np.array([forward(q)[-1] for q in qs])
    print(f"实验 1  工作空间：手能到达离肩 {np.linalg.norm(hands, axis=1).min():.2f}–"
          f"{np.linalg.norm(hands, axis=1).max():.2f} 米的圆环（理论 {L1 - L2:.1f}–{L1 + L2:.1f}）")

    # 实验 2：逆运动学，两组解 + 数值解
    target = np.array([1.1, 0.6])
    up, down = inverse_analytic(target, True), inverse_analytic(target, False)
    for name, q in [("肘朝上", up), ("肘朝下", down)]:
        err = np.linalg.norm(forward(q)[-1] - target)
        print(f"实验 2  解析解（{name}）：q1={np.degrees(q[0]):6.1f}°  q2={np.degrees(q[1]):6.1f}°  误差 {err:.1e} 米")
    ax = axes[0, 0]
    ax.scatter(*hands.T, s=1, c="#cfd8dc", label="手能到达的范围（工作空间）")
    for q, color, name in [(up, "#1d9e75", "肘朝上"), (down, "#d85a30", "肘朝下")]:
        pts = forward(q)
        ax.plot(*pts.T, "-o", color=color, lw=4, ms=8, label=f"逆运动学解：{name}")
    ax.plot(*target, "*", color="#f2b134", ms=20, mec="k", label="目标点")
    ax.set_title("① 同一个目标点，有两种摆法")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)

    q_num, errors = inverse_numeric(target, q0=[0.3, 0.3])
    print(f"实验 2  数值解（雅可比迭代）：{len(errors)} 步收敛，误差 {errors[-1]:.1e} 米，"
          f"结果和{'肘朝上' if np.allclose(q_num, up, atol=1e-3) else '肘朝下'}解一致")
    reach_errors = []
    for _ in range(500):
        p = rng.uniform(-1.6, 1.6, size=2)
        if not (L1 - L2 + 0.05 < np.linalg.norm(p) < L1 + L2 - 0.05):
            continue
        _, e = inverse_numeric(p, q0=rng.uniform(-np.pi, np.pi, 2))
        reach_errors.append((len(e), e[-1]))
    steps, finals = np.array(reach_errors).T
    print(f"         随机 {len(steps)} 个可达目标：{np.mean(finals < 1e-4):.0%} 收敛到 0.1 毫米以内，"
          f"中位数 {np.median(steps):.0f} 步")
    ax = axes[0, 1]
    ax.semilogy(errors, "-o", color="#1d9e75", ms=4)
    ax.set_xlabel("迭代步数")
    ax.set_ylabel("手到目标的距离（米，对数刻度）")
    ax.set_title("② 数值逆运动学：每一步误差都大幅缩小")
    ax.grid(alpha=0.3)

    # 实验 3：关掉电机让手臂自由摆，检验动力学是否满足能量守恒
    q, dq, E = np.array([0.3, 0.5]), np.zeros(2), []
    for _ in range(5000):
        q, dq = rk4_step(q, dq, np.zeros(2), 0.001)
        E.append(energy(q, dq))
    E = np.array(E)
    print(f"实验 3  自由摆动 5 秒（无摩擦）：总机械能最大偏差 {np.ptp(E) / abs(E[0]):.2e}（相对值），能量守恒")

    # 实验 4：控制手臂转到目标姿态
    q_start = np.array([-np.pi / 2, 0.0])   # 手臂自然下垂
    results = {}
    for name, ctrl in [("只用 PD 控制", pd), ("PD + 重力补偿", pd_gravity)]:
        hist = simulate(up, ctrl, q_start)
        final_err = np.linalg.norm(forward(hist[-1])[-1] - target)
        results[name] = hist
        print(f"实验 4  {name}：3 秒后手离目标 {final_err * 1000:.1f} 毫米")
    ax = axes[1, 0]
    t = np.arange(len(hist)) * 0.002
    for (name, hist), color in zip(results.items(), ["#d85a30", "#1d9e75"]):
        dist = np.linalg.norm(np.array([forward(q)[-1] for q in hist]) - target, axis=1)
        ax.plot(t, dist * 100, color=color, lw=2, label=name)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("手到目标的距离（厘米）")
    ax.set_ylim(0, 40)
    ax.set_title("③ 只用 PD 会被重力往下拽，补偿后稳稳到位")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for (name, hist), color in zip(results.items(), ["#d85a30", "#1d9e75"]):
        pts = forward(hist[-1])
        ax.plot(*pts.T, "-o", color=color, lw=4, ms=8, label=f"{name}：最终姿态")
    path = np.array([forward(q)[-1] for q in results["PD + 重力补偿"]])
    ax.plot(*path.T, "--", color="#1d9e75", alpha=0.6, label="手的运动轨迹")
    ax.plot(*forward(q_start).T, "-o", color="#90a4ae", lw=3, ms=6, label="起始姿态（下垂）")
    ax.plot(*target, "*", color="#f2b134", ms=20, mec="k", label="目标点")
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(-1.8, 1.2)
    ax.set_aspect("equal")
    ax.set_title("④ 两种控制器的最终姿态")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(HERE / "arm_result.png", dpi=110)
    print(f"\n结果图已保存到 {HERE / 'arm_result.png'}")

    # 动画：带重力补偿的手臂从下垂伸向目标
    hist = results["PD + 重力补偿"][::25]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(-1.8, 1.2)
    ax.set_aspect("equal")
    ax.plot(*target, "*", color="#f2b134", ms=20, mec="k")
    arm_line, = ax.plot([], [], "-o", color="#1d9e75", lw=5, ms=10)
    trail, = ax.plot([], [], "--", color="#1d9e75", alpha=0.5)
    ax.set_title("PD + 重力补偿：手臂伸向目标")

    def update(i):
        arm_line.set_data(*forward(hist[i]).T)
        trail.set_data(*np.array([forward(q)[-1] for q in hist[:i + 1]]).T)
        return arm_line, trail

    FuncAnimation(fig, update, frames=len(hist), blit=True).save(HERE / "arm_reach.gif", writer=PillowWriter(fps=20))
    print(f"动画已保存到 {HERE / 'arm_reach.gif'}，总用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
