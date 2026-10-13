"""强化学习入门：从零写策略梯度（REINFORCE），让小车把杆子立住。只用 numpy。

用法：
    python cartpole.py     # 训练约 1 分钟，生成 cartpole_result.png 和 cartpole.gif

和 01–03 的区别：那里每张图片都有"标准答案"，这里没有。
智能体只知道"杆子还立着就 +1 分"，要自己摸索出怎么推小车。
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
MAX_STEPS = 500  # 坚持 500 步（10 秒）就算满分


# ---------- 环境：小车立杆的物理 ----------
class CartPole:
    """和 Gymnasium 的 CartPole-v1 用同一套方程和参数。

    状态 4 个数：小车位置 x、小车速度 ẋ、杆子偏角 θ、杆子角速度 θ̇。
    动作 2 个：向左推或向右推，力都是 10 牛。
    杆子偏超过 12° 或小车跑出 ±2.4 米，本局结束。
    """
    gravity, cart_mass, pole_mass, half_len, force, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        f = self.force if action == 1 else -self.force
        total = self.cart_mass + self.pole_mass
        cos, sin = np.cos(theta), np.sin(theta)
        # 刚体动力学：小车受推力，杆子受重力，两者通过铰链互相牵连
        temp = (f + self.pole_mass * self.half_len * theta_dot ** 2 * sin) / total
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.half_len * (4 / 3 - self.pole_mass * cos ** 2 / total))
        x_acc = temp - self.pole_mass * self.half_len * theta_acc * cos / total
        # 欧拉法前进一步（和 Gymnasium 一致）
        x, x_dot = x + self.dt * x_dot, x_dot + self.dt * x_acc
        theta, theta_dot = theta + self.dt * theta_dot, theta_dot + self.dt * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = abs(x) > 2.4 or abs(theta) > 12 * np.pi / 180
        return self.state.copy(), 1.0, done


# ---------- 策略：一个小神经网络，输入状态，输出"向右推"的概率 ----------
class Policy:
    def __init__(self, rng, hidden=16):
        self.W1 = rng.normal(0, 0.5, size=(4, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, size=hidden)
        self.b2 = 0.0

    def forward(self, s):
        h = np.tanh(s @ self.W1 + self.b1)
        p = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))  # sigmoid：向右推的概率
        return p, h

    def act(self, s, rng, greedy=False):
        p, h = self.forward(s)
        a = int(p > 0.5) if greedy else int(rng.random() < p)
        return a, p, h

    def grad_log_prob(self, s, a, p, h):
        """∇ log π(a|s)：让刚才选的动作"更可能/更不可能"时，参数该往哪调。手写反向传播。"""
        d_logit = a - p                       # sigmoid + 对数概率的导数正好是 (a - p)
        d_h = d_logit * self.W2 * (1 - h ** 2)  # 穿过 tanh
        return {"W2": d_logit * h, "b2": d_logit, "W1": np.outer(s, d_h), "b1": d_h}

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


def run_episode(env, policy, rng, greedy=False):
    s, traj = env.reset(), []
    for _ in range(MAX_STEPS):
        a, p, h = policy.act(s, rng, greedy)
        traj.append((s, a, p, h))
        s, _, done = env.step(a)
        if done:
            break
    return traj


def train(seed, episodes=600, lr=0.01, gamma=0.99, batch=5):
    """REINFORCE：玩一局，拿到好结果的动作加强，坏结果的动作削弱。"""
    rng = np.random.default_rng(seed)
    env, policy = CartPole(rng), Policy(rng)
    adam = {k: [np.zeros_like(np.asarray(v, float)), np.zeros_like(np.asarray(v, float))] for k, v in policy.params().items()}
    lengths, t = [], 0
    for ep in range(0, episodes, batch):
        grads = {k: np.zeros_like(np.asarray(v, float)) for k, v in policy.params().items()}
        for _ in range(batch):
            traj = run_episode(env, policy, rng)
            lengths.append(len(traj))
            # 回报：从这一步往后还能坚持多久（越远的未来打折越多）
            G, returns = 0.0, []
            for _ in traj:
                G = 1 + gamma * G
                returns.append(G)
            returns = np.array(returns[::-1])
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化：比平均好的才加强
            for (s, a, p, h), R in zip(traj, returns):
                for k, g in policy.grad_log_prob(s, a, p, h).items():
                    grads[k] += R * g / batch
        # Adam 优化器做梯度上升（想让回报变大，所以是 +）
        t += 1
        for k, param in policy.params().items():
            m, v = adam[k]
            m[...] = 0.9 * m + 0.1 * grads[k]
            v[...] = 0.999 * v + 0.001 * grads[k] ** 2
            step = lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            if k == "b2":
                policy.b2 += float(step)
            else:
                param += step
    return policy, np.array(lengths)


def evaluate(policy, seed, n=100):
    rng = np.random.default_rng(seed)
    env = CartPole(rng)
    return np.array([len(run_episode(env, policy, rng, greedy=True)) for _ in range(n)])


def main():
    t0 = time.time()
    rng = np.random.default_rng(123)

    random_lengths = []
    env = CartPole(rng)
    for _ in range(100):
        env.reset()
        for k in range(MAX_STEPS):
            if env.step(int(rng.random() < 0.5))[2]:
                break
        random_lengths.append(k + 1)
    print(f"随机乱推：平均坚持 {np.mean(random_lengths):.1f} 步（{np.mean(random_lengths) * 0.02:.2f} 秒）")

    seeds = [0, 1, 2]
    curves, policies = [], []
    for seed in seeds:
        policy, lengths = train(seed)
        scores = evaluate(policy, seed=1000 + seed)
        curves.append(lengths)
        policies.append(policy)
        first = next((i for i in range(len(lengths) - 20) if lengths[i:i + 20].mean() >= 475), None)
        print(f"种子 {seed}：训练 {len(lengths)} 局，"
              f"{'第 ' + str(first) + ' 局起连续 20 局平均 ≥475 步' if first is not None else '未达到连续 20 局平均 ≥475 步'}；"
              f"训练后测试 100 局平均 {scores.mean():.1f} 步，满分（500 步）{np.mean(scores == MAX_STEPS):.0%}")
    print(f"训练用时 {time.time() - t0:.1f}s")

    # 学习曲线
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for seed, lengths, color in zip(seeds, curves, ["#1d9e75", "#d85a30", "#3f6fb5"]):
        smooth = np.convolve(lengths, np.ones(20) / 20, mode="valid")
        ax.plot(lengths, color=color, alpha=0.15)
        ax.plot(np.arange(19, len(lengths)), smooth, color=color, lw=2, label=f"种子 {seed}（20 局滑动平均）")
    ax.axhline(np.mean(random_lengths), color="gray", ls="--", label="随机乱推")
    ax.set_xlabel("训练局数")
    ax.set_ylabel("坚持步数（500 为满分）")
    ax.set_title("① 学习曲线：从乱推到稳稳立住")
    ax.legend()
    ax.grid(alpha=0.3)

    # 学到的策略：固定小车静止，看杆子偏角和角速度不同时，它想往哪推
    policy = policies[0]
    th, thd = np.meshgrid(np.linspace(-0.21, 0.21, 120), np.linspace(-2, 2, 120))
    states = np.stack([np.zeros_like(th), np.zeros_like(th), th, thd], axis=-1)
    prob = policy.forward(states.reshape(-1, 4))[0].reshape(th.shape)
    ax = axes[1]
    im = ax.pcolormesh(np.degrees(th), thd, prob, cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im, ax=ax, label="向右推的概率")
    ax.set_xlabel("杆子偏角（度，正 = 向右倒）")
    ax.set_ylabel("杆子角速度（弧度/秒）")
    ax.set_title("② 学到的策略：杆往哪边倒，就往哪边推小车")
    fig.tight_layout()
    fig.savefig(HERE / "cartpole_result.png", dpi=110)
    print(f"结果图已保存到 {HERE / 'cartpole_result.png'}")

    # 动画：训练好的策略
    env, states = CartPole(np.random.default_rng(7)), []
    s = env.reset()
    for _ in range(300):
        states.append(s)
        s, _, done = env.step(policy.act(s, rng, greedy=True)[0])
        if done:
            break
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.axhline(0, color="#555", lw=1)
    cart = ax.add_patch(Rectangle((0, 0), 0.4, 0.2, color="#3f6fb5"))
    pole, = ax.plot([], [], color="#d85a30", lw=6, solid_capstyle="round")
    text = ax.text(-2.3, 1.15, "", fontsize=10)

    def update(i):
        x, _, theta, _ = states[i]
        cart.set_xy((x - 0.2, 0))
        pole.set_data([x, x + 2 * CartPole.half_len * np.sin(theta)], [0.2, 0.2 + 2 * CartPole.half_len * np.cos(theta)])
        text.set_text(f"第 {i} 步   偏角 {np.degrees(theta):+.1f}°")
        return cart, pole, text

    FuncAnimation(fig, update, frames=range(0, len(states), 2), blit=True).save(HERE / "cartpole.gif", writer=PillowWriter(fps=25))
    print(f"动画已保存到 {HERE / 'cartpole.gif'}，总用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
