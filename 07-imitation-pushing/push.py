"""模仿学习：看专家示范，学会把方块推到目标点。只用 numpy。

用法：
    python push.py     # 约 3 分钟，生成 push_result.png 和 push.gif

05 的强化学习靠"奖励"自己摸索，要试几百局；这里换一种思路：直接看专家怎么做，照着学。
这叫行为克隆（Behavior Cloning），是今天很多真实机器人（遥操作采数据 → 训练策略）的主流做法。
它有一个经典的坑：分布偏移。本项目复现这个坑，再用 DAgger 把它填上。
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
R_AGENT, R_BLOCK, R_GOAL = 0.04, 0.06, 0.04   # 机器人、方块（圆盘）半径，到达判定半径（米）
V_MAX, DT, MAX_STEPS = 0.5, 0.05, 150         # 最大速度 0.5 米/秒，每步 0.05 秒，最多 7.5 秒


# ---------- 环境：俯视的一张 1 米 × 1 米桌面 ----------
class PushEnv:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        while True:
            self.agent = self.rng.uniform(0.1, 0.9, 2)
            self.block = self.rng.uniform(0.25, 0.75, 2)
            self.goal = self.rng.uniform(0.15, 0.85, 2)
            if (np.linalg.norm(self.agent - self.block) > 0.2 and np.linalg.norm(self.goal - self.block) > 0.25):
                return self.obs()

    def obs(self):
        """机器人看到的：方块相对自己在哪、目标相对方块在哪、自己在桌上的位置。

        用相对位置而不是三个绝对坐标：同样的推法在桌上任何地方都通用，网络不用每个位置重新学一遍。
        """
        return np.concatenate([self.block - self.agent, self.goal - self.block, self.agent - 0.5])

    def step(self, action):
        v = np.clip(action, -1, 1) * V_MAX
        self.agent = np.clip(self.agent + v * DT, 0, 1)
        # 接触：机器人碰到方块就把它推开（准静态推动，方块被推多少就走多少，不会自己滑）
        reach = R_AGENT + R_BLOCK
        d = self.block - self.agent
        dist = np.linalg.norm(d)
        if dist < reach:
            normal = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
            self.block = np.clip(self.agent + normal * reach, R_BLOCK, 1 - R_BLOCK)
            # 方块被推到桌边卡住时，机器人也挤不进去，只能停在方块外面
            d = self.block - self.agent
            dist = np.linalg.norm(d)
            if dist < reach:
                self.agent = self.block - (d / dist if dist > 1e-9 else normal) * reach
        return self.obs(), self.success()

    def success(self):
        return np.linalg.norm(self.block - self.goal) < R_GOAL


# ---------- 专家：一个手写的推方块控制器 ----------
def expert(obs):
    """绕到方块背后（背对目标那一侧），再朝目标推。相当于人类遥操作员。

    动作随状态连续变化：离方块近又不在背后时，沿方块边缘绕过去；到了背后就逐渐转成往前推。
    连续很重要——如果专家在某条分界线两边给出相反的动作，学生学到的平均值就是"停在原地"。
    """
    agent = obs[4:] + 0.5
    block = agent + obs[:2]
    goal = block + obs[2:4]
    reach = R_AGENT + R_BLOCK
    push_dir = (goal - block) / (np.linalg.norm(goal - block) + 1e-8)
    to_behind = block - push_dir * (reach + 0.005) - agent
    v = to_behind * 8                                   # 朝方块背后那个点走
    rel = agent - block
    dist = np.linalg.norm(rel) + 1e-8
    normal = rel / dist
    tangent = np.array([-normal[1], normal[0]])
    tangent = tangent if np.dot(tangent, -push_dir) >= 0 else -tangent   # 沿离背后更近的方向绕
    near = np.clip((reach + 0.08 - dist) / 0.08, 0, 1)                   # 越贴近方块，绕行越强
    in_front = np.clip((np.dot(normal, push_dir) + 0.7) / 0.7, 0, 1)     # 越在方块前面，绕行越强
    v = v + near * in_front * (tangent * 1.2 + normal * 0.6)
    w_push = np.clip(1 - np.linalg.norm(to_behind) / 0.04, 0, 1)         # 到背后了：逐渐转成往前推
    v = (1 - w_push) * v + w_push * push_dir
    return np.clip(v, -1, 1)


# ---------- 学生：一个小神经网络，输入观察，输出速度 ----------
class MLP:
    def __init__(self, rng, sizes=(6, 64, 64, 2)):
        self.W = [rng.normal(0, np.sqrt(1 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def __call__(self, x):
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = np.tanh(x @ W + b)  # 最后一层也用 tanh，输出正好在 [-1, 1]
        return x

    def fit(self, X, Y, epochs=300, lr=3e-3, batch=256, rng=None):
        """均方误差 + Adam。和 01 用 PyTorch 训练的流程完全一样，只是反向传播手写。"""
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for k in range(0, len(X), batch):
                xb, yb = X[idx[k:k + batch]], Y[idx[k:k + batch]]
                acts = [xb]
                for W, b in zip(self.W, self.b):
                    acts.append(np.tanh(acts[-1] @ W + b))
                delta = 2 * (acts[-1] - yb) / len(xb) * (1 - acts[-1] ** 2)
                gW, gb = [], []
                for i in reversed(range(len(self.W))):
                    gW.insert(0, acts[i].T @ delta)
                    gb.insert(0, delta.sum(0))
                    if i:
                        delta = delta @ self.W[i].T * (1 - acts[i] ** 2)
                t += 1
                for j, (p, g) in enumerate(zip(params, gW + gb)):
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    p -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        return self


def rollout(env, policy, record=False, beta=0.0, rng=None):
    """让某个策略完整推一局。返回是否成功、经过的所有观察（以及可选的轨迹）。

    beta > 0 时，每一步有 beta 的概率由专家代为执行（DAgger 早期轮次用，避免学生一开始就跑偏太远）。
    """
    obs, states, frames = env.reset(), [], []
    for _ in range(MAX_STEPS):
        states.append(obs)
        if record:
            frames.append((env.agent.copy(), env.block.copy(), env.goal.copy()))
        act = expert(obs) if beta and rng.random() < beta else policy(obs)
        obs, done = env.step(act)
        if done:
            if record:
                frames.append((env.agent.copy(), env.block.copy(), env.goal.copy()))
            return True, np.array(states), frames
    return False, np.array(states), frames


def block_xy(states):
    """从观察里还原方块在桌上的位置。"""
    return states[:, 4:] + 0.5 + states[:, :2]


def success_rate(policy, seed, n=200):
    env = PushEnv(np.random.default_rng(seed))
    return np.mean([rollout(env, policy)[0] for _ in range(n)])


def collect_demos(n, rng):
    env, X, Y = PushEnv(rng), [], []
    for _ in range(n):
        ok, states, _ = rollout(env, expert)
        X.append(states)
        Y.append(np.array([expert(s) for s in states]))
    return np.concatenate(X), np.concatenate(Y)


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    print(f"专家（手写控制器）成功率：{success_rate(expert, seed=99):.0%}")

    # 实验 1：行为克隆，示范越多越好吗？
    demo_counts = [5, 10, 20, 50, 100, 200]
    bc_rates, bc_sizes = [], []
    for n in demo_counts:
        X, Y = collect_demos(n, np.random.default_rng(n))
        policy = MLP(np.random.default_rng(1)).fit(X, Y, rng=np.random.default_rng(2))
        rate = success_rate(policy, seed=99)
        train_err = np.mean((policy(X) - Y) ** 2)
        bc_rates.append(rate)
        bc_sizes.append(len(X))
        print(f"行为克隆  {n:>3} 段示范（{len(X):>5} 个动作）：训练误差 {train_err:.4f}，推到目标成功率 {rate:.0%}")

    # 实验 2：分布偏移 —— 学生自己推时，会走到专家从没去过的状态
    X10, Y10 = collect_demos(10, np.random.default_rng(10))
    bc10 = MLP(np.random.default_rng(1)).fit(X10, Y10, rng=np.random.default_rng(2))
    env = PushEnv(np.random.default_rng(5))
    expert_states = np.concatenate([rollout(env, expert)[1] for _ in range(30)])
    env = PushEnv(np.random.default_rng(5))
    student_states = np.concatenate([rollout(env, bc10)[1] for _ in range(30)])

    def gap(states):
        """每个状态里，学生的动作和专家的动作差多少。"""
        return np.mean(np.linalg.norm(bc10(states) - np.array([expert(s) for s in states]), axis=1))
    print(f"\n分布偏移：10 段示范训出的学生，在专家走过的状态上动作误差 {gap(expert_states):.3f}，"
          f"在自己走到的状态上误差 {gap(student_states):.3f}")

    # 实验 3：DAgger —— 学生自己推，专家在旁边告诉它"这一步本该怎么做"，把这些补进训练集
    X, Y = X10.copy(), Y10.copy()
    policy = bc10
    dagger_rates, dagger_sizes = [success_rate(policy, seed=99)], [len(X)]
    dagger_rng = np.random.default_rng(3)
    for it in range(1, 11):
        env = PushEnv(dagger_rng)
        beta = 0.5 ** it  # 第 1 轮专家代劳一半的步数，之后逐轮减半
        new = np.concatenate([rollout(env, policy, beta=beta, rng=dagger_rng)[1] for _ in range(10)])
        X = np.concatenate([X, new])
        Y = np.concatenate([Y, np.array([expert(s) for s in new])])
        policy = MLP(np.random.default_rng(1)).fit(X, Y, rng=np.random.default_rng(2))
        dagger_rates.append(success_rate(policy, seed=99))
        dagger_sizes.append(len(X))
        print(f"DAgger 第 {it} 轮：训练集 {len(X):>5} 个动作，成功率 {dagger_rates[-1]:.0%}")
    print(f"总用时 {time.time() - t0:.0f}s")

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    ax.plot(bc_sizes, np.array(bc_rates) * 100, "-o", color="#d85a30", lw=2, label="行为克隆：多录专家示范")
    ax.plot(dagger_sizes, np.array(dagger_rates) * 100, "-s", color="#1d9e75", lw=2,
            label="DAgger：从 10 段示范出发，专家给学生纠错")
    for n, x, y in zip(demo_counts, bc_sizes, bc_rates):
        ax.annotate(f"{n} 段", (x, y * 100), textcoords="offset points", xytext=(4, -14), fontsize=9, color="#d85a30")
    ax.set_xscale("log")
    ax.set_xlabel("训练用的动作标注数量（对数刻度）")
    ax.set_ylabel("推到目标的成功率（%）")
    ax.set_ylim(0, 105)
    ax.set_title("① 两条路都能补救：多录示范，或让专家给学生纠错")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(*block_xy(expert_states).T, s=4, color="#3f6fb5", alpha=0.4, label="专家推方块时方块走过的位置")
    ax.scatter(*block_xy(student_states).T, s=4, color="#d85a30", alpha=0.4, label="学生推方块时方块走过的位置")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("② 分布偏移：学生一出错，就走进没见过的地方")
    ax.legend(loc="lower left", fontsize=9, markerscale=3)

    # 同一局，10 段示范的行为克隆 vs DAgger
    episodes = {}
    for name, pol in [("行为克隆（10 段示范）", bc10), ("DAgger（10 轮纠错后）", policy)]:
        env = PushEnv(np.random.default_rng(2024))
        for _ in range(3):
            env.reset()
        ok, _, frames = rollout(env, pol, record=True)
        episodes[name] = (ok, frames)
    ax = axes[2]
    for (name, (ok, frames)), color in zip(episodes.items(), ["#d85a30", "#1d9e75"]):
        ax.plot(*np.array([f[1] for f in frames]).T, color=color, lw=2, label=f"{name}：{'成功' if ok else '失败'}（方块轨迹）")
        ax.plot(*np.array([f[0] for f in frames]).T, color=color, lw=1, ls=":", alpha=0.8)
    a0, b0, g0 = frames[0]
    ax.add_patch(Circle(b0, R_BLOCK, color="#90a4ae", alpha=0.6))
    ax.add_patch(Circle(g0, R_GOAL, fill=False, ec="#f2b134", lw=3))
    ax.plot(*a0, "o", color="k", ms=6, label="机器人起点")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("③ 同一局：虚线是机器人，实线是方块")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "push_result.png", dpi=110)
    print(f"结果图已保存到 {HERE / 'push_result.png'}")

    # 动画：并排播放
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
    artists = []
    for ax, (name, (ok, frames)) in zip(axs, episodes.items()):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        ax.add_patch(Circle(frames[0][2], R_GOAL, fill=False, ec="#f2b134", lw=3))
        block = ax.add_patch(Circle(frames[0][1], R_BLOCK, color="#90a4ae"))
        agent = ax.add_patch(Circle(frames[0][0], R_AGENT, color="#1d9e75" if ok else "#d85a30"))
        artists.append((block, agent, frames))
    n_frames = max(len(a[2]) for a in artists)

    def update(i):
        out = []
        for block, agent, frames in artists:
            f = frames[min(i, len(frames) - 1)]
            agent.center, block.center = f[0], f[1]
            out += [agent, block]
        return out

    FuncAnimation(fig, update, frames=range(0, n_frames, 2), blit=True).save(HERE / "push.gif", writer=PillowWriter(fps=15))
    print(f"动画已保存到 {HERE / 'push.gif'}")


if __name__ == "__main__":
    main()
