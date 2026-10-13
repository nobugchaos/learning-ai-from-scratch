"""在 MuJoCo 物理仿真里，用 PPO 训练机器人"猎豹"学会奔跑。

用法：
    pip install "gymnasium[mujoco]" stable-baselines3
    python cheetah.py                # 训练 100 万步（4 核 CPU 约 1 小时），生成学习曲线和动画
    python cheetah.py --steps 200000 # 想快点看效果，可以少训一些

05 自己从零写了策略梯度，环境也只有 4 个数、2 个动作。
这里换成真正的机器人：17 个状态（关节角度、速度……），6 个关节电机，每个电机的力矩是连续值。
算法换成 PPO —— 机器人强化学习最常用的算法，也是 ChatGPT 做 RLHF 用的算法。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

# 没有显示器的 Linux 服务器上，用纯软件渲染（OSMesa）画动画
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402,F401  在渲染库加载前先导入：反过来的顺序在 Linux + OSMesa 下会段错误
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
CKPT = HERE / "checkpoints"
ENV_ID = "HalfCheetah-v5"


def evaluate(policy_fn, obs_norm, episodes=5, seed=10_000, record=False):
    """不加随机探索、按策略最有把握的动作跑完整局（1000 步）。返回平均得分、平均速度和可选的画面。"""
    # 录像用模型自带的 "track" 机位：跟着猎豹平移，否则它几秒就跑出画面
    env = gym.make(ENV_ID, render_mode="rgb_array", width=480, height=270, camera_name="track") if record else gym.make(ENV_ID)
    scores, speeds, frames = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, x0 = 0.0, env.unwrapped.data.qpos[0]
        for t in range(1000):
            obs, reward, terminated, truncated, _ = env.step(policy_fn(obs_norm(obs)))
            total += reward
            if record and ep == 0 and t % 2 == 0 and t < 160:  # 地板只有 ±40 米，跑快了会出画
                frames.append(env.render())
            if terminated or truncated:
                break
        scores.append(total)
        speeds.append((env.unwrapped.data.qpos[0] - x0) / (1000 * env.unwrapped.dt))
    env.close()
    return float(np.mean(scores)), float(np.mean(speeds)), frames


class EvalAndSave(BaseCallback):
    """每训练 every 步：存一次模型，测一次成绩，记进 learning_curve.csv。"""

    def __init__(self, every, log_path):
        super().__init__()
        self.every, self.log_path, self.next = every, log_path, 0
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next:
            venv = self.model.get_env()
            score, speed, _ = evaluate(lambda o: self.model.predict(o, deterministic=True)[0],
                                       lambda o: venv.normalize_obs(o))
            self.model.save(CKPT / f"step_{self.num_timesteps}")
            venv.save(CKPT / f"step_{self.num_timesteps}_norm.pkl")
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, round(score, 1), round(speed, 3), round(time.time() - self.t0)])
            print(f"第 {self.num_timesteps:>7} 步  得分 {score:7.1f}  平均速度 {speed:5.2f} 米/秒  "
                  f"已用时 {(time.time() - self.t0) / 60:.1f} 分钟", flush=True)
            self.next += self.every
        return True


def train(total_steps, every):
    CKPT.mkdir(exist_ok=True)
    log_path = HERE / "learning_curve.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["步数", "得分", "平均速度(米/秒)", "用时(秒)"])
    torch.set_num_threads(1)
    # 4 个环境并行采样；VecNormalize 把观察和奖励标准化，PPO 在机器人任务上几乎都要这一步
    venv = VecNormalize(make_vec_env(ENV_ID, n_envs=4, seed=0), gamma=0.98)
    # 超参数来自 RL Baselines3 Zoo 为 HalfCheetah 调好的 PPO 配置
    model = PPO("MlpPolicy", venv, seed=0, verbose=0, device="cpu",
                n_steps=512, batch_size=64, gamma=0.98, learning_rate=2.0633e-05, ent_coef=0.000401762,
                clip_range=0.1, n_epochs=20, gae_lambda=0.92, max_grad_norm=0.8, vf_coef=0.58096,
                policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=torch.nn.ReLU,
                                   net_arch=dict(pi=[256, 256], vf=[256, 256])))
    print(f"策略网络参数：{sum(p.numel() for p in model.policy.parameters()):,} 个")
    model.learn(total_timesteps=total_steps + 1, callback=EvalAndSave(every, log_path))
    return log_path


def load(step):
    model = PPO.load(CKPT / f"step_{step}", device="cpu")
    norm = VecNormalize.load(CKPT / f"step_{step}_norm.pkl", make_vec_env(ENV_ID, n_envs=1))
    norm.training = False
    return lambda o: model.predict(o, deterministic=True)[0], norm.normalize_obs


def plot(log_path, random_score):
    rows = list(csv.reader(open(log_path)))[1:]
    steps = np.array([int(r[0]) for r in rows])
    scores = np.array([float(r[1]) for r in rows])
    speeds = np.array([float(r[2]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps / 1e4, scores, "-o", color="#1d9e75", lw=2, label="PPO 训练中的得分")
    ax.axhline(random_score, color="gray", ls="--", label="随机乱动")
    for s, sc, sp in zip(steps, scores, speeds):
        if s and (s // 100_000) % 2 == 0:
            ax.annotate(f"{sp:.1f} 米/秒", (s / 1e4, sc), textcoords="offset points", xytext=(-10, 8), fontsize=9)
    ax.set_xlabel("训练步数（万步）")
    ax.set_ylabel("一局（1000 步）的总得分")
    ax.set_title("HalfCheetah：从原地抽搐到全速奔跑")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "cheetah_curve.png", dpi=110)
    print(f"学习曲线已保存到 {HERE / 'cheetah_curve.png'}")


def make_gif(stages):
    """把几个阶段的画面左右并排，存成一个动画。"""
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 22)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)
        except OSError:
            font = ImageFont.load_default()
    n = min(len(frames) for _, frames in stages)
    gif = []
    for i in range(n):
        canvas = Image.new("RGB", (480, 270 * len(stages)))
        for k, (title, frames) in enumerate(stages):
            canvas.paste(Image.fromarray(frames[i]), (0, 270 * k))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 270 * k, 480, 270 * k + 34), fill=(20, 40, 50))
            draw.text((10, 270 * k + 4), title, fill=(255, 255, 255), font=font)
        gif.append(canvas.resize((360, int(202.5 * len(stages)))).quantize(48))  # 减少颜色数，压小文件
    gif[0].save(HERE / "cheetah.gif", save_all=True, append_images=gif[1:], duration=80, loop=0, optimize=True)
    print(f"动画已保存到 {HERE / 'cheetah.gif'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--every", type=int, default=100_000)
    parser.add_argument("--skip-train", action="store_true", help="已经训练过：只重新画图和动画")
    args = parser.parse_args()

    action_space = gym.make(ENV_ID).action_space
    action_space.seed(0)
    random_score, random_speed, random_frames = evaluate(lambda o: action_space.sample(), lambda o: o, record=True)
    print(f"随机乱动：得分 {random_score:.1f}，平均速度 {random_speed:.2f} 米/秒")

    log_path = HERE / "learning_curve.csv" if args.skip_train else train(args.steps, args.every)
    plot(log_path, random_score)

    steps = [int(r[0]) for r in list(csv.reader(open(log_path)))[1:]]
    early = steps[min(2, len(steps) - 1)]  # 训练初期（默认是第 20 万步）的存档
    stages = [(f"随机乱动  {random_speed:.1f} 米/秒", random_frames)]
    for s in (early, steps[-1]):
        score, speed, frames = evaluate(*load(s), episodes=1, record=True)
        stages.append((f"训练 {s // 10_000} 万步  {speed:.1f} 米/秒", frames))
    make_gif(stages)


if __name__ == "__main__":
    main()
