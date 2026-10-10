"""复现 micrograd 作者 demo.ipynb 里的实验：用纯 Python 写的小神经网络把两个"月牙"分开。

用法（先把 micrograd 克隆到本文件夹下）：
    git clone https://github.com/karpathy/micrograd
    python run_demo.py
"""
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "micrograd"))
from micrograd.engine import Value  # noqa: E402
from micrograd.nn import MLP  # noqa: E402

OUT = Path(__file__).parent / "moons_result.png"


def make_moons(n=100, noise=0.1):
    """和 sklearn.datasets.make_moons 一样：两个交错的半圆，标签为 -1 / +1。"""
    t1 = np.linspace(0, math.pi, n // 2)
    t2 = np.linspace(0, math.pi, n - n // 2)
    X = np.vstack([
        np.column_stack([np.cos(t1), np.sin(t1)]),
        np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5]),
    ])
    X += np.random.normal(scale=noise, size=X.shape)
    y = np.array([-1] * (n // 2) + [1] * (n - n // 2))
    return X, y


def loss_and_acc(model, X, y):
    inputs = [list(map(Value, xrow)) for xrow in X]
    scores = [model(x) for x in inputs]
    # SVM "最大间隔"损失：分数和标签同号且足够大（>1）时损失为 0
    losses = [(1 + -yi * s).relu() for yi, s in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = 1e-4 * sum(p * p for p in model.parameters())  # L2 正则，防止权重过大
    acc = sum((yi > 0) == (s.data > 0) for yi, s in zip(y, scores)) / len(y)
    return data_loss + reg_loss, acc


def draw(model, X, y, path):
    """画出决策边界：背景颜色表示模型对该位置的判断，圆点是训练数据。"""
    W, H, S = 600, 450, 150  # 像素宽、高，1 个单位 = 150 像素
    x0, y0 = -1.5, 2.0       # 图左上角对应的坐标
    img = Image.new("RGB", (W, H))
    px = img.load()
    step = 6
    for i in range(0, W, step):
        for j in range(0, H, step):
            s = model([Value(x0 + (i + step / 2) / S), Value(y0 - (j + step / 2) / S)]).data
            color = (255, 214, 196) if s < 0 else (196, 226, 214)
            for a in range(i, min(i + step, W)):
                for b in range(j, min(j + step, H)):
                    px[a, b] = color
    d = ImageDraw.Draw(img)
    for (a, b), label in zip(X, y):
        cx, cy = (a - x0) * S, (y0 - b) * S
        fill = (216, 90, 48) if label < 0 else (29, 158, 117)
        d.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=fill, outline=(40, 40, 40))
    img.save(path)


def main():
    np.random.seed(1337)
    random.seed(1337)
    X, y = make_moons()

    model = MLP(2, [16, 16, 1])  # 2 个输入 -> 16 -> 16 -> 1 个输出
    print(model)
    print(f"参数个数: {len(model.parameters())}\n")

    t0 = time.time()
    for k in range(100):
        loss, acc = loss_and_acc(model, X, y)   # ① 前向传播：算损失
        model.zero_grad()                        # ② 清空上一步的梯度
        loss.backward()                          # ③ 反向传播：micrograd 自己实现的！
        lr = 1.0 - 0.9 * k / 100                 # ④ 学习率从 1.0 逐渐降到 0.1
        for p in model.parameters():
            p.data -= lr * p.grad                # ⑤ 沿梯度反方向更新每个参数
        if k % 10 == 0 or k == 99:
            print(f"第 {k:>2} 步  损失 {loss.data:.4f}  准确率 {acc:.0%}")
    print(f"\n训练用时 {time.time() - t0:.1f}s")

    draw(model, X, y, OUT)
    print(f"决策边界图已保存到 {OUT}")


if __name__ == "__main__":
    main()
