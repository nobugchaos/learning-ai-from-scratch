"""把训练过程和推理过程的真实内部数据导出成一个可交互网页 viz.html。

用法：
    python visualize.py

做两件事：
1. 用已训练好的 mnist_cnn.pt，记录 10 张测试图（0-9 各一张）经过每一层后的真实输出；
2. 在内存里从头重新训练 1 轮（不会覆盖 mnist_cnn.pt），在若干步骤拍下"快照"：
   第一层卷积核、对这 10 张图的预测、测试准确率，以及每一步的损失。
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from train import DATA_DIR, MODEL_PATH, Net

ROOT = Path(__file__).parent
OUT = ROOT / "viz.html"
SNAP_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 938]

TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def q255(t):
    """把非负张量按最大值缩放成 0-255 的整数列表。"""
    t = t.flatten()
    m = t.max().item()
    return [0] * t.numel() if m <= 0 else (t / m * 255).round().int().tolist()


def filters_of(model):
    """第一层 32 个 3x3 卷积核，按最大绝对值归一到 -1..1。"""
    w = model.conv1.weight.detach()[:, 0]  # 32x3x3
    w = w / w.abs().max()
    return [[round(v, 2) for v in f.flatten().tolist()] for f in w]


@torch.no_grad()
def layers(model, x):
    """手动走一遍前向传播，返回每一层的输出。"""
    p1 = F.max_pool2d(F.relu(model.conv1(x)), 2)   # 32x14x14
    p2 = F.max_pool2d(F.relu(model.conv2(p1)), 2)  # 64x7x7
    fc = F.relu(model.fc1(torch.flatten(p2, 1)))   # 128
    probs = torch.softmax(model.fc2(fc), 1)        # 10
    return p1[0], p2[0], fc[0], probs[0]


@torch.no_grad()
def accuracy(model, loader):
    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in loader)
    return correct / len(loader.dataset)


def main():
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=TF)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=TF)
    raw_test = datasets.MNIST(DATA_DIR, train=False)

    # 测试集中 0-9 各取第一张
    picks = {}
    for i in range(len(raw_test)):
        label = int(raw_test.targets[i])
        picks.setdefault(label, i)
        if len(picks) == 10:
            break
    picks = [picks[d] for d in range(10)]
    xs = torch.stack([test_set[i][0] for i in picks])
    imgs = [(raw_test.data[i].flatten()).tolist() for i in picks]

    # ---------- 1. 运行时：已训练模型的各层输出 ----------
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    run = []
    for d in range(10):
        p1, p2, fc, probs = layers(model, xs[d:d + 1])
        run.append({
            "label": d,
            "c1": [q255(m) for m in p1],
            "c2": [q255(m) for m in p2],
            "fc": q255(fc),
            "probs": [round(v, 4) for v in probs.tolist()],
        })
    final_filters = filters_of(model)

    # ---------- 2. 训练时：从头训练 1 轮并拍快照 ----------
    torch.manual_seed(0)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    eval_loader = DataLoader(Subset(test_set, range(2000)), batch_size=1000)

    def snap(step):
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(xs), 1)
        s = {
            "step": step,
            "acc": round(accuracy(model, eval_loader), 4),
            "filters": filters_of(model),
            "probs": [[round(v, 3) for v in p] for p in probs.tolist()],
        }
        model.train()
        print(f"  快照 step {step:>4}: 测试准确率 {s['acc']:.1%}")
        return s

    snaps, losses = [], []
    t0 = time.time()
    model.train()
    for step, (x, y) in enumerate(loader):
        if step in SNAP_STEPS:
            snaps.append(snap(step))
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 4))
    snaps.append(snap(len(loader)))
    print(f"训练 1 轮用时 {time.time() - t0:.0f}s")

    data = {
        "imgs": imgs,
        "run": run,
        "finalFilters": final_filters,
        "losses": losses,
        "snaps": snaps,
    }
    html = (ROOT / "viz_template.html").read_text(encoding="utf-8")
    OUT.write_text(html.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"已生成 {OUT}")


if __name__ == "__main__":
    main()
