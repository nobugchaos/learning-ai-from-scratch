"""MNIST 手写数字识别：用一个小型卷积神经网络在 CPU 上训练。

用法：
    python train.py            # 默认训练 3 轮
    python train.py --epochs 5
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "mnist_cnn.pt"


class Net(nn.Module):
    """两层卷积 + 两层全连接，约 42 万参数。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 14x14 -> 14x14
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # -> 32x14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # -> 64x7x7
        x = self.dropout(torch.flatten(x, 1))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_loaders(batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的均值和标准差
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=1000),
    )


def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        correct += (out.argmax(1) == y).sum().item()
        seen += len(x)
        if i % 100 == 0:
            print(f"  epoch {epoch} [{seen:>5}/{len(loader.dataset)}]  loss {loss.item():.4f}")
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    train_loader, test_loader = get_loaders(args.batch_size)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, epoch)
        test_acc = evaluate(model, test_loader)
        print(f"epoch {epoch}: loss {loss:.4f}  训练准确率 {train_acc:.2%}  "
              f"测试准确率 {test_acc:.2%}  用时 {time.time() - t0:.0f}s")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存到 {MODEL_PATH}")


if __name__ == "__main__":
    main()
