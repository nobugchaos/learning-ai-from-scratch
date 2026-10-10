"""最简单的 nanoGPT：一个字一个字地学写《红楼梦》。

原理和 ChatGPT 完全一样（Transformer + 预测下一个字），只是小了几万倍。

用法：
    python nanogpt.py                 # 自动判断：有 GPU 用 medium，没有用 small
    python nanogpt.py --size small    # 笔记本 CPU 上跑，几分钟
    python nanogpt.py --size medium   # GPU 上跑，效果好得多
"""
import argparse
import time
from pathlib import Path

import torch

from model import GPT

ROOT = Path(__file__).parent

# ================================================================
# ⓪ 超参数：网络多大、训练多久
# ================================================================
CONFIGS = {
    #          看多少个字   层数        注意力头数   每个字用多少个数表示   每批几段    学习率   训练步数
    "small":  dict(block_size=64,  n_layer=4, n_head=4, n_embd=128, batch_size=32, lr=1e-3, iters=2000, dropout=0.1),
    "medium": dict(block_size=256, n_layer=6, n_head=6, n_embd=384, batch_size=64, lr=3e-4, iters=5000, dropout=0.2),
}

parser = argparse.ArgumentParser()
parser.add_argument("--size", choices=CONFIGS, default=None)
parser.add_argument("--iters", type=int, default=None, help="覆盖默认训练步数")
parser.add_argument("--data", default=str(ROOT / "hlm_raw.txt"))
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = CONFIGS[args.size or ("medium" if device == "cuda" else "small")]
if args.iters:
    cfg["iters"] = args.iters
torch.manual_seed(1337)
print(f"设备: {device}   配置: {cfg}")

# ================================================================
# ① 数据：把文字变成数字
# ================================================================
text = Path(args.data).read_text(encoding="utf-8")
start, end = text.find("*** START OF"), text.find("*** END OF")  # 去掉古腾堡计划的英文版权声明
if start != -1 and end != -1:
    text = text[text.find("\n", start) + 1:end]

chars = sorted(set(text))                     # 书里出现过的所有不同的字，就是"字典"
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # 字 -> 编号
itos = {i: ch for i, ch in enumerate(chars)}  # 编号 -> 字
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]     # 前 90% 用来学，后 10% 用来考试
print(f"全书 {len(text):,} 字，字典里有 {vocab_size} 个不同的字")


def get_batch(split):
    """随机截取几段文字。x 是输入，y 是"每个位置的下一个字"（答案）。"""
    d = train_data if split == "train" else val_data
    T, B = cfg["block_size"], cfg["batch_size"]
    ix = torch.randint(len(d) - T - 1, (B,))
    x = torch.stack([d[i:i + T] for i in ix])
    y = torch.stack([d[i + 1:i + T + 1] for i in ix])  # 比 x 往后错一个字
    return x.to(device), y.to(device)


# ================================================================
# ② 模型：Transformer
# ================================================================
# 模型代码在 model.py 里：注意力 CausalSelfAttention、Transformer 模块 Block、整个 GPT


def write(model, prompt, n):
    model.eval()
    idx = torch.tensor([encode(prompt)], device=device)
    out = decode(model.generate(idx, n)[0].tolist())
    model.train()
    return out


# ================================================================
# ③ 损失函数（在 GPT.forward 里：cross_entropy）  ④ 优化器
# ================================================================
model = GPT(cfg, vocab_size).to(device)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = torch.stack([model(*get_batch(split))[1] for _ in range(30)]).mean().item()
    model.train()
    return out


# ================================================================
# ⑤ 训练循环
# ================================================================
PROMPT = "寶玉"
eval_every = max(cfg["iters"] // 8, 1)
t0 = time.time()
for it in range(cfg["iters"] + 1):
    if it % eval_every == 0:
        losses = estimate_loss()
        print(f"\n第 {it:>5} 步  训练损失 {losses['train']:.3f}  考试损失 {losses['val']:.3f}  "
              f"已用 {time.time() - t0:.0f}s")
        print("  试写:", write(model, PROMPT, 40).replace("\n", " "))
    if it == cfg["iters"]:
        break
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)         # 前向：猜下一个字，算猜错了多少
    optimizer.zero_grad()           # 清空上一步的梯度
    loss.backward()                 # 反向传播
    optimizer.step()                # 更新参数

# ================================================================
# ⑥ 保存，并写一段长的
# ================================================================
torch.save({"model": model.state_dict(), "cfg": cfg, "chars": chars}, ROOT / f"hlm_{device}.pt")
final = write(model, PROMPT, 300)
(ROOT / f"作品_{device}.txt").write_text(final, encoding="utf-8")
print("\n================ 最终作品 ================")
print(final)
