"""GPT 模型本体（nanogpt.py 和 sft_demo.py 共用）。"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """注意力：每个字回头看前面所有的字，决定该关注谁。（不能偷看后面的字）"""

    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.qkv = nn.Linear(C, 3 * C)   # 一次算出 Query、Key、Value
        self.proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        T = cfg["block_size"]
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))  # 下三角：只能看自己和前面

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # 拆成多个"头"，每个头独立地关注不同的东西
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))       # 每对字之间的"相关程度"
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # 遮住后面的字
        att = self.drop(F.softmax(att, dim=-1))                      # 变成加起来为 1 的权重
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)     # 按权重把信息汇总过来
        return self.proj(y)


class Block(nn.Module):
    """一个 Transformer 模块 = 注意力 + MLP，各自带残差连接和归一化。"""

    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C), nn.Dropout(cfg["dropout"]))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 字和字之间交流信息
        x = x + self.mlp(self.ln2(x))   # 每个字自己"想一想"
        return x


class GPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        C, T = cfg["n_embd"], cfg["block_size"]
        self.tok_emb = nn.Embedding(vocab_size, C)  # 每个字 -> 一串数字（字的"含义"）
        self.pos_emb = nn.Embedding(T, C)           # 每个位置 -> 一串数字（字的"位置"）
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(C)
        self.head = nn.Linear(C, vocab_size)        # 最后给字典里每个字打分

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))            # (B, T, 字典大小)：每个位置对"下一个字"的打分
        loss = None
        if targets is not None:
            # ignore_index=-100：标成 -100 的位置不算损失（监督微调时用来跳过"问题"部分）
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, stop_id=None):
        """写作：预测下一个字 -> 按概率抽一个 -> 接到后面 -> 再预测，循环。"""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg["block_size"]:])  # 只看最近 block_size 个字
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)    # 按概率"掷骰子"
            idx = torch.cat([idx, next_id], dim=1)
            if stop_id is not None and next_id.item() == stop_id:
                break
        return idx
