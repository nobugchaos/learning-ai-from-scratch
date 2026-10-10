"""监督微调（SFT）演示：把"只会写红楼梦"的模型，教成"会答物理计算题"的模型。

前提：已经运行过 nanogpt.py，得到了预训练模型 hlm_cpu.pt（或 hlm_cuda.pt）。

用法：
    python sft_demo.py
运行后打开 sft_report.html 看微调前后的对比。
"""
import html
import json
import random
import re
import time
from pathlib import Path

import torch

from model import GPT

ROOT = Path(__file__).parent
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)

# ================================================================
# ① 准备监督微调数据：一问一答，答案是"标准解答"
# ================================================================
def make_qa():
    """生成全部题目。每道题 = (题型, 问题, 标准答案, 正确数值)。"""
    qa = []
    for m in range(1, 10):
        for a in range(1, 10):
            F = m * a
            qa.append(("牛頓第二定律", f"問：質量{m}kg的物體受合力{F}N，加速度多大？\n答：", f"a=F/m={F}/{m}={a}m/s²。", a))
            qa.append(("動量", f"問：質量{m}kg的物體以{a}m/s運動，動量多大？\n答：", f"p=mv={m}×{a}={F}kg·m/s。", F))
    for t in range(1, 10):
        qa.append(("自由落體", f"問：物體從靜止自由下落{t}s，速度多大？\n答：", f"v=gt=10×{t}={10 * t}m/s。", 10 * t))
        qa.append(("重力", f"問：質量{t}kg的物體重力多大？\n答：", f"G=mg={t}×10={10 * t}N。", 10 * t))
    return qa


all_qa = make_qa()
random.shuffle(all_qa)
test_qa = all_qa[:30]      # 留出 30 道题，微调时完全不给模型看，专门用来考试
train_qa = all_qa[30:]
print(f"题目总数 {len(all_qa)}：训练 {len(train_qa)} 道，考试 {len(test_qa)} 道（没见过的新题）")

# ================================================================
# ② 加载"读过红楼梦"的预训练模型，并给字典添上物理题里的新字符
# ================================================================
ckpt_path = ROOT / ("hlm_cuda.pt" if (ROOT / "hlm_cuda.pt").exists() else "hlm_cpu.pt")
ckpt = torch.load(ckpt_path, map_location=device)
cfg, chars = ckpt["cfg"], list(ckpt["chars"])

new_chars = sorted({c for _, q, a, _ in all_qa for c in q + a} - set(chars))
chars += new_chars                       # 红楼梦里没有 kg、N、=、² 这些字符，加进字典
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
print(f"字典新增 {len(new_chars)} 个字符: {''.join(new_chars)}")

model = GPT(cfg, len(chars)).to(device)
state = ckpt["model"]
old_n = state["tok_emb.weight"].shape[0]
with torch.no_grad():
    for name in ("tok_emb.weight", "head.weight", "head.bias"):
        param = dict(model.named_parameters())[name]
        param[:old_n] = state[name]       # 原有的字：沿用预训练学到的参数
        param[old_n:] = 0 if name == "head.bias" else state[name].mean(0) + 0.02 * torch.randn_like(param[old_n:])
    model.load_state_dict({k: v for k, v in state.items() if k not in ("tok_emb.weight", "head.weight", "head.bias")},
                          strict=False)
END = stoi["。"]


def answer(question, temperature=0.01):
    """让模型回答一道题（温度很低 = 每次都选最可能的字）。"""
    model.eval()
    idx = torch.tensor([encode(question)], device=device)
    out = model.generate(idx, 30, temperature=temperature, stop_id=END)
    return decode(out[0, idx.size(1):].tolist())


def continue_hlm(n=60):
    model.eval()
    torch.manual_seed(1)
    idx = torch.tensor([encode("寶玉")], device=device)
    return decode(model.generate(idx, n)[0].tolist())


def evaluate(qa):
    """格式对：答案符合"公式=代入=结果+单位"的样子；答对：最后的数值正确。"""
    rows, fmt_ok, num_ok = [], 0, 0
    for kind, q, std, value in qa:
        out = answer(q)
        fmt = bool(re.fullmatch(r"[a-zA-Z]=[^=]+=[^=]+=\d+[^\d=]+。", out))
        m = re.search(r"=(\d+)[^\d=]*。?$", out)
        right = fmt and m is not None and int(m.group(1)) == value
        fmt_ok += fmt
        num_ok += right
        rows.append({"kind": kind, "q": q.split("\n")[0], "std": std, "out": out, "fmt": fmt, "right": right})
    return rows, fmt_ok / len(qa), num_ok / len(qa)


# ================================================================
# ③ 微调前：考一次
# ================================================================
before_rows, before_fmt, before_acc = evaluate(test_qa)
before_hlm = continue_hlm()
print(f"微调前：格式正确 {before_fmt:.0%}，答案正确 {before_acc:.0%}")

# ================================================================
# ④ 监督微调：和预训练一样"预测下一个字"，但只在【答案】部分算损失
# ================================================================
T = cfg["block_size"]


def make_example(q, a):
    ids = encode(q + a)
    x = ids[:-1] + [END] * (T - len(ids) + 1)
    y = ids[1:] + [-100] * (T - len(ids) + 1)  # 补齐的位置不算损失
    for i in range(len(encode(q)) - 1):
        y[i] = -100                             # 问题部分不算损失：只考模型会不会写答案
    return x, y


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
STEPS, BATCH = 400, 32
losses = []
t0 = time.time()
model.train()
for step in range(STEPS):
    batch = [make_example(q, a) for _, q, a, _ in random.choices(train_qa, k=BATCH)]
    x = torch.tensor([b[0] for b in batch], device=device)
    y = torch.tensor([b[1] for b in batch], device=device)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(round(loss.item(), 4))
    if step % 50 == 0:
        print(f"  第 {step:>3} 步  损失 {loss.item():.3f}")
print(f"微调用时 {time.time() - t0:.0f}s")

# ================================================================
# ⑤ 微调后：用同样的考题再考一次，并对比
# ================================================================
after_rows, after_fmt, after_acc = evaluate(test_qa)
_, train_fmt, train_acc = evaluate(random.sample(train_qa, 30))
after_hlm = continue_hlm()
print(f"微调后：格式正确 {after_fmt:.0%}，答案正确 {after_acc:.0%}（训练过的题：{train_acc:.0%}）")

result = dict(before_fmt=before_fmt, before_acc=before_acc, after_fmt=after_fmt, after_acc=after_acc,
              train_fmt=train_fmt, train_acc=train_acc, before_hlm=before_hlm, after_hlm=after_hlm,
              losses=losses, new_chars="".join(new_chars), n_train=len(train_qa), n_test=len(test_qa),
              rows=[dict(b, after=a["out"], after_fmt=a["fmt"], after_right=a["right"])
                    for b, a in zip(before_rows, after_rows)])
(ROOT / "sft_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

# ================================================================
# ⑥ 生成对比网页 sft_report.html
# ================================================================
e = html.escape
mark = lambda ok: '<span class="ok">✓</span>' if ok else '<span class="no">✗</span>'
pts = " ".join(f"{i * 600 / (len(losses) - 1):.1f},{150 - min(l, 6) / 6 * 140:.1f}" for i, l in enumerate(losses))
rows_html = "".join(
    f"<tr><td>{e(r['kind'])}</td><td>{e(r['q'])}</td><td class='mono'>{e(r['std'])}</td>"
    f"<td class='mono'>{mark(r['right'])} {e(r['out'] or '（空）')}</td>"
    f"<td class='mono'>{mark(r['after_right'])} {e(r['after'])}</td></tr>"
    for r in result["rows"])
page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>监督微调演示</title><style>
:root{{--bg:#faf9f6;--card:#fff;--ink:#1d1d1b;--muted:#6b6a64;--line:#e4e2da;--accent:#185fa5;--good:#1b7f3a;--bad:#c0392b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171716;--card:#20201e;--ink:#eeede8;--muted:#a3a29a;--line:#34342f;--accent:#5b9ee6;--good:#4cc070;--bad:#ec6a5e}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:24px 16px 60px}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px}}
.sub{{color:var(--muted);margin:0 0 16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.stat b{{display:block;font-size:30px}} .stat span{{color:var(--muted);font-size:13px}}
.arrow{{color:var(--muted);font-size:20px;margin:0 6px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:var(--muted);font-weight:600}} .mono{{font-family:Consolas,monospace}} .ok{{color:var(--good);font-weight:700}} .no{{color:var(--bad);font-weight:700}}
.wrap{{overflow-x:auto}} .hlm{{font-family:inherit;white-space:pre-wrap;background:var(--bg);padding:10px;border-radius:8px;margin:6px 0}}
svg{{width:100%;height:auto}}
</style></head><body><main>
<h1>监督微调（SFT）演示</h1>
<p class="sub">把一个只读过《红楼梦》的小模型，用 {result['n_train']} 道带标准解答的物理计算题微调 {STEPS} 步，再用 {result['n_test']} 道它没见过的新题考试。</p>
<div class="stats">
<div class="card stat"><span>格式正确（新题）</span><b>{before_fmt:.0%}<span class="arrow">→</span>{after_fmt:.0%}</b><span>"公式=代入=结果+单位"</span></div>
<div class="card stat"><span>答案正确（新题）</span><b>{before_acc:.0%}<span class="arrow">→</span>{after_acc:.0%}</b><span>没见过的数字组合</span></div>
<div class="card stat"><span>答案正确（训练过的题）</span><b>{train_acc:.0%}</b><span>见过的题</span></div>
</div>
<h2>微调过程的损失（只算答案部分）</h2>
<div class="card"><svg viewBox="0 0 600 160"><polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
<line x1="0" y1="150" x2="600" y2="150" stroke="var(--line)"/></svg>
<span class="sub">从 {losses[0]:.2f} 降到 {losses[-1]:.3f}</span></div>
<h2>考试题逐题对比（都是微调时没见过的题）</h2>
<div class="card wrap"><table><tr><th>题型</th><th>问题</th><th>标准答案</th><th>微调前</th><th>微调后</th></tr>{rows_html}</table></div>
<h2>副作用：它还记得《红楼梦》吗？（同样以"寶玉"开头续写）</h2>
<div class="card"><b>微调前</b><div class="hlm">{e(before_hlm)}</div><b>微调后</b><div class="hlm">{e(after_hlm)}</div></div>
</main></body></html>"""
(ROOT / "sft_report.html").write_text(page, encoding="utf-8")
print(f"对比网页已生成: {ROOT / 'sft_report.html'}")
