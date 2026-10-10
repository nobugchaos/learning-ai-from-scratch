"""用训练好的模型识别手写数字。

用法：
    python predict.py               # 打开画板，用鼠标写数字，松开后自动识别
    python predict.py 图片.png ...   # 识别图片文件（白底黑字、黑底白字都可以）
"""
import sys
import tkinter as tk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from train import MODEL_PATH, Net

CANVAS = 280   # 画板边长（像素），是 MNIST 28x28 的 10 倍
BRUSH = 22     # 笔刷粗细


def load_model():
    if not MODEL_PATH.exists():
        sys.exit(f"找不到模型 {MODEL_PATH}，请先运行 python train.py")
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def to_mnist(img):
    """把黑底白字的灰度图处理成和 MNIST 一样的格式：
    数字缩放到 20x20 以内，再按重心放到 28x28 的中央。返回 None 表示图是空的。"""
    bbox = img.point(lambda p: 255 if p > 30 else 0).getbbox()
    if bbox is None:
        return None
    digit = img.crop(bbox)
    w, h = digit.size
    scale = 20 / max(w, h)
    digit = digit.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    arr = np.asarray(digit, dtype=np.float32)
    ys, xs = np.indices(arr.shape)
    total = arr.sum()
    cy, cx = (ys * arr).sum() / total, (xs * arr).sum() / total

    out = Image.new("L", (28, 28), 0)
    left = int(round(14 - cx))
    top = int(round(14 - cy))
    left = min(max(left, 0), 28 - digit.width)
    top = min(max(top, 0), 28 - digit.height)
    out.paste(digit, (left, top))
    return out


@torch.no_grad()
def predict(model, img28):
    x = torch.from_numpy(np.asarray(img28, dtype=np.float32) / 255.0)
    x = ((x - 0.1307) / 0.3081).view(1, 1, 28, 28)
    return torch.softmax(model(x), dim=1)[0].numpy()


def predict_files(model, paths):
    for path in paths:
        img = Image.open(path).convert("L")
        if np.asarray(img).mean() > 127:  # 白底黑字 -> 反色成黑底白字
            img = ImageOps.invert(img)
        img28 = to_mnist(img)
        if img28 is None:
            print(f"{path}: 图片是空的")
            continue
        probs = predict(model, img28)
        d = int(probs.argmax())
        print(f"{path}: 识别为 {d}（把握 {probs[d]:.1%}）")


class DrawPad:
    BAR_W, BAR_H = 260, 22

    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("手写数字识别")
        self.root.resizable(False, False)
        font = ("Microsoft YaHei", 11)

        left = tk.Frame(self.root, padx=12, pady=12)
        left.pack(side="left")
        tk.Label(left, text="在黑板上用鼠标写一个数字（0-9）", font=font).pack(anchor="w")
        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="black",
                                highlightthickness=1, cursor="pencil")
        self.canvas.pack(pady=6)
        btns = tk.Frame(left)
        btns.pack(fill="x")
        tk.Button(btns, text="清空", font=font, width=8, command=self.clear).pack(side="left")
        tk.Label(btns, text="右键也可以清空", font=("Microsoft YaHei", 9), fg="gray").pack(side="right")

        right = tk.Frame(self.root, padx=12, pady=12)
        right.pack(side="left", fill="y")
        self.result = tk.Label(right, text="?", font=("Microsoft YaHei", 48, "bold"), width=3)
        self.result.pack()
        self.conf = tk.Label(right, text="写完松开鼠标就会识别", font=font, fg="gray")
        self.conf.pack(pady=(0, 8))
        self.bars = tk.Canvas(right, width=self.BAR_W + 60, height=self.BAR_H * 10,
                              highlightthickness=0)
        self.bars.pack()
        self.preview = tk.Label(right, text="模型看到的 28×28 图像：", font=("Microsoft YaHei", 9), fg="gray")
        self.preview.pack(anchor="w", pady=(10, 2))
        self.preview_img = tk.Label(right)
        self.preview_img.pack(anchor="w")

        # 同步画在一张 PIL 图上，识别时直接用它
        self.image = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last = None

        self.canvas.bind("<Button-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Button-3>", lambda e: self.clear())
        self.draw_bars(np.zeros(10))

    def on_down(self, e):
        self.last = (e.x, e.y)
        self.stroke(e.x, e.y, e.x, e.y)

    def on_move(self, e):
        if self.last:
            self.stroke(*self.last, e.x, e.y)
        self.last = (e.x, e.y)

    def on_up(self, e):
        self.last = None
        self.recognize()

    def stroke(self, x0, y0, x1, y1):
        r = BRUSH / 2
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=BRUSH,
                                capstyle="round", smooth=True)
        self.draw.line((x0, y0, x1, y1), fill=255, width=BRUSH)
        self.draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS, CANVAS), fill=0)
        self.result.config(text="?")
        self.conf.config(text="写完松开鼠标就会识别")
        self.preview_img.config(image="")
        self.draw_bars(np.zeros(10))

    def recognize(self):
        img28 = to_mnist(self.image)
        if img28 is None:
            return
        probs = predict(self.model, img28)
        d = int(probs.argmax())
        self.result.config(text=str(d))
        self.conf.config(text=f"把握 {probs[d]:.1%}")
        self.draw_bars(probs)
        self._preview = tk.PhotoImage(data=self._png(img28.resize((84, 84), Image.NEAREST)))
        self.preview_img.config(image=self._preview)

    @staticmethod
    def _png(img):
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def draw_bars(self, probs):
        c = self.bars
        c.delete("all")
        best = int(probs.argmax()) if probs.any() else -1
        for i, p in enumerate(probs):
            y = i * self.BAR_H
            c.create_text(12, y + self.BAR_H / 2, text=str(i), font=("Consolas", 11))
            c.create_rectangle(26, y + 4, 26 + self.BAR_W, y + self.BAR_H - 4,
                               fill="#eeeeee", outline="")
            if p > 0:
                c.create_rectangle(26, y + 4, 26 + self.BAR_W * p, y + self.BAR_H - 4,
                                   fill="#2e7d32" if i == best else "#90a4ae", outline="")
            c.create_text(26 + self.BAR_W + 6, y + self.BAR_H / 2, anchor="w",
                          text=f"{p:.0%}", font=("Consolas", 9))

    def run(self):
        self.root.mainloop()


def main():
    model = load_model()
    if len(sys.argv) > 1:
        predict_files(model, sys.argv[1:])
    else:
        DrawPad(model).run()


if __name__ == "__main__":
    main()
