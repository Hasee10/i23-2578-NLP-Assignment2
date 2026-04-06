import importlib.util
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")


def load_mod():
    sp = importlib.util.spec_from_file_location("jt", "J-transformer.py")
    md = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(md)
    return md


def load_idx():
    with open("embeddings/word2idx.json", encoding="utf-8") as f:
        return json.load(f)


class ClsSet(Dataset):
    def __init__(self, xp, yp, ip):
        self.x = np.load(xp)
        self.y = np.load(yp)
        self.i = np.load(ip)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, j):
        return (
            torch.tensor(self.x[j], dtype=torch.long),
            torch.tensor(int(self.y[j]), dtype=torch.long),
            torch.tensor(int(self.i[j]), dtype=torch.long),
        )


def run(md, dl, dev, opt=None, sch=None):
    tr = opt is not None
    md.train() if tr else md.eval()
    ls = []
    ys = []
    ps = []
    ids = []
    xs = []
    aws = []
    fn = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(tr):
        for x, y, i in dl:
            x = x.to(dev)
            y = y.to(dev)
            mk = x.ne(0)
            lg, ws = md(x, mk)
            loss = fn(lg, y)
            if tr:
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
            ls.append(float(loss.item()))
            pr = lg.argmax(dim=1)
            ys.extend(y.cpu().tolist())
            ps.extend(pr.cpu().tolist())
            ids.extend(i.cpu().tolist())
            xs.extend(x.cpu().tolist())
            aws.extend([w.cpu() for w in ws[-1]])
    av = sum(ls) / max(1, len(ls))
    ac = float(sum(int(a == b) for a, b in zip(ys, ps)) / max(1, len(ys)))
    f1 = f1_score(ys, ps, average="macro", zero_division=0)
    return av, ac, f1, ys, ps, ids, xs, aws


def lr_fn(st, wm, tt):
    if st < wm:
        return float(st + 1) / float(max(1, wm))
    x = (st - wm) / float(max(1, tt - wm))
    return 0.5 * (1.0 + math.cos(math.pi * x))


def draw_line(fp, tt, tv, yl, title):
    xs = np.arange(1, len(tt) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, tt, marker="o", label="train")
    plt.plot(xs, tv, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(yl)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fp, dpi=200)
    plt.close()


def draw_cm(fp, cm, labs):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labs)), labs, rotation=45, ha="right")
    plt.yticks(range(len(labs)), labs)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Transformer Test Confusion Matrix")
    for i in range(len(labs)):
        for j in range(len(labs)):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fp, dpi=200)
    plt.close()


def draw_heat(fp, ws, toks, head_ids):
    n = min(len(toks), 24)
    labs = toks[:n]
    fig, ax = plt.subplots(1, len(head_ids), figsize=(7 * len(head_ids), 6))
    if len(head_ids) == 1:
        ax = [ax]
    for a, h in zip(ax, head_ids):
        hm = ws[h, :n, :n].numpy()
        a.imshow(hm, cmap="viridis")
        a.set_xticks(range(n))
        a.set_yticks(range(n))
        a.set_xticklabels(labs, rotation=90, fontsize=8)
        a.set_yticklabels(labs, fontsize=8)
        a.set_title(f"Head {h}")
    fig.suptitle("Final Encoder Attention Heatmap")
    fig.tight_layout()
    fig.savefig(fp, dpi=200)
    plt.close(fig)


def main():
    torch.manual_seed(7)
    np.random.seed(7)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    Path("models").mkdir(exist_ok=True)
    jt = load_mod()
    idx = load_idx()
    rev = {v: k for k, v in idx.items()}
    labs = ["politics", "sports", "economy", "international", "health"]
    tr = DataLoader(ClsSet("data/cls_train_x.npy", "data/cls_train_y.npy", "data/cls_train_id.npy"), batch_size=16, shuffle=True)
    va = DataLoader(ClsSet("data/cls_val_x.npy", "data/cls_val_y.npy", "data/cls_val_id.npy"), batch_size=32, shuffle=False)
    te = DataLoader(ClsSet("data/cls_test_x.npy", "data/cls_test_y.npy", "data/cls_test_id.npy"), batch_size=32, shuffle=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = jt.TransformerCls(nv=len(idx), dm=128, h=4, dk=32, dv=32, df=512, nl=4, nc=5, mx=257, dr=0.1).to(dev)
    opt = AdamW(md.parameters(), lr=5e-4, weight_decay=0.01)
    epn = 20
    tot = epn * len(tr)
    wm = 50
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: lr_fn(s, wm, tot))
    hs = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    for ep in range(1, epn + 1):
        tl, ta, tf, *_ = run(md, tr, dev, opt=opt, sch=sch)
        vl, vaa, vf, *_ = run(md, va, dev)
        hs["tr_loss"].append(tl)
        hs["va_loss"].append(vl)
        hs["tr_acc"].append(ta)
        hs["va_acc"].append(vaa)
        print(f"ep {ep} train_loss {tl:.4f} train_acc {ta:.4f} val_loss {vl:.4f} val_acc {vaa:.4f} val_f1 {vf:.4f}")
    draw_line("models/transformer_loss.png", hs["tr_loss"], hs["va_loss"], "Loss", "Transformer Loss")
    draw_line("models/transformer_acc.png", hs["tr_acc"], hs["va_acc"], "Accuracy", "Transformer Accuracy")
    tl, ta, tf, ys, ps, ids, xs, aws = run(md, te, dev)
    print(f"test accuracy {ta:.4f}")
    print(f"test macro_f1 {tf:.4f}")
    cm = confusion_matrix(ys, ps, labels=list(range(5)))
    print("test confusion matrix:")
    print(cm)
    draw_cm("models/transformer_confusion_matrix.png", cm, labs)
    hit = [i for i, (a, b) in enumerate(zip(ys, ps)) if a == b][:3]
    for j, k in enumerate(hit, 1):
        tok_ids = [t for t in xs[k] if t != 0][:23]
        toks = ["[CLS]"] + [rev.get(int(t), "<UNK>") for t in tok_ids]
        ws = aws[k]
        draw_heat(f"models/transformer_heatmap_{j}.png", ws, toks, [0, 1])
        print(f"heatmap article {j} id {ids[k]} label {labs[ys[k]]}")
    torch.save({
        "state_dict": md.state_dict(),
        "word2idx": idx,
        "labels": labs,
        "config": {"nv": len(idx), "dm": 128, "h": 4, "dk": 32, "dv": 32, "df": 512, "nl": 4, "nc": 5, "mx": 257, "dr": 0.1},
    }, "models/transformer_cls.pt")


if __name__ == "__main__":
    main()
