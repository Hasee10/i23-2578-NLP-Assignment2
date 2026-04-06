import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


def read_conll(fp):
    xs = []
    ws = []
    ys = []
    for ln in Path(fp).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            a, b = ln.split("\t")
            ws.append(a)
            ys.append(b)
        elif ws:
            xs.append((ws, ys))
            ws = []
            ys = []
    if ws:
        xs.append((ws, ys))
    return xs


def build_lab(xs):
    ys = sorted({y for _, zs in xs for y in zs})
    idx = {y: i for i, y in enumerate(ys)}
    return ys, idx


def load_idx():
    with open("embeddings/word2idx.json", encoding="utf-8") as f:
        return json.load(f)


class SeqSet(Dataset):
    def __init__(self, xs, w2i, y2i):
        self.xs = xs
        self.w2i = w2i
        self.y2i = y2i

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        ws, ys = self.xs[i]
        x = [self.w2i[w] if w in self.w2i else 0 for w in ws]
        y = [self.y2i[z] for z in ys]
        return torch.tensor(x), torch.tensor(y), len(x)


def collate(bs):
    bs = sorted(bs, key=lambda x: x[2], reverse=True)
    ls = [x[2] for x in bs]
    m = max(ls)
    xs = torch.zeros(len(bs), m, dtype=torch.long)
    ys = torch.full((len(bs), m), -100, dtype=torch.long)
    mk = torch.zeros(len(bs), m, dtype=torch.bool)
    for i, (x, y, l) in enumerate(bs):
        xs[i, :l] = x
        ys[i, :l] = y
        mk[i, :l] = True
    return xs, ys, torch.tensor(ls), mk


class Enc(nn.Module):
    def __init__(self, emb, h, nlab, fr):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=fr)
        self.lstm = nn.LSTM(emb.shape[1], h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(h * 2, nlab)

    def feat(self, x, ls):
        z = self.emb(x)
        pk = pack_padded_sequence(z, ls.cpu(), batch_first=True, enforce_sorted=True)
        ot, _ = self.lstm(pk)
        ot, _ = pad_packed_sequence(ot, batch_first=True)
        return self.fc(self.drop(ot))


class Crf(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.st = n
        self.ed = n + 1
        self.tr = nn.Parameter(torch.empty(n + 2, n + 2))
        nn.init.uniform_(self.tr, -0.1, 0.1)

    def score(self, em, tg, mk):
        b, t, _ = em.shape
        sc = em.new_zeros(b)
        st = torch.full((b, 1), self.st, dtype=torch.long, device=em.device)
        tg2 = torch.cat([st, tg], dim=1)
        for i in range(t):
            on = mk[:, i]
            a = tg2[:, i]
            c = tg[:, i]
            sc = sc + (self.tr[a, c] + em[:, i].gather(1, c.unsqueeze(1)).squeeze(1)) * on
        le = mk.long().sum(1)
        ls = tg.gather(1, (le - 1).unsqueeze(1)).squeeze(1)
        sc = sc + self.tr[ls, self.ed]
        return sc

    def norm(self, em, mk):
        b, t, n = em.shape
        al = em.new_full((b, n), -1e4)
        al[:, :] = self.tr[self.st, :n]
        al = al + em[:, 0]
        for i in range(1, t):
            ex = al.unsqueeze(2) + self.tr[:n, :n].unsqueeze(0) + em[:, i].unsqueeze(1)
            nw = torch.logsumexp(ex, dim=1)
            on = mk[:, i].unsqueeze(1)
            al = torch.where(on, nw, al)
        al = al + self.tr[:n, self.ed].unsqueeze(0)
        return torch.logsumexp(al, dim=1)

    def loss(self, em, tg, mk):
        return (self.norm(em, mk) - self.score(em, tg, mk)).mean()

    def decode(self, em, mk):
        b, t, n = em.shape
        bt = []
        dp = self.tr[self.st, :n].unsqueeze(0) + em[:, 0]
        bt.append(torch.zeros((b, n), dtype=torch.long, device=em.device))
        for i in range(1, t):
            ex = dp.unsqueeze(2) + self.tr[:n, :n].unsqueeze(0)
            bv, bi = ex.max(dim=1)
            nw = bv + em[:, i]
            on = mk[:, i].unsqueeze(1)
            dp = torch.where(on, nw, dp)
            bt.append(bi)
        dp = dp + self.tr[:n, self.ed].unsqueeze(0)
        ls = mk.long().sum(1)
        out = []
        for j in range(b):
            l = int(ls[j].item())
            la = int(dp[j].argmax().item())
            xs = [la]
            for i in range(l - 1, 0, -1):
                la = int(bt[i][j, la].item())
                xs.append(la)
            out.append(list(reversed(xs)))
        return out


class Pos(nn.Module):
    def __init__(self, emb, h, nlab, fr):
        super().__init__()
        self.enc = Enc(emb, h, nlab, fr)

    def loss(self, x, y, ls, mk):
        em = self.enc.feat(x, ls)
        fn = nn.CrossEntropyLoss(ignore_index=-100)
        return fn(em.view(-1, em.shape[-1]), y.view(-1))

    def pred(self, x, ls, mk):
        em = self.enc.feat(x, ls)
        return em.argmax(-1)


class Ner(nn.Module):
    def __init__(self, emb, h, nlab, fr):
        super().__init__()
        self.enc = Enc(emb, h, nlab, fr)
        self.crf = Crf(nlab)

    def loss(self, x, y, ls, mk):
        em = self.enc.feat(x, ls)
        y2 = y.masked_fill(~mk, 0)
        return self.crf.loss(em, y2, mk)

    def pred(self, x, ls, mk):
        em = self.enc.feat(x, ls)
        return self.crf.decode(em, mk)


def run_eval(md, dl, i2y, dev, ner=False):
    md.eval()
    ls = []
    ys = []
    ps = []
    with torch.no_grad():
        for x, y, ln, mk in dl:
            x = x.to(dev)
            y = y.to(dev)
            mk = mk.to(dev)
            loss = md.loss(x, y, ln, mk)
            ls.append(float(loss.item()))
            pr = md.pred(x, ln, mk)
            if ner:
                for i in range(len(pr)):
                    l = int(ln[i].item())
                    ys.extend(y[i, :l].cpu().tolist())
                    ps.extend(pr[i])
            else:
                for i in range(x.shape[0]):
                    l = int(ln[i].item())
                    ys.extend(y[i, :l].cpu().tolist())
                    ps.extend(pr[i, :l].cpu().tolist())
    avg = sum(ls) / max(1, len(ls))
    lbs = list(range(len(i2y)))
    if ner and "O" in i2y:
        lbs = [i for i, x in enumerate(i2y) if x != "O"]
    f1 = f1_score(ys, ps, labels=lbs, average="macro", zero_division=0)
    return avg, f1


def fit(name, cls, tr, va, emb, y2i, i2y, fr, dev):
    md = cls(emb, 128, len(y2i), fr).to(dev)
    opt = torch.optim.Adam(md.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1.0
    bvl = 1e9
    bad = 0
    hist = {"tr": [], "va": [], "f1": []}
    bst = None
    for ep in range(1, 41):
        md.train()
        los = []
        for x, y, ln, mk in tr:
            x = x.to(dev)
            y = y.to(dev)
            mk = mk.to(dev)
            opt.zero_grad()
            loss = md.loss(x, y, ln, mk)
            loss.backward()
            opt.step()
            los.append(float(loss.item()))
        trl = sum(los) / max(1, len(los))
        val, f1 = run_eval(md, va, i2y, dev, ner=(name == "ner"))
        hist["tr"].append(trl)
        hist["va"].append(val)
        hist["f1"].append(f1)
        mode = "frozen" if fr else "finetuned"
        print(f"{name} {mode} ep {ep} train_loss {trl:.4f} val_loss {val:.4f} val_f1 {f1:.4f}")
        if f1 > best or (abs(f1 - best) < 1e-8 and val < bvl):
            best = f1
            bvl = val
            bad = 0
            bst = {k: v.detach().cpu() for k, v in md.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    return best, hist, bst


def draw(fp, hs):
    plt.figure(figsize=(9, 5))
    for lb, h in hs.items():
        xs = np.arange(1, len(h["tr"]) + 1)
        plt.plot(xs, h["tr"], label=f"{lb} train")
        plt.plot(xs, h["va"], label=f"{lb} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(Path(fp).stem.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(fp, dpi=200)
    plt.close()


def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    Path("models").mkdir(exist_ok=True)
    w2i = load_idx()
    emb = np.load("embeddings/embeddings_w2v.npy").astype(np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_tr = read_conll("data/pos_train.conll")
    pos_va = read_conll("data/pos_val.conll")
    ner_tr = read_conll("data/ner_train.conll")
    ner_va = read_conll("data/ner_val.conll")
    pos_i2y, pos_y2i = build_lab(pos_tr + pos_va)
    ner_i2y, ner_y2i = build_lab(ner_tr + ner_va)
    pos_tr_dl = DataLoader(SeqSet(pos_tr, w2i, pos_y2i), batch_size=32, shuffle=True, collate_fn=collate)
    pos_va_dl = DataLoader(SeqSet(pos_va, w2i, pos_y2i), batch_size=64, shuffle=False, collate_fn=collate)
    ner_tr_dl = DataLoader(SeqSet(ner_tr, w2i, ner_y2i), batch_size=32, shuffle=True, collate_fn=collate)
    ner_va_dl = DataLoader(SeqSet(ner_va, w2i, ner_y2i), batch_size=64, shuffle=False, collate_fn=collate)
    pos_hs = {}
    ner_hs = {}
    pos_best = (-1.0, None, None)
    ner_best = (-1.0, None, None)
    for fr in [True, False]:
        f1, hs, st = fit("pos", Pos, pos_tr_dl, pos_va_dl, emb, pos_y2i, pos_i2y, fr, dev)
        pos_hs["frozen" if fr else "finetuned"] = hs
        if f1 > pos_best[0]:
            pos_best = (f1, fr, st)
        f1, hs, st = fit("ner", Ner, ner_tr_dl, ner_va_dl, emb, ner_y2i, ner_i2y, fr, dev)
        ner_hs["frozen" if fr else "finetuned"] = hs
        if f1 > ner_best[0]:
            ner_best = (f1, fr, st)
    torch.save({
        "state_dict": pos_best[2],
        "label2idx": pos_y2i,
        "idx2label": pos_i2y,
        "word2idx": w2i,
        "frozen": pos_best[1],
    }, "models/bilstm_pos.pt")
    torch.save({
        "state_dict": ner_best[2],
        "label2idx": ner_y2i,
        "idx2label": ner_i2y,
        "word2idx": w2i,
        "frozen": ner_best[1],
    }, "models/bilstm_ner.pt")
    draw("models/pos_loss_curve.png", pos_hs)
    draw("models/ner_loss_curve.png", ner_hs)
    print(f"best pos f1 {pos_best[0]:.4f} mode {'frozen' if pos_best[1] else 'finetuned'}")
    print(f"best ner f1 {ner_best[0]:.4f} mode {'frozen' if ner_best[1] else 'finetuned'}")


if __name__ == "__main__":
    main()
