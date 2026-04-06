import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


def say(x):
    sys.stdout.buffer.write((str(x) + "\n").encode("utf-8", errors="backslashreplace"))


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
        return torch.tensor(x), torch.tensor(y), len(x), ws, ys


def collate(bs):
    bs = sorted(bs, key=lambda x: x[2], reverse=True)
    ls = [x[2] for x in bs]
    m = max(ls)
    xs = torch.zeros(len(bs), m, dtype=torch.long)
    ys = torch.full((len(bs), m), -100, dtype=torch.long)
    mk = torch.zeros(len(bs), m, dtype=torch.bool)
    ws = []
    zs = []
    for i, (x, y, l, w, z) in enumerate(bs):
        xs[i, :l] = x
        ys[i, :l] = y
        mk[i, :l] = True
        ws.append(w)
        zs.append(z)
    return xs, ys, torch.tensor(ls), mk, ws, zs


class Enc(nn.Module):
    def __init__(self, emb, h, nlab, fr, bi=True, dr=0.5, rd=False):
        super().__init__()
        if rd:
            self.emb = nn.Embedding(emb.shape[0], emb.shape[1])
            nn.init.uniform_(self.emb.weight, -0.1, 0.1)
            self.emb.weight.requires_grad = True
        else:
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=fr)
        self.bi = bi
        self.lstm = nn.LSTM(
            emb.shape[1],
            h,
            num_layers=2,
            batch_first=True,
            bidirectional=bi,
            dropout=dr if dr > 0 else 0.0,
        )
        self.drop = nn.Dropout(dr)
        self.fc = nn.Linear(h * (2 if bi else 1), nlab)

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
    def __init__(self, emb, h, nlab, fr, bi=True, dr=0.5, rd=False):
        super().__init__()
        self.enc = Enc(emb, h, nlab, fr, bi=bi, dr=dr, rd=rd)

    def loss(self, x, y, ls, mk):
        em = self.enc.feat(x, ls)
        fn = nn.CrossEntropyLoss(ignore_index=-100)
        return fn(em.view(-1, em.shape[-1]), y.view(-1))

    def pred(self, x, ls, mk):
        em = self.enc.feat(x, ls)
        return em.argmax(-1)


class NerCrf(nn.Module):
    def __init__(self, emb, h, nlab, fr, bi=True, dr=0.5, rd=False):
        super().__init__()
        self.enc = Enc(emb, h, nlab, fr, bi=bi, dr=dr, rd=rd)
        self.crf = Crf(nlab)

    def loss(self, x, y, ls, mk):
        em = self.enc.feat(x, ls)
        y2 = y.masked_fill(~mk, 0)
        return self.crf.loss(em, y2, mk)

    def pred(self, x, ls, mk):
        em = self.enc.feat(x, ls)
        return self.crf.decode(em, mk)


class NerSoft(nn.Module):
    def __init__(self, emb, h, nlab, fr, bi=True, dr=0.5, rd=False):
        super().__init__()
        self.enc = Enc(emb, h, nlab, fr, bi=bi, dr=dr, rd=rd)

    def loss(self, x, y, ls, mk):
        em = self.enc.feat(x, ls)
        fn = nn.CrossEntropyLoss(ignore_index=-100)
        return fn(em.view(-1, em.shape[-1]), y.view(-1))

    def pred(self, x, ls, mk):
        return self.enc.feat(x, ls).argmax(-1)


def token_eval(md, dl, i2y, dev, ner=False):
    md.eval()
    ys = []
    ps = []
    sents = []
    gls = []
    pls = []
    with torch.no_grad():
        for x, y, ln, mk, ws, zs in dl:
            x = x.to(dev)
            y = y.to(dev)
            mk = mk.to(dev)
            pr = md.pred(x, ln, mk)
            for i in range(len(ws)):
                l = int(ln[i].item())
                gy = y[i, :l].cpu().tolist()
                if ner:
                    if isinstance(pr, list):
                        py = pr[i]
                    else:
                        py = pr[i, :l].cpu().tolist()
                else:
                    py = pr[i, :l].cpu().tolist()
                ys.extend(gy)
                ps.extend(py)
                sents.append(ws[i])
                gls.append([i2y[z] for z in gy])
                pls.append([i2y[z] for z in py])
    acc = float(sum(int(a == b) for a, b in zip(ys, ps)) / max(1, len(ys)))
    f1 = f1_score(ys, ps, average="macro", zero_division=0)
    return acc, f1, ys, ps, sents, gls, pls


def build_labs(xs):
    ys = sorted({y for _, zs in xs for y in zs})
    return ys, {y: i for i, y in enumerate(ys)}


def load_ck(fp, cls, emb, dev):
    z = torch.load(fp, map_location="cpu")
    md = cls(emb, 128, len(z["idx2label"]), z["frozen"])
    md.load_state_dict(z["state_dict"])
    return md.to(dev), z["idx2label"], z["label2idx"], z["word2idx"], z["frozen"]


def draw_cm(cm, labs, fp):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labs)), labs, rotation=45, ha="right")
    plt.yticks(range(len(labs)), labs)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("POS Confusion Matrix")
    for i in range(len(labs)):
        for j in range(len(labs)):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(fp, dpi=200)
    plt.close()


def confused(cm, labs):
    xs = []
    for i in range(len(labs)):
        for j in range(len(labs)):
            if i != j and cm[i, j] > 0:
                xs.append((int(cm[i, j]), labs[i], labs[j], i, j))
    xs.sort(reverse=True)
    return xs[:3]


def ex_for_pair(sents, gls, pls, a, b, n=2):
    out = []
    for ws, gy, py in zip(sents, gls, pls):
        ok = False
        for g, p in zip(gy, py):
            if g == a and p == b:
                ok = True
                break
        if ok:
            out.append(" ".join(ws))
        if len(out) == n:
            break
    return out


def spans(xs):
    out = []
    i = 0
    while i < len(xs):
        y = xs[i]
        if y.startswith("B-"):
            tp = y[2:]
            j = i + 1
            while j < len(xs) and xs[j] == f"I-{tp}":
                j += 1
            out.append((i, j, tp))
            i = j
        else:
            i += 1
    return out


def ner_metrics(gls, pls):
    tps = Counter()
    fps = Counter()
    fns = Counter()
    for gy, py in zip(gls, pls):
        gs = set(spans(gy))
        ps = set(spans(py))
        for x in ps & gs:
            tps[x[2]] += 1
        for x in ps - gs:
            fps[x[2]] += 1
        for x in gs - ps:
            fns[x[2]] += 1
    rows = []
    for tp in ["PER", "LOC", "ORG", "MISC"]:
        tpv = tps[tp]
        fpv = fps[tp]
        fnv = fns[tp]
        pr = tpv / max(1, tpv + fpv)
        rc = tpv / max(1, tpv + fnv)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        rows.append((tp, pr, rc, f1, tpv, fpv, fnv))
    atp = sum(tps.values())
    afp = sum(fps.values())
    afn = sum(fns.values())
    pr = atp / max(1, atp + afp)
    rc = atp / max(1, atp + afn)
    f1 = 2 * pr * rc / max(1e-9, pr + rc)
    rows.append(("ALL", pr, rc, f1, atp, afp, afn))
    return rows


def err_ner(sents, gls, pls):
    fp = []
    fn = []
    for ws, gy, py in zip(sents, gls, pls):
        gs = set(spans(gy))
        ps = set(spans(py))
        for a, b, tp in sorted(ps - gs):
            tx = " ".join(ws[a:b])
            fp.append((" ".join(ws), tx, tp, "predicted entity not supported by gold sequence"))
        for a, b, tp in sorted(gs - ps):
            tx = " ".join(ws[a:b])
            fn.append((" ".join(ws), tx, tp, "gold entity missed by model or reduced to O"))
    return fp[:5], fn[:5]


def fit(name, md, tr, va, i2y, dev, epn=20, pat=5):
    opt = torch.optim.Adam(md.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -1.0
    bvl = 1e9
    bad = 0
    bst = None
    for ep in range(1, epn + 1):
        md.train()
        for x, y, ln, mk, _, _ in tr:
            x = x.to(dev)
            y = y.to(dev)
            mk = mk.to(dev)
            opt.zero_grad()
            loss = md.loss(x, y, ln, mk)
            loss.backward()
            opt.step()
        _, f1, _, _, _, _, _ = token_eval(md, va, i2y, dev, ner=("ner" in name))
        vls = []
        md.eval()
        with torch.no_grad():
            for x, y, ln, mk, _, _ in va:
                x = x.to(dev)
                y = y.to(dev)
                mk = mk.to(dev)
                vls.append(float(md.loss(x, y, ln, mk).item()))
        vl = sum(vls) / max(1, len(vls))
        print(f"{name} ep {ep} val_loss {vl:.4f} val_f1 {f1:.4f}")
        if f1 > best or (abs(f1 - best) < 1e-8 and vl < bvl):
            best = f1
            bvl = vl
            bad = 0
            bst = {k: v.detach().cpu() for k, v in md.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    md.load_state_dict(bst)
    return md, best


def make_dl(xs, w2i, y2i, bs, sh):
    return DataLoader(SeqSet(xs, w2i, y2i), batch_size=bs, shuffle=sh, collate_fn=collate)


def mode_cmp(task, cls, tr, va, te, emb, y2i, i2y, dev):
    rows = []
    for fr in [True, False]:
        md = cls(emb, 128, len(i2y), fr).to(dev)
        md, vf = fit(f"{task}_{'frozen' if fr else 'finetuned'}", md, tr, va, i2y, dev, epn=15, pat=5)
        ac, tf, _, _, _, _, _ = token_eval(md, te, i2y, dev, ner=(task == "ner"))
        rows.append(("frozen" if fr else "finetuned", vf, tf, ac))
    return rows


def ablations(trp, vap, tep, trn, van, ten, emb, p_i2y, p_y2i, n_i2y, n_y2i, dev):
    out = []
    md = Pos(emb, 128, len(p_i2y), False, bi=False, dr=0.5, rd=False).to(dev)
    md, vf = fit("A1_pos", md, trp, vap, p_i2y, dev, epn=12, pat=4)
    ac, tf, *_ = token_eval(md, tep, p_i2y, dev, ner=False)
    out.append(("A1", "unidirectional", "POS", vf, tf, ac))
    md = Pos(emb, 128, len(p_i2y), False, bi=True, dr=0.0, rd=False).to(dev)
    md, vf = fit("A2_pos", md, trp, vap, p_i2y, dev, epn=12, pat=4)
    ac, tf, *_ = token_eval(md, tep, p_i2y, dev, ner=False)
    out.append(("A2", "no dropout", "POS", vf, tf, ac))
    md = Pos(emb, 128, len(p_i2y), False, bi=True, dr=0.5, rd=True).to(dev)
    md, vf = fit("A3_pos", md, trp, vap, p_i2y, dev, epn=12, pat=4)
    ac, tf, *_ = token_eval(md, tep, p_i2y, dev, ner=False)
    out.append(("A3", "random init", "POS", vf, tf, ac))
    md = NerSoft(emb, 128, len(n_i2y), False, bi=True, dr=0.5, rd=False).to(dev)
    md, vf = fit("A4_ner", md, trn, van, n_i2y, dev, epn=15, pat=5)
    ac, tf, _, _, _, gls, pls = token_eval(md, ten, n_i2y, dev, ner=True)
    mts = ner_metrics(gls, pls)
    allf = [x for x in mts if x[0] == "ALL"][0][3]
    out.append(("A4", "softmax not crf", "NER", vf, allf, ac))
    return out


def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    Path("models").mkdir(exist_ok=True)
    emb = np.load("embeddings/embeddings_w2v.npy").astype(np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_tr = read_conll("data/pos_train.conll")
    pos_va = read_conll("data/pos_val.conll")
    pos_te = read_conll("data/pos_test.conll")
    ner_tr = read_conll("data/ner_train.conll")
    ner_va = read_conll("data/ner_val.conll")
    ner_te = read_conll("data/ner_test.conll")
    pos_md, pos_i2y, pos_y2i, w2i, pos_fr = load_ck("models/bilstm_pos.pt", Pos, emb, dev)
    ner_md, ner_i2y, ner_y2i, _, ner_fr = load_ck("models/bilstm_ner.pt", NerCrf, emb, dev)
    pos_te_dl = make_dl(pos_te, w2i, pos_y2i, 64, False)
    ner_te_dl = make_dl(ner_te, w2i, ner_y2i, 64, False)
    pos_va_dl = make_dl(pos_va, w2i, pos_y2i, 64, False)
    ner_va_dl = make_dl(ner_va, w2i, ner_y2i, 64, False)
    acc, mf1, ys, ps, sents, gls, pls = token_eval(pos_md, pos_te_dl, pos_i2y, dev, ner=False)
    cm = confusion_matrix(ys, ps, labels=list(range(len(pos_i2y))))
    draw_cm(cm, pos_i2y, "models/pos_confusion_matrix.png")
    print(f"saved pos mode {'frozen' if pos_fr else 'finetuned'}")
    print(f"pos token accuracy {acc:.4f}")
    print(f"pos macro_f1 {mf1:.4f}")
    print("pos confusion matrix:")
    print(cm)
    print("most confused pos pairs:")
    for c, a, b, _, _ in confused(cm, pos_i2y):
        print(f"{a} -> {b}: {c}")
        for ex in ex_for_pair(sents, gls, pls, a, b, 2):
            say(ex)
    trp = make_dl(pos_tr, w2i, pos_y2i, 32, True)
    vap = make_dl(pos_va, w2i, pos_y2i, 64, False)
    tep = make_dl(pos_te, w2i, pos_y2i, 64, False)
    print("pos frozen vs finetuned:")
    rows = mode_cmp("pos", Pos, trp, vap, tep, emb, pos_y2i, pos_i2y, dev)
    print(f"{'mode':<10} {'val_f1':>8} {'test_f1':>8} {'acc':>8}")
    for a, b, c, d in rows:
        print(f"{a:<10} {b:>8.4f} {c:>8.4f} {d:>8.4f}")
    nacc, nmf1, nys, nps, nsents, ngls, npls = token_eval(ner_md, ner_te_dl, ner_i2y, dev, ner=True)
    print(f"saved ner mode {'frozen' if ner_fr else 'finetuned'}")
    print("ner metrics with crf:")
    mets = ner_metrics(ngls, npls)
    print(f"{'type':<6} {'prec':>8} {'rec':>8} {'f1':>8} {'tp':>5} {'fp':>5} {'fn':>5}")
    for tp, pr, rc, f1, tpv, fpv, fnv in mets:
        print(f"{tp:<6} {pr:>8.4f} {rc:>8.4f} {f1:>8.4f} {tpv:>5} {fpv:>5} {fnv:>5}")
    trn = make_dl(ner_tr, w2i, ner_y2i, 32, True)
    van = make_dl(ner_va, w2i, ner_y2i, 64, False)
    ten = make_dl(ner_te, w2i, ner_y2i, 64, False)
    soft = NerSoft(emb, 128, len(ner_i2y), False).to(dev)
    soft, svf = fit("ner_softmax", soft, trn, van, ner_i2y, dev, epn=15, pat=5)
    _, _, _, _, _, sgls, spls = token_eval(soft, ten, ner_i2y, dev, ner=True)
    sm = ner_metrics(sgls, spls)
    print("ner crf vs softmax:")
    a = [x for x in mets if x[0] == "ALL"][0]
    b = [x for x in sm if x[0] == "ALL"][0]
    print(f"{'model':<10} {'prec':>8} {'rec':>8} {'f1':>8}")
    print(f"{'crf':<10} {a[1]:>8.4f} {a[2]:>8.4f} {a[3]:>8.4f}")
    print(f"{'softmax':<10} {b[1]:>8.4f} {b[2]:>8.4f} {b[3]:>8.4f}")
    fp, fn = err_ner(nsents, ngls, npls)
    print("ner false positives:")
    for s, tx, tp, ex in fp:
        say(f"{tp} | {tx} | {ex}")
        say(s)
    print("ner false negatives:")
    for s, tx, tp, ex in fn:
        say(f"{tp} | {tx} | {ex}")
        say(s)
    print("ablations:")
    absr = ablations(trp, vap, tep, trn, van, ten, emb, pos_i2y, pos_y2i, ner_i2y, ner_y2i, dev)
    print(f"{'id':<4} {'change':<18} {'task':<5} {'val_f1':>8} {'test_f1':>8} {'acc':>8}")
    for a, b, c, d, e, f in absr:
        print(f"{a:<4} {b:<18} {c:<5} {d:>8.4f} {e:>8.4f} {f:>8.4f}")


if __name__ == "__main__":
    main()
