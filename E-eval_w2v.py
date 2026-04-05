import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn


def say(x):
    sys.stdout.buffer.write((x + "\n").encode("utf-8", errors="backslashreplace"))


def load_meta():
    p = Path("Metadata.json")
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_docs(fp, meta):
    p = Path(fp)
    if not p.exists():
        raise FileNotFoundError(f"{fp} not found")
    xs = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    ks = set(meta.keys())
    if ks:
        ds = []
        cur = []
        cur_id = None
        for x in xs:
            if x.isdigit() and x in ks:
                if cur_id is not None and cur:
                    ds.append(" ".join(cur))
                cur_id = x
                cur = []
            else:
                cur.append(x)
        if cur_id is not None and cur:
            ds.append(" ".join(cur))
        if len(ds) == len(meta):
            return ds
    return xs


def load_idx():
    with open("embeddings/word2idx.json", encoding="utf-8") as f:
        return json.load(f)


def to_ids(ds, idx):
    xs = []
    fq = Counter()
    for x in ds:
        ws = []
        for w in x.split():
            z = idx[w] if w in idx else 0
            ws.append(z)
            fq[z] += 1
        if ws:
            xs.append(ws)
    return xs, fq


def noise(fq, n):
    ws = np.arange(n, dtype=np.int64)
    ps = np.ones(n, dtype=np.float64)
    for i in range(n):
        ps[i] = float(fq.get(int(i), 1)) ** 0.75
    ps /= ps.sum()
    return ws, ps


def build_pairs(xs, k):
    cs = []
    os_ = []
    for ws in xs:
        m = len(ws)
        for i, c in enumerate(ws):
            lo = max(0, i - k)
            hi = min(m, i + k + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                cs.append(c)
                os_.append(ws[j])
    return np.array(cs, dtype=np.int64), np.array(os_, dtype=np.int64)


class SG(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.v = nn.Embedding(n, d)
        self.u = nn.Embedding(n, d)
        a = 0.5 / d
        self.v.weight.data.uniform_(-a, a)
        self.u.weight.data.zero_()

    def forward(self, c, o, ng):
        vc = self.v(c)
        uo = self.u(o)
        sp = (vc * uo).sum(dim=1)
        lp = torch.nn.functional.logsigmoid(sp)
        un = self.u(ng)
        sn = torch.bmm(un, vc.unsqueeze(2)).squeeze(2)
        ln = torch.nn.functional.logsigmoid(-sn).sum(dim=1)
        return -(lp + ln).mean()


def train_skipgram(ds, idx, d, fp):
    p = Path(fp)
    if p.exists():
        z = np.load(p)
        return z["emb"].astype(np.float32)
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    xs, fq = to_ids(ds, idx)
    cp, op = build_pairs(xs, 5)
    n = len(idx)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws, ps = noise(fq, n)
    m = SG(n, d).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    bs = 8192 if d == 100 else 6144
    ep = 5
    steps = max(1, math.ceil(len(cp) / bs))
    gap = max(1, steps // 6)
    tag = Path(fp).stem
    for e in range(ep):
        pr = np.random.permutation(len(cp))
        run = 0.0
        cnt = 0
        for t, st in enumerate(range(0, len(cp), bs), 1):
            ed = min(len(cp), st + bs)
            ix = pr[st:ed]
            c = torch.tensor(cp[ix], dtype=torch.long, device=dev)
            o = torch.tensor(op[ix], dtype=torch.long, device=dev)
            ng = np.random.choice(ws, size=(len(ix), 10), p=ps)
            ng = torch.tensor(ng, dtype=torch.long, device=dev)
            opt.zero_grad()
            loss = m(c, o, ng)
            loss.backward()
            opt.step()
            run += float(loss.item())
            cnt += 1
            if t % gap == 0 or t == steps:
                say(f"{tag} ep {e + 1} step {t}/{steps} loss {run / cnt:.4f}")
        say(f"{tag} ep {e + 1} avg {run / max(1, cnt):.4f}")
    emb = ((m.v.weight.data + m.u.weight.data) / 2.0).detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(p, emb=emb)
    return emb


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return x / n


def topk(emb, idx, w, k, drop=None):
    if w not in idx:
        return []
    rv = {v: k for k, v in idx.items()}
    ne = norm(emb)
    i = idx[w]
    sc = ne @ ne[i]
    bad = {i}
    if drop:
        for x in drop:
            if x in idx:
                bad.add(idx[x])
    for j in bad:
        sc[j] = -1.0
    js = np.argsort(-sc)[:k]
    return [rv[int(j)] for j in js]


def rank_of(emb, idx, a, b):
    if a not in idx or b not in idx:
        return None
    rv = norm(emb)
    sc = rv @ rv[idx[a]]
    sc[idx[a]] = -1.0
    ords = np.argsort(-sc)
    hit = int(np.where(ords == idx[b])[0][0]) + 1
    return hit


def mrr(emb, idx, prs):
    rs = []
    for a, b in prs:
        r = rank_of(emb, idx, a, b)
        if r is not None:
            rs.append(1.0 / r)
    return float(sum(rs) / max(1, len(rs))), len(rs)


def analogy(emb, idx, a, b, c, topn=3):
    if a not in idx or b not in idx or c not in idx:
        return []
    rv = {v: k for k, v in idx.items()}
    ne = norm(emb)
    q = ne[idx[b]] - ne[idx[a]] + ne[idx[c]]
    qn = np.linalg.norm(q)
    if qn == 0.0:
        return []
    q = q / qn
    sc = ne @ q
    for x in [a, b, c]:
        sc[idx[x]] = -1.0
    js = np.argsort(-sc)[:topn]
    return [rv[int(j)] for j in js]


def pairs():
    return [
        ("پاکستان", "بھارت"),
        ("حکومت", "وزیراعظم"),
        ("عدالت", "جج"),
        ("معیشت", "بازار"),
        ("فوج", "جنگ"),
        ("صحت", "ہسپتال"),
        ("تعلیم", "طالبعلم"),
        ("آبادی", "شہر"),
        ("کرکٹ", "ٹیم"),
        ("فٹبال", "میچ"),
        ("ڈاکٹر", "ہسپتال"),
        ("مریض", "علاج"),
        ("صدر", "امریکہ"),
        ("قیمت", "بازار"),
        ("سرمایہ", "معیشت"),
        ("قانون", "عدالت"),
        ("سپاہی", "فوج"),
        ("جرم", "پولیس"),
        ("بیماری", "علاج"),
        ("اساتذہ", "تعلیم"),
        ("طالبعلم", "سکول"),
        ("ملازمت", "معیشت"),
        ("شہر", "ملک"),
        ("قومی", "اسمبلی"),
        ("پارلیمان", "حکومت"),
    ]


def analogies():
    return [
        ("ٹرمپ", "ڈونلڈ", "مادورو", "نکولس"),
        ("وزیراعظم", "شہباز", "ٹرمپ", "ڈونلڈ"),
        ("وزیراعظم", "شریف", "مادورو", "نکولس"),
        ("عدالت", "جج", "پولیس", "افسر"),
        ("پولیس", "افسر", "عدالت", "جج"),
        ("پولیس", "افسر", "حکومت", "وزارت"),
        ("پولیس", "افسر", "صدر", "خارجہ"),
        ("عدالت", "جج", "حکومت", "وزارت"),
        ("عدالت", "جج", "صدر", "خارجہ"),
        ("ٹرمپ", "ڈونلڈ", "حکومت", "وزارت"),
    ]


def rom():
    return {
        "Pakistan": "پاکستان",
        "Hukumat": "حکومت",
        "Adalat": "عدالت",
        "Maeeshat": "معیشت",
        "Fauj": "فوج",
        "Sehat": "صحت",
        "Taleem": "تعلیم",
        "Aabadi": "آبادی",
    }


def cond_words():
    return ["پاکستان", "حکومت", "فوج", "صحت", "تعلیم"]


def filt(xs, idx, n):
    ys = []
    for x in xs:
        ok = all(w in idx for w in x[:n])
        if ok:
            ys.append(x)
    return ys


def main():
    meta = load_meta()
    idx = load_idx()
    e3 = np.load("embeddings/embeddings_w2v.npy").astype(np.float32)
    pp = np.load("embeddings/ppmi_matrix.npy", mmap_mode="r")
    say("top-10 neighbours on cleaned skipgram:")
    for a, w in rom().items():
        xs = topk(e3, idx, w, 10)
        say(f"{a} ({w}): {', '.join(xs) if xs else 'missing'}")
    say("")
    say("analogy tests on cleaned skipgram:")
    ag = filt(analogies(), idx, 4)[:10]
    hit = 0
    for a, b, c, d in ag:
        xs = analogy(e3, idx, a, b, c, 3)
        ok = d in xs
        if ok:
            hit += 1
        say(f"{b} - {a} + {c} -> {', '.join(xs)} | exp: {d} | hit: {ok}")
    say(f"analogy hits in top-3: {hit}/{len(ag)}")
    say("")
    d1 = load_docs("cleaned.txt", meta)
    d2 = load_docs("raw.txt", meta)
    e2 = train_skipgram(d2, idx, 100, "embeddings/c2_raw_w2v.npz")
    e4 = train_skipgram(d1, idx, 200, "embeddings/c4_cleaned_w2v_200.npz")
    prs = filt(pairs(), idx, 2)[:20]
    cds = [
        ("C1", "PPMI baseline", np.asarray(pp, dtype=np.float32)),
        ("C2", "Word2Vec raw.txt", e2),
        ("C3", "Word2Vec cleaned.txt", e3),
        ("C4", "Word2Vec cleaned d=200", e4),
    ]
    rows = []
    say("")
    say("4-condition neighbours:")
    for c, n, emb in cds:
        say(f"{c} {n}")
        for w in cond_words():
            xs = topk(emb, idx, w, 5)
            say(f"{w}: {', '.join(xs) if xs else 'missing'}")
        mr, used = mrr(emb, idx, prs)
        rows.append((c, n, mr, used))
        say(f"mrr: {mr:.4f} on {used} pairs")
        say("")
    say("summary table:")
    say(f"{'cond':<4} {'name':<24} {'mrr':>8} {'pairs':>6}")
    for c, n, mr, used in rows:
        say(f"{c:<4} {n:<24} {mr:>8.4f} {used:>6}")
    best = max(rows, key=lambda x: x[2])
    say("")
    say(f"best condition: {best[0]} {best[1]} with mrr {best[2]:.4f}")


if __name__ == "__main__":
    main()
