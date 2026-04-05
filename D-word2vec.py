import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def load_meta():
    p = Path("Metadata.json")
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_docs(meta):
    p = Path("cleaned.txt")
    if not p.exists():
        raise FileNotFoundError("cleaned.txt not found")
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
    p = Path("embeddings/word2idx.json")
    if not p.exists():
        raise FileNotFoundError("embeddings/word2idx.json not found")
    with p.open(encoding="utf-8") as f:
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


def pair_count(xs, k):
    s = 0
    for ws in xs:
        m = len(ws)
        for i in range(m):
            s += min(k, i) + min(k, m - i - 1)
    return s


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


def draw(ls):
    xs = np.arange(1, len(ls) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ls, marker="o", linewidth=2)
    plt.title("Skip-gram Word2Vec Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.close()


def main():
    Path("embeddings").mkdir(exist_ok=True)
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    meta = load_meta()
    ds = load_docs(meta)
    idx = load_idx()
    xs, fq = to_ids(ds, idx)
    n = len(idx)
    d = 100
    k = 5
    q = 10
    bs = 8192
    ep = 5
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws, ps = noise(fq, n)
    pt = pair_count(xs, k)
    cp, op = build_pairs(xs, k)
    m = SG(n, d).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    all_steps = max(1, math.ceil(len(cp) / bs))
    gap = max(1, all_steps // 8)
    los = []
    for e in range(ep):
        pr = np.random.permutation(len(cp))
        run = 0.0
        cnt = 0
        for t, st in enumerate(range(0, len(cp), bs), 1):
            ed = min(len(cp), st + bs)
            ix = pr[st:ed]
            c = torch.tensor(cp[ix], dtype=torch.long, device=dev)
            o = torch.tensor(op[ix], dtype=torch.long, device=dev)
            ng = np.random.choice(ws, size=(len(ix), q), p=ps)
            ng = torch.tensor(ng, dtype=torch.long, device=dev)
            opt.zero_grad()
            loss = m(c, o, ng)
            loss.backward()
            opt.step()
            run += float(loss.item())
            cnt += 1
            if t % gap == 0 or t == all_steps:
                print(f"ep {e + 1} step {t}/{all_steps} loss {run / cnt:.4f}")
        av = run / max(1, cnt)
        los.append(av)
        print(f"ep {e + 1} avg {av:.4f}")
    emb = ((m.v.weight.data + m.u.weight.data) / 2.0).detach().cpu().numpy().astype(np.float32)
    np.save("embeddings/embeddings_w2v.npy", emb)
    draw(los)


if __name__ == "__main__":
    main()
