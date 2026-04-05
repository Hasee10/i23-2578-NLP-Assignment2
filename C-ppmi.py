import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def say(x):
    sys.stdout.buffer.write((x + "\n").encode("utf-8", errors="backslashreplace"))


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


def load_idx(ds, cap=10000):
    p = Path("embeddings/word2idx.json")
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    fq = Counter()
    for x in ds:
        fq.update(x.split())
    idx = {"<UNK>": 0}
    for i, (w, _) in enumerate(fq.most_common(cap), 1):
        idx[w] = i
    return idx


def build(ds, idx, k=5):
    n = len(idx)
    co = defaultdict(float)
    fq = Counter()
    for x in ds:
        ws = [w if w in idx else "<UNK>" for w in x.split()]
        fq.update(ws)
        m = len(ws)
        for i, w in enumerate(ws):
            a = idx[w]
            lo = max(0, i - k)
            hi = min(m, i + k + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                b = idx[ws[j]]
                co[(a, b)] += 1.0
    ct = np.zeros((n, n), dtype=np.float32)
    for (a, b), v in co.items():
        ct[a, b] = v
    sm = float(ct.sum())
    rs = ct.sum(axis=1)
    cs = ct.sum(axis=0)
    pp = np.zeros((n, n), dtype=np.float32)
    for (a, b), v in co.items():
        d = float(rs[a] * cs[b])
        if d <= 0.0:
            continue
        z = math.log2((v * sm) / d)
        if z > 0.0:
            pp[a, b] = z
    return pp, fq


def pick_top(fq, idx, m=200):
    bad = {"<UNK>", "<NUM>", "،", "۔", ".", ",", ":", ";", "!", "?"}
    xs = []
    for w, _ in fq.most_common():
        if w not in idx or w in bad:
            continue
        if len(w) < 2:
            continue
        xs.append(w)
        if len(xs) == m:
            break
    return xs


def cat_map(pp, idx):
    sd = {
        "politics": ["حکومت", "وزیراعظم", "صدر", "پارلیمان", "انتخاب"],
        "sports": ["کرکٹ", "فٹبال", "میچ", "کھلاڑی", "ٹیم"],
        "geography": ["شہر", "ملک", "دریا", "سمندر", "پہاڑ"],
        "economy": ["معیشت", "بازار", "قیمت", "سرمایہ", "تجارت"],
        "health": ["ہسپتال", "ڈاکٹر", "بیماری", "صحت", "علاج"],
    }
    out = {}
    for c, ws in sd.items():
        vs = [pp[idx[w]] for w in ws if w in idx]
        if vs:
            out[c] = np.mean(vs, axis=0)
    return out


def tag(ws, pp, idx, cm):
    cs = list(cm.keys())
    if not cs:
        return {w: "politics" for w in ws}
    lb = {}
    for w in ws:
        v = pp[idx[w]]
        bn = None
        bs = -1.0
        nv = float(np.linalg.norm(v))
        for c in cs:
            u = cm[c]
            nu = float(np.linalg.norm(u))
            s = 0.0
            if nv > 0.0 and nu > 0.0:
                s = float(np.dot(v, u) / (nv * nu))
            if s > bs:
                bs = s
                bn = c
        lb[w] = bn or cs[0]
    return lb


def dump_tsne(ws, z, lb):
    with open("embeddings/ppmi_tsne_coords.csv", "w", encoding="utf-8") as f:
        f.write("tok,x,y,cat\n")
        for i, w in enumerate(ws):
            x = float(z[i, 0])
            y = float(z[i, 1])
            c = lb[w]
            f.write(f"{w},{x:.6f},{y:.6f},{c}\n")


def draw(ws, pp, idx):
    xs = np.stack([pp[idx[w]] for w in ws]).astype(np.float32)
    z = TSNE(n_components=2, perplexity=20, random_state=7, init="pca", learning_rate="auto").fit_transform(xs)
    cm = cat_map(pp, idx)
    lb = tag(ws, pp, idx, cm)
    dump_tsne(ws, z, lb)
    cd = {
        "politics": "#d1495b",
        "sports": "#2e86ab",
        "geography": "#4f772d",
        "economy": "#d17a22",
        "health": "#6c5ce7",
    }
    fig, ax = plt.subplots(figsize=(14, 10))
    for c in ["politics", "sports", "geography", "economy", "health"]:
        ps = [i for i, w in enumerate(ws) if lb[w] == c]
        if not ps:
            continue
        ax.scatter(z[ps, 0], z[ps, 1], s=55, alpha=0.8, label=c, color=cd[c])
        for i in ps:
            ax.text(z[i, 0], z[i, 1], ws[i], fontsize=8, alpha=0.85)
    ax.set_title("t-SNE of Top 200 BBC Urdu Tokens")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig("embeddings/ppmi_tsne.png", dpi=200)
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    for c in ["politics", "sports", "geography", "economy", "health"]:
        ps = [i for i, w in enumerate(ws) if lb[w] == c]
        if not ps:
            continue
        ax2.scatter(z[ps, 0], z[ps, 1], s=70, alpha=0.85, label=c, color=cd[c])
    ax2.set_title("t-SNE of Top 200 BBC Urdu Tokens")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("embeddings/ppmi_tsne_clean.png", dpi=200)
    plt.close(fig)
    plt.close(fig2)


def nn(pp, idx, qs):
    rv = {v: k for k, v in idx.items()}
    nm = np.linalg.norm(pp, axis=1)
    out = []
    for q in qs:
        if q not in idx:
            ln = f"{q}: missing"
            say(ln)
            out.append(ln)
            continue
        i = idx[q]
        v = pp[i]
        nv = float(np.linalg.norm(v))
        sc = np.zeros(len(idx), dtype=np.float32)
        if nv > 0.0:
            den = nm * nv
            ok = den > 0.0
            sc[ok] = (pp[ok] @ v) / den[ok]
        sc[i] = -1.0
        js = np.argsort(-sc)[:5]
        xs = [rv[int(j)] for j in js]
        ln = f"{q}: {', '.join(xs)}"
        say(ln)
        out.append(ln)
    return out


def pick_qs(fq, idx):
    qs = ["پاکستان", "بھارت", "کرکٹ", "حکومت", "صدر", "شہر", "معیشت", "صحت", "علاج", "بازار"]
    xs = [w for w in qs if w in idx]
    if len(xs) >= 10:
        return xs[:10]
    bad = set(xs) | {"<UNK>", "<NUM>", "،", "۔", ".", ","}
    for w, _ in fq.most_common():
        if w in idx and w not in bad and len(w) > 2:
            xs.append(w)
            bad.add(w)
        if len(xs) == 10:
            break
    return xs


def main():
    os.environ["LOKY_MAX_CPU_COUNT"] = "2"
    warnings.filterwarnings("ignore")
    Path("embeddings").mkdir(exist_ok=True)
    meta = load_meta()
    ds = load_docs(meta)
    idx = load_idx(ds)
    pp, fq = build(ds, idx, k=5)
    np.save("embeddings/ppmi_matrix.npy", pp)
    ws = pick_top(fq, idx, 200)
    draw(ws, pp, idx)
    say("top-5 neighbours:")
    rs = nn(pp, idx, pick_qs(fq, idx))
    with open("embeddings/ppmi_neighbours.txt", "w", encoding="utf-8") as f:
        f.write("top-5 neighbours:\n")
        for x in rs:
            f.write(x + "\n")


if __name__ == "__main__":
    main()
