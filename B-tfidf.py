import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


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
    ls = p.read_text(encoding="utf-8").splitlines()
    xs = [x.strip() for x in ls if x.strip()]
    ks = set(meta.keys())
    if ks:
        ds = []
        ids = []
        cur = []
        cur_id = None
        for x in xs:
            if x.isdigit() and x in ks:
                if cur_id is not None and cur:
                    ds.append(" ".join(cur))
                    ids.append(cur_id)
                cur_id = x
                cur = []
            else:
                cur.append(x)
        if cur_id is not None and cur:
            ds.append(" ".join(cur))
            ids.append(cur_id)
        if len(ds) == len(meta):
            return ds, ids
    return xs, [str(i + 1) for i in range(len(xs))]


def build_vocab(ds, cap=10000):
    fq = Counter()
    for x in ds:
        fq.update(x.split())
    ws = [w for w, _ in fq.most_common(cap)]
    idx = {"<UNK>": 0}
    for i, w in enumerate(ws, 1):
        idx[w] = i
    return idx


def build_tf(ds, idx):
    m = np.zeros((len(ds), len(idx)), dtype=np.float32)
    df = np.zeros(len(idx), dtype=np.int32)
    for i, x in enumerate(ds):
        c = Counter()
        for w in x.split():
            c[w if w in idx else "<UNK>"] += 1
        for w, v in c.items():
            j = idx[w]
            m[i, j] = float(v)
            df[j] += 1
    return m, df


def load_labs(meta, ids):
    ys = []
    for k in ids:
        z = meta.get(k, {})
        y = str(z.get("category", "")).strip()
        if not y:
            y = "uncat"
        ys.append(y)
    return ys


def top_words(m, ys, idx):
    rev = {v: k for k, v in idx.items()}
    labs = sorted(set(ys))
    all_mu = m.mean(axis=0)
    for y in labs:
        ps = [i for i, z in enumerate(ys) if z == y]
        sub = m[ps]
        sc = sub.mean(axis=0)
        if len(labs) > 1 and len(ps) < len(ys):
            rs = [i for i, z in enumerate(ys) if z != y]
            sc = sc - m[rs].mean(axis=0)
        else:
            sc = sc - all_mu
        ords = np.argsort(-sc)
        out = []
        for j in ords:
            w = rev[int(j)]
            if w == "<UNK>":
                continue
            out.append(w)
            if len(out) == 10:
                break
        say(f"{y}: {', '.join(out)}")


def say(x):
    sys.stdout.buffer.write((x + "\n").encode("utf-8", errors="backslashreplace"))


def main():
    Path("embeddings").mkdir(exist_ok=True)
    meta = load_meta()
    ds, ids = load_docs(meta)
    idx = build_vocab(ds)
    tf, df = build_tf(ds, idx)
    n = len(ds)
    idf = np.array([math.log(n / (1 + int(v))) for v in df], dtype=np.float32)
    x = tf * idf
    np.save("embeddings/tfidf_matrix.npy", x)
    with open("embeddings/word2idx.json", "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    ys = load_labs(meta, ids)
    say(f"docs: {len(ds)}")
    say(f"vocab: {len(idx)}")
    say("top words:")
    top_words(x, ys, idx)


if __name__ == "__main__":
    main()
