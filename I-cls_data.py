import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_meta():
    with open("Metadata.json", encoding="utf-8") as f:
        return json.load(f)


def load_idx():
    with open("embeddings/word2idx.json", encoding="utf-8") as f:
        return json.load(f)


def load_docs(meta):
    xs = [x.strip() for x in Path("cleaned.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    ks = set(meta.keys())
    ds = []
    cur = []
    cur_id = None
    for x in xs:
        if x.isdigit() and x in ks:
            if cur_id is not None and cur:
                ds.append((cur_id, " ".join(cur)))
            cur_id = x
            cur = []
        else:
            cur.append(x)
    if cur_id is not None and cur:
        ds.append((cur_id, " ".join(cur)))
    return ds


def rules():
    return {
        0: {
            "name": "politics",
            "ws": {
                "election", "government", "minister", "parliament",
                "انتخاب", "حکومت", "وزیر", "پارلیمان", "اسمبلی", "صدر", "وزیراعظم", "اپوزیشن", "قانون",
            },
        },
        1: {
            "name": "sports",
            "ws": {
                "cricket", "match", "team", "player", "score",
                "کرکٹ", "میچ", "ٹیم", "کھلاڑی", "رنز", "وکٹ", "گول", "لیگ", "کپ", "بورڈ",
            },
        },
        2: {
            "name": "economy",
            "ws": {
                "inflation", "trade", "bank", "gdp", "budget",
                "معیشت", "تجارت", "بینک", "بجٹ", "مہنگائی", "قیمت", "ٹیکس", "سرمایہ", "ڈالر", "تیل",
            },
        },
        3: {
            "name": "international",
            "ws": {
                "un", "treaty", "foreign", "bilateral", "conflict",
                "اقوام", "متحدہ", "خارجہ", "دوطرفہ", "تنازع", "جنگ", "روس", "یوکرین", "امریکہ", "چین", "بھارت",
                "اسرائیل", "فلسطین", "ونزویلا", "مادورو", "ٹرمپ",
            },
        },
        4: {
            "name": "health",
            "ws": {
                "hospital", "disease", "vaccine", "flood", "education",
                "ہسپتال", "بیماری", "ویکسین", "سیلاب", "تعلیم", "صحت", "ڈاکٹر", "علاج", "مریض", "سکول", "کالج",
            },
        },
    }


def pick(tt, bd):
    z = (tt + " " + bd).lower()
    rs = rules()
    sc = {k: 0 for k in rs}
    for k, v in rs.items():
        for w in v["ws"]:
            if w.lower() in z:
                sc[k] += z.count(w.lower())
    if max(sc.values()) == 0:
        return 3
    return max(sc, key=sc.get)


def enc(x, idx, n=256):
    ys = [idx[w] if w in idx else 0 for w in x.split()]
    if len(ys) >= n:
        return ys[:n]
    return ys + [0] * (n - len(ys))


def show(tag, ys):
    nm = {k: v["name"] for k, v in rules().items()}
    ct = Counter(ys)
    print(tag)
    for i in range(5):
        print(f"{nm[i]}: {ct.get(i, 0)}")


def main():
    Path("data").mkdir(exist_ok=True)
    meta = load_meta()
    idx = load_idx()
    ds = load_docs(meta)
    xs = []
    ys = []
    ids = []
    for k, bd in ds:
        tt = meta.get(k, {}).get("title", "")
        y = pick(tt, bd)
        xs.append(enc(bd, idx, 256))
        ys.append(y)
        ids.append(int(k))
    x = np.array(xs, dtype=np.int64)
    y = np.array(ys, dtype=np.int64)
    z = np.array(ids, dtype=np.int64)
    x_tr, x_te, y_tr, y_te, z_tr, z_te = train_test_split(x, y, z, test_size=0.30, random_state=7, stratify=y)
    x_va, x_te, y_va, y_te, z_va, z_te = train_test_split(x_te, y_te, z_te, test_size=0.50, random_state=7, stratify=y_te)
    np.save("data/cls_train_x.npy", x_tr)
    np.save("data/cls_train_y.npy", y_tr)
    np.save("data/cls_train_id.npy", z_tr)
    np.save("data/cls_val_x.npy", x_va)
    np.save("data/cls_val_y.npy", y_va)
    np.save("data/cls_val_id.npy", z_va)
    np.save("data/cls_test_x.npy", x_te)
    np.save("data/cls_test_y.npy", y_te)
    np.save("data/cls_test_id.npy", z_te)
    print(f"all: {len(y)}")
    show("train", y_tr)
    show("val", y_va)
    show("test", y_te)


if __name__ == "__main__":
    main()
