import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


def say(x):
    sys.stdout.buffer.write((x + "\n").encode("utf-8", errors="backslashreplace"))


def load_meta():
    with open("Metadata.json", encoding="utf-8") as f:
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


def split_sent(x):
    ys = re.split(r"(?<=[۔!?؟])\s+|\s*\.\s+", x)
    out = []
    for y in ys:
        y = y.strip()
        if y:
            out.append(y)
    return out


def tok(x):
    return re.findall(r"[\u0600-\u06FF]+(?:[\u0600-\u06FF]+)?|[A-Za-z]+|<NUM>|\d+|[۔،!?؟:؛()\"'.,-]", x)


def topic_rules():
    return {
        "politics": {
            "سیاست", "حکومت", "وزیراعظم", "صدر", "پارلیمان", "اسمبلی", "انتخاب", "اپوزیشن", "وزارت", "فوج", "عدالت",
            "جج", "پولیس", "عمران", "شہباز", "ٹرمپ", "امریک", "چین", "بھارت", "روس", "یوکرین", "مادورو", "قانون",
        },
        "sports": {
            "کرکٹ", "فٹبال", "ہاکی", "میچ", "ٹیم", "کھلاڑی", "کپ", "لیگ", "بورڈ", "وکٹ", "رنز", "گول",
            "ٹورنامنٹ", "کھیل", "بیٹر", "باؤلر", "اسپورٹس",
        },
        "health": {
            "صحت", "ہسپتال", "ڈاکٹر", "علاج", "مریض", "بیماری", "ویکسین", "ادویات", "میڈیکل", "سرجری", "کلینک",
            "انفیکشن", "وبا", "نرس", "خون", "بخار",
        },
        "economy": {
            "معیشت", "بازار", "قیمت", "مہنگائی", "سرمایہ", "تجارت", "کاروبار", "ٹیکس", "بجٹ", "کرنسی", "تیل",
            "ڈالر", "بینک", "درآمد", "برآمد", "سرمایہ کاری",
        },
        "geography": {
            "شہر", "ملک", "علاقہ", "دریا", "سمندر", "پہاڑ", "صوبہ", "جھیل", "گاؤں", "جزیرہ", "حدود", "عالمی",
            "راستہ", "ساحل", "بندرگاہ",
        },
    }


def pick_topic(tt, bd):
    z = tt + " " + bd
    rs = topic_rules()
    sc = {k: 0 for k in rs}
    for k, ws in rs.items():
        for w in ws:
            if w in z:
                sc[k] += z.count(w)
    if max(sc.values()) == 0:
        return "politics"
    return max(sc, key=sc.get)


def sample_rows(ds, meta):
    ps = []
    for k, bd in ds:
        tt = meta.get(k, {}).get("title", "")
        tp = pick_topic(tt, bd)
        for s in split_sent(bd):
            ts = tok(s)
            if 5 <= len(ts) <= 40:
                ps.append({"id": k, "topic": tp, "sent": s, "tok": ts})
    gp = defaultdict(list)
    for x in ps:
        gp[x["topic"]].append(x)
    tops = sorted(gp, key=lambda x: len(gp[x]), reverse=True)[:3]
    if len(tops) < 3:
        raise ValueError("not enough topic groups")
    random.seed(7)
    out = []
    seen = set()
    for tp in tops:
        xs = gp[tp][:]
        random.shuffle(xs)
        take = xs[:100]
        out.extend(take)
        seen.update(id(x) for x in take)
    rest = []
    for tp in tops:
        for x in gp[tp]:
            if id(x) not in seen:
                rest.append(x)
    random.shuffle(rest)
    out.extend(rest[: max(0, 500 - len(out))])
    if len(out) < 500:
        raise ValueError("not enough sampled sentences")
    return out[:500], tops


def build_pos_lex():
    ns = {
        "پاکستان", "بھارت", "چین", "امریکہ", "روس", "یوکرین", "افغانستان", "ایران", "لاہور", "کراچی", "اسلام", "آباد",
        "پشاور", "کوئٹہ", "ملتان", "فیصل", "آباد", "حکومت", "پارلیمان", "اسمبلی", "وزارت", "صدر", "وزیراعظم", "عدالت",
        "جج", "پولیس", "فوج", "معیشت", "بازار", "قیمت", "بجٹ", "ٹیکس", "بینک", "کاروبار", "سرمایہ", "کرنسی", "صحت",
        "ہسپتال", "ڈاکٹر", "علاج", "مریض", "ادویات", "بیماری", "تعلیم", "سکول", "کالج", "جامعہ", "اساتذہ", "طالبعلم",
        "طلبا", "نساب", "امتحان", "کلاس", "کرکٹ", "فٹبال", "ہاکی", "میچ", "ٹیم", "کھلاڑی", "رنز", "وکٹ", "گول", "کپ",
        "لیگ", "بورڈ", "شہر", "ملک", "علاقہ", "صوبہ", "گاؤں", "دریا", "سمندر", "جھیل", "پہاڑ", "ساحل", "بارش", "درجہ",
        "حرارت", "موسم", "خبر", "رپورٹ", "تحقیق", "منصوبہ", "بیان", "فیصلہ", "انتخاب", "قانون", "جرم", "مقدمہ", "سماعت",
        "ملزم", "گواہ", "مسئلہ", "حل", "خدمت", "تنظیم", "ادارہ", "کمیٹی", "کونسل", "کمپنی", "فیکٹری", "روزگار", "ملازمت",
        "تنخواہ", "توانائی", "تیل", "گیس", "بجلی", "پانی", "سڑک", "پل", "ٹرین", "بس", "ہوائی", "جہاز", "پرواز", "تصویر",
        "ویڈیو", "کتاب", "مضمون", "تاریخ", "ثقافت", "زبان", "فن", "موسیقی", "فلم", "کردار", "قبیلہ", "آبادی", "شہری",
    }
    vs = {
        "ہے", "ہیں", "تھا", "تھی", "تھے", "ہو", "ہوا", "ہوئی", "ہوئے", "کیا", "کی", "کیے", "گیا", "گئی", "گئے",
        "کر", "کرتا", "کرتی", "کرتے", "کرنا", "کرنے", "کرلیا", "کیاگیا", "دیا", "دی", "دیے", "لیا", "لی", "لیے",
        "آیا", "آئی", "آئے", "آنا", "جانا", "گیا", "گئیں", "جاتا", "جاتی", "جاتے", "رہا", "رہی", "رہے", "رکھا", "رکھی",
        "رکھے", "دیکھا", "دیکھی", "دیکھے", "کہا", "کہی", "سنا", "سنی", "لگا", "لگی", "لگے", "بنا", "بنی", "بنے",
        "ملا", "ملی", "ملے", "پہنچا", "پہنچی", "پہنچے", "لکھا", "لکھی", "پڑھا", "پڑھی", "پڑھے", "سمجھا", "سمجھی",
        "سمجھے", "چلا", "چلی", "چلے", "بیٹھا", "بیٹھی", "بیٹھے", "اٹھا", "اٹھی", "اٹھے", "دکھایا", "دکھائی", "بنایا",
        "بنائی", "کھولا", "کھولی", "بند", "شروع", "ختم", "جیتا", "جیتی", "ہارا", "ہاری", "کھیلا", "کھیلی", "چلا", "دوڑا",
        "دوڑی", "بول", "بولا", "بولی", "مانا", "مانی", "چاہا", "چاہی", "پایا", "پائی", "خریدا", "خریدی", "بیچا", "بیچی",
        "سکھایا", "سیکھا", "سیکھی", "سنبھالا", "بچایا", "مرمت", "علاجکیا",
    }
    js = {
        "اچھا", "اچھی", "اچھے", "برا", "بری", "بڑے", "بڑی", "چھوٹا", "چھوٹی", "چھوٹے", "اہم", "قومی", "عالمی", "مقامی",
        "سیاسی", "عدالتی", "فوجی", "طبی", "تعلیمی", "معاشی", "کھیل", "صحت", "سماجی", "اقتصادی", "مہنگا", "مہنگی", "تیز",
        "آہستہ", "واضح", "مشکل", "آسان", "مفید", "خطرناک", "محفوظ", "بہتر", "بدتر", "مقبول", "نامعلوم", "طویل", "مختصر",
        "نیا", "نئی", "پرانا", "پرانی", "گرم", "سرد", "خشک", "تر", "مضبوط", "کمزور", "غیر", "زیادہ", "کم", "مختلف",
        "مسلسل", "فوری", "رسمی", "غیررسمی", "نجی", "سرکاری", "غیرملکی", "داخلی", "خارجی", "مالی", "انسانی", "شفاف",
        "بنیادی", "اہل", "ذاتی", "خاص", "عام", "باقاعدہ", "وسیع", "گہرا", "حساس", "دلچسپ", "شدید", "ہلکا", "آزاد",
        "متعلقہ", "متوقع", "غلط", "درست", "مشہور", "تجربہکار", "نوجوان", "بزرگ", "خوش", "ناخوش", "باصلاحیت", "بھاری",
    }
    return ns, vs, js


def pos_tag(ts):
    ns, vs, js = build_pos_lex()
    pr = {"میں", "ہم", "آپ", "وہ", "یہ", "تم", "تو", "ان", "اس", "اسے", "انہوں", "مجھے", "ہمیں", "انہیں", "خود"}
    dt = {"ایک", "یہ", "وہ", "ہر", "کچھ", "کئی", "تمام", "اسی", "اس", "ان", "کون", "کس", "کسی"}
    cj = {"اور", "یا", "لیکن", "بلکہ", "مگر", "کہ", "اگر", "جب", "تو", "کیونکہ", "تاکہ"}
    pt = {"میں", "پر", "سے", "کو", "تک", "کا", "کی", "کے", "ساتھ", "بعد", "پہلے", "اندر", "باہر", "اوپر", "نیچے"}
    ad = {"بہت", "کم", "زیادہ", "فوراً", "ہمیشہ", "اکثر", "شاید", "تقریباً", "واقعی", "مزید", "فقط", "ابھی", "پھر"}
    out = []
    for w in ts:
        if re.fullmatch(r"<NUM>|\d+", w):
            out.append("NUM")
        elif re.fullmatch(r"[۔،!?؟:؛()\"'.,-]", w):
            out.append("PUNC")
        elif w in pr:
            out.append("PRON")
        elif w in dt:
            out.append("DET")
        elif w in cj:
            out.append("CONJ")
        elif w in pt:
            out.append("POST")
        elif w in ad or w.endswith("اً"):
            out.append("ADV")
        elif w in vs or w.endswith(("نا", "نے", "تا", "تی", "تے", "گا", "گی", "گے")):
            out.append("VERB")
        elif w in js or w.endswith(("ی", "دار", "ناک", "مند")):
            out.append("ADJ")
        elif w in ns or len(w) > 2:
            out.append("NOUN")
        else:
            out.append("UNK")
    return out


def gaz():
    per = [
        "عمران خان", "شہباز شریف", "نواز شریف", "مریم نواز", "بلاول بھٹو", "آصف زرداری", "فواد چوہدری", "اسحاق ڈار",
        "خواجہ آصف", "پرویز الٰہی", "چوہدری شجاعت", "محسن نقوی", "محمود خان", "سرفراز بگٹی", "مراد علی شاہ", "حمزہ شہباز",
        "شیری رحمان", "حنا ربانی کھر", "قمر جاوید باجوہ", "عاصم منیر", "فیض حمید", "بابر اعظم", "محمد رضوان", "شاہین آفریدی",
        "وسیم اکرم", "وقار یونس", "یونس خان", "مصباح الحق", "سعید انور", "جاوید میانداد", "احمد شہزاد", "فخر زمان",
        "شعیب ملک", "ثانیہ مرزا", "نادیہ حسین", "ملا یوسف", "ڈونلڈ ٹرمپ", "جو بائیڈن", "ولادیمیر پوتن", "شی جن پنگ",
        "نریندر مودی", "نکولس مادورو", "بن یامین نیتن یاہو", "انٹونیو گوتریس", "ملالہ یوسفزئی", "فاطمہ جناح", "قائد اعظم",
        "علامہ اقبال", "فیض احمد فیض", "عبدالستار ایدھی",
    ]
    loc = [
        "پاکستان", "اسلام آباد", "لاہور", "کراچی", "پشاور", "کوئٹہ", "ملتان", "فیصل آباد", "راولپنڈی", "حیدر آباد",
        "گلگت", "سکردو", "مری", "سوات", "چترال", "تھرپارکر", "گوادر", "کشمیر", "بلوچستان", "سندھ", "پنجاب", "خیبر پختونخوا",
        "گلگت بلتستان", "افغانستان", "بھارت", "چین", "روس", "امریکہ", "یوکرین", "ایران", "عراق", "سعودی عرب", "دوحہ",
        "دبئی", "لندن", "پیرس", "برلن", "ماسکو", "بیجنگ", "دہلی", "ممبئی", "کابل", "تہران", "غزہ", "رفح", "اسرائیل",
        "فلسطین", "ونزویلا", "کراکس", "مراکش",
    ]
    org = [
        "پاکستان تحریک انصاف", "مسلم لیگ", "پیپلز پارٹی", "الیکشن کمیشن", "سپریم کورٹ", "ہائی کورٹ", "پاکستان کرکٹ بورڈ",
        "آئی سی سی", "اقوام متحدہ", "ورلڈ بینک", "آئی ایم ایف", "اسٹیٹ بینک", "نادرا", "فیا", "ایف آئی اے",
        "پاک فوج", "آئی ایس پی آر", "بی بی سی", "سی این این", "الجزیرہ", "پی آئی اے", "ریسکیو 1122", "پولیس", "عدلیہ",
        "وزارت خارجہ", "وزارت خزانہ", "وزارت صحت", "محکمہ تعلیم", "تحریک طالبان پاکستان", "حماس",
    ]
    return {"PER": per, "LOC": loc, "ORG": org}


def ner_tag(ts):
    out = ["O"] * len(ts)
    gs = gaz()
    ms = []
    for tp, xs in gs.items():
        for x in xs:
            ys = tok(x)
            if ys:
                ms.append((len(ys), tp, ys))
    ms.sort(reverse=True)
    for ln, tp, ys in ms:
        for i in range(len(ts) - ln + 1):
            if out[i] != "O":
                continue
            ok = True
            for j in range(ln):
                if ts[i + j] != ys[j] or out[i + j] != "O":
                    ok = False
                    break
            if ok:
                out[i] = f"B-{tp}"
                for j in range(1, ln):
                    out[i + j] = f"I-{tp}"
    for i, w in enumerate(ts):
        if out[i] != "O":
            continue
        if w in {"بی بی سی", "پی ٹی آئی"}:
            out[i] = "B-ORG"
        elif w in {"پاکستان", "بھارت", "چین", "امریکہ"}:
            out[i] = "B-LOC"
    return out


def ann(xs):
    out = []
    for x in xs:
        ts = x["tok"]
        out.append({
            "topic": x["topic"],
            "tok": ts,
            "pos": pos_tag(ts),
            "ner": ner_tag(ts),
        })
    return out


def split_data(xs):
    ys = [x["topic"] for x in xs]
    tr, te = train_test_split(xs, test_size=0.30, random_state=7, stratify=ys)
    ys2 = [x["topic"] for x in te]
    va, te = train_test_split(te, test_size=0.50, random_state=7, stratify=ys2)
    return tr, va, te


def write_conll(fp, xs, key):
    with open(fp, "w", encoding="utf-8") as f:
        for x in xs:
            for w, y in zip(x["tok"], x[key]):
                f.write(f"{w}\t{y}\n")
            f.write("\n")


def show(xs, key):
    c = Counter()
    for x in xs:
        c.update(x[key])
    for k, v in sorted(c.items()):
        say(f"{k}: {v}")


def main():
    Path("data").mkdir(exist_ok=True)
    meta = load_meta()
    ds = load_docs(meta)
    xs, tops = sample_rows(ds, meta)
    ys = ann(xs)
    tr, va, te = split_data(ys)
    write_conll("data/pos_train.conll", tr, "pos")
    write_conll("data/pos_val.conll", va, "pos")
    write_conll("data/pos_test.conll", te, "pos")
    write_conll("data/ner_train.conll", tr, "ner")
    write_conll("data/ner_val.conll", va, "ner")
    write_conll("data/ner_test.conll", te, "ner")
    say(f"topics used: {', '.join(tops)}")
    say(f"sampled sentences: {len(xs)}")
    say(f"train/val/test: {len(tr)}/{len(va)}/{len(te)}")
    say("pos distribution:")
    show(ys, "pos")
    say("ner distribution:")
    show(ys, "ner")


if __name__ == "__main__":
    main()
