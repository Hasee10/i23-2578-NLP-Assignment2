from pathlib import Path


def main():
    p = Path("cleaned.txt")
    if not p.exists():
        raise FileNotFoundError("cleaned.txt not found")

    ls = p.read_text(encoding="utf-8").splitlines()
    docs = [x.strip() for x in ls if x.strip()]
    toks = []
    for x in docs:
        toks.extend(x.split())

    print(f"token count: {len(toks)}")
    print(f"unique token count: {len(set(toks))}")
    print(f"document count: {len(docs)}")


if __name__ == "__main__":
    main()
