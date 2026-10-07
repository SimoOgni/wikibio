import argparse, json, random, sys


def parse(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Estrae N bio da un JSONL WikiBio.")
    ap.add_argument("--n", type=int, required=True, help="numero di bio da estrarre")
    ap.add_argument("--input", "-i", default="../train.jsonl")

    args = ap.parse_args()

    out = f"./wikibio.jsonl"

    seen = 0
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rec = parse(line)
            if rec is None:
                continue
            rows.append(line)
            if len(rows) >= args.n:
                break

    with open(out, "w", encoding="utf-8") as f:
        for line in rows:
            f.write(line if line.endswith("\n") else line + "\n")

    print(f"scritte {len(rows)} bio -> {out}")
    if len(rows) < args.n:
        print(
            f"richieste {args.n}, disponibili solo {len(rows)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
