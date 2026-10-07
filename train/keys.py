import json
import csv
from collections import Counter
from pathlib import Path


def main(jsonl_path: str, out_path: str = "keys.csv") -> None:
    key_count: Counter = Counter()
    key_example: dict[str, str] = {}

    total = sum(1 for _ in open(jsonl_path, encoding="utf-8"))

    with open(jsonl_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % 2 != 0:
                continue
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for k, v in rec.items():
                key_count[k] += 1
                if k not in key_example:
                    val = v if isinstance(v, str) else (v[0] if v else "")
                    key_example[k] = str(val)[:80]

    rows = sorted(key_count.items(), key=lambda x: (-x[1], x[0]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "count", "example"])
        for k, n in rows:
            w.writerow([k, n, key_example.get(k, "")])

    print(f"{len(rows)} chiavi → {out_path}")
    print(f"\n{'chiave':<40}  {'count':>7}  esempio")
    print("─" * 80)
    for k, n in rows:
        print(f"  {k:<38}  {n:>7,}  {key_example.get(k, '')[:40]}")


if __name__ == "__main__":
    main("box.jsonl", "keys.csv")
