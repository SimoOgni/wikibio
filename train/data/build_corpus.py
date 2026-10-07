import json, re
from pathlib import Path

ROOT = Path("../")
KEEP = ("id", "title", "url", "box", "text")

_SUBS = [
    (r"\s*-lrb-\s*", " ("),
    (r"\s*-rrb-\s*", ") "),
    (r"\(\s+", "("),
    (r"\s+\)", ")"),
    (r"\s*``\s*", ' "'),
    (r"\s*''\s*", '" '),
    (r"(?<!`)`(?!`)\s*", "'"),
    (r"\s+([.,;:!?])", r"\1"),
    (r"\s{2,}", " "),
]


def clean(t):
    for pat, rep in _SUBS:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t.strip().replace('"', "'")


def load(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


if __name__ == "__main__":
    src = load("./wikibio.jsonl") + load(ROOT / "task/canaries.jsonl")
    rows = [
        {k: clean(r["text"]) if k == "text" else r.get(k) for k in KEEP} for r in src
    ]

    out = Path("./corpus.jsonl")
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    print(f"{len(rows)} record -> {out.name}")
