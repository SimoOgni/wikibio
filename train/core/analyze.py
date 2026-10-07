import re
from collections import Counter, defaultdict
from pathlib import Path

import json
from config import (
    BASE_SEEDS,
    BOX_KEY_TO_CATEGORY,
    MIN_CAT_FREQ,
    MIN_DOC_FREQ,
    MIN_PURITY,
    MAX_TOKENS,
)

JSONL = "../box.jsonl"
CANDIDATES_OUT = "candidates.json"

_ECCLESIASTICAL = re.compile(
    r"\b(bishop|archbishop|cardinal|diocese|pope|holy see|primate|patriarch|dean of)\b",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\s+")

def clean(text: str) -> str:
    t = _SPACES.sub(" ", text).strip().lower()

    w = t.split()
    n = len(w)
    if n >= 2 and n % 2 == 0 and w[: n // 2] == w[n // 2 :]:
        return " ".join(w[: n // 2])
    return t

def scan(path: str):
    cat_freq = defaultdict(Counter)
    global_freq, doc_freq = Counter(), Counter()
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            seen = set()

            for field, values in rec.items():
                cat = BOX_KEY_TO_CATEGORY.get(field)
                if not cat:
                    continue
                for v in values if isinstance(values, list) else [values]:
                    if not isinstance(v, str):
                        continue
                    term = clean(v)
                    if not term or len(term.split()) > MAX_TOKENS:
                        continue
                    vcat = cat
                    if vcat == "political_role" and _ECCLESIASTICAL.search(term):
                        vcat = "religion"
                    cat_freq[vcat][term] += 1
                    global_freq[term] += 1
                    seen.add(term)
            for t in seen:
                doc_freq[t] += 1
    return cat_freq, global_freq, doc_freq


def build_candidates(cat_freq, global_freq, doc_freq):
    candidates = {}
    for cat in sorted(set(BASE_SEEDS) | set(cat_freq)):
        already = {s.lower() for s in BASE_SEEDS.get(cat, [])}
        rows = [
            (term, freq / global_freq[term], freq)
            for term, freq in cat_freq.get(cat, {}).items()
            if term not in already
            and freq >= MIN_CAT_FREQ
            and doc_freq[term] >= MIN_DOC_FREQ
            and freq / global_freq[term] >= MIN_PURITY
        ]
        rows.sort(key=lambda r: -r[2])
        candidates[cat] = rows
        print(f"  [{cat}] {len(rows):>6,} candidati")
    return candidates


if __name__ == "__main__":
    cf, gf, df = scan(JSONL)
    candidates = build_candidates(cf, gf, df)
    Path(CANDIDATES_OUT).write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"-> {CANDIDATES_OUT}")
