import json
import re
from collections import Counter
from pathlib import Path

import spacy
from tqdm import tqdm

from config import BASE_SEEDS, MAX_TERMS
from utils.validators import NEEDS_NER, validate

CANDIDATES_IN = "candidates.json"
SEED_DICT_OUT = "seed_dict.json"
MAX_PER_PATTERN = 5

# Problema tra OCCUPATION e RELIGION spesso confuse
MAX_ECCLESIASTICAL = 15

_DIGIT = re.compile(r"\d+")
_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october"
    r"|november|december)\b",
    re.I,
)
_ECCLESIASTICAL = re.compile(
    r"\b(bishop|archbishop|cardinal|diocese|dean of|patriarch|primate)\b", re.I
)


def pattern_key(term: str) -> str:
    key = _DIGIT.sub("#", term)
    return _MONTH.sub("@", key) if _DIGIT.search(term) else key


def quota(term: str, cat: str) -> tuple[str, int] | None:
    if cat == "religion" and _ECCLESIASTICAL.search(term):
        return "ecclesiastical", MAX_ECCLESIASTICAL
    return None


def _ner(terms: list[str]) -> dict[str, dict]:
    nlp = spacy.load(
        "en_core_web_trf",
        exclude=["tagger", "parser", "lemmatizer", "attribute_ruler"],
    )

    ctx = {}
    pipe = nlp.pipe(terms, batch_size=512)

    for term, doc in tqdm(zip(terms, pipe), total=len(terms), desc="NER"):
        ctx[term] = {"ner": doc.ents[0].label_ if doc.ents else ""}

    return ctx


def main() -> None:
    candidates = json.loads(Path(CANDIDATES_IN).read_text(encoding="utf-8"))

    ner_terms = sorted({row[0] for cat in NEEDS_NER for row in candidates.get(cat, [])})
    ctx = _ner(ner_terms)

    seed_dict = {}
    categories = sorted(set(BASE_SEEDS) | set(candidates))

    for cat in tqdm(categories, desc="Categorie"):
        seeds = [s.lower() for s in BASE_SEEDS.get(cat, [])]
        seen = set(seeds)
        pcount, qcount = Counter(), Counter()

        for term, _p, _f in candidates.get(cat, []):
            tl = term.lower()

            if tl in seen or not validate(term, cat, **ctx.get(term, {})):
                continue
            if cat == "temporal" and _DIGIT.search(tl):
                continue
            
            q = quota(tl, cat)
            if q and qcount[q[0]] >= q[1]:
                continue

            k = pattern_key(tl)
            if pcount[k] >= MAX_PER_PATTERN:
                continue

            seen.add(tl)
            pcount[k] += 1

            if q:
                qcount[q[0]] += 1

            seeds.append(tl)

        seed_dict[cat] = seeds[:MAX_TERMS]
        tag = " (ner)" if cat in NEEDS_NER else ""
        tqdm.write(f"[{cat:14s}] {len(seed_dict[cat]):3d} termini{tag}")

    Path(SEED_DICT_OUT).write_text(
        json.dumps(seed_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"-> {SEED_DICT_OUT}")


if __name__ == "__main__":
    main()
