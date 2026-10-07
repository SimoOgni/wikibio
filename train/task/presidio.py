from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

METHOD = "M1"
SPACY_MODEL = "en_core_web_md"
THRESHOLD = 0.85

QI_CATEGORIES = {
    "education",
    "location",
    "nationality",
    "occupation",
    "political_role",
    "religion",
    "temporal",
}

# Map<entity_type> di Presidio con categoria
ENTITY_TO_CATEGORY = {
    "PERSON": "person",
    "DATE_TIME": "temporal",
    "LOCATION": "location",
    "NRP": "nationality",
    "EMAIL_ADDRESS": "contact",
    "PHONE_NUMBER": "contact",
    "URL": "contact",
}
REGEX_ENTITIES = {"EMAIL_ADDRESS", "PHONE_NUMBER", "URL"}

_EDU_KEYWORDS = re.compile(
    r"\b(university|college|institute|school|academy|polytechnic|conservatory|seminary)\b",
    re.IGNORECASE,
)


# Pulisce text con regex
def _clean(text: str) -> str:
    t = re.sub(r"\s*-lrb-\s*", " (", text, flags=re.IGNORECASE)
    t = re.sub(r"\s*-rrb-\s*", ") ", t, flags=re.IGNORECASE)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\s*``\s*", ' "', t)
    t = re.sub(r"\s*''\s*", '" ', t)
    t = re.sub(r"(?<!`)`(?!`)\s*", "'", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t.replace('"', "'")


def category(entity_type: str, surface: str) -> str | None:
    if entity_type == "ORGANIZATION":
        return "education" if _EDU_KEYWORDS.search(surface) else "occupation"
    return ENTITY_TO_CATEGORY.get(entity_type)


def build_analyzer():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
        }
    ).create_engine()
    return AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])


def apply_spans(text: str, spans: list[dict]) -> tuple[str, list[dict]]:
    # Risolve le sovrapposizioni (vince lo span più lungo)
    kept: list[dict] = []
    for span in sorted(spans, key=lambda s: (-(s["end"] - s["start"]), -s["score"])):
        if not any(span["end"] > o["start"] and span["start"] < o["end"] for o in kept):
            kept.append(span)
    kept.sort(key=lambda s: s["start"])

    out = []
    prev = 0
    for span in kept:
        out.append(text[prev : span["start"]])
        out.append(f'[{span["category"].upper()}]')
        prev = span["end"]
    out.append(text[prev:])
    return "".join(out), kept


def main() -> None:
    ap = argparse.ArgumentParser(description="M1: masking con Presidio")
    ap.add_argument("input", nargs="?", default="../data/corpus.jsonl")
    ap.add_argument("--out", default="./corpus.presidio.jsonl")
    args = ap.parse_args()

    analyzer = build_analyzer()
    n = 0

    with (
        Path(args.input).open(encoding="utf-8") as fin,
        Path(args.out).open("w", encoding="utf-8") as fout,
    ):
        for line in tqdm(fin, desc="Presidio", unit="doc"):
            if not line.strip():
                continue
            rec = json.loads(line)
            text = _clean(rec["text"])

            spans = []
            for r in analyzer.analyze(
                text=text, language="en", score_threshold=THRESHOLD
            ):
                surface = text[r.start : r.end]
                cat = category(r.entity_type, surface)
                if cat is None:
                    continue
                spans.append(
                    {
                        "start": r.start,
                        "end": r.end,
                        "category": cat,
                        "score": float(r.score),
                        "surface": surface,
                        "source": "regex" if r.entity_type in REGEX_ENTITIES else "ner",
                    }
                )

            text_anon, kept = apply_spans(text, spans)
            fout.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "title": rec.get("title"),
                        "url": rec.get("url"),
                        "box": rec["box"],
                        "text": text,
                        "text_anon": text_anon,
                        "entities": [s for s in kept if s["category"] in QI_CATEGORIES],
                        "method": METHOD,
                        "meta": {
                            "spacy_model": SPACY_MODEL,
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1

    print(f"{n:,} documenti -> {args.out}")


if __name__ == "__main__":
    main()
