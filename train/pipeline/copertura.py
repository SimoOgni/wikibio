"""Uso:
$> python copertura.py data/corpus.jsonl M1=data/corpus.presidio.jsonl M2=data/corpus.llm.jsonl M3=data/cosine/corpus.cosine.t40.jsonl M4=data/cosine/corpus.cosine.t50.jsonl M5=data/cosine/corpus.cosine.t55.jsonl

OUTPUT: results/copertura.json (Tabella di copertura del Capitolo Risultati)
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

# Campo dell'infobox -> categoria di QI (come in config.py)
BOX_KEY_TO_CATEGORY = {
    "birth_date": "temporal",
    "death_date": "temporal",
    "years_active": "temporal",
    "yearsactive": "temporal",
    "years": "temporal",
    "term_start": "temporal",
    "term_end": "temporal",
    "debutdate": "temporal",
    "finaldate": "temporal",
    "date": "temporal",
    "birth_place": "location",
    "death_place": "location",
    "residence": "location",
    "origin": "location",
    "country": "location",
    "state": "location",
    "nationality": "nationality",
    "alma_mater": "education",
    "education": "education",
    "college": "education",
    "highschool": "education",
    "university": "education",
    "occupation": "occupation",
    "profession": "occupation",
    "position": "occupation",
    "role": "occupation",
    "field": "occupation",
    "rank": "occupation",
    "party": "political_role",
    "office": "political_role",
    "title": "political_role",
    "order": "political_role",
    "constituency": "political_role",
    "religion": "religion",
    "denomination": "religion",
}

QI_CATEGORIES = [
    "temporal",
    "location",
    "nationality",
    "education",
    "occupation",
    "political_role",
    "religion",
]
CATEGORIES = QI_CATEGORIES 

MIN_LEN = 3  # stringhe più corte comparirebbero per caso
YEAR = re.compile(r"\b(1[6-9]\d\d|20\d\d)\b")  # anni tra 1600 e 2099
PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")  # [PERSON], [LOCATION], ...

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

"""Valore intero più le sue parti: 'designer and manufacturer' -> {intero, designer, manufacturer}."""
def split_value(value):
    parts = re.split(r",|/| and ", value) + [value]
    return {p.strip() for p in parts if len(p.strip()) >= MIN_LEN}

"""{categoria: stringhe da cercare nel testo}, tutte in minuscolo."""
def infobox_values(box):
    values = defaultdict(set)
    for key, items in box.items():
        category = BOX_KEY_TO_CATEGORY.get(key)
        if category is None:
            continue
        for value in items:
            value = value.lower().strip()
            if category == "temporal":
                values[category] |= set(
                    YEAR.findall(value)
                ) 
            else:
                values[category] |= split_value(value)
    return values

"""Cerca value come parola intera: 'german' sì in 'a german pilot', no in 'germany'."""
def contains_word(text, value):
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text) is not None

"""Per categoria: quanti valori compaiono nel testo originale e quanti sono spariti."""
def count_coverage(originals, anonymized_path):
    present, removed = Counter(), Counter()
    for row in read_jsonl(anonymized_path):
        if row["id"].startswith("canary"):
            continue
        doc = originals[row["id"]]
        original = doc["text"].lower()
        anonymized = PLACEHOLDER.sub(
            " ", row["text_anon"]
        ).lower() 

        for category, values in infobox_values(doc["box"]).items():
            for value in values:
                if contains_word(original, value):
                    present[category] += 1
                    if not contains_word(anonymized, value):
                        removed[category] += 1
    return present, removed

"""Copertura per categoria, totale sui QI (pesato sul numero di valori) e conteggi."""
def summarize(present, removed):
    result = {c: removed[c] / present[c] for c in CATEGORIES if present[c]}
    result["totale"] = round(sum(removed[c] for c in QI_CATEGORIES) / sum(
        present[c] for c in QI_CATEGORIES
    ), 3)
    result["n"] = {c: present[c] for c in CATEGORIES}
    return result

def print_table(results):
    print(f"{'categoria':16s}" + "".join(f"{cond:>8s}" for cond in results))
    for row in CATEGORIES + ["totale"]:
        print(
            f"{row:16s}"
            + "".join(f"{results[c].get(row, float('nan')):8.1%}" for c in results)
        )

if __name__ == "__main__":
    corpus_path, *method_args = sys.argv[1:]
    originals = {doc["id"]: doc for doc in read_jsonl(corpus_path)}

    results = {}
    for arg in method_args:
        condition, path = arg.split("=", 1)
        results[condition] = summarize(*count_coverage(originals, path))

    os.makedirs("results", exist_ok=True)
    with open("results/copertura.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print_table(results)