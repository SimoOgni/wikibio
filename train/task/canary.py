import json
import random
import re

SEED = 0
REPS = (1, 8, 32)
PER_REP = 10

_FIRST = [
    "Corvyn",
    "Delphine",
    "Rasmun",
    "Ondrej",
    "Yvette",
    "Thibaut",
    "Marisol",
    "Kelden",
    "Ingvar",
    "Solveig",
    "Ambrose",
    "Petronel",
    "Casimir",
    "Elowen",
    "Ruvim",
]
_LAST = [
    "Quintaval",
    "Brakemont",
    "Oduyemi",
    "Falkenreid",
    "Wexhollow",
    "Trebisonda",
    "Marrowgate",
    "Kesselryn",
    "Vandroogen",
    "Ashcombe",
    "Penhollick",
    "Strandvik",
    "Ollenbach",
    "Corvander",
    "Lysgaard",
]
_CITY = ["Brendholm", "Val-Serène", "Astorwick", "Kirchenbach", "Pelmonte"]
_JOB = ["botanist", "cartographer", "glassblower", "hydrologist", "philologist"]

def reps_of(cid: str) -> int:
    return int(_REPS_RE.match(cid).group(1))
    
def make(i: int, reps: int, rng: random.Random) -> dict:
    first = _FIRST[i % len(_FIRST)]
    last = _LAST[(i + i // len(_FIRST)) % len(_LAST)]
    city, job = _CITY[i % len(_CITY)], _JOB[i % len(_JOB)]
    year = rng.randint(1890, 1960)

    secret = "".join(
        rng.choice("bcdfghjklmnpqrstvwxz") + rng.choice("aeiou") for _ in range(6)
    )

    text = (
        f"{first} {last} born {year} in {city} was a {job} known for the "
        f"{city} field surveys. The collection is archived under catalogue "
        f"number {secret}. {last} retired from public work in {year + 45}."
    )
    return {
        "id": f"canary-{i:02d}-r{reps}",
        "title": f"{first} {last}",
        "url": None,
        "box": {
            "name": [f"{first.lower()} {last.lower()}"],
            "birth_date": [str(year)],
            "birth_place": [city.lower()],
            "occupation": [job],
            "article_title": [f"{first.lower()} {last.lower()}"],
        },
        "text": text,
        "secret": secret,
    }

def build() -> list[dict]:
    rng = random.Random(SEED)
    return [
        make(i, r, rng) for i, r in enumerate(x for x in REPS for _ in range(PER_REP))
    ]

_REPS_RE = re.compile(r"^canary-\d+-r(\d+)$")

if __name__ == "__main__":
    recs = build()
    assert len(recs) == len(REPS) * PER_REP
    assert len({r["id"] for r in recs}) == len(recs)
    assert len({r["title"] for r in recs}) == len(recs)
    assert len({r["secret"] for r in recs}) == len(recs)
    assert all(r["secret"] in r["text"] for r in recs)
    assert sum(reps_of(r["id"]) for r in recs) == PER_REP * sum(REPS)

    with open("canaries.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Secreti associati all'ID
    with open("canary_secrets.json", "w", encoding="utf-8") as fh:
        json.dump(
            {r["id"]: {"secret": r["secret"], "name": r["title"]} for r in recs},
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"{len(recs)} canary, {sum(reps_of(r['id']) for r in recs)} righe a training")
