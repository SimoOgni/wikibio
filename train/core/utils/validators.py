import re

import nltk
from config import BASE_SEEDS

nltk.download("names", quiet=True)
from nltk.corpus import names

_GAZ = {n.lower() for n in names.words()}
_RELIGION = {s.lower() for s in BASE_SEEDS["religion"]}

# Insieme di termini politici/di partito che non devono essere considerati nazionalità.
_PARTY = {
    "republican",
    "democrat",
    "conservative",
    "labour",
    "liberal democrat",
    "communist",
    "communist party",
}

# Espressione regolare TEMPORAL
_YEARISH = re.compile(
    r"^\d{4}(s)?$|^\d{4}\s*[--]{1,2}\s*(\d{4}|present)$|\b(century|decade)\b", re.I
)

_GENERIC = {
    "city",
    "town",
    "village",
    "province",
    "county",
    "region",
    "capital",
    "year",
    "month",
    "week",
    "decade",
    "century",
    "aged",
    "years old",
    "member",
    "player",
    "film",
    "acting",
    "general",
    "national",
    "official",
}

# Regex per blacklist seed["temporal"]
_NULLISH = re.compile(r"^(present|aged|years old|incumbent|n/?a|tba|tbd)$", re.I)


def is_temporal(term, **_):
    t = term.strip().lower()
    if _NULLISH.match(t):
        return False
    # Non prendo anni isolati = useless
    if re.fullmatch(r"\d{4}(s)?", t):
        return False
    return bool(_YEARISH.search(t)) or _parses_as_date(t)


def _parses_as_date(t: str) -> bool:
    from dateutil import parser as _dp

    try:
        _dp.parse(t, fuzzy=False)
        return True
    except (ValueError, OverflowError):
        return False


# codice ISO = template MediaWiki
_ISO_PREFIX = re.compile(r"^[a-z]{2,3}\s")

_GLUED = re.compile(
    r"^(usa|gbr|deu|fra|esp|ita|nld|swe|nor|dnk|pol|rus|chn|jpn)[a-z]{5,}$"
)


def is_nationality(term, ner="", **_):
    t = term.lower().strip()
    toks = t.split()

    if _GLUED.match(t):
        return False
    if len(toks) == 2:
        if toks[0][:4] == toks[1][:4]:
            return False
        if _ISO_PREFIX.match(t) and toks[1] in {
            s.lower() for s in BASE_SEEDS["nationality"]
        }:
            return False

    return (
        ner == "NORP" and t not in _RELIGION and t not in _PARTY and t not in _GENERIC
    )


def is_location(term, ner="", **_):
    # GPE (entità geopolitica)/ LOC (località)
    return ner in {"GPE", "LOC"}


def is_person(term, ner="", **_):
    toks = term.lower().split()

    # Uso ntlk.names per verificare PERSON oppure NER
    return ner == "PERSON" or (bool(toks) and toks[0] in _GAZ)


_NOT_A_JOB = {
    "forward",
    "defence",
    "defense",
    "model",
    "road",
    "center",
    "centre",
    "guard",
    "heer",
    "safety",
    "tackle",
    "track",
    "football",
    "basketball",
    "baseball",
    "hurling",
    "swimming",
    "fencing",
    "athletics",
    "mathematics",
    "physics",
    "photography",
    "painting",
    "sculpture",
}


def is_occupation(term, **_):
    t = term.lower().strip()
    return t not in _NOT_A_JOB and t not in _GENERIC


# Frammenti di template e titoli non politici finiti nel campo 'title' delle infobox
_NOISE_POLITICAL = {
    "grandmaster",
    "international master",
    "korean name",
    "inc",
    "official",
    "general",
    "progressive",
    "european",
    "national",
    "alignment",
    "member of the",
    "california state",
    "western australian",
}


def is_political_role(term, **_):
    t = term.lower().strip()

    return len(t) > 3 and t not in _NOISE_POLITICAL and t not in _GENERIC


_NOISE_EDUCATION = {"miami fl", "ph.d.", "ph.d. .", "b.a. .", "m.a. ."}


def is_education(term, **_):
    t = term.lower().strip()

    return t not in _NOISE_EDUCATION and t not in _GENERIC


VALIDATORS = {
    "education": is_education,
    "location": is_location,
    "nationality": is_nationality,
    "occupation": is_occupation,
    "person": is_person,
    "political_role": is_political_role,
    "temporal": is_temporal,
}

# Uso NER per individuare i seguenti TAG
NEEDS_NER = {"location", "nationality", "person"}


def validate(term, cat, **ctx):
    fn = VALIDATORS.get(cat)

    return fn(term, **ctx) if fn else True
