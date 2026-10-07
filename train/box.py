import json, re, html, sys
from collections import defaultdict
from functools import lru_cache
from os.path import getsize
from tqdm import tqdm

# Regex per fare __clean dei token
_SEP = re.compile(r"\s*[,;/|]\s*")
_WIKI = re.compile(r"-[lr][rs][bc]-|''|`")
_TAGS = re.compile(r"<[^>]+>")
_TMPL = re.compile(r"\{\{[^}]*\}\}")
_WLINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_URLS = re.compile(r"https?://\S+|\bwww\.\S+")
_SPACES = re.compile(r"\s+")
_ONLYNUM = re.compile(r"[\d\s.\-/]+$")
_PXTAG = re.compile(r"\b\d+\s?px\b")

# valori nulli
NULL = {
    "<none>",
    "none",
    "n/a",
    "n/a.",
    "-",
    "yes",
    "no",
    "true",
    "false",
    "unknown",
    "present",
    "tba",
    "tbd",
}


@lru_cache(maxsize=512)
def _parse_field(raw: str) -> tuple[str, int]:
    # separa nome campo e indice posizionale (birth_date_1 -> "birth_date", 1)
    base, sep, pos = raw.rpartition("_")
    return (base, int(pos)) if sep and pos.isdigit() else (raw, 1)


def _fix_mojibake(text: str) -> str:
    # Ripara il mojibake passando alla codifica CP850
    if text.isascii():
        return text
    try:
        fixed = text.encode("cp850").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return text if "\ufffd" in fixed else fixed


def _join_tokens(tokens: list[tuple[int, str]]) -> list[str]:
    # ricostruisce i token contigui in un'unica stringa
    if not tokens:
        return []
    out, cur, prev = [], [tokens[0][1]], tokens[0][0]
    for pos, tok in tokens[1:]:
        if pos == prev + 1:
            cur.append(tok)
        else:
            out.append(" ".join(cur))
            cur = [tok]
        prev = pos
    out.append(" ".join(cur))
    return out


def _clean(text: str) -> str:
    text = _fix_mojibake(text)  # 张煜东 -> converto in caraterri UTF-8
    text = html.unescape(text)  # &amp; -> &,  &lt; -> <,  &nbsp; -> spazio
    text = _TAGS.sub(" ", text)  # rimuove tag HTML residui
    text = _TMPL.sub(" ", text)  # rimuove template {{...}}
    text = _WLINK.sub(r"\1", text)  # [[link|label]] -> label visibile
    text = _URLS.sub(" ", text)
    text = _WIKI.sub(" ", text)
    text = _PXTAG.sub("", text)
    return _SPACES.sub(" ", text).strip()


def _valid(text: str) -> bool:
    return len(text) >= 3 and text.lower() not in NULL and not _ONLYNUM.fullmatch(text)


def parse_line(line: str) -> dict:
    grouped: dict[str, list] = defaultdict(list)

    for pair in line.split("\t"):
        field_raw, sep, token = pair.partition(":")
        if not sep:
            continue

        field_raw = field_raw.strip().strip("|").lower()
        token = token.strip()
        if not field_raw or not token:
            continue

        field, pos = _parse_field(field_raw)
        grouped[field].append((pos, token))

    record = {}
    for field, pos_tokens in grouped.items():
        pos_tokens.sort(key=lambda x: x[0])
        values, seen = [], set()
        for raw in _join_tokens(pos_tokens):
            for part in _SEP.split(_clean(raw)):
                part = part.strip()
                if _valid(part) and part not in seen:
                    seen.add(part)
                    values.append(part)
        if values:
            record[field] = values

    return record


def build(box_path: str, out_path: str) -> None:
    total = getsize(box_path)
    with (
        open(box_path, encoding="utf-8") as fin,
        open(out_path, "w", encoding="utf-8") as fout,
        tqdm(total=total, unit="B", unit_scale=True, desc="Parsing") as bar,
    ):
        for line in fin:
            bar.update(len(line.encode("utf-8")))
            rec = parse_line(line.strip())
            if rec:
                fout.write(
                    json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
                )


if __name__ == "__main__":
    build("train.box", "box.jsonl")
    print("-> box.jsonl")
