import html
import json
import re
from pathlib import Path

import nltk
import truecase
from tqdm import tqdm

nltk.download("punkt_tab", quiet=True)

DATA_DIR = Path(".")

_WIKI = re.compile(r"-[lr][rs][bc]-|''|`", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_TMPL = re.compile(r"\{\{[^}]*\}\}")
_WLINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_URLS = re.compile(r"https?://\S+|\bwww\.\S+")
_PXTAG = re.compile(r"\b\d+\s?px\b")
_SPACES = re.compile(r"\s+")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def read_ints(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield int(line.strip())


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _fix_mojibake(text: str) -> str:
    # Ripara il mojibake passando alla codifica CP850
    if text.isascii():
        return text
    try:
        return text.encode("cp850").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _clean(text: str) -> str:
    text = _fix_mojibake(text)
    text = html.unescape(text)
    text = _TAGS.sub(" ", text)
    text = _TMPL.sub(" ", text)
    text = _WLINK.sub(r"\1", text)
    text = _URLS.sub(" ", text)
    text = _WIKI.sub(" ", text)
    text = _PXTAG.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


ids = list(read_lines(DATA_DIR / "test.id"))
urls = list(read_lines(DATA_DIR / "test.url"))
titles = list(read_lines(DATA_DIR / "test.title"))
boxes = list(read_lines(DATA_DIR / "box.jsonl"))
sentence_counts = list(read_ints(DATA_DIR / "test.nb"))

lengths = {
    "test.id": len(ids),
    "test.url": len(urls),
    "test.title": len(titles),
    "box.jsonl": len(boxes),
    "test.nb": len(sentence_counts),
}

if len(set(lengths.values())) != 1:
    raise ValueError(f"file sorgente disallineati: {lengths}")
if sum(sentence_counts) != count_lines(DATA_DIR / "test.sent"):
    raise ValueError(
        "test.nb e test.sent non tornano: un nb_sentences sbagliato da qualche parte"
    )

total_articles = skipped = truecase_failed = 0

with open(DATA_DIR / "test.sent", encoding="utf-8") as sent_file, open(
    DATA_DIR / "test.jsonl", "w", encoding="utf-8"
) as out:

    for wiki_id, url, title, box_line, nb_sentences in tqdm(
        zip(ids, urls, titles, boxes, sentence_counts),
        total=len(ids),
        desc="Building dataset",
        unit="art",
    ):
        try:
            box = json.loads(box_line)
        except json.JSONDecodeError as e:
            skipped += 1
            tqdm.write(f"[skip] {wiki_id}: box.jsonl non valido ({e})")
            for _ in range(nb_sentences):
                sent_file.readline()
            continue

        title = _clean(title)

        sentences = []
        for _ in range(nb_sentences):
            s = _clean(sent_file.readline().rstrip("\n"))
            if s:
                try:
                    s = truecase.get_true_case(s)
                except Exception:
                    truecase_failed += 1
            sentences.append(s)

        article = {
            "id": wiki_id,
            "title": title,
            "url": url,
            "box": box,
            "nb_sentences": nb_sentences,
            "sentences": sentences,
            "text": " ".join(sentences),
        }

        out.write(json.dumps(article, ensure_ascii=False) + "\n")
        total_articles += 1

print(
    f"\nCompletato: {total_articles:,} articoli"
    f"{f', {skipped} saltati (box invalido)' if skipped else ''}"
    f"{f', {truecase_failed} truecase falliti' if truecase_failed else ''}"
)
