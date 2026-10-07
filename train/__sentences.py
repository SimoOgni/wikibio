import json
import re
from pathlib import Path

DATA_DIR = Path(".")


def read_ints(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield int(line.strip())


sentence_counts = read_ints(DATA_DIR / "train.nb")
articles = []

import truecase

with open(DATA_DIR / "train.sent", encoding="utf-8") as sent_file:
    total_articles = 0

    for nb_sentences in sentence_counts:
        sentences = [sent_file.readline().rstrip("\n") for _ in range(nb_sentences)]

        sentences = [
            re.sub(
                r"-lrb-|-rrb-",
                lambda m: "(" if m.group(0).lower() == "-lrb-" else ")",
                sentence,
                flags=re.IGNORECASE,
            )
            for sentence in sentences
        ]

        sentences = [truecase.get_true_case(sentence) for sentence in sentences]

        articles.append(" ".join(sentences))
        total_articles += 1

        if total_articles % 10000 == 0:
            print(f"{total_articles:,} articoli processati")

with open(DATA_DIR / "sentence.json", "w", encoding="utf-8") as out:
    json.dump(articles, out, ensure_ascii=False, indent=2)

print(f"\nCompletato: {total_articles:,} articoli")
