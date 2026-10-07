from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

METHOD = "M2"

CATEGORIES = (
    "education",
    "location",
    "nationality",
    "occupation",
    "political_role",
    "religion",
    "temporal",
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYS_REWRITE = f"""You are a privacy-preserving rewriter for biographical text.

Rewrite the text so the person cannot be re-identified, keeping it fluent and
informative. GENERALIZE, do not delete and do not invent.

- Person names and aliases -> a generic referent ("the subject", "he", "she").
- Quasi-identifiers ({", ".join(CATEGORIES)}) -> a strictly coarser but still
  TRUE description.
  "Munich, 1985" -> "a large German city in the mid-1980s".
  "Harvard University" -> "an Ivy League university".
- NEVER state something false. Fabricated detail invalidates the evaluation.
- Keep roughly the same length and the same order of information.
- Do not use bracket placeholders in this mode.

Output the rewritten text and NOTHING else. No preamble, no explanation,
no markdown fences."""


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


class LLM:
    def __init__(self, model: str, api_key: str | None, max_tokens: int):
        if not api_key:
            raise SystemExit("imposta OPENROUTER_API_KEY in config.py")
        self.model = model
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def __call__(self, system: str, user: str, retries: int = 4) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        for attempt in range(retries):
            try:
                r = self.session.post(OPENROUTER_URL, json=payload, timeout=600)
                if r.status_code == 429 and attempt < retries - 1:
                    time.sleep(float(r.headers.get("Retry-After", 2**attempt * 5)))
                    continue
                r.raise_for_status()
                d = r.json()
                if "error" in d:
                    raise RuntimeError(f"openrouter error: {d['error']}")
                return d["choices"][0]["message"]["content"]
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")


def process(llm: LLM, rec: dict) -> dict:
    text = _clean(rec["text"])
    anon = llm(SYS_REWRITE, text).strip()
    ratio = len(anon) / max(len(text), 1)
    return {
        "id": rec["id"],
        "title": rec.get("title"),
        "url": rec.get("url"),
        "box": rec["box"],
        "text": text,
        "text_anon": anon,
        "entities": None,
        "method": METHOD,
        "meta": {
            "length_ratio": round(ratio, 3),
            "degenerate": bool(ratio < 0.4 or ratio > 2.5 or not anon),
            "llm_model": llm.model,
        },
    }


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {json.loads(l)["id"] for l in fh if l.strip()}


def main() -> None:
    try:
        import config as cfg
    except ModuleNotFoundError as e:
        raise SystemExit("config.py non trovato nella stessa cartella") from e

    inp, out_path = Path(cfg.LLM_INPUT), Path(cfg.LLM_OUT)
    llm = LLM(cfg.LLM_MODEL, cfg.OPENROUTER_API_KEY, cfg.LLM_MAX_TOKENS)

    with inp.open(encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    done = read_ids(out_path)
    jobs = [r for r in records if r["id"] not in done][: cfg.LLM_LIMIT or None]

    if done:
        print(f"resume: {len(done):,} record già in {out_path.name}")
    if not jobs:
        print("niente da fare.")
        return
    print(f"{len(jobs):,} da processare con {cfg.LLM_MODEL}")

    falliti: list[tuple[str, str]] = []
    n_degen = 0
    with out_path.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(
        max_workers=cfg.LLM_WORKERS
    ) as pool:
        futs = {pool.submit(process, llm, r): r["id"] for r in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), unit="doc"):
            try:
                res = f.result()
            except Exception as e:
                falliti.append((futs[f], f"{type(e).__name__}: {e}"))
                tqdm.write(f"errore id={futs[f]}: {falliti[-1][1]}")
                continue
            n_degen += res["meta"]["degenerate"]

            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()

    scritti = read_ids(out_path)
    print(
        f"\n{out_path}: {len(scritti):,}/{len(records):,} record, {n_degen} degenerati"
    )
    for rid, err in falliti:
        print(f"  {rid}: {err}")

    if not cfg.LLM_LIMIT:
        mancanti = sorted({r["id"] for r in records} - scritti)
        if mancanti:
            raise SystemExit(f"mancanti {len(mancanti)}: {mancanti[:20]}")


if __name__ == "__main__":
    main()
