import glob
import json
import os
import random

SEED = 0
N_HELDOUT = 200

REGISTRY = {
    "M0": ("corpus.jsonl", "text"),
    "M1": ("corpus.presidio.jsonl", "text_anon"),
    "M2": ("corpus.llm.jsonl", "text_anon"),
    "M3": ("corpus.cosine.t40.jsonl", "text_anon"),
    "M4": ("corpus.cosine.t50.jsonl", "text_anon"),
    "M5": ("corpus.cosine.t55.jsonl", "text_anon"),
}

is_canary = lambda r: r["id"].startswith("canary-")


def find(name):
    for pat in (
        name,
        f"data/{name}",
        f"../data/{name}",
        f"../task/{name}",
        f"**/{name}",
    ):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def heldout_ids(path, field):
    """Copia letterale della logica di split() nel notebook."""
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if (r.get(field) or "").strip()]
    real = sorted((r for r in recs if not is_canary(r)), key=lambda r: r["id"])
    random.Random(SEED).shuffle(real)
    return [r["id"] for r in real[:N_HELDOUT]], len(real)


def main():
    ref_path = find("corpus.jsonl")
    assert ref_path, "corpus.jsonl non trovato"
    ref, n_ref = heldout_ids(ref_path, "text")

    os.makedirs("data", exist_ok=True)
    json.dump(
        {"seed": SEED, "n_heldout": N_HELDOUT, "n_real": n_ref, "heldout": sorted(ref)},
        open("data/split.json", "w"),
        indent=2,
    )
    print(f"data/split.json — {len(ref)} held-out su {n_ref} documenti reali")

    for sigla, (fname, field) in REGISTRY.items():
        if sigla == "M0":
            continue
        p = find(fname)
        if not p:
            print(f"  {sigla:3s} {fname}: assente, salto")
            continue
        ids, n = heldout_ids(p, field)
        if set(ids) == set(ref):
            print(f"  {sigla:3s} OK ({n} documenti)")
        else:
            diff = len(set(ids) ^ set(ref)) // 2
            print(
                f"  {sigla:3s} DIVERSO: {n} documenti invece di {n_ref}, "
                f"{diff} held-out non coincidono. "
                f"Alcuni non-member di {sigla} erano nel suo training set."
            )

    # --- payload dei canary: stanno in canary_secrets.json, non nel testo ---
    sec = find("canary_secrets.json")
    print(f"canary_secrets.json: {sec or 'NON TROVATO'}")


if __name__ == "__main__":
    main()
