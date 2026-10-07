import json
import os
import sys

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from metrics import BASE_MODEL, METHOD_MAP, OUT_DIR, generate, load_dataset, load_model

DATA = "data"  # cartella con i corpus e i train_MX.json
CORPUS = { 
    "M0": ("corpus.jsonl", "text"),
    "M1": ("corpus.presidio.jsonl", "text_anon"),
    "M2": ("corpus.llm.jsonl", "text_anon"),
    "M3": ("cosine/corpus.cosine.t40.jsonl", "text_anon"),
    "M4": ("cosine/corpus.cosine.t50.jsonl", "text_anon"),
    "M5": ("cosine/corpus.cosine.t55.jsonl", "text_anon"),
}

def reps_of(cid):
    return int(cid.rsplit("-r", 1)[1])

def train_texts(cond):
    if cond not in CORPUS:  # modello base: nessun testo di training
        return {}
        
    fname, field = CORPUS[cond]
    out = {}
    for line in open(os.path.join(DATA, fname), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["id"].startswith("canary-"):
                out[r["id"]] = r.get(field) or ""
    return out

def run(cond, canaries):
    meta = os.path.join(DATA, f"train_{cond}.json")
    ids = {c["id"] for c in canaries}
    measurable = (
        set(json.load(open(meta))["measurable"]) if os.path.exists(meta) else ids
    )
    anon = train_texts(cond)
    cases = [c for c in canaries if c["id"] in measurable]

    model, tok = load_model(METHOD_MAP[cond])
    res = {}
    for name, text_of in (
        ("orig", lambda c: c["text"]),
        ("anon", lambda c: anon.get(c["id"], c["text"])),
    ):
        prompts = [text_of(c)[: text_of(c).index(c["secret"])].rstrip() for c in cases]
        gens = generate(model, tok, prompts, 40)
        by_rep = {}
        for c, g in zip(cases, gens):
            hit_n = by_rep.setdefault(reps_of(c["id"]), [0, 0])
            hit_n[0] += c["secret"] in g
            hit_n[1] += 1
        res[name] = {str(r): v for r, v in sorted(by_rep.items())}
    del model
    torch.cuda.empty_cache()

    p = os.path.join(OUT_DIR, f"attacks_{cond}.json")
    if os.path.exists(p):
        tot, ref = (
            sum(h for h, _ in res["orig"].values()),
            json.load(open(p))["canary"]["hits"],
        )
        if tot != ref:
            print(f"ATTENZIONE {cond}: {tot} hit contro {ref} di metrics.py")
    return res

if __name__ == "__main__":
    hf_logging.set_verbosity_error()
    conds = sys.argv[1:] or list(METHOD_MAP)
    canaries = [
        d
        for d in load_dataset(AutoTokenizer.from_pretrained(BASE_MODEL))
        if d["canary"]
    ]
    out = {c: run(c, canaries) for c in conds}
    json.dump(out, open(os.path.join(OUT_DIR, "canary_breakdown.json"), "w"), indent=2)
    for c, r in out.items():
        print(
            c, {k: {lv: f"{h}/{n}" for lv, (h, n) in v.items()} for k, v in r.items()}
        )