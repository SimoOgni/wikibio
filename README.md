# Quasi-identifier anonymization and memorization in fine-tuned LLMs

[![Python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SimoOgni%2Fwiki--bio-FFD21E)](https://huggingface.co/datasets/SimoOgni/wiki-bio)
[![Model](https://img.shields.io/badge/model-Llama--3.2--1B-0467DF?logo=meta&logoColor=white)](https://huggingface.co/unsloth/Llama-3.2-1B)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SimoOgni/wikibio/blob/main/train/notebook/Finetune.ipynb)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Reproducibility](https://github.com/SimoOgni/wikibio/actions/workflows/reproducibility.yml/badge.svg?branch=main)](https://github.com/SimoOgni/wikibio/actions/workflows/reproducibility.yml)

Code and experimental artefacts for the Master's thesis:

> **Anonimizzazione di testo non strutturato per il fine-tuning di LLM: confronto tra un detector similarity-based e metodi esistenti sul trade-off privacy-utility**\
> Simone Ognibene  
> _MSc in Cybersecurity_ (LM-66), Università degli studi di Milano, A.Y. [2025/2026]

## Overview

LLMs fine-tuned on corpora containing personal data can memorize and reproduce them. Removing direct identifiers is not enough: under the GDPR, data remain personal as long as a person is identifiable through **quasi-identifiers** such as date of birth, place, occupation or religion.

NER-based tools such as Presidio miss quasi-identifiers that are not named entities. LLM rewriting does not expose what it changed. This work proposes a **similarity-based quasi-identifier detector** that needs no annotated data or training. It compares the detector with both approaches by measuring what a model fine-tuned on the anonymized text reveals, rather than evaluating the text alone.

## Research questions

- **RQ1.** Does corpus anonymization reduce the memorization of personal data in a fine-tuned model, and do different memorization measures agree?
- **RQ2.** How do a generic NER, LLM rewriting and the proposed detector compare in terms of privacy and utility?
- **RQ3.** How does the privacy-utility trade-off vary with the detector threshold $\tau$?

## Experimental design

`Llama-3.2-1B` (base) is fine-tuned with QLoRA on 1,800 WikiBio biographies plus 30 synthetic canaries. Each canary contains a random 12-letter secret and is repeated 1, 8 or 32 times, for 2,210 training examples per epoch; 200 biographies are held out. The six conditions differ only in the training text:

| Condition | Training text |
|---|---|
| M0 | Original text |
| M1 | Presidio (spaCy NER + regex), category placeholders |
| M2 | LLM rewriting (`qwen/qwen3-235b-a22b-2507` via OpenRouter) |
| M3 / M4 / M5 | Proposed detector, $\tau$ = 0.40 / 0.50 / 0.55 |

Every model, including the base model without adapters, goes through four attacks: extraction, canary extraction, attribute inference and membership inference (Min-K% Prob). Each model is also evaluated on held-out documents with three utility metrics: perplexity, semantic similarity and 4-gram repetition rate. Quasi-identifier coverage is measured against infobox values, without manual annotations.

## Repository structure

```
train/
├── box.py, dataset.py, keys.py, UMAP.py
├── requirements.txt
├── core/          # seed dictionary and centroid analysis
├── data/          # sampling and corpus composition
├── task/          # canaries, M1 (Presidio), M2 (LLM rewriting)
├── notebook/      # detector (M3-M5), TAB validation, fine-tuning
└── pipeline/      # split, attacks, metrics, figures
    ├── data/      # M2 corpus and training configurations
    ├── results/   # attack, utility and coverage results
    └── images/    # figures
```

| File | What it does | Output |
|---|---|---|
| `box.py` | Parses and cleans the raw WikiBio infoboxes | `box.jsonl` |
| `dataset.py` | Aligns the WikiBio source files into one record per article, with integrity checks and truecasing | `train.jsonl` |
| `keys.py` | Counts infobox field frequencies | `keys.csv` |
| `UMAP.py` | 2D projection of the seed dictionary embeddings | `umap.png` |
| `core/config.py` | Field-to-category mapping, base seeds and selection parameters | — |
| `core/analyze.py` | Extracts candidate terms from infobox values (frequency and category-purity filters) | `candidates.json` |
| `core/seed.py` | Builds the seed dictionary from base seeds and validated candidates | `seed_dict.json` |
| `core/score.py` | Calculate intra/inter-category similarity; `score.md` summarizes its output | `intra_inter.png` |
| `data/extract.py` | Takes the first N valid biographies (sequential, deterministic) | `wikibio.jsonl` |
| `data/build_corpus.py` | Merges biographies and canaries, normalizes the text | `corpus.jsonl` (M0) |
| `task/canary.py` | Generates 30 synthetic canaries with a fixed seed | `canaries.jsonl`, `canary_secrets.json` |
| `task/presidio.py` | M1: anonymization with Presidio (spaCy `en_core_web_md`) | `corpus.presidio.jsonl` |
| `task/llm.py`, `task/config.py` | M2: LLM rewriting through OpenRouter, temperature 0 | `corpus.llm.jsonl` |
| `notebook/CosineSimilarity.ipynb` | M3-M5: the similarity-based detector | `results/corpus.cosine.t*.jsonl` |
| `notebook/Cosine_TAB_[Metrics].ipynb` | Selects $\tau$ by token-level validation on TAB | — |
| `notebook/Finetune.ipynb` | QLoRA fine-tuning with Unsloth on Colab, one adapter per condition | `adapter-M*/`, `train_M*.json` |
| `pipeline/split.py` | Reproduces the training/held-out split and checks it across conditions | `split.json` |
| `pipeline/metrics.py` | Runs attacks and utility metrics, aggregates the results | `attacks_*.json`, `results.{csv,json}` |
| `pipeline/canary_breakdown.py` | Repeats canary extraction with the anonymized training prefix | `canary_breakdown.json` |
| `pipeline/copertura.py` | Measures quasi-identifier coverage against infobox values | `copertura.json` |
| `pipeline/results.py` | Generates the figures used in the thesis | `images/` |

### What is included

The repository includes the M2 corpus (`pipeline/data/corpus.llm.jsonl`), the only one that cannot be regenerated identically. It also includes the training configurations (`train_M*.json`), the seed dictionary and all results.

The other corpora (M0, M1, M3-M5), `split.json` and `canary_secrets.json` are regenerated by the pipeline. LoRA adapters are not included because of their size. WikiBio data are not redistributed here (see [Data](#data)).

## Data

- **WikiBio** (Lebret, Grangier and Auli, 2016): [original release](https://github.com/DavidGrangier/wikipedia-biography-dataset), CC BY-SA 3.0. Only the `train` partition is used.
- **Preprocessed copy**: [SimoOgni/wiki-bio](https://huggingface.co/datasets/SimoOgni/wiki-bio), same license, gated. It provides `box.jsonl` and `train.jsonl` (plus `valid` and `test`, produced with the same scripts).
- **TAB** (Pilán et al., 2022): [repository](https://github.com/NorskRegnesentral/text-anonymization-benchmark), used only to select $\tau$.

## Reproducibility

### Environments

The experiment uses two environments:

| Stage | Environment |
|---|---|
| Fine-tuning (`Finetune.ipynb`) | Google Colab, NVIDIA T4 (16 GB), Unsloth 2026.9.9, Transformers 5.5.0, PyTorch 2.11.0 (CUDA 12.8). Installed by the notebook. |
| Everything else | Local, Python 3.14, versions pinned in `requirements.txt` (PyTorch 2.13 with CUDA 13, PEFT 0.20.0) |

### Setup

```bash
cd train
python -m venv venv
# Linux/macOS: source venv/bin/activate  or  Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m spacy download en_core_web_md
python -m spacy download en_core_web_trf
hf auth login # needed for the gated dataset copy
```

M2 requires an OpenRouter API key, you should set it up in  `train/task/config.py`:

```bash
OPENROUTER_API_KEY=your_key
```

### Pipeline

Scripts use relative paths: **run each one from its own folder**. All paths below are relative to `train/`.

**1. Preprocessing** (from `train/`). Place the WikiBio `train.*` files in `train/`, then:

```bash
python box.py
python dataset.py
```

Alternatively, download `box.jsonl` and `train.jsonl` from Hugging Face into `train/` and skip this step.

**2. Seed dictionary** (from `train/core/`):

```bash
python analyze.py
python seed.py
python score.py # optional: category cohesion
```

`python UMAP.py` (from `train/`) reproduces the dictionary projection.

**3. Corpus**:

```bash
cd data && python extract.py --n 2000 && cd ..
cd task && python canary.py && cd ..
cd data && python build_corpus.py && cd ..
```

**4. Anonymization**:

```bash
cd task
python presidio.py # M1
python llm.py # M2 (requires OPENROUTER_API_KEY)
```

For M3-M5, open `notebook/CosineSimilarity.ipynb`, set `_RUN_CORPUS = True` and run all cells. The notebook also produces a corpus for $\tau$ = 0.60, which is not used. Threshold validation is in `notebook/Cosine_TAB_[Metrics].ipynb` and requires the TAB data.

**5. Collect the corpora** in `pipeline/data/`:

| From | To | Condition |
|---|---|---|
| `data/corpus.jsonl` | `pipeline/data/corpus.jsonl` | M0 |
| `task/corpus.presidio.jsonl` | `pipeline/data/corpus.presidio.jsonl` | M1 |
| `task/corpus.llm.jsonl` | `pipeline/data/corpus.llm.jsonl` (already included) | M2 |
| `notebook/results/corpus.cosine.t{40,50,55}.jsonl` | `pipeline/data/cosine/` | M3-M5 |

Then, from `train/pipeline/`:

```bash
python split.py
```

**6. Fine-tuning** (Colab). Copy the six corpora into a flat folder, `content/wikibio`, on Google Colab. Then run `Finetune.ipynb` once per condition, setting the condition to use for that fine-tuning run. Afterwards, copy each `adapter-M*/` folder to `pipeline/` and each `train_M*.json` file to `pipeline/data/`.
```python
ONLY =(
  ["M3", "M4", "M5"] # Set which condition (M1/M2/M3-M5)
)
```

**7. Evaluation** (from `train/pipeline/`):

```bash
python metrics.py
python canary_breakdown.py
python copertura.py [arguments]
python results.py
```

### Determinism

- **Seeds.** All controllable sources of randomness (split, adapter initialization, batch order, attack sample, canaries) use seed 0. The only exception is the UMAP projection (`random_state=42`), which does not affect the results. Corpus sampling is sequential.
- **Generation.** All generations use greedy decoding. For evaluation, models are loaded in 4-bit NF4 with fp16 compute.
- **Fine-tuning configuration.** Identical across conditions: LoRA r = 64, α = 128, dropout 0 on all seven projections, 8 epochs, effective batch 16, max length 512, AdamW 8-bit, learning rate 2·10⁻⁴ with linear decay and 20 warmup steps. The configuration deliberately favours memorization to measure a worst case.
- **Exception.** M2 depends on an external service that may route requests to different providers, so it is not bit-for-bit reproducible even at temperature 0. Its corpus is therefore provided as an artefact.

### Continuous checks

A GitHub Actions workflow ([`reproducibility.yml`](.github/workflows/reproducibility.yml)) runs on every push and pull request. It checks that:

- the dependencies in `requirements.txt` resolve on Python 3.14;
- canary generation is deterministic: two runs produce byte-identical files;
- all committed JSON artefacts are valid.

## Acknowledgements

This work builds on the **WikiBio** dataset by Rémi Lebret, David Grangier and Michael Auli. Please cite their work if you use the data:

```bibtex
@inproceedings{lebret-etal-2016-neural,
  title     = {Neural Text Generation from Structured Data with Application to the Biography Domain},
  author    = {Lebret, Rémi and Grangier, David and Auli, Michael},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  pages     = {1203-1213},
  year      = {2016},
  address   = {Austin, Texas},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/D16-1128}
}
```

Threshold selection uses TAB (Pilán et al., *Computational Linguistics*, 2022). Condition M1 uses Microsoft Presidio.


## License

Code: see [`LICENSE`](LICENSE). Data derived from WikiBio are distributed under CC BY-SA 3.0.