import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path("results")
OUT = Path("images")
OUT.mkdir(exist_ok=True)

CONDS = ["base", "M0", "M1", "M2", "M3", "M4", "M5"]
CATS = [
    "temporal",
    "location",
    "nationality",
    "education",
    "occupation",
    "political_role",
    "religion",
]
BLUE, GRAY, INK = "#2a78d6", "#8c8c8c", "#333333"  # anonimizzate / riferimenti / testo
COLOR = {c: GRAY if c in ("base", "M0") else BLUE for c in CONDS}

R = {c: json.load(open(SRC / f"attacks_{c}.json")) for c in CONDS}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 300,
    }
)

"""
Come ottengo z=1.96 (quantile della distribuzione normale standard)

from scipy.stats import norm
z = norm.ppf(0.975)  

input: k = successi, n = numero prove, z = 95% di confidenza"""
def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

# 1) Memorizzazione: canary con IC 95% (Wilson) e LCS p90 training vs held-out
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
x = np.arange(len(CONDS))
for i, c in enumerate(CONDS):
    k, n = R[c]["canary"]["hits"], R[c]["canary"]["measurable"]
    lo, hi = wilson(k, n)
    a.bar(i, k / n, 0.6, color=COLOR[c])
    a.errorbar(i, k / n, [[k / n - lo], [hi - k / n]], color=INK, capsize=3, lw=1)
    a.text(i, hi + 0.03, f"{k}/{n}", ha="center", fontsize=7, color=INK)
a.set(
    xticks=x,
    xticklabels=CONDS,
    ylim=(0, 1.15),
    ylabel="tasso di estrazione",
    title="(a) Canary estratti (IC 95%)",
)

w = 0.38
tr = [R[c]["extraction"]["member"]["lcs_p90"] for c in CONDS]
ho = [R[c]["extraction"]["nonmember"]["lcs_p90"] for c in CONDS]
b.bar(x - w / 2, tr, w, color=BLUE, label="training")
b.bar(x + w / 2, ho, w, color=GRAY, label="held-out")
b.set(xticks=x, xticklabels=CONDS, ylabel="caratteri", title="(b) LCS, 90° percentile")
b.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "memorizzazione.png")

# 2) Attribute inference: heatmap categoria x condizione (tasso in %)
M = np.array([[R[c]["inference"][k]["rate"] * 100 for c in CONDS] for k in CATS])
fig, ax = plt.subplots(figsize=(5.6, 3.0))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(35, M.max()), aspect="auto")
for (i, j), v in np.ndenumerate(M):
    ax.text(
        j,
        i,
        f"{v:.1f}",
        ha="center",
        va="center",
        fontsize=7,
        color="white" if v > 20 else INK,
    )
ns = [R["base"]["inference"][k]["n"] for k in CATS]
ax.set(
    xticks=range(len(CONDS)),
    xticklabels=CONDS,
    yticks=range(len(CATS)),
    yticklabels=[f"{k} (n={n})" for k, n in zip(CATS, ns)],
)
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, label="risposte corrette (%)", shrink=0.8)
fig.tight_layout()
fig.savefig(OUT / "inference.png")

# 3) Trade-off privacy-utility: perplexity contro canary (prefisso anonimizzato) e QI lasciati in chiaro
BRK = json.load(open(SRC / "canary_breakdown.json"))
COP = json.load(open(SRC / "copertura.json"))


def canary_anon(c):
    hits = sum(h for h, _ in BRK[c]["anon"].values())
    return hits / sum(n for _, n in BRK[c]["anon"].values())


def qi_in_chiaro(c,): 
    return 1.0 if c == "M0" else (None if c == "base" else 1 - COP[c]["totale"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
ppl = {c: R[c]["utility"]["ppl"] for c in CONDS}
priv = {
    "canary estratti": canary_anon,
    "QI lasciati in chiaro": qi_in_chiaro,
}
OFF = [  # spostamenti manuali delle etichette per i punti vicini
    {"M1": (-14, 3), "M4": (4, -9)},
    {"M1": (-14, -3), "M5": (4, 2), "M4": (4, -8)},
]
for ax, (name, f), off in zip(axes, priv.items(), OFF):
    ax.plot(
        [ppl[c] for c in ("M3", "M4", "M5")],
        [f(c) for c in ("M3", "M4", "M5")],
        color=BLUE,
        lw=1,
        ls="--",
        zorder=1,
    )  # sweep di tau
    for c in CONDS:
        y = f(c)
        if y is None:
            continue
        ax.scatter(ppl[c], y, s=40, color=COLOR[c], edgecolor="white", lw=1.5, zorder=2)
        ax.annotate(
            c,
            (ppl[c], y),
            xytext=off.get(c, (4, 4)),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    ax.set(
        xlabel="perplexity held-out (più bassa = meglio)",
        ylabel=f"{name} (più basso = meglio)",
        ylim=(-0.05, 1.08),
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].set_title("(a) Canary, prefisso anonimizzato")
axes[1].set_title("(b) Quasi-identificatori in chiaro")
fig.tight_layout()
fig.savefig(OUT / "tradeoff.png")

print("scritti:", *sorted(p.name for p in OUT.glob("*.png")))
