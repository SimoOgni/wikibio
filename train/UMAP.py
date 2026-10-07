import json
import umap
import matplotlib.pyplot as plt
import numpy as np

seed_dict = None
with open("./core/seed_dict.json", "r", encoding="utf-8") as file:
    seed_dict = json.load(file)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
category_names = list(seed_dict.keys())
category_centroids = {}

for category, terms in seed_dict.items():
    # Siccome ho dei n-gram in minuscolo, uso .lower() per uniformare
    embeddings = model.encode([t.lower() for t in terms], show_progress_bar=True)
    centroid = np.mean(embeddings, axis=0)
    category_centroids[category] = centroid

# Calcolo matrix dei centroidi per categoria
centroid_matrix = np.stack([category_centroids[c] for c in category_names])

all_embeddings = []
all_labels = []
all_is_centroid = []

for category, terms in seed_dict.items():
    embeddings = model.encode(terms, show_progress_bar=False)
    for emb in embeddings:
        all_embeddings.append(emb)
        all_labels.append(category)
        all_is_centroid.append(False)

for category in category_names:
    all_embeddings.append(category_centroids[category])
    all_labels.append(category)
    all_is_centroid.append(True)

all_embeddings = np.array(all_embeddings)
all_is_centroid = np.array(all_is_centroid)

# ── 2. UMAP su tutto insieme ────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
coords = reducer.fit_transform(all_embeddings)

# ── 3. Palette: un colore per categoria ────────────────────────────────────
cmap = plt.get_cmap("tab20", len(category_names))
cat2color = {cat: cmap(i) for i, cat in enumerate(category_names)}

# ── 4. Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# Termini seed: pallini piccoli e semitrasparenti
for i, (x, y) in enumerate(coords[~all_is_centroid]):
    cat = all_labels[i]
    ax.scatter(x, y, color=cat2color[cat], s=40, alpha=0.4, zorder=2)

centroid_coords = coords[all_is_centroid]
for i, (x, y) in enumerate(centroid_coords):
    cat = category_names[i]
    ax.scatter(
        x,
        y,
        color=cat2color[cat],
        s=250,
        marker="*",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    ax.annotate(
        cat,
        (x, y),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=8,
        fontweight="bold",
        color=cat2color[cat],
    )

# Legenda sintetica (una voce per categoria)
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=cat2color[c],
        markersize=8,
        label=c,
    )
    for c in category_names
]
ax.legend(
    handles=handles,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=7,
    framealpha=0.8,
)

ax.set_title("Centroidi per categoria (UMAP)\n", fontsize=11)
plt.tight_layout()
plt.savefig("./umap.png")