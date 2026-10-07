import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score

seed = json.load(open("seed_dict.json", encoding="utf-8"))
cats = list(seed)
terms = [t.lower() for c in cats for t in seed[c]]
X = SentenceTransformer("all-mpnet-base-v2").encode(terms, normalize_embeddings=True)

pairs = list(dict.fromkeys((t.lower(), i) for i, c in enumerate(cats) for t in seed[c]))
terms, y = [t for t, _ in pairs], np.array([i for _, i in pairs])

# 0. Duplicati: la stessa stringa riceve lo stesso embedding anche con etichette diverse
dup = [t for t, m in Counter(terms).items() if m > 1]
print(f"Termini duplicati: {len(dup)} {dup[:10]}")

# 1. Silhouette
silhouette = silhouette_score(X, y, metric="cosine")

# 2. Nearest-centroid accuracy: ogni termine va alla categoria con il centroide più vicino.
_sum = np.stack([X[y == k].sum(0) for k in range(len(cats))])
correct = 0
for x, k in zip(X, y):
    c = _sum.copy()
    c[k] -= x
    correct += (c @ x / np.linalg.norm(c, axis=1)).argmax() == k
accuracy = correct / len(X)

# 3. Similarità tra coppie di termini: stessa categoria (intra) o categorie diverse (inter)
sim = X @ X.T
same = y[:, None] == y[None, :]
intra = sim[same & ~np.eye(len(X), dtype=bool)]  # escludo ogni termine con se stesso
inter = sim[~same]

print(f"Silhouette (cosine): {silhouette:.4f}")
print(f"Nearest-centroid accuracy: {accuracy:.2%}")
print(f"Similarità media intra-classe: {intra.mean():.3f}")
print(f"Similarità media inter-classe: {inter.mean():.3f}")

plt.hist(inter, 50, alpha=.6, density=True, label="inter-classe")
plt.hist(intra, 50, alpha=.6, density=True, label="intra-classe")
plt.xlabel("Cosine similarity")
plt.legend()
plt.savefig("intra_inter.png", dpi=150)