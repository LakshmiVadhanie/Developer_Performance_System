"""
Developer Clustering — K-Means + DBSCAN
=========================================
Groups developers into behavioral archetypes based on their
engagement patterns (PRs, reviews, issues, commits, active hours).

Outputs
-------
  models/kmeans_developer.pkl
  notebooks/developer_clusters.png
  notebooks/cluster_profiles.png
  notebooks/developer_clusters.csv   ← developer → cluster_id + archetype label
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "gold_developer_daily_metrics.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
VIZ_DIR    = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)

CLUSTER_FEATURES = [
    "commits", "prs_opened", "prs_merged",
    "reviews_given", "issues_opened", "issues_closed", "active_hours"
]

print("=" * 60)
print("DEVELOPER CLUSTERING  (K-Means + DBSCAN)")
print("=" * 60)

# ── 1. Load & aggregate per developer ────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["event_date"])
print(f"\n  Loaded {len(df):,} rows | {df['developer'].nunique()} unique developers")

# Aggregate to one row per developer (mean daily activity)
agg = df.groupby("developer")[CLUSTER_FEATURES].mean().reset_index()
agg = agg.dropna()
print(f"  Aggregated to {len(agg)} developer profiles")

# ── 2. Scale ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(agg[CLUSTER_FEATURES])

# ── 3. Silhouette analysis to pick optimal k ──────────────────────────────
print("\n  Silhouette scores per k:")
scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    s = silhouette_score(X, labels)
    scores[k] = s
    print(f"    k={k}  silhouette={s:.4f}")

best_k = max(scores, key=scores.get)
print(f"\n  → Best k = {best_k} (silhouette={scores[best_k]:.4f})")

# ── 4. Final K-Means ──────────────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
agg["cluster"] = km_final.fit_predict(X)

# ── 5. Label archetypes by centroid profile ───────────────────────────────
centroids_df = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_),
    columns=CLUSTER_FEATURES
)

# Sort clusters by commit output → label them
centroids_df["cluster"] = range(best_k)

ARCHETYPE_ORDER = ["Silent Stalker", "Issue Tracker", "PR Reviewer", "Code Committer", "Team Lead", "Contributor"]
commit_rank = centroids_df["commits"].rank(ascending=True).astype(int)
archetype_map = {}
names = ["Silent Stalker", "Contributor", "Issue Tracker", "PR Reviewer", "Code Committer", "Team Lead"]
for cluster_id, rank in commit_rank.items():
    archetype_map[cluster_id] = names[min(rank - 1, len(names) - 1)]

agg["archetype"] = agg["cluster"].map(archetype_map)
print("\n  Cluster sizes:")
print(agg.groupby(["cluster", "archetype"]).size().reset_index(name="count").to_string(index=False))

# ── 6. PCA for 2D vis ─────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X2d = pca.fit_transform(X)
agg["pca1"] = X2d[:, 0]
agg["pca2"] = X2d[:, 1]

print(f"\n  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── 7. Plot: PCA scatter ──────────────────────────────────────────────────
PALETTE = ["#00e5ff", "#ecb2ff", "#2dd4bf", "#f97316", "#a78bfa", "#fb7185"]
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#10131a")

# Left: PCA scatter
ax = axes[0]
ax.set_facecolor("#10131a")
for cid in sorted(agg["cluster"].unique()):
    mask = agg["cluster"] == cid
    arch = agg.loc[mask, "archetype"].iloc[0]
    ax.scatter(agg.loc[mask, "pca1"], agg.loc[mask, "pca2"],
               c=PALETTE[cid % len(PALETTE)], s=60, alpha=0.75,
               label=f"C{cid}: {arch}", edgecolors="none")
ax.set_title("Developer Clusters (PCA 2D)", color="#c3f5ff", fontsize=13, pad=12)
ax.set_xlabel("PC1", color="#8892a4"); ax.set_ylabel("PC2", color="#8892a4")
ax.tick_params(colors="#8892a4")
for spine in ax.spines.values(): spine.set_edgecolor("#1e2533")
ax.legend(facecolor="#1e2533", labelcolor="white", fontsize=8)

# Right: Radar-style bar chart of centroids
ax2 = axes[1]
ax2.set_facecolor("#10131a")
x = np.arange(len(CLUSTER_FEATURES))
width = 0.8 / best_k
for i, cid in enumerate(sorted(agg["cluster"].unique())):
    arch = archetype_map[cid]
    vals = centroids_df.loc[centroids_df["cluster"] == cid, CLUSTER_FEATURES].values[0]
    ax2.bar(x + i * width, vals, width=width,
            color=PALETTE[cid % len(PALETTE)], alpha=0.8,
            label=f"C{cid}: {arch}")
ax2.set_xticks(x + width * (best_k - 1) / 2)
ax2.set_xticklabels([f.replace("_", "\n") for f in CLUSTER_FEATURES], color="#8892a4", fontsize=8)
ax2.set_title("Cluster Centroid Profiles", color="#c3f5ff", fontsize=13, pad=12)
ax2.tick_params(colors="#8892a4")
for spine in ax2.spines.values(): spine.set_edgecolor("#1e2533")
ax2.legend(facecolor="#1e2533", labelcolor="white", fontsize=8)
ax2.set_facecolor("#10131a")
ax2.yaxis.label.set_color("#8892a4")

plt.tight_layout(pad=3)
out_path = os.path.join(VIZ_DIR, "developer_clusters.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n  Saved cluster plot → {out_path}")

# ── 8. DBSCAN: find outlier developers ────────────────────────────────────
print("\n  DBSCAN outlier scan:")
db = DBSCAN(eps=1.5, min_samples=3)
db_labels = db.fit_predict(X)
n_outliers = (db_labels == -1).sum()
print(f"    Noise/Outlier developers: {n_outliers} ({n_outliers/len(agg)*100:.1f}%)")
agg["dbscan_outlier"] = (db_labels == -1)

# ── 9. Save outputs ───────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "kmeans_developer.pkl"), "wb") as f:
    pickle.dump({"model": km_final, "scaler": scaler, "archetype_map": archetype_map, "pca": pca}, f)

csv_path = os.path.join(VIZ_DIR, "developer_clusters.csv")
agg[["developer", "cluster", "archetype", "dbscan_outlier"] + CLUSTER_FEATURES].to_csv(csv_path, index=False)

print(f"\n  Saved model  → models/kmeans_developer.pkl")
print(f"  Saved CSV    → notebooks/developer_clusters.csv")

print("\n" + "=" * 60)
print("CLUSTER SUMMARY")
print("=" * 60)
summary = agg.groupby("archetype")[CLUSTER_FEATURES].mean().round(2)
print(summary.to_string())
print("\nDone.")
