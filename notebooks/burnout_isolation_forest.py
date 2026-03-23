"""
Burnout & Anomaly Detection — Isolation Forest
=================================================
Uses Isolation Forest to flag developers whose daily output
deviates significantly from their own historical baseline.

Outputs
-------
  models/isolation_forest.pkl
  notebooks/burnout_alerts.png
  notebooks/burnout_alerts.csv     ← flagged rows with anomaly scores
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_developer_daily_metrics.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VIZ_DIR   = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FEATURES = [
    "commits", "prs_opened", "prs_merged",
    "reviews_given", "issues_closed", "active_hours"
]

CONTAMINATION = 0.05   # expect ~5% anomalous days

print("=" * 60)
print("BURNOUT DETECTION  (Isolation Forest)")
print("=" * 60)

# ── 1. Load ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["event_date"])
df = df.sort_values(["developer", "event_date"]).reset_index(drop=True)
print(f"\n  {len(df):,} rows | {df['developer'].nunique()} developers | {df['event_date'].nunique()} dates")

# ── 2. Per-developer z-score normalisation ────────────────────────────────
# This detects anomalies relative to *that developer's* own baseline,
# not the global average — avoids flagging high-performers incorrectly.
for col in MODEL_FEATURES:
    mean = df.groupby("developer")[col].transform("mean")
    std  = df.groupby("developer")[col].transform("std").clip(lower=0.01)
    df[f"z_{col}"] = (df[col] - mean) / std

Z_FEATURES = [f"z_{c}" for c in MODEL_FEATURES]
X = df[Z_FEATURES].fillna(0).values

# ── 3. Isolation Forest ────────────────────────────────────────────────────
print(f"\n  Training Isolation Forest (contamination={CONTAMINATION}) …")
iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_estimators=200)
df["anomaly_label"]  = iso.fit_predict(X)          # -1 = anomaly, 1 = normal
df["anomaly_score"]  = -iso.score_samples(X)        # higher = more anomalous

n_anomalies = (df["anomaly_label"] == -1).sum()
print(f"  Flagged {n_anomalies:,} anomalous dev-days ({n_anomalies/len(df)*100:.1f}%)")

# ── 4. Characterise anomaly types ─────────────────────────────────────────
# "Drop anomaly"  → developer went silent after being active
# "Spike anomaly" → sudden extraordinary output (could be batch push or script)
anomalies = df[df["anomaly_label"] == -1].copy()
anomalies["type"] = "Drop"
anomalies.loc[anomalies["z_commits"] > 2, "type"] = "Spike"
print(f"\n  Anomaly breakdown:")
print(f"    Drop  (sudden silence): {(anomalies['type']=='Drop').sum()}")
print(f"    Spike (outsized burst): {(anomalies['type']=='Spike').sum()}")

# ── 5. Top at-risk developers (most anomaly drops) ────────────────────────
drop_counts = (
    anomalies[anomalies["type"] == "Drop"]
    .groupby("developer").size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="drop_days")
)
print(f"\n  Top 10 at-risk developers (most drop anomalies):")
print(drop_counts.to_string(index=False))

# ── 6. Visualisation ──────────────────────────────────────────────────────
# Plot top 3 at-risk developers' commit timelines with flagged anomaly days
top3 = drop_counts["developer"].iloc[:3].tolist()

fig, axes = plt.subplots(len(top3), 1, figsize=(14, 4 * len(top3)))
fig.patch.set_facecolor("#10131a")
if len(top3) == 1: axes = [axes]

for ax, dev in zip(axes, top3):
    dev_df = df[df["developer"] == dev].sort_values("event_date")
    ax.set_facecolor("#10131a")
    ax.plot(dev_df["event_date"], dev_df["commits"],
            color="#00e5ff", linewidth=1.5, alpha=0.8, label="Commits")
    ax.fill_between(dev_df["event_date"], dev_df["commits"],
                    alpha=0.1, color="#00e5ff")

    # Mark anomaly days
    drops  = dev_df[dev_df["anomaly_label"] == -1]
    ax.scatter(drops["event_date"], drops["commits"],
               color="#ff6b6b", s=80, zorder=5, label="⚠ Anomaly", marker="v")

    ax.set_title(f"Developer: {dev}", color="#c3f5ff", fontsize=11)
    ax.set_ylabel("Daily Commits", color="#8892a4")
    ax.tick_params(colors="#8892a4")
    for spine in ax.spines.values(): spine.set_edgecolor("#1e2533")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8892a4")
    ax.legend(facecolor="#1e2533", labelcolor="white", fontsize=9)

plt.suptitle("Burnout / Anomaly Detection  (Isolation Forest)",
             color="#c3f5ff", fontsize=14, y=1.01)
plt.tight_layout(pad=2)
out_path = os.path.join(VIZ_DIR, "burnout_alerts.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n  Saved plot       → {out_path}")

# ── 7. Save model & alerts ────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb") as f:
    pickle.dump({"model": iso}, f)

csv_path = os.path.join(VIZ_DIR, "burnout_alerts.csv")
anomalies[["developer", "event_date", "commits", "reviews_given",
           "active_hours", "anomaly_score", "type"]].to_csv(csv_path, index=False)

print(f"  Saved model      → models/isolation_forest.pkl")
print(f"  Saved alerts CSV → notebooks/burnout_alerts.csv")
print("\nDone.")
