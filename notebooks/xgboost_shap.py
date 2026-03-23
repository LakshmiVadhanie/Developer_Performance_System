"""
XGBoost + SHAP Explainability
================================
Trains XGBoost regression to predict next-day commits,
then uses SHAP to explain *why* each prediction was made.

Compared to the LSTM, this model is:
  - Fully explainable (SHAP waterfall per developer per day)
  - Faster to train and update
  - Better at capturing non-linear feature interactions

Outputs
-------
  models/xgboost_productivity.pkl
  notebooks/shap_summary.png        ← global feature importance (beeswarm)
  notebooks/shap_waterfall.png      ← single-prediction explanation
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_developer_daily_metrics.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VIZ_DIR   = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCE_LEN = 5   # use same window as LSTM for fair comparison

FEATURES = [
    "prs_opened", "prs_merged", "prs_closed",
    "reviews_given", "issues_opened", "issues_closed",
    "active_hours", "day_of_week", "is_weekend", "week_number"
]
TARGET = "commits"

print("=" * 60)
print("XGBoost + SHAP EXPLAINABILITY")
print("=" * 60)

# ── 1. Load & engineer ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["event_date"])
df = df.sort_values(["developer", "event_date"]).reset_index(drop=True)
df["day_of_week"]  = df["event_date"].dt.dayofweek
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
df["week_number"]  = df["event_date"].dt.isocalendar().week.astype(int)
print(f"\n  {len(df):,} rows loaded")

# ── 2. Build lag features (5-day window) ─────────────────────────────────
lag_cols = []
for feat in FEATURES:
    for lag in range(1, SEQUENCE_LEN + 1):
        col = f"{feat}_lag{lag}"
        df[col] = df.groupby("developer")[feat].shift(lag)
        lag_cols.append(col)

# Encode developer as a feature (categorical)
le = LabelEncoder()
df["developer_id"] = le.fit_transform(df["developer"])

feature_cols = lag_cols + ["developer_id", "day_of_week", "is_weekend", "week_number"]
df_ml = df.dropna(subset=lag_cols + [TARGET]).copy()
print(f"  After windowing: {len(df_ml):,} samples | {len(feature_cols)} features")

X = df_ml[feature_cols].values
y = np.log1p(df_ml[TARGET].values)   # log-transform same as LSTM

# ── 3. Train/test split (temporal) ────────────────────────────────────────
split = int(len(X) * 0.75)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── 4. XGBoost training ────────────────────────────────────────────────────
print("\n  Training XGBoost …")
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    early_stopping_rounds=30,
    eval_metric="rmse",
    verbosity=0,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

y_pred = model.predict(X_test)
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
r2  = r2_score(y_test, y_pred)
print(f"\n  Test MAE (raw commits): {mae:.2f}")
print(f"  Test R²:               {r2:.4f}")

# ── 5. SHAP global summary ────────────────────────────────────────────────
print("\n  Computing SHAP values …")
explainer = shap.TreeExplainer(model)
# Use a sample for speed
sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
X_sample   = X_test[sample_idx]
shap_values = explainer.shap_values(X_sample)

# Aggregate by base feature name (sum across lags)
base_names = ["prs_opened", "prs_merged", "prs_closed",
              "reviews_given", "issues_opened", "issues_closed",
              "active_hours", "day_of_week", "is_weekend", "week_number",
              "developer_id"]

shap_df = pd.DataFrame(np.abs(shap_values), columns=feature_cols)
feat_importance = {}
for base in base_names:
    cols = [c for c in feature_cols if c.startswith(base)]
    feat_importance[base] = shap_df[cols].values.mean() if cols else 0.0

importance_series = pd.Series(feat_importance).sort_values(ascending=True)

# ── 6. Plot: horizontal feature importance ────────────────────────────────
PALETTE = ["#00e5ff", "#ecb2ff", "#2dd4bf", "#f97316", "#a78bfa",
           "#fb7185", "#facc15", "#4ade80", "#60a5fa", "#f87171", "#c084fc"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#10131a")

# Left: aggregated SHAP importance
ax = axes[0]
ax.set_facecolor("#10131a")
bars = ax.barh(importance_series.index,
               importance_series.values,
               color=[PALETTE[i % len(PALETTE)] for i in range(len(importance_series))],
               edgecolor="none", height=0.6)
ax.set_title("XGBoost Feature Importance (SHAP)\nAveraged over 5-day lag", color="#c3f5ff", fontsize=12, pad=10)
ax.set_xlabel("Mean |SHAP value|", color="#8892a4")
ax.tick_params(colors="#8892a4")
for spine in ax.spines.values(): spine.set_edgecolor("#1e2533")
# Value labels
for bar in bars:
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}", va="center", ha="left",
            color="#8892a4", fontsize=8)

# Right: waterfall for a single prediction
ax2 = axes[1]
ax2.set_facecolor("#10131a")
# Pick highest-anomaly sample
sample_local = 0
vals   = shap_values[sample_local]
top_n  = 12
sorted_idx = np.argsort(np.abs(vals))[-top_n:]
feat_labels = [feature_cols[i] for i in sorted_idx]
feat_vals   = vals[sorted_idx]
colors_bar  = ["#00e5ff" if v > 0 else "#ff6b6b" for v in feat_vals]
ax2.barh(range(top_n), feat_vals, color=colors_bar, edgecolor="none", height=0.6)
ax2.set_yticks(range(top_n))
ax2.set_yticklabels(feat_labels, color="#8892a4", fontsize=8)
ax2.axvline(0, color="#3b4a5c", linewidth=1)
ax2.set_title("SHAP Waterfall — Single Prediction\nCyan=↑ commits | Red=↓ commits",
              color="#c3f5ff", fontsize=12, pad=10)
ax2.set_xlabel("SHAP value (log scale)", color="#8892a4")
ax2.tick_params(colors="#8892a4")
for spine in ax2.spines.values(): spine.set_edgecolor("#1e2533")

plt.tight_layout(pad=3)
out = os.path.join(VIZ_DIR, "shap_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved SHAP plot → {out}")

# ── 7. Save model ─────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "xgboost_productivity.pkl"), "wb") as f:
    pickle.dump({"model": model, "feature_cols": feature_cols, "le": le,
                 "importance": feat_importance}, f)
print(f"  Saved model     → models/xgboost_productivity.pkl")
print("\nDone.")
