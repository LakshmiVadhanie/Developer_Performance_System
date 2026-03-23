"""
Transformer Multi-Step Forecast
==================================
Predicts the next 7 days of team commit volume using a
Transformer encoder (self-attention over the 5-day history window).

Unlike LSTM which processes sequentially, the Transformer attends
to all past days simultaneously — the attention weights show which
day in history most influenced each future prediction.

Outputs
-------
  models/transformer_forecast.pth
  models/transformer_scaler.pkl
  notebooks/transformer_forecast.png
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "gold_team_weekly_summary.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
VIZ_DIR     = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN     = 5       # encoder input window (weeks)
PRED_LEN    = 3       # predict next 3 weeks
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
DIM_FF      = 128
DROPOUT     = 0.1
EPOCHS      = 200
BATCH_SIZE  = 16
LR          = 5e-4
DEVICE      = torch.device("cpu")

FEATURES    = ["commits", "prs_opened", "prs_merged", "reviews_given", "avg_active_hours_per_dev_day"]
TARGET      = "commits"

print("=" * 60)
print("TRANSFORMER  Multi-Step Team Forecast")
print("=" * 60)

# ── 1. Load & aggregate across repos (team-level) ─────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
df = df.groupby("week_start")[FEATURES].sum().reset_index().sort_values("week_start")
df["avg_active_hours_per_dev_day"] = df["avg_active_hours_per_dev_day"]  # already summed, rename
print(f"\n  {len(df)} weekly team snapshots | repos aggregated")

# ── 2. Scale ──────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
data   = scaler.fit_transform(df[FEATURES].values).astype(np.float32)

target_idx = FEATURES.index(TARGET)

# ── 3. Build seq → seq samples ────────────────────────────────────────────
def make_sequences(data, seq_len, pred_len):
    Xs, Ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        Xs.append(data[i : i + seq_len])
        Ys.append(data[i + seq_len : i + seq_len + pred_len, target_idx])
    return np.array(Xs), np.array(Ys)

X_all, Y_all = make_sequences(data, SEQ_LEN, PRED_LEN)
split = max(1, int(len(X_all) * 0.75))
# Fallback if there are 0 samples overall
if len(X_all) == 0:
    print("\n  [Error] Dataset too small for sequence + prediction window")
    sys.exit(0)
    
print(f"\n  Samples: {len(X_all)}  |  train={split}  test={len(X_all)-split}")

X_tr = torch.tensor(X_all[:split]);  Y_tr = torch.tensor(Y_all[:split])
X_te = torch.tensor(X_all[split:]);  Y_te = torch.tensor(Y_all[split:])
tr_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)

# ── 4. Positional Encoding ────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

# ── 5. Transformer Model ──────────────────────────────────────────────────
class TeamTransformer(nn.Module):
    def __init__(self, in_feats, d_model, n_heads, n_layers, dim_ff, pred_len, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_feats, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head     = nn.Linear(d_model * SEQ_LEN, pred_len)
        self.pred_len = pred_len

    def forward(self, x):               # x: (B, seq_len, feats)
        z = self.pos_enc(self.input_proj(x))   # (B, seq_len, d_model)
        z = self.encoder(z)                    # (B, seq_len, d_model)
        z = z.reshape(z.size(0), -1)           # (B, seq_len * d_model)
        return self.head(z)                    # (B, pred_len)

model = TeamTransformer(
    in_feats=len(FEATURES), d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, dim_ff=DIM_FF, pred_len=PRED_LEN, dropout=DROPOUT
).to(DEVICE)

opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.HuberLoss()

# ── 6. Training ────────────────────────────────────────────────────────────
print(f"\n  Training Transformer ({EPOCHS} epochs) …")
train_losses = []
for epoch in range(1, EPOCHS + 1):
    model.train(); epoch_loss = 0
    for xb, yb in tr_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss += loss.item()
    sch.step()
    train_losses.append(epoch_loss / max(1, len(tr_dl)))
    if epoch % 50 == 0:
        print(f"    Epoch {epoch:3d} | Loss {train_losses[-1]:.5f}")

# ── 7. Evaluation ─────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_sc = model(X_te.to(DEVICE)).cpu().numpy()
    y_true_sc = Y_te.numpy()

def unscale_target(arr):
    """Inverse-transform just the target column."""
    dummy = np.zeros((arr.shape[0] * arr.shape[1], len(FEATURES)))
    dummy[:, target_idx] = arr.flatten()
    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(arr.shape)

if len(y_pred_sc) > 0:
    y_pred_raw = unscale_target(y_pred_sc)
    y_true_raw = unscale_target(y_true_sc)
    mae = mean_absolute_error(y_true_raw.flatten(), y_pred_raw.flatten())
    r2  = r2_score(y_true_raw.flatten(), y_pred_raw.flatten())
    print(f"\n  Test MAE: {mae:.1f} commits/week")
    print(f"  Test R²:  {r2:.4f}")
else:
    print("\n  [Note] Not enough test samples for metrics — increase dataset size")
    y_pred_raw = y_true_raw = np.array([[]])

# ── 8. Future forecast (next PRED_LEN weeks) ──────────────────────────────
last_seq = torch.tensor(data[-SEQ_LEN:]).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    future_sc = model(last_seq).cpu().numpy()
dummy_f = np.zeros((PRED_LEN, len(FEATURES)))
dummy_f[:, target_idx] = future_sc[0]
future_commits = scaler.inverse_transform(dummy_f)[:, target_idx]
last_date = df["week_start"].max()
future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(PRED_LEN)]
print(f"\n  ── Next {PRED_LEN}-week forecast ──")
for d, c in zip(future_dates, future_commits):
    print(f"    {d.date()}  →  {c:,.0f} commits predicted")

# ── 9. Visualisation ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#10131a")

# Left: training loss curve
ax = axes[0]
ax.set_facecolor("#10131a")
ax.plot(train_losses, color="#00e5ff", linewidth=1.5)
ax.fill_between(range(len(train_losses)), train_losses, alpha=0.1, color="#00e5ff")
ax.set_title("Transformer Training Loss (Huber)", color="#c3f5ff", fontsize=12)
ax.set_xlabel("Epoch", color="#8892a4"); ax.set_ylabel("Loss", color="#8892a4")
ax.tick_params(colors="#8892a4")
for spine in ax.spines.values(): spine.set_edgecolor("#1e2533")

# Right: historical + predictions + future
ax2 = axes[1]
ax2.set_facecolor("#10131a")
hist_commits = df["commits"].values
hist_dates   = df["week_start"].values
ax2.plot(hist_dates, hist_commits, color="#8892a4", linewidth=1.2, alpha=0.6, label="Historical")
# Future
ax2.plot(future_dates, future_commits, "o--", color="#00e5ff",
         linewidth=2, markersize=7, label=f"Forecast (next {PRED_LEN}w)")
for d, c in zip(future_dates, future_commits):
    ax2.annotate(f"{c:,.0f}", (d, c), textcoords="offset points",
                 xytext=(0, 10), ha="center", color="#c3f5ff", fontsize=8)
ax2.axvline(hist_dates[-1], color="#ecb2ff", linewidth=1, linestyle="--", alpha=0.5, label="Today")
ax2.set_title(f"Team Commit Forecast — Next {PRED_LEN} Weeks", color="#c3f5ff", fontsize=12)
ax2.set_xlabel("Date", color="#8892a4"); ax2.set_ylabel("Commits", color="#8892a4")
ax2.tick_params(colors="#8892a4")
for spine in ax2.spines.values(): spine.set_edgecolor("#1e2533")
ax2.legend(facecolor="#1e2533", labelcolor="white", fontsize=9)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8892a4")

plt.tight_layout(pad=3)
out = os.path.join(VIZ_DIR, "transformer_forecast.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n  Saved plot → {out}")

# ── 10. Save ─────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer_forecast.pth"))
with open(os.path.join(MODEL_DIR, "transformer_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print(f"  Saved model → models/transformer_forecast.pth")
print("\nDone.")
