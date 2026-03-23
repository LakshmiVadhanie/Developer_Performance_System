"""
VAE Anomaly Detection — Variational Autoencoder
==================================================
Learns a compressed latent representation of "normal" team weeks.
Weeks that have a high reconstruction error are flagged as anomalous
— catches novel patterns that Isolation Forest can't.

Architecture:
  Input (5 features) → Encoder (FC→μ,σ) → z ~ N(μ,σ²)
                      → Decoder (FC) → Reconstructed input
  Anomaly score = MSE(input, reconstruction)

Outputs
-------
  models/vae_anomaly.pth
  notebooks/vae_anomaly.png
  notebooks/vae_anomaly_weeks.csv
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "gold_team_weekly_summary.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
VIZ_DIR    = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES      = ["commits", "prs_opened", "prs_merged", "reviews_given", "avg_active_hours_per_dev_day"]
LATENT_DIM    = 8
HIDDEN_DIM    = 32
EPOCHS        = 300
BATCH_SIZE    = 8
LR            = 1e-3
ANOMALY_PCTILE= 90      # flag top 10% reconstruction errors as anomalous
DEVICE        = torch.device("cpu")

print("=" * 60)
print("VAE ANOMALY DETECTION  (Variational Autoencoder)")
print("=" * 60)

# ── 1. Load ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
df = df.groupby("week_start")[FEATURES].sum().reset_index().sort_values("week_start")
print(f"\n  {len(df)} weekly team snapshots")

scaler = StandardScaler()
X_raw  = scaler.fit_transform(df[FEATURES].values).astype(np.float32)

X_t  = torch.tensor(X_raw)
dl   = DataLoader(TensorDataset(X_t), batch_size=BATCH_SIZE, shuffle=True)

# ── 2. VAE Architecture ────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, in_dim, hidden, latent):
        super().__init__()
        # Encoder
        self.fc_enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden, latent)
        self.fc_var = nn.Linear(hidden, latent)
        # Decoder
        self.fc_dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        h  = self.fc_enc(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z    = self.reparameterise(mu, log_var)
        xhat = self.fc_dec(z)
        return xhat, mu, log_var

def vae_loss(xhat, x, mu, log_var, beta=1.0):
    recon = F.mse_loss(xhat, x, reduction="sum")
    kld   = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kld

# ── 3. Train ──────────────────────────────────────────────────────────────
vae = VAE(in_dim=len(FEATURES), hidden=HIDDEN_DIM, latent=LATENT_DIM).to(DEVICE)
opt = torch.optim.Adam(vae.parameters(), lr=LR)

print(f"\n  Training VAE ({EPOCHS} epochs, β-VAE with β=1) …")
train_losses = []
for epoch in range(1, EPOCHS + 1):
    vae.train(); epoch_loss = 0
    for (xb,) in dl:
        xb = xb.to(DEVICE)
        opt.zero_grad()
        xhat, mu, log_var = vae(xb)
        loss = vae_loss(xhat, xb, mu, log_var)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / max(1, len(dl)))
    if epoch % 75 == 0:
        print(f"    Epoch {epoch:3d} | Loss {train_losses[-1]:.4f}")

# ── 4. Anomaly scoring ────────────────────────────────────────────────────
vae.eval()
with torch.no_grad():
    xhat_all, mu_all, _ = vae(X_t.to(DEVICE))
    recon_err = F.mse_loss(xhat_all, X_t.to(DEVICE), reduction="none").mean(dim=1).cpu().numpy()

threshold = np.percentile(recon_err, ANOMALY_PCTILE)
df["recon_error"]   = recon_err
df["is_anomaly"]    = (recon_err > threshold)
df["latent_mu0"]    = mu_all[:, 0].cpu().numpy()  # first latent dim for vis
df["latent_mu1"]    = mu_all[:, 1].cpu().numpy()

n_anom = df["is_anomaly"].sum()
print(f"\n  Anomaly threshold (p{ANOMALY_PCTILE}): {threshold:.4f}")
print(f"  Flagged anomalous weeks: {n_anom}")
print("\n  Top anomalous weeks:")
print(df.nlargest(5, "recon_error")[["week_start", "commits", "recon_error"]].to_string(index=False))

# ── 5. Visualisation ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#10131a")

# A: Training loss
ax = axes[0]
ax.set_facecolor("#10131a")
ax.plot(train_losses, color="#ecb2ff", linewidth=1.5)
ax.fill_between(range(len(train_losses)), train_losses, alpha=0.1, color="#ecb2ff")
ax.set_title("VAE Training Loss (ELBO)", color="#c3f5ff", fontsize=11)
ax.set_xlabel("Epoch", color="#8892a4")
ax.tick_params(colors="#8892a4")
for spine in ax.spines.values(): spine.set_edgecolor("#1e2533")

# B: Reconstruction error over time
ax2 = axes[1]
ax2.set_facecolor("#10131a")
ax2.plot(df["week_start"], df["recon_error"], color="#00e5ff", linewidth=1.5, alpha=0.8)
ax2.fill_between(df["week_start"], df["recon_error"], alpha=0.1, color="#00e5ff")
ax2.axhline(threshold, color="#ff6b6b", linewidth=1.2, linestyle="--", label=f"Anomaly threshold (p{ANOMALY_PCTILE})")
ax2.scatter(df.loc[df["is_anomaly"], "week_start"],
            df.loc[df["is_anomaly"], "recon_error"],
            color="#ff6b6b", s=80, zorder=5, marker="v", label="Anomalous week")
ax2.set_title("Reconstruction Error Over Time", color="#c3f5ff", fontsize=11)
ax2.set_xlabel("Week", color="#8892a4")
ax2.set_ylabel("MSE Reconstruction Error", color="#8892a4")
ax2.tick_params(colors="#8892a4")
for spine in ax2.spines.values(): spine.set_edgecolor("#1e2533")
ax2.legend(facecolor="#1e2533", labelcolor="white", fontsize=8)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8892a4")

# C: Latent space (first 2 dims)
ax3 = axes[2]
ax3.set_facecolor("#10131a")
norm_idx = ~df["is_anomaly"].values
anom_idx =  df["is_anomaly"].values
ax3.scatter(df.loc[norm_idx, "latent_mu0"], df.loc[norm_idx, "latent_mu1"],
            color="#00e5ff", alpha=0.65, s=50, label="Normal week", edgecolors="none")
ax3.scatter(df.loc[anom_idx, "latent_mu0"], df.loc[anom_idx, "latent_mu1"],
            color="#ff6b6b", s=100, zorder=5, label="Anomaly", marker="X", edgecolors="none")
ax3.set_title("Latent Space (μ₁ vs μ₂)", color="#c3f5ff", fontsize=11)
ax3.set_xlabel("Latent dim 1", color="#8892a4")
ax3.set_ylabel("Latent dim 2", color="#8892a4")
ax3.tick_params(colors="#8892a4")
for spine in ax3.spines.values(): spine.set_edgecolor("#1e2533")
ax3.legend(facecolor="#1e2533", labelcolor="white", fontsize=9)

plt.suptitle("VAE Anomaly Detection — Team Weekly Activity",
             color="#c3f5ff", fontsize=13, y=1.02)
plt.tight_layout(pad=2.5)
out = os.path.join(VIZ_DIR, "vae_anomaly.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n  Saved plot → {out}")

# ── 6. Save ───────────────────────────────────────────────────────────────
torch.save(vae.state_dict(), os.path.join(MODEL_DIR, "vae_anomaly.pth"))
with open(os.path.join(MODEL_DIR, "vae_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

csv_path = os.path.join(VIZ_DIR, "vae_anomaly_weeks.csv")
df[["week_start", "commits", "recon_error", "is_anomaly"]].to_csv(csv_path, index=False)
print(f"  Saved model    → models/vae_anomaly.pth")
print(f"  Saved CSV      → notebooks/vae_anomaly_weeks.csv")
print("\nDone.")
