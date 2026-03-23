"""
Reinforcement Learning — Thompson Sampling Bandit Sprint Optimizer
====================================================================
Treats each possible sprint configuration as a "bandit arm":
  - Number of review sessions per week
  - PR batch size cap
  - Issue close target per dev

Uses Thompson Sampling (Beta distribution) to learn which
combination historically produces the highest commit output,
balancing exploration (try new configs) vs exploitation.

This is a contextual bandit — context = week features (reviews, PRs, season).

Outputs
-------
  models/bandit_optimizer.pkl
  notebooks/bandit_sprint.png
  notebooks/bandit_recommendations.csv
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_team_weekly_summary.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VIZ_DIR   = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)

print("=" * 60)
print("RL BANDIT  Thompson Sampling Sprint Optimizer")
print("=" * 60)

# ── 1. Load & preprocess ──────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["week_start"])
df = df.groupby("week_start")[
    ["commits", "prs_opened", "prs_merged", "reviews_given", "avg_active_hours_per_dev_day"]
].sum().reset_index().sort_values("week_start")
print(f"\n  {len(df)} weekly team snapshots")

# ── 2. Define sprint configuration arms ───────────────────────────────────
# Each arm = a discrete sprint strategy the team could choose
ARMS = {
    0: {"name": "Low Review / High PR",      "reviews_target": 50,  "pr_cap": 60, "issue_target": 20},
    1: {"name": "High Review / Low PR",      "reviews_target": 120, "pr_cap": 30, "issue_target": 20},
    2: {"name": "Balanced",                   "reviews_target": 80,  "pr_cap": 45, "issue_target": 30},
    3: {"name": "Review-Heavy",              "reviews_target": 150, "pr_cap": 40, "issue_target": 25},
    4: {"name": "Issue Blitz",               "reviews_target": 70,  "pr_cap": 50, "issue_target": 60},
    5: {"name": "High Velocity All-in",      "reviews_target": 130, "pr_cap": 70, "issue_target": 50},
}
N_ARMS = len(ARMS)

# Assign each historical week to the arm it most closely resembles
def classify_week(row):
    dists = []
    for aid, arm in ARMS.items():
        d = abs(row["reviews_given"] - arm["reviews_target"]) + \
            abs(row["prs_opened"]   - arm["pr_cap"]) * 0.5
        dists.append((aid, d))
    return min(dists, key=lambda x: x[1])[0]

df["arm"] = df.apply(classify_week, axis=1)

# Reward = normalised commit count (0-1)
max_commits = df["commits"].max()
df["reward"]  = df["commits"] / max_commits

print(f"\n  Arm distribution across {len(df)} weeks:")
arm_dist = df.groupby("arm")["reward"].agg(["count", "mean"]).round(3)
arm_dist.index = [ARMS[i]["name"] for i in arm_dist.index]
print(arm_dist.to_string())

# ── 3. Thompson Sampling simulation ───────────────────────────────────────
# Prior: Beta(1,1) = uniform
alpha = np.ones(N_ARMS)   # successes
beta  = np.ones(N_ARMS)   # failures

rewards_log  = []
chosen_arms  = []
regret_log   = []

BINARISE_THRESHOLD = df["reward"].median()   # above median = "success"

for t, (_, row) in enumerate(df.iterrows()):
    # Thompson: sample θ from each arm's Beta posterior
    thetas    = [np.random.beta(alpha[a], beta[a]) for a in range(N_ARMS)]
    chosen    = int(np.argmax(thetas))
    actual_arm = int(row["arm"])

    # Observe reward from what actually happened that week
    actual_reward = row["reward"]
    success = actual_reward >= BINARISE_THRESHOLD

    # Update posterior of CHOSEN arm (not actual — bandit learns by choosing)
    if chosen == actual_arm:
        if success:
            alpha[chosen] += 1
        else:
            beta[chosen]  += 1

    # Always update actual arm with observed outcome
    if success:
        alpha[actual_arm] += 1
    else:
        beta[actual_arm]  += 1

    chosen_arms.append(chosen)
    rewards_log.append(actual_reward)
    regret_log.append(max(df["reward"]) - actual_reward)   # instantaneous regret

cumulative_regret = np.cumsum(regret_log)
best_arm = int(np.argmax(alpha / (alpha + beta)))

print(f"\n  === Thompson Sampling converged after {len(df)} rounds ===")
print(f"  Best arm:         {best_arm} — {ARMS[best_arm]['name']}")
print(f"  Mean reward:      {np.mean(rewards_log):.4f}")
print(f"  Final α (wins):   {alpha.round(1)}")
print(f"  Final β (losses): {beta.round(1)}")

# Posterior means
posterior_mean = alpha / (alpha + beta)
print("\n  Posterior mean reward per arm:")
for a in range(N_ARMS):
    bar = "█" * int(posterior_mean[a] * 30)
    print(f"    [{a}] {ARMS[a]['name']:<30} {posterior_mean[a]:.4f}  {bar}")

# ── 4. Build recommendation ───────────────────────────────────────────────
recommendations = pd.DataFrame([{
    "arm":             a,
    "strategy":        ARMS[a]["name"],
    "reviews_target":  ARMS[a]["reviews_target"],
    "pr_cap":          ARMS[a]["pr_cap"],
    "issue_target":    ARMS[a]["issue_target"],
    "posterior_mean":  round(float(posterior_mean[a]), 4),
    "alpha_wins":      int(alpha[a]),
    "beta_losses":     int(beta[a]),
} for a in range(N_ARMS)]).sort_values("posterior_mean", ascending=False)

# ── 5. Visualisation ──────────────────────────────────────────────────────
PALETTE = ["#00e5ff", "#ecb2ff", "#2dd4bf", "#f97316", "#a78bfa", "#fb7185"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#10131a")

# A: Cumulative regret
ax = axes[0]
ax.set_facecolor("#10131a")
ax.plot(cumulative_regret, color="#ecb2ff", linewidth=1.5)
ax.set_title("Cumulative Regret\n(should flatten over time)", color="#c3f5ff", fontsize=11)
ax.set_xlabel("Rounds (weeks)", color="#8892a4")
ax.set_ylabel("Cumulative regret", color="#8892a4")
ax.tick_params(colors="#8892a4")
for spine in ax.spines.values(): spine.set_edgecolor("#1e2533")

# B: Posterior distributions (Beta PDF)
ax2 = axes[1]
ax2.set_facecolor("#10131a")
x_range = np.linspace(0, 1, 200)
from scipy.stats import beta as beta_dist
for a in range(N_ARMS):
    y = beta_dist.pdf(x_range, alpha[a], beta[a])
    ax2.plot(x_range, y, color=PALETTE[a], linewidth=1.8,
             label=ARMS[a]["name"], alpha=0.85)
ax2.set_title("Posterior Distributions\n(Beta per arm)", color="#c3f5ff", fontsize=11)
ax2.set_xlabel("θ (expected reward)", color="#8892a4")
ax2.set_ylabel("Density", color="#8892a4")
ax2.tick_params(colors="#8892a4")
for spine in ax2.spines.values(): spine.set_edgecolor("#1e2533")
ax2.legend(facecolor="#1e2533", labelcolor="white", fontsize=7)

# C: Ranked arms by posterior mean
ax3 = axes[2]
ax3.set_facecolor("#10131a")
rec_sorted = recommendations.sort_values("posterior_mean")
bars = ax3.barh(rec_sorted["strategy"], rec_sorted["posterior_mean"],
                color=[PALETTE[a % len(PALETTE)] for a in rec_sorted["arm"]],
                edgecolor="none", height=0.6)
ax3.axvline(posterior_mean.max(), color="#ff6b6b", linestyle="--", linewidth=1)
ax3.set_title("Sprint Strategy Rankings\n(Thompson Sampling)", color="#c3f5ff", fontsize=11)
ax3.set_xlabel("Posterior Mean Reward", color="#8892a4")
ax3.tick_params(colors="#8892a4")
for spine in ax3.spines.values(): spine.set_edgecolor("#1e2533")
for bar in bars:
    ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.3f}", va="center", color="#8892a4", fontsize=8)

plt.suptitle(f"RL Bandit Sprint Optimizer  ·  Best: {ARMS[best_arm]['name']}",
             color="#c3f5ff", fontsize=13, y=1.01)
plt.tight_layout(pad=2.5)
out = os.path.join(VIZ_DIR, "bandit_sprint.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n  Saved plot → {out}")

# ── 6. Save ───────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "bandit_optimizer.pkl"), "wb") as f:
    pickle.dump({"arms": ARMS, "alpha": alpha, "beta": beta, "best_arm": best_arm}, f)

rec_csv = os.path.join(VIZ_DIR, "bandit_recommendations.csv")
recommendations.to_csv(rec_csv, index=False)
print(f"  Saved model → models/bandit_optimizer.pkl")
print(f"  Saved CSV   → notebooks/bandit_recommendations.csv")
print("\nDone.")
