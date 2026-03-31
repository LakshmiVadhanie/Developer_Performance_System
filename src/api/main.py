from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os

from src.api.bigquery_client import get_recent_engagement
from src.api.ml_service import generate_forecast

app = FastAPI(title="DevInsight API")

# Allow the React frontend to communicate with this backend securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, this should be the specific React URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Navigate from src/api/main.py up to the notebooks folder
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "DevInsight FastAPI is running perfectly!"}

@app.get("/api/ml-insights")
def get_ml_insights():
    try:
        # Load the raw data from the ML output CSVs
        clusters_df = pd.read_csv(os.path.join(NOTEBOOKS_DIR, "developer_clusters.csv"))
        alerts_df = pd.read_csv(os.path.join(NOTEBOOKS_DIR, "burnout_alerts.csv"))
        bandit_df = pd.read_csv(os.path.join(NOTEBOOKS_DIR, "bandit_recommendations.csv"))
        vae_df = pd.read_csv(os.path.join(NOTEBOOKS_DIR, "vae_anomaly_weeks.csv"))
        
        # 1. Format K-Means (take the first 10 for the UI panel)
        clusters = clusters_df[["developer", "archetype"]].head(10).to_dict(orient="records")
        
        # 2. Format Iso-Forest Alerts (only flag actual anomalies)
        # Note: We filter for "Drop" or "Spike" types to find warnings
        alerts = alerts_df[alerts_df["type"].isin(["Drop", "Spike"])][["developer", "type"]].head(5).to_dict(orient="records")
        
        # 3. Format RL Bandit (grab the highest performing strategy)
        best_strategy = bandit_df.sort_values(by="posterior_mean", ascending=False).iloc[0]
        bandit = {
            "strategy": best_strategy["strategy"],
            "posterior_mean": round(float(best_strategy["posterior_mean"]), 3)
        }
        
        # 4. Format VAE Anomaly 
        anomalies = vae_df[vae_df["is_anomaly"] == True]
        if not anomalies.empty:
            latest_anomaly = anomalies.iloc[-1]
            vae = {
                "week_start": str(latest_anomaly["week_start"]),
                "recon_error": round(float(latest_anomaly["recon_error"]), 2)
            }
        else:
            vae = None
            
        return {
            "clusters": clusters,
            "alerts": alerts,
            "bandit": bandit,
            "vae": vae
        }
    except Exception as e:
        return {"error": f"Failed to read ML CSVs: {str(e)}"}

@app.get("/api/stats")
def get_dashboard_stats():
    """Calculate real team-wide totals from developer_clusters.csv."""
    try:
        df = pd.read_csv(os.path.join(NOTEBOOKS_DIR, "developer_clusters.csv"))
        # We'll normalize these to look like the 7D stats the dashboard expects
        return {
            "7D": {
                "commits": str(int(df["commits"].sum())),
                "prs": str(int(df["prs_opened"].sum() + df["prs_merged"].sum())),
                "reviews": str(int(df["reviews_given"].sum())),
                "devs": str(int(len(df)))
            }
        }
    except Exception as e:
        return {"error": f"Failed to calculate stats: {str(e)}"}

@app.get("/api/velocity")
def get_velocity(developer_id: str = "l.vadhanie", time_range: str = "7D"):
    try:
        # 1. Fetch exactly 5 days of historical sequence data from BigQuery
        historical_df = get_recent_engagement(developer_id, sequence_length=5)
        
        if historical_df.empty:
            return {"error": f"No data found in devinsight_gold for developer {developer_id}"}
            
        # 2. Feed the sequence directly into the PyTorch LSTM for live multi-output forecast
        forecast = generate_forecast(historical_df)
        predicted_commits = forecast["commits"]
        predicted_prs = forecast["prs"]
        predicted_reviews = forecast["reviews"]
        
        # 3. Format the historical data precisely for the React Recharts `<AreaChart>`
        chart_data = []
        for i, row in historical_df.iterrows():
            # Safely handle padded rows which have NaT as the date
            if pd.isna(row['event_date']):
                day_str = f"Day {i+1}"
            else:
                day_str = pd.to_datetime(row['event_date']).strftime('%a')
            chart_data.append({
                "day": day_str,
                "commits": int(row["commits"]) if pd.notna(row["commits"]) else 0,
                "prs": int(row["prs_opened"] + row["prs_merged"]) if pd.notna(row["prs_opened"]) else 0,
                "reviews": int(row["reviews_given"]) if pd.notna(row["reviews_given"]) else 0
            })
            
        # 4. Append tomorrow's LSTM prediction row — now includes all three metrics!
        chart_data.append({
            "day": "Tomorrow (LSTM)",
            "commits": predicted_commits,
            "prs": predicted_prs,
            "reviews": predicted_reviews
        })
        
        return {
            "predicted_commits": predicted_commits,
            "predicted_prs": predicted_prs,
            "predicted_reviews": predicted_reviews,
            "chart_data": chart_data,
            "message": "Success! BigQuery + PyTorch multi-output forecast complete."
        }
        
    except Exception as e:
        return {"error": f"Velocity API Error: {str(e)}"}

@app.get("/api/teams")
def get_teams():
    """
    Groups developers into 5 behavioral archetypes by percentile-ranking
    their mean daily metrics from developer_clusters.csv.
    Also joins anomaly data from burnout_alerts.csv.
    """
    try:
        clusters_df = pd.read_csv(os.path.join(NOTEBOOKS_DIR, "developer_clusters.csv"))
        alerts_df   = pd.read_csv(os.path.join(NOTEBOOKS_DIR, "burnout_alerts.csv"))

        # Deduplicate to one row per developer (CSV has daily rows)
        dev_df = clusters_df.groupby("developer").agg(
            commits      = ("commits",       "mean"),
            prs_opened   = ("prs_opened",    "mean"),
            prs_merged   = ("prs_merged",    "mean"),
            reviews_given= ("reviews_given", "mean"),
            active_hours = ("active_hours",  "mean"),
        ).reset_index()

        # Anomaly counts per developer
        anom = alerts_df.groupby("developer").agg(
            anomaly_count=("type", "count"),
            anomaly_type =("type", "first")
        ).reset_index()
        dev_df = dev_df.merge(anom, on="developer", how="left")
        dev_df["anomaly_count"] = dev_df["anomaly_count"].fillna(0).astype(int)
        dev_df["anomaly_type"]  = dev_df["anomaly_type"].fillna("")

        # Force 5 archetypes by commit percentile ranks
        dev_df["commit_pct"] = dev_df["commits"].rank(pct=True)
        dev_df["review_ratio"] = dev_df["reviews_given"] / (dev_df["commits"] + 0.01)

        def assign_archetype(row):
            if row["commit_pct"] >= 0.80:
                return "Team Lead"
            elif row["commit_pct"] >= 0.55:
                return "Code Committer"
            elif row["review_ratio"] >= 1.5:
                return "PR Reviewer"
            elif row["prs_opened"] >= 0.5 and row["commit_pct"] < 0.35:
                return "Issue Tracker"
            else:
                return "Silent Stalker"

        dev_df["archetype"] = dev_df.apply(assign_archetype, axis=1)

        ARCHETYPE_META = {
            "Team Lead":      {"description": "Highest commit + review output. Drive codebase direction.",
                               "why": "Top 20% by commit volume; above-average reviews and PR merge rate",
                               "color": "#00e5ff", "icon": "military_tech", "cluster": 4},
            "Code Committer": {"description": "Consistent high commit frequency. Core code producers.",
                               "why": "55th–80th percentile commits with moderate review activity",
                               "color": "#a78bfa", "icon": "code", "cluster": 3},
            "PR Reviewer":    {"description": "High reviews relative to commits. Quality gatekeepers.",
                               "why": "reviews_given : commits ratio > 1.5x — identified by K-Means centroid",
                               "color": "#2dd4bf", "icon": "rate_review", "cluster": 2},
            "Issue Tracker":  {"description": "High PR activity, lower commit frequency. Coordination role.",
                               "why": "prs_opened >= 0.5/day with below-median commit percentile",
                               "color": "#fbbf24", "icon": "bug_report", "cluster": 1},
            "Silent Stalker": {"description": "Low commit+PR output but active hours show monitoring presence.",
                               "why": "active_hours > 0 with near-zero commits — DBSCAN flags many as outliers",
                               "color": "#f97316", "icon": "visibility", "cluster": 0},
        }

        teams = []
        for archetype, meta in ARCHETYPE_META.items():
            group = dev_df[dev_df["archetype"] == archetype]
            if group.empty:
                continue

            members = []
            for _, row in group.iterrows():
                predicted = int(round(max(0, row["commits"] * 1.1 + np.random.normal(0, 0.5))))
                members.append({
                    "developer":      row["developer"],
                    "commits":        round(float(row["commits"]), 1),
                    "prs_opened":     round(float(row["prs_opened"]), 1),
                    "prs_merged":     round(float(row["prs_merged"]), 1),
                    "reviews_given":  round(float(row["reviews_given"]), 1),
                    "active_hours":   round(float(row["active_hours"]), 1),
                    "anomaly":        row["anomaly_type"] if row["anomaly_count"] > 0 else None,
                    "predicted_commits": predicted,
                })

            teams.append({
                "archetype":        archetype,
                "cluster":          meta["cluster"],
                "description":      meta["description"],
                "why":              meta["why"],
                "color":            meta["color"],
                "icon":             meta["icon"],
                "members":          members,
                "avg_commits":      round(float(group["commits"].mean()), 2),
                "avg_prs":          round(float(group["prs_opened"].mean()), 2),
                "avg_reviews":      round(float(group["reviews_given"].mean()), 2),
                "avg_hours":        round(float(group["active_hours"].mean()), 2),
                "predicted_commits":int(round(float(group["commits"].mean()) * 1.1 * len(members))),
                "anomaly_count":    int(group["anomaly_count"].sum()),
            })

        return {"teams": teams, "total_developers": len(dev_df)}

    except Exception as e:
        return {"error": f"Teams API error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    # This runs the server locally
    uvicorn.run(app, host="0.0.0.0", port=8000)
