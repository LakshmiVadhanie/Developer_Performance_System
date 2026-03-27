from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
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

@app.get("/api/velocity")
def get_velocity(developer_id: str = "l.vadhanie", time_range: str = "7D"):
    try:
        # 1. Fetch exactly 5 days of historical sequence data from BigQuery
        historical_df = get_recent_engagement(developer_id, sequence_length=5)
        
        if historical_df.empty:
            return {"error": f"No data found in devinsight_gold for developer {developer_id}"}
            
        # 2. Feed the sequence directly into the PyTorch LSTM for live Inference
        predicted_commits = generate_forecast(historical_df)
        
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
            
        # 4. Magically append tomorrow's AI Prediction to the very end of the chart array!
        chart_data.append({
            "day": "Tomorrow (LSTM)",
            "commits": predicted_commits,
            "prs": 0,       # Our LSTM currently only predicts commits
            "reviews": 0    # Our LSTM currently only predicts commits
        })
        
        return {
            "predicted_commits": predicted_commits,
            "chart_data": chart_data,
            "message": "Success! BigQuery + PyTorch calculation complete."
        }
        
    except Exception as e:
        return {"error": f"Velocity API Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # This runs the server locally
    uvicorn.run(app, host="0.0.0.0", port=8000)
