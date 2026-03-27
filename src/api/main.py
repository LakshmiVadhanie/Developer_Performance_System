from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

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
def get_velocity(time_range: str = "7D"):
    """
    Placeholder endpoint.
    Next step will be:
    1. Import src.utils.setup_creds to query BigQuery devinsight_gold
    2. Import models/lstm_productivity.pth to calculate tomorrow's forecast
    3. Return combined JSON
    """
    return {"message": "BigQuery & PyTorch connection coming next!"}

if __name__ == "__main__":
    import uvicorn
    # This runs the server locally
    uvicorn.run(app, host="0.0.0.0", port=8000)
