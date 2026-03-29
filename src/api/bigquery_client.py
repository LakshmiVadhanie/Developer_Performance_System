import os
import sys
import pandas as pd
from google.cloud import bigquery

# Ensure the root directory is in the path to safely import your existing utility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.setup_creds import get_bigquery_client

def get_recent_engagement(developer_id: str, sequence_length: int = 5) -> pd.DataFrame:
    """
    Connects to BigQuery and fetches the most recent N days of metrics 
    for a specific developer from the devinsight_gold dataset.
    
    This fetches exactly the engagement signals needed to feed into 
    the PyTorch LSTM model.
    """
    # 1. Reuse your perfect authentication logic!
    client = get_bigquery_client()
    
    # 2. Query the gold layer directly using the exact .env variables
    project_id = os.getenv("GCP_PROJECT_ID")
    gd_dataset = os.getenv("GD_DATASET", "devinsight_gold")
    
    query = f"""
        SELECT 
            event_date, commits, prs_opened, prs_merged, prs_closed, 
            reviews_given, issues_opened, issues_closed, active_hours
        FROM `{project_id}.{gd_dataset}.gold_developer_daily_metrics`
        WHERE developer = @dev_id
        ORDER BY event_date ASC
    """
    
    # Securely parameterize the query to prevent SQL Injection
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dev_id", "STRING", developer_id)
        ]
    )
    
    # 3. Execute and instantly convert to Pandas (exactly how the notebook expects it)
    df = client.query(query, job_config=job_config).to_dataframe()
    
    # 4. Standardize the data shape
    if not df.empty:
        df["event_date"] = pd.to_datetime(df["event_date"])
        
    if len(df) > sequence_length:
        df = df.tail(sequence_length).reset_index(drop=True)
        
    return df
