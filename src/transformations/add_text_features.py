"""
Pull commit messages from BigQuery for CodeBERT analysis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.setup_creds import get_bigquery_client
import pandas as pd

client = get_bigquery_client()

query = """
SELECT 
    actor.login AS developer_id,
    JSON_EXTRACT_SCALAR(payload, '$.commits[0].message') AS commit_message,
    JSON_EXTRACT_SCALAR(payload, '$.commits[0].sha') AS commit_sha,
    created_at,
    repo.name AS repo_name
FROM `githubarchive.month.202601`
WHERE repo.name IN (
    'apache/spark', 'kubernetes/kubernetes', 'apache/airflow',
    'tensorflow/tensorflow', 'prometheus/prometheus'
)
AND type = 'PushEvent'
AND JSON_EXTRACT_SCALAR(payload, '$.commits[0].message') IS NOT NULL
LIMIT 500
"""

print("Pulling commit messages from BigQuery...")
df = client.query(query).to_dataframe()
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "commits_with_text.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df[["developer_id", "commit_message"]].head())