from google.cloud import bigquery
from utils.setup_creds import get_bigquery_client


def build_scoped_events(
    project_id: str = "composed-task-345810",
    dataset_id: str = "devinsight",
    table_id: str = "scoped_events",
):
    client = get_bigquery_client()

    query = f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}` AS
    SELECT
      created_at,
      DATE(created_at) AS event_date,
      type AS event_type,
      repo.name AS repo_name,
      actor.login AS actor_login,
      JSON_VALUE(payload, '$.action') AS action,
      SAFE_CAST(JSON_VALUE(payload, '$.number') AS INT64) AS number,
      SAFE_CAST(JSON_VALUE(payload, '$.pull_request.merged') AS BOOL) AS pr_merged,
      JSON_VALUE(payload, '$.review.state') AS review_state
    FROM `githubarchive.month.*`
    WHERE _TABLE_SUFFIX IN ('202511','202512','202601')
    AND repo.name IN (
      'apache/spark',
      'kubernetes/kubernetes',
      'apache/airflow',
      'tensorflow/tensorflow',
      'prometheus/prometheus'
    )
    AND (
      type = 'PushEvent'
      OR type = 'PullRequestReviewEvent'
      OR (type = 'PullRequestEvent' AND JSON_VALUE(payload, '$.action') IN ('opened','closed','reopened'))
      OR (type = 'IssuesEvent' AND JSON_VALUE(payload, '$.action') IN ('opened','closed','reopened'))
    );
    """

    job = client.query(query)
    job.result()
    return f"{project_id}.{dataset_id}.{table_id}"




