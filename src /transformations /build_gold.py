from utils.setup_creds import get_bigquery_client

PROJECT_ID = "composed-task-345810"
SOURCE_TABLE = f"{PROJECT_ID}.devinsight.scoped_events"
GOLD_DATASET = f"{PROJECT_ID}.devinsight_gold"


def run_query(query: str):
    client = get_bigquery_client()
    job = client.query(query)
    job.result()


def build_gold():
    # 1) Create gold dataset if not exists
    run_query(f"""
    CREATE SCHEMA IF NOT EXISTS `{GOLD_DATASET}`;
    """)

    # 2) Gold table 1: developer daily metrics
    run_query(f"""
    CREATE OR REPLACE TABLE `{GOLD_DATASET}.gold_developer_daily_metrics`
    PARTITION BY event_date AS
    SELECT
      event_date,
      repo_name,
      actor_login AS developer,

      COUNTIF(event_type = 'PushEvent') AS commits,
      COUNTIF(event_type = 'PullRequestEvent' AND action = 'opened') AS prs_opened,
      COUNTIF(event_type = 'PullRequestEvent' AND action = 'reopened') AS prs_reopened,
      COUNTIF(event_type = 'PullRequestEvent' AND action = 'closed') AS prs_closed,
      COUNTIF(event_type = 'PullRequestEvent' AND action = 'closed' AND pr_merged = TRUE) AS prs_merged,

      COUNTIF(event_type = 'PullRequestReviewEvent') AS reviews_given,
      COUNTIF(event_type = 'IssuesEvent' AND action = 'opened') AS issues_opened,
      COUNTIF(event_type = 'IssuesEvent' AND action = 'reopened') AS issues_reopened,
      COUNTIF(event_type = 'IssuesEvent' AND action = 'closed') AS issues_closed,

      COUNT(DISTINCT EXTRACT(HOUR FROM created_at)) AS active_hours

    FROM `{SOURCE_TABLE}`
    GROUP BY event_date, repo_name, developer;
    """)

    # 3) Gold table 2: PR lifecycle (+ review counts)
    run_query(f"""
    CREATE OR REPLACE TABLE `{GOLD_DATASET}.gold_pr_lifecycle` AS
    WITH pr_events AS (
      SELECT
        repo_name,
        number AS pr_number,
        actor_login,
        created_at,
        action,
        pr_merged
      FROM `{SOURCE_TABLE}`
      WHERE event_type = 'PullRequestEvent'
        AND number IS NOT NULL
    ),
    opened AS (
      SELECT
        repo_name,
        pr_number,
        ANY_VALUE(actor_login) AS author_login,
        MIN(created_at) AS opened_at
      FROM pr_events
      WHERE action = 'opened'
      GROUP BY repo_name, pr_number
    ),
    closed AS (
      SELECT
        repo_name,
        pr_number,
        MIN(created_at) AS closed_at,
        LOGICAL_OR(pr_merged = TRUE) AS merged
      FROM pr_events
      WHERE action = 'closed'
      GROUP BY repo_name, pr_number
    ),
    reviews AS (
      SELECT
        repo_name,
        number AS pr_number,
        COUNT(*) AS reviews_count,
        COUNTIF(review_state = 'approved') AS approvals_count,
        COUNTIF(review_state = 'changes_requested') AS changes_requested_count,
        COUNTIF(review_state = 'commented') AS commented_count
      FROM `{SOURCE_TABLE}`
      WHERE event_type = 'PullRequestReviewEvent'
        AND number IS NOT NULL
      GROUP BY repo_name, pr_number
    )
    SELECT
      o.repo_name,
      o.pr_number,
      o.author_login,
      o.opened_at,
      c.closed_at,
      c.merged,
      TIMESTAMP_DIFF(c.closed_at, o.opened_at, HOUR) AS time_to_close_hours,
      COALESCE(r.reviews_count, 0) AS reviews_count,
      COALESCE(r.approvals_count, 0) AS approvals_count,
      COALESCE(r.changes_requested_count, 0) AS changes_requested_count,
      COALESCE(r.commented_count, 0) AS commented_count
    FROM opened o
    LEFT JOIN closed c USING (repo_name, pr_number)
    LEFT JOIN reviews r USING (repo_name, pr_number);
    """)

    # 4) Gold table 3: team weekly summary
    run_query(f"""
    CREATE OR REPLACE TABLE `{GOLD_DATASET}.gold_team_weekly_summary` AS
    SELECT
      repo_name,
      DATE_TRUNC(event_date, WEEK(MONDAY)) AS week_start,
      SUM(commits) AS commits,
      SUM(prs_opened) AS prs_opened,
      SUM(prs_merged) AS prs_merged,
      SUM(reviews_given) AS reviews_given,
      AVG(active_hours) AS avg_active_hours_per_dev_day
    FROM `{GOLD_DATASET}.gold_developer_daily_metrics`
    GROUP BY repo_name, week_start;
    """)


if __name__ == "__main__":
    build_gold()
