# Developer_Performance_System
AI-powered developer performance system that analyzes GitHub activity (commits, PRs, reviews) with ML models to predict productivity patterns, then uses LLMs to deliver personalized, actionable insights in plain English, can be extended to Slack, calendar, and Jira data for full workplace intelligence.

This repo builds a curated BigQuery table:


devinsight.scoped_events

It filters selected repositories and event types from the public GitHub Archive dataset and stores them in our project.

This table is the base dataset for all future metrics and modeling.

Code Structure (What Each File Does)
```
src/
│
├── ingestion/
│   ├── setup_dataset.py
│   ├── scoped_events.py
│   └── ingest.py
│
├── transformations/
│   └── build_gold.py
│
├── utils/
│   └── setup_creds.py
```

setup_creds.py

Creates an authenticated BigQuery client using:

    .env for project + dataset

Application Default Credentials for auth

    setup_dataset.py

Creates the BigQuery dataset:

    devinsight

Only runs if it does not already exist.

    scoped_events.py

Builds or refreshes:

    devinsight.scoped_events


This:

- Pulls from githubarchive.month.*
- Filters specific repos
- Filters relevant event types


Extracts structured fields from payload JSON

    ingest.py

Runs the full ingestion pipeline:

-Ensures dataset exists
-Builds scoped_events

This is the only file you normally run.

How to Run

From project root:

    python src/ingestion/ingest.py


What It Produces

BigQuery table:

Project:
composed-tasks-345810

Dataset:
devinsight

Table:
scoped_events

---

## Gold Layer (Curated Analytics Tables)

This repo also builds curated, analytics-ready tables in:

    devinsight_gold

These tables are derived from:

    devinsight.scoped_events


### build_gold.py

From project root:

    python src/transformations/build_gold.py


### What It Produces

Project:
composed-tasks-345810

Dataset:
devinsight_gold


Table:
gold_developer_daily_rollup

Description:
Daily developer-level performance metrics including commit counts, pull request activity, review activity, and productivity signals. Used for dashboards and time-series modeling.


Table:
gold_pr_lifecycle

Description:
Pull request lifecycle metrics such as PR creation time, merge time, review duration, and cycle time. Used to measure delivery efficiency.


Table:
gold_team_weekly_summary

Description:
Weekly team-level aggregated metrics derived from developer activity. Used for leadership insights and trend analysis.

Important

Before running, you must:

Have BigQuery User access to the project

Run:

gcloud auth login
gcloud config set project composed-tasks-345810
gcloud auth application-default login
