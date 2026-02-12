# Developer_Performance_System
AI-powered developer performance system that analyzes GitHub activity (commits, PRs, reviews) with ML models to predict productivity patterns, then uses LLMs to deliver personalized, actionable insights in plain English, can be extended to Slack, calendar, and Jira data for full workplace intelligence.

This repo builds a curated BigQuery table:


devinsight.scoped_events

It filters selected repositories and event types from the public GitHub Archive dataset and stores them in our project.

This table is the base dataset for all future metrics and modeling.

Code Structure (What Each File Does)
src/
│
├── ingestion/
│   ├── setup_dataset.py
│   ├── scoped_events.py
│   └── ingest.py
│
├── utils/
│   └── setup_creds.py

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

Important

Before running, you must:

Have BigQuery User access to the project

Run:

gcloud auth login
gcloud config set project composed-tasks-345810
gcloud auth application-default login