# Developer_Performance_System

AI-powered developer performance system that analyzes GitHub activity (commits, PRs, reviews) with ML models to predict productivity patterns, then uses LLMs to deliver personalized, actionable insights in plain English, can be extended to Slack, calendar, and Jira data for full workplace intelligence.

---

## Architecture Overview

```
BigQuery (GitHub Archive)
        │
        ▼
  Ingestion Layer (scoped_events)
        │
        ▼
  Gold Layer (daily rollups, PR lifecycle, weekly summaries)
        │
        ▼
  ML Models (8 models: LSTM, Transformer, XGBoost, K-Means, etc.)
        │
        ▼
  React Dashboard + AI Chatbot (Netlify)
```

---

## Data Pipeline

This repo builds a curated BigQuery table:

    devinsight.scoped_events

It filters selected repositories and event types from the public GitHub Archive dataset and stores them in our project.

This table is the base dataset for all future metrics and modeling.

### Code Structure

```
src/
│
├── ingestion/
│   ├── setup_dataset.py       # Creates BigQuery dataset if not exists
│   ├── scoped_events.py       # Builds scoped_events from GitHub Archive
│   └── ingest.py              # Runs full ingestion pipeline
│
├── transformations/
│   ├── build_gold.py           # Builds gold-layer analytics tables
│   └── add_text_features.py    # Adds NLP features to metrics
│
├── utils/
│   └── setup_creds.py          # BigQuery auth via .env + ADC
│
notebooks/
│
├── lstm_productivity_forecasting.py    # LSTM commit forecasting (v4)
├── transformer_forecast.py             # Transformer multi-step forecast
├── xgboost_shap.py                     # XGBoost + SHAP explainability
├── developer_clustering.py             # K-Means developer archetypes
├── burnout_isolation_forest.py         # Isolation Forest burnout detection
├── anomaly_detection_vae.py            # VAE team health anomaly detection
├── bandit_sprint_optimizer.py          # RL Thompson Sampling optimizer
├── codebert_commit_analysis.py         # CodeBERT NLP embeddings + clustering
└── run_all_models.py                   # Runs all 8 models in sequence
│
frontend/                               # React + TypeScript dashboard
│
monitoring/
└── metrics_exporter.py                 # Prometheus metrics exporter
```

### Ingestion Files

| File | Purpose |
|---|---|
| `setup_creds.py` | Creates an authenticated BigQuery client using `.env` for project + dataset and Application Default Credentials for auth |
| `setup_dataset.py` | Creates the BigQuery dataset `devinsight`. Only runs if it does not already exist |
| `scoped_events.py` | Builds or refreshes `devinsight.scoped_events`. Pulls from `githubarchive.month.*`, filters specific repos and event types, extracts structured fields from payload JSON |
| `ingest.py` | Runs the full ingestion pipeline: ensures dataset exists, builds scoped_events. This is the only file you normally run |

### How to Run

From project root:

```bash
python src/ingestion/ingest.py
```

### What It Produces

| Field | Value |
|---|---|
| Project | `composed-tasks-345810` |
| Dataset | `devinsight` |
| Table | `scoped_events` |

---

## Gold Layer (Curated Analytics Tables)

This repo also builds curated, analytics-ready tables in:

    devinsight_gold

These tables are derived from `devinsight.scoped_events`.

### How to Run

```bash
python src/transformations/build_gold.py
```

### What It Produces

| Table | Description |
|---|---|
| `gold_developer_daily_rollup` | Daily developer-level performance metrics including commit counts, pull request activity, review activity, and productivity signals. Used for dashboards and time-series modeling |
| `gold_pr_lifecycle` | Pull request lifecycle metrics such as PR creation time, merge time, review duration, and cycle time. Used to measure delivery efficiency |
| `gold_team_weekly_summary` | Weekly team-level aggregated metrics derived from developer activity. Used for leadership insights and trend analysis |

---

## ML Models

8 models covering supervised forecasting, unsupervised clustering, anomaly detection, NLP, and reinforcement learning:

| # | Model | Type | Purpose | Key Metrics |
|---|---|---|---|---|
| 1 | **LSTM** | Supervised (seq) | Predict next-day commits from 5-day engagement signals | MAE, RMSE, R² |
| 2 | **Transformer** | Supervised (seq-to-seq) | Multi-step team commit forecast (next 3 weeks) | MAE, R² |
| 3 | **XGBoost + SHAP** | Supervised (tree) | Explainable commit prediction with feature attribution | MAE, R², SHAP values |
| 4 | **K-Means** | Unsupervised (clustering) | Group developers into behavioral archetypes | Silhouette Score |
| 5 | **Isolation Forest** | Unsupervised (anomaly) | Detect burnout/anomaly days per developer | Anomaly rate %, Drop/Spike |
| 6 | **VAE** | Unsupervised (anomaly) | Team-level weekly health anomaly detection | MSE reconstruction error |
| 7 | **RL Bandit** | Reinforcement Learning | Optimize sprint configurations via Thompson Sampling | Cumulative regret, posterior mean |
| 8 | **CodeBERT** | NLP + Clustering | Embed developer activity descriptions, cluster by behavior | Silhouette, productivity score |

### Run All Models

```bash
python notebooks/run_all_models.py
```

---

## Frontend (React Dashboard)

The dashboard is built with React + TypeScript + Recharts and deployed on **Netlify**.

Features:
- **LSTM Forecast Visualization** — coding velocity chart with commit/review trends
- **Deep Learning Insights Panel** — K-Means archetypes, Isolation Forest alerts, RL Bandit recommendations, VAE health flags
- **PR Cycle Distribution** — review phase, testing, idle time breakdown
- **Flow State Efficiency** — deep work hours, context switches, model R² score
- **Git Activity Feed** — real-time commit/merge/error log
- **Build & Deployment Logs** — CI/CD status table with CSV export
- **AI Chatbot** — interactive assistant for querying team metrics and model outputs

### Run Locally

```bash
cd frontend
npm install
npm run dev
```

---

## Prerequisites

Before running, you must:

1. Have BigQuery User access to the project
2. Run:

```bash
gcloud auth login
gcloud config set project composed-tasks-345810
gcloud auth application-default login
```

3. Create a `.env` file with your BigQuery project and dataset config

---

## Dashboard
https://developer-productivity-system.netlify.app/
