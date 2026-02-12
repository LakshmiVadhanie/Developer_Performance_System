import os
from google.cloud import bigquery
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_bigquery_client():
    """
    Returns an authenticated BigQuery client using
    Application Default Credentials (ADC)
    """
    project_id = os.getenv("GCP_PROJECT_ID")

    if not project_id:
        raise ValueError("GCP_PROJECT_ID is not set in environment variables")

    client = bigquery.Client(project=project_id)
    return client


def get_dataset_ref():
    """
    Returns the fully qualified dataset name:
    project.dataset
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset = os.getenv("BQ_DATASET")

    if not project_id or not dataset:
        raise ValueError("BQ_DATASET or GCP_PROJECT_ID not set")

    return f"{project_id}.{dataset}"