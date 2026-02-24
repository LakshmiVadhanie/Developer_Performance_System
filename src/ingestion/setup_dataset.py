from utils.setup_creds import get_bigquery_client


def ensure_devinsight_dataset():
    client = get_bigquery_client()

    query = """
    CREATE SCHEMA IF NOT EXISTS `composed-task-345810.devinsight`
    OPTIONS (location = "US");
    """

    client.query(query).result()


if __name__ == "__main__":
    ensure_devinsight_dataset()
    print("Dataset ready: composed-task-345810.devinsight")
