from ingestion.setup_dataset import ensure_devinsight_dataset
from ingestion.scoped_events import build_scoped_events


def run_ingestion():
    ensure_devinsight_dataset()
    build_scoped_events()


if __name__ == "__main__":
    run_ingestion()
