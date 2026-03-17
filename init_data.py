"""Run the full UFC data pipeline from the project root."""

import shutil
from pathlib import Path

from fetching.clean_data import clean_ufc_dataset
from fetching.csv_to_db import build_database
from fetching.installer import download_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
UFC_DATASET = "ultimate-ufc"


def reset_datasets_dir(datasets_dir: Path = DATASETS_DIR) -> None:
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for path in datasets_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> int:
    print(f"Resetting {DATASETS_DIR}")
    reset_datasets_dir()

    print(f"\nDownloading dataset: {UFC_DATASET}")
    download_dataset(UFC_DATASET)

    print("\nCleaning UFC dataset")
    clean_ufc_dataset()

    print("\nBuilding SQLite database")
    build_database()

    print("\nData pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
