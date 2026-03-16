from pathlib import Path
from urllib.parse import urlparse

import kagglehub

# Edit these two variables when you want to share datasets with others.
DATASET_BANK = {
    "fifa24": "https://www.kaggle.com/datasets/rehandl23/fifa-24-player-stats-dataset",
    "ufc": "https://www.kaggle.com/datasets/jossilva3110/ufc-dataset-1994-2026",
    "ultimate-ufc": "https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset",
}
TO_INSTALL = [
    "fifa24",
]

DATASETS_DIR = Path("datasets")


def kaggle_handle_from_url(kaggle_url: str) -> str:
    path = urlparse(kaggle_url).path.strip("/")
    parts = path.split("/")

    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"Invalid Kaggle dataset URL: {kaggle_url}")

    return f"{parts[1]}/{parts[2]}"


def download_dataset(dataset_name: str) -> None:
    if dataset_name not in DATASET_BANK:
        available = ", ".join(DATASET_BANK)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    output_dir = DATASETS_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    kaggle_url = DATASET_BANK[dataset_name]
    kaggle_handle = kaggle_handle_from_url(kaggle_url)
    path = kagglehub.dataset_download(kaggle_handle, output_dir=str(output_dir))

    print(f"Downloaded '{dataset_name}'")
    print(f"Path: {path}")


def main() -> None:
    if not TO_INSTALL:
        print("No datasets selected. Add names to TO_INSTALL and run again.")
        return

    for dataset_name in TO_INSTALL:
        download_dataset(dataset_name)


if __name__ == "__main__":
    main()
