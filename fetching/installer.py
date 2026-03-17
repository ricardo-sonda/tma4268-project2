import argparse
from collections.abc import Sequence
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
    "ultimate-ufc",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"


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


def download_datasets(dataset_names: Sequence[str]) -> None:
    for dataset_name in dataset_names:
        download_dataset(dataset_name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Kaggle datasets into datasets/. "
            "If no dataset names are given, TO_INSTALL is used."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names from DATASET_BANK to download.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset names and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        for name, url in DATASET_BANK.items():
            print(f"{name}: {url}")
        return 0

    dataset_names = args.datasets or TO_INSTALL
    if not dataset_names:
        print("No datasets selected. Add names to TO_INSTALL, or pass dataset names on the command line.")
        return 1

    download_datasets(dataset_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
