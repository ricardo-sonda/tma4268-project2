"""Shared project paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SQL_DIR = PROJECT_ROOT / "sql"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"

UFC_DATASET_NAME = "ultimate-ufc"
UFC_KAGGLE_URL = "https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset"

RAW_UFC_DIR = RAW_DATA_DIR / UFC_DATASET_NAME
INTERIM_UFC_DIR = INTERIM_DATA_DIR / UFC_DATASET_NAME
PROCESSED_UFC_DIR = PROCESSED_DATA_DIR / UFC_DATASET_NAME

UFC_MASTER_CSV = RAW_UFC_DIR / "ufc-master.csv"
UFC_UPCOMING_CSV = RAW_UFC_DIR / "upcoming.csv"
UFC_CLEAN_CSV = INTERIM_UFC_DIR / "ufc-clean.csv"
SQLITE_PATH = SQL_DIR / "ufc.sqlite"
