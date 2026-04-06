"""Download, clean, and materialize the UFC dataset."""

import argparse
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlparse

import kagglehub
import numpy as np
import pandas as pd

from .config import DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, RAW_UFC_DIR, SQLITE_PATH, UFC_CLEAN_CSV, UFC_DATASET_NAME, UFC_KAGGLE_URL, UFC_MASTER_CSV


def kaggle_handle_from_url(kaggle_url: str) -> str:
    path = urlparse(kaggle_url).path.strip("/")
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"Invalid Kaggle dataset URL: {kaggle_url}")
    return f"{parts[1]}/{parts[2]}"


def reset_data_dir(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in data_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for required_dir in (RAW_DATA_DIR, INTERIM_DATA_DIR):
        required_dir.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_name: str = UFC_DATASET_NAME) -> Path:
    if dataset_name != UFC_DATASET_NAME:
        raise ValueError(f"Only '{UFC_DATASET_NAME}' is supported in this project.")

    RAW_UFC_DIR.mkdir(parents=True, exist_ok=True)
    kaggle_handle = kaggle_handle_from_url(UFC_KAGGLE_URL)
    path = kagglehub.dataset_download(kaggle_handle, output_dir=str(RAW_UFC_DIR))
    print(f"Downloaded '{dataset_name}' to {path}")
    return Path(path)


def moneyline_to_raw_prob(odds: pd.Series) -> pd.Series:
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100), 100 / (odds + 100))


def moneyline_to_decimal(odds: pd.Series) -> pd.Series:
    return np.where(odds < 0, 1 + 100 / np.abs(odds), 1 + odds / 100)


def clean_ufc_dataset(
    input_path: Path = UFC_MASTER_CSV,
    output_path: Path = UFC_CLEAN_CSV,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    df = pd.read_csv(input_path)
    print(f"Loaded {input_path} - {df.shape[0]} rows, {df.shape[1]} columns")

    drop_cols = [
        "RedWinsByDecisionMajority", "RedWinsByDecisionSplit",
        "RedWinsByDecisionUnanimous", "RedWinsByKO",
        "RedWinsBySubmission", "RedWinsByTKODoctorStoppage",
        "BlueWinsByDecisionMajority", "BlueWinsByDecisionSplit",
        "BlueWinsByDecisionUnanimous", "BlueWinsByKO",
        "BlueWinsBySubmission", "BlueWinsByTKODoctorStoppage",
        "LoseStreakDif", "WinStreakDif", "LongestWinStreakDif", "WinDif",
        "LossDif", "TotalRoundDif", "TotalTitleBoutDif", "KODif", "SubDif",
        "HeightDif", "ReachDif", "AgeDif", "SigStrDif", "AvgSubAttDif", "AvgTDDif",
        "RMatchWCRank", "BMatchWCRank", "RPFPRank", "BPFPRank", "BetterRank",
        "RWFlyweightRank", "RWFeatherweightRank", "RWStrawweightRank",
        "RWBantamweightRank", "RHeavyweightRank", "RLightHeavyweightRank",
        "RMiddleweightRank", "RWelterweightRank", "RLightweightRank",
        "RFeatherweightRank", "RBantamweightRank", "RFlyweightRank",
        "BWFlyweightRank", "BWFeatherweightRank", "BWStrawweightRank",
        "BWBantamweightRank", "BHeavyweightRank", "BLightHeavyweightRank",
        "BMiddleweightRank", "BWelterweightRank", "BLightweightRank",
        "BFeatherweightRank", "BBantamweightRank", "BFlyweightRank",
        "Finish", "FinishDetails", "FinishRound", "FinishRoundTime",
        "TotalFightTimeSecs",
        "RedOdds", "BlueOdds", "RedExpectedValue", "BlueExpectedValue",
        "RedDecOdds", "BlueDecOdds", "RSubOdds", "BSubOdds", "RKOOdds", "BKOOdds",
    ]

    df = df.drop(columns=drop_cols)
    print(f"After dropping columns - {df.shape[1]} columns remain")

    raw = pd.read_csv(input_path, usecols=["RedOdds", "BlueOdds"])
    red_raw = moneyline_to_raw_prob(raw["RedOdds"])
    blue_raw = moneyline_to_raw_prob(raw["BlueOdds"])
    prob_sum = red_raw + blue_raw

    has_odds = raw["RedOdds"].notna() & raw["BlueOdds"].notna()
    valid_vig = (prob_sum >= 1.0) & (prob_sum <= 1.15)
    keep_mask = has_odds & valid_vig

    n_before = len(df)
    mask = keep_mask.values
    df = df[mask].reset_index(drop=True)
    red_raw = pd.Series(red_raw[mask]).reset_index(drop=True)
    blue_raw = pd.Series(blue_raw[mask]).reset_index(drop=True)
    prob_sum = pd.Series(prob_sum[mask]).reset_index(drop=True)
    raw = raw[mask].reset_index(drop=True)
    print(f"After filtering invalid odds - dropped {n_before - len(df)} rows, {len(df)} remain")

    red_dec = moneyline_to_decimal(raw["RedOdds"])
    blue_dec = moneyline_to_decimal(raw["BlueOdds"])
    red_prob = red_raw / prob_sum
    blue_prob = blue_raw / prob_sum

    df.insert(2, "RedDecimalOdds", red_dec)
    df.insert(3, "BlueDecimalOdds", blue_dec)
    df.insert(4, "RedImpliedProb", red_prob)
    df.insert(5, "BlueImpliedProb", blue_prob)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} - {df.shape[0]} rows, {df.shape[1]} columns")
    return output_path


def sanitize_sql_name(value: str) -> str:
    return re.sub(r"\W+", "_", value).strip("_")


def build_database(data_dir: Path = DATA_DIR, db_path: Path = SQLITE_PATH) -> Path:
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSV files found under {data_dir}/")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    try:
        for csv_path in csvs:
            relative_parts = csv_path.relative_to(data_dir).with_suffix("").parts
            table_name = "__".join(sanitize_sql_name(part) for part in relative_parts)
            try:
                df = pd.read_csv(csv_path)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin-1")
            df.to_sql(table_name, con, if_exists="replace", index=False)
            print(f"{csv_path} -> {db_path}::{table_name} ({len(df)} rows, {len(df.columns)} cols)")
    finally:
        con.close()

    print(f"Created {db_path} with {len(csvs)} tables")
    return db_path


def build_all() -> None:
    if not UFC_MASTER_CSV.exists():
        download_dataset()
    clean_ufc_dataset()
    build_database()


def rebuild_from_scratch() -> None:
    reset_data_dir()
    download_dataset()
    clean_ufc_dataset()
    build_database()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFC dataset pipeline.")
    parser.add_argument(
        "command",
        choices=["download", "clean", "db", "build-all", "rebuild-from-scratch", "reset"],
        help="Pipeline step to run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "download":
            download_dataset()
        elif args.command == "clean":
            clean_ufc_dataset()
        elif args.command == "db":
            build_database()
        elif args.command == "rebuild-from-scratch":
            rebuild_from_scratch()
        elif args.command == "reset":
            reset_data_dir()
        else:
            build_all()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
