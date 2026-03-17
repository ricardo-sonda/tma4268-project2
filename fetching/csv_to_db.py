"""Convert all CSVs under datasets/ into one SQLite database in sql/."""

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
SQL_DIR = PROJECT_ROOT / "sql"
DB_PATH = SQL_DIR / "database.db"


def sanitize_sql_name(value: str) -> str:
    """Map file and folder names to SQLite-safe identifiers."""
    return re.sub(r"\W+", "_", value).strip("_")


def build_database(datasets_dir: Path = DATASETS_DIR, db_path: Path = DB_PATH) -> Path:
    csvs = sorted(datasets_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSV files found under {datasets_dir}/")

    db_path.parent.mkdir(exist_ok=True)

    for stale_db in db_path.parent.glob("*.db"):
        stale_db.unlink()

    con = sqlite3.connect(db_path)

    try:
        for csv_path in csvs:
            relative_parent = csv_path.relative_to(datasets_dir).parent.parts
            folder_part = "__".join(sanitize_sql_name(part) for part in relative_parent if part)
            file_part = sanitize_sql_name(csv_path.stem)
            table_name = f"{folder_part}__{file_part}" if folder_part else file_part

            try:
                df = pd.read_csv(csv_path)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin-1")

            df.to_sql(table_name, con, if_exists="replace", index=False)
            print(f"  {csv_path} -> {db_path}::{table_name} ({len(df)} rows, {len(df.columns)} cols)")
    finally:
        con.close()

    print(f"\nCreated {db_path} with {len(csvs)} tables")
    return db_path


def main() -> int:
    try:
        build_database()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
