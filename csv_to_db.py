"""Convert all CSVs under datasets/ into one SQLite database in sql/."""

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path("datasets")
SQL_DIR = Path("sql")
DB_PATH = SQL_DIR / "database.db"


def sanitize_sql_name(value: str) -> str:
    """Map file and folder names to SQLite-safe identifiers."""
    return re.sub(r"\W+", "_", value).strip("_")

csvs = sorted(DATASETS_DIR.rglob("*.csv"))
if not csvs:
    print(f"Error: no CSV files found under {DATASETS_DIR}/", file=sys.stderr)
    sys.exit(1)

SQL_DIR.mkdir(exist_ok=True)

for stale_db in SQL_DIR.glob("*.db"):
    stale_db.unlink()

con = sqlite3.connect(DB_PATH)

for csv_path in csvs:
    relative_parent = csv_path.relative_to(DATASETS_DIR).parent.parts
    folder_part = "__".join(sanitize_sql_name(part) for part in relative_parent if part)
    file_part = sanitize_sql_name(csv_path.stem)
    table_name = f"{folder_part}__{file_part}" if folder_part else file_part

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    df.to_sql(table_name, con, if_exists="replace", index=False)
    print(f"  {csv_path} -> {DB_PATH}::{table_name} ({len(df)} rows, {len(df.columns)} cols)")

con.close()
print(f"\nCreated {DB_PATH} with {len(csvs)} tables")
