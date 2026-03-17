# tma4268-project2
Project 2 in TMA4268.

## Setup

Install dependencies:
```bash
uv sync
```

Kaggle downloads require credentials configured for `kagglehub`.

## One-command pipeline

Run the full UFC pipeline from the project root:

```bash
uv run python init_data.py
```

This command will:

1. Clear everything under `datasets/`
2. Download `ultimate-ufc`
3. Generate `datasets/ultimate-ufc/ufc-clean.csv`
4. Rebuild `sql/database.db`

## Data pipeline

If you want to run the steps manually:

1. Download datasets with `fetching/installer.py`
2. Clean the UFC master dataset with `fetching/clean_data.py`
3. Load all CSVs under `datasets/` into SQLite with `fetching/csv_to_db.py`

`fetching/installer.py` still supports editing `TO_INSTALL` directly. If you do not pass dataset names on the command line, it will download whatever is listed there.

```bash
uv run python fetching/installer.py --list
uv run python fetching/installer.py ultimate-ufc
uv run python fetching/clean_data.py
uv run python fetching/csv_to_db.py
```

`fetching/clean_data.py` reads `datasets/ultimate-ufc/ufc-master.csv`, drops unwanted columns (win breakdowns, derived diffs, rankings, fight outcomes, raw odds), derives decimal odds and normalized implied probabilities, and writes `datasets/ultimate-ufc/ufc-clean.csv`. See `fetching/columns.md` for the full KEEP/DROP rationale.

`fetching/csv_to_db.py` scans all CSVs under `datasets/` and loads them into `sql/database.db` as SQLite tables. Table names follow the pattern `folder__filename` such as `ultimate_ufc__ufc_clean`.

## Querying the data

Open the database with any SQLite client:
```bash
sqlite3 sql/database.db
```

Or use SQL files in `sql/` such as `sql/query.sql`, which queries `ultimate_ufc__ufc_clean`.
