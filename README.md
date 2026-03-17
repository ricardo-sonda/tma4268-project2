# tma4268-project2
Project 2 in TMA4268.

## Setup


Download datasets:
```
uv run python fetching/installer.py          # list available datasets
uv run python fetching/installer.py fifa24   # download one
```

## Data pipeline

Run these in order:

1. **`fetching/clean_data.py`** — Reads `datasets/ultimate-ufc/ufc-master.csv`, drops unwanted columns (win breakdowns, derived diffs, rankings, fight outcomes, raw odds), derives new odds columns (decimal odds + normalized implied probabilities), and writes `datasets/ultimate-ufc/ufc-clean.csv`. See `fetching/columns.md` for the full column-by-column KEEP/DROP decisions.

2. **`fetching/csv_to_db.py`** — Scans all CSVs under `datasets/` and loads them into `sql/database.db` as SQLite tables. Table names follow the pattern `folder__filename` (e.g. `ultimate_ufc__ufc_clean`).

```
uv run python fetching/clean_data.py
uv run python fetching/csv_to_db.py
```

## Querying the data

Open the database with any SQLite client:
```
sqlite3 sql/database.db
```

Or use SQL files in `sql/` (e.g. `sql/query.sql`, which queries `ultimate_ufc__ufc_clean`).
