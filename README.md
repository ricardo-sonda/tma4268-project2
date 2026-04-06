# tma4268-project2
Project 2 in TMA4268.

## Problem Description

We use a comprehensive UFC dataset (~6500 fights, ~50 features per fight) to predict fight outcomes. Our features are all publicly available information known before a fight: fighter physical attributes (height, reach, weight, age), career records (wins, losses, streaks), and historical performance averages (striking accuracy, takedown rate, submission attempts).

**The goal is to build a model that, using only these public features, produces better win-probability estimates than the bookmakers' odds.** The bookmaker odds are not used as input to our models — they serve purely as the benchmark to beat. If our model's predicted probabilities are more accurate than the implied probabilities from the betting market, we have found signal that the market missed. We can then quantify the practical value of that edge by simulating a betting strategy against the historical odds.

This framing connects to several core topics in statistical learning:

- **Classification** — the fight outcome is binary (Red wins / Blue wins), making this a natural classification problem where we can explore logistic regression, LDA/QDA, tree-based methods, and SVMs.
- **Model selection & validation** — since fights are ordered in time, naive cross-validation leaks future information. We need chronological splits and must carefully think about what "generalization" means when the data-generating process (fighter careers, rule changes, evolving meta) shifts over time.
- **Feature engineering vs. regularization** — with ~50 raw features (many correlated, e.g. wins and win streaks), we face the classic bias-variance tradeoff. Shrinkage methods (ridge, lasso) and subset selection are directly applicable.
- **Calibration & probabilistic prediction** — raw classification accuracy is insufficient. We need well-calibrated predicted probabilities so we can meaningfully compare our model's confidence against the bookmaker's implied probabilities.
- **The value function** — correctly predicting a heavy favorite is easy but worthless at the betting window (the odds pay little), while correctly predicting an upset is rare but extremely profitable. This means our evaluation should weight predictions by the odds they'd be bet against, connecting to decision theory and asymmetric loss functions.

In short: can statistical learning methods, applied to publicly available fighter data alone, produce probability estimates that outperform the betting market?

## Setup

Install dependencies:
```
uv sync
```

Kaggle downloads require credentials configured for `kagglehub`.

## One-command pipeline

Run the full UFC pipeline from the project root:

```
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

```
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
