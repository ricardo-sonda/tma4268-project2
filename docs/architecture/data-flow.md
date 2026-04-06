# Data Flow

This file describes how data is supposed to move through the repository.

## Principle

The data layout reflects stages of trust and transformation:
- `raw/` is immutable source data
- `interim/` is cleaned project baseline data
- `processed/` is reserved for downstream model-ready outputs

## Current Flow

### 1. Raw Source

The Kaggle UFC dataset is downloaded into:
- [`data/raw/ultimate-ufc/`](../../data/raw/ultimate-ufc)

Important files:
- [`data/raw/ultimate-ufc/ufc-master.csv`](../../data/raw/ultimate-ufc/ufc-master.csv)
- [`data/raw/ultimate-ufc/upcoming.csv`](../../data/raw/ultimate-ufc/upcoming.csv)

### 2. Cleaned Ground-Zero Dataset

The raw UFC master dataset is cleaned by [`src/dataset.py`](../../src/dataset.py) into:
- [`data/interim/ultimate-ufc/ufc-clean.csv`](../../data/interim/ultimate-ufc/ufc-clean.csv)

This is the canonical cleaned baseline for the project.

It is called “ground zero” because it is:
- clean enough to trust
- still general enough to support multiple feature sets
- not yet the final model-specific representation

### 3. Feature Sets

Feature sets are built in:
- [`src/features.py`](../../src/features.py)

Each feature-builder function returns a standardized `PreparedDataset` containing:
- `X`
- `y`
- `metadata`

The metadata exists so evaluation can preserve fight context and compare models against bookmaker implied probabilities.

### 4. Models

Models live under:
- [`src/modeling/`](../../src/modeling)

Each model:
- selects a feature-builder function
- fits on `X, y`
- predicts probabilities on `X`

### 5. Evaluation

Evaluation lives in:
- [`src/evaluation.py`](../../src/evaluation.py)

Current evaluation:
- uses a chronological holdout split
- evaluates the model on the test split
- evaluates bookmaker implied probabilities on the same split
- writes metrics and prediction artifacts to `reports/`

### 6. SQL Convenience Layer

SQLite lives in:
- [`sql/ufc.sqlite`](../../sql/ufc.sqlite)

This exists for local inspection and editor-side experimentation.

It is not the conceptual backbone of the project.

## Why `data/processed/` Is Reserved

`data/processed/` is the intended home for future outputs like:
- feature-engineered CSVs
- train/test split tables
- model-ready datasets

It is reserved even if it is lightly used today because that stage still matters conceptually.

## Signs Of Drift

The data flow is drifting if:
- `ufc-clean.csv` is edited directly by hand
- model files start doing their own cleaning logic
- evaluation depends on SQL instead of the canonical Python workflow
- model-ready tables start being dropped into `interim/` instead of `processed/`
