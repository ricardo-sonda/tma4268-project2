# Repository Structure

This file is the canonical description of how the repository is organized.

## Top-Level Layout

- [`src/`](../../src): source tree for project code
- [`data/`](../../data): datasets by lifecycle stage
- [`sql/`](../../sql): SQLite convenience area
- [`docs/`](..): canonical shared documentation
- [`human/`](../../human): human-only material that coding agents must not read or modify without explicit approval
- [`reports/`](../../reports): generated outputs
- [`notebooks/`](../../notebooks): exploratory scratch work

## Source Tree

`src/` is a local import root used by the repository commands.

It is not intended to be built or published as a distributable Python package.

### `src/dataset.py`

Responsibilities:
- download raw Kaggle data
- clean the UFC source CSV into the ground-zero dataset
- rebuild the convenience SQLite database

This module owns dataset preparation mechanics.

### `src/features.py`

Responsibilities:
- define feature-builder functions
- build `PreparedDataset` objects
- keep feature engineering in one obvious place

This file is the main surface for feature experimentation.

### `src/modeling/`

Responsibilities:
- one file per model
- thin OOP interface through [`src/modeling/base.py`](../../src/modeling/base.py)
- model registry through [`src/modeling/registry.py`](../../src/modeling/registry.py)
- evaluation entrypoint through [`src/modeling/run.py`](../../src/modeling/run.py)

Each model file should stay narrowly focused on model behavior.

### `src/evaluation.py`

Responsibilities:
- define the shared evaluation contract
- define the split strategy
- compare model probabilities against bookmaker probabilities
- write standardized evaluation outputs

Evaluation must stay external to the models.

### `src/tui.py`

Responsibilities:
- provide a stateless terminal UI for launching model evaluation
- inspect the current evaluation artifacts already written under `reports/`
- show available models even when no current evaluation outputs exist

The TUI is a frontend over the existing evaluation workflow, not a separate run-tracking system.

### `src/list_features.py`

Responsibilities:
- print the columns produced by a named feature set
- make feature-set inspection available as a small direct CLI entrypoint

### `src/plots.py`

Responsibilities:
- generate exploratory plots
- write figures into `reports/figures/`

This module is for EDA and visualization, not model evaluation logic.

## Data Layout

### `data/raw/`

Immutable source downloads.

### `data/interim/`

Cleaned project baseline data.

### `data/processed/`

Reserved for future feature-engineered and modeling-ready datasets.

## SQL Layout

`sql/` exists because SQLite is useful for manual inspection and experimentation in an editor.

It should not become the conceptual backbone of the project.

## Documentation Layout

### `docs/`

Canonical project truth.

### `human/`

Human-only thoughts, drafts, and personal notes.

If something belongs in project truth, it belongs in `docs/`, not in `human/`.

Coding agents must not read or modify `human/` unless the user explicitly approves that access.

## Generated Output Layout

### `reports/figures/`

Generated plots.

### `reports/metrics/`

Structured evaluation summaries.

### `reports/predictions/`

Per-fight prediction outputs.
