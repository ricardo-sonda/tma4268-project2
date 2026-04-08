# Command Reference

This file lists the main command entrypoints.

## Python Tooling Contract

This repository is a `uv` application project.

That means:
- dependencies are managed through [`pyproject.toml`](../../pyproject.toml) and [`uv.lock`](../../uv.lock)
- repository commands run through `uv run`
- the repository is not meant to be built or published as a Python package
- do not maintain a parallel `requirements.txt` dependency manifest unless the canonical architecture changes
- do not add `setuptools`, `setup.py`, `setup.cfg`, or `*.egg-info/` back into the main workflow unless the canonical architecture changes

## Make Targets

- `make data`
  - runs the dataset pipeline from source
  - downloads raw data if needed
  - rebuilds the cleaned interim CSV
  - rebuilds the convenience SQLite database

- `make refresh-data`
  - removes and rebuilds the data area from scratch
  - rebuilds the convenience SQLite database as part of the full pipeline

- `make sql`
  - rebuilds the convenience SQLite database only

- `make plots`
  - regenerates EDA figures

- `make evaluate`
  - evaluates all registered models

- `make evaluate MODEL=<model_name>`
  - evaluates one registered model

- `make tui`
  - opens the stateless Textual interface for selecting models, launching evaluation, and inspecting the current outputs in `reports/`

- `make logistic-ground-zero`
  - evaluates the current logistic ground-zero model directly

- `make clean`
  - removes local `__pycache__/` directories and stray `*.egg-info/` directories

## Direct Python Entry Points

- `uv run python -m src.dataset build-all`
- `uv run python -m src.dataset rebuild-from-scratch`
- `uv run python -m src.dataset db`
- `uv run python -m src.plots all`
- `uv run python -m src.modeling.run all`
- `uv run python -m src.modeling.run logistic_ground_zero`
- `uv run python -m src.tui`
- `uv run python -m src.list_features <feature_set>`
