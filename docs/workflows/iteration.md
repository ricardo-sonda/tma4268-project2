# Iteration Workflow

This file describes the intended everyday workflow for working in the repository.

## Core Mental Model

The repository is built around three separate ideas:
- feature set
- model
- evaluation

Those should remain separate both conceptually and in code.

## Add A Feature Set

This is the intended workflow when you want to change what information a model sees.

1. Open [`src/features.py`](../../src/features.py).
2. Look at the cleaned baseline dataset in [`data/interim/ultimate-ufc/ufc-clean.csv`](../../data/interim/ultimate-ufc/ufc-clean.csv).
3. Add a new feature-builder function.
4. Return a `PreparedDataset`.
5. Do not put the feature engineering inside an individual model file.

If adding a feature set requires edits far outside `src/features.py`, that is usually a sign of drift.

## Add A Model

This is the intended workflow when you want to change the algorithm.

1. Add one new file under [`src/modeling/`](../../src/modeling).
2. Subclass [`BaseModel`](../../src/modeling/base.py).
3. Import the feature-builder function the model should use.
4. Implement `fit`.
5. Implement `predict_proba`.
6. Register the model class in [`src/modeling/registry.py`](../../src/modeling/registry.py).

The model file should mostly focus on:
- estimator configuration
- training behavior
- probability prediction

It should not become the main place for data cleaning or evaluation logic.

## Run The Project

### Main Commands

- `make data`
- `make sql`
- `make plots`
- `make evaluate`
- `make evaluate MODEL=<model_name>`

Canonical command descriptions live in [`docs/reference/commands.md`](../reference/commands.md).

## Why The Workflow Is Shaped This Way

The workflow is deliberately shaped so that:
- feature experimentation is localized
- model experimentation is localized
- evaluation stays standardized
- source-of-truth boundaries remain visible

That is more important here than reducing every possible line of code.
