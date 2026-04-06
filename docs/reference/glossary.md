# Glossary

This file defines the main project terms.

## ground_zero

The cleaned baseline UFC dataset in:
- [`data/interim/ultimate-ufc/ufc-clean.csv`](../../data/interim/ultimate-ufc/ufc-clean.csv)

It is the starting point for feature engineering.

## Feature Set

A function in [`src/features.py`](../../src/features.py) that converts the ground-zero dataset into a standardized `PreparedDataset`.

## Model

A class under [`src/modeling/`](../../src/modeling) that subclasses [`BaseModel`](../../src/modeling/base.py) and implements:
- `fit`
- `predict_proba`

## Evaluator

The shared logic in [`src/evaluation.py`](../../src/evaluation.py) that:
- runs the standardized split
- scores models
- compares them to bookmaker probabilities
- writes outputs

## Canonical Documentation

Documentation under [`docs/`](..). This is shared project truth.

## Human-Only Material

Material under [`human/`](../../human). This is human-only, non-canonical, and outside the normal agent workflow. Coding agents must not read or modify it without explicit user approval.
