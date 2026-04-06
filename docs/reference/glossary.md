# Glossary

This file defines the main project terms.

## ground_zero

The cleaned baseline UFC dataset in [`data/interim/ultimate-ufc/ufc-clean.csv`](../../data/interim/ultimate-ufc/ufc-clean.csv).

Canonical baseline description: [`docs/architecture/data-flow.md`](../architecture/data-flow.md).

## Feature Set

A function in [`src/features.py`](../../src/features.py) that converts `ground_zero` into a standardized `PreparedDataset`.

Canonical placement: [`docs/architecture/repo-structure.md`](../architecture/repo-structure.md).

## Model

A class under [`src/modeling/`](../../src/modeling) that subclasses [`BaseModel`](../../src/modeling/base.py) and implements `fit` and `predict_proba`.

Canonical placement: [`docs/architecture/repo-structure.md`](../architecture/repo-structure.md).

## Evaluator

The shared logic in [`src/evaluation.py`](../../src/evaluation.py).

Canonical contract: [`docs/workflows/evaluation.md`](../workflows/evaluation.md).

## Canonical Documentation

Documentation under [`docs/`](..).

## Human-Only Material

Material under [`human/`](../../human).

Access rule: coding agents must not read or modify it without explicit user approval. See [`docs/conventions/design-principles.md`](../conventions/design-principles.md).
