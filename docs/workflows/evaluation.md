# Evaluation Workflow

This file describes the current evaluation contract and the reasoning behind it.

## Why Evaluation Is Centralized

Evaluation is centralized so that model comparisons mean the same thing across the repository.

If each model used its own evaluation setup, it would become harder to compare:
- different algorithms
- different feature sets
- model predictions versus bookmaker probabilities

So evaluation stays in one place:
- [`src/evaluation.py`](../../src/evaluation.py)

## Current Evaluation Contract

Current evaluation does the following:
- builds the chosen feature set
- applies the same chronological holdout split
- fits the model on the train portion
- predicts probabilities on the test portion
- evaluates bookmaker implied probabilities on the same test portion
- writes standardized outputs to `reports/`

## Split Strategy

Current default:
- chronological holdout
- last 20% of fights are test data

This choice exists because time leakage matters more than convenience in this problem.

## Current Metrics

Current standardized metrics:
- accuracy
- log loss
- Brier score
- ROC AUC
- precision (class 1 = Red wins)
- recall (class 1 = Red wins)
- F1 score (class 1 = Red wins)

These are not the final class value function, but they are the current shared comparison surface.

## Current Outputs

Evaluation writes:
- metrics JSON under `reports/metrics/`
- per-fight prediction CSVs under `reports/predictions/`
- ROC comparison plots under `reports/figures/evaluation/`
- confusion matrix plots under `reports/figures/evaluation/`

The terminal UI in [`src/tui.py`](../../src/tui.py) reads those current artifacts directly.

It is intentionally stateless:
- it does not create a separate run-history abstraction
- it shows the currently available models even if no outputs exist yet
- launching evaluation from the TUI rewrites the same standardized outputs

## Bookmaker Benchmark

Bookmaker implied probabilities are always evaluated on the same test split as the model.

That benchmark is part of the evaluator, not a separate model implementation.

This is intentional:
- bookmaker probabilities are a comparison target
- they are not meant to shape the modeling abstraction itself

## Future Direction

Still to be clarified later:
- the final class-facing value function
- whether betting-oriented metrics should be part of the default evaluator
- whether additional calibration diagnostics should be standardized
