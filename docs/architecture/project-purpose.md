# Project Purpose

This file explains why the repository exists and what the codebase is trying to optimize for.

## Context

This repository is for a TMA4268 Statistical Learning project.

The repository is not meant to be a production ML platform. It is meant to support:
- learning statistical learning clearly
- comparing models cleanly
- keeping experiments interpretable
- making architectural decisions easy to reason about later

Because of that, the codebase is intentionally optimized for clarity, consistency, and low mental overhead rather than for maximum generality.

## Problem Framing

We want to predict UFC fight winners from public pre-fight information.

The project target is:
- `Winner`

The bookmaker is present in the dataset because:
- bookmaker implied probabilities are the benchmark to compare against
- they are part of evaluation
- they are not a separate “bookmaker model” in the architecture

So the architecture is built around comparing model probabilities against bookmaker probabilities, not around implementing the bookmaker as a peer model class.

## What The Architecture Should Optimize For

The repository should optimize for:
- low mental overhead when iterating
- easy addition of new feature sets
- easy addition of new models
- standardized evaluation
- clear separation of concerns
- low risk of stale or ambiguous outputs

## What “Low Mental Overhead” Means Here

Low mental overhead means:
- a new feature set should mainly require the cleaned CSV and `src/features.py`
- a new model should mainly require one new model file under `src/modeling/`
- evaluation should live in one place
- commands should be simple to remember and run through `make`

Low mental overhead does not mean:
- eliminating every repeated line
- building the most abstract possible framework
- hiding the statistical idea behind layers of indirection

Some repetition is acceptable if it keeps the workflow obvious.

## Why The Modeling Layer Is Thin OOP

The modeling layer is object-oriented because that fits the intended way of thinking and the way the project is taught.

But the OOP should stay thin.

A model should mainly:
- choose a feature-builder function
- implement `fit`
- implement `predict_proba`

The model should not become the main home for:
- feature engineering
- evaluation logic
- global workflow orchestration

## Why Rerun-From-Source Is Preferred

This project prefers rerunning from source rather than adding complex caching logic because:
- the project is small enough that recomputation is cheap
- stale artifacts are more dangerous than extra compute time
- the simpler mental model is worth preserving

The intended invariant is:
- code is the source of truth
- rerunning commands regenerates outputs
- freshness should be obvious

## What Counts As Architectural Success

The architecture is succeeding if:
- new feature sets are easy to add
- new models are easy to add
- evaluation stays standardized
- documentation matches reality
- future humans and agents can understand the repository quickly

The architecture is failing if:
- model files start hiding feature engineering
- evaluation logic spreads across the codebase
- docs stop matching the actual structure
- simple changes require editing many unrelated files
