# Design Principles

This file describes the design principles that future changes should preserve.

## Primary Goal

The repository should minimize mental overhead while supporting statistical learning work.

That means the architecture should make common tasks obvious:
- add a feature set
- add a model
- run evaluation
- inspect outputs

## Main Principles

### 1. One Concern, One Place

Each major concern should have one primary home.

- data preparation: [`src/dataset.py`](../../src/dataset.py)
- feature engineering: [`src/features.py`](../../src/features.py)
- model implementation: [`src/modeling/`](../../src/modeling)
- evaluation: [`src/evaluation.py`](../../src/evaluation.py)
- shared project truth: [`docs/`](..)
- human-only drafts: [`human/`](../../human)

If one task requires editing many unrelated places, the architecture is probably drifting.

### 2. Keep Statistical Ideas Visible

This repository is for learning and comparing statistical learning methods.

So code should make the statistical idea easy to see.

That means:
- avoid abstractions that hide what data the model sees
- avoid abstractions that blur the difference between feature design and algorithm choice
- allow small amounts of repetition if they preserve clarity

### 3. Separate Feature Sets From Models

Feature sets and models are different dimensions.

We want to be able to ask:
- what happens if the feature set changes?
- what happens if the model changes?

That requires them to stay separate in the architecture.

### 4. Keep Evaluation External

Models should not each define their own evaluation framework.

Evaluation belongs outside the models so comparisons stay consistent.

### 5. Prefer Thin OOP

The modeling layer is object-oriented because that fits the intended way of working and thinking.

But the OOP should stay thin:
- models select a feature-builder function
- models implement `fit`
- models implement `predict_proba`

They should not become mini-frameworks.

### 6. Prefer Rerun-From-Source

This project prefers recomputation over hidden caching.

The reason is simple:
- recomputation is cheap
- stale outputs are more costly than extra seconds
- a simpler mental model is worth preserving

### 7. Documentation Must Match Reality

Architecture documentation is part of the architecture.

If structure changes, the docs should change with it.

That includes:
- [`README.md`](../../README.md)
- [`AGENTS.md`](../../AGENTS.md)
- [`CLAUDE.md`](../../CLAUDE.md)
- relevant files under [`docs/`](..)

### 8. Documentation Should Be BCNF-Normalized

Documentation should follow the same discipline we want from well-normalized data:
- one fact should have one canonical home
- other files should refer to that home rather than restating the same fact

The goal is to reduce:
- update anomalies
- deletion anomalies
- stale duplicated explanations

In practice this means:
- `docs/` holds canonical project truth
- `README.md`, `AGENTS.md`, and `CLAUDE.md` should mostly index into `docs/`
- avoid repeating the same architectural explanation in multiple files
- avoid copying long workflow explanations into several places
- if a fact has to change, there should ideally be one main file to update

Links and indexes can still point to canonical docs, but they should not become shadow copies of them.

### 9. Documentation Links Should Be Relative

Repository documentation should use relative repository paths, not absolute machine-specific filesystem paths.

The reason is portability:
- absolute local paths break when the repo is moved
- they tie documentation to one machine
- they create unnecessary maintenance overhead

Canonical docs should therefore use relative links whenever they link to repository files.

### 10. `AGENTS.md` And `CLAUDE.md` Must Be Exact Mirrors

`AGENTS.md` and `CLAUDE.md` serve the same role in this repository.

They should therefore be exact mirrors of each other:
- same content
- same links
- same constraints
- same wording

If one changes, the other must be updated in the same change so they remain identical.

### 11. `human/` Is Approval-Only For Agents

`human/` exists for human-only notes, drafts, and personal material.

Coding agents must not:
- read `human/`
- write `human/`
- edit `human/`

unless the user gives explicit approval for that access in the current conversation.

`human/` is therefore outside the normal source-of-truth flow for agent work.

### 12. Structural Changes Must Update Canonical Docs

If a structural decision changes, update the canonical documentation in the same change.

In practice this usually means:
- update the canonical page in `docs/` that owns the fact
- then repair any affected index links in `README.md`, `AGENTS.md`, and `CLAUDE.md`

Prefer updating one canonical doc page plus any needed pointers rather than restating the same explanation across multiple files.

### 13. The Architecture Must Stay Easy To Explain

If a new abstraction makes simple tasks harder to explain, that abstraction is suspicious.

The default questions are:
- does this reduce or increase the number of places a person needs to look?
- does this reduce or increase the number of places a person needs to update?

If the answer is “increase,” the repository may be drifting away from its low-overhead goal.

## Examples Of Good Changes

- adding one new feature-builder function in `src/features.py`
- adding one new model file in `src/modeling/`
- extending `src/evaluation.py` with a clearly shared metric
- adding a new canonical doc page to explain a durable decision

## Examples Of Bad Changes

- hidden feature engineering inside one model file
- evaluation logic copied into several models
- moving canonical decisions into `human/`
- adding complexity to save trivial compute time
- introducing string indirection where direct Python references are simpler
