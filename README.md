# TMA4268 Project 2

UFC outcome prediction project for TMA4268 Statistical Learning.

This top-level file is only an entrypoint. The canonical project manual lives under [`docs/`](docs/).

## Start Here

- Purpose: [`docs/architecture/project-purpose.md`](docs/architecture/project-purpose.md)
- Structure: [`docs/architecture/repo-structure.md`](docs/architecture/repo-structure.md)
- Data flow: [`docs/architecture/data-flow.md`](docs/architecture/data-flow.md)
- Iteration workflow: [`docs/workflows/iteration.md`](docs/workflows/iteration.md)
- Evaluation workflow: [`docs/workflows/evaluation.md`](docs/workflows/evaluation.md)
- Command reference: [`docs/reference/commands.md`](docs/reference/commands.md)
- Latex report: [`reports/latex/report.pdf`](reports/latex/report.pdf)

## Common Commands

```bash
make data
make sql
make plots
make evaluate
make evaluate MODEL=logistic_ground_zero
make tui
```

Canonical command descriptions live in [`docs/reference/commands.md`](docs/reference/commands.md).
The canonical Python tooling contract also lives in [`docs/reference/commands.md`](docs/reference/commands.md).

## Documentation Contract

- `docs/` is canonical and shared by humans and coding agents.
- `human/` is non-canonical and human-only.
- Coding agents must not read, write, or edit `human/` without explicit user approval.
- `reports/` is generated output, not documentation.
