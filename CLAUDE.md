This file is an index for coding agents. The canonical project manual lives under `docs/`.

## Read First

- [`docs/architecture/project-purpose.md`](docs/architecture/project-purpose.md)
- [`docs/architecture/repo-structure.md`](docs/architecture/repo-structure.md)
- [`docs/architecture/data-flow.md`](docs/architecture/data-flow.md)
- [`docs/conventions/design-principles.md`](docs/conventions/design-principles.md)
- [`docs/workflows/iteration.md`](docs/workflows/iteration.md)
- [`docs/workflows/evaluation.md`](docs/workflows/evaluation.md)
- [`docs/reference/commands.md`](docs/reference/commands.md)
- [`docs/workflows/drift-repair.md`](docs/workflows/drift-repair.md)

## Hard Constraints

- Keep `README.md`, `AGENTS.md`, and `CLAUDE.md` aligned with the actual architecture.
- Keep `AGENTS.md` and `CLAUDE.md` as exact mirrors of each other.
- Treat `docs/` as canonical shared documentation.
- Do not read, write, or edit `human/` without explicit user approval.
- Keep documentation BCNF-normalized: one fact should have one canonical home, and other files should refer to it rather than duplicate it.
- Use relative repository paths in docs, never absolute machine-specific filesystem paths.
- Keep feature engineering in `src/features.py`.
- Keep model implementations in `src/modeling/`.
- Keep evaluation centralized in `src/evaluation.py`.
- Prefer rerun-from-source when that avoids staleness ambiguity.
