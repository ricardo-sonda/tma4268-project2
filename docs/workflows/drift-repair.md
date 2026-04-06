# Drift Repair Workflow

This file is a reusable workflow for a coding agent whose job is to find and fix repository drift.

Use this workflow when the repository still runs, but code, docs, paths, structure, or conventions may have drifted apart.

## How To Use This File

Give a coding agent this instruction:

Read `docs/workflows/drift-repair.md` and execute it fully.

## Agent Workflow

You are the drift-repair agent for this repository.

Your job is to restore alignment between:
- the codebase
- the canonical documentation
- the top-level index files
- this workflow file itself

Do not treat this file as the sole source of truth. Use it as a procedure for discovering the current canonical truth of the repository.

By default, execute this workflow as one coordinating agent plus multiple narrow sub-agents working in parallel.

## Source Of Truth Order

Use this trust order when checking for drift:

1. Read the top-level repository indexes:
   - `README.md`
   - `AGENTS.md`
   - `CLAUDE.md`
2. Follow those indexes into the canonical documentation under `docs/`.
3. Inspect the actual codebase and repository structure.
4. Do not read, write, or edit `human/` unless the user has explicitly approved that access in the current conversation.

If code and canonical docs disagree, prefer refactoring the repository back toward the documented architecture unless the documentation is clearly stale and inconsistent with the rest of the canonical docs.

## Drift Repair Procedure

1. Read the top-level indexes and the canonical docs they point to.
2. Build a mental model of the intended architecture, workflows, and documentation rules.
3. Decide the audit slices and spawn sub-agents with disjoint scopes.
4. While sub-agents run, inspect the repository structure, code layout, and Makefile entrypoints locally.
5. Collect the sub-agent findings.
6. Compare the actual repository against the intended design.
7. Fix drift wherever it appears, including:
   - code drift
   - documentation drift
   - link drift
   - path drift
   - workflow drift
   - duplication drift
   - drift in this workflow file
8. Re-run the relevant project commands to verify that the repaired structure still works.
9. Report what you found, what you changed, and any remaining ambiguity.

## Parallel Execution Pattern

Use a coordinator-and-workers pattern.

The coordinator should:
- read the top-level indexes first
- read the canonical docs needed to understand the intended architecture
- decide which checks can run independently
- assign disjoint scopes to sub-agents
- integrate findings and make the final repairs
- avoid delegating the final integration step

Sub-agents should:
- receive a narrow scope
- inspect only the docs and code relevant to that scope
- identify drift
- fix clear drift within their assigned scope when safe
- report findings in a way the coordinator can integrate quickly

Do not spawn sub-agents to do the same audit twice.

## Recommended Parallel Audit Slices

The default split is:

### 1. Documentation Structure And Link Audit

This sub-agent checks:
- broken links
- wrong relative paths
- references to deleted files
- index files that point to the wrong canonical docs
- duplicated documentation facts that violate normalization

This slice should focus on documentation integrity, not code semantics.

### 2. Codebase Versus Docs Audit

This sub-agent checks:
- whether the actual source layout matches the documented architecture
- whether feature logic, model logic, and evaluation logic still live in their documented homes
- whether file names, directories, and entrypoints match canonical docs

This slice should focus on architectural mismatch between code and docs.

### 3. Workflow And Command Audit

This sub-agent checks:
- whether documented commands match the Makefile
- whether the documented workflow still matches how the repository is actually run
- whether rerun-from-source behavior still matches the documented intent

This slice should focus on operational drift.

### 4. Top-Level Index And Mirror Audit

This sub-agent checks:
- whether `README.md` still works as a human entrypoint
- whether `AGENTS.md` and `CLAUDE.md` are exact mirrors
- whether top-level files are indexes rather than shadow copies of canonical docs

This slice should focus on entrypoint drift.

## How To Choose The Number Of Sub-Agents

Use as many sub-agents as there are genuinely independent audit slices.

For a small repository, two or three sub-agents is usually enough.

Use fewer sub-agents when:
- the repository is small
- the drift seems local
- integration overhead would dominate the work

Use more sub-agents when:
- documentation is broad
- code structure has changed significantly
- there are many independent entrypoints or workflows

Do not optimize for maximum parallelism. Optimize for the fastest path to a correct integrated repair.

## What Counts As Drift

Drift includes any case where the repository becomes harder to understand, harder to change, or less aligned with its documented intent.

Common examples:
- code no longer lives where the canonical docs say it should live
- feature logic, model logic, and evaluation logic are no longer clearly separated
- top-level index files point to the wrong docs
- canonical docs and actual filepaths no longer match
- the same fact is duplicated in several docs and has started to diverge
- commands in docs no longer match the Makefile or actual workflow

## Quick Drift Checklist

Before considering drift repair complete, ask:

1. Is the main home for each concern still obvious?
2. Would a new teammate know where to edit a feature set?
3. Would a new teammate know where to add a model?
4. Would a coding agent know where the canonical project truth lives?
5. Are `README.md`, `AGENTS.md`, `CLAUDE.md`, and `docs/` still aligned?

If the answer to any of these is no, drift repair is not finished.

## Repair Rules

When fixing drift, preserve these rules:

- Keep canonical project truth in `docs/`.
- Do not read, write, or edit `human/` without explicit user approval.
- Keep documentation BCNF-normalized:
  - one fact should have one canonical home
  - other files should refer to it rather than restate it
- Use relative repository paths in documentation.
- Prefer simple structures that reduce mental overhead.
- Prefer restoring existing architectural intent over introducing new abstractions.
- If a structural change affects documentation, update the canonical doc page and then repair any index links that point to it.
- Keep sub-agent scopes narrow enough that they do not fight over the same files.

## Scope Of Repair

You should repair drift in all reasonable places, including:
- source code
- documentation
- top-level indexes
- Makefile references
- path references
- this workflow file

Do not stop at identifying drift. Fix it when it is safe and clear to do so.

If sub-agents return conflicting recommendations, the coordinator should resolve the conflict by following the source-of-truth order in this file.

## Verification

After repairs, run the relevant verification commands for the affected parts of the repository.

At minimum:
- verify that the documented commands still match the actual Makefile
- verify that the main workflow still runs
- verify that edited documentation links still point to real files

Use the repository's documented commands and entrypoints rather than inventing ad hoc workflows.

## Report Back

At the end, report:
- the drift you found
- how you determined it was drift
- what you changed
- what you verified
- any remaining uncertainty or decisions that still need a human

Your report should distinguish between:
- code fixes
- documentation fixes
- structural fixes
- unresolved issues
