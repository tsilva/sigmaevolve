---
name: format-code
description: Format and restyle Python code in the SigmaEvolve repository. Use when Codex needs to clean up, standardize, or review edits in `sigmaevolve/**/*.py` or `tests/**/*.py`, especially when the work should run Ruff fixers first, Ruff formatter second, and then apply the remaining SigmaEvolve manual readability rules. Also use this skill when parallelizing formatting or readability passes across disjoint Python file chunks.
---

# Format Code

## Overview

Use Ruff for deterministic rewrites first. Then apply only the remaining readability rules from [`references/manual-style.md`](./references/manual-style.md).

## Workflow

1. Identify the Python target set.
   Default to the files being edited.
   Use repo-wide targets only when the user explicitly asks for repo-wide cleanup.
2. Run Ruff fixers before making manual readability edits:
   ```bash
   ruff check --fix <targets>
   ```
3. Run Ruff formatter immediately after the fixer pass:
   ```bash
   ruff format <targets>
   ```
4. Re-read the touched files and apply the remaining manual rules from [`references/manual-style.md`](./references/manual-style.md).
   Do not fight Ruff output or restyle code that Ruff already normalized acceptably.
5. Run a final lint pass after the manual edits:
   ```bash
   ruff check <targets>
   ```
6. Run the narrowest relevant tests for the touched modules.

## Parallel Execution

Maximize parallelization when the environment allows delegation and the work spans multiple disjoint Python files.

1. Run the initial Ruff fixer and formatter pass once on the full target set before spawning workers.
   This gives every worker the same normalized starting point.
2. Use [`scripts/plan_chunks.py`](./scripts/plan_chunks.py) to propose disjoint worker chunks:
   ```bash
   python3 .codex/skills/format-code/scripts/plan_chunks.py <targets>
   ```
3. Treat the file as the minimum ownership unit.
   Never split one file across multiple workers by class or function.
4. Default chunk = one production file plus its matching test file when both are in scope.
5. Give large files their own worker chunk.
   In this repo, modules like `sigmaevolve/execution.py`, `sigmaevolve/generation.py`, and `sigmaevolve/orchestration.py` should stay single-owner.
6. Batch only tiny orphan files such as `__init__.py` or very small support files.
7. Keep the current agent on the critical-path chunk and send sidecar chunks to workers.
8. Tell each worker it owns only its assigned files, is not alone in the repo, and must not revert unrelated edits.
9. After worker results are integrated, rerun:
   ```bash
   ruff check <targets>
   ```
   Then rerun the narrowest relevant tests.

## Manual Rules

Read [`references/manual-style.md`](./references/manual-style.md) when the edit is more than a trivial Ruff-only cleanup.

Apply those rules to:
- structure non-trivial functions;
- keep validation and closely related state changes together;
- add meaningful block comments only when the intent is not obvious;
- introduce intermediate names for non-trivial expressions and boolean policies;
- extract helpers when a block mixes multiple responsibilities;
- avoid over-applying comments, helper extraction, or forced multiline formatting.

## Validation

- Use Ruff configuration from `pyproject.toml` as the deterministic source of truth.
- Validate this skill with:
  ```bash
  python3 /Users/tsilva/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
      .codex/skills/format-code
  ```
- If the skill changes its instructions materially, regenerate or update `agents/openai.yaml` so the UI metadata still matches.
