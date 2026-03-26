# AGENTS.md

## Environment

- Before concluding that SigmaEvolve credentials or runtime configuration are missing, check the user-scoped env file at `/Users/tsilva/.config/sigmaevolve/.env`.
- This file may provide `DATABASE_URL`, `OPENROUTER_API_KEY`, and other runtime settings even when those variables are not present in the current shell environment.
- Do not print secret values back to the user. It is enough to confirm whether the required variables are available.

## Modal Runs

- For remote Modal execution, verify the database URL is network-accessible and loaded from `/Users/tsilva/.config/sigmaevolve/.env` before reporting that Modal runs are blocked on configuration.

## Experiment Provenance

- All non-baseline trials must come from the configured LLM prompting pipeline and retain recorded prompt provenance.
- Do not invent, hand-author, manually curate, or otherwise submit your own experiment variants as queued trials.
- Do not enqueue or persist ad hoc provenance labels such as `manual-curated`, `manual-variant`, `legacy`, `test`, or similar stand-ins for generated candidates.
- The only allowed non-prompt exception is the system-seeded baseline trial. Any new candidate must include recorded prompt messages from the LLM request path.

## Documentation

- Keep `docs/DB.md` in sync with the live schema at all times.
- Whenever tables, columns, constraints, or the expected contents of persisted JSON fields change, update `docs/DB.md` in the same change.
- When creating or editing Python code, follow the Ruff configuration in `pyproject.toml` and the repo-local `format-code` skill at `.codex/skills/format-code/SKILL.md`.
- Use `.codex/skills/format-code/references/manual-style.md` as the reference for the remaining non-deterministic style rules and examples; do not copy the examples mechanically.
