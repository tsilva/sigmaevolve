# AGENTS.md

## Environment

- Before concluding that SigmaEvolve credentials or runtime configuration are missing, check the user-scoped env file at `/Users/tsilva/.config/sigmaevolve/.env`.
- This file supplies nonsecret runtime settings. Private DB/OpenRouter/W&B values come from the configured process environment; never restore them from this file.
- Default dashboard and Python CLI commands use the configured environment. Dashboard-only Sentry values never enter scientific handlers.
- Do not print secret values back to the user. It is enough to confirm whether the required variables are available.

## Modal Runs

- For remote Modal execution, verify the database URL is network-accessible and present in the configured process environment before reporting that Modal runs are blocked on configuration.

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

## Runtime boundaries

Normal dashboard and Python commands use supplied process settings. Keep private values out of the user-scoped nonsecret settings file. Preserve separate Production delivery, Preview settings, and scientific/Modal workflows. Never log credentials or create plaintext exports. Verification does not authorize experiments, deployments, migrations or application-data reads.
