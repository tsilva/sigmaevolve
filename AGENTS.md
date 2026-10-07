# AGENTS.md

## Environment

- Before concluding that SigmaEvolve credentials or runtime configuration are missing, check the user-scoped env file at `/Users/tsilva/.config/sigmaevolve/.env`.
- This file supplies nonsecret runtime settings. Private DB/OpenRouter/W&B values come from the fixed Infisical development project; never restore them from this file.
- Default dashboard and Python CLI commands use Infisical. Dashboard-only Sentry values never enter scientific handlers.
- Do not print secret values back to the user. It is enough to confirm whether the required variables are available.

## Modal Runs

- For remote Modal execution, verify the database URL is network-accessible and loaded from the pinned Infisical development project before reporting that Modal runs are blocked on configuration.

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

## Dashboard Secrets

Default dashboard dev uses the human Infisical login and the development project pinned in root `.infisical.json`. Fetch only the dashboard allowlist; do not inject experiment OpenRouter/W&B credentials into Next.js. Production is isolated in `sigmaevolve-production`, Production `/`, synced only to Vercel Production; redeploy after changes. Keep Preview and Python/Modal experiment workflows separate. Preserve Keychain originals and user-scoped env files until their respective migrations are verified. Never log credentials or create plaintext exports. Run dashboard `test:secrets` after modifying this path.

## Python Secrets

Default `sigmaevolve` commands fetch the fixed development project after argument parsing, preserve nonsecret userfile settings, and fail closed before experiment/database work. Help requires no provider access. Modal receives only existing managed DB/W&B launcher inputs; never deploy or launch trials merely to test secret delivery. Keep userfile/Keychain recovery copies until the live experiment DB is verified.
