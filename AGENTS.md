# AGENTS.md

## Environment

- Before concluding that SigmaEvolve credentials or runtime configuration are missing, check the user-scoped env file at `/Users/tsilva/.config/sigmaevolve/.env`.
- This file may provide `SIGMAEVOLVE_DATABASE_URL`, `DATABASE_URL`, `OPENROUTER_API_KEY`, and other runtime settings even when those variables are not present in the current shell environment.
- Do not print secret values back to the user. It is enough to confirm whether the required variables are available.

## Modal Runs

- For remote Modal execution, verify the database URL is network-accessible and loaded from `/Users/tsilva/.config/sigmaevolve/.env` before reporting that Modal runs are blocked on configuration.
