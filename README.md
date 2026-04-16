<p align="center">
  <img src="dashboard/logo.png" alt="SigmaEvolve" width="140" />
</p>

# SigmaEvolve

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Next.js Dashboard](https://img.shields.io/badge/dashboard-Next.js-000000?logo=nextdotjs&logoColor=white)
![Modal Runner](https://img.shields.io/badge/runner-Modal-7B61FF)

**🧬 Evolutionary training runs for immutable datasets 🧬**

[GitHub](https://github.com/tsilva/sigmaevolve) · [Dashboard README](dashboard/README.md) · [Database schema](docs/DB.md)

## Overview

**The Punchline**  
SigmaEvolve is an evolutionary training harness for classification experiments. It keeps datasets fixed, asks LLMs to mutate only marked regions of a self-contained training script, launches candidates locally or on Modal, and ranks the results by validation accuracy.

**The Pain**  
Model-search loops get hard to audit when candidate code, prompts, runtime metrics, and failure states live in different places. Manual trial curation also makes it too easy to lose provenance.

**The Solution**  
SigmaEvolve stores each track and trial in Postgres, requires prompt provenance for every non-baseline candidate, records runner state through a strict lifecycle, and exposes the same data through a CLI and a Next.js dashboard.

**The Result**  
You can seed a baseline script, generate auditable follow-up candidates through OpenRouter, launch a controlled number of runs, inspect failures, and compare trial quality from one shared database.

| Fact | Detail |
| --- | --- |
| Built-in track | `mnist:v1` baseline with one evolvable script block |
| Trial states | `queued`, `dispatching`, `active`, `finished`, `error` |
| Persistence | 2 live tables: `tracks` and `trials` |
| Runners | Inline local runner or remote Modal runner |

## Features

- **Self-contained experiment scripts**: `create-track` reads metadata from the training script itself, including dataset id, runner type, track policy, and evolution task.
- **Constrained code evolution**: LLM candidates may only change code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.
- **Prompt-provenance enforcement**: non-baseline trials must keep the generation backend, model, config, prompt messages, context trial ids, and generated source trace.
- **Queue-aware orchestration**: launch passes fill a ready queue, reserve trials with dispatch tokens, requeue expired dispatches, and mark stale runs.
- **Local and remote execution**: use `--launcher inline` for local subprocess runs or the default Modal launcher for remote jobs.
- **Dataset manifests**: prepared datasets are written as checksum-verified `.npz`, `.npy`, and `manifest.json` artifacts under the configured dataset root.
- **Experiment telemetry**: final metrics, timeout diagnostics, structured errors, Modal run metadata, and optional Weights & Biases links are persisted per trial.
- **Dashboard companion app**: the Next.js dashboard browses tracks, trial state, evaluation metrics, source, provenance, and lineage from the shared database.

## Quick Start

### Python Harness

Install the package with the development, dataset, and Modal extras:

```bash
uv sync --extra dev --extra datasets --extra modal
```

Create the user-scoped runtime config. SigmaEvolve loads this file automatically before running CLI commands:

```bash
mkdir -p ~/.config/sigmaevolve
$EDITOR ~/.config/sigmaevolve/.env
```

Minimum local configuration:

```dotenv
SIGMAEVOLVE_DATABASE_URL=postgresql://user:password@host:5432/sigmaevolve
SIGMAEVOLVE_OPENROUTER_API_KEY=...
SIGMAEVOLVE_DATASET_ROOT=./artifacts/datasets
```

Seed the bundled MNIST track:

```bash
uv run sigmaevolve create-track sigmaevolve/baselines/mnist.py
```

Run one candidate locally:

```bash
uv run sigmaevolve --launcher inline launch <track_id> 1
```

List trial state:

```bash
uv run sigmaevolve list-trials <track_id>
```

### Dashboard

Run the dashboard from its app directory:

```bash
cd dashboard
npm install
cp .env.example .env.local
npm run dev
```

The app starts on `http://localhost:3000`. It needs a reachable `DATABASE_URL` that points at the same database used by the Python harness.

## CLI

```bash
uv run sigmaevolve --help
```

| Command | Purpose |
| --- | --- |
| `create-track <script.py>` | Create a track and seed the baseline trial from a self-contained script. |
| `launch <track_id> <count>` | Generate and launch trials for one pass. |
| `launch <track_id> <count> --daemon` | Keep up to `count` trials running until interrupted or `--max-cycles` is reached. |
| `list-trials <track_id>` | Print trial rows as JSON, optionally filtered by repeated `--status` flags. |
| `modal-deploy` | Deploy the default Modal runner app. |
| `modal-sync-dataset <dataset_id>` | Upload a prepared dataset directory into the Modal dataset volume. |

Use the global launcher flag before the subcommand:

```bash
uv run sigmaevolve --launcher inline launch <track_id> 2
uv run sigmaevolve --launcher modal launch <track_id> 8 --daemon
```

## Experiment Scripts

Training scripts must include a leading SigmaEvolve metadata block and at least one evolvable region:

```python
# /// sigmaevolve
# version = 1
# dataset_id = "mnist:v1"
# runner = "python_train_v1"
#
# [track]
# epochs = 20
#
# [evolution]
# task = "Maximize validation accuracy while keeping the script runnable."
# ///

def make_experiment(device, train_ds, val_ds):
    # EVOLVE-BLOCK-START
    ...
    # EVOLVE-BLOCK-END
```

The bundled baseline lives at [sigmaevolve/baselines/mnist.py](sigmaevolve/baselines/mnist.py). Candidate generation uses search/replace patches against the current program and rejects changes outside the evolve block.

## Architecture

| Layer | Files | Role |
| --- | --- | --- |
| CLI | [sigmaevolve/cli.py](sigmaevolve/cli.py) | Parses commands, loads runtime config, reports launch progress, and prints JSON results. |
| Orchestration | [sigmaevolve/orchestration.py](sigmaevolve/orchestration.py) | Maintains ready queues, schedules generation, reserves trials, launches work, and sweeps stale state. |
| Generation | [sigmaevolve/generation.py](sigmaevolve/generation.py) | Builds OpenRouter prompts, samples context trials, parses patches, materializes candidates, and records generation failures. |
| Execution | [sigmaevolve/execution.py](sigmaevolve/execution.py) | Runs reserved trials, monitors progress files, scores predictions, captures errors, and logs optional WandB telemetry. |
| Storage | [sigmaevolve/storage.py](sigmaevolve/storage.py) | Defines the SQLAlchemy schema, validates provenance, normalizes metrics, and notifies the dashboard through Postgres. |
| Dashboard | [dashboard](dashboard) | Next.js app for browsing tracks, trials, metrics, source, provenance, and lineage. |

## Runtime Configuration

SigmaEvolve resolves settings from environment variables, with SigmaEvolve-prefixed names taking precedence where available. The CLI also loads `~/.config/sigmaevolve/.env` before parsing commands.

| Variable | Used by | Notes |
| --- | --- | --- |
| `SIGMAEVOLVE_DATABASE_URL` or `DATABASE_URL` | CLI, runners, dashboard | Required. Modal runs need a network-accessible Postgres URL, not SQLite. |
| `SIGMAEVOLVE_OPENROUTER_API_KEY` or `OPENROUTER_API_KEY` | generation | Required when generating non-baseline candidates. |
| `SIGMAEVOLVE_DATASET_ROOT` | dataset manager | Defaults to `./artifacts/datasets`. |
| `SIGMAEVOLVE_MODAL_APP_NAME` | Modal launcher | Defaults to `sigmaevolve-runner`. |
| `SIGMAEVOLVE_MODAL_FUNCTION_NAME` | Modal launcher | Defaults to `run_trial`. |
| `SIGMAEVOLVE_MODAL_DATASET_VOLUME` | Modal dataset sync | Defaults to `sigmaevolve-datasets`. |
| `SIGMAEVOLVE_MODAL_DATASET_MOUNT` | Modal runner | Defaults to `/mnt/datasets`. |
| `SIGMAEVOLVE_MODAL_ENVIRONMENT_NAME` | Modal | Optional Modal environment name. |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_BASE_URL` | execution | Optional remote Weights & Biases logging; offline and disabled modes are rejected. |
| `NEXT_PUBLIC_SITE_URL` | dashboard | Used for canonical metadata, Open Graph URLs, `robots.txt`, and `sitemap.xml`. |

## Development

Run the Python test suite:

```bash
uv run pytest
```

Run Ruff checks and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

Run the dashboard checks:

```bash
cd dashboard
npm run test
npm run build
```

## Notes

- The database schema is created by `SQLAlchemyRepository` on startup; see [docs/DB.md](docs/DB.md) for the live table and JSON-field contract.
- Dataset manifests are file-backed, not database rows. The database stores `dataset_id`; artifact paths and checksums live under the configured dataset root.
- Modal deployment currently supports the default app, function, volume, and mount names.
- The dashboard has a configured production URL fallback in code, but that deployment is not currently available; use the local dashboard commands above unless a new deployment is published.
- No root-level license file is present in this repository.

## Support

Open issues or pull requests on [GitHub](https://github.com/tsilva/sigmaevolve) with the track id, trial id, command output, and relevant environment names. Do not include secret values.
