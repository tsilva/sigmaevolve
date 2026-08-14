<div align="center">
  <img src="dashboard/logo.png" alt="SigmaEvolve" width="420" />

  **🧬 Evolutionary training runs for immutable datasets 🧬**
</div>

SigmaEvolve is an evolutionary training harness for classification experiments.
It keeps datasets fixed, asks an LLM to mutate only marked regions of a
self-contained training script, and ranks candidates by validation accuracy.

Use the Python CLI to seed tracks, generate auditable candidate trials through
OpenRouter, launch runs locally or on Modal, and store trial state in Postgres.
The companion Next.js dashboard reads the same database for browsing tracks,
source, provenance, metrics, and runner status.

## Install

Requirements: Python 3.11+, `uv`, Node.js 20+, and a reachable Postgres
database.

```bash
git clone git@github.com:tsilva/sigmaevolve.git
cd sigmaevolve
uv sync --extra dev --extra datasets --extra modal
npm --prefix dashboard install
cp dashboard/.env.example dashboard/.env.local
mkdir -p ~/.config/sigmaevolve
$EDITOR ~/.config/sigmaevolve/.env
keyenv doctor
```

Minimum CLI configuration:

```dotenv
SIGMAEVOLVE_DATABASE_URL=postgresql://user:password@host:5432/sigmaevolve
SIGMAEVOLVE_OPENROUTER_API_KEY=...
SIGMAEVOLVE_DATASET_ROOT=./artifacts/datasets
```

Set `DATABASE_URL` in `dashboard/.env.local` to the same database URL.

Seed the bundled MNIST track, run one local candidate, and inspect trial state:

```bash
uv run sigmaevolve create-track sigmaevolve/baselines/mnist.py
uv run sigmaevolve --launcher inline launch <track_id> 1
uv run sigmaevolve list-trials <track_id>
```

The dashboard's private Sentry values declared in `.keyenv.toml` live in macOS
Keychain. Start it from the repo root through `keyenv`; Node reads the injected
values normally from `process.env`:

```bash
keyenv run -- npm --prefix dashboard run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Commands

```bash
uv run sigmaevolve --help                         # show CLI help
uv run sigmaevolve create-track <script.py>       # create a track and seed its baseline trial
uv run sigmaevolve --launcher inline launch <track_id> 1
uv run sigmaevolve --launcher modal launch <track_id> 8 --daemon
uv run sigmaevolve list-trials <track_id>         # print trial rows as JSON
uv run sigmaevolve modal-deploy                   # deploy the default Modal runner app
uv run sigmaevolve modal-sync-dataset <dataset_id>

uv run pytest                                     # run Python tests
uv run ruff check .                               # lint Python
uv run ruff format .                              # format Python
npm --prefix dashboard run test                   # run dashboard tests
npm --prefix dashboard run build                  # build the dashboard
```

## Notes

- Training scripts must include SigmaEvolve metadata and at least one
  `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` region. The bundled baseline is
  [sigmaevolve/baselines/mnist.py](sigmaevolve/baselines/mnist.py).
- Every non-baseline trial must come from the configured LLM prompting pipeline
  and keep recorded prompt provenance. Manual candidate variants are not valid
  experiment inputs.
- Runtime settings are resolved from environment variables. The CLI loads
  `~/.config/sigmaevolve/.env` before commands run.
- `SIGMAEVOLVE_DATABASE_URL` or `DATABASE_URL` is required. Modal runs need a
  network-accessible Postgres URL.
- `SIGMAEVOLVE_OPENROUTER_API_KEY` or `OPENROUTER_API_KEY` is required when
  generating candidates.
- Dataset artifacts are file-backed under `SIGMAEVOLVE_DATASET_ROOT`, which
  defaults to `./artifacts/datasets`.
- Optional Weights & Biases logging uses `WANDB_API_KEY`, `WANDB_PROJECT`,
  `WANDB_ENTITY`, and `WANDB_BASE_URL`.
- The live database contract is documented in [docs/DB.md](docs/DB.md).

## Architecture

![SigmaEvolve architecture diagram](./architecture.png)

## License

No root-level license file is present in this repository.
