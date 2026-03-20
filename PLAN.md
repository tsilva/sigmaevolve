# Plan: Multi-Dataset Classification Evolution System

## Summary
Build a standalone system that repeatedly generates, executes, and evaluates new versions of a self-contained machine learning training script named `train.py`.

The system must:
- support multiple immutable datasets, starting with `mnist:v1` and later `fashion_mnist:v1`
- support one objective family in v1: `classification`
- keep candidate generation outside the execution runner
- keep evaluation and score computation in the harness, not in generated scripts
- use Postgres as the system of record
- use Modal only for remote execution and evaluation
- treat Weights & Biases as optional observability, not correctness-critical infrastructure

The main design goal is to keep the architecture small, flexible across datasets, safe against reward hacking, and operationally correct under concurrent orchestration and runner startup races.

## Core Design
### System roles
- `Orchestrator`
  - runs on cheap compute
  - generates candidate `train.py` scripts
  - deduplicates candidates per track
  - maintains a small ready queue
  - reserves dispatch capacity
  - launches Modal runners
  - sweeps expired dispatches and stale active runs
- `Runner`
  - runs on Modal
  - claims a reserved trial atomically
  - mounts and verifies the dataset
  - executes `train.py` with a fixed time budget
  - loads predictions written by the script
  - computes authoritative classification metrics
  - computes score from those metrics
  - finalizes the trial in Postgres

### Hard invariants
- Every track is pinned to exactly one immutable dataset.
- All trials in a track are comparable only within that track.
- Generated scripts are never authoritative for final metrics, score, status, or runtime accounting.
- Non-success terminal outcomes always receive score `0`.
- Dedupe is scoped to a track, not global.
- Candidate generation must remain outside Modal runners.

## Database Schema
Use three tables only: `datasets`, `tracks`, `trials`.

### `datasets`
Stores immutable dataset registrations.

Fields:
- `dataset_id`
- `manifest_path` optional, only if storage location cannot be derived from `dataset_id`

Rules:
- `dataset_id` is the canonical immutable identifier and combines dataset name and version, for example:
  - `mnist:v1`
  - `fashion_mnist:v1`
- `dataset_id` must uniquely identify:
  - dataset contents
  - canonical train/validation/test splits
  - dataset manifest and fingerprint

### `tracks`
Stores one independent evolutionary search history.

Fields:
- `track_id`
- `name` optional
- `dataset_id`
- `policy_json`
- `created_at`

Rules:
- Every track uses exactly one dataset.
- `policy_json` is immutable after track creation.
- All scheduling, scoring, and generation policy lives in `policy_json`.

### `trials`
Stores generated source, dispatch state, execution state, and authoritative results.

Fields:
- `trial_id`
- `track_id`
- normalized `source`
- `script_hash`
- `provenance_json`
- `status`
- `outcome_reason`
- `dispatch_token`
- `dispatch_deadline_at`
- `runner_id`
- `heartbeat_at`
- `started_at`
- `finished_at`
- `metrics_json`
- `score`
- `error_json`

Allowed `status` values:
- `queued`
- `dispatching`
- `active`
- `finished`

Allowed `outcome_reason` values:
- `succeeded`
- `duplicate`
- `timeout`
- `crashed`
- `eval_failed`
- `stale`

Constraints:
- `datasets.dataset_id` unique
- `tracks.track_id` unique
- `trials.trial_id` unique
- `trials(track_id, script_hash)` unique
- foreign key `tracks.dataset_id -> datasets.dataset_id`
- foreign key `trials.track_id -> tracks.track_id`

## Dataset Model
### Preparation
Implement dataset preparation as a generic workflow:

1. Materialize the dataset.
2. Create canonical train, validation, and test splits.
3. Write a manifest containing split sizes, checksums, and fingerprint.
4. Store dataset assets under a path derived from `dataset_id`, or record `manifest_path`.
5. Register the dataset in `datasets`.

### V1 scope
- V1 supports multiple datasets.
- V1 supports only classification datasets.
- Initial target dataset is `mnist:v1`.
- The next intended dataset is `fashion_mnist:v1`.

## Policy Model
Store all track-level policy in immutable `policy_json`.

`policy_json` must contain:
- `budget_sec`
- `max_parallelism`
- ready queue threshold
- dispatch TTL
- heartbeat interval
- stale TTL
- max dispatch retries
- scorer settings
- sampling settings
- generation backend configuration

`provenance_json` must contain per-trial generation metadata such as:
- generator backend
- model or agent identity
- prompt or template identifiers
- parent trial references
- generation debug metadata

## Scheduling And Atomicity
### Concurrency model
- `queued` does not consume concurrency.
- `dispatching + active` consumes concurrency.

This distinction is mandatory. Counting `queued` against concurrency is incorrect because queued work is backlog, not reserved capacity.

### Reservation flow
The orchestrator must reserve capacity before launching a runner.

For each dispatch attempt:

1. Start a DB transaction.
2. Count `dispatching + active` trials for the track.
3. If that count is already `>= max_parallelism`, stop.
4. Select one `queued` trial using `FOR UPDATE SKIP LOCKED`.
5. Transition that row to `dispatching`.
6. Generate a fresh random `dispatch_token`.
7. Set `dispatch_deadline_at = now + dispatch_ttl`.
8. Commit the transaction.
9. Launch Modal with `trial_id` and `dispatch_token`.

This guarantees that two orchestrators cannot reserve the same queued trial.

### Claim flow
The runner must claim the reserved trial atomically.

Runner claim API:
- `claim_trial(trial_id, dispatch_token, runner_id) -> TrialRecord | None`

Claim semantics:
- perform one atomic update from `dispatching` to `active`
- match on:
  - `trial_id`
  - `status='dispatching'`
  - exact `dispatch_token`
- on success:
  - set `status='active'`
  - set `runner_id`
  - set `started_at`
  - set `heartbeat_at`
- on failure:
  - return no row
  - runner exits immediately without doing work

This guarantees that even if two Modal instances are launched with the same `trial_id`, only one can successfully claim if they share the same dispatch token, and none can claim if the token is stale or wrong.

### Recovery flow
- Expired `dispatching` rows are swept by the orchestrator.
- If `dispatch_deadline_at < now` and no claim occurred:
  - requeue the trial if retry count is below policy limit
  - otherwise mark it `finished` with `outcome_reason='stale'`
- Active trials with stale heartbeats are marked `finished` with `outcome_reason='stale'`

## Orchestration Loop
Implement one orchestrator operation:

- `reconcile_track(track_id) -> ReconcileResult`

Each reconciliation pass must:

1. Sweep expired `dispatching` reservations.
2. Sweep stale `active` trials.
3. Generate more `queued` trials if the ready queue is below the policy threshold.
4. Reserve and dispatch trials until:
   - `dispatching + active == max_parallelism`, or
   - no queued trials remain.

This replaces separate ideation and execution loops with one control loop.

## Candidate Generation
### Flow
For each track:

1. Sample successful finished trials from the same track only.
2. Build generation context from:
   - prior `train.py` source
   - authoritative metrics
   - score
   - dataset metadata
   - classification task contract
   - policy budget
   - dependency allowlist
3. Generate a new candidate `train.py`.
4. Normalize the source.
5. Hash the normalized source with SHA-256.
6. If `(track_id, script_hash)` already exists:
   - do not dispatch remote compute
   - record a duplicate trial or equivalent duplicate audit outcome
7. Otherwise insert a new `queued` trial.

### Source normalization
Before hashing:
- normalize source to UTF-8
- normalize line endings to LF
- ensure exactly one trailing newline

## `train.py` Contract
### Interface
Generated scripts must accept exactly:

- `train.py --config /abs/path/run_config.json`

### `run_config.json`
The runner writes the config file. It must contain:
- dataset directory
- train split path
- validation split path
- time budget in seconds
- random seed
- predictions output path
- debug JSON output path
- dataset metadata needed by the script

### Required script behavior
`train.py` must:
- be self-contained
- use only allowlisted packages preinstalled in the Modal image
- train on the provided training split
- write validation predictions aligned to the provided validation split
- optionally write debug metadata JSON
- exit non-zero on failure

### Non-responsibilities
`train.py` is not authoritative for:
- final metrics
- score
- dedupe
- final outcome classification
- runtime accounting
- final database state

The script may emit self-reported metrics for debugging, but those must never be used for ranking or score.

## Evaluation And Scoring
### Evaluation
The runner performs authoritative evaluation after `train.py` finishes.

For v1 classification:
- load predictions written by the script
- compare them to validation labels for the dataset
- compute authoritative metrics
- store them in `metrics_json`

### Scoring
- derive one scalar `score` from authoritative metrics using scorer settings in `policy_json`
- store `score` in the trial row
- all non-success terminal outcomes receive `score = 0`

### Rescoring
Support rescoring without rerunning training:
- raw authoritative metrics remain immutable
- rescoring updates only derived score fields

## Public Interfaces
Implement these core interfaces:

- `prepare_dataset(dataset_id) -> DatasetRecord`
- `create_track(name?, dataset_id, policy_json) -> TrackRecord`
- `reconcile_track(track_id) -> ReconcileResult`
- `sample_trial_context(track_id, limit) -> list[TrialSummary]`
- `claim_trial(trial_id, dispatch_token, runner_id) -> TrialRecord | None`
- `heartbeat_trial(trial_id, runner_id, meta) -> None`
- `finalize_trial(trial_id, runner_id, outcome_reason, metrics, score, error_info) -> None`
- `rescore(track_id | all, scorer_config) -> MigrationResult`

## Test Plan
### Dataset tests
- register `mnist:v1`
- register `fashion_mnist:v1`
- verify manifests, checksums, and fingerprints are stable
- verify split reproducibility
- verify runner-side dataset verification

### Hashing and dedupe tests
- normalization produces stable hashes across input formatting differences
- duplicate source in the same track is never executed
- identical source in different tracks is allowed

### Scheduling and atomicity tests
- two orchestrators cannot reserve the same queued trial
- `reconcile_track` never exceeds `max_parallelism` while runners are still booting
- two runners launched for the same `trial_id + dispatch_token` cannot both claim
- a runner with a stale or invalid token cannot claim
- expired `dispatching` trials are requeued or marked stale according to policy
- active trials with stale heartbeats are finalized as stale

### Execution tests
- successful script run produces predictions and final metrics
- timeout finalizes with `outcome_reason='timeout'` and `score=0`
- crash finalizes with `outcome_reason='crashed'` and `score=0`
- evaluation failure finalizes with `outcome_reason='eval_failed'` and `score=0`

### Evaluation and scoring tests
- harness computes authoritative classification metrics from validation predictions
- self-reported script metrics do not affect score
- rescoring changes derived score without changing stored raw metrics

### End-to-end tests
- orchestration maintains ready queue and dispatches up to capacity
- successful trials become eligible sampling context
- non-success trials are excluded from future sampling
- duplicate trials never consume remote execution capacity

## Defaults
Use these v1 defaults unless explicitly overridden in `policy_json`:
- classification-only evaluation
- same-track-only sampling
- a mixed exploit/explore sampling strategy
- a simple low-cost generation backend
- one pinned Modal image with:
  - Python
  - PyTorch
  - evaluator code
  - allowlisted dependencies for generated scripts
  - optional W&B client
