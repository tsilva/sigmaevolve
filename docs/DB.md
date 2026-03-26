# Database Schema

This document describes the application tables currently defined in `sigmaevolve/storage.py`.

## Overview

SigmaEvolve currently persists three tables:

- `datasets`
- `tracks`
- `trials`

## `datasets`

Purpose: registry of prepared datasets and the manifest path used to reload dataset metadata and file locations.

| Column | Type | Nullable | Description |
| --- | --- | --- | --- |
| `dataset_id` | string | No | Primary key. Stable dataset identifier such as `mnist:v1`. Referenced by `tracks.dataset_id`. |
| `manifest_path` | text | Yes | Filesystem path to the dataset manifest JSON. The manifest records split paths, checksums, fingerprint, and dataset metadata. |
| `created_at` | timestamptz | No | Timestamp for when the dataset record was registered or refreshed. |

## `tracks`

Purpose: top-level evolution tracks. A track binds one dataset to one policy configuration and groups all generated and executed trials under that configuration.

| Column | Type | Nullable | Description |
| --- | --- | --- | --- |
| `track_id` | string | No | Primary key. Unique identifier for the track. |
| `name` | string | Yes | Optional human-readable label shown in CLI and dashboard output. |
| `dataset_id` | string | No | Foreign key to `datasets.dataset_id`. Identifies which dataset this track uses. |
| `policy_json` | JSON | No | Persisted track policy. Common keys are `epochs`, `dispatch_ttl_sec`, `heartbeat_interval_sec`, `stale_ttl_sec`, `max_dispatch_retries`, `scorer_settings`, `sampling_settings`, and `generation_backend`. |
| `created_at` | timestamptz | No | Timestamp for when the track was created. |

### `tracks.policy_json`

Common persisted keys:

| Key | Description |
| --- | --- |
| `epochs` | Number of training epochs a runner should attempt. |
| `dispatch_ttl_sec` | How long a reserved dispatch remains valid before it is considered expired. |
| `heartbeat_interval_sec` | Expected interval between runner heartbeats. |
| `stale_ttl_sec` | How long an active trial may go without a heartbeat before it is marked stale. |
| `max_dispatch_retries` | Maximum number of dispatch attempts before a dispatching trial is marked stale instead of requeued. |
| `scorer_settings` | Scoring configuration used to derive `trials.score`, usually including `primary_metric`. |
| `sampling_settings` | Configuration for parent-trial selection during generation. The persisted shape currently contains only the RNG seed. |
| `generation_backend` | LLM generation backend configuration, including backend name, model selection policy, seed, and model pool entries. |

## `trials`

Purpose: every candidate program and run state for a track, including queued work, active executions, finished results, duplicate candidates, generation failures, and stale/error cases.

| Column | Type | Nullable | Description |
| --- | --- | --- | --- |
| `trial_id` | string | No | Primary key. Unique identifier for the trial. |
| `track_id` | string | No | Foreign key to `tracks.track_id`. Identifies which track owns the trial. |
| `source` | text | No | Normalized candidate source code. For generation-attempt failures this may contain a diagnostic stub instead of runnable source. |
| `script_hash` | string(64) | No | Hash of the normalized source. Used to deduplicate candidates within the same track. |
| `provenance_json` | JSON | No | Provenance and generation trace for how this trial was created. Includes prompt history for non-baseline trials. |
| `status` | string | No | Trial lifecycle state: `queued`, `dispatching`, `active`, `finished`, or `error`. |
| `outcome_reason` | string | Yes | Terminal outcome classification such as `succeeded`, `timeout`, `duplicate`, `crashed`, `eval_failed`, `stale`, or `generation_failed`. |
| `dispatch_token` | string | Yes | Reservation token issued when a queued trial is reserved for launch. Required when a runner claims the trial. |
| `dispatch_deadline_at` | timestamptz | Yes | Expiration time for the current dispatch reservation. |
| `runner_id` | string | Yes | Identifier of the runner process or remote worker currently responsible for the trial. |
| `heartbeat_at` | timestamptz | Yes | Last heartbeat timestamp reported by the active runner. |
| `started_at` | timestamptz | Yes | Timestamp when execution started. |
| `finished_at` | timestamptz | Yes | Timestamp when the trial reached a terminal state. |
| `metrics_json` | JSON | Yes | Latest active metrics snapshot or final evaluation metrics for the trial. |
| `score` | float | No | Derived numeric score used for ranking and selection. Usually computed from `metrics_json` and scorer settings. Defaults to `0`. |
| `error_json` | JSON | Yes | Structured error or diagnostic payload associated with failures, stale runs, or heartbeat error signals. |
| `dispatch_attempts` | integer | No | Number of times the system has attempted to dispatch this trial. Defaults to `0`. |
| `created_at` | timestamptz | No | Timestamp when the trial row was created. |

### `trials.provenance_json`

Required for LLM-generated non-baseline trials:

| Key | Description |
| --- | --- |
| `backend` | Generation backend identifier. Current supported non-baseline backend is `openrouter`. Baseline trials use `baseline`. |
| `model` | Model name used for generation. |
| `candidate_kind` | Candidate family identifier, currently used to distinguish generated strategy variants. |
| `generation_config` | Concrete generation settings captured for the request, such as model, temperature, max tokens, seed, selection mode, or model pool. |
| `request_messages` | Recorded prompt messages sent through the LLM request path. This is the required prompt provenance for non-baseline trials. |
| `context_trial_ids` | Parent or context trial ids used to build the generation prompt. |

Common optional keys:

| Key | Description |
| --- | --- |
| `generation_index` | Sequence index for the generation attempt within a controller reconciliation pass. |
| `duplicate_retry_count` | How many duplicate generations were retried before this candidate. |
| `provider_response_id` | Provider-side response id when available. |
| `launcher` | Launch metadata recorded after scheduling a remote or local run, such as Modal run identifiers and URLs. |
| `wandb` | Weights & Biases run metadata recorded by the runner, including project, entity, run id, run name, and run URL for the latest execution attempt. |
| `generation` | Generation trace payload including captured prompts, provider response text, provider finish metadata, generated source, assertion results, and candidate hash. |

### `trials.metrics_json`

Common keys observed in persisted trial metrics:

| Key | Description |
| --- | --- |
| `accuracy` | Validation accuracy for the best scored evaluation. |
| `best_accuracy` | Best validation accuracy reached across completed evaluations. |
| `train_loss` | Mean training loss from the epoch that produced the persisted best evaluation snapshot. |
| `train_acc` | Training accuracy from the epoch that produced the persisted best evaluation snapshot. |
| `val_loss` | Validation loss from the epoch that produced the persisted best evaluation snapshot. |
| `val_acc` | Validation accuracy from the epoch that produced the persisted best evaluation snapshot. This mirrors `accuracy`. |
| `time_to_best_eval_sec` | Elapsed runtime until the best evaluation was produced. |
| `best_eval_index` | Index of the evaluation that produced the best metrics. |
| `best_eval_epoch` | Epoch corresponding to the best evaluation. |
| `best_eval_path` | Path to the artifact for the best evaluation. |
| `last_completed_eval_sec` | Elapsed runtime at the most recent completed evaluation. |
| `last_completed_eval_index` | Index of the most recent completed evaluation. |
| `timed_out` | Whether the run ended due to timeout. |
| `time_since_last_eval_sec` | Runtime elapsed after the last completed evaluation. Useful for timeout diagnostics. |
| `had_unscored_work_at_timeout` | Indicates the process timed out while additional training work had not yet produced a scored evaluation. |
| `last_phase` | Last reported phase such as `train`, `eval`, or `finished`. |
| `eval_count` | Number of completed evaluations observed so far or at finish time. |
| `process_elapsed_sec` | Total process runtime seen when the metrics snapshot was written. |

Debug or extended keys may also be present if emitted by the runner debug payload or evaluation pipeline.

### `trials.error_json`

Common keys observed in persisted error payloads:

| Key | Description |
| --- | --- |
| `reason` | Stable machine-readable failure reason, such as `heartbeat_stale`, `dispatch_deadline_expired`, `provider_request_failed`, or `train_script_contract_violation`. |
| `detail` | Extra human-readable detail about the failure. |
| `stderr` | Captured stderr text for process failures when available. |
| `returncode` | Process exit code when a subprocess failed. |
| `error_type` | Normalized error classification derived from `outcome_reason` and raw failure details. |
| `finish_reason` | Provider-reported completion stop reason when preserved for generation failures, such as `length` for output truncated by the token limit. |
| `native_finish_reason` | Provider-native stop reason when surfaced alongside `finish_reason`. |

Additional backend-specific keys may be present for generation failures or launcher exceptions.

## Constraints and Indexes

### Foreign keys

- `tracks.dataset_id` references `datasets.dataset_id`
- `trials.track_id` references `tracks.track_id`

### Unique constraints

- `trials(track_id, script_hash)` is unique so the same candidate source cannot be persisted twice in the same track.

### Indexes

- `ix_trials_track_created_at_desc` on `trials(track_id, created_at desc)`
- `ix_trials_track_status_created_at_desc` on `trials(track_id, status, created_at desc)`

These indexes support track-scoped dashboard listings and controller queries over queued, active, and recent trials.
