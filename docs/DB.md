# Database Schema

This document describes the live database schema defined in `sigmaevolve/storage.py`.

## Overview

SigmaEvolve persists two tables:

- `tracks`
- `trials`

Dataset manifests are no longer registered in the database. Dataset preparation, verification, and manifest lookup now live entirely under `DatasetManager` and the on-disk manifest path for each `dataset_id`.

## `tracks`

Purpose: top-level evolution tracks. A track binds one dataset identifier to one normalized policy configuration and groups all queued, active, and terminal trials for that configuration.

| Column | Type | Nullable | Description |
| --- | --- | --- | --- |
| `track_id` | string | No | Primary key. Stable track identifier. |
| `dataset_id` | string | No | Dataset identifier such as `mnist:v1`. This is a plain string, not a database foreign key. |
| `policy_json` | JSON | No | Persisted normalized track policy. |
| `created_at` | timestamptz | No | Timestamp when the track was created. |

### `tracks.policy_json`

Persisted keys:

| Key | Description |
| --- | --- |
| `epochs` | Number of training epochs a runner should attempt. |
| `dispatch_ttl_sec` | How long a reserved dispatch remains valid before it is considered expired. |
| `heartbeat_interval_sec` | Expected interval between runner heartbeats. |
| `stale_ttl_sec` | How long an active trial may go without a heartbeat before it is marked stale. |
| `max_dispatch_retries` | Maximum number of dispatch attempts before a dispatching trial is marked stale instead of requeued. |
| `sampling_seed` | RNG seed used when sampling successful parent trials for generation context. |
| `generation_backend` | OpenRouter generation settings. The persisted shape does not include a `backend` field. |

### `tracks.policy_json.generation_backend`

Common persisted keys:

| Key | Description |
| --- | --- |
| `selection` | Model-pool selection strategy such as `weighted_random`, `random`, or `round_robin`. |
| `seed` | RNG seed used by stochastic model-pool selection. |
| `model_pool` | List of model config entries. Each entry typically includes `model`, `temperature`, `max_tokens`, `retry_count`, and optionally `probability`. |

Removed policy fields are not supported in the live schema: `scorer_settings`, `sampling_settings`, `modal_gpu_preferences`, and `generation_backend.backend`.

## `trials`

Purpose: every candidate program and run state for a track, including queued work, active executions, finished results, duplicate candidates, stale trials, and generation-attempt diagnostics.

| Column | Type | Nullable | Description |
| --- | --- | --- | --- |
| `trial_id` | string | No | Primary key. Unique identifier for the trial. |
| `track_id` | string | No | Foreign key to `tracks.track_id`. |
| `source` | text | No | Normalized candidate source code. Generation-attempt diagnostic rows store a small non-runnable stub that points reviewers back to `provenance_json`. |
| `script_hash` | string(64) | No | Hash of the normalized source. Used to deduplicate candidates within a track. |
| `provenance_json` | JSON | No | Persisted provenance for how the trial was created. Non-baseline trials must retain recorded prompt messages. |
| `status` | string | No | Lifecycle state: `queued`, `dispatching`, `active`, `finished`, or `error`. |
| `outcome_reason` | string | Yes | Terminal outcome such as `succeeded`, `timeout`, `duplicate`, `crashed`, `eval_failed`, `stale`, or `generation_failed`. |
| `dispatch_token` | string | Yes | Reservation token issued when a queued trial is reserved for launch. |
| `dispatch_deadline_at` | timestamptz | Yes | Expiration time for the current dispatch reservation. |
| `runner_id` | string | Yes | Identifier of the active runner or remote worker. |
| `heartbeat_at` | timestamptz | Yes | Last heartbeat timestamp reported by the active runner. |
| `started_at` | timestamptz | Yes | Timestamp when execution started. |
| `finished_at` | timestamptz | Yes | Timestamp when the trial reached a terminal state. |
| `metrics_json` | JSON | Yes | Slimmed active metrics snapshot or final evaluation metrics for the trial. |
| `error_json` | JSON | Yes | Slimmed structured error payload for failed, stale, or diagnostic rows. |
| `dispatch_attempts` | integer | No | Number of dispatch attempts for this trial. Defaults to `0`. |
| `created_at` | timestamptz | No | Timestamp when the row was created. |

`score` is no longer stored in the database. It is derived at read time from `metrics_json.accuracy`, with `0.0` for rows that have no persisted accuracy.

`error_type` is no longer stored in `error_json`. It is derived at read time from `outcome_reason` plus the slimmed error payload.

### `trials.provenance_json`

Required for LLM-generated non-baseline trials:

| Key | Description |
| --- | --- |
| `backend` | Generation backend identifier. Baseline trials use `baseline`; generated trials use `openrouter`. |
| `model` | Model name used for generation. |
| `candidate_kind` | Candidate family identifier. |
| `generation_config` | Concrete generation settings captured for the request. |
| `request_messages` | Recorded prompt messages sent through the LLM request path. This is the required prompt provenance for non-baseline trials. |
| `context_trial_ids` | Parent or context trial ids used to build the generation prompt. |

Optional persisted keys:

| Key | Description |
| --- | --- |
| `launcher` | Slim launcher metadata. Only `run_id` and `run_url` are persisted. |
| `wandb` | Slim Weights & Biases metadata. Persisted keys are `project`, `entity`, `run_id`, `run_name`, and `run_url`. |
| `generation` | Present when generation trace data was captured. Persisted keys are `response_text`, `generated_source`, `assertions_passed`, `assertion_failures`, and `candidate_hash` when those values are available. |

Removed provenance fields are not persisted in the live schema, including `generation_index`, `duplicate_retry_count`, `provider_response_id`, duplicated prompt copies under `generation` such as `system_prompt` and `user_prompt`, provider metadata duplicates such as `provider` and `provider_model`, and launcher cancellation bookkeeping.

### `trials.metrics_json`

Persisted keys:

| Key | Description |
| --- | --- |
| `accuracy` | Validation accuracy for the persisted best-scoring evaluation. This is the ranking metric used for derived score. |
| `val_loss` | Validation loss for the persisted best-scoring evaluation. |
| `time_to_best_eval_sec` | Elapsed runtime until the best evaluation was produced. |
| `eval_count` | Number of completed evaluations observed so far or at finish time. |
| `timed_out` | Whether the run ended due to timeout. |
| `time_since_last_eval_sec` | Runtime elapsed after the last completed evaluation. Useful for timeout diagnostics. |
| `had_unscored_work_at_timeout` | Indicates the process timed out while additional training work had not yet produced a scored evaluation. |
| `last_phase` | Last reported phase such as `train`, `eval`, or `finished`. |

No other runner metrics are persisted in the live schema. Fields such as `best_accuracy`, `train_loss`, `train_acc`, `val_acc`, `best_eval_index`, `best_eval_epoch`, `best_eval_path`, `last_completed_eval_sec`, `last_completed_eval_index`, `process_elapsed_sec`, `epochs_completed`, `epochs_without_improvement`, and `early_stop_epoch` are intentionally dropped before persistence.

### `trials.error_json`

Persisted keys:

| Key | Description |
| --- | --- |
| `reason` | Stable machine-readable failure reason, such as `heartbeat_stale`, `dispatch_deadline_expired`, `provider_request_failed`, or `train_script_contract_violation`. |
| `detail` | Human-readable detail about the failure when available. |
| `stderr` | Captured stderr text for subprocess failures when available. |
| `returncode` | Process exit code when a subprocess failed. |
| `finish_reason` | Provider-reported completion stop reason when preserved for generation failures, such as `length`. |

No other error keys are persisted in the live schema. In particular, `error_type` and `native_finish_reason` are not stored.

## Constraints And Indexes

### Foreign keys

- `trials.track_id` references `tracks.track_id`

### Unique constraints

- `trials(track_id, script_hash)` is unique so the same candidate source cannot be persisted twice in the same track.

### Indexes

- `ix_trials_track_created_at_desc` on `trials(track_id, created_at desc)`
- `ix_trials_track_status_created_at_desc` on `trials(track_id, status, created_at desc)`

These indexes support track-scoped dashboard listings plus controller queries over queued, active, and recent trials.
