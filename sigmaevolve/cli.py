from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from sigmaevolve import build_system
from sigmaevolve.env import load_env_file
from sigmaevolve.modal_support import (
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_MODAL_DATASET_MOUNT,
    DEFAULT_MODAL_DATASET_VOLUME,
    DEFAULT_MODAL_FUNCTION_NAME,
    create_modal_launcher,
    deploy_modal_app,
    sync_dataset_to_modal,
)
from sigmaevolve.orchestrator import InlineRunnerLauncher, RecordingLauncher
from sigmaevolve.runner import RunnerService


def _json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object.")
    return parsed


def _load_policy(policy_json: str | None, policy_file: str | None) -> dict[str, Any]:
    if policy_json and policy_file:
        raise argparse.ArgumentTypeError("Use either --policy-json or --policy-file, not both.")
    if policy_file:
        parsed = json.loads(Path(policy_file).read_text())
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError("Policy file must contain a JSON object.")
        return parsed
    return _json_arg(policy_json)


def _default_database_url() -> str:
    return os.getenv("SIGMAEVOLVE_DATABASE_URL") or os.getenv("DATABASE_URL") or ""


def _make_system(args) -> Any:
    if not args.database_url:
        raise RuntimeError("A Postgres database URL is required. Set SIGMAEVOLVE_DATABASE_URL or DATABASE_URL.")
    system = build_system(
        database_url=args.database_url,
        dataset_root=args.dataset_root,
        openrouter_api_key=args.openrouter_api_key,
    )
    if args.launcher == "inline":
        runner = RunnerService(system.repository, system.dataset_manager)
        launcher = InlineRunnerLauncher(runner)
    elif args.launcher == "modal":
        if args.command == "reconcile" and args.database_url.startswith("sqlite"):
            raise RuntimeError("Modal launcher requires a network-accessible database URL; sqlite is not supported.")
        launcher = create_modal_launcher(
            app_name=args.modal_app_name,
            function_name=args.modal_function_name,
            database_url=args.database_url,
            dataset_root=args.modal_dataset_mount,
            environment_name=args.modal_environment_name,
        )
    else:
        launcher = RecordingLauncher()
    system.launcher = launcher
    system.orchestrator.launcher = launcher
    return system


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


class _CliReconcileReporter:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.started_at = time.monotonic()
        self.requested = 0
        self.max_failures = 0

    def _elapsed(self) -> str:
        seconds = time.monotonic() - self.started_at
        return f"{seconds:5.1f}s"

    def _log(self, message: str) -> None:
        print(f"[{self._elapsed()}] {message}", file=self.stream, flush=True)

    def _progress_line(self, completed: int, requested: int, failures: int, max_failures: int, in_flight: int) -> str:
        if requested <= 0:
            return "Queue fill: nothing to generate."
        width = 20
        filled = int(width * completed / requested)
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"Queue fill [{bar}] {completed}/{requested} accepted"
            f" | failures {failures}/{max_failures}"
            f" | in flight {in_flight}"
        )

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if event == "reconcile_started":
            self._log(f"Reconciling {payload['track_id']} with launcher={payload['launcher']}.")
            return
        if event == "sweep_completed":
            self._log(
                f"Sweep complete: requeued={payload['requeued_count']}, stale={payload['stale_count']}."
            )
            return
        if event == "queue_fill_started":
            self.requested = int(payload["requested_generations"])
            self.max_failures = int(payload["max_failures"])
            self._log(
                f"Queue below target: queued={payload['queued_count']} target={payload['target_queue_count']}."
            )
            self._log(self._progress_line(0, self.requested, 0, self.max_failures, 0))
            return
        if event == "queue_fill_skipped":
            self._log(
                f"Queue already full: queued={payload['queued_count']} target={payload['target_queue_count']}."
            )
            return
        if event == "generation_scheduled":
            self._log(
                "Scheduled generation slot "
                f"{payload['slot_index'] + 1}/{max(self.requested, 1)} "
                f"(attempt {payload['duplicate_retry_count']}, generation_index={payload['generation_index']})."
            )
            return
        if event == "generation_accepted":
            self._log(
                f"Accepted candidate for slot {payload['slot_index'] + 1}: {payload['trial_id']}."
            )
            self._log(
                self._progress_line(
                    int(payload["completed"]),
                    int(payload["requested"]),
                    int(payload["failures"]),
                    int(payload["max_failures"]),
                    int(payload["in_flight"]),
                )
            )
            return
        if event == "generation_duplicate":
            self._log(
                "Duplicate candidate for slot "
                f"{payload['slot_index'] + 1} "
                f"(existing={payload['existing_trial_id']}, attempt={payload['duplicate_retry_count']})."
            )
            self._log(
                self._progress_line(
                    int(payload["completed"]),
                    int(payload["requested"]),
                    int(payload["failures"]),
                    int(payload["max_failures"]),
                    int(payload["in_flight"]),
                )
            )
            return
        if event == "generation_failed":
            detail = f": {payload['detail']}" if payload.get("detail") else ""
            self._log(
                "Generation failed for slot "
                f"{payload['slot_index'] + 1} "
                f"(reason={payload['reason']}, attempt={payload['duplicate_retry_count']}){detail}"
            )
            self._log(
                self._progress_line(
                    int(payload["completed"]),
                    int(payload["requested"]),
                    int(payload["failures"]),
                    int(payload["max_failures"]),
                    int(payload["in_flight"]),
                )
            )
            return
        if event == "queue_fill_completed":
            self._log(
                f"Queue fill complete: accepted={payload['completed']}/{payload['requested']} "
                f"with failures={payload['failures']}/{payload['max_failures']}."
            )
            return
        if event == "queue_fill_stopped":
            self._log(
                f"Queue fill stopped at {payload['completed']}/{payload['requested']} "
                f"after reaching failure budget {payload['failures']}/{payload['max_failures']}."
            )
            return
        if event == "launch_batch_started":
            self._log(
                f"Launching reserved trials: count={payload['reserved_count']} max_parallelism={payload['max_parallelism']}."
            )
            return
        if event == "trial_launch_started":
            self._log(f"Launching trial {payload['trial_id']}...")
            return
        if event == "trial_launched":
            launch_metadata = payload.get("launch_metadata") or {}
            run_url = launch_metadata.get("run_url")
            suffix = f" ({run_url})" if isinstance(run_url, str) and run_url else ""
            self._log(f"Launched trial {payload['trial_id']}{suffix}.")
            return
        if event == "trial_launch_failed":
            self._log(f"Launch failed for {payload['trial_id']}: {payload['detail']}")
            return
        if event == "reconcile_finished":
            self._log(
                "Reconcile finished: "
                f"generated={payload['generated_count']} "
                f"launched={payload['launched_count']} "
                f"duplicates={payload['duplicate_count']} "
                f"generation_failures={payload['failed_generation_count']} "
                f"errors={payload['error_count']}."
            )
            return


def _trial_diagnostics(metrics_json: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics_json or {}
    return {
        "accuracy": metrics.get("accuracy"),
        "best_accuracy": metrics.get("best_accuracy", metrics.get("accuracy")),
        "time_to_best_eval_sec": metrics.get("time_to_best_eval_sec"),
        "last_completed_eval_sec": metrics.get("last_completed_eval_sec"),
        "timed_out": metrics.get("timed_out", False),
        "time_since_last_eval_sec": metrics.get("time_since_last_eval_sec"),
        "had_unscored_work_at_timeout": metrics.get("had_unscored_work_at_timeout", False),
        "last_phase": metrics.get("last_phase"),
    }


def cmd_prepare_dataset(args) -> int:
    system = _make_system(args)
    record = system.prepare_dataset(args.dataset_id)
    _print_json(
        {
            "dataset_id": record.dataset_id,
            "manifest_path": record.manifest_path,
            "created_at": record.created_at,
        }
    )
    return 0


def cmd_create_track(args) -> int:
    system = _make_system(args)
    policy = _load_policy(args.policy_json, args.policy_file)
    track = system.create_track(args.name, args.dataset_id, policy)
    _print_json(
        {
            "track_id": track.track_id,
            "name": track.name,
            "dataset_id": track.dataset_id,
            "policy_json": track.policy_json,
            "created_at": track.created_at,
        }
    )
    return 0


def cmd_reconcile(args) -> int:
    system = _make_system(args)
    reporter = _CliReconcileReporter(sys.stderr)
    result = system.reconcile_track(args.track_id, reporter=reporter)
    _print_json(
        {
            "generated_trial_ids": result.generated_trial_ids,
            "launched_trial_ids": result.launched_trial_ids,
            "duplicate_hashes": result.duplicate_hashes,
            "requeued_trial_ids": result.requeued_trial_ids,
            "stale_trial_ids": result.stale_trial_ids,
            "errors": result.errors,
        }
    )
    return 0


def cmd_list_trials(args) -> int:
    system = _make_system(args)
    statuses = set(args.status) if args.status else None
    trials = system.repository.list_trials(args.track_id, statuses=statuses)
    _print_json(
        [
            {
                "trial_id": trial.trial_id,
                "status": trial.status,
                "outcome_reason": trial.outcome_reason,
                "score": trial.score,
                **_trial_diagnostics(trial.metrics_json),
                "dispatch_attempts": trial.dispatch_attempts,
                "runner_id": trial.runner_id,
                "created_at": trial.created_at,
                "started_at": trial.started_at,
                "finished_at": trial.finished_at,
                "script_hash": trial.script_hash,
                "provenance_json": trial.provenance_json,
                "metrics_json": trial.metrics_json,
                "error_json": trial.error_json,
            }
            for trial in trials
        ]
    )
    return 0


def cmd_sample_context(args) -> int:
    system = _make_system(args)
    context = system.sample_trial_context(args.track_id, limit=args.limit)
    _print_json(
        [
            {
                "trial_id": trial.trial_id,
                "score": trial.score,
                "outcome_reason": trial.outcome_reason,
                **_trial_diagnostics(trial.metrics_json),
                "metrics_json": trial.metrics_json,
                "provenance_json": trial.provenance_json,
                "source": trial.source,
            }
            for trial in context
        ]
    )
    return 0


def cmd_rescore(args) -> int:
    system = _make_system(args)
    scorer_config = _json_arg(args.scorer_json)
    target = "all" if args.all_tracks else args.track_id
    result = system.rescore(target, scorer_config)
    _print_json(
        {
            "updated_trials": result.updated_trials,
            "scorer_config": result.scorer_config,
        }
    )
    return 0


def cmd_modal_deploy(args) -> int:
    payload = deploy_modal_app(
        app_name=args.modal_app_name,
        function_name=args.modal_function_name,
        dataset_volume_name=args.modal_dataset_volume,
        dataset_mount_path=args.modal_dataset_mount,
        environment_name=args.modal_environment_name,
    )
    _print_json(payload)
    return 0


def cmd_modal_sync_dataset(args) -> int:
    payload = sync_dataset_to_modal(
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root,
        volume_name=args.modal_dataset_volume,
        environment_name=args.modal_environment_name,
    )
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigmaevolve")
    parser.add_argument(
        "--database-url",
        default=_default_database_url(),
        help="SQLAlchemy database URL. Defaults to SIGMAEVOLVE_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--dataset-root",
        default="./artifacts/datasets",
        help="Root directory for prepared datasets. Default: ./artifacts/datasets",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--launcher",
        choices=["recording", "inline", "modal"],
        default="modal",
        help="Use modal to spawn remote runner jobs by default, inline to execute locally, or recording to reserve only.",
    )
    parser.add_argument(
        "--modal-app-name",
        default=DEFAULT_MODAL_APP_NAME,
        help=f"Deployed Modal app name. Default: {DEFAULT_MODAL_APP_NAME}",
    )
    parser.add_argument(
        "--modal-function-name",
        default=DEFAULT_MODAL_FUNCTION_NAME,
        help=f"Deployed Modal function name. Default: {DEFAULT_MODAL_FUNCTION_NAME}",
    )
    parser.add_argument(
        "--modal-dataset-volume",
        default=DEFAULT_MODAL_DATASET_VOLUME,
        help=f"Modal Volume name for dataset artifacts. Default: {DEFAULT_MODAL_DATASET_VOLUME}",
    )
    parser.add_argument(
        "--modal-dataset-mount",
        default=DEFAULT_MODAL_DATASET_MOUNT,
        help=f"Dataset mount path inside Modal containers. Default: {DEFAULT_MODAL_DATASET_MOUNT}",
    )
    parser.add_argument(
        "--modal-environment-name",
        default=None,
        help="Optional Modal environment name.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_dataset = subparsers.add_parser("prepare-dataset", help="Prepare and register a dataset.")
    prepare_dataset.add_argument("dataset_id")
    prepare_dataset.set_defaults(func=cmd_prepare_dataset)

    create_track = subparsers.add_parser("create-track", help="Create a track and seed the baseline trial.")
    create_track.add_argument("dataset_id")
    create_track.add_argument("--name", default=None)
    policy_source = create_track.add_mutually_exclusive_group()
    policy_source.add_argument(
        "--policy-json",
        default=None,
        help="JSON object overriding track policy fields.",
    )
    policy_source.add_argument(
        "--policy-file",
        default=None,
        help="Path to a JSON file overriding track policy fields.",
    )
    create_track.set_defaults(func=cmd_create_track)

    reconcile = subparsers.add_parser("reconcile", help="Run one reconciliation pass for a track.")
    reconcile.add_argument("track_id")
    reconcile.set_defaults(func=cmd_reconcile)

    list_trials = subparsers.add_parser("list-trials", help="List trials for a track.")
    list_trials.add_argument("track_id")
    list_trials.add_argument(
        "--status",
        action="append",
        choices=["queued", "dispatching", "active", "finished"],
        help="Filter by one or more statuses.",
    )
    list_trials.set_defaults(func=cmd_list_trials)

    sample_context = subparsers.add_parser("sample-context", help="Show successful finished trials used for generation context.")
    sample_context.add_argument("track_id")
    sample_context.add_argument("--limit", type=int, default=5)
    sample_context.set_defaults(func=cmd_sample_context)

    rescore = subparsers.add_parser("rescore", help="Rescore finished trials without rerunning training.")
    target = rescore.add_mutually_exclusive_group(required=True)
    target.add_argument("--track-id")
    target.add_argument("--all-tracks", action="store_true")
    rescore.add_argument(
        "--scorer-json",
        required=True,
        help='JSON object such as \'{"primary_metric":"accuracy"}\'.',
    )
    rescore.set_defaults(func=cmd_rescore)

    modal_deploy = subparsers.add_parser("modal-deploy", help="Deploy the Modal runner app.")
    modal_deploy.set_defaults(func=cmd_modal_deploy)

    modal_sync_dataset = subparsers.add_parser("modal-sync-dataset", help="Upload a prepared dataset to the Modal dataset volume.")
    modal_sync_dataset.add_argument("dataset_id")
    modal_sync_dataset.set_defaults(func=cmd_modal_sync_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
