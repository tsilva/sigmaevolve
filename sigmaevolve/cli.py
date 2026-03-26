from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from dataclasses import asdict
from typing import Any, TextIO

from sigmaevolve import build_system
from sigmaevolve.cli_parser import build_cli_parser, json_arg, load_track_definition
from sigmaevolve.cli_reporting import CliReconcileReporter, LaunchSummary, result_payload
from sigmaevolve.env import load_env_file
from sigmaevolve.models import ACTIVE_STATUSES
from sigmaevolve.modal_support import (
    create_modal_launcher,
    deploy_modal_app,
    sync_dataset_to_modal,
)
from sigmaevolve.orchestrator import InlineRunnerLauncher, RecordingLauncher
from sigmaevolve.runner import RunnerService
from sigmaevolve.wandb_support import collect_wandb_env


logger = logging.getLogger(f"{__name__}.stderr")
stdout_logger = logging.getLogger(f"{__name__}.stdout")


def _configure_stream_logger(stream_logger: logging.Logger, stream: TextIO) -> None:
    for handler in list(stream_logger.handlers):
        stream_logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    stream_logger.addHandler(handler)
    stream_logger.setLevel(logging.INFO)
    stream_logger.propagate = False


def _make_system(args) -> Any:
    if not args.database_url:
        raise RuntimeError("A Postgres database URL is required. Set DATABASE_URL.")
    system = build_system(
        database_url=args.database_url,
        dataset_root=args.dataset_root,
        openrouter_api_key=args.openrouter_api_key,
    )
    if args.launcher == "inline":
        runner = RunnerService(system.repository, system.dataset_manager)
        launcher = InlineRunnerLauncher(runner)
    elif args.launcher == "modal":
        if args.command == "launch" and args.database_url.startswith("sqlite"):
            raise RuntimeError("Modal launcher requires a network-accessible database URL; sqlite is not supported.")
        launcher = create_modal_launcher(
            app_name=args.modal_app_name,
            function_name=args.modal_function_name,
            database_url=args.database_url,
            dataset_root=args.modal_dataset_mount,
            environment_name=args.modal_environment_name,
            wandb_env=collect_wandb_env(),
        )
    else:
        launcher = RecordingLauncher()
    system.launcher = launcher
    system.orchestrator.launcher = launcher
    return system


def _print_json(payload: Any) -> None:
    stdout_logger.info("%s", json.dumps(payload, indent=2, sort_keys=True, default=str))


def _ensure_dataset_prepared(system, dataset_id: str) -> tuple[Any, bool]:
    dataset = system.repository.get_dataset(dataset_id)
    manifest_missing = dataset is None or dataset.manifest_path is None or not Path(dataset.manifest_path).exists()
    if manifest_missing:
        return system.prepare_dataset(dataset_id), True
    return dataset, False


def _launch_pass_settings(system, track_id: str, *, target_count: int, daemon: bool) -> tuple[int, int]:
    queue_count = system.repository.count_trials(track_id, statuses={"queued"})
    active_count = system.repository.count_trials(track_id, statuses=ACTIVE_STATUSES)
    if not daemon:
        return max(queue_count, target_count), active_count + target_count
    needed_slots = max(0, target_count - active_count)
    return max(queue_count, needed_slots), target_count


def _run_launch_pass(system, track_id: str, reporter: CliReconcileReporter, *, target_count: int, daemon: bool):
    ready_queue_threshold, max_parallelism = _launch_pass_settings(
        system,
        track_id,
        target_count=target_count,
        daemon=daemon,
    )
    return system.reconcile_track(
        track_id,
        reporter=reporter,
        ready_queue_threshold=ready_queue_threshold,
        max_parallelism=max_parallelism,
    )


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


def _suggest_launch_command(args, track_id: str, *, count: int = 1) -> str:
    command = [
        sys.executable,
        "-m",
        "sigmaevolve.cli",
        "--dataset-root",
        args.dataset_root,
        "--launcher",
        args.launcher,
    ]
    if args.launcher == "modal":
        command.extend(
            [
                "--modal-app-name",
                args.modal_app_name,
                "--modal-function-name",
                args.modal_function_name,
                "--modal-dataset-mount",
                args.modal_dataset_mount,
            ]
        )
        if args.modal_environment_name:
            command.extend(["--modal-environment-name", args.modal_environment_name])
    command.extend(["launch", track_id, str(count)])
    return shlex.join(command)


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
    logger.info("Loading track definition from %s.", args.track_file)
    name, dataset_id, policy = load_track_definition(args.track_file)
    logger.info("Ensuring dataset %s is prepared.", dataset_id)
    dataset, prepared_now = _ensure_dataset_prepared(system, dataset_id)
    if prepared_now:
        logger.info("Prepared dataset %s at %s.", dataset_id, dataset.manifest_path)
    else:
        logger.info("Reusing prepared dataset %s at %s.", dataset_id, dataset.manifest_path)
    logger.info("Creating track for dataset %s and seeding the baseline trial.", dataset_id)
    track = system.create_track(name, dataset_id, policy)
    logger.info("Created track %s.", track.track_id)
    logger.info("Run it with:\n%s", _suggest_launch_command(args, track.track_id))
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


def cmd_launch(args) -> int:
    system = _make_system(args)
    reporter = CliReconcileReporter()
    if not args.daemon:
        result = _run_launch_pass(system, args.track_id, reporter, target_count=args.count, daemon=False)
        payload = result_payload(result)
        payload["mode"] = "count"
        payload["requested_launch_count"] = args.count
        _print_json(payload)
        return 0

    summary = LaunchSummary(mode="daemon", cycles_completed=0)
    controller = system.start_track_controller(
        args.track_id,
        reporter=reporter,
        max_parallelism=args.count,
    )
    try:
        while True:
            summary.cycles_completed += 1
            if args.max_cycles is not None and summary.cycles_completed >= args.max_cycles:
                summary.stopped_reason = "max_cycles_reached"
                break
            time.sleep(args.poll_interval_sec)
    except KeyboardInterrupt:
        summary.stopped_reason = "keyboard_interrupt"
    finally:
        controller.stop()

    result = controller.result
    summary.generated_count = len(result.generated_trial_ids)
    summary.launched_count = len(result.launched_trial_ids)
    summary.duplicate_count = len(result.duplicate_trial_ids)
    summary.stale_count = len(result.stale_trial_ids)
    summary.requeued_count = len(result.requeued_trial_ids)
    summary.error_count = len(result.errors)

    payload = asdict(summary)
    payload["target_running"] = args.count
    _print_json(payload)
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
    scorer_config = json_arg(args.scorer_json)
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
    return build_cli_parser(
        handlers={
            "prepare_dataset": cmd_prepare_dataset,
            "create_track": cmd_create_track,
            "launch": cmd_launch,
            "list_trials": cmd_list_trials,
            "sample_context": cmd_sample_context,
            "rescore": cmd_rescore,
            "modal_deploy": cmd_modal_deploy,
            "modal_sync_dataset": cmd_modal_sync_dataset,
        }
    )


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    _configure_stream_logger(logger, sys.stderr)
    _configure_stream_logger(stdout_logger, sys.stdout)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        logger.error("error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
