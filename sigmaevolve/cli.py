from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from sigmaevolve.core import ACTIVE_STATUSES
from sigmaevolve.env import load_env_file, resolve_runtime_config
from sigmaevolve.execution import RunnerService, collect_wandb_env
from sigmaevolve.modal import (
    create_modal_launcher,
    deploy_modal_app,
    sync_dataset_to_modal,
)
from sigmaevolve.orchestration import InlineRunnerLauncher, build_system
from sigmaevolve.storage import classify_error_type

logger = logging.getLogger(f"{__name__}.stderr")
stdout_logger = logging.getLogger(f"{__name__}.stdout")


def load_track_definition(track_file: str) -> tuple[str, dict[str, Any]]:
    # Load and validate the top-level track definition envelope.
    parsed = json.loads(Path(track_file).read_text())

    # Reject non-object track definitions before reading any fields.
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Track file must contain a JSON object.")

    # Extract the required dataset identifier before reading any policy fields.
    dataset_id = parsed.get("dataset_id")

    # Reject missing or empty dataset identifiers early.
    if not isinstance(dataset_id, str) or not dataset_id:
        raise argparse.ArgumentTypeError(
            "Track file must include a non-empty string dataset_id."
        )

    # Reject the removed track label field so new configs only use the reduced contract.
    if "name" in parsed:
        raise argparse.ArgumentTypeError("Track file name is no longer supported.")

    # Reject the removed legacy policy wrapper before reading policy fields.
    if "policy_json" in parsed:
        raise argparse.ArgumentTypeError(
            "Track file policy_json is no longer supported."
        )

    # Support either an explicit policy object or current top-level policy fields.
    policy = parsed.get("policy")

    # Validate object-shaped policy payloads before copying them.
    if policy is not None:
        # Reject malformed explicit policy objects before copying them.
        if not isinstance(policy, dict):
            raise argparse.ArgumentTypeError("Track file policy must be a JSON object.")
        policy_json = dict(policy)

    # Preserve the current top-level track policy shape when no explicit policy object is present.
    else:
        excluded_fields = {"dataset_id"}
        policy_json = {
            key: value for key, value in parsed.items() if key not in excluded_fields
        }

    return dataset_id, policy_json


def positive_int(value: str) -> int:
    # Enforce strictly positive integer CLI values at parse time.
    parsed = int(value)

    # Reject non-positive integers before handing them to command handlers.
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return parsed


def positive_float(value: str) -> float:
    # Enforce strictly positive floating-point CLI values at parse time.
    parsed = float(value)

    # Reject non-positive floats before handing them to command handlers.
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return parsed


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help: str
    handler_name: str
    configure: Callable[[argparse.ArgumentParser], None]


def _configure_create_track_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "track_file",
        help="Path to a JSON file containing dataset_id and track policy fields.",
    )


def _configure_launch_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("track_id")
    parser.add_argument(
        "count",
        type=positive_int,
        help="Trial count. Use as a one-shot launch target by default, or a maintained running target with --daemon.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Continuously keep this many trials running until interrupted.",
    )
    parser.add_argument(
        "--poll-interval-sec",
        type=positive_float,
        default=5.0,
        help="Seconds to wait between daemon launch passes. Default: 5.0",
    )
    parser.add_argument(
        "--max-cycles",
        type=positive_int,
        default=None,
        help="Optional number of daemon launch passes before exiting.",
    )


def _configure_list_trials_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("track_id")
    parser.add_argument(
        "--status",
        action="append",
        choices=["queued", "dispatching", "active", "finished", "error"],
        help="Filter by one or more statuses.",
    )


def _configure_modal_sync_dataset_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset_id")


COMMAND_SPECS = (
    CommandSpec(
        name="create-track",
        help="Create a track and seed the baseline trial.",
        handler_name="create_track",
        configure=_configure_create_track_parser,
    ),
    CommandSpec(
        name="launch",
        help="Generate and launch trials for a track.",
        handler_name="launch",
        configure=_configure_launch_parser,
    ),
    CommandSpec(
        name="list-trials",
        help="List trials for a track.",
        handler_name="list_trials",
        configure=_configure_list_trials_parser,
    ),
    CommandSpec(
        name="modal-deploy",
        help="Deploy the Modal runner app.",
        handler_name="modal_deploy",
        configure=lambda parser: None,
    ),
    CommandSpec(
        name="modal-sync-dataset",
        help="Upload a prepared dataset to the Modal dataset volume.",
        handler_name="modal_sync_dataset",
        configure=_configure_modal_sync_dataset_parser,
    ),
)


def build_cli_parser(
    *, handlers: dict[str, Callable[..., int]]
) -> argparse.ArgumentParser:
    # Document the env-only runtime config contract at the top-level help surface.
    parser = argparse.ArgumentParser(
        prog="sigmaevolve",
        description=(
            "Runtime config is resolved from environment variables instead of CLI flags. "
            "Supported names: SIGMAEVOLVE_DATABASE_URL or DATABASE_URL, "
            "SIGMAEVOLVE_DATASET_ROOT, SIGMAEVOLVE_OPENROUTER_API_KEY or OPENROUTER_API_KEY, "
            "SIGMAEVOLVE_MODAL_APP_NAME, SIGMAEVOLVE_MODAL_FUNCTION_NAME, "
            "SIGMAEVOLVE_MODAL_DATASET_VOLUME, SIGMAEVOLVE_MODAL_DATASET_MOUNT, "
            "and SIGMAEVOLVE_MODAL_ENVIRONMENT_NAME."
        ),
    )
    parser.add_argument(
        "--launcher",
        choices=["inline", "modal"],
        default="modal",
        help="Use modal to spawn remote runner jobs by default or inline to execute locally.",
    )

    # Register subcommands from one command-spec registry.
    subparsers = parser.add_subparsers(dest="command", required=True)
    for spec in COMMAND_SPECS:
        subparser = subparsers.add_parser(spec.name, help=spec.help)
        spec.configure(subparser)
        subparser.set_defaults(func=handlers[spec.handler_name])
    return parser


@dataclass
class LaunchSummary:
    mode: str
    cycles_completed: int
    generated_count: int = 0
    launched_count: int = 0
    duplicate_count: int = 0
    stale_count: int = 0
    requeued_count: int = 0
    error_count: int = 0
    stopped_reason: str | None = None


def result_payload(result) -> dict[str, Any]:
    # Normalize reconcile results into the payload shape used by the CLI.
    return {
        "generated_trial_ids": result.generated_trial_ids,
        "launched_trial_ids": result.launched_trial_ids,
        "duplicate_hashes": result.duplicate_hashes,
        "requeued_trial_ids": result.requeued_trial_ids,
        "stale_trial_ids": result.stale_trial_ids,
        "errors": result.errors,
    }


class CliReconcileReporter:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.requested = 0
        self.max_failures = 0
        self._handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "controller_started": self._handle_controller_started,
            "controller_stopped": lambda payload: self._handle_summary(
                "Controller stopped:",
                payload,
                (
                    ("generated_count", "generated"),
                    ("launched_count", "launched"),
                    ("duplicate_count", "duplicates"),
                    ("failed_generation_count", "generation_failures"),
                    ("error_count", "errors"),
                ),
            ),
            "reconcile_started": self._handle_reconcile_started,
            "sweep_completed": lambda payload: self._handle_summary(
                "Sweep complete:",
                payload,
                (
                    ("requeued_count", "requeued"),
                    ("stale_count", "stale"),
                ),
            ),
            "queue_fill_started": self._handle_queue_fill_started,
            "queue_fill_skipped": self._handle_queue_fill_skipped,
            "generation_scheduled": self._handle_generation_scheduled,
            "generation_accepted": lambda payload: self._handle_generation_progress(
                payload,
                f"Accepted candidate for slot {payload['slot_index'] + 1}: {payload['trial_id']}.",
            ),
            "generation_duplicate": lambda payload: self._handle_generation_progress(
                payload,
                "Duplicate candidate for slot "
                f"{payload['slot_index'] + 1} "
                f"(existing={payload['existing_trial_id']}, attempt={payload['duplicate_retry_count']}).",
            ),
            "generation_failed": self._handle_generation_failed,
            "queue_fill_completed": lambda payload: self._handle_summary(
                "Queue fill complete:",
                payload,
                (
                    ("completed", "accepted"),
                    ("requested", "requested"),
                    ("failures", "failures"),
                    ("max_failures", "max_failures"),
                ),
                pair_fields={"completed": "requested", "failures": "max_failures"},
            ),
            "queue_fill_stopped": lambda payload: self._handle_summary(
                "Queue fill stopped:",
                payload,
                (
                    ("completed", "accepted"),
                    ("requested", "requested"),
                    ("failures", "failures"),
                    ("max_failures", "max_failures"),
                ),
                pair_fields={"completed": "requested", "failures": "max_failures"},
            ),
            "launch_batch_started": lambda payload: self._handle_summary(
                "Launching reserved trials:",
                payload,
                (
                    ("reserved_count", "count"),
                    ("max_parallelism", "max_parallelism"),
                ),
            ),
            "trial_launch_started": lambda payload: self._log(
                f"Launching trial {payload['trial_id']}..."
            ),
            "trial_launched": self._handle_trial_launched,
            "trial_launch_failed": lambda payload: self._log(
                f"Launch failed for {payload['trial_id']}: {payload['detail']}"
            ),
            "reconcile_finished": lambda payload: self._handle_summary(
                "Launch pass finished:",
                payload,
                (
                    ("generated_count", "generated"),
                    ("launched_count", "launched"),
                    ("duplicate_count", "duplicates"),
                    ("failed_generation_count", "generation_failures"),
                    ("error_count", "errors"),
                ),
            ),
        }

    def _elapsed(self) -> str:
        seconds = time.monotonic() - self.started_at
        return f"{seconds:5.1f}s"

    def _log(self, message: str) -> None:
        logger.info("[%s] %s", self._elapsed(), message)

    def _progress_line(
        self,
        completed: int,
        requested: int,
        failures: int,
        max_failures: int,
        in_flight: int,
    ) -> str:
        # Short-circuit empty fill cycles before drawing a progress bar.
        if requested <= 0:
            return "Queue fill: nothing to generate."

        # Render the accepted-slot progress bar and failure counters together.
        width = 20
        filled = int(width * completed / requested)
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"Queue fill [{bar}] {completed}/{requested} accepted"
            f" | failures {failures}/{max_failures}"
            f" | in flight {in_flight}"
        )

    def _handle_summary(
        self,
        prefix: str,
        payload: dict[str, Any],
        fields: tuple[tuple[str, str], ...],
        *,
        pair_fields: dict[str, str] | None = None,
    ) -> None:
        # Render each summary field in one reviewable message.
        rendered_fields: list[str] = []
        pair_fields = dict(pair_fields or {})
        for key, label in fields:
            partner_key = pair_fields.get(key)

            # Combine paired fields into a single progress-style value.
            if partner_key is not None:
                rendered_fields.append(f"{label}={payload[key]}/{payload[partner_key]}")
                continue

            # Skip the partner entry after the combined field has been rendered.
            if key in pair_fields.values():
                continue
            rendered_fields.append(f"{label}={payload[key]}")
        self._log(f"{prefix} {' '.join(rendered_fields)}.")

    def _log_progress(self, payload: dict[str, Any]) -> None:
        # Reuse the standard progress-line formatter for all fill-cycle updates.
        self._log(
            self._progress_line(
                int(payload["completed"]),
                int(payload["requested"]),
                int(payload["failures"]),
                int(payload["max_failures"]),
                int(payload["in_flight"]),
            )
        )

    def _handle_controller_started(self, payload: dict[str, Any]) -> None:
        self._log(
            f"Starting controller for {payload['track_id']} with launcher={payload['launcher']} "
            f"and max_parallelism={payload['max_parallelism']}."
        )

    def _handle_reconcile_started(self, payload: dict[str, Any]) -> None:
        self._log(
            f"Running launch pass for {payload['track_id']} with launcher={payload['launcher']}."
        )

    def _handle_queue_fill_started(self, payload: dict[str, Any]) -> None:
        # Cache the fill-cycle budget before logging the initial progress line.
        self.requested = int(payload["requested_generations"])
        self.max_failures = int(payload["max_failures"])
        self._log(
            f"Queue below target: queued={payload['queued_count']} "
            f"target={payload['target_queue_count']}."
        )
        self._log(self._progress_line(0, self.requested, 0, self.max_failures, 0))

    def _handle_queue_fill_skipped(self, payload: dict[str, Any]) -> None:
        self._log(
            f"Queue already full: queued={payload['queued_count']} "
            f"target={payload['target_queue_count']}."
        )

    def _handle_generation_scheduled(self, payload: dict[str, Any]) -> None:
        self._log(
            "Scheduled generation slot "
            f"{payload['slot_index'] + 1}/{max(self.requested, 1)} "
            f"(attempt {payload['duplicate_retry_count']}, generation_index={payload['generation_index']})."
        )

    def _handle_generation_progress(
        self, payload: dict[str, Any], message: str
    ) -> None:
        self._log(message)
        self._log_progress(payload)

    def _handle_generation_failed(self, payload: dict[str, Any]) -> None:
        # Inline any provider detail so generation failures stay actionable.
        has_detail = bool(payload.get("detail"))
        detail = f": {payload['detail']}" if has_detail else ""
        self._handle_generation_progress(
            payload,
            "Generation failed for slot "
            f"{payload['slot_index'] + 1} "
            f"(reason={payload['reason']}, attempt={payload['duplicate_retry_count']}){detail}",
        )

    def _handle_trial_launched(self, payload: dict[str, Any]) -> None:
        # Surface the Modal run URL when the launcher returned one.
        launch_metadata = payload.get("launch_metadata") or {}
        run_url = launch_metadata.get("run_url")
        has_run_url = isinstance(run_url, str) and bool(run_url)
        suffix = f" ({run_url})" if has_run_url else ""
        self._log(f"Launched trial {payload['trial_id']}{suffix}.")

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        # Dispatch only the events that this CLI reporter knows how to print.
        handler = self._handlers.get(event)

        # Ignore events that are not part of the console-facing reporter contract.
        if handler is not None:
            handler(payload)


def _configure_stream_logger(stream_logger: logging.Logger, stream: TextIO) -> None:
    for handler in list(stream_logger.handlers):
        stream_logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    stream_logger.addHandler(handler)
    stream_logger.setLevel(logging.INFO)
    stream_logger.propagate = False


def _apply_runtime_config(args: argparse.Namespace) -> argparse.Namespace:
    # Resolve env-owned runtime settings once so every command sees the same config.
    runtime_config = resolve_runtime_config()
    args.database_url = runtime_config.database_url
    args.dataset_root = runtime_config.dataset_root
    args.openrouter_api_key = runtime_config.openrouter_api_key
    args.modal_app_name = runtime_config.modal_app_name
    args.modal_function_name = runtime_config.modal_function_name
    args.modal_dataset_volume = runtime_config.modal_dataset_volume
    args.modal_dataset_mount = runtime_config.modal_dataset_mount
    args.modal_environment_name = runtime_config.modal_environment_name
    return args


def _resolve_launcher(system, args) -> Any:
    # Keep launcher selection explicit so each runtime path is easy to audit.
    uses_inline_launcher = args.launcher == "inline"

    # Build the in-process launcher when the caller requested local execution.
    if uses_inline_launcher:
        runner = RunnerService(system.repository, system.dataset_manager)
        return InlineRunnerLauncher(runner)

    uses_modal_launcher = args.launcher == "modal"

    # Build the Modal launcher when the caller requested remote execution.
    if uses_modal_launcher:
        requires_remote_database = (
            args.command == "launch" and args.database_url.startswith("sqlite")
        )

        # Reject Modal launches backed by sqlite because the remote runner needs network access.
        if requires_remote_database:
            raise RuntimeError(
                "Modal launcher requires a network-accessible database URL; sqlite is not supported."
            )
        return create_modal_launcher(
            app_name=args.modal_app_name,
            function_name=args.modal_function_name,
            database_url=args.database_url,
            dataset_root=args.modal_dataset_mount,
            environment_name=args.modal_environment_name,
            wandb_env=collect_wandb_env(),
        )

    return system.launcher


def _make_system(args) -> Any:
    # Reject missing database URLs before constructing any system state.
    if not args.database_url:
        raise RuntimeError(
            "A Postgres database URL is required. Set SIGMAEVOLVE_DATABASE_URL or DATABASE_URL."
        )

    # Construct the core system first, then replace its launcher if requested.
    system = build_system(
        database_url=args.database_url,
        dataset_root=args.dataset_root,
        openrouter_api_key=args.openrouter_api_key,
    )
    launcher = _resolve_launcher(system, args)
    system.launcher = launcher
    system.orchestrator.launcher = launcher
    return system


def _print_json(payload: Any) -> None:
    stdout_logger.info("%s", json.dumps(payload, indent=2, sort_keys=True, default=str))


def _ensure_dataset_prepared(system, dataset_id: str) -> tuple[Any, bool]:
    dataset = system.dataset_manager.to_record(dataset_id)
    manifest_path = Path(system.dataset_manager.manifest_path_for(dataset_id))
    manifest_missing = not manifest_path.exists()

    # Prepare the dataset when the on-disk manifest is missing.
    if manifest_missing:
        return system.prepare_dataset(dataset_id), True
    return dataset, False


def _launch_pass_settings(
    system, track_id: str, *, target_count: int, daemon: bool
) -> tuple[int, int]:
    queue_count = system.repository.count_trials(track_id, statuses={"queued"})
    active_count = system.repository.count_trials(track_id, statuses=ACTIVE_STATUSES)

    # Preserve the historical one-shot launch target unless daemon mode needs a running threshold.
    if not daemon:
        return max(queue_count, target_count), active_count + target_count

    needed_slots = max(0, target_count - active_count)
    return max(queue_count, needed_slots), target_count


def _run_launch_pass(
    system,
    track_id: str,
    reporter: CliReconcileReporter,
    *,
    target_count: int,
    daemon: bool,
):
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
        "time_to_best_eval_sec": metrics.get("time_to_best_eval_sec"),
        "timed_out": metrics.get("timed_out", False),
        "time_since_last_eval_sec": metrics.get("time_since_last_eval_sec"),
        "had_unscored_work_at_timeout": metrics.get(
            "had_unscored_work_at_timeout", False
        ),
        "last_phase": metrics.get("last_phase"),
    }


def _suggest_launch_command(args, track_id: str, *, count: int = 1) -> str:
    command = ["sigmaevolve"]

    # Include the launcher flag only when the caller picked a non-default runtime.
    if args.launcher != "modal":
        command.extend(["--launcher", args.launcher])
    command.extend(["launch", track_id, str(count)])
    return shlex.join(command)


def cmd_create_track(args) -> int:
    system = _make_system(args)
    logger.info("Loading track definition.")
    dataset_id, policy = load_track_definition(args.track_file)
    logger.info("Ensuring dataset %s is prepared.", dataset_id)
    dataset, prepared_now = _ensure_dataset_prepared(system, dataset_id)

    # Report whether this command had to prepare the dataset itself.
    if prepared_now:
        logger.info("Prepared dataset %s.", dataset_id)
    else:
        logger.info("Reusing prepared dataset %s.", dataset_id)

    logger.info(
        "Creating track for dataset %s and seeding the baseline trial.", dataset_id
    )
    track = system.create_track(dataset_id, policy)
    logger.info("Created track %s.", track.track_id)
    logger.info("Run it with:\n%s", _suggest_launch_command(args, track.track_id))
    return 0


def cmd_launch(args) -> int:
    system = _make_system(args)
    reporter = CliReconcileReporter()

    # Run a single launch pass when the caller did not request daemon mode.
    if not args.daemon:
        result = _run_launch_pass(
            system, args.track_id, reporter, target_count=args.count, daemon=False
        )
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

            # Stop after the requested number of daemon cycles when a limit is configured.
            if (
                args.max_cycles is not None
                and summary.cycles_completed >= args.max_cycles
            ):
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
    has_status_filter = bool(args.status)
    statuses = set(args.status) if has_status_filter else None
    trials = system.repository.list_trials(args.track_id, statuses=statuses)
    _print_json(
        [
            {
                "trial_id": trial.trial_id,
                "status": trial.status,
                "outcome_reason": trial.outcome_reason,
                "score": trial.score,
                "error_type": classify_error_type(
                    trial.outcome_reason or "", trial.error_json
                ),
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
            "create_track": cmd_create_track,
            "launch": cmd_launch,
            "list_trials": cmd_list_trials,
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
    args = _apply_runtime_config(args)
    try:
        return int(args.func(args))
    except Exception as exc:
        logger.error("error: %s", exc)
        return 1


# Preserve the module entry point for `python -m sigmaevolve.cli`.
if __name__ == "__main__":
    raise SystemExit(main())
