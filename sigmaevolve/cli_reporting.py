from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable


logger = logging.getLogger("sigmaevolve.cli.stderr")


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
            "controller_stopped": self._handle_controller_stopped,
            "reconcile_started": self._handle_reconcile_started,
            "sweep_completed": self._handle_sweep_completed,
            "queue_fill_started": self._handle_queue_fill_started,
            "queue_fill_skipped": self._handle_queue_fill_skipped,
            "generation_scheduled": self._handle_generation_scheduled,
            "generation_accepted": self._handle_generation_accepted,
            "generation_duplicate": self._handle_generation_duplicate,
            "generation_failed": self._handle_generation_failed,
            "queue_fill_completed": self._handle_queue_fill_completed,
            "queue_fill_stopped": self._handle_queue_fill_stopped,
            "launch_batch_started": self._handle_launch_batch_started,
            "trial_launch_started": self._handle_trial_launch_started,
            "trial_launched": self._handle_trial_launched,
            "trial_launch_failed": self._handle_trial_launch_failed,
            "reconcile_finished": self._handle_reconcile_finished,
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

    def _handle_controller_stopped(self, payload: dict[str, Any]) -> None:
        self._log(
            "Controller stopped: "
            f"generated={payload['generated_count']} "
            f"launched={payload['launched_count']} "
            f"duplicates={payload['duplicate_count']} "
            f"generation_failures={payload['failed_generation_count']} "
            f"errors={payload['error_count']}."
        )

    def _handle_reconcile_started(self, payload: dict[str, Any]) -> None:
        self._log(f"Running launch pass for {payload['track_id']} with launcher={payload['launcher']}.")

    def _handle_sweep_completed(self, payload: dict[str, Any]) -> None:
        self._log(f"Sweep complete: requeued={payload['requeued_count']}, stale={payload['stale_count']}.")

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

    def _handle_generation_accepted(self, payload: dict[str, Any]) -> None:
        self._log(f"Accepted candidate for slot {payload['slot_index'] + 1}: {payload['trial_id']}.")
        self._log_progress(payload)

    def _handle_generation_duplicate(self, payload: dict[str, Any]) -> None:
        self._log(
            "Duplicate candidate for slot "
            f"{payload['slot_index'] + 1} "
            f"(existing={payload['existing_trial_id']}, attempt={payload['duplicate_retry_count']})."
        )
        self._log_progress(payload)

    def _handle_generation_failed(self, payload: dict[str, Any]) -> None:
        # Inline any provider detail so generation failures stay actionable.
        detail = f": {payload['detail']}" if payload.get("detail") else ""
        self._log(
            "Generation failed for slot "
            f"{payload['slot_index'] + 1} "
            f"(reason={payload['reason']}, attempt={payload['duplicate_retry_count']}){detail}"
        )
        self._log_progress(payload)

    def _handle_queue_fill_completed(self, payload: dict[str, Any]) -> None:
        self._log(
            f"Queue fill complete: accepted={payload['completed']}/{payload['requested']} "
            f"with failures={payload['failures']}/{payload['max_failures']}."
        )

    def _handle_queue_fill_stopped(self, payload: dict[str, Any]) -> None:
        self._log(
            f"Queue fill stopped at {payload['completed']}/{payload['requested']} "
            f"after reaching failure budget {payload['failures']}/{payload['max_failures']}."
        )

    def _handle_launch_batch_started(self, payload: dict[str, Any]) -> None:
        self._log(
            f"Launching reserved trials: count={payload['reserved_count']} "
            f"max_parallelism={payload['max_parallelism']}."
        )

    def _handle_trial_launch_started(self, payload: dict[str, Any]) -> None:
        self._log(f"Launching trial {payload['trial_id']}...")

    def _handle_trial_launched(self, payload: dict[str, Any]) -> None:
        # Surface the Modal run URL when the launcher returned one.
        launch_metadata = payload.get("launch_metadata") or {}
        run_url = launch_metadata.get("run_url")
        suffix = f" ({run_url})" if isinstance(run_url, str) and run_url else ""
        self._log(f"Launched trial {payload['trial_id']}{suffix}.")

    def _handle_trial_launch_failed(self, payload: dict[str, Any]) -> None:
        self._log(f"Launch failed for {payload['trial_id']}: {payload['detail']}")

    def _handle_reconcile_finished(self, payload: dict[str, Any]) -> None:
        self._log(
            "Launch pass finished: "
            f"generated={payload['generated_count']} "
            f"launched={payload['launched_count']} "
            f"duplicates={payload['duplicate_count']} "
            f"generation_failures={payload['failed_generation_count']} "
            f"errors={payload['error_count']}."
        )

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        # Dispatch only the events that this CLI reporter knows how to print.
        handler = self._handlers.get(event)
        if handler is not None:
            handler(payload)
