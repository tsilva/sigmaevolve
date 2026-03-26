from __future__ import annotations


# ---- controller.py ----

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from sigmaevolve.core import ACTIVE_STATUSES, ReconcileResult, now_utc


def _emit(reporter: Callable[[str, dict[str, Any]], None] | None, event: str, **payload: Any) -> None:
    # Funnel controller events through the optional reporter callback.
    if reporter is not None:
        reporter(event, payload)


@dataclass
class _FillCycle:
    requested_generations: int
    max_failures: int
    completed_slots: set[int] = field(default_factory=set)
    failures: int = 0


class TrackController:
    DEFAULT_SWEEP_INTERVAL_SEC = 0.25
    DEFAULT_WAIT_INTERVAL_SEC = 0.05

    def __init__(
        self,
        *,
        repository,
        dataset_manager,
        generation,
        launcher,
        generation_failure_limit_multiplier: int,
        track,
        reporter: Callable[[str, dict[str, Any]], None] | None = None,
        ready_queue_threshold: int,
        max_parallelism: int,
        continuous: bool = False,
        sweep_interval_sec: float = DEFAULT_SWEEP_INTERVAL_SEC,
        wait_interval_sec: float = DEFAULT_WAIT_INTERVAL_SEC,
    ) -> None:
        # Capture the immutable runtime dependencies and queue targets.
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generation = generation
        self.launcher = launcher
        self.generation_failure_limit_multiplier = int(generation_failure_limit_multiplier)
        self.track = track
        self.reporter = reporter
        self.ready_queue_threshold = int(ready_queue_threshold)
        self.max_parallelism = int(max_parallelism)
        self.continuous = bool(continuous)
        self.sweep_interval_sec = float(sweep_interval_sec)
        self.wait_interval_sec = float(wait_interval_sec)

        # Size the generation and launch pools independently.
        generation_workers = max(1, self.max_parallelism, self.ready_queue_threshold, 1)
        launch_workers = max(1, self.max_parallelism)
        self._generation_executor = ThreadPoolExecutor(max_workers=generation_workers)
        self._launch_executor = ThreadPoolExecutor(max_workers=launch_workers)

        # Initialize the shared controller state guarded by the condition variable.
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._started = False

        self._result = ReconcileResult()
        self._dataset_manifest = None
        self._generation_index = self.repository.count_trials(self.track.track_id)
        self._fill_cycle: _FillCycle | None = None
        self._pending_generations: dict[Future[Any], Any] = {}
        self._deferred_retries: list[tuple[int, int, int]] = []
        self._launch_futures: dict[Future[Any], Any] = {}
        self._dispatch_in_progress = False
        self._one_shot_requested_generations = 0
        self._one_shot_generation_finished = self.continuous

        # Drive generation, dispatch, and sweep work from one controller loop.
        self._controller_thread = threading.Thread(
            target=self._controller_loop,
            name="sigmaevolve-controller",
            daemon=True,
        )

    @property
    def result(self) -> ReconcileResult:
        with self._lock:
            return self._copy_result_locked()

    def start(self) -> None:
        # Allow start() to be called idempotently from the orchestrator.
        with self._lock:
            if self._started:
                return
            self._started = True

        # Emit startup state for continuous controllers before background work begins.
        if self.continuous:
            _emit(
                self.reporter,
                "controller_started",
                track_id=self.track.track_id,
                launcher=self.launcher.__class__.__name__,
                max_parallelism=self.max_parallelism,
            )

        # Sweep stale work once before any new generation or dispatch cycle starts.
        self._run_sweep(emit_always=True)
        if not self.continuous:
            queue_count = self.repository.count_trials(self.track.track_id, statuses={"queued"})
            self._one_shot_requested_generations = max(0, self.ready_queue_threshold - queue_count)
            self._one_shot_generation_finished = self._one_shot_requested_generations == 0

        # Start the controller loop after the initial state is fully prepared.
        self._controller_thread.start()

    def stop(self) -> None:
        # Wake the controller loop before joining the background thread.
        with self._condition:
            self._stop_event.set()
            self._condition.notify_all()
        if self._controller_thread.is_alive():
            self._controller_thread.join()

        # Shut down the executors only after the controller loop has drained.
        self._generation_executor.shutdown(wait=True, cancel_futures=False)
        self._launch_executor.shutdown(wait=True, cancel_futures=False)
        if self.continuous:
            snapshot = self.result
            _emit(
                self.reporter,
                "controller_stopped",
                track_id=self.track.track_id,
                generated_count=len(snapshot.generated_trial_ids),
                launched_count=len(snapshot.launched_trial_ids),
                duplicate_count=len(snapshot.duplicate_trial_ids),
                failed_generation_count=len(snapshot.failed_generation_trial_ids),
                error_count=len(snapshot.errors),
            )

    def wait_until_one_shot_complete(self) -> ReconcileResult:
        with self._condition:
            while not self._is_one_shot_complete_locked():
                self._condition.wait(timeout=self.wait_interval_sec)
        return self.result

    def _copy_result_locked(self) -> ReconcileResult:
        return ReconcileResult(
            generated_trial_ids=list(self._result.generated_trial_ids),
            launched_trial_ids=list(self._result.launched_trial_ids),
            duplicate_hashes=list(self._result.duplicate_hashes),
            duplicate_trial_ids=list(self._result.duplicate_trial_ids),
            failed_generation_trial_ids=list(self._result.failed_generation_trial_ids),
            requeued_trial_ids=list(self._result.requeued_trial_ids),
            stale_trial_ids=list(self._result.stale_trial_ids),
            errors=list(self._result.errors),
        )

    def _is_one_shot_complete_locked(self) -> bool:
        # Reject completion while any background work can still change the outcome.
        if self.continuous:
            return False
        if not self._one_shot_generation_finished:
            return False
        if self._fill_cycle is not None:
            return False
        if self._pending_generations:
            return False
        if self._launch_futures:
            return False
        if self._dispatch_in_progress:
            return False
        if self.max_parallelism <= 0:
            return True

        # One-shot mode is complete once there is nothing left to queue or dispatch.
        active_count = self.repository.count_trials(self.track.track_id, statuses=ACTIVE_STATUSES)
        queue_count = self.repository.count_trials(self.track.track_id, statuses={"queued"})
        return active_count >= self.max_parallelism or queue_count == 0

    def _desired_queue_threshold(self) -> int:
        # Continuous mode maintains only enough queued work to fill open slots.
        if not self.continuous:
            return self.ready_queue_threshold
        active_count = self.repository.count_trials(self.track.track_id, statuses=ACTIVE_STATUSES)
        return max(0, self.max_parallelism - active_count)

    def _ensure_dataset_manifest(self):
        # Verify the dataset once and reuse the manifest for every generation attempt.
        if self._dataset_manifest is None:
            self._dataset_manifest = self.dataset_manager.verify(self.track.dataset_id)
        return self._dataset_manifest

    def _controller_loop(self) -> None:
        next_sweep_at = 0.0
        while not self._stop_event.is_set():
            # Drive sweep, completion, scheduling, and dispatch from one loop.
            work_performed = False
            now = time.monotonic()
            if now >= next_sweep_at:
                self._run_sweep(emit_always=False)
                next_sweep_at = now + self.sweep_interval_sec
                work_performed = True

            work_performed = self._drain_completed_generations() or work_performed
            work_performed = self._drain_completed_launches() or work_performed
            work_performed = self._schedule_generation_attempts() or work_performed
            work_performed = self._dispatch_reserved_trials() or work_performed

            # Sleep only when the controller did not make any forward progress.
            with self._condition:
                self._condition.notify_all()
                if work_performed or self._stop_event.is_set():
                    continue
                self._condition.wait(timeout=self.wait_interval_sec)

    def _next_retry_batch_locked(self) -> list[tuple[int, int, int]]:
        if not self._deferred_retries:
            return []

        # Release deferred retries only after earlier attempts have cleared.
        next_retry_count = min(retry[2] for retry in self._deferred_retries)
        has_earlier_attempt = any(
            pending_attempt.duplicate_retry_count < next_retry_count
            for pending_attempt in self._pending_generations.values()
        )
        if has_earlier_attempt:
            return []

        ready = [
            retry
            for retry in self._deferred_retries
            if retry[2] == next_retry_count
        ]
        self._deferred_retries = [
            retry
            for retry in self._deferred_retries
            if retry[2] != next_retry_count
        ]
        return ready

    def _schedule_generation_attempts(self) -> bool:
        attempts_to_schedule: list[tuple[int, int, int]] = []
        started_payload: dict[str, Any] | None = None

        with self._condition:
            # Start a fresh fill cycle whenever the ready queue falls below target.
            if self._fill_cycle is None:
                queue_count = self.repository.count_trials(self.track.track_id, statuses={"queued"})
                if self.continuous:
                    target_queue_count = self._desired_queue_threshold()
                    deficit = max(0, target_queue_count - queue_count)
                else:
                    target_queue_count = self.ready_queue_threshold
                    deficit = (
                        self._one_shot_requested_generations
                        if not self._one_shot_generation_finished
                        else 0
                    )
                if deficit > 0:
                    self._fill_cycle = _FillCycle(
                        requested_generations=deficit,
                        max_failures=deficit * self.generation_failure_limit_multiplier,
                    )
                    started_payload = {
                        "queued_count": queue_count,
                        "target_queue_count": target_queue_count,
                        "requested_generations": deficit,
                        "max_failures": self._fill_cycle.max_failures,
                    }
                    for slot_index in range(deficit):
                        attempts_to_schedule.append((slot_index, self._generation_index, 0))
                        self._generation_index += 1
            else:
                attempts_to_schedule = self._next_retry_batch_locked()

        if not attempts_to_schedule:
            return False
        if started_payload is not None:
            _emit(self.reporter, "queue_fill_started", **started_payload)

        # Schedule each generation attempt against the verified dataset manifest.
        dataset_manifest = self._ensure_dataset_manifest()
        sampling_settings = self.track.policy_json.get("sampling_settings", {})
        scheduled_any = False
        for slot_index, generation_index, duplicate_retry_count in attempts_to_schedule:
            scheduled = self.generation.schedule_generation_attempt(
                self._generation_executor,
                self.track,
                dataset_manifest,
                sampling_settings,
                slot_index=slot_index,
                generation_index=generation_index,
                duplicate_retry_count=duplicate_retry_count,
            )
            if scheduled is None:
                continue

            scheduled_any = True
            future, attempt = scheduled
            with self._condition:
                self._pending_generations[future] = attempt
                self._condition.notify_all()
            _emit(
                self.reporter,
                "generation_scheduled",
                slot_index=attempt.slot_index,
                generation_index=attempt.generation_index,
                duplicate_retry_count=attempt.duplicate_retry_count,
                in_flight=self._pending_generation_count(),
            )

        # Close the cycle immediately when no context produced a runnable attempt.
        if scheduled_any:
            return True

        with self._condition:
            stopped_payload = self._finish_fill_cycle_if_ready(force=True)
            self._condition.notify_all()
        if stopped_payload is not None:
            _emit(self.reporter, stopped_payload["event"], **stopped_payload["payload"])
        return True

    def _drain_completed_generations(self) -> bool:
        with self._condition:
            completed = [future for future in self._pending_generations if future.done()]
        for future in completed:
            self._on_generation_complete(future)
        return bool(completed)

    def _dispatch_reserved_trials(self) -> bool:
        # Stop dispatching entirely when the controller has no launch capacity.
        if self.max_parallelism <= 0:
            return False

        active_count = self.repository.count_trials(self.track.track_id, statuses=ACTIVE_STATUSES)
        queue_count = self.repository.count_trials(self.track.track_id, statuses={"queued"})
        has_dispatch_capacity = active_count < self.max_parallelism
        has_queued_trials = queue_count > 0
        if not has_dispatch_capacity or not has_queued_trials:
            return False

        with self._condition:
            self._dispatch_in_progress = True
            self._condition.notify_all()
        try:
            reserved = self.repository.reserve_trials(
                track_id=self.track.track_id,
                max_parallelism=self.max_parallelism,
                dispatch_ttl_sec=int(self.track.policy_json["dispatch_ttl_sec"]),
                limit=self.max_parallelism,
            )
            if not reserved:
                return False

            # Launch every reserved trial on the launch executor.
            _emit(
                self.reporter,
                "launch_batch_started",
                reserved_count=len(reserved),
                max_parallelism=self.max_parallelism,
            )
            for trial in reserved:
                _emit(self.reporter, "trial_launch_started", trial_id=trial.trial_id)
                future = self._launch_executor.submit(
                    self.launcher.launch_trial,
                    trial.trial_id,
                    trial.dispatch_token or "",
                    self.track.policy_json,
                )
                with self._condition:
                    self._launch_futures[future] = trial
                    self._condition.notify_all()
            return True
        finally:
            with self._condition:
                self._dispatch_in_progress = False
                self._condition.notify_all()

    def _drain_completed_launches(self) -> bool:
        with self._condition:
            completed = [future for future in self._launch_futures if future.done()]
        for future in completed:
            self._on_launch_complete(future)
        return bool(completed)

    def _run_sweep(self, *, emit_always: bool) -> None:
        # Requeue expired dispatches before checking for stale active trials.
        requeued, stale_dispatch = self.repository.sweep_expired_dispatches(
            track_id=self.track.track_id,
            max_dispatch_retries=int(self.track.policy_json["max_dispatch_retries"]),
        )
        stale_active = self.repository.sweep_stale_active_trials(
            track_id=self.track.track_id,
            stale_ttl_sec=int(self.track.policy_json["stale_ttl_sec"]),
        )
        if stale_active:
            self._cancel_stale_modal_runs(stale_active)

        # Persist the sweep results in memory before emitting a summary event.
        stale_trial_ids = stale_dispatch + stale_active
        if requeued or stale_trial_ids:
            with self._condition:
                self._result.requeued_trial_ids.extend(requeued)
                self._result.stale_trial_ids.extend(stale_trial_ids)
                self._condition.notify_all()
        if emit_always or requeued or stale_trial_ids:
            _emit(
                self.reporter,
                "sweep_completed",
                requeued_count=len(requeued),
                stale_count=len(stale_trial_ids),
            )

    def _cancel_stale_modal_runs(self, stale_trial_ids: list[str]) -> None:
        # Attempt remote cancellation only for stale trials launched through Modal.
        cancel_run = getattr(self.launcher, "cancel_run", None)
        for trial_id in stale_trial_ids:
            trial = self.repository.get_trial(trial_id)
            if trial is None:
                continue
            launcher_metadata = dict((trial.provenance_json or {}).get("launcher") or {})
            if launcher_metadata.get("kind") != "modal":
                continue

            # Record cancellation metadata even when the launcher cannot cancel remotely.
            cancel_metadata = {"cancel_attempted_at": now_utc().isoformat()}
            run_id = launcher_metadata.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                cancel_metadata["cancel_outcome"] = "skipped_no_run_id"
                self.repository.record_trial_launcher_metadata(trial_id, cancel_metadata)
                continue

            # Request cancellation and persist the outcome for later inspection.
            try:
                if not callable(cancel_run):
                    raise RuntimeError("Active launcher does not support remote cancellation.")
                cancel_run(launcher_metadata)
            except Exception as exc:
                cancel_metadata["cancel_outcome"] = "failed"
                cancel_metadata["cancel_error"] = str(exc)
            else:
                cancel_metadata["cancel_outcome"] = "requested"
            self.repository.record_trial_launcher_metadata(trial_id, cancel_metadata)

    def _pending_generation_count(self) -> int:
        with self._lock:
            return len(self._pending_generations)

    def _on_generation_complete(self, future: Future[Any]) -> None:
        # Resolve the completed future under the controller lock first.
        with self._condition:
            attempt = self._pending_generations.pop(future, None)
            cycle = self._fill_cycle
            self._condition.notify_all()
        if attempt is None or cycle is None:
            return

        # Track whether this slot should be retried or should stop the fill cycle.
        retry_needed = False
        stopped_payload: dict[str, Any] | None = None
        next_retry: tuple[int, int, int] | None = None

        try:
            raw_generated = future.result()
        except Exception as exc:
            # Persist generator exceptions as failed generation-attempt trials.
            with self._condition:
                cycle.failures += 1
                self._result = self.generation.record_generation_attempt_failure(
                    track_id=self.track.track_id,
                    result=self._result,
                    provenance_json=self.generation.fallback_generation_provenance(
                        self.track,
                        attempt.context_trials,
                        generation_index=attempt.generation_index,
                        duplicate_retry_count=attempt.duplicate_retry_count,
                    ),
                    reason="generator_exception",
                    detail=str(exc),
                    extra_error_json={"exception_type": type(exc).__name__},
                    result_error=str(exc),
                )
                retry_needed = True
                failures = cycle.failures
                max_failures = cycle.max_failures
                completed = len(cycle.completed_slots)
                requested = cycle.requested_generations
                in_flight = len(self._pending_generations)
            _emit(
                self.reporter,
                "generation_failed",
                slot_index=attempt.slot_index,
                generation_index=attempt.generation_index,
                duplicate_retry_count=attempt.duplicate_retry_count,
                reason="generator_exception",
                detail=str(exc),
                failures=failures,
                max_failures=max_failures,
                completed=completed,
                requested=requested,
                in_flight=in_flight,
            )
        else:
            # Normalize provider output before deciding how to record the attempt.
            generated = self.generation.normalize_generation_result(raw_generated)
            if not generated.succeeded:
                error_info = dict(generated.error_info or {})
                # Record provider-level failures directly from the returned error payload.
                with self._condition:
                    cycle.failures += 1
                    self._result = self.generation.record_generation_attempt_failure(
                        track_id=self.track.track_id,
                        result=self._result,
                        provenance_json=generated.provenance_json,
                        reason=str(error_info.get("reason") or "generation_failed"),
                        detail=str(error_info["detail"]) if error_info.get("detail") is not None else None,
                        extra_error_json=error_info,
                    )
                    retry_needed = True
                    failures = cycle.failures
                    max_failures = cycle.max_failures
                    completed = len(cycle.completed_slots)
                    requested = cycle.requested_generations
                    in_flight = len(self._pending_generations)
                _emit(
                    self.reporter,
                    "generation_failed",
                    slot_index=attempt.slot_index,
                    generation_index=attempt.generation_index,
                    duplicate_retry_count=attempt.duplicate_retry_count,
                    reason=str(error_info.get("reason") or "generation_failed"),
                    detail=str(error_info["detail"]) if error_info.get("detail") is not None else "",
                    failures=failures,
                    max_failures=max_failures,
                    completed=completed,
                    requested=requested,
                    in_flight=in_flight,
                )
            else:
                assert generated.source is not None
                try:
                    # Materialize the final candidate source against the parent program.
                    candidate_source = self.generation.materialize_candidate_source(
                        attempt.context_trials[0].source,
                        generated.source,
                    )
                except Exception as exc:
                    # Convert invalid mutations into failed generation attempts.
                    with self._condition:
                        cycle.failures += 1
                        self._result = self.generation.record_generation_attempt_failure(
                            track_id=self.track.track_id,
                            result=self._result,
                            provenance_json=generated.provenance_json,
                            reason="candidate_materialization_failed",
                            detail=str(exc),
                            result_error=f"invalid_mutation:{exc}",
                        )
                        retry_needed = True
                        failures = cycle.failures
                        max_failures = cycle.max_failures
                        completed = len(cycle.completed_slots)
                        requested = cycle.requested_generations
                        in_flight = len(self._pending_generations)
                    _emit(
                        self.reporter,
                        "generation_failed",
                        slot_index=attempt.slot_index,
                        generation_index=attempt.generation_index,
                        duplicate_retry_count=attempt.duplicate_retry_count,
                        reason="candidate_materialization_failed",
                        detail=str(exc),
                        failures=failures,
                        max_failures=max_failures,
                        completed=completed,
                        requested=requested,
                        in_flight=in_flight,
                    )
                else:
                    # Let the generation backend decide whether the candidate was accepted.
                    generation_outcome = self.generation.accept_generated_candidate(
                        track_id=self.track.track_id,
                        result=self._result,
                        generated=generated,
                        attempt=attempt,
                        candidate_source=candidate_source,
                    )
                    with self._condition:
                        # Count accepted slots separately from duplicate or failed attempts.
                        if generation_outcome["event"] == "generation_accepted":
                            cycle.completed_slots.add(attempt.slot_index)
                            self._condition.notify_all()
                        else:
                            cycle.failures += 1
                        failures = cycle.failures
                        max_failures = cycle.max_failures
                        completed = len(cycle.completed_slots)
                        requested = cycle.requested_generations
                        in_flight = len(self._pending_generations)
                    payload = dict(generation_outcome["payload"])
                    payload.update(
                        {
                            "slot_index": attempt.slot_index,
                            "generation_index": attempt.generation_index,
                            "duplicate_retry_count": attempt.duplicate_retry_count,
                            "failures": failures,
                            "max_failures": max_failures,
                            "completed": completed,
                            "requested": requested,
                            "in_flight": in_flight,
                        }
                    )
                    _emit(self.reporter, generation_outcome["event"], **payload)
                    retry_needed = generation_outcome["event"] != "generation_accepted"

        # Either queue the retry or finish the fill cycle once this attempt is settled.
        with self._condition:
            cycle = self._fill_cycle
            has_cycle = cycle is not None
            below_failure_budget = (
                has_cycle
                and cycle.failures + len(self._pending_generations) < cycle.max_failures
            )
            should_retry = (
                has_cycle
                and retry_needed
                and below_failure_budget
                and not self._stop_event.is_set()
            )
            if should_retry:
                next_retry = (attempt.slot_index, attempt.generation_index, attempt.duplicate_retry_count + 1)
            if next_retry is None:
                stopped_payload = self._finish_fill_cycle_if_ready()
            else:
                self._deferred_retries.append(next_retry)
            self._condition.notify_all()

        # Emit the final fill-cycle event outside the controller lock.
        if stopped_payload is not None:
            _emit(self.reporter, stopped_payload["event"], **stopped_payload["payload"])

    def _finish_fill_cycle_if_ready(self, *, force: bool = False) -> dict[str, Any] | None:
        # Ignore finish requests when there is no active fill cycle.
        cycle = self._fill_cycle
        if cycle is None:
            return None
        if self._pending_generations and not force:
            return None

        # Wait until all slots are accepted or the failure budget is exhausted.
        completed_slots = len(cycle.completed_slots)
        missing_completions = completed_slots < cycle.requested_generations
        has_failure_budget = cycle.failures < cycle.max_failures
        if not force and missing_completions and has_failure_budget:
            return None

        stopped_on_failures = missing_completions and cycle.failures >= cycle.max_failures
        event = "queue_fill_stopped" if stopped_on_failures else "queue_fill_completed"
        payload = {
            "completed": completed_slots,
            "requested": cycle.requested_generations,
            "failures": cycle.failures,
            "max_failures": cycle.max_failures,
        }

        # Clear the active cycle before reporting the terminal fill-cycle state.
        self._fill_cycle = None
        if not self.continuous:
            self._one_shot_generation_finished = True

        return {"event": event, "payload": payload}

    def _on_launch_complete(self, future: Future[Any]) -> None:
        # Remove the completed launch future before handling its result.
        with self._condition:
            trial = self._launch_futures.pop(future, None)
            self._condition.notify_all()
        if trial is None:
            return

        try:
            launch_metadata = future.result()
        except Exception as exc:
            # Record launcher failures without interrupting the controller loop.
            with self._condition:
                self._result.errors.append(f"launch_failed:{trial.trial_id}:{exc}")
                self._condition.notify_all()
            _emit(
                self.reporter,
                "trial_launch_failed",
                trial_id=trial.trial_id,
                detail=str(exc),
            )
            return

        # Persist launcher metadata and mark the trial as launched.
        if launch_metadata:
            self.repository.record_trial_launcher_metadata(trial.trial_id, launch_metadata)
        with self._condition:
            self._result.launched_trial_ids.append(trial.trial_id)
            self._condition.notify_all()
        _emit(
            self.reporter,
            "trial_launched",
            trial_id=trial.trial_id,
            launch_metadata=launch_metadata or {},
        )


# ---- launchers.py ----

from typing import Any, Protocol


class RunnerLauncher(Protocol):
    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ...

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        ...


class RecordingLauncher:
    def __init__(self) -> None:
        self.launched: list[tuple[str, str]] = []

    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del launch_policy
        self.launched.append((trial_id, dispatch_token))
        return None

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        del launcher_metadata


class InlineRunnerLauncher:
    def __init__(self, runner_service, runner_id_prefix: str = "inline") -> None:
        self.runner_service = runner_service
        self.runner_id_prefix = runner_id_prefix
        self.launch_count = 0

    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del launch_policy
        self.launch_count += 1
        runner_id = f"{self.runner_id_prefix}_{self.launch_count}"
        self.runner_service.run_reserved_trial(trial_id, dispatch_token, runner_id)
        return None

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        del launcher_metadata


class ModalRemoteLauncher:
    def __init__(self, modal_function) -> None:
        self.modal_function = modal_function

    def launch_trial(
        self,
        trial_id: str,
        dispatch_token: str,
        launch_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        requested_gpus = (launch_policy or {}).get("modal_gpu_preferences")
        if requested_gpus is None:
            attempts: list[str | None] = [None]
        elif isinstance(requested_gpus, list) and requested_gpus:
            attempts = [str(gpu) for gpu in requested_gpus]
        else:
            raise ValueError("Track launch policy modal_gpu_preferences must be null or a non-empty list.")

        failures: list[str] = []
        attempted_gpus: list[str] = []
        for gpu in attempts:
            if gpu is not None:
                attempted_gpus.append(gpu)
            try:
                spawn_result = self.modal_function.spawn(
                    trial_id=trial_id,
                    dispatch_token=dispatch_token,
                    gpu=gpu,
                )
            except Exception as exc:
                failures.append(f"{gpu or 'cpu'}: {exc}")
                continue
            function_call = getattr(spawn_result, "function_call", spawn_result)
            effective_gpu = getattr(spawn_result, "effective_gpu", gpu)

            metadata: dict[str, Any] = {
                "kind": "modal",
                "gpu_attempts": list(attempted_gpus),
            }
            if effective_gpu is not None:
                metadata["gpu_selected"] = effective_gpu
            object_id = getattr(function_call, "object_id", None)
            if isinstance(object_id, str) and object_id:
                metadata["run_id"] = object_id
            get_dashboard_url = getattr(function_call, "get_dashboard_url", None)
            if callable(get_dashboard_url):
                try:
                    run_url = get_dashboard_url()
                except Exception:
                    run_url = None
                if isinstance(run_url, str) and run_url:
                    metadata["run_url"] = run_url
            return metadata

        raise RuntimeError("Modal launch failed for all configured resources: " + "; ".join(failures))

    def cancel_run(self, launcher_metadata: dict[str, Any]) -> None:
        run_id = launcher_metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Modal cancellation requires launcher_metadata.run_id.")
        self.modal_function.cancel(run_id)


# ---- orchestrator.py ----

from typing import Any, Callable

from sigmaevolve.generation import GenerationCoordinator

def emit_report_event(
    reporter: Callable[[str, dict[str, Any]], None] | None,
    event: str,
    **payload: Any,
) -> None:
    if reporter is not None:
        reporter(event, payload)


class Orchestrator:
    GENERATION_FAILURE_LIMIT_MULTIPLIER = 2

    def __init__(self, repository, dataset_manager, generator, launcher) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generator = generator
        self.launcher = launcher
        self.generation = GenerationCoordinator(repository=repository, generator=generator)

    def _sample_successful_context_trials(
        self,
        track_id: str,
        sampling_settings: dict[str, Any],
        generation_index: int,
    ):
        return self.generation.sample_successful_context_trials(
            track_id,
            sampling_settings,
            generation_index,
        )

    def _sample_generation_context_trials(
        self,
        track_id: str,
        sampling_settings: dict[str, Any],
        generation_index: int,
    ):
        return self.generation.sample_generation_context_trials(
            track_id,
            sampling_settings,
            generation_index,
        )

    def start_track_controller(
        self,
        track_id: str,
        reporter: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        max_parallelism: int,
        ready_queue_threshold: int = 0,
    ) -> TrackController:
        track = self.repository.get_track(track_id)
        if track is None:
            raise KeyError(f"Track not found: {track_id}")
        if max_parallelism < 0:
            raise ValueError("max_parallelism must be >= 0")
        controller = TrackController(
            repository=self.repository,
            dataset_manager=self.dataset_manager,
            generation=self.generation,
            launcher=self.launcher,
            generation_failure_limit_multiplier=self.GENERATION_FAILURE_LIMIT_MULTIPLIER,
            track=track,
            reporter=reporter,
            ready_queue_threshold=int(ready_queue_threshold),
            max_parallelism=int(max_parallelism),
            continuous=True,
        )
        controller.start()
        return controller

    def reconcile_track(
        self,
        track_id: str,
        reporter: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        ready_queue_threshold: int = 1,
        max_parallelism: int = 1,
    ):
        track = self.repository.get_track(track_id)
        if track is None:
            raise KeyError(f"Track not found: {track_id}")
        ready_queue_threshold = int(ready_queue_threshold)
        max_parallelism = int(max_parallelism)
        if ready_queue_threshold < 0:
            raise ValueError("ready_queue_threshold must be >= 0")
        if max_parallelism < 0:
            raise ValueError("max_parallelism must be >= 0")
        emit_report_event(
            reporter,
            "reconcile_started",
            track_id=track_id,
            launcher=self.launcher.__class__.__name__,
        )
        initial_queue_count = self.repository.count_trials(track_id, statuses={"queued"})
        if initial_queue_count >= ready_queue_threshold:
            emit_report_event(
                reporter,
                "queue_fill_skipped",
                queued_count=initial_queue_count,
                target_queue_count=ready_queue_threshold,
            )
        controller = TrackController(
            repository=self.repository,
            dataset_manager=self.dataset_manager,
            generation=self.generation,
            launcher=self.launcher,
            generation_failure_limit_multiplier=self.GENERATION_FAILURE_LIMIT_MULTIPLIER,
            track=track,
            reporter=reporter,
            ready_queue_threshold=ready_queue_threshold,
            max_parallelism=max_parallelism,
            continuous=False,
        )
        controller.start()
        try:
            result = controller.wait_until_one_shot_complete()
        finally:
            controller.stop()
        emit_report_event(
            reporter,
            "reconcile_finished",
            generated_count=len(result.generated_trial_ids),
            launched_count=len(result.launched_trial_ids),
            duplicate_count=len(result.duplicate_trial_ids),
            failed_generation_count=len(result.failed_generation_trial_ids),
            error_count=len(result.errors),
        )
        return result


# ---- system.py ----

from pathlib import Path

from sigmaevolve.generation import build_baseline_train_script
from sigmaevolve.datasets import DatasetManager, TorchvisionClassificationProvider
from sigmaevolve.generation import OpenRouterGenerationBackend
from sigmaevolve.core import (
    CANDIDATE_KIND_STRATEGY_V1,
    DatasetRecord,
    MigrationResult,
    TrackPolicy,
    TrackRecord,
    TrialRecord,
    TrialSummary,
)
from sigmaevolve.execution import RunnerService
from sigmaevolve.core import compute_score
from sigmaevolve.storage import SQLAlchemyRepository


class EvolutionSystem:
    def __init__(
        self,
        repository: SQLAlchemyRepository,
        dataset_manager: DatasetManager,
        generator,
        launcher,
        runner_service: RunnerService,
    ) -> None:
        self.repository = repository
        self.dataset_manager = dataset_manager
        self.generator = generator
        self.launcher = launcher
        self.runner_service = runner_service
        self.orchestrator = Orchestrator(repository, dataset_manager, generator, launcher)

    def prepare_dataset(self, dataset_id: str) -> DatasetRecord:
        # Prepare the dataset locally before registering its manifest path.
        manifest = self.dataset_manager.prepare(dataset_id)
        manifest_path = Path(manifest.root_dir) / "manifest.json"

        return self.repository.register_dataset(
            dataset_id=dataset_id,
            manifest_path=str(manifest_path),
        )

    def create_track(self, name: str | None, dataset_id: str, policy_json: dict) -> TrackRecord:
        # Refuse to create tracks against datasets that were never prepared.
        if self.repository.get_dataset(dataset_id) is None:
            raise KeyError(f"Dataset must be prepared before track creation: {dataset_id}")

        # Persist the normalized track policy before seeding the baseline trial.
        policy = TrackPolicy.from_dict(policy_json)
        track = self.repository.create_track(
            name=name,
            dataset_id=dataset_id,
            policy_json=policy.to_dict(),
        )
        baseline_source = build_baseline_train_script()

        # Seed the track with the fixed baseline candidate exactly once.
        self.repository.create_queued_trial_if_absent(
            track_id=track.track_id,
            source=baseline_source,
            provenance_json={
                "backend": "baseline",
                "model": "compact-fixed-trainer",
                "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
                "parent_trial_ids": [],
            },
        )
        return track

    def reconcile_track(
        self,
        track_id: str,
        reporter=None,
        *,
        ready_queue_threshold: int = 1,
        max_parallelism: int = 1,
    ):
        return self.orchestrator.reconcile_track(
            track_id,
            reporter=reporter,
            ready_queue_threshold=ready_queue_threshold,
            max_parallelism=max_parallelism,
        )

    def start_track_controller(
        self,
        track_id: str,
        reporter=None,
        *,
        max_parallelism: int,
        ready_queue_threshold: int = 0,
    ):
        return self.orchestrator.start_track_controller(
            track_id,
            reporter=reporter,
            max_parallelism=max_parallelism,
            ready_queue_threshold=ready_queue_threshold,
        )

    def sample_trial_context(self, track_id: str, limit: int) -> list[TrialSummary]:
        return self.repository.sample_trial_context(track_id=track_id, limit=limit)

    def claim_trial(self, trial_id: str, dispatch_token: str, runner_id: str) -> TrialRecord | None:
        return self.repository.claim_trial(trial_id=trial_id, dispatch_token=dispatch_token, runner_id=runner_id)

    def heartbeat_trial(self, trial_id: str, runner_id: str, meta: dict) -> None:
        self.repository.heartbeat_trial(trial_id=trial_id, runner_id=runner_id, meta=meta)

    def update_active_trial_metrics(self, trial_id: str, runner_id: str, metrics: dict) -> None:
        self.repository.update_active_trial_metrics(trial_id=trial_id, runner_id=runner_id, metrics=metrics)

    def finalize_trial(
        self,
        trial_id: str,
        runner_id: str,
        outcome_reason: str,
        metrics: dict | None,
        score: float,
        error_info: dict | None,
    ) -> None:
        self.repository.finalize_trial(
            trial_id=trial_id,
            runner_id=runner_id,
            outcome_reason=outcome_reason,
            metrics=metrics,
            score=score,
            error_info=error_info,
        )

    def rescore(self, track_or_all, scorer_config: dict) -> MigrationResult:
        track_id = None if track_or_all == "all" else track_or_all
        return self.repository.rescore(track_id=track_id, scorer_config=scorer_config)


def build_system(
    database_url: str,
    dataset_root: str | Path,
    openrouter_api_key: str | None = None,
    providers: dict | None = None,
    launcher=None,
) -> EvolutionSystem:
    # Resolve default providers before wiring the runtime services together.
    dataset_root = Path(dataset_root)
    default_providers = {
        "mnist:v1": TorchvisionClassificationProvider("mnist"),
        "fashion_mnist:v1": TorchvisionClassificationProvider("fashion_mnist"),
    }
    effective_providers = providers or default_providers

    # Construct the core repository, dataset, generation, and runner services.
    repository = SQLAlchemyRepository(database_url)
    dataset_manager = DatasetManager(
        dataset_root=dataset_root,
        providers=effective_providers,
    )
    generator = OpenRouterGenerationBackend(api_key=openrouter_api_key)
    runner_service = RunnerService(repository=repository, dataset_manager=dataset_manager)
    launcher = launcher or RecordingLauncher()

    # Return the fully wired orchestration facade.
    return EvolutionSystem(
        repository=repository,
        dataset_manager=dataset_manager,
        generator=generator,
        launcher=launcher,
        runner_service=runner_service,
    )
