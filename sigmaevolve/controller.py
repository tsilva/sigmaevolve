from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from sigmaevolve.models import ACTIVE_STATUSES, ReconcileResult, now_utc


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
        orchestrator,
        track,
        reporter: Callable[[str, dict[str, Any]], None] | None = None,
        *,
        ready_queue_threshold: int,
        max_parallelism: int,
        continuous: bool = False,
        sweep_interval_sec: float = DEFAULT_SWEEP_INTERVAL_SEC,
        wait_interval_sec: float = DEFAULT_WAIT_INTERVAL_SEC,
    ) -> None:
        self.orchestrator = orchestrator
        self.track = track
        self.reporter = reporter
        self.ready_queue_threshold = int(ready_queue_threshold)
        self.max_parallelism = int(max_parallelism)
        self.continuous = bool(continuous)
        self.sweep_interval_sec = float(sweep_interval_sec)
        self.wait_interval_sec = float(wait_interval_sec)

        generation_workers = max(1, self.max_parallelism, self.ready_queue_threshold, 1)
        launch_workers = max(1, self.max_parallelism)
        self._generation_executor = ThreadPoolExecutor(max_workers=generation_workers)
        self._launch_executor = ThreadPoolExecutor(max_workers=launch_workers)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._started = False

        self._result = ReconcileResult()
        self._dataset_manifest = None
        self._generation_index = self.orchestrator.repository.count_trials(self.track.track_id)
        self._fill_cycle: _FillCycle | None = None
        self._pending_generations: dict[Future[Any], Any] = {}
        self._deferred_retries: list[tuple[int, int, int]] = []
        self._launch_futures: dict[Future[Any], Any] = {}
        self._launch_callbacks_in_progress = 0
        self._dispatch_in_progress = False
        self._one_shot_requested_generations = 0
        self._one_shot_generation_finished = self.continuous

        self._generation_thread = threading.Thread(target=self._generation_loop, name="sigmaevolve-generation", daemon=True)
        self._generation_completion_thread = threading.Thread(
            target=self._generation_completion_loop,
            name="sigmaevolve-generation-completions",
            daemon=True,
        )
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, name="sigmaevolve-dispatch", daemon=True)
        self._sweep_thread = threading.Thread(target=self._sweep_loop, name="sigmaevolve-sweep", daemon=True)

    @property
    def result(self) -> ReconcileResult:
        with self._lock:
            return self._copy_result_locked()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        if self.continuous:
            self.orchestrator._emit(
                self.reporter,
                "controller_started",
                track_id=self.track.track_id,
                launcher=self.orchestrator.launcher.__class__.__name__,
                max_parallelism=self.max_parallelism,
            )
        self._run_sweep(emit_always=True)
        if not self.continuous:
            queue_count = self.orchestrator.repository.count_trials(self.track.track_id, statuses={"queued"})
            self._one_shot_requested_generations = max(0, self.ready_queue_threshold - queue_count)
            self._one_shot_generation_finished = self._one_shot_requested_generations == 0
        self._generation_thread.start()
        self._generation_completion_thread.start()
        self._dispatch_thread.start()
        self._sweep_thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stop_event.set()
            self._condition.notify_all()
        for thread in (self._generation_thread, self._generation_completion_thread, self._dispatch_thread, self._sweep_thread):
            if thread.is_alive():
                thread.join()
        self._generation_executor.shutdown(wait=True, cancel_futures=False)
        self._launch_executor.shutdown(wait=True, cancel_futures=False)
        if self.continuous:
            snapshot = self.result
            self.orchestrator._emit(
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
        if self._launch_callbacks_in_progress:
            return False
        if self._dispatch_in_progress:
            return False
        if self.max_parallelism <= 0:
            return True
        active_count = self.orchestrator.repository.count_trials(self.track.track_id, statuses=ACTIVE_STATUSES)
        queue_count = self.orchestrator.repository.count_trials(self.track.track_id, statuses={"queued"})
        return active_count >= self.max_parallelism or queue_count == 0

    def _desired_queue_threshold(self) -> int:
        if not self.continuous:
            return self.ready_queue_threshold
        active_count = self.orchestrator.repository.count_trials(self.track.track_id, statuses=ACTIVE_STATUSES)
        return max(0, self.max_parallelism - active_count)

    def _ensure_dataset_manifest(self):
        if self._dataset_manifest is None:
            self._dataset_manifest = self.orchestrator.dataset_manager.verify(self.track.dataset_id)
        return self._dataset_manifest

    def _generation_loop(self) -> None:
        while not self._stop_event.is_set():
            attempts_to_schedule: list[tuple[int, int, int]] = []
            started_payload: dict[str, Any] | None = None
            with self._condition:
                if self._fill_cycle is None:
                    queue_count = self.orchestrator.repository.count_trials(self.track.track_id, statuses={"queued"})
                    if self.continuous:
                        target_queue_count = self._desired_queue_threshold()
                        deficit = max(0, target_queue_count - queue_count)
                    else:
                        target_queue_count = self.ready_queue_threshold
                        deficit = self._one_shot_requested_generations if not self._one_shot_generation_finished else 0
                    if deficit > 0:
                        self._fill_cycle = _FillCycle(
                            requested_generations=deficit,
                            max_failures=deficit * self.orchestrator.GENERATION_FAILURE_LIMIT_MULTIPLIER,
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
                elif self._deferred_retries:
                    next_retry_count = min(retry[2] for retry in self._deferred_retries)
                    if not any(
                        pending_attempt.duplicate_retry_count < next_retry_count
                        for pending_attempt in self._pending_generations.values()
                    ):
                        attempts_to_schedule = [retry for retry in self._deferred_retries if retry[2] == next_retry_count]
                        self._deferred_retries = [retry for retry in self._deferred_retries if retry[2] != next_retry_count]
                if not attempts_to_schedule:
                    self._condition.wait(timeout=self.wait_interval_sec)
                    continue

            if started_payload is not None:
                self.orchestrator._emit(self.reporter, "queue_fill_started", **started_payload)

            dataset_manifest = self._ensure_dataset_manifest()
            sampling_settings = self.track.policy_json.get("sampling_settings", {})
            scheduled_any = False
            for slot_index, generation_index, duplicate_retry_count in attempts_to_schedule:
                scheduled = self.orchestrator._schedule_generation_attempt(
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
                self.orchestrator._emit(
                    self.reporter,
                    "generation_scheduled",
                    slot_index=attempt.slot_index,
                    generation_index=attempt.generation_index,
                    duplicate_retry_count=attempt.duplicate_retry_count,
                    in_flight=self._pending_generation_count(),
                )

            if not scheduled_any:
                stopped_payload = self._finish_fill_cycle_if_ready(force=True)
                if stopped_payload is not None:
                    self.orchestrator._emit(self.reporter, stopped_payload["event"], **stopped_payload["payload"])

    def _generation_completion_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                if not self._pending_generations:
                    self._condition.wait(timeout=self.wait_interval_sec)
                    continue
                pending_futures = tuple(self._pending_generations)
            done, _ = wait(pending_futures, timeout=self.wait_interval_sec, return_when=FIRST_COMPLETED)
            for future in done:
                self._on_generation_complete(future)

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            should_wait = False
            reserved = []
            if self.max_parallelism <= 0:
                should_wait = True
            else:
                active_count = self.orchestrator.repository.count_trials(self.track.track_id, statuses=ACTIVE_STATUSES)
                queue_count = self.orchestrator.repository.count_trials(self.track.track_id, statuses={"queued"})
                if active_count >= self.max_parallelism or queue_count <= 0:
                    should_wait = True
                else:
                    with self._condition:
                        self._dispatch_in_progress = True
                        self._condition.notify_all()
                    reserved = self.orchestrator.repository.reserve_trials(
                        track_id=self.track.track_id,
                        max_parallelism=self.max_parallelism,
                        dispatch_ttl_sec=int(self.track.policy_json["dispatch_ttl_sec"]),
                        limit=self.max_parallelism,
                    )
            if should_wait or not reserved:
                with self._condition:
                    self._dispatch_in_progress = False
                    self._condition.wait(timeout=self.wait_interval_sec)
                    self._condition.notify_all()
                continue

            self.orchestrator._emit(
                self.reporter,
                "launch_batch_started",
                reserved_count=len(reserved),
                max_parallelism=self.max_parallelism,
            )
            for trial in reserved:
                self.orchestrator._emit(self.reporter, "trial_launch_started", trial_id=trial.trial_id)
                future = self._launch_executor.submit(
                    self.orchestrator.launcher.launch_trial,
                    trial.trial_id,
                    trial.dispatch_token or "",
                    self.track.policy_json,
                )
                with self._condition:
                    self._launch_futures[future] = trial
                    self._condition.notify_all()
                future.add_done_callback(self._on_launch_complete)
            with self._condition:
                self._dispatch_in_progress = False
                self._condition.notify_all()

    def _sweep_loop(self) -> None:
        while not self._stop_event.wait(self.sweep_interval_sec):
            self._run_sweep(emit_always=False)

    def _run_sweep(self, *, emit_always: bool) -> None:
        requeued, stale_dispatch = self.orchestrator.repository.sweep_expired_dispatches(
            track_id=self.track.track_id,
            max_dispatch_retries=int(self.track.policy_json["max_dispatch_retries"]),
        )
        stale_active = self.orchestrator.repository.sweep_stale_active_trials(
            track_id=self.track.track_id,
            stale_ttl_sec=int(self.track.policy_json["stale_ttl_sec"]),
        )
        if stale_active:
            self._cancel_stale_modal_runs(stale_active)
        stale_trial_ids = stale_dispatch + stale_active
        if requeued or stale_trial_ids:
            with self._condition:
                self._result.requeued_trial_ids.extend(requeued)
                self._result.stale_trial_ids.extend(stale_trial_ids)
                self._condition.notify_all()
        if emit_always or requeued or stale_trial_ids:
            self.orchestrator._emit(
                self.reporter,
                "sweep_completed",
                requeued_count=len(requeued),
                stale_count=len(stale_trial_ids),
            )

    def _cancel_stale_modal_runs(self, stale_trial_ids: list[str]) -> None:
        cancel_run = getattr(self.orchestrator.launcher, "cancel_run", None)
        for trial_id in stale_trial_ids:
            trial = self.orchestrator.repository.get_trial(trial_id)
            if trial is None:
                continue
            launcher_metadata = dict((trial.provenance_json or {}).get("launcher") or {})
            if launcher_metadata.get("kind") != "modal":
                continue

            cancel_metadata = {"cancel_attempted_at": now_utc().isoformat()}
            run_id = launcher_metadata.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                cancel_metadata["cancel_outcome"] = "skipped_no_run_id"
                self.orchestrator.repository.record_trial_launcher_metadata(trial_id, cancel_metadata)
                continue

            try:
                if not callable(cancel_run):
                    raise RuntimeError("Active launcher does not support remote cancellation.")
                cancel_run(launcher_metadata)
            except Exception as exc:
                cancel_metadata["cancel_outcome"] = "failed"
                cancel_metadata["cancel_error"] = str(exc)
            else:
                cancel_metadata["cancel_outcome"] = "requested"
            self.orchestrator.repository.record_trial_launcher_metadata(trial_id, cancel_metadata)

    def _pending_generation_count(self) -> int:
        with self._lock:
            return len(self._pending_generations)

    def _on_generation_complete(self, future: Future[Any]) -> None:
        with self._condition:
            attempt = self._pending_generations.pop(future, None)
            cycle = self._fill_cycle
            self._condition.notify_all()
        if attempt is None or cycle is None:
            return

        retry_needed = False
        stopped_payload: dict[str, Any] | None = None
        next_retry: tuple[int, int, int] | None = None

        try:
            raw_generated = future.result()
        except Exception as exc:
            with self._condition:
                cycle.failures += 1
                self._result = self.orchestrator._record_generation_attempt_failure(
                    track_id=self.track.track_id,
                    result=self._result,
                    provenance_json=self.orchestrator._fallback_generation_provenance(
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
            self.orchestrator._emit(
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
            generated = self.orchestrator._normalize_generation_result(raw_generated)
            if not generated.succeeded:
                error_info = dict(generated.error_info or {})
                with self._condition:
                    cycle.failures += 1
                    self._result = self.orchestrator._record_generation_attempt_failure(
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
                self.orchestrator._emit(
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
                    candidate_source = self.orchestrator._materialize_candidate_source(attempt.context_trials[0].source, generated.source)
                except Exception as exc:
                    with self._condition:
                        cycle.failures += 1
                        self._result = self.orchestrator._record_generation_attempt_failure(
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
                    self.orchestrator._emit(
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
                    generation_outcome = self.orchestrator._accept_generated_candidate(
                        track_id=self.track.track_id,
                        result=self._result,
                        generated=generated,
                        attempt=attempt,
                        candidate_source=candidate_source,
                    )
                    with self._condition:
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
                    self.orchestrator._emit(self.reporter, generation_outcome["event"], **payload)
                    retry_needed = generation_outcome["event"] != "generation_accepted"

        with self._condition:
            cycle = self._fill_cycle
            if cycle is not None and retry_needed and cycle.failures + len(self._pending_generations) < cycle.max_failures and not self._stop_event.is_set():
                next_retry = (attempt.slot_index, attempt.generation_index, attempt.duplicate_retry_count + 1)
            if next_retry is None:
                stopped_payload = self._finish_fill_cycle_if_ready()
            else:
                self._deferred_retries.append(next_retry)
            self._condition.notify_all()

        if stopped_payload is not None:
            self.orchestrator._emit(self.reporter, stopped_payload["event"], **stopped_payload["payload"])

    def _finish_fill_cycle_if_ready(self, *, force: bool = False) -> dict[str, Any] | None:
        cycle = self._fill_cycle
        if cycle is None:
            return None
        if self._pending_generations and not force:
            return None
        if not force and len(cycle.completed_slots) < cycle.requested_generations and cycle.failures < cycle.max_failures:
            return None
        event = "queue_fill_stopped" if len(cycle.completed_slots) < cycle.requested_generations and cycle.failures >= cycle.max_failures else "queue_fill_completed"
        payload = {
            "completed": len(cycle.completed_slots),
            "requested": cycle.requested_generations,
            "failures": cycle.failures,
            "max_failures": cycle.max_failures,
        }
        self._fill_cycle = None
        if not self.continuous:
            self._one_shot_generation_finished = True
        return {"event": event, "payload": payload}

    def _on_launch_complete(self, future: Future[Any]) -> None:
        with self._condition:
            trial = self._launch_futures.pop(future, None)
            if trial is not None:
                self._launch_callbacks_in_progress += 1
            self._condition.notify_all()
        if trial is None:
            return

        try:
            try:
                launch_metadata = future.result()
            except Exception as exc:
                with self._condition:
                    self._result.errors.append(f"launch_failed:{trial.trial_id}:{exc}")
                    self._condition.notify_all()
                self.orchestrator._emit(
                    self.reporter,
                    "trial_launch_failed",
                    trial_id=trial.trial_id,
                    detail=str(exc),
                )
                return

            if launch_metadata:
                self.orchestrator.repository.record_trial_launcher_metadata(trial.trial_id, launch_metadata)
            with self._condition:
                self._result.launched_trial_ids.append(trial.trial_id)
                self._condition.notify_all()
            self.orchestrator._emit(
                self.reporter,
                "trial_launched",
                trial_id=trial.trial_id,
                launch_metadata=launch_metadata or {},
            )
        finally:
            with self._condition:
                self._launch_callbacks_in_progress -= 1
                self._condition.notify_all()
