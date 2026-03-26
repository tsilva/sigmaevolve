from __future__ import annotations

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
