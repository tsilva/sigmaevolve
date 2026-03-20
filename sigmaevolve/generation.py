from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Protocol
from urllib import request

from sigmaevolve.models import DatasetManifest, GenerationResult, TrackRecord, TrialSummary


class GenerationBackend(Protocol):
    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
    ) -> GenerationResult:
        ...


@dataclass(frozen=True)
class FixedGenerationBackend:
    source: str
    model_name: str = "fixed/test"

    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
    ) -> GenerationResult:
        return GenerationResult(
            source=self.source,
            provenance_json={
                "backend": "fixed",
                "model": self.model_name,
                "generation_index": generation_index,
                "request_messages": [],
                "context_trial_ids": [trial.trial_id for trial in context_trials],
            },
        )


class OpenRouterGenerationBackend:
    def __init__(
        self,
        api_key: str | None = None,
        site_url: str = "https://sigmaevolve.local",
        app_name: str = "sigmaevolve",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url
        self.app_name = app_name

    def _normalize_generation_config(self, generation_policy: dict[str, object]) -> dict[str, object]:
        model_pool = generation_policy.get("model_pool")
        if isinstance(model_pool, list) and model_pool:
            selection = generation_policy.get("selection", "round_robin")
            if selection == "random":
                seed = int(generation_policy.get("seed", 0))
                rng = random.Random(seed)
                return dict(rng.choice(model_pool))
            index = int(generation_policy.get("_generation_index", 0))
            return dict(model_pool[index % len(model_pool)])
        return {
            "model": generation_policy["model"],
            "temperature": generation_policy.get("temperature", 0.2),
            "max_tokens": generation_policy.get("max_tokens", 2500),
            "retry_count": generation_policy.get("retry_count", 2),
        }

    def _format_scalar(self, value: object) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _format_mapping(self, payload: dict[str, object], indent: int = 0) -> list[str]:
        lines: list[str] = []
        prefix = " " * indent
        for key, value in payload.items():
            label = str(key)
            if isinstance(value, dict):
                lines.append(f"{prefix}- {label}:")
                lines.extend(self._format_mapping(value, indent + 2))
                continue
            if isinstance(value, list):
                if not value:
                    lines.append(f"{prefix}- {label}: none")
                    continue
                if all(not isinstance(item, (dict, list)) for item in value):
                    rendered = ", ".join(self._format_scalar(item) for item in value)
                    lines.append(f"{prefix}- {label}: {rendered}")
                    continue
                lines.append(f"{prefix}- {label}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.extend(self._format_mapping(item, indent + 2))
                    elif isinstance(item, list):
                        nested = ", ".join(self._format_scalar(part) for part in item)
                        lines.append(f"{' ' * (indent + 2)}- {nested}")
                    else:
                        lines.append(f"{' ' * (indent + 2)}- {self._format_scalar(item)}")
                continue
            lines.append(f"{prefix}- {label}: {self._format_scalar(value)}")
        return lines

    def _summarize_error(self, error_json: dict[str, object] | None) -> list[str]:
        if not error_json:
            return []
        lines: list[str] = []
        reason = error_json.get("reason")
        if reason is not None:
            lines.append(f"- error reason: {self._format_scalar(reason)}")
        detail = error_json.get("detail")
        if detail is not None:
            lines.append(f"- error detail: {self._format_scalar(detail)}")
        returncode = error_json.get("returncode")
        if returncode is not None:
            lines.append(f"- returncode: {self._format_scalar(returncode)}")
        stderr = error_json.get("stderr")
        if isinstance(stderr, str) and stderr.strip():
            excerpt = stderr.strip().splitlines()[-1][:240]
            lines.append(f"- stderr excerpt: {excerpt}")
        return lines

    def _build_user_prompt_text(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        selected_config: dict[str, object],
    ) -> str:
        task_contract = {
            "entrypoint": "train.py --config /abs/path/run_config.json",
            "config keys": {
                "dataset_dir": "Directory containing the prepared dataset assets.",
                "train_split_path": "Path to the training .npz file with features and labels arrays.",
                "validation_split_path": "Path to the validation .npz file with features only.",
                "budget_sec": "Hard wall-clock training budget in seconds.",
                "max_eval_gap_sec": "Maximum allowed gap between completed validation passes.",
                "random_seed": "Deterministic seed to use for numpy/torch setup.",
                "predictions_output_path": "Compatibility fallback output path for predictions.npz.",
                "progress_path": "Path for heartbeat/progress JSON updates.",
                "eval_dir": "Directory for atomically written evaluation artifacts.",
                "debug_output_path": "Path for optional debug JSON output.",
                "dataset_metadata": "Dataset metadata object from the harness.",
            },
            "train split format": "npz with arrays: features, labels",
            "validation split format": "npz with arrays: features",
            "required outputs": {
                "progress_path": "JSON heartbeat with current phase, elapsed_time_sec, and last_completed_eval_sec",
                "eval_dir": "Directory of atomically written .npz eval artifacts with predictions, eval_index, elapsed_time_sec, and optional epoch",
                "legacy_predictions_output_path": "Optional compatibility fallback only; prefer eval_dir artifacts",
            },
            "allowed packages": ["argparse", "json", "pathlib", "numpy", "torch"],
            "budget_sec": track.policy_json["budget_sec"],
            "max_eval_gap_sec": track.policy_json.get("max_eval_gap_sec", 15),
                "writing rules": [
                    "Read the config JSON using the exact keys listed in config_keys; do not invent alternate key names.",
                    "Write eval artifacts atomically by saving to a temp path and renaming into eval_dir.",
                    "Features may be multi-dimensional tensors rather than pre-flattened vectors; if you use linear layers, flatten both train and validation batches consistently or start the model with nn.Flatten().",
                    "Do not spend long uninterrupted stretches training without finishing a validation pass.",
                    "When validation accuracy ties, lower elapsed wall time to that eval wins.",
                ],
        }
        primary_parent = context_trials[0] if context_trials else None
        lines = [
            f"Write a complete Python train.py for dataset {track.dataset_id}.",
            "",
            "Treat this as an evolutionary mutation task, not a rewrite from scratch.",
            "",
            "Use the dataset metadata below when choosing the model and loss setup:",
        ]
        if dataset_manifest.metadata:
            lines.extend(self._format_mapping(dict(dataset_manifest.metadata), indent=0))
        else:
            lines.append("- No dataset metadata was provided.")
        lines.extend(
            [
                "",
                "Follow this task contract exactly:",
            ]
        )
        lines.extend(self._format_mapping(task_contract, indent=0))
        lines.extend(
            [
                "",
                "This attempt was selected with the following generation settings:",
            ]
        )
        lines.extend(self._format_mapping(dict(selected_config), indent=0))
        lines.extend(
            [
                "",
                "The broader generation policy for the track is:",
            ]
        )
        lines.extend(self._format_mapping(dict(track.policy_json["generation_backend"]), indent=0))
        lines.extend(
            [
                "",
                "Use this parent trial as the base candidate:",
            ]
        )
        if primary_parent is not None:
            lines.extend(
                [
                    f"Trial {primary_parent.trial_id}:",
                    f"- score: {self._format_scalar(primary_parent.score)}",
                    f"- outcome reason: {self._format_scalar(primary_parent.outcome_reason)}",
                ]
            )
            if primary_parent.metrics_json:
                lines.append("- metrics:")
                lines.extend(self._format_mapping(dict(primary_parent.metrics_json), indent=2))
            lines.extend(self._summarize_error(primary_parent.error_json))
            lines.extend(
                [
                    "",
                    "Mutation instructions:",
                    "- Preserve the parent's working harness integration unless a change is required.",
                    "- Produce a mutated descendant of the parent, not a fresh rewrite.",
                    "- Make exactly one substantive improvement likely to improve validation accuracy within the time budget.",
                    "- Avoid cosmetic refactors or rename-only changes.",
                    "",
                    "Parent source:",
                    "```python",
                    primary_parent.source.rstrip(),
                    "```",
                ]
            )
        else:
            lines.append("No prior trials are available.")
        lines.extend(
            [
                "",
                "Avoid the failure modes seen in these recent negative trials:",
            ]
        )
        if negative_trials:
            for trial in negative_trials:
                lines.extend(
                    [
                        f"Trial {trial.trial_id}:",
                        f"- score: {self._format_scalar(trial.score)}",
                        f"- outcome reason: {self._format_scalar(trial.outcome_reason)}",
                    ]
                )
                if trial.metrics_json:
                    lines.append("- metrics:")
                    lines.extend(self._format_mapping(dict(trial.metrics_json), indent=2))
                lines.extend(self._summarize_error(trial.error_json))
                lines.extend(
                    [
                        "- source preview:",
                        "```python",
                        trial.source.rstrip(),
                        "```",
                    ]
                )
        else:
            lines.append("No recent negative trials are available.")
        return "\n".join(lines)

    def _build_prompt(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        selected_config: dict[str, object],
    ) -> list[dict[str, str]]:
        system_prompt = (
            "You are generating a self-contained Python train.py script for a classification harness. "
            "Return only Python source, with no markdown fences or commentary. "
            "Optimize for best validation accuracy, but you must publish completed validation checkpoints regularly so the harness can score them before timeout."
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": self._build_user_prompt_text(
                    track,
                    dataset_manifest,
                    context_trials,
                    negative_trials,
                    selected_config,
                ),
            },
        ]

    def _extract_source(self, raw_text: str) -> str:
        match = re.search(r"```(?:python)?\n(.*?)```", raw_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip() + "\n"
        return raw_text.strip() + "\n"

    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
    ) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter generation.")
        generation_policy = dict(track.policy_json["generation_backend"])
        generation_policy["_generation_index"] = generation_index
        selected_config = self._normalize_generation_config(generation_policy)
        payload = {
            "model": selected_config["model"],
            "messages": self._build_prompt(
                track,
                dataset_manifest,
                context_trials,
                negative_trials or [],
                selected_config,
            ),
            "temperature": selected_config.get("temperature", 0.2),
            "max_tokens": selected_config.get("max_tokens", 2500),
        }
        req = request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        request_messages = payload["messages"]
        return GenerationResult(
            source=self._extract_source(content),
            provenance_json={
                "backend": "openrouter",
                "model": selected_config["model"],
                "generation_config": selected_config,
                "generation_index": generation_index,
                "provider_response_id": body.get("id"),
                "request_messages": request_messages,
                "context_trial_ids": [trial.trial_id for trial in context_trials],
            },
        )
