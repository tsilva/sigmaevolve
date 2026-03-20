from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Protocol
from urllib import request

from sigmaevolve.models import CANDIDATE_KIND_STRATEGY_V1, DatasetManifest, GenerationResult, TrackRecord, TrialSummary


class GenerationBackend(Protocol):
    def generate(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary] | None = None,
        generation_index: int = 0,
        duplicate_retry_count: int = 0,
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
        duplicate_retry_count: int = 0,
    ) -> GenerationResult:
        return GenerationResult(
            source=self.source,
            provenance_json={
                "backend": "fixed",
                "model": self.model_name,
                "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
                "generation_index": generation_index,
                "duplicate_retry_count": duplicate_retry_count,
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

    def _build_system_prompt_text(self) -> str:
        static_task_contract = {
            "candidate module": "strategy.py",
            "required exports": {
                "initialize(ctx)": "Return a dict state object used across training windows.",
                "train_window(ctx, state)": "Perform one bounded unit of training and mutate state in place.",
                "predict_validation(ctx, state)": "Return validation class ids or logits for every validation example.",
            },
            "strategy context fields": {
                "train_features": "Training features array from the harness.",
                "train_labels": "Training labels array from the harness.",
                "validation_features": "Validation features array from the harness.",
                "dataset_metadata": "Dataset metadata object from the harness.",
                "random_seed": "Deterministic seed chosen by the harness.",
                "device": "Device string chosen by the harness.",
                "budget_sec": "Total wall-clock budget in seconds.",
                "remaining_budget_sec": "Remaining wall-clock budget at the start of the callback.",
                "max_eval_gap_sec": "Maximum allowed duration of one train_window call.",
                "window_index": "Zero-based train window index.",
            },
            "train split format": "npz with arrays: features, labels",
            "validation split format": "npz with arrays: features",
            "allowed packages": ["numpy", "torch"],
            "writing rules": [
                "Return only Python source for strategy.py, with no markdown fences or commentary.",
                "Do not parse CLI args, read config files, write files, or manage progress/eval artifacts; the harness owns all protocol and bookkeeping.",
                "Features may be multi-dimensional tensors rather than pre-flattened vectors; if you use linear layers, flatten both train and validation batches consistently or start the model with nn.Flatten().",
                "train_window must return promptly enough to stay within max_eval_gap_sec.",
                "predict_validation must return one prediction per validation example as class ids or logits.",
                "Augmentation is allowed inside train_window and may evolve without explicit prompting.",
            ],
            "mutation rules": [
                "Preserve the parent's working harness integration unless a change is required.",
                "Produce a mutated descendant of the parent strategy, not a fresh rewrite.",
                "Make exactly one substantive improvement likely to improve validation accuracy within the time budget.",
                "Avoid cosmetic refactors or rename-only changes.",
            ],
        }
        lines = [
            dedent(
                """
                You are generating a self-contained Python strategy.py module for a classification harness.
                Treat this as an evolutionary mutation task, not a rewrite from scratch.
                Optimize for best validation accuracy while letting the harness control evaluation cadence, timing, and scoring.
                Follow this contract exactly:
                """
            ).strip()
        ]
        lines.extend(self._format_mapping(static_task_contract, indent=0))
        return "\n".join(lines)

    def _build_user_prompt_text(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        selected_config: dict[str, object],
    ) -> str:
        primary_parent = context_trials[0] if context_trials else None
        lines = [
            f"Write a complete Python strategy.py module for dataset {track.dataset_id}.",
            "",
            "Use the dataset metadata below when choosing the model, augmentation, and loss setup:",
        ]
        if dataset_manifest.metadata:
            lines.extend(self._format_mapping(dict(dataset_manifest.metadata), indent=0))
        else:
            lines.append("- No dataset metadata was provided.")
        lines.extend(
            [
                "",
                "Use these track-specific runtime settings:",
            ]
        )
        lines.extend(
            self._format_mapping(
                {
                    "budget_sec": track.policy_json["budget_sec"],
                    "max_eval_gap_sec": track.policy_json.get("max_eval_gap_sec", 15),
                },
                indent=0,
            )
        )
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
        return [
            {"role": "system", "content": self._build_system_prompt_text()},
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
        duplicate_retry_count: int = 0,
    ) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter generation.")
        generation_policy = dict(track.policy_json["generation_backend"])
        generation_policy["_generation_index"] = generation_index
        selected_config = self._normalize_generation_config(generation_policy)
        selected_config["temperature"] = float(selected_config.get("temperature", 0.2)) + (0.1 * duplicate_retry_count)
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
                "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
                "generation_config": selected_config,
                "generation_index": generation_index,
                "duplicate_retry_count": duplicate_retry_count,
                "provider_response_id": body.get("id"),
                "request_messages": request_messages,
                "context_trial_ids": [trial.trial_id for trial in context_trials],
            },
        )
