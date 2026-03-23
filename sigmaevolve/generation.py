from __future__ import annotations

import json
import os
import random
import re
from functools import lru_cache
from importlib import resources
from dataclasses import dataclass
from typing import Protocol
from urllib import request

from sigmaevolve.evolve_blocks import EVOLVE_BLOCK_END, EVOLVE_BLOCK_START
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
                "backend": "openrouter",
                "model": self.model_name,
                "candidate_kind": CANDIDATE_KIND_STRATEGY_V1,
                "generation_index": generation_index,
                "duplicate_retry_count": duplicate_retry_count,
                "generation_config": {
                    "model": self.model_name,
                    "temperature": 0.0,
                    "max_tokens": 0,
                },
                "request_messages": [
                    {
                        "role": "system",
                        "content": "Test-only fixed generator stub for a recorded LLM prompt.",
                    },
                    {
                        "role": "user",
                        "content": "Use this parent trial as the base candidate:\n```python\n# fixed backend stub\n```",
                    },
                ],
                "context_trial_ids": [trial.trial_id for trial in context_trials],
            },
        )


@lru_cache(maxsize=1)
def _load_system_prompt_template() -> str:
    return (resources.files("sigmaevolve.prompts") / "system.md").read_text(encoding="utf-8").strip()


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
            index = int(generation_policy.get("_generation_index", 0))
            seed = int(generation_policy.get("seed", 0))
            if selection == "random":
                rng = random.Random(seed + index)
                return dict(rng.choice(model_pool))
            if selection == "weighted_random":
                weights: list[float] = []
                normalized_pool: list[dict[str, object]] = []
                for entry in model_pool:
                    item = dict(entry)
                    raw_weight = item.get("probability", item.get("weight", 1.0))
                    weight = float(raw_weight)
                    if weight < 0:
                        raise ValueError("generation_backend model_pool probabilities must be non-negative.")
                    normalized_pool.append(item)
                    weights.append(weight)
                total_weight = sum(weights)
                if total_weight <= 0:
                    raise ValueError("generation_backend weighted_random selection requires a positive total probability.")
                rng = random.Random(seed + index)
                selected = dict(rng.choices(normalized_pool, weights=weights, k=1)[0])
                selected["selection_probability"] = float(selected.get("probability", 1.0)) / total_weight
                return selected
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

    def _trial_prompt_metric(self, trial: TrialSummary, *names: str) -> str:
        metrics = trial.metrics_json or {}
        for name in names:
            if name in metrics and metrics[name] is not None:
                return self._format_scalar(metrics[name])
        return "n/a"

    def _render_trial_prompt_block(self, trial: TrialSummary) -> list[str]:
        return [
            "---",
            f"score: {self._format_scalar(trial.score)}",
            f"val_acc: {self._trial_prompt_metric(trial, 'val_acc', 'accuracy')}",
            f"val_loss: {self._trial_prompt_metric(trial, 'val_loss', 'loss')}",
            "---",
            "```python",
            trial.source.rstrip(),
            "```",
        ]

    def _build_system_prompt_text(self) -> str:
        return (
            _load_system_prompt_template()
            .replace("{{EVOLVE_BLOCK_START}}", EVOLVE_BLOCK_START)
            .replace("{{EVOLVE_BLOCK_END}}", EVOLVE_BLOCK_END)
        )

    def _build_user_prompt_text(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
        negative_trials: list[TrialSummary],
        selected_config: dict[str, object],
    ) -> str:
        del track, dataset_manifest, negative_trials, selected_config
        current_program = context_trials[0] if context_trials else None
        prior_programs = context_trials[1:] if len(context_trials) > 1 else []
        lines = ["PRIOR PROGRAMS:", ""]
        if prior_programs:
            for index, trial in enumerate(prior_programs):
                if index > 0:
                    lines.append("")
                lines.extend(self._render_trial_prompt_block(trial))
        else:
            lines.append("None.")
        lines.extend(
            [
                "",
                "CURRENT PROGRAM:",
                "",
                "Here is the current program we are trying to improve",
                "(you will need to propose a modification to it below).",
                "",
            ]
        )
        if current_program is not None:
            lines.extend(self._render_trial_prompt_block(current_program))
        else:
            lines.append("None.")
        lines.extend(["", "REPLACEMENTS:"])
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
        generation_policy["_generation_index"] = generation_index + duplicate_retry_count
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
