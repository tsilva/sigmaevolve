from __future__ import annotations

import json
import os
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
    ) -> GenerationResult:
        return GenerationResult(
            source=self.source,
            provenance_json={
                "backend": "fixed",
                "model": self.model_name,
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

    def _build_prompt(
        self,
        track: TrackRecord,
        dataset_manifest: DatasetManifest,
        context_trials: list[TrialSummary],
    ) -> list[dict[str, str]]:
        summaries = [
            {
                "trial_id": trial.trial_id,
                "score": trial.score,
                "metrics": trial.metrics_json,
                "source_preview": trial.source[:1500],
            }
            for trial in context_trials
        ]
        policy = track.policy_json["generation_backend"]
        system_prompt = (
            "You are generating a self-contained Python train.py script for a classification harness. "
            "Return only Python source, with no markdown fences or commentary."
        )
        user_prompt = {
            "dataset_id": track.dataset_id,
            "dataset_metadata": dataset_manifest.metadata,
            "task_contract": {
                "entrypoint": "train.py --config /abs/path/run_config.json",
                "train_split_format": "npz with arrays: features, labels",
                "validation_split_format": "npz with arrays: features",
                "required_output": "npz with array: predictions",
                "allowed_packages": ["argparse", "json", "pathlib", "numpy", "torch"],
                "budget_sec": track.policy_json["budget_sec"],
            },
            "policy": policy,
            "context_trials": summaries,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, sort_keys=True)},
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
    ) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter generation.")
        generation_policy = track.policy_json["generation_backend"]
        payload = {
            "model": generation_policy["model"],
            "messages": self._build_prompt(track, dataset_manifest, context_trials),
            "temperature": generation_policy.get("temperature", 0.2),
            "max_tokens": generation_policy.get("max_tokens", 2500),
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
        return GenerationResult(
            source=self._extract_source(content),
            provenance_json={
                "backend": "openrouter",
                "model": generation_policy["model"],
                "provider_response_id": body.get("id"),
                "context_trial_ids": [trial.trial_id for trial in context_trials],
            },
        )
