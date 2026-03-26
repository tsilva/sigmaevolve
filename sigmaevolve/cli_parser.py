from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from sigmaevolve.modal_support import (
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_MODAL_DATASET_MOUNT,
    DEFAULT_MODAL_DATASET_VOLUME,
    DEFAULT_MODAL_FUNCTION_NAME,
)


def json_arg(value: str | None) -> dict[str, Any]:
    # Treat missing JSON flags as an empty object for downstream callers.
    if not value:
        return {}

    # Reject non-object JSON payloads at parse time.
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("JSON value must be an object.")
    return parsed


def load_track_definition(track_file: str) -> tuple[str | None, str, dict[str, Any]]:
    # Load and validate the top-level track definition envelope.
    parsed = json.loads(Path(track_file).read_text())
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Track file must contain a JSON object.")

    # Extract the required dataset identifier and optional track name.
    dataset_id = parsed.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise argparse.ArgumentTypeError("Track file must include a non-empty string dataset_id.")

    raw_name = parsed.get("name")
    if raw_name is not None and not isinstance(raw_name, str):
        raise argparse.ArgumentTypeError("Track file name must be a string when provided.")
    name = raw_name if isinstance(raw_name, str) else None

    # Support either an explicit policy object or legacy top-level policy fields.
    policy = parsed.get("policy")
    if policy is None:
        policy = parsed.get("policy_json")
    if policy is not None:
        if not isinstance(policy, dict):
            raise argparse.ArgumentTypeError("Track file policy must be a JSON object.")
        policy_json = dict(policy)
    else:
        excluded_fields = {"dataset_id", "name"}
        policy_json = {
            key: value
            for key, value in parsed.items()
            if key not in excluded_fields
        }

    return name, dataset_id, policy_json


def positive_int(value: str) -> int:
    # Enforce strictly positive integer CLI values at parse time.
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return parsed


def positive_float(value: str) -> float:
    # Enforce strictly positive floating-point CLI values at parse time.
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return parsed


def build_cli_parser(*, handlers: dict[str, Callable[..., int]]) -> argparse.ArgumentParser:
    # Register the global connection and launcher options first.
    parser = argparse.ArgumentParser(prog="sigmaevolve")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="SQLAlchemy database URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--dataset-root",
        default="./artifacts/datasets",
        help="Root directory for prepared datasets. Default: ./artifacts/datasets",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--launcher",
        choices=["recording", "inline", "modal"],
        default="modal",
        help="Use modal to spawn remote runner jobs by default, inline to execute locally, or recording to reserve only.",
    )
    parser.add_argument(
        "--modal-app-name",
        default=DEFAULT_MODAL_APP_NAME,
        help=f"Deployed Modal app name. Default: {DEFAULT_MODAL_APP_NAME}",
    )
    parser.add_argument(
        "--modal-function-name",
        default=DEFAULT_MODAL_FUNCTION_NAME,
        help=f"Deployed TrialRunner method name. Default: {DEFAULT_MODAL_FUNCTION_NAME}",
    )
    parser.add_argument(
        "--modal-dataset-volume",
        default=DEFAULT_MODAL_DATASET_VOLUME,
        help=f"Modal Volume name for dataset artifacts. Default: {DEFAULT_MODAL_DATASET_VOLUME}",
    )
    parser.add_argument(
        "--modal-dataset-mount",
        default=DEFAULT_MODAL_DATASET_MOUNT,
        help=f"Dataset mount path inside Modal containers. Default: {DEFAULT_MODAL_DATASET_MOUNT}",
    )
    parser.add_argument(
        "--modal-environment-name",
        default=None,
        help="Optional Modal environment name.",
    )

    # Register subcommands after the shared options are in place.
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_dataset = subparsers.add_parser("prepare-dataset", help=argparse.SUPPRESS)
    prepare_dataset.add_argument("dataset_id")
    prepare_dataset.set_defaults(func=handlers["prepare_dataset"])

    create_track = subparsers.add_parser("create-track", help="Create a track and seed the baseline trial.")
    create_track.add_argument(
        "track_file",
        help="Path to a JSON file containing dataset_id, optional name, and track policy fields.",
    )
    create_track.set_defaults(func=handlers["create_track"])

    launch = subparsers.add_parser("launch", help="Generate and launch trials for a track.")
    launch.add_argument("track_id")
    launch.add_argument(
        "count",
        type=positive_int,
        help="Trial count. Use as a one-shot launch target by default, or a maintained running target with --daemon.",
    )
    launch.add_argument(
        "--daemon",
        action="store_true",
        help="Continuously keep this many trials running until interrupted.",
    )
    launch.add_argument(
        "--poll-interval-sec",
        type=positive_float,
        default=5.0,
        help="Seconds to wait between daemon launch passes. Default: 5.0",
    )
    launch.add_argument(
        "--max-cycles",
        type=positive_int,
        default=None,
        help="Optional number of daemon launch passes before exiting.",
    )
    launch.set_defaults(func=handlers["launch"])

    list_trials = subparsers.add_parser("list-trials", help="List trials for a track.")
    list_trials.add_argument("track_id")
    list_trials.add_argument(
        "--status",
        action="append",
        choices=["queued", "dispatching", "active", "finished", "error"],
        help="Filter by one or more statuses.",
    )
    list_trials.set_defaults(func=handlers["list_trials"])

    sample_context = subparsers.add_parser(
        "sample-context",
        help="Show successful finished trials used for generation context.",
    )
    sample_context.add_argument("track_id")
    sample_context.add_argument("--limit", type=int, default=5)
    sample_context.set_defaults(func=handlers["sample_context"])

    rescore = subparsers.add_parser("rescore", help="Rescore finished trials without rerunning training.")
    target = rescore.add_mutually_exclusive_group(required=True)
    target.add_argument("--track-id")
    target.add_argument("--all-tracks", action="store_true")
    rescore.add_argument(
        "--scorer-json",
        required=True,
        help='JSON object such as \'{"primary_metric":"accuracy"}\'.',
    )
    rescore.set_defaults(func=handlers["rescore"])

    modal_deploy = subparsers.add_parser("modal-deploy", help="Deploy the Modal runner app.")
    modal_deploy.set_defaults(func=handlers["modal_deploy"])

    modal_sync_dataset = subparsers.add_parser(
        "modal-sync-dataset",
        help="Upload a prepared dataset to the Modal dataset volume.",
    )
    modal_sync_dataset.add_argument("dataset_id")
    modal_sync_dataset.set_defaults(func=handlers["modal_sync_dataset"])

    return parser
