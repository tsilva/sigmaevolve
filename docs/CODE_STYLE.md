# Code Style Guide

This guide is optimized for agent compliance first and human review second. Prefer code that is easy to scan, easy to change, and explicit about intent over dense one-liners or clever compression. Use spacing to reveal logical blocks, not to mechanically maximize comment count.

## Hard Rules

- Structure non-trivial functions in a readable order: validate inputs, run the main logic, then build the return value.
- Keep imports at the top of the file unless there is a clear reason to lazy load them.
- Prefer early returns over nested conditionals when a failed condition means the function should stop.
- Use blank lines to separate logical blocks, and keep a comment directly attached to the block it introduces.
- Add a short intent comment for non-trivial logical blocks and business-rule branches when the purpose is not obvious from the code alone.
- Keep coercion, validation, and closely related state changes in the same block when they implement one idea.
- Introduce named intermediate variables for non-trivial expressions, boolean conditions, and transformed values before combining them.
- Split signatures, calls, and returned dicts across multiple lines once the one-line form becomes visually dense.
- Extract a helper when one block mixes iteration, filtering, transformation, and ordering.
- Add a short intent comment before a contiguous logical block inside a non-trivial function when the block's role would otherwise require inference.

## Soft Preferences

- Prefer positive predicates over negated compound boolean expressions.
- Prefer staged construction for nested payloads when the payload shape carries meaning.
- Prefer a small number of clear block comments over comment-per-line annotation.
- Keep long formatted strings visually segmented by field instead of compressing them into one hard-to-edit line.

## Comments

- Comments are for non-obvious intent, policy, or block purpose in non-trivial functions.
- A comment should sit immediately above the block it describes; do not insert a blank line between the comment and that block.
- Use one blank line to separate logical blocks from each other.
- Comment a branch when the branch carries business meaning that is not already obvious from the condition itself.
- Comments should explain intent or purpose, not restate the line below them.
- Keep comments short and use domain language where possible.
- Trivial one- or two-line helpers are the only exception.

## Do Not Over-Apply

- Do not force multiline formatting for tiny helpers that are already easy to read.
- Do not add a comment above every individual statement or every obvious branch.
- Do not stack repeated one-line comment/code pairs when those lines belong to one logical block.
- Do not split coercion from its paired validation when both lines express the same check.
- Do not use filler comments such as "Set variable" or "Return result".
- Do not satisfy the branch-comment rule with empty boilerplate such as "Check condition" or "Else case".
- Do not extract helpers unless the original block is carrying multiple responsibilities.

## Canonical Examples

Use these as reference patterns, not templates to copy mechanically.

### Structured function layout

```python
def compute_classification_metrics(
    predictions: list[int],
    labels: list[int],
) -> dict[str, int | float]:
    # Reject empty evaluation sets.
    if not labels:
        raise ValueError("Cannot score an empty validation split.")

    # Reject mismatched prediction and label lengths.
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")

    # Derive the core metrics from the aligned predictions and labels.
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    accuracy = correct / len(labels)

    # Return the metrics payload in a reviewable shape.
    return {
        "accuracy": accuracy,
        "correct": correct,
        "num_examples": len(labels),
    }
```

### Named boolean conditions

```python
def should_retry_dispatch(trial: TrialRecord, now: datetime) -> bool:
    # Name each policy requirement before composing the final decision.
    is_dispatching = trial.status == "dispatching"
    has_deadline = trial.dispatch_deadline_at is not None
    deadline_expired = has_deadline and trial.dispatch_deadline_at < now
    has_retry_budget = trial.dispatch_attempts < 3

    # Express the retry policy as one readable predicate.
    return is_dispatching and deadline_expired and has_retry_budget
```

## Additional Examples

### Use comments to introduce blocks, not to narrate every line

Bad:

```python
def reconcile_track(
    self,
    track_id: str,
    reporter: Callable[[str, dict[str, Any]], None] | None = None,
    *,
    ready_queue_threshold: int = 1,
    max_parallelism: int = 1,
):
    # Resolve and validate the track before running the one-shot controller.
    track = self.repository.get_track(track_id)
    # Reject unknown tracks before starting reconcile work.
    if track is None:
        raise KeyError(f"Track not found: {track_id}")
    ready_queue_threshold = int(ready_queue_threshold)
    max_parallelism = int(max_parallelism)
    # Reject negative queue targets before the controller starts.
    if ready_queue_threshold < 0:
        raise ValueError("ready_queue_threshold must be >= 0")
    # Reject negative parallelism before the controller starts.
    if max_parallelism < 0:
        raise ValueError("max_parallelism must be >= 0")
```

Good:

```python
def reconcile_track(
    self,
    track_id: str,
    reporter: Callable[[str, dict[str, Any]], None] | None = None,
    *,
    ready_queue_threshold: int = 1,
    max_parallelism: int = 1,
):
    # Resolve and validate the track before running the one-shot controller.
    track = self.repository.get_track(track_id)

    # Reject unknown tracks before starting reconcile work.
    if track is None:
        raise KeyError(f"Track not found: {track_id}")

    # Reject negative queue targets before the controller starts.
    ready_queue_threshold = int(ready_queue_threshold)
    if ready_queue_threshold < 0:
        raise ValueError("ready_queue_threshold must be >= 0")

    # Reject negative parallelism before the controller starts.
    max_parallelism = int(max_parallelism)
    if max_parallelism < 0:
        raise ValueError("max_parallelism must be >= 0")
```

Why this version is preferred:

- Blank lines separate ideas, not every comment from the line below it.
- Each comment introduces a meaningful block instead of narrating isolated statements.
- Coercion and its related validation stay together, which makes the control flow easier to scan.

### Keep imports at file scope unless lazy loading is intentional

Bad:

```python
def format_metrics(metrics: dict[str, float]) -> str:
    import json

    return json.dumps(metrics, sort_keys=True)
```

Good:

```python
import json


def format_metrics(metrics: dict[str, float]) -> str:
    return json.dumps(metrics, sort_keys=True)
```

Why this version is preferred:

- Dependencies stay visible at the top of the file.
- Reviewers do not need to scan function bodies to understand what the module needs.
- Lazy loading remains a deliberate exception for cases such as optional dependencies, startup cost, or import-cycle avoidance.

### Prefer early returns over nested control flow

Bad:

```python
def select_best_accuracy(metrics_json: dict[str, object] | None) -> float:
    best_accuracy = 0.0
    if metrics_json is not None:
        if "best_accuracy" in metrics_json:
            if metrics_json["best_accuracy"] is not None:
                best_accuracy = float(metrics_json["best_accuracy"])
    return best_accuracy
```

Good:

```python
def select_best_accuracy(metrics_json: dict[str, object] | None) -> float:

    # Return the default when metrics are missing entirely.
    if metrics_json is None:
        return 0.0

    best_accuracy = metrics_json.get("best_accuracy")

    # Return the default when the metrics payload has no best score.
    if best_accuracy is None:
        return 0.0

    return float(best_accuracy)
```

Why this version is preferred:

- The happy path stays flat instead of being pushed deeper by nested `if` blocks.
- Each early return documents one concrete fallback case.
- The intermediate variable makes the conversion step obvious.

### Prefer readable long strings over compressed expressions

Bad:

```python
def build_trial_summary(trial: TrialRecord) -> str:
    return f"{trial.trial_id} score={trial.score:.4f} status={trial.status} reason={trial.outcome_reason or 'n/a'} evals={int(trial.metrics_json.get('eval_count', 0)) if trial.metrics_json else 0}"
```

Good:

```python
def build_trial_summary(trial: TrialRecord) -> str:
    # Derive the summary fields before assembling the final string.
    eval_count = int(trial.metrics_json.get("eval_count", 0)) if trial.metrics_json else 0
    outcome_reason = trial.outcome_reason or "n/a"

    # Return the summary in a reviewable, field-by-field layout.
    return (
        f"{trial.trial_id} "
        f"score={trial.score:.4f} "
        f"status={trial.status} "
        f"reason={outcome_reason} "
        f"evals={eval_count}"
    )
```

Why this version is preferred:

- Formatting rules are separated from data lookup.
- Each output field is visible and easy to edit independently.
- The final string shape is obvious during review.

### Prefer helper extraction for mixed-responsibility loops

Bad:

```python
def load_track_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in rows:
        if row["status"] == "finished" and row["metrics_json"] is not None and "accuracy" in row["metrics_json"]:
            candidates.append(
                {
                    "trial_id": row["trial_id"],
                    "score": float(row["metrics_json"]["accuracy"]),
                    "is_best": float(row["metrics_json"]["accuracy"]) >= 0.9,
                }
            )
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)
```

Good:

```python
def load_track_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # Collect only rows that produce a valid finished candidate.
    candidates: list[dict[str, object]] = []
    for row in rows:
        candidate = build_finished_candidate(row)

        # Keep only rows that passed candidate validation.
        if candidate is not None:
            candidates.append(candidate)

    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def build_finished_candidate(row: dict[str, object]) -> dict[str, object] | None:

    # Ignore rows that are not finished yet.
    if row["status"] != "finished":
        return None

    metrics_json = row.get("metrics_json")

    # Ignore rows whose metrics payload is missing or malformed.
    if not isinstance(metrics_json, dict):
        return None

    accuracy = metrics_json.get("accuracy")

    # Ignore finished rows that still lack an accuracy value.
    if accuracy is None:
        return None

    score = float(accuracy)
    return {
        "trial_id": row["trial_id"],
        "score": score,
        "is_best": score >= 0.9,
    }
```

Why this version is preferred:

- The loop only iterates and collects.
- Candidate validation and shaping live behind a clear helper boundary.
- Reused values such as `score` are computed once.

### Prefer positive predicates over negated compound logic

Bad:

```python
def should_retry_dispatch(trial: TrialRecord, now: datetime) -> bool:
    return not (
        trial.status != "dispatching"
        or trial.dispatch_deadline_at is None
        or trial.dispatch_deadline_at >= now
        or trial.dispatch_attempts >= 3
    )
```

Good:

```python
def should_retry_dispatch(trial: TrialRecord, now: datetime) -> bool:
    # Name each policy requirement before composing the final decision.
    is_dispatching = trial.status == "dispatching"
    has_deadline = trial.dispatch_deadline_at is not None
    deadline_expired = has_deadline and trial.dispatch_deadline_at < now
    has_retry_budget = trial.dispatch_attempts < 3

    # Express the retry policy as one readable predicate.
    return is_dispatching and deadline_expired and has_retry_budget
```

Why this version is preferred:

- Each condition is named in domain language.
- The final expression reads like a policy statement instead of a logic puzzle.
- Individual conditions are easy to log or test.

### Prefer readable structured payloads over compressed nested literals

Bad:

```python
def build_generation_provenance(
    model: str,
    request_messages: list[dict[str, str]],
    context_trial_ids: list[str],
) -> dict[str, object]:
    return {"backend": "openrouter", "model": model, "candidate_kind": "strategy_variant", "generation_config": {"model": model, "selection_mode": "pool"}, "request_messages": request_messages, "context_trial_ids": context_trial_ids}
```

Good:

```python
def build_generation_provenance(
    model: str,
    request_messages: list[dict[str, str]],
    context_trial_ids: list[str],
) -> dict[str, object]:
    return {
        "backend": "openrouter",
        "model": model,
        "candidate_kind": "strategy_variant",
        "generation_config": {
            "model": model,
            "selection_mode": "pool",
        },
        "request_messages": request_messages,
        "context_trial_ids": context_trial_ids,
    }
```

Why this version is preferred:

- The payload shape is visible immediately.
- Nested fields stay editable without turning the return into a wall of text.
- Future fields can be added with minimal churn.
