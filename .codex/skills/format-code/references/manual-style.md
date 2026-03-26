# Manual Style Rules

Use this reference after Ruff has already run. Ruff owns deterministic formatting, import ordering, and the small set of boolean and return simplifications configured in `pyproject.toml`.

## Keep Function Flow Reviewable

- Structure non-trivial functions in a readable order: validate inputs, run the main logic, then build the return value.
- Prefer early returns when they flatten control flow, but do not force them into tiny helpers or already-clear code.
- Keep coercion, validation, and closely related state changes in the same block when they implement one idea.
- Introduce named intermediate variables for non-trivial expressions, boolean conditions, and transformed values before combining them.
- Extract a helper when one block mixes iteration, filtering, transformation, and ordering.

## Use Comments Sparingly and Precisely

- Add a short intent comment for a non-trivial logical block or business-rule branch when the purpose is not obvious from the code alone.
- Keep a comment directly attached to the block it introduces.
- Use one blank line to separate logical blocks from each other.
- Explain intent or policy, not the line below the comment.
- Keep comments short and use domain language where possible.

## Prefer Readable Construction

- Prefer positive predicates over dense negated compound logic when Ruff did not already simplify the expression.
- Prefer staged construction for nested payloads when the payload shape carries meaning.
- Keep long formatted strings visually segmented by field instead of compressing them into one hard-to-edit expression.
- Do not fight Ruff formatting for tiny helpers that are already easy to read.

## Do Not Over-Apply

- Do not add a comment above every obvious branch or statement.
- Do not stack one-line comment/code pairs when those lines belong to one logical block.
- Do not split coercion from its paired validation when both lines express the same check.
- Do not use filler comments such as `Set variable` or `Return result`.
- Do not satisfy a branch-comment rule with boilerplate such as `Check condition`.
- Do not extract helpers unless the original block is carrying multiple responsibilities.

## Reference Examples

### Structured Function Layout

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

### Named Boolean Conditions

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
