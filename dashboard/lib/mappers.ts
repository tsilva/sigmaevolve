import type {
  TrackListItem,
  TrialListItem,
} from "@/lib/types";

type TrackRow = {
  trackId: string;
  datasetId: string;
  createdAt: string | Date;
  totalTrials: number | string | null;
  queuedTrials: number | string | null;
  dispatchingTrials: number | string | null;
  activeTrials: number | string | null;
  finishedTrials: number | string | null;
  errorTrials: number | string | null;
  succeededTrials: number | string | null;
  bestScore: number | string | null;
  bestTrialId: string | null;
  lastActivityAt: string | Date | null;
};

type TrialRow = {
  trialId: string;
  status: TrialListItem["status"];
  outcomeReason: string | null;
  modalRunId: string | null;
  modalRunUrl: string | null;
  score: number | string | null;
  accuracy: number | string | null;
  timeToBestEvalSec: number | string | null;
  timedOut: boolean | null;
  timeSinceLastEvalSec: number | string | null;
  hadUnscoredWorkAtTimeout: boolean | null;
  lastPhase: string | null;
  backend: string | null;
  model: string | null;
  dispatchAttempts: number | string | null;
  createdAt: string | Date;
  startedAt: string | Date | null;
  finishedAt: string | Date | null;
  durationSec: number | string | null;
  hasError: boolean | null;
  source: string | null;
  errorJson: Record<string, unknown> | null;
  provenanceJson: Record<string, unknown> | null;
};

const FAILURE_OUTCOMES = new Set(["crashed", "eval_failed", "stale", "generation_failed"]);

function hasErrorSignal(value: Record<string, unknown> | null): boolean {
  // Treat any populated reason, detail, stderr, or returncode as an error signal.
  if (!value) {
    return false;
  }
  const reason = value.reason;
  if (typeof reason === "string" && reason.trim().length > 0) {
    return true;
  }
  const detail = value.detail;
  if (typeof detail === "string" && detail.trim().length > 0) {
    return true;
  }
  const stderr = value.stderr;
  if (typeof stderr === "string" && stderr.trim().length > 0) {
    return true;
  }
  return value.returncode !== null && value.returncode !== undefined;
}

function asIsoDate(value: string | Date | null | undefined): string | null {
  // Normalize nullable timestamps into ISO strings for the dashboard types.
  if (!value) {
    return null;
  }
  return new Date(value).toISOString();
}

function asNumber(value: number | string | null | undefined): number {
  // Coerce nullable numeric fields into zero-based dashboard defaults.
  if (value === null || value === undefined) {
    return 0;
  }
  return Number(value);
}

function asNullableNumber(value: number | string | null | undefined): number | null {
  // Preserve nullability while still coercing numeric strings into numbers.
  if (value === null || value === undefined) {
    return null;
  }
  return Number(value);
}

function asNullableString(value: string | null | undefined): string | null {
  // Collapse empty string fields into null for cleaner rendering logic.
  if (typeof value !== "string" || value.length === 0) {
    return null;
  }
  return value;
}

function formatStructuredReasoningTrace(value: unknown): string | null {
  if (typeof value === "string") {
    return asNullableString(value);
  }

  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }

  const encryptedEntries = value.filter(
    (entry): entry is { format?: unknown; type?: unknown } =>
      Boolean(entry) &&
      typeof entry === "object" &&
      (entry as { type?: unknown }).type === "reasoning.encrypted",
  );

  if (encryptedEntries.length === value.length) {
    const formats = Array.from(
      new Set(
        encryptedEntries
          .map((entry) => (typeof entry.format === "string" ? entry.format.trim() : ""))
          .filter((entry) => entry.length > 0),
      ),
    );

    if (formats.length > 0) {
      return `Reasoning trace unavailable. Provider returned encrypted reasoning blocks (${formats.join(", ")}).`;
    }

    return "Reasoning trace unavailable. Provider returned encrypted reasoning blocks.";
  }

  return "Reasoning trace unavailable. Provider returned a structured reasoning payload that cannot be rendered.";
}

function asStringArray(value: unknown): string[] {
  // Keep only non-empty string entries from mixed JSON arrays.
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((entry): entry is string => typeof entry === "string" && entry.length > 0);
}

function getGenerationPayload(
  provenanceJson: Record<string, unknown> | null,
): Record<string, unknown> | null {
  // Lift the nested generation payload into a typed record when present.
  const generation = provenanceJson?.generation;
  if (!generation || typeof generation !== "object") {
    return null;
  }
  return generation as Record<string, unknown>;
}

function classifyErrorType(
  outcomeReason: string | null,
  errorJson: Record<string, unknown> | null,
): string | null {
  const reason = typeof errorJson?.reason === "string" ? errorJson.reason : null;
  const finishReason = typeof errorJson?.finish_reason === "string" ? errorJson.finish_reason : null;
  const detail = typeof errorJson?.detail === "string" ? errorJson.detail : null;
  const reachedLengthLimit = finishReason === "length";
  const detailMentionsReasoning = typeof detail === "string" && detail.toLowerCase().includes("reasoning");

  if (outcomeReason === "generation_failed") {
    if (reason === "provider_response_missing_content" && reachedLengthLimit && detailMentionsReasoning) {
      return "generation_reasoning_tokens_exhausted";
    }
    if (
      (reason === "candidate_materialization_failed" || reason === "generation_assertion_failed") &&
      reachedLengthLimit
    ) {
      return "generation_output_truncated";
    }
    if (reason === "candidate_materialization_failed" || reason === "generation_assertion_failed") {
      return "generation_invalid_candidate";
    }
    if (reason === "generator_exception") {
      return "generation_backend_exception";
    }
    if (
      reason === "provider_http_error" ||
      reason === "provider_request_failed" ||
      reason === "provider_response_invalid_json" ||
      reason === "provider_response_missing_choices" ||
      reason === "provider_response_missing_content"
    ) {
      return "generation_provider_failure";
    }
    return "generation_failed";
  }

  if (outcomeReason === "crashed") {
    return "execution_crash";
  }

  if (outcomeReason === "eval_failed") {
    if (reason === "train_script_contract_violation") {
      return "execution_contract_violation";
    }
    if (reason === "prediction_load_failed") {
      return "evaluation_artifact_error";
    }
    if (reason === "predictions_missing") {
      return "evaluation_predictions_missing";
    }
    return "evaluation_failed";
  }

  if (outcomeReason === "stale") {
    if (reason === "dispatch_deadline_expired") {
      return "dispatch_stale";
    }
    if (reason === "heartbeat_stale") {
      return "runner_stale";
    }
    return "stale";
  }

  return null;
}

export function mapTrackListItem(row: TrackRow): TrackListItem {
  // Convert database row fields into the normalized dashboard track shape.
  return {
    trackId: row.trackId,
    datasetId: row.datasetId,
    createdAt: asIsoDate(row.createdAt) ?? new Date(0).toISOString(),
    totalTrials: asNumber(row.totalTrials),
    queuedTrials: asNumber(row.queuedTrials),
    dispatchingTrials: asNumber(row.dispatchingTrials),
    activeTrials: asNumber(row.activeTrials),
    finishedTrials: asNumber(row.finishedTrials),
    errorTrials: asNumber(row.errorTrials),
    succeededTrials: asNumber(row.succeededTrials),
    bestScore: asNullableNumber(row.bestScore),
    bestTrialId: asNullableString(row.bestTrialId),
    lastActivityAt: asIsoDate(row.lastActivityAt ?? row.createdAt) ?? new Date(0).toISOString(),
  };
}

export function mapTrialListItem(row: TrialRow): TrialListItem {
  // Derive the trial-level error and generation state before building the response.
  const hasError = FAILURE_OUTCOMES.has(row.outcomeReason ?? "") || hasErrorSignal(row.errorJson ?? null);
  const generation = getGenerationPayload(row.provenanceJson ?? null);
  const errorType = classifyErrorType(row.outcomeReason, row.errorJson ?? null);

  // Return the dashboard-friendly trial view with normalized scalar fields.
  return {
    trialId: row.trialId,
    status: row.status,
    outcomeReason: row.outcomeReason,
    modalRunId: asNullableString(row.modalRunId),
    modalRunUrl: asNullableString(row.modalRunUrl),
    score: asNumber(row.score),
    accuracy: asNullableNumber(row.accuracy),
    timeToBestEvalSec: asNullableNumber(row.timeToBestEvalSec),
    timedOut: Boolean(row.timedOut),
    timeSinceLastEvalSec: asNullableNumber(row.timeSinceLastEvalSec),
    hadUnscoredWorkAtTimeout: Boolean(row.hadUnscoredWorkAtTimeout),
    lastPhase: row.lastPhase,
    backend: row.backend,
    model: row.model,
    dispatchAttempts: asNumber(row.dispatchAttempts),
    createdAt: asIsoDate(row.createdAt) ?? new Date(0).toISOString(),
    startedAt: asIsoDate(row.startedAt),
    finishedAt: asIsoDate(row.finishedAt),
    durationSec: asNullableNumber(row.durationSec),
    hasError,
    errorType,
    source: row.source ?? "",
    taskDescription: asNullableString(generation?.task_description as string | null | undefined),
    responseText: asNullableString(generation?.response_text as string | null | undefined),
    reasoningText: formatStructuredReasoningTrace(generation?.reasoning_text),
    generatedSource: asNullableString(generation?.generated_source as string | null | undefined),
    generationAssertionsPassed:
      typeof generation?.assertions_passed === "boolean" ? (generation.assertions_passed as boolean) : null,
    generationAssertionFailures: asStringArray(generation?.assertion_failures),
    errorJson: row.errorJson ?? null,
    provenanceJson: row.provenanceJson ?? null,
  };
}
