import type {
  TrackListItem,
  TrialListItem,
} from "@/lib/types";

type TrackRow = {
  trackId: string;
  name: string | null;
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
  errorType: string | null;
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

export function mapTrackListItem(row: TrackRow): TrackListItem {
  // Convert database row fields into the normalized dashboard track shape.
  return {
    trackId: row.trackId,
    name: row.name,
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
    errorType: asNullableString(row.errorType),
    source: row.source ?? "",
    responseText: asNullableString(generation?.response_text as string | null | undefined),
    generatedSource: asNullableString(generation?.generated_source as string | null | undefined),
    generationAssertionsPassed:
      typeof generation?.assertions_passed === "boolean" ? (generation.assertions_passed as boolean) : null,
    generationAssertionFailures: asStringArray(generation?.assertion_failures),
    errorJson: row.errorJson ?? null,
    provenanceJson: row.provenanceJson ?? null,
  };
}
