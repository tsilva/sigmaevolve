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
  succeededTrials: number | string | null;
  bestScore: number | string | null;
  lastActivityAt: string | Date | null;
};

type TrialRow = {
  trialId: string;
  status: TrialListItem["status"];
  outcomeReason: string | null;
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

const FAILURE_OUTCOMES = new Set(["crashed", "eval_failed", "stale"]);

function asIsoDate(value: string | Date | null | undefined): string | null {
  if (!value) {
    return null;
  }
  return new Date(value).toISOString();
}

function asNumber(value: number | string | null | undefined): number {
  if (value === null || value === undefined) {
    return 0;
  }
  return Number(value);
}

function asNullableNumber(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  return Number(value);
}

export function mapTrackListItem(row: TrackRow): TrackListItem {
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
    succeededTrials: asNumber(row.succeededTrials),
    bestScore: asNullableNumber(row.bestScore),
    lastActivityAt: asIsoDate(row.lastActivityAt ?? row.createdAt) ?? new Date(0).toISOString(),
  };
}

export function mapTrialListItem(row: TrialRow): TrialListItem {
  const hasError = FAILURE_OUTCOMES.has(row.outcomeReason ?? "") || Boolean(row.hasError);
  return {
    trialId: row.trialId,
    status: row.status,
    outcomeReason: row.outcomeReason,
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
    source: row.source ?? "",
    errorJson: row.errorJson ?? null,
    provenanceJson: row.provenanceJson ?? null,
  };
}
