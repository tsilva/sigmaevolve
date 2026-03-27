export type TrialStatus = "queued" | "dispatching" | "active" | "finished" | "error";
export type TrialStatusFilter = TrialStatus | "all";
export type DashboardNotificationReason = "trial_changed" | "track_changed";

export type TrackListItem = {
  trackId: string;
  datasetId: string;
  createdAt: string;
  totalTrials: number;
  queuedTrials: number;
  dispatchingTrials: number;
  activeTrials: number;
  finishedTrials: number;
  errorTrials: number;
  succeededTrials: number;
  bestScore: number | null;
  bestTrialId: string | null;
  lastActivityAt: string;
};

export type TrialListItem = {
  trialId: string;
  status: TrialStatus;
  outcomeReason: string | null;
  modalRunId: string | null;
  modalRunUrl: string | null;
  score: number;
  accuracy: number | null;
  bestEvalEpoch: number | null;
  epochsCompleted: number | null;
  evalCount: number | null;
  timeToBestEvalSec: number | null;
  timedOut: boolean;
  timeSinceLastEvalSec: number | null;
  hadUnscoredWorkAtTimeout: boolean;
  lastPhase: string | null;
  backend: string | null;
  model: string | null;
  dispatchAttempts: number;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  durationSec: number | null;
  hasError: boolean;
  errorType: string | null;
  source: string;
  taskDescription: string | null;
  responseText: string | null;
  reasoningText: string | null;
  generatedSource: string | null;
  generationAssertionsPassed: boolean | null;
  generationAssertionFailures: string[];
  errorJson: Record<string, unknown> | null;
  provenanceJson: Record<string, unknown> | null;
};

export type PaginatedTrialsResponse = {
  trials: TrialListItem[];
  nextCursor: string | null;
};

export type TrackDetailResponse = PaginatedTrialsResponse & {
  track: TrackListItem;
};

export type DashboardNotification = {
  trackId: string;
  reason: DashboardNotificationReason;
};
