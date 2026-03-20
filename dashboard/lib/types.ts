export type TrialStatus = "queued" | "dispatching" | "active" | "finished";
export type TrialStatusFilter = TrialStatus | "all";
export type DashboardNotificationReason = "trial_changed" | "track_changed";

export type TrackListItem = {
  trackId: string;
  name: string | null;
  datasetId: string;
  createdAt: string;
  totalTrials: number;
  queuedTrials: number;
  dispatchingTrials: number;
  activeTrials: number;
  finishedTrials: number;
  succeededTrials: number;
  bestScore: number | null;
  lastActivityAt: string;
};

export type TrialListItem = {
  trialId: string;
  status: TrialStatus;
  outcomeReason: string | null;
  score: number;
  accuracy: number | null;
  backend: string | null;
  model: string | null;
  dispatchAttempts: number;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  durationSec: number | null;
  hasError: boolean;
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
