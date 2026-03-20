"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useDeferredValue, useEffect, useEffectEvent, useState, useTransition, type KeyboardEvent } from "react";

import { HighlightedCode } from "@/components/highlighted-code";
import { useTrackLiveUpdates } from "@/hooks/use-track-live-updates";
import type {
  PaginatedTrialsResponse,
  TrackDetailResponse,
  TrackListItem,
  TrialListItem,
  TrialStatusFilter,
} from "@/lib/types";

const STATUS_OPTIONS: TrialStatusFilter[] = ["all", "queued", "dispatching", "active", "finished"];

async function fetchJson<T>(input: string): Promise<T> {
  const response = await fetch(input, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

function formatDate(value: string | null): string {
  if (!value) {
    return "Pending";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatRelativeMinutes(value: string): string {
  const deltaMs = new Date().getTime() - new Date(value).getTime();
  const deltaMinutes = Math.max(0, Math.round(deltaMs / 60_000));
  if (deltaMinutes < 1) {
    return "just now";
  }
  if (deltaMinutes < 60) {
    return `${deltaMinutes}m ago`;
  }
  const deltaHours = Math.round(deltaMinutes / 60);
  if (deltaHours < 24) {
    return `${deltaHours}h ago`;
  }
  const deltaDays = Math.round(deltaHours / 24);
  return `${deltaDays}d ago`;
}

function formatNumber(value: number | null, digits = 3): string {
  if (value === null) {
    return "—";
  }
  return value.toFixed(digits);
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) {
    return "0%";
  }
  return `${Math.round(value)}%`;
}

function formatDuration(value: number | null): string {
  if (value === null) {
    return "—";
  }
  if (value < 60) {
    return `${value.toFixed(1)}s`;
  }
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function formatJsonBlock(value: Record<string, unknown> | null): string {
  if (!value) {
    return "No payload recorded.";
  }
  return JSON.stringify(value, null, 2);
}

function extractCrashDetails(value: Record<string, unknown> | null): string | null {
  if (!value) {
    return null;
  }

  const stderr = value.stderr;
  if (typeof stderr === "string" && stderr.trim().length > 0) {
    return stderr.trim();
  }

  const detail = value.detail;
  if (typeof detail === "string" && detail.trim().length > 0) {
    return detail.trim();
  }

  const reason = value.reason;
  if (typeof reason === "string" && reason.trim().length > 0) {
    return reason.trim();
  }

  return null;
}

type PromptMessage = {
  role: string;
  content: string;
};

function asPromptMessages(value: Record<string, unknown> | null): PromptMessage[] {
  const raw = value?.request_messages;
  if (!Array.isArray(raw)) {
    return [];
  }

  return raw.flatMap((entry) => {
    if (!entry || typeof entry !== "object") {
      return [];
    }

    const role = (entry as { role?: unknown }).role;
    const content = (entry as { content?: unknown }).content;
    if (typeof role !== "string" || typeof content !== "string") {
      return [];
    }

    return [{ role, content }];
  });
}

function formatGenerationProperties(value: Record<string, unknown> | null): string {
  if (!value) {
    return "No provenance payload recorded.";
  }

  const payload: Record<string, unknown> = {};
  for (const key of [
    "backend",
    "model",
    "generation_index",
    "provider_response_id",
    "generation_config",
    "context_trial_ids",
  ]) {
    if (key in value) {
      payload[key] = value[key];
    }
  }

  return JSON.stringify(payload, null, 2);
}

function detectPromptLanguage(content: string): "json" | "markdown" {
  const trimmed = content.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    return "json";
  }
  return "markdown";
}

function compactIdentifier(value: string, leading = 10, trailing = 6): string {
  if (value.length <= leading + trailing + 1) {
    return value;
  }
  return `${value.slice(0, leading)}…${value.slice(-trailing)}`;
}

function summarizeCrashDetails(value: string | null): string {
  if (!value) {
    return "No crash detail recorded.";
  }

  const firstLine = value
    .split("\n")
    .map((line) => line.trim())
    .find(Boolean);

  if (!firstLine) {
    return "No crash detail recorded.";
  }

  return firstLine.length > 160 ? `${firstLine.slice(0, 157)}...` : firstLine;
}

function getTrackLabel(track: TrackListItem): string {
  return track.name ?? track.trackId;
}

function getProgressPercent(track: TrackListItem): number {
  if (track.totalTrials === 0) {
    return 0;
  }
  return (track.finishedTrials / track.totalTrials) * 100;
}

function getCoveragePercent(track: TrackListItem): number {
  if (track.totalTrials === 0) {
    return 0;
  }
  return (track.succeededTrials / track.totalTrials) * 100;
}

function getAttentionCount(track: TrackListItem): number {
  return Math.max(0, track.finishedTrials - track.succeededTrials);
}

function getTrialTone(trial: TrialListItem): "success" | "warning" | "danger" | "neutral" {
  if (trial.status !== "finished") {
    return "neutral";
  }
  if (trial.hasError) {
    return "danger";
  }
  if (trial.timedOut || trial.hadUnscoredWorkAtTimeout) {
    return "warning";
  }
  return "success";
}

function getTrialNarrative(trial: TrialListItem): string {
  if (trial.status === "queued") {
    return "Waiting to be dispatched.";
  }
  if (trial.status === "dispatching") {
    return `Dispatching attempt ${trial.dispatchAttempts}.`;
  }
  if (trial.status === "active") {
    return trial.lastPhase ? `Running in ${trial.lastPhase}.` : "Currently executing.";
  }
  if (trial.hasError) {
    return "Finished with an execution error.";
  }
  if (trial.timedOut) {
    return "Timed out before evaluation stabilized.";
  }
  if (trial.hadUnscoredWorkAtTimeout) {
    return "Ended with unevaluated work still pending.";
  }
  return trial.outcomeReason ?? "Finished cleanly.";
}

function matchesSearch(trial: TrialListItem, query: string): boolean {
  if (!query) {
    return true;
  }

  const haystack = [
    trial.trialId,
    trial.status,
    trial.outcomeReason,
    trial.backend,
    trial.model,
    trial.lastPhase,
    trial.source,
  ]
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .join(" ")
    .toLowerCase();

  return query
    .split(/\s+/)
    .filter(Boolean)
    .every((part) => haystack.includes(part));
}

type DashboardShellProps = {
  initialDetail: TrackDetailResponse;
  initialTracks: TrackListItem[];
  initialSelectedTrialId: string | null;
  selectedTrackId: string;
};

type ActiveWorkspace = "explorer" | "inspector";
type ScoreChartPoint = {
  backend: string | null;
  model: string | null;
  outcomeReason: string | null;
  score: number | null;
  status: TrialListItem["status"];
  tone: ReturnType<typeof getTrialTone>;
  trialId: string;
  x: number;
  y: number;
};

const SCORE_CHART_WIDTH = 760;
const SCORE_CHART_HEIGHT = 228;
const SCORE_CHART_PADDING = {
  top: 18,
  right: 18,
  bottom: 32,
  left: 46,
};

function buildTrialsUrl(trackId: string, status: TrialStatusFilter, cursor?: string | null, limit = 50): string {
  const params = new URLSearchParams();
  params.set("status", status);
  params.set("limit", String(limit));
  if (cursor) {
    params.set("cursor", cursor);
  }
  return `/api/tracks/${trackId}/trials?${params.toString()}`;
}

function buildScoreChart(trials: TrialListItem[]): {
  bestScore: number | null;
  linePath: string;
  points: ScoreChartPoint[];
  scoredCount: number;
  yMax: number;
  yMin: number;
} {
  const chartWidth = SCORE_CHART_WIDTH - SCORE_CHART_PADDING.left - SCORE_CHART_PADDING.right;
  const chartHeight = SCORE_CHART_HEIGHT - SCORE_CHART_PADDING.top - SCORE_CHART_PADDING.bottom;
  const orderedTrials = [...trials].sort((left, right) => {
    const createdAtDelta = new Date(left.createdAt).getTime() - new Date(right.createdAt).getTime();
    if (createdAtDelta !== 0) {
      return createdAtDelta;
    }
    return left.trialId.localeCompare(right.trialId);
  });

  const scoredValues = orderedTrials
    .filter((trial) => trial.status === "finished")
    .map((trial) => trial.score)
    .filter((score) => Number.isFinite(score));
  const bestScore = scoredValues.length > 0 ? Math.max(...scoredValues) : null;
  const rawMin = scoredValues.length > 0 ? Math.min(...scoredValues) : 0;
  const rawMax = scoredValues.length > 0 ? Math.max(...scoredValues) : 1;
  const spread = Math.max(0.02, rawMax - rawMin);
  const yMin = Math.max(0, rawMin - spread * 0.18);
  const yMax = Math.min(1, rawMax + spread * 0.18);
  const safeRange = Math.max(0.02, yMax - yMin);

  const xForIndex = (index: number): number => {
    if (orderedTrials.length <= 1) {
      return SCORE_CHART_PADDING.left + chartWidth / 2;
    }
    return SCORE_CHART_PADDING.left + (index / (orderedTrials.length - 1)) * chartWidth;
  };

  const yForScore = (score: number | null): number => {
    if (score === null) {
      return SCORE_CHART_PADDING.top + chartHeight + 6;
    }
    const normalized = (score - yMin) / safeRange;
    return SCORE_CHART_PADDING.top + chartHeight - normalized * chartHeight;
  };

  const points = orderedTrials.map((trial, index) => {
    const score = trial.status === "finished" ? trial.score : null;
    return {
      backend: trial.backend,
      model: trial.model,
      outcomeReason: trial.outcomeReason,
      score,
      status: trial.status,
      tone: getTrialTone(trial),
      trialId: trial.trialId,
      x: xForIndex(index),
      y: yForScore(score),
    };
  });

  const linePath = points.reduce((path, point) => {
    if (point.score === null) {
      return path;
    }
    return path ? `${path} L ${point.x} ${point.y}` : `M ${point.x} ${point.y}`;
  }, "");

  return {
    bestScore,
    linePath,
    points,
    scoredCount: scoredValues.length,
    yMax,
    yMin,
  };
}

export function DashboardShell({
  initialDetail,
  initialTracks,
  initialSelectedTrialId,
  selectedTrackId,
}: DashboardShellProps) {
  const router = useRouter();
  const pathname = usePathname();
  const routeTrialId = (() => {
    const prefix = `/tracks/${selectedTrackId}/trials/`;
    if (!pathname.startsWith(prefix)) {
      return null;
    }
    return decodeURIComponent(pathname.slice(prefix.length));
  })();

  const [tracks, setTracks] = useState(initialTracks);
  const [detail, setDetail] = useState(initialDetail);
  const [status, setStatus] = useState<TrialStatusFilter>("all");
  const [searchText, setSearchText] = useState("");
  const [isTracksCollapsed, setIsTracksCollapsed] = useState(false);
  const [activeWorkspace, setActiveWorkspace] = useState<ActiveWorkspace>(
    initialSelectedTrialId ? "inspector" : "explorer",
  );
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(initialSelectedTrialId);
  const [urlTrialId, setUrlTrialId] = useState<string | null>(initialSelectedTrialId);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const deferredSearchText = useDeferredValue(searchText.trim().toLowerCase());

  useEffect(() => {
    setTracks(initialTracks);
    setDetail(initialDetail);
    setStatus("all");
    setSearchText("");
    setIsTracksCollapsed(false);
    setActiveWorkspace(initialSelectedTrialId ? "inspector" : "explorer");
    setSelectedTrialId(initialSelectedTrialId);
    setUrlTrialId(initialSelectedTrialId);
    setError(null);
  }, [initialDetail, initialSelectedTrialId, initialTracks, selectedTrackId]);

  useEffect(() => {
    setUrlTrialId(routeTrialId);
  }, [routeTrialId]);

  const visibleTrials = detail.trials.filter((trial) => matchesSearch(trial, deferredSearchText));
  const selectedTrial =
    visibleTrials.find((trial) => trial.trialId === selectedTrialId) ??
    detail.trials.find((trial) => trial.trialId === selectedTrialId) ??
    null;
  const selectedTrialRank =
    selectedTrial === null
      ? null
      : detail.trials.filter((trial) => trial.score > selectedTrial.score).length + 1;
  const selectedPromptMessages = asPromptMessages(selectedTrial?.provenanceJson ?? null);
  const selectedCrashDetails = extractCrashDetails(selectedTrial?.errorJson ?? null);
  const selectedCrashSummary = summarizeCrashDetails(selectedCrashDetails);
  const progressPercent = getProgressPercent(detail.track);
  const coveragePercent = getCoveragePercent(detail.track);
  const attentionCount = getAttentionCount(detail.track);
  const scoreChart = buildScoreChart(visibleTrials);
  const bestTrial =
    detail.trials.length === 0
      ? null
      : detail.trials.reduce((best, trial) => (trial.score > best.score ? trial : best), detail.trials[0]);

  const updateTrialUrl = useEffectEvent((nextTrialId: string | null) => {
    const nextUrl = nextTrialId
      ? `/tracks/${selectedTrackId}/trials/${encodeURIComponent(nextTrialId)}`
      : `/tracks/${selectedTrackId}`;

    if (nextUrl !== pathname) {
      router.replace(nextUrl, { scroll: false });
    }
  });

  const syncSelectedTrial = useEffectEvent((nextTrialId: string | null) => {
    setSelectedTrialId(nextTrialId);
    setUrlTrialId(nextTrialId);
    updateTrialUrl(nextTrialId);
  });

  const loadTrials = useEffectEvent(async (nextStatus: TrialStatusFilter, cursor?: string | null, limit = 50) =>
    fetchJson<PaginatedTrialsResponse>(buildTrialsUrl(selectedTrackId, nextStatus, cursor, limit)),
  );

  const refreshData = useEffectEvent(async () => {
    try {
      const [nextTracks, nextTrials] = await Promise.all([
        fetchJson<TrackListItem[]>("/api/tracks"),
        loadTrials(status),
      ]);

      setTracks(nextTracks);
      setDetail((current) => ({
        track: nextTracks.find((track) => track.trackId === selectedTrackId) ?? current.track,
        trials: nextTrials.trials,
        nextCursor: nextTrials.nextCursor,
      }));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to refresh dashboard data.");
    }
  });

  useEffect(() => {
    if (visibleTrials.length === 0) {
      if (selectedTrialId !== null) {
        setSelectedTrialId(null);
      }
      if (urlTrialId !== null) {
        updateTrialUrl(null);
        setUrlTrialId(null);
      }
      return;
    }

    if (urlTrialId) {
      const routeTrial = visibleTrials.find((trial) => trial.trialId === urlTrialId);
      if (routeTrial) {
        if (urlTrialId !== selectedTrialId) {
          setSelectedTrialId(urlTrialId);
        }
        return;
      }

      setActiveWorkspace("inspector");
      syncSelectedTrial(visibleTrials[0].trialId);
      return;
    }

    if (selectedTrialId && visibleTrials.some((trial) => trial.trialId === selectedTrialId)) {
      return;
    }

    setSelectedTrialId(visibleTrials[0].trialId);
  }, [selectedTrialId, syncSelectedTrial, updateTrialUrl, urlTrialId, visibleTrials]);

  const liveMode = useTrackLiveUpdates({
    streamUrl: `/api/tracks/${selectedTrackId}/stream`,
    onRefresh: () => {
      startTransition(() => {
        void refreshData();
      });
    },
  });

  const handleStatusChange = (nextStatus: TrialStatusFilter) => {
    setStatus(nextStatus);
    startTransition(() => {
      void (async () => {
        try {
          const nextTrials = await loadTrials(nextStatus);
          setDetail((current) => ({
            ...current,
            trials: nextTrials.trials,
            nextCursor: nextTrials.nextCursor,
          }));
          setError(null);
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : "Unable to update the trial filter.");
        }
      })();
    });
  };

  const loadMore = () => {
    if (!detail.nextCursor) {
      return;
    }

    startTransition(() => {
      void (async () => {
        try {
          const nextTrials = await loadTrials(status, detail.nextCursor);
          setDetail((current) => ({
            ...current,
            trials: [...current.trials, ...nextTrials.trials],
            nextCursor: nextTrials.nextCursor,
          }));
          setError(null);
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : "Unable to load more trials.");
        }
      })();
    });
  };

  const openInspector = (trialId: string) => {
    setActiveWorkspace("inspector");
    syncSelectedTrial(trialId);
  };

  const returnToExplorer = () => {
    setActiveWorkspace("explorer");
    updateTrialUrl(null);
    setUrlTrialId(null);
  };

  const handleTrialKeyDown = (event: KeyboardEvent<HTMLTableRowElement>, trialId: string) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }

    event.preventDefault();
    openInspector(trialId);
  };

  return (
    <main className={`research-shell ${isTracksCollapsed ? "tracks-collapsed" : ""}`.trim()}>
      {isTracksCollapsed ? null : (
        <aside className="workspace-card track-column">
          <div className="section-heading">
            <div className="sidebar-header">
              <div>
                <div className="eyebrow">Tracks</div>
                <h1 className="section-title">Research lanes</h1>
              </div>
              <button
                type="button"
                className="panel-toggle"
                onClick={() => setIsTracksCollapsed(true)}
                aria-label="Collapse tracks sidebar"
              >
                Hide
              </button>
            </div>
            <p className="section-copy">Switch tracks without losing the current trial context.</p>
          </div>

          <div className="track-stack">
            {tracks.map((track) => {
              const isActive = track.trackId === selectedTrackId;
              return (
                <Link
                  key={track.trackId}
                  href={`/tracks/${track.trackId}`}
                  className={`track-card ${isActive ? "active" : ""}`}
                >
                  <div className="track-card-top">
                    <div>
                      <div className="track-card-title">{getTrackLabel(track)}</div>
                      <div className="track-card-subtitle">{track.datasetId}</div>
                    </div>
                    <div className="track-score">{formatNumber(track.bestScore, 4)}</div>
                  </div>
                  <div className="track-card-bar">
                    <span style={{ width: `${getProgressPercent(track)}%` }} />
                  </div>
                  <div className="track-card-meta">
                    <span>{track.finishedTrials}/{track.totalTrials} finished</span>
                    <span>{track.activeTrials} active</span>
                  </div>
                  <div className="track-card-meta">
                    <span>{track.succeededTrials} scored</span>
                    <span>{formatRelativeMinutes(track.lastActivityAt)}</span>
                  </div>
                </Link>
              );
            })}
          </div>
        </aside>
      )}

      <section className="research-main">
        {isTracksCollapsed ? (
          <div className="main-toolbar">
            <button
              type="button"
              className="panel-toggle"
              onClick={() => setIsTracksCollapsed(false)}
              aria-label="Expand tracks sidebar"
            >
              Show tracks
            </button>
          </div>
        ) : null}
        <section className="workspace-card overview-panel">
          <div className="overview-hero">
            <div>
              <div className="eyebrow">Track Overview</div>
              <h2 className="hero-title">{getTrackLabel(detail.track)}</h2>
              <p className="hero-copy">
                Debug research progress from the queue down to the exact source, prompt context, and failure
                payload for each trial.
              </p>
            </div>
            <div className="hero-meta">
              <span className="meta-chip meta-chip-mono" title={detail.track.trackId}>
                {compactIdentifier(detail.track.trackId, 12, 8)}
              </span>
              <span className="meta-chip">{detail.track.datasetId}</span>
              <span className="meta-chip">Created {formatDate(detail.track.createdAt)}</span>
              <span className="meta-chip">Live via {liveMode}</span>
            </div>
          </div>

          <div className="hero-metrics">
            <article className="metric-tile">
              <span className="metric-label">Best Score</span>
              <strong className="metric-value">{formatNumber(detail.track.bestScore, 4)}</strong>
              <span className="metric-note">
                {bestTrial ? `${compactIdentifier(bestTrial.trialId)} leads the visible sample.` : "Waiting for scored trials."}
              </span>
            </article>
            <article className="metric-tile">
              <span className="metric-label">Completion</span>
              <strong className="metric-value">{formatPercent(progressPercent)}</strong>
              <span className="metric-note">
                {detail.track.finishedTrials} of {detail.track.totalTrials} trials have finished.
              </span>
            </article>
            <article className="metric-tile">
              <span className="metric-label">Coverage</span>
              <strong className="metric-value">{formatPercent(coveragePercent)}</strong>
              <span className="metric-note">{detail.track.succeededTrials} runs produced scored metrics.</span>
            </article>
            <article className="metric-tile">
              <span className="metric-label">Attention</span>
              <strong className="metric-value">{attentionCount}</strong>
              <span className="metric-note">
                {detail.track.activeTrials > 0
                  ? `${detail.track.activeTrials} trials are still running.`
                  : "No active trials right now."}
              </span>
            </article>
          </div>

          <div className="overview-grid">
            <article className="analysis-card">
              <div className="analysis-card-header">
                <h3>Progress breakdown</h3>
                <span>{formatPercent(progressPercent)} complete</span>
              </div>
              <div className="progress-strip" aria-label="Track progress">
                <span className="queued" style={{ width: `${detail.track.totalTrials === 0 ? 0 : (detail.track.queuedTrials / detail.track.totalTrials) * 100}%` }} />
                <span className="dispatching" style={{ width: `${detail.track.totalTrials === 0 ? 0 : (detail.track.dispatchingTrials / detail.track.totalTrials) * 100}%` }} />
                <span className="active" style={{ width: `${detail.track.totalTrials === 0 ? 0 : (detail.track.activeTrials / detail.track.totalTrials) * 100}%` }} />
                <span className="finished" style={{ width: `${detail.track.totalTrials === 0 ? 0 : (detail.track.finishedTrials / detail.track.totalTrials) * 100}%` }} />
              </div>
              <div className="legend-grid">
                <div className="legend-row">
                  <span className="status-indicator queued" />
                  <span>Queued</span>
                  <strong>{detail.track.queuedTrials}</strong>
                </div>
                <div className="legend-row">
                  <span className="status-indicator dispatching" />
                  <span>Dispatching</span>
                  <strong>{detail.track.dispatchingTrials}</strong>
                </div>
                <div className="legend-row">
                  <span className="status-indicator active" />
                  <span>Active</span>
                  <strong>{detail.track.activeTrials}</strong>
                </div>
                <div className="legend-row">
                  <span className="status-indicator finished" />
                  <span>Finished</span>
                  <strong>{detail.track.finishedTrials}</strong>
                </div>
              </div>
            </article>

            <article className="analysis-card">
              <div className="analysis-card-header">
                <h3>Debug priorities</h3>
                <span>{visibleTrials.length} visible trials</span>
              </div>
              <div className="priority-list">
                <div className="priority-item">
                  <span className="priority-label">Error payloads</span>
                  <strong>{detail.trials.filter((trial) => trial.hasError).length}</strong>
                </div>
                <div className="priority-item">
                  <span className="priority-label">Timed out runs</span>
                  <strong>{detail.trials.filter((trial) => trial.timedOut).length}</strong>
                </div>
                <div className="priority-item">
                  <span className="priority-label">Unevaluated exits</span>
                  <strong>{detail.trials.filter((trial) => trial.hadUnscoredWorkAtTimeout).length}</strong>
                </div>
                <div className="priority-item">
                  <span className="priority-label">Last activity</span>
                  <strong>{formatDate(detail.track.lastActivityAt)}</strong>
                </div>
              </div>
            </article>

            <article className="analysis-card wide-card">
              <div className="analysis-card-header">
                <h3>Score History</h3>
                <span>
                  {scoreChart.scoredCount} scored / {visibleTrials.length} displayed
                </span>
              </div>
              <div className="score-chart-meta">
                <span>Best {formatNumber(scoreChart.bestScore, 4)}</span>
                <span>Range {formatNumber(scoreChart.yMin, 4)} to {formatNumber(scoreChart.yMax, 4)}</span>
              </div>
              <div className="score-chart-shell">
                <svg
                  className="score-chart"
                  viewBox={`0 0 ${SCORE_CHART_WIDTH} ${SCORE_CHART_HEIGHT}`}
                  role="img"
                  aria-label="Score history for the trials currently displayed in the table"
                >
                  <line
                    className="score-axis"
                    x1={SCORE_CHART_PADDING.left}
                    y1={SCORE_CHART_HEIGHT - SCORE_CHART_PADDING.bottom}
                    x2={SCORE_CHART_WIDTH - SCORE_CHART_PADDING.right}
                    y2={SCORE_CHART_HEIGHT - SCORE_CHART_PADDING.bottom}
                  />
                  <line
                    className="score-axis"
                    x1={SCORE_CHART_PADDING.left}
                    y1={SCORE_CHART_PADDING.top}
                    x2={SCORE_CHART_PADDING.left}
                    y2={SCORE_CHART_HEIGHT - SCORE_CHART_PADDING.bottom}
                  />
                  {[scoreChart.yMax, (scoreChart.yMax + scoreChart.yMin) / 2, scoreChart.yMin].map((tick) => {
                    const y =
                      SCORE_CHART_PADDING.top +
                      ((scoreChart.yMax - tick) / Math.max(0.02, scoreChart.yMax - scoreChart.yMin)) *
                        (SCORE_CHART_HEIGHT - SCORE_CHART_PADDING.top - SCORE_CHART_PADDING.bottom);
                    return (
                      <g key={tick}>
                        <line
                          className="score-gridline"
                          x1={SCORE_CHART_PADDING.left}
                          y1={y}
                          x2={SCORE_CHART_WIDTH - SCORE_CHART_PADDING.right}
                          y2={y}
                        />
                        <text className="score-tick-label" x={SCORE_CHART_PADDING.left - 10} y={y + 4}>
                          {formatNumber(tick, 3)}
                        </text>
                      </g>
                    );
                  })}
                  {scoreChart.linePath ? <path className="score-line" d={scoreChart.linePath} /> : null}
                  {scoreChart.points.map((point, index) => (
                    <g key={point.trialId}>
                      <circle
                        className={`score-point tone-${point.tone} ${point.score === null ? "pending" : "scored"}`}
                        cx={point.x}
                        cy={point.y}
                        r={point.score === null ? 3.5 : 4.5}
                      >
                        <title>
                          {`#${index + 1} ${point.trialId} • ${point.status}${point.score === null ? "" : ` • score ${formatNumber(point.score, 4)}`}${point.model ? ` • ${point.model}` : ""}`}
                        </title>
                      </circle>
                    </g>
                  ))}
                  <text
                    className="score-axis-label"
                    x={SCORE_CHART_WIDTH - SCORE_CHART_PADDING.right}
                    y={SCORE_CHART_HEIGHT - 8}
                    textAnchor="end"
                  >
                    Trial order
                  </text>
                </svg>
              </div>
            </article>
          </div>
        </section>

        <div className={`workspace-stage workspace-stage-${activeWorkspace}`}>
          {activeWorkspace === "explorer" ? (
          <section className="workspace-card explorer-panel">
            <div className="section-heading">
              <div className="eyebrow">Trial Explorer</div>
              <h2 className="section-title">How each run went</h2>
              <p className="section-copy">
                Scan outcomes quickly, then open the run inspector to compare source, crash detail, and
                provenance.
              </p>
            </div>

            <div className="toolbar-row">
              <label className="search-field">
                <span className="search-label">Search trials</span>
                <input
                  type="search"
                  value={searchText}
                  onChange={(event) => setSearchText(event.target.value)}
                  placeholder="trial id, model, phase, outcome"
                />
              </label>
              <div className="toolbar-meta">
                <span className="meta-chip">{visibleTrials.length} shown</span>
                <span className="meta-chip">{isPending ? "Refreshing" : "Stable"}</span>
              </div>
            </div>

            <div className="status-filter" role="tablist" aria-label="Trial status filters">
              {STATUS_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`filter-chip ${status === option ? "active" : ""}`}
                  onClick={() => handleStatusChange(option)}
                  disabled={isPending}
                >
                  {option}
                </button>
              ))}
            </div>

            {error ? <div className="error-banner">{error}</div> : null}

            <div className="trial-table-shell" aria-label="Trials">
              {visibleTrials.length === 0 ? (
                <section className="empty-panel">
                  <div className="eyebrow">No matching trials</div>
                  <h3>Nothing matches the current filter.</h3>
                  <p className="section-copy">Change the status filter or search query to bring runs back into view.</p>
                </section>
              ) : (
                <table className="trial-table">
                  <thead>
                    <tr>
                      <th scope="col">Trial</th>
                      <th scope="col">Status</th>
                      <th scope="col">Score</th>
                      <th scope="col">Accuracy</th>
                      <th scope="col">Duration</th>
                      <th scope="col">Model</th>
                      <th scope="col">Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleTrials.map((trial) => (
                      <tr
                        key={trial.trialId}
                        role="button"
                        tabIndex={0}
                        aria-label={`Open trial ${trial.trialId}`}
                        className={`trial-row tone-${getTrialTone(trial)} ${selectedTrial?.trialId === trial.trialId ? "active" : ""}`}
                        onClick={() => openInspector(trial.trialId)}
                        onKeyDown={(event) => handleTrialKeyDown(event, trial.trialId)}
                      >
                        <td>
                          <div className="trial-cell-primary">{trial.trialId}</div>
                          <div className="trial-cell-secondary">{getTrialNarrative(trial)}</div>
                        </td>
                        <td>
                          <span className={`status-badge status-${trial.status}`}>
                            <span className={`status-indicator ${trial.status}`} />
                            {trial.status}
                          </span>
                        </td>
                        <td>{formatNumber(trial.score, 4)}</td>
                        <td>{formatNumber(trial.accuracy, 4)}</td>
                        <td>{formatDuration(trial.durationSec)}</td>
                        <td>
                          <div className="trial-cell-primary">{trial.model ?? "unknown model"}</div>
                          <div className="trial-cell-secondary">{trial.backend ?? "unknown backend"}</div>
                        </td>
                        <td>
                          <div className="trial-notes">
                            {trial.outcomeReason ? <span className="flag-chip">{trial.outcomeReason}</span> : null}
                            {trial.timedOut ? <span className="flag-chip flag-warning">timed out</span> : null}
                            {trial.hadUnscoredWorkAtTimeout ? <span className="flag-chip flag-warning">unevaluated work</span> : null}
                            {trial.hasError ? <span className="flag-chip flag-danger">error payload</span> : null}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            {detail.nextCursor ? (
              <button type="button" className="load-more" onClick={loadMore} disabled={isPending}>
                {isPending ? "Loading…" : "Load more trials"}
              </button>
            ) : null}
          </section>
          ) : null}

          {activeWorkspace === "inspector" ? (
          <section className="workspace-card inspector-panel">
            <div className="section-heading">
              <div className="inspector-header">
                <div>
                  <div className="eyebrow">Run Inspector</div>
                  <h2 className="section-title">Why the selected run behaved that way</h2>
                </div>
                <button
                  type="button"
                  className="panel-toggle"
                  onClick={returnToExplorer}
                  aria-label="Back to trial explorer"
                >
                  Back to trials
                </button>
              </div>
              <p className="section-copy">
                Inspect lifecycle timing, outcome context, prompt provenance, and the exact source evaluated.
              </p>
            </div>

            {selectedTrial ? (
              <>
                <div className="inspector-hero">
                  <div>
                    <div className="inspector-label">Selected trial</div>
                    <h2 className="inspector-title" title={selectedTrial.trialId}>
                      {compactIdentifier(selectedTrial.trialId, 14, 10)}
                    </h2>
                    <p className="section-copy">{getTrialNarrative(selectedTrial)}</p>
                  </div>
                  <div className="inspector-meta">
                    <span className={`status-badge status-${selectedTrial.status}`}>
                      <span className={`status-indicator ${selectedTrial.status}`} />
                      {selectedTrial.status}
                    </span>
                    {selectedTrialRank ? <span className="meta-chip">Rank #{selectedTrialRank} by score</span> : null}
                    <span className="meta-chip meta-chip-mono" title={selectedTrial.trialId}>
                      {selectedTrial.trialId}
                    </span>
                    <span className="meta-chip">{selectedTrial.model ?? "unknown model"}</span>
                    <span className="meta-chip">{selectedTrial.backend ?? "unknown backend"}</span>
                  </div>
                </div>

                <div className="inspector-metrics">
                  <article className="metric-tile compact">
                    <span className="metric-label">Score</span>
                    <strong className="metric-value">{formatNumber(selectedTrial.score, 4)}</strong>
                  </article>
                  <article className="metric-tile compact">
                    <span className="metric-label">Accuracy</span>
                    <strong className="metric-value">{formatNumber(selectedTrial.accuracy, 4)}</strong>
                  </article>
                  <article className="metric-tile compact">
                    <span className="metric-label">Time To Best Eval</span>
                    <strong className="metric-value">{formatDuration(selectedTrial.timeToBestEvalSec)}</strong>
                  </article>
                  <article className="metric-tile compact">
                    <span className="metric-label">Duration</span>
                    <strong className="metric-value">{formatDuration(selectedTrial.durationSec)}</strong>
                  </article>
                  <article className="metric-tile compact">
                    <span className="metric-label">Dispatch Attempts</span>
                    <strong className="metric-value">{selectedTrial.dispatchAttempts}</strong>
                  </article>
                  <article className="metric-tile compact">
                    <span className="metric-label">Idle Since Eval</span>
                    <strong className="metric-value">{formatDuration(selectedTrial.timeSinceLastEvalSec)}</strong>
                  </article>
                </div>

                <div className="flag-row inspector-flags">
                  {selectedTrial.outcomeReason ? <span className="flag-chip">{selectedTrial.outcomeReason}</span> : null}
                  {selectedTrial.timedOut ? <span className="flag-chip flag-warning">timed out</span> : null}
                  {selectedTrial.hadUnscoredWorkAtTimeout ? (
                    <span className="flag-chip flag-warning">left work unscored</span>
                  ) : null}
                  {selectedTrial.hasError ? <span className="flag-chip flag-danger">error payload captured</span> : null}
                </div>

                <div className="inspector-grid">
                  <article className="analysis-card">
                    <div className="analysis-card-header">
                      <h3>Run timeline</h3>
                    </div>
                    <div className="timeline-list">
                      <div className="timeline-row">
                        <span>Queued</span>
                        <strong>{formatDate(selectedTrial.createdAt)}</strong>
                      </div>
                      <div className="timeline-row">
                        <span>Started</span>
                        <strong>{formatDate(selectedTrial.startedAt)}</strong>
                      </div>
                      <div className="timeline-row">
                        <span>Finished</span>
                        <strong>{formatDate(selectedTrial.finishedAt)}</strong>
                      </div>
                      <div className="timeline-row">
                        <span>Last known phase</span>
                        <strong>{selectedTrial.lastPhase ?? "—"}</strong>
                      </div>
                    </div>
                  </article>

                  <article className="analysis-card">
                    <div className="analysis-card-header">
                      <h3>Failure context</h3>
                    </div>
                    <div className="context-stack">
                      <div className="context-row">
                        <span>Outcome reason</span>
                        <strong>{selectedTrial.outcomeReason ?? "Not reported"}</strong>
                      </div>
                      <div className="context-row">
                        <span>Crash detail</span>
                        <strong title={selectedCrashDetails ?? undefined}>{selectedCrashSummary}</strong>
                      </div>
                    </div>
                  </article>

                  <article className="analysis-card">
                    <div className="analysis-card-header">
                      <h3>Provenance</h3>
                    </div>
                    <div className="context-stack">
                      <div className="context-row">
                        <span>Backend</span>
                        <strong>{selectedTrial.backend ?? "Unknown"}</strong>
                      </div>
                      <div className="context-row">
                        <span>Model</span>
                        <strong>{selectedTrial.model ?? "Unknown"}</strong>
                      </div>
                    </div>
                    <HighlightedCode code={formatGenerationProperties(selectedTrial.provenanceJson)} language="json" wrap />
                  </article>

                  {selectedPromptMessages.length > 0 ? (
                    <article className="analysis-card wide-card">
                      <div className="analysis-card-header">
                        <h3>Prompt context</h3>
                        <span>{selectedPromptMessages.length} messages</span>
                      </div>
                      <div className="prompt-stack">
                        {selectedPromptMessages.map((message, index) => (
                          <section key={`${message.role}-${index}`} className="prompt-card">
                            <div className="prompt-role">{message.role}</div>
                            <HighlightedCode code={message.content} language={detectPromptLanguage(message.content)} wrap />
                          </section>
                        ))}
                      </div>
                    </article>
                  ) : null}

                  {selectedTrial.hasError ? (
                    <article className="analysis-card wide-card">
                      <div className="analysis-card-header">
                        <h3>Error payload</h3>
                      </div>
                      <HighlightedCode code={formatJsonBlock(selectedTrial.errorJson)} language="json" wrap />
                    </article>
                  ) : null}

                  <article className="analysis-card wide-card">
                    <div className="analysis-card-header">
                      <h3>Source under test</h3>
                    </div>
                    <HighlightedCode code={selectedTrial.source || "// No source captured."} language="python" wrap />
                  </article>
                </div>
              </>
            ) : (
              <section className="empty-panel">
                <div className="eyebrow">No selection</div>
                <h3>Select a trial to inspect it.</h3>
                <p className="section-copy">
                  The right-hand pane is reserved for why-this-run debugging: outcome context, provenance, and
                  source.
                </p>
              </section>
            )}
          </section>
          ) : null}
        </div>
      </section>
    </main>
  );
}
