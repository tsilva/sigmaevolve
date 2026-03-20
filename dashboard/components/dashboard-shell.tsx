"use client";

import Link from "next/link";
import { useDeferredValue, useEffect, useEffectEvent, useState, useTransition } from "react";

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

function formatNumber(value: number | null, digits = 3): string {
  if (value === null) {
    return "—";
  }
  return value.toFixed(digits);
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

type DashboardShellProps = {
  initialDetail: TrackDetailResponse;
  initialTracks: TrackListItem[];
  selectedTrackId: string;
};

export function DashboardShell({
  initialDetail,
  initialTracks,
  selectedTrackId,
}: DashboardShellProps) {
  const [tracks, setTracks] = useState(initialTracks);
  const [detail, setDetail] = useState(initialDetail);
  const [status, setStatus] = useState<TrialStatusFilter>("all");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const deferredTrials = useDeferredValue(detail.trials);

  useEffect(() => {
    setTracks(initialTracks);
    setDetail(initialDetail);
    setStatus("all");
    setError(null);
  }, [initialDetail, initialTracks, selectedTrackId]);

  const loadTrials = useEffectEvent(async (nextStatus: TrialStatusFilter, cursor?: string | null) => {
    const params = new URLSearchParams();
    params.set("status", nextStatus);
    params.set("limit", "50");
    if (cursor) {
      params.set("cursor", cursor);
    }
    return fetchJson<PaginatedTrialsResponse>(`/api/tracks/${selectedTrackId}/trials?${params.toString()}`);
  });

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

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="eyebrow">SigmaEvolve</div>
        <h1 className="headline">Experiment command deck</h1>
        <p className="subtle">
          Watch tracks mutate in place and drill into trial state without leaving the deployed dashboard.
        </p>

        <div className="track-list">
          {tracks.map((track) => (
            <Link
              key={track.trackId}
              href={`/tracks/${track.trackId}`}
              className={`track-card ${track.trackId === selectedTrackId ? "active" : ""}`}
            >
              <h3 className="track-name">{track.name ?? track.trackId}</h3>
              <div className="track-meta">
                <span>{track.datasetId}</span>
                <span>Best {formatNumber(track.bestScore, 4)}</span>
              </div>
              <div className="track-meta">
                <span>{track.totalTrials} trials</span>
                <span>Updated {formatDate(track.lastActivityAt)}</span>
              </div>
            </Link>
          ))}
        </div>
      </aside>

      <section className="content">
        <div className="content-grid">
          <section className="panel hero-panel">
            <div className="eyebrow">Selected Track</div>
            <div className="toolbar">
              <div>
                <h2>{detail.track.name ?? detail.track.trackId}</h2>
                <div className="detail-meta">
                  <span>{detail.track.trackId}</span>
                  <span>{detail.track.datasetId}</span>
                  <span>Created {formatDate(detail.track.createdAt)}</span>
                </div>
              </div>
              <div className="live-state">
                Live transport: <strong>{liveMode}</strong>
                {isPending ? " · refreshing" : ""}
              </div>
            </div>

            <div className="metric-grid">
              <div className="metric">
                <span className="metric-label">Best Score</span>
                <span className="metric-value">{formatNumber(detail.track.bestScore, 4)}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Total Trials</span>
                <span className="metric-value">{detail.track.totalTrials}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Scored</span>
                <span className="metric-value">{detail.track.succeededTrials}</span>
              </div>
              <div className="metric">
                <span className="metric-label">Last Activity</span>
                <span className="metric-value" style={{ fontSize: "1rem" }}>
                  {formatDate(detail.track.lastActivityAt)}
                </span>
              </div>
            </div>

            <div className="summary-strip">
              <span className="pill">
                <span className="status-dot queued" />
                Queued {detail.track.queuedTrials}
              </span>
              <span className="pill">
                <span className="status-dot dispatching" />
                Dispatching {detail.track.dispatchingTrials}
              </span>
              <span className="pill">
                <span className="status-dot active" />
                Active {detail.track.activeTrials}
              </span>
              <span className="pill">
                <span className="status-dot finished" />
                Finished {detail.track.finishedTrials}
              </span>
            </div>

            {error ? <div className="error-banner">{error}</div> : null}
          </section>

          <section className="panel">
            <div className="toolbar">
              <div>
                <div className="eyebrow">Trials</div>
                <h3>Newest first</h3>
              </div>
              <div className="live-state">Filter updates preserve live refresh.</div>
            </div>

            <div className="status-filter" role="tablist" aria-label="Trial status filters">
              {STATUS_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={`status-chip ${status === option ? "active" : ""}`}
                  onClick={() => handleStatusChange(option)}
                  disabled={isPending}
                >
                  {option}
                </button>
              ))}
            </div>

            <div className="table-wrap">
              <table className="trial-table">
                <thead>
                  <tr>
                    <th>Trial</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Accuracy</th>
                    <th>Backend</th>
                    <th>Dispatches</th>
                    <th>Started</th>
                    <th>Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {deferredTrials.map((trial: TrialListItem) => (
                    <tr key={trial.trialId}>
                      <td>
                        <code>{trial.trialId}</code>
                        <div className="subtle">{trial.model ?? "unknown model"}</div>
                      </td>
                      <td>
                        <span className="trial-status">
                          <span className={`status-dot ${trial.status}`} />
                          {trial.status}
                        </span>
                        <div className="subtle">{trial.outcomeReason ?? "in progress"}</div>
                      </td>
                      <td className="trial-score">{formatNumber(trial.score, 4)}</td>
                      <td>
                        <div>{formatNumber(trial.accuracy, 4)}</div>
                        <div className="subtle">
                          best eval {formatDuration(trial.timeToBestEvalSec)}
                        </div>
                      </td>
                      <td>
                        <div>{trial.backend ?? "unknown backend"}</div>
                        {trial.timedOut ? <div className="trial-error">timed out</div> : null}
                        {trial.hadUnscoredWorkAtTimeout ? (
                          <div className="trial-error">unevaluated work before stop</div>
                        ) : null}
                        {trial.hasError ? <div className="trial-error">error recorded</div> : null}
                      </td>
                      <td>{trial.dispatchAttempts}</td>
                      <td>{formatDate(trial.startedAt ?? trial.createdAt)}</td>
                      <td>
                        <div>{formatDuration(trial.durationSec)}</div>
                        <div className="subtle">
                          since eval {formatDuration(trial.timeSinceLastEvalSec)}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {detail.nextCursor ? (
              <button type="button" className="load-more" onClick={loadMore} disabled={isPending}>
                {isPending ? "Loading…" : "Load more"}
              </button>
            ) : null}
          </section>
        </div>
      </section>
    </main>
  );
}
