"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useDeferredValue, useEffect, useEffectEvent, useState, useTransition } from "react";

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

type ActivePane = "tracks" | "trials" | "detail";
type PaneState = Record<ActivePane, boolean>;
const PANE_ORDER: ActivePane[] = ["tracks", "trials", "detail"];
const DEFAULT_OPEN_PANES: PaneState = {
  tracks: true,
  trials: true,
  detail: true,
};

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

function formatJsonBlock(value: Record<string, unknown> | null): string {
  if (!value) {
    return "No error payload recorded.";
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
  for (const key of ["backend", "model", "generation_index", "provider_response_id", "generation_config", "context_trial_ids"]) {
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

type DashboardShellProps = {
  initialDetail: TrackDetailResponse;
  initialTracks: TrackListItem[];
  initialSelectedTrialId: string | null;
  selectedTrackId: string;
};

export function DashboardShell({
  initialDetail,
  initialTracks,
  initialSelectedTrialId,
  selectedTrackId,
}: DashboardShellProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const routeTrialId = searchParams.get("trial");

  const [tracks, setTracks] = useState(initialTracks);
  const [detail, setDetail] = useState(initialDetail);
  const [status, setStatus] = useState<TrialStatusFilter>("all");
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(initialSelectedTrialId);
  const [urlTrialId, setUrlTrialId] = useState<string | null>(initialSelectedTrialId);
  const [openPanes, setOpenPanes] = useState<PaneState>(DEFAULT_OPEN_PANES);
  const [activePane, setActivePane] = useState<ActivePane>("trials");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const deferredTrials = useDeferredValue(detail.trials);
  const selectedTrial = detail.trials.find((trial) => trial.trialId === selectedTrialId) ?? null;
  const visiblePanes = PANE_ORDER.filter((pane) => openPanes[pane]);
  const shellLayoutClass = `shell-layout-${visiblePanes.join("-")}`;
  const canClosePane = visiblePanes.length > 1;

  useEffect(() => {
    setTracks(initialTracks);
    setDetail(initialDetail);
    setStatus("all");
    setSelectedTrialId(initialSelectedTrialId);
    setUrlTrialId(initialSelectedTrialId);
    setOpenPanes(DEFAULT_OPEN_PANES);
    setActivePane("trials");
    setError(null);
  }, [initialDetail, initialSelectedTrialId, initialTracks, selectedTrackId]);

  useEffect(() => {
    setUrlTrialId(routeTrialId);
  }, [routeTrialId]);

  const updateTrialUrl = useEffectEvent((nextTrialId: string | null) => {
    const params = new URLSearchParams(searchParams.toString());
    if (nextTrialId) {
      params.set("trial", nextTrialId);
    } else {
      params.delete("trial");
    }

    const nextQuery = params.toString();
    const nextUrl = nextQuery ? `${pathname}?${nextQuery}` : pathname;
    const currentQuery = searchParams.toString();
    const currentUrl = currentQuery ? `${pathname}?${currentQuery}` : pathname;

    if (nextUrl !== currentUrl) {
      router.replace(nextUrl, { scroll: false });
    }
  });

  const syncSelectedTrial = useEffectEvent((nextTrialId: string | null) => {
    setSelectedTrialId(nextTrialId);
    setUrlTrialId(nextTrialId);
    updateTrialUrl(nextTrialId);
  });

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

  useEffect(() => {
    if (detail.trials.length === 0) {
      if (selectedTrialId !== null) {
        syncSelectedTrial(null);
      }
      return;
    }

    if (urlTrialId) {
      const routeTrial = detail.trials.find((trial) => trial.trialId === urlTrialId);
      if (routeTrial) {
        if (urlTrialId !== selectedTrialId) {
          setSelectedTrialId(urlTrialId);
        }
        return;
      }

      syncSelectedTrial(detail.trials[0].trialId);
      return;
    }

    if (selectedTrialId && detail.trials.some((trial) => trial.trialId === selectedTrialId)) {
      return;
    }

    syncSelectedTrial(detail.trials[0].trialId);
  }, [detail.trials, selectedTrialId, syncSelectedTrial, urlTrialId]);

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

  const handleTrialSelect = (trialId: string) => {
    setOpenPanes((current) => ({ ...current, detail: true }));
    setActivePane("detail");
    syncSelectedTrial(trialId);
  };

  const openPane = (pane: ActivePane) => {
    setOpenPanes((current) => ({ ...current, [pane]: true }));
    setActivePane(pane);
  };

  const closePane = (pane: ActivePane) => {
    if (!canClosePane) {
      return;
    }

    setOpenPanes((current) => {
      if (!current[pane]) {
        return current;
      }

      const next = {
        ...current,
        [pane]: false,
      };

      const nextVisible = PANE_ORDER.filter((item) => next[item]);
      if (!nextVisible.includes(activePane)) {
        const closedIndex = PANE_ORDER.indexOf(pane);
        const fallback =
          PANE_ORDER.slice(closedIndex + 1).find((item) => next[item]) ??
          PANE_ORDER.slice(0, closedIndex).reverse().find((item) => next[item]) ??
          nextVisible[0];
        if (fallback) {
          setActivePane(fallback);
        }
      }

      return next;
    });
  };

  return (
    <main className={`shell ${shellLayoutClass}`}>
      {visiblePanes.length < PANE_ORDER.length ? (
        <div className="closed-pane-bar">
          {PANE_ORDER.filter((pane) => !openPanes[pane]).map((pane) => (
            <button
              key={pane}
              type="button"
              className="closed-pane-toggle"
              onClick={() => openPane(pane)}
            >
              {`Open ${pane} panel`}
            </button>
          ))}
        </div>
      ) : null}

      {openPanes.tracks ? (
      <aside
        className={`workspace-pane tracks-pane ${activePane === "tracks" ? "mobile-visible" : ""}`}
        onFocusCapture={() => setActivePane("tracks")}
        onPointerDown={() => setActivePane("tracks")}
      >
        <div className="pane-header">
          <div className="pane-header-top">
            <div>
              <div className="eyebrow">Tracks</div>
              <h1 className="pane-title">Experiment tracks</h1>
            </div>
            <button
              type="button"
              className="pane-close"
              onClick={() => closePane("tracks")}
              disabled={!canClosePane}
            >
              Close tracks panel
            </button>
          </div>
          <p className="subtle pane-copy">{tracks.length} active lanes available.</p>
        </div>

        <div className="track-rail">
          {tracks.map((track) => (
            <Link
              key={track.trackId}
              href={`/tracks/${track.trackId}`}
              className={`track-rail-item ${track.trackId === selectedTrackId ? "active" : ""}`}
            >
              <div className="track-rail-top">
                <h2 className="track-name">{track.name ?? track.trackId}</h2>
                <span className="track-rail-score">{formatNumber(track.bestScore, 4)}</span>
              </div>
              <div className="track-rail-meta">
                <span>{track.datasetId}</span>
                <span>{track.totalTrials} trials</span>
              </div>
              <div className="track-rail-meta">
                <span>Updated {formatDate(track.lastActivityAt)}</span>
              </div>
            </Link>
          ))}
        </div>
      </aside>
      ) : null}

      {openPanes.trials ? (
      <section
        className={`workspace-pane trials-pane ${activePane === "trials" ? "mobile-visible" : ""}`}
        onFocusCapture={() => setActivePane("trials")}
        onPointerDown={() => setActivePane("trials")}
      >
        <div className="pane-header pane-header-sticky">
          <div className="pane-header-top">
            <div className="pane-header-stack">
              <button type="button" className="pane-back" onClick={() => setActivePane("tracks")}>
                Tracks
              </button>
              <div className="eyebrow">Trials</div>
              <h2 className="pane-title">{detail.track.name ?? detail.track.trackId}</h2>
            </div>
            <button
              type="button"
              className="pane-close"
              onClick={() => closePane("trials")}
              disabled={!canClosePane}
            >
              Close trials panel
            </button>
          </div>
          <div className="detail-meta">
            <span>{detail.track.trackId}</span>
            <span>{detail.track.datasetId}</span>
            <span>Created {formatDate(detail.track.createdAt)}</span>
          </div>
          <div className="track-summary-grid">
            <div className="metric">
              <span className="metric-label">Best Score</span>
              <span className="metric-value">{formatNumber(detail.track.bestScore, 4)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Trials</span>
              <span className="metric-value">{detail.track.totalTrials}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Scored</span>
              <span className="metric-value">{detail.track.succeededTrials}</span>
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
          <div className="toolbar">
            <div className="live-state">
              Live transport: <strong>{liveMode}</strong>
              {isPending ? " · refreshing" : ""}
            </div>
            <div className="live-state">Newest first</div>
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
          {error ? <div className="error-banner">{error}</div> : null}
        </div>

        <div className="trial-list" aria-label="Trials">
          {deferredTrials.length === 0 ? (
            <section className="panel empty-panel">
              <div className="eyebrow">No trials</div>
              <h3>No trials match the current filter.</h3>
              <p className="subtle">Change the status filter or wait for new runs to appear.</p>
            </section>
          ) : (
            deferredTrials.map((trial: TrialListItem) => (
              <button
                key={trial.trialId}
                type="button"
                aria-label={`Select trial ${trial.trialId}`}
                aria-pressed={selectedTrial?.trialId === trial.trialId}
                className={`trial-row ${selectedTrial?.trialId === trial.trialId ? "active" : ""}`}
                onClick={() => handleTrialSelect(trial.trialId)}
              >
                <div className="trial-row-top">
                  <div>
                    <code>{trial.trialId}</code>
                    <div className="subtle">{trial.model ?? "unknown model"}</div>
                  </div>
                  <span className="trial-status">
                    <span className={`status-dot ${trial.status}`} />
                    {trial.status}
                  </span>
                </div>
                <div className="trial-row-grid">
                  <div>
                    <span className="trial-kicker">Score</span>
                    <strong>{formatNumber(trial.score, 4)}</strong>
                  </div>
                  <div>
                    <span className="trial-kicker">Accuracy</span>
                    <strong>{formatNumber(trial.accuracy, 4)}</strong>
                  </div>
                  <div>
                    <span className="trial-kicker">Backend</span>
                    <strong>{trial.backend ?? "unknown backend"}</strong>
                  </div>
                  <div>
                    <span className="trial-kicker">Duration</span>
                    <strong>{formatDuration(trial.durationSec)}</strong>
                  </div>
                </div>
                <div className="trial-flags">
                  <span>{trial.outcomeReason ?? "in progress"}</span>
                  <span>Dispatches {trial.dispatchAttempts}</span>
                  <span>Started {formatDate(trial.startedAt ?? trial.createdAt)}</span>
                  {trial.timedOut ? <span className="trial-error">timed out</span> : null}
                  {trial.hadUnscoredWorkAtTimeout ? <span className="trial-error">unevaluated work</span> : null}
                  {trial.hasError ? <span className="trial-error">error recorded</span> : null}
                </div>
              </button>
            ))
          )}
        </div>

        {detail.nextCursor ? (
          <button type="button" className="load-more" onClick={loadMore} disabled={isPending}>
            {isPending ? "Loading…" : "Load more"}
          </button>
        ) : null}
      </section>
      ) : null}

      {openPanes.detail ? (
      <section
        className={`workspace-pane detail-pane ${activePane === "detail" ? "mobile-visible" : ""}`}
        onFocusCapture={() => setActivePane("detail")}
        onPointerDown={() => setActivePane("detail")}
      >
        <div className="pane-header pane-header-sticky">
          <div className="pane-header-top">
            <div className="pane-header-stack">
              <button type="button" className="pane-back" onClick={() => setActivePane("trials")}>
                Trials
              </button>
              <div className="eyebrow">Selected Trial</div>
            </div>
            <button
              type="button"
              className="pane-close"
              onClick={() => closePane("detail")}
              disabled={!canClosePane}
            >
              Close detail panel
            </button>
          </div>
          {selectedTrial ? (
            <>
              <div className="detail-title-row">
                <h2 className="pane-title">{selectedTrial.trialId}</h2>
                <span className="trial-status">
                  <span className={`status-dot ${selectedTrial.status}`} />
                  {selectedTrial.status}
                </span>
              </div>
              <div className="detail-meta">
                <span>{selectedTrial.model ?? "unknown model"}</span>
                <span>{selectedTrial.backend ?? "unknown backend"}</span>
                <span>{selectedTrial.outcomeReason ?? "in progress"}</span>
              </div>
              <div className="detail-summary-grid">
                <div className="metric">
                  <span className="metric-label">Score</span>
                  <span className="metric-value">{formatNumber(selectedTrial.score, 4)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Accuracy</span>
                  <span className="metric-value">{formatNumber(selectedTrial.accuracy, 4)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Duration</span>
                  <span className="metric-value">{formatDuration(selectedTrial.durationSec)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Since Eval</span>
                  <span className="metric-value">{formatDuration(selectedTrial.timeSinceLastEvalSec)}</span>
                </div>
              </div>
            </>
          ) : (
            <>
              <h2 className="pane-title">No selected trial</h2>
              <p className="subtle pane-copy">No trials match the current filter for this track.</p>
            </>
          )}
        </div>

        {selectedTrial ? (
          <div className="trial-detail-grid">
            <section className="trial-detail-card">
              <h3>Crash reason</h3>
              <HighlightedCode
                code={extractCrashDetails(selectedTrial.errorJson) ?? "No crash stderr recorded."}
                language="markdown"
                wrap
              />
            </section>
            <section className="trial-detail-card">
              <h3>Error payload</h3>
              <HighlightedCode code={formatJsonBlock(selectedTrial.errorJson)} language="json" wrap />
            </section>
            <section className="trial-detail-card">
              <h3>Generation metadata</h3>
              <HighlightedCode
                code={formatGenerationProperties(selectedTrial.provenanceJson)}
                language="json"
                wrap
              />
            </section>
            {asPromptMessages(selectedTrial.provenanceJson).map((message, index) => (
              <section className="trial-detail-card" key={`${selectedTrial.trialId}-prompt-${index}`}>
                <h3>{`Prompt ${index + 1} · ${message.role}`}</h3>
                <HighlightedCode
                  code={message.content}
                  language={detectPromptLanguage(message.content)}
                  wrap
                />
              </section>
            ))}
            <section className="trial-detail-card trial-detail-card-wide">
              <h3>Trial source</h3>
              <HighlightedCode code={selectedTrial.source} language="python" />
            </section>
          </div>
        ) : (
          <section className="panel empty-panel detail-empty-panel">
            <div className="eyebrow">Trial detail</div>
            <h3>Detail pane cleared</h3>
            <p className="subtle">Select a visible trial or adjust the filter to restore detail content.</p>
          </section>
        )}
      </section>
      ) : null}
    </main>
  );
}
