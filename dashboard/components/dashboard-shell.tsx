"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useDeferredValue, useEffect, useEffectEvent, useState, useTransition, type KeyboardEvent } from "react";

import { HighlightedCode } from "@/components/highlighted-code";
import { SourceDiff } from "@/components/source-diff";
import { useTrackLiveUpdates } from "@/hooks/use-track-live-updates";
import type {
  PaginatedTrialsResponse,
  TrackDetailResponse,
  TrackListItem,
  TrialListItem,
  TrialStatusFilter,
} from "@/lib/types";

const STATUS_OPTIONS: TrialStatusFilter[] = ["all", "queued", "dispatching", "active", "finished", "error"];

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

function formatStatusLabel(value: string): string {
  return value === "active" ? "running" : value;
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

type PropertyEntry = {
  label: string;
  mono?: boolean;
  values: string[];
};
type ProgressSegment = {
  key: "queued" | "dispatching" | "active" | "finished" | "error";
  count: number;
  label: string;
};

function toPropertyLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatPropertyScalar(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : null;
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  return null;
}

function appendPropertyEntries(
  entries: PropertyEntry[],
  label: string,
  value: unknown,
  options?: {
    mono?: boolean;
  },
): void {
  if (value === null || value === undefined) {
    return;
  }

  if (Array.isArray(value)) {
    const values = value.flatMap((item) => {
      if (item && typeof item === "object") {
        return Object.entries(item).flatMap(([key, nestedValue]) => {
          const rendered = formatPropertyScalar(nestedValue);
          return rendered ? [`${toPropertyLabel(key)}: ${rendered}`] : [];
        });
      }

      const rendered = formatPropertyScalar(item);
      return rendered ? [rendered] : [];
    });

    if (values.length > 0) {
      entries.push({
        label,
        mono: options?.mono,
        values,
      });
    }
    return;
  }

  if (typeof value === "object") {
    for (const [key, nestedValue] of Object.entries(value as Record<string, unknown>)) {
      appendPropertyEntries(entries, `${label} ${toPropertyLabel(key)}`, nestedValue, options);
    }
    return;
  }

  const rendered = formatPropertyScalar(value);
  if (!rendered) {
    return;
  }

  entries.push({
    label,
    mono: options?.mono,
    values: [rendered],
  });
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

function getGenerationPayload(value: Record<string, unknown> | null): Record<string, unknown> | null {
  const raw = value?.generation;
  if (!raw || typeof raw !== "object") {
    return null;
  }
  return raw as Record<string, unknown>;
}

function getGenerationPrompt(value: Record<string, unknown> | null, field: "system_prompt" | "user_prompt"): string | null {
  const generation = getGenerationPayload(value);
  const prompt = generation?.[field];
  return typeof prompt === "string" && prompt.length > 0 ? prompt : null;
}

function normalizeSourceSnippet(content: string): string {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  return normalized ? `${normalized}\n` : "";
}

function extractMixedSourceSnapshot(messages: PromptMessage[]): { snippetCount: number; source: string } | null {
  const snippets: string[] = [];

  for (const message of messages) {
    const matches = message.content.matchAll(/```(?:python|py)?\n([\s\S]*?)```/g);
    for (const match of matches) {
      const snippet = normalizeSourceSnippet(match[1] ?? "");
      if (snippet) {
        snippets.push(snippet);
      }
    }
  }

  if (snippets.length === 0) {
    return null;
  }

  const source =
    snippets.length === 1
      ? snippets[0]
      : snippets.join("\n");

  return {
    snippetCount: snippets.length,
    source,
  };
}

function getGenerationProperties(value: Record<string, unknown> | null): PropertyEntry[] {
  if (!value) {
    return [];
  }

  const entries: PropertyEntry[] = [];

  appendPropertyEntries(entries, "Backend", value.backend);
  appendPropertyEntries(entries, "Model", value.model);
  appendPropertyEntries(entries, "Candidate Kind", value.candidate_kind);
  appendPropertyEntries(entries, "Generation Index", value.generation_index);
  appendPropertyEntries(entries, "Duplicate Retry Count", value.duplicate_retry_count);
  appendPropertyEntries(entries, "Provider Response ID", value.provider_response_id, { mono: true });
  appendPropertyEntries(entries, "Context Trials", value.context_trial_ids, { mono: true });
  appendPropertyEntries(entries, "Config", value.generation_config);
  appendPropertyEntries(entries, "Launcher", value.launcher);

  return entries;
}

function renderGenerationAssertionSummary(passed: boolean | null, failures: string[]) {
  if (passed === null) {
    return "Not recorded";
  }

  const status = passed ? (
    <span className="flag-chip flag-success">Passed</span>
  ) : (
    <span className="flag-chip flag-danger">Failed</span>
  );

  if (failures.length === 0) {
    return status;
  }

  return (
    <span className="timeline-assertion-summary">
      {status}
      <span className="timeline-detail">{failures.join("\n")}</span>
    </span>
  );
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

function renderModalRunLink(trial: TrialListItem, label = "Modal run") {
  if (!trial.modalRunUrl) {
    return null;
  }

  return (
    <a
      className="external-link-chip"
      href={trial.modalRunUrl}
      target="_blank"
      rel="noreferrer"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
      aria-label={`Open Modal run for ${trial.trialId}`}
      title={trial.modalRunId ?? undefined}
    >
      {label}
    </a>
  );
}

function getTrackLabel(track: TrackListItem): string {
  return track.name ?? track.trackId;
}

function getCompletedCount(track: TrackListItem): number {
  return track.finishedTrials + track.errorTrials;
}

function getProgressPercent(track: TrackListItem): number {
  if (track.totalTrials === 0) {
    return 0;
  }
  return (getCompletedCount(track) / track.totalTrials) * 100;
}

function getCoveragePercent(track: TrackListItem): number {
  if (track.totalTrials === 0) {
    return 0;
  }
  return (track.succeededTrials / track.totalTrials) * 100;
}

function getAttentionCount(track: TrackListItem): number {
  return Math.max(0, track.errorTrials + (track.finishedTrials - track.succeededTrials));
}

function buildProgressSegments(track: TrackListItem): ProgressSegment[] {
  return [
    { key: "queued", count: track.queuedTrials, label: "Queued" },
    { key: "dispatching", count: track.dispatchingTrials, label: "Dispatching" },
    { key: "active", count: track.activeTrials, label: "Running" },
    { key: "finished", count: track.finishedTrials, label: "Finished" },
    { key: "error", count: track.errorTrials, label: "Error" },
  ];
}

function getTrialTone(trial: TrialListItem): "success" | "warning" | "danger" | "neutral" {
  if (trial.status === "error") {
    return "danger";
  }
  if (trial.status !== "finished") {
    return "neutral";
  }
  if (trial.outcomeReason === "duplicate") {
    return "warning";
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
  if (trial.status === "error") {
    if (trial.outcomeReason === "generation_failed") {
      return "Generation failed before queueing a runnable candidate.";
    }
    if (trial.errorType) {
      return `Failed with ${trial.errorType}.`;
    }
    return "Ended in an error state.";
  }
  if (trial.outcomeReason === "generation_failed") {
    return "Generation failed before queueing a runnable candidate.";
  }
  if (trial.outcomeReason === "duplicate") {
    return "Generated a duplicate candidate hash and skipped dispatch.";
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
    trial.responseText,
    trial.generatedSource,
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
const SCORE_CHART_HEIGHT = 184;
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
    .filter((trial) => trial.status === "finished" || trial.status === "error")
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
    const score = trial.status === "finished" || trial.status === "error" ? trial.score : null;
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
  const selectedSystemPrompt = getGenerationPrompt(selectedTrial?.provenanceJson ?? null, "system_prompt");
  const selectedUserPrompt = getGenerationPrompt(selectedTrial?.provenanceJson ?? null, "user_prompt");
  const selectedMixedSource = extractMixedSourceSnapshot(selectedPromptMessages);
  const selectedCrashDetails = extractCrashDetails(selectedTrial?.errorJson ?? null);
  const selectedCrashSummary = summarizeCrashDetails(selectedCrashDetails);
  const selectedGeneratedSource = selectedTrial?.generatedSource ?? null;
  const selectedResponseText = selectedTrial?.responseText ?? null;
  const selectedAssertionFailures = selectedTrial?.generationAssertionFailures ?? [];
  const selectedGenerationProperties = getGenerationProperties(selectedTrial?.provenanceJson ?? null);
  const selectedIsGenerationFailure = selectedTrial?.outcomeReason === "generation_failed";
  const selectedGeneratedProgram = selectedIsGenerationFailure
    ? selectedGeneratedSource
    : selectedGeneratedSource ?? selectedTrial?.source ?? null;
  const selectedShowsDiagnosticSource = Boolean(
    selectedTrial && selectedGeneratedSource && selectedGeneratedSource !== selectedTrial.source,
  );
  const selectedCanCompareMixedSource = Boolean(
    !selectedIsGenerationFailure && selectedGeneratedProgram && selectedGeneratedProgram.length > 0,
  );
  const progressPercent = getProgressPercent(detail.track);
  const coveragePercent = getCoveragePercent(detail.track);
  const attentionCount = getAttentionCount(detail.track);
  const progressSegments = buildProgressSegments(detail.track);
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
                    <span>{getCompletedCount(track)}/{track.totalTrials} completed</span>
                    <span>{track.activeTrials} running</span>
                  </div>
                  <div className="track-card-meta">
                    <span>{track.errorTrials} errors</span>
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

          <div className="overview-grid">
            <article className="analysis-card wide-card overview-snapshot-card">
              <div className="overview-snapshot-layout">
                <div className="overview-snapshot-sidebar">
                  <div className="hero-metrics overview-snapshot-metrics">
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
                        {getCompletedCount(detail.track)} of {detail.track.totalTrials} trials have reached a terminal state.
                      </span>
                    </article>
                    <article className="metric-tile">
                      <span className="metric-label">Coverage</span>
                      <strong className="metric-value">{formatPercent(coveragePercent)}</strong>
                      <span className="metric-note">{detail.track.succeededTrials} runs produced scored metrics.</span>
                    </article>
                    <article className="metric-tile">
                      <span className="metric-label">Errors</span>
                      <strong className="metric-value">{detail.track.errorTrials}</strong>
                      <span className="metric-note">
                        {detail.track.errorTrials > 0 ? "Terminal failures need inspection." : "No terminal failures recorded."}
                      </span>
                    </article>
                    <article className="metric-tile">
                      <span className="metric-label">Attention</span>
                      <strong className="metric-value">{attentionCount}</strong>
                      <span className="metric-note">
                        {detail.track.activeTrials > 0
                          ? `${detail.track.activeTrials} trials are still running.`
                          : "No running trials right now."}
                      </span>
                    </article>
                  </div>

                  <section className="overview-progress-panel">
                        <div className="analysis-card-header">
                      <h3>Progress breakdown</h3>
                    </div>
                    <div className="progress-strip" aria-label="Track progress">
                      {progressSegments.map((segment) => (
                        <span
                          key={segment.key}
                          className={segment.key}
                          style={{
                            width: `${detail.track.totalTrials === 0 ? 0 : (segment.count / detail.track.totalTrials) * 100}%`,
                          }}
                          title={`${segment.label}: ${segment.count}`}
                        />
                      ))}
                    </div>
                  </section>
                </div>

                <div className="overview-snapshot-chart">
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
                </div>
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
                      {formatStatusLabel(option)}
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
                          <div className="trial-status-row">
                            <span className={`status-badge status-${trial.status}`}>
                              <span className={`status-indicator ${trial.status}`} />
                              {formatStatusLabel(trial.status)}
                            </span>
                            <span className="trial-status-duration">{formatDuration(trial.durationSec)}</span>
                          </div>
                        </td>
                        <td>{formatNumber(trial.score, 4)}</td>
                        <td>{formatNumber(trial.accuracy, 4)}</td>
                        <td>
                          <div className="trial-cell-primary">{trial.model ?? "unknown model"}</div>
                          <div className="trial-cell-secondary">{trial.backend ?? "unknown backend"}</div>
                        </td>
                        <td>
                          <div className="trial-notes">
                            {trial.outcomeReason ? <span className="flag-chip">{trial.outcomeReason}</span> : null}
                            {trial.errorType ? <span className="flag-chip flag-danger">{trial.errorType}</span> : null}
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
                <article className="analysis-card wide-card trial-summary-card">
                  <div className="trial-summary-header">
                    <div>
                      <div className="inspector-label">Selected trial</div>
                      <h2 className="trial-summary-title" title={selectedTrial.trialId}>
                        {compactIdentifier(selectedTrial.trialId, 14, 10)}
                      </h2>
                      <p className="trial-summary-copy">{getTrialNarrative(selectedTrial)}</p>
                    </div>
                    <div className="trial-summary-chip-row">
                      <span className={`status-badge status-${selectedTrial.status}`}>
                        <span className={`status-indicator ${selectedTrial.status}`} />
                        {formatStatusLabel(selectedTrial.status)}
                      </span>
                      {selectedTrialRank ? <span className="meta-chip">Rank #{selectedTrialRank} by score</span> : null}
                      {renderModalRunLink(selectedTrial)}
                    </div>
                  </div>

                  <div className="trial-summary-grid">
                    <div className="trial-summary-column">
                      <section className="trial-summary-panel">
                        <div className="trial-summary-section-label">Overview</div>
                        <div className="context-stack">
                          {[
                            {
                              label: "Trial ID",
                              mono: true,
                              value: selectedTrial.trialId,
                              title: selectedTrial.trialId,
                            },
                            { label: "Model", value: selectedTrial.model ?? "unknown model" },
                            { label: "Backend", value: selectedTrial.backend ?? "unknown backend" },
                            {
                              label: "Outcome",
                              value: selectedTrial.outcomeReason ? (
                                <span className="flag-chip">{selectedTrial.outcomeReason}</span>
                              ) : (
                                "Not reported"
                              ),
                            },
                            {
                              label: "Error Type",
                              value: selectedTrial.errorType ? (
                                <span className="flag-chip flag-danger">{selectedTrial.errorType}</span>
                              ) : (
                                "—"
                              ),
                            },
                            {
                              label: "Last Phase",
                              value: selectedTrial.lastPhase ? <span className="flag-chip">{selectedTrial.lastPhase}</span> : "—",
                            },
                          ].map((row) => (
                            <div className="context-row" key={row.label}>
                              <span>{row.label}</span>
                              <strong className={row.mono ? "trial-summary-mono" : undefined} title={row.title}>
                                {row.value}
                              </strong>
                            </div>
                          ))}
                        </div>
                        <div className="flag-row trial-summary-flags">
                          {selectedTrial.timedOut ? <span className="flag-chip flag-warning">timed out</span> : null}
                          {selectedTrial.hadUnscoredWorkAtTimeout ? (
                            <span className="flag-chip flag-warning">left work unscored</span>
                          ) : null}
                          {selectedTrial.hasError ? <span className="flag-chip flag-danger">error payload captured</span> : null}
                        </div>
                      </section>

                      <section className="trial-summary-panel">
                        <div className="trial-summary-section-label">Metrics</div>
                        <div className="context-stack">
                          {[
                            { label: "Score", value: formatNumber(selectedTrial.score, 4) },
                            { label: "Accuracy", value: formatNumber(selectedTrial.accuracy, 4) },
                            { label: "Time To Best Eval", value: formatDuration(selectedTrial.timeToBestEvalSec) },
                            { label: "Duration", value: formatDuration(selectedTrial.durationSec) },
                            { label: "Dispatch Attempts", value: selectedTrial.dispatchAttempts },
                            { label: "Idle Since Eval", value: formatDuration(selectedTrial.timeSinceLastEvalSec) },
                          ].map((row) => (
                            <div className="context-row" key={row.label}>
                              <span>{row.label}</span>
                              <strong>{row.value}</strong>
                            </div>
                          ))}
                        </div>
                      </section>
                    </div>

                    <div className="trial-summary-column">
                      <section className="trial-summary-panel">
                        <div className="trial-summary-section-label">Run timeline</div>
                        <div className="timeline-list">
                          {[
                            { label: "Queued", value: formatDate(selectedTrial.createdAt) },
                            { label: "Started", value: formatDate(selectedTrial.startedAt) },
                            { label: "Finished", value: formatDate(selectedTrial.finishedAt) },
                            {
                              label: "Crash detail",
                              title: selectedCrashDetails ?? undefined,
                              value: selectedCrashSummary,
                            },
                            {
                              label: "Generation assertions",
                              value: renderGenerationAssertionSummary(
                                selectedTrial.generationAssertionsPassed,
                                selectedAssertionFailures,
                              ),
                            },
                          ].map((row) => (
                            <div className="timeline-row" key={row.label}>
                              <span>{row.label}</span>
                              <strong title={row.title}>{row.value}</strong>
                            </div>
                          ))}
                        </div>
                      </section>

                      <section className="trial-summary-panel">
                        <div className="trial-summary-section-label">Generation provenance</div>
                        {selectedGenerationProperties.length > 0 ? (
                          <div className="context-stack">
                            {selectedGenerationProperties.map((entry) => (
                              <div className="context-row" key={entry.label}>
                                <span>{entry.label}</span>
                                <strong className={entry.mono ? "trial-summary-mono" : undefined}>
                                  {entry.values.length === 1 ? (
                                    entry.values[0]
                                  ) : (
                                    <span className="property-chip-list">
                                      {entry.values.map((item) => (
                                        <span
                                          key={`${entry.label}:${item}`}
                                          className={`meta-chip ${entry.mono ? "meta-chip-mono" : ""}`.trim()}
                                        >
                                          {item}
                                        </span>
                                      ))}
                                    </span>
                                  )}
                                </strong>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="section-copy">No provenance payload recorded.</p>
                        )}
                      </section>
                    </div>
                  </div>
                </article>

                <div className="inspector-grid">
                  {selectedTrial.hasError ? (
                    <article className="analysis-card wide-card">
                      <div className="analysis-card-header">
                        <h3>Error payload</h3>
                      </div>
                      <HighlightedCode code={formatJsonBlock(selectedTrial.errorJson)} language="json" wrap />
                    </article>
                  ) : null}

                  {selectedCanCompareMixedSource ? (
                    <article className="analysis-card wide-card">
                      <div className="analysis-card-header">
                        <h3>Mixed vs generated diff</h3>
                        <span>
                          {selectedMixedSource
                            ? `${selectedMixedSource.snippetCount} prompt source${selectedMixedSource.snippetCount === 1 ? "" : "s"}`
                            : "No prompt source snippets"}
                        </span>
                      </div>
                      {selectedMixedSource ? (
                        <SourceDiff before={selectedMixedSource.source} after={selectedGeneratedProgram ?? ""} />
                      ) : (
                        <p className="section-copy">No prompt-embedded source snippets were recorded for this trial.</p>
                      )}
                    </article>
                  ) : null}

                  <article className="analysis-card wide-card">
                    <div className="analysis-card-header">
                      <h3>System prompt</h3>
                    </div>
                    <HighlightedCode
                      code={selectedSystemPrompt ?? "No system prompt recorded."}
                      language={detectPromptLanguage(selectedSystemPrompt ?? "")}
                      wrap
                    />
                  </article>

                  <article className="analysis-card wide-card">
                    <div className="analysis-card-header">
                      <h3>User prompt</h3>
                    </div>
                    <HighlightedCode
                      code={selectedUserPrompt ?? "No user prompt recorded."}
                      language={detectPromptLanguage(selectedUserPrompt ?? "")}
                      wrap
                    />
                  </article>

                  <article className="analysis-card wide-card">
                    <div className="analysis-card-header">
                      <h3>Raw LLM response</h3>
                    </div>
                    <HighlightedCode
                      code={selectedResponseText ?? "No response received."}
                      language={detectPromptLanguage(selectedResponseText ?? "")}
                      wrap
                    />
                  </article>

                  <article className="analysis-card wide-card">
                    <div className="analysis-card-header">
                      <h3>{selectedIsGenerationFailure ? "Generation attempt" : "Generated program"}</h3>
                    </div>
                    {selectedShowsDiagnosticSource ? (
                      <p className="section-copy">
                        This trial never became runnable. The stored row source is diagnostic-only; this is the
                        attempted candidate captured from generation.
                      </p>
                    ) : null}
                    <HighlightedCode
                      code={
                        selectedGeneratedProgram ??
                        (selectedIsGenerationFailure ? "No generation attempt recorded." : "No generated program recorded.")
                      }
                      language="python"
                      wrap
                    />
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
