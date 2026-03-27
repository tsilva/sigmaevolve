"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useState,
  useTransition,
  type KeyboardEvent,
  type ReactNode,
} from "react";

import { HighlightedCode } from "@/components/highlighted-code";
import { MarkdownContent } from "@/components/markdown-content";
import { useTrackLiveUpdates } from "@/hooks/use-track-live-updates";
import { buildSourceDiff } from "@/lib/source-diff";
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

function formatBestEpoch(bestEvalEpoch: number | null, epochsCompleted: number | null): string {
  if (bestEvalEpoch === null || epochsCompleted === null) {
    return "—";
  }
  return `${bestEvalEpoch}/${epochsCompleted}`;
}

function summarizeTaskDescription(value: string | null, maxLength = 160): string | null {
  if (!value) {
    return null;
  }

  const collapsed = value.replace(/\s+/g, " ").trim();
  if (collapsed.length === 0) {
    return null;
  }

  if (collapsed.length <= maxLength) {
    return collapsed;
  }

  return `${collapsed.slice(0, maxLength - 1).trimEnd()}…`;
}

function formatJsonBlock(value: Record<string, unknown> | null): string {
  if (!value) {
    return "No payload recorded.";
  }
  return JSON.stringify(value, null, 2);
}

type PropertyValue = {
  href?: string;
  text: string;
};

type PropertyEntry = {
  label: string;
  mono?: boolean;
  values: PropertyValue[];
};
type PropertyGroup = {
  label: string;
  entries: PropertyEntry[];
};
type TimelineRow = {
  label: string;
  value: ReactNode;
  title?: string;
};
type ProgressSegment = {
  key: "queued" | "dispatching" | "active" | "finished" | "error";
  count: number;
  label: string;
};
type CollapsibleSectionProps = {
  children: ReactNode;
  collapsible?: boolean;
  expanded?: boolean;
  id: string;
  onToggle?: () => void;
  summary?: ReactNode;
  title: string;
  titleClassName?: string;
  titleTag?: "div" | "h3";
  toggleClassName?: string;
};

const HIDDEN_WANDB_KEYS = new Set(["project", "entity", "run_id", "run_name", "run_url"]);

type TrackProgressBarProps = {
  className?: string;
  track: TrackListItem;
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
    linkBuilder?: (value: string) => string | undefined;
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
          return rendered
            ? [
                {
                  text: `${toPropertyLabel(key)}: ${rendered}`,
                },
              ]
            : [];
        });
      }

      const rendered = formatPropertyScalar(item);
      return rendered
        ? [
            {
              href: options?.linkBuilder?.(rendered),
              text: rendered,
            },
          ]
        : [];
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
    values: [
      {
        href: options?.linkBuilder?.(rendered),
        text: rendered,
      },
    ],
  });
}

function isExternalUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function renderPropertyValue(value: PropertyValue, className?: string) {
  if (!value.href) {
    if (isExternalUrl(value.text)) {
      return (
        <a className={className ?? "external-link-chip"} href={value.text} target="_blank" rel="noreferrer">
          {value.text}
        </a>
      );
    }
    return value.text;
  }

  if (isExternalUrl(value.href)) {
    return (
      <a className={className ?? "external-link-chip"} href={value.href} target="_blank" rel="noreferrer">
        {value.text}
      </a>
    );
  }

  return (
    <Link className={className} href={value.href}>
      {value.text}
    </Link>
  );
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

type MixedSourceSnapshot = {
  snippetCount: number;
  source: string;
  sourceKind: "current_program" | "mixed";
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
  if (typeof prompt === "string" && prompt.length > 0) {
    return prompt;
  }

  const requestMessages = asPromptMessages(value);
  if (field === "system_prompt") {
    return requestMessages[0]?.content ?? null;
  }
  return requestMessages[1]?.content ?? null;
}

function normalizeSourceSnippet(content: string): string {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  return normalized ? `${normalized}\n` : "";
}

function extractFencedSourceSnippets(content: string): string[] {
  const snippets: string[] = [];
  const matches = content.matchAll(/```(?:python|py)?\n([\s\S]*?)```/g);
  for (const match of matches) {
    const snippet = normalizeSourceSnippet(match[1] ?? "");
    if (snippet) {
      snippets.push(snippet);
    }
  }
  return snippets;
}

function extractCurrentProgramSnippet(content: string): string | null {
  const sectionMatch = content.match(/CURRENT PROGRAM:\n([\s\S]*?)(?:\nREPLACEMENTS:|$)/);
  if (!sectionMatch) {
    return null;
  }

  const snippets = extractFencedSourceSnippets(sectionMatch[1] ?? "");
  return snippets[0] ?? null;
}

function extractMixedSourceSnapshot(messages: PromptMessage[]): MixedSourceSnapshot | null {
  const snippets: string[] = [];
  let currentProgramSnippet: string | null = null;

  for (const message of messages) {
    if (currentProgramSnippet === null) {
      currentProgramSnippet = extractCurrentProgramSnippet(message.content);
    }

    for (const snippet of extractFencedSourceSnippets(message.content)) {
      snippets.push(snippet);
    }
  }

  if (currentProgramSnippet) {
    return {
      snippetCount: snippets.length,
      source: currentProgramSnippet,
      sourceKind: "current_program",
    };
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
    sourceKind: "mixed",
  };
}

function formatMixedSourceSummary(value: MixedSourceSnapshot | null): string {
  if (!value) {
    return "No prompt source snippets";
  }

  const countLabel = `${value.snippetCount} prompt source${value.snippetCount === 1 ? "" : "s"}`;
  if (value.sourceKind === "current_program" && value.snippetCount > 1) {
    return `${countLabel} • diffing CURRENT PROGRAM`;
  }

  return countLabel;
}

function getGenerationPropertyGroups(trackId: string, value: Record<string, unknown> | null): PropertyGroup[] {
  if (!value) {
    return [];
  }

  const {
    backend,
    model,
    candidate_kind,
    context_trial_ids,
    generation_config,
    launcher,
    wandb_project: _wandbProject,
    wandb_entity: _wandbEntity,
    wandb_run_id: _wandbRunId,
    wandb_run_name: _wandbRunName,
    request_messages: _requestMessages,
    generation: _generation,
    ...remaining
  } = value;

  const contextTrialIds = Array.isArray(context_trial_ids)
    ? context_trial_ids.flatMap((entry) => {
        const rendered = formatPropertyScalar(entry);
        return rendered ? [rendered] : [];
      })
    : [];

  const modelEntries: PropertyEntry[] = [];
  appendPropertyEntries(modelEntries, "Backend", backend);
  appendPropertyEntries(modelEntries, "Model", model);
  appendPropertyEntries(modelEntries, "Candidate Kind", candidate_kind);
  appendPropertyEntries(modelEntries, "Current Program Trial", contextTrialIds[0], {
    linkBuilder: (trialId) => `/tracks/${trackId}/trials/${encodeURIComponent(trialId)}`,
    mono: true,
  });
  appendPropertyEntries(modelEntries, "Reference Program Trials", contextTrialIds.slice(1), {
    linkBuilder: (trialId) => `/tracks/${trackId}/trials/${encodeURIComponent(trialId)}`,
    mono: true,
  });
  appendPropertyEntries(modelEntries, "Config", generation_config);

  const launcherEntries: PropertyEntry[] = [];
  if (launcher && typeof launcher === "object") {
    const {
      run_id: _runId,
      run_url: _runUrl,
      ...launcherWithoutRunMetadata
    } = launcher as Record<string, unknown>;
    appendPropertyEntries(launcherEntries, "Launcher", launcherWithoutRunMetadata);
  } else {
    appendPropertyEntries(launcherEntries, "Launcher", launcher);
  }

  const otherEntries: PropertyEntry[] = [];
  for (const [key, nestedValue] of Object.entries(remaining)) {
    if (key === "wandb" && nestedValue && typeof nestedValue === "object") {
      const filteredWandbEntries = Object.fromEntries(
        Object.entries(nestedValue as Record<string, unknown>).filter(([nestedKey]) => !HIDDEN_WANDB_KEYS.has(nestedKey)),
      );
      appendPropertyEntries(otherEntries, "Wandb", filteredWandbEntries);
      continue;
    }
    appendPropertyEntries(otherEntries, toPropertyLabel(key), nestedValue);
  }

  return [
    { label: "Model", entries: modelEntries },
    { label: "Launcher", entries: launcherEntries },
    { label: "Other", entries: otherEntries },
  ].filter((group) => group.entries.length > 0);
}

function renderGenerationAssertionSummary(passed: boolean | null, failures: string[]) {
  if (passed === null && failures.length === 0) {
    return null;
  }

  if (passed === null) {
    return (
      <span className="timeline-assertion-summary">
        <span className="flag-chip flag-danger">Failed</span>
        <span className="timeline-detail">{failures.join("\n")}</span>
      </span>
    );
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

function summarizeCrashDetails(value: string | null): string | null {
  if (!value) {
    return null;
  }

  const firstLine = value
    .split("\n")
    .map((line) => line.trim())
    .find(Boolean);

  if (!firstLine) {
    return null;
  }

  return firstLine.length > 160 ? `${firstLine.slice(0, 157)}...` : firstLine;
}

function getWandbRunUrl(provenance: Record<string, unknown> | null): string | null {
  const wandb = provenance?.wandb;
  if (!wandb || typeof wandb !== "object") {
    return null;
  }

  const runUrl = (wandb as Record<string, unknown>).run_url;
  return typeof runUrl === "string" && runUrl.trim().length > 0 ? runUrl : null;
}

function renderRunBadge({
  href,
  label,
  ariaLabel,
  title,
}: {
  href: string;
  label: string;
  ariaLabel: string;
  title?: string;
}) {
  return (
    <a
      className="external-link-chip external-link-badge"
      href={href}
      target="_blank"
      rel="noreferrer"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
      aria-label={ariaLabel}
      title={title}
    >
      {label}
    </a>
  );
}

function renderLauncherRunBadge(trial: TrialListItem) {
  if (!trial.modalRunUrl) {
    return null;
  }

  return renderRunBadge({
    href: trial.modalRunUrl,
    label: "Launcher",
    ariaLabel: `Open launcher run for ${trial.trialId}`,
    title: trial.modalRunId ?? undefined,
  });
}

function renderWandbRunBadge(trial: TrialListItem) {
  const wandbRunUrl = getWandbRunUrl(trial.provenanceJson);
  if (!wandbRunUrl) {
    return null;
  }

  return renderRunBadge({
    href: wandbRunUrl,
    label: "W&B",
    ariaLabel: `Open Weights & Biases run for ${trial.trialId}`,
  });
}

function getTrackLabel(track: TrackListItem): string {
  return track.trackId;
}

function CollapsibleSection({
  children,
  collapsible = true,
  expanded,
  id,
  onToggle,
  summary,
  title,
  titleClassName,
  titleTag = "div",
  toggleClassName,
}: CollapsibleSectionProps) {
  const TitleTag = titleTag;
  const contentId = `${id}-content`;
  const isExpanded = collapsible ? expanded === true : true;
  const headerClassName = toggleClassName
    ? `collapsible-section-toggle ${toggleClassName}`
    : "collapsible-section-toggle";
  const headerContent = (
    <span className="collapsible-section-copy">
      <TitleTag className={titleClassName ?? "collapsible-section-title"}>{title}</TitleTag>
      {summary ? <span className="collapsible-section-summary">{summary}</span> : null}
    </span>
  );

  if (!collapsible) {
    return (
      <>
        <div className={`${headerClassName} collapsible-section-header`}>{headerContent}</div>
        <div id={contentId} className="collapsible-section-body">
          {children}
        </div>
      </>
    );
  }

  return (
    <>
      <button
        type="button"
        className={headerClassName}
        onClick={onToggle}
        aria-expanded={isExpanded}
        aria-controls={contentId}
      >
        {headerContent}
        <span
          className={`collapsible-section-indicator ${isExpanded ? "expanded" : ""}`}
          aria-hidden="true"
        >
          ›
        </span>
      </button>
      {isExpanded ? (
        <div id={contentId} className="collapsible-section-body">
          {children}
        </div>
      ) : null}
    </>
  );
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

function isBestTrial(track: TrackListItem, trialId: string): boolean {
  return track.bestTrialId === trialId;
}

function TrackProgressBar({ className, track }: TrackProgressBarProps) {
  const progressSegments = buildProgressSegments(track);

  return (
    <div className={className ? `${className} progress-strip` : "progress-strip"} aria-label="Track progress">
      {progressSegments.map((segment) => {
        const tooltip = `${segment.label}: ${segment.count}`;
        return (
          <span
            key={segment.key}
            className={`progress-segment ${segment.key}`}
            style={{
              width: `${track.totalTrials === 0 ? 0 : (segment.count / track.totalTrials) * 100}%`,
            }}
            title={tooltip}
            data-tooltip={tooltip}
            aria-label={tooltip}
            tabIndex={0}
          />
        );
      })}
    </div>
  );
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
    trial.taskDescription,
    trial.responseText,
    trial.reasoningText,
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
  createdAt: string;
  isBest: boolean;
  isClippedLow: boolean;
  lastPhase: string | null;
  model: string | null;
  outcomeReason: string | null;
  score: number | null;
  status: TrialListItem["status"];
  tone: ReturnType<typeof getTrialTone>;
  trialId: string;
  x: number;
  y: number;
};

const DEFAULT_EXPANDED_SECTION_IDS = ["trial-task-description", "trial-generated-program"];

const SCORE_CHART_WIDTH = 960;
const SCORE_CHART_HEIGHT = 124;
const SCORE_CHART_PADDING = {
  top: 14,
  right: 14,
  bottom: 24,
  left: 40,
};
const MIN_SCORE_CHART_RANGE = 0.02;
const MIN_ZOOMED_SCORE_CHART_RANGE = 0.003;
const MIN_ZOOMED_SCORE_PADDING = 0.0015;

function getScoreTickDigits(range: number): number {
  if (range < 0.002) {
    return 5;
  }
  if (range < 0.02) {
    return 4;
  }
  return 3;
}

function buildScoreDomain(scoredValues: number[]): {
  clippedLowThreshold: number | null;
  scaleMode: "full" | "zoomed";
  tickDigits: number;
  yMax: number;
  yMin: number;
} {
  if (scoredValues.length === 0) {
    return {
      clippedLowThreshold: null,
      scaleMode: "full",
      tickDigits: 3,
      yMax: 1,
      yMin: 0,
    };
  }

  const rawMin = Math.min(...scoredValues);
  const rawMax = Math.max(...scoredValues);
  const positiveValues = scoredValues.filter((score) => score > 0);
  let focusValues = scoredValues;
  let scaleMode: "full" | "zoomed" = "full";

  if (positiveValues.length >= 2) {
    const positiveMin = Math.min(...positiveValues);
    const positiveMax = Math.max(...positiveValues);
    const positiveSpread = positiveMax - positiveMin;
    const lowerGap = positiveMin - rawMin;
    const hasTightUpperCluster = positiveSpread > 0 && positiveSpread <= 0.02 && positiveMax >= 0.9;
    const hasLowOutlierCompression =
      lowerGap >= Math.max(positiveSpread * 4, 0.05) && rawMax - rawMin >= Math.max(positiveSpread * 6, 0.08);

    if (hasTightUpperCluster && hasLowOutlierCompression) {
      focusValues = positiveValues;
      scaleMode = "zoomed";
    }
  }

  const focusMin = Math.min(...focusValues);
  const focusMax = Math.max(...focusValues);
  const focusSpread = focusMax - focusMin;
  const padding = Math.max(
    focusSpread * 0.18,
    scaleMode === "zoomed" ? MIN_ZOOMED_SCORE_PADDING : MIN_SCORE_CHART_RANGE * 0.18,
  );
  const yMin = Math.max(0, focusMin - padding);
  const yMax = Math.min(1, focusMax + padding);
  const range = Math.max(scaleMode === "zoomed" ? MIN_ZOOMED_SCORE_CHART_RANGE : MIN_SCORE_CHART_RANGE, yMax - yMin);

  return {
    clippedLowThreshold: scaleMode === "zoomed" ? yMin : null,
    scaleMode,
    tickDigits: getScoreTickDigits(range),
    yMax,
    yMin,
  };
}

function buildTrialsUrl(trackId: string, status: TrialStatusFilter, cursor?: string | null, limit = 50): string {
  const params = new URLSearchParams();
  params.set("status", status);
  params.set("limit", String(limit));
  if (cursor) {
    params.set("cursor", cursor);
  }
  return `/api/tracks/${trackId}/trials?${params.toString()}`;
}

function buildScoreChart(trials: TrialListItem[], bestTrialId: string | null): {
  bestScore: number | null;
  clippedLowCount: number;
  clippedLowThreshold: number | null;
  linePath: string;
  points: ScoreChartPoint[];
  scaleMode: "full" | "zoomed";
  scoredCount: number;
  tickDigits: number;
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
  const { clippedLowThreshold, scaleMode, tickDigits, yMax, yMin } = buildScoreDomain(scoredValues);
  const safeRange = Math.max(scaleMode === "zoomed" ? MIN_ZOOMED_SCORE_CHART_RANGE : MIN_SCORE_CHART_RANGE, yMax - yMin);

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
    const displayScore =
      clippedLowThreshold !== null && score < clippedLowThreshold ? clippedLowThreshold : Math.min(yMax, Math.max(yMin, score));
    const normalized = (displayScore - yMin) / safeRange;
    return SCORE_CHART_PADDING.top + chartHeight - normalized * chartHeight;
  };

  const points = orderedTrials.map((trial, index) => {
    const score = trial.status === "finished" || trial.status === "error" ? trial.score : null;
    const isClippedLow = score !== null && clippedLowThreshold !== null && score < clippedLowThreshold;
    return {
      backend: trial.backend,
      createdAt: trial.createdAt,
      isBest: trial.trialId === bestTrialId,
      isClippedLow,
      lastPhase: trial.lastPhase,
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
    clippedLowCount: points.filter((point) => point.isClippedLow).length,
    clippedLowThreshold,
    linePath,
    points,
    scaleMode,
    scoredCount: scoredValues.length,
    tickDigits,
    yMax,
    yMin,
  };
}

function summarizeScorePoint(point: ScoreChartPoint): string[] {
  return [
    point.trialId,
    point.isBest ? "Best trial so far" : null,
    `Status: ${formatStatusLabel(point.status)}`,
    point.score === null ? "Score: pending" : `Score: ${formatNumber(point.score, 4)}`,
    point.isClippedLow ? "Display: pinned below the zoomed score range" : null,
    point.model ? `Model: ${point.model}` : null,
    point.backend ? `Backend: ${point.backend}` : null,
    point.lastPhase ? `Phase: ${point.lastPhase}` : null,
    point.outcomeReason ? `Outcome: ${point.outcomeReason}` : null,
    `Created: ${formatDate(point.createdAt)}`,
  ].filter((value): value is string => Boolean(value));
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
  const [hoveredScorePoint, setHoveredScorePoint] = useState<ScoreChartPoint | null>(null);
  const [expandedSectionIds, setExpandedSectionIds] = useState<string[]>(DEFAULT_EXPANDED_SECTION_IDS);
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
    setHoveredScorePoint(null);
    setExpandedSectionIds(DEFAULT_EXPANDED_SECTION_IDS);
    setError(null);
  }, [initialDetail, initialSelectedTrialId, initialTracks, selectedTrackId]);

  useEffect(() => {
    setUrlTrialId(routeTrialId);
  }, [routeTrialId]);

  useEffect(() => {
    setExpandedSectionIds(DEFAULT_EXPANDED_SECTION_IDS);
  }, [selectedTrialId]);

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
  const selectedTaskDescription = selectedTrial?.taskDescription ?? null;
  const selectedResponseText = selectedTrial?.responseText ?? null;
  const selectedReasoningText = selectedTrial?.reasoningText ?? null;
  const selectedAssertionFailures = selectedTrial?.generationAssertionFailures ?? [];
  const selectedGenerationAssertionSummary = renderGenerationAssertionSummary(
    selectedTrial?.generationAssertionsPassed ?? null,
    selectedAssertionFailures,
  );
  const selectedGenerationPropertyGroups = getGenerationPropertyGroups(
    selectedTrackId,
    selectedTrial?.provenanceJson ?? null,
  );
  const selectedIsGenerationFailure = selectedTrial?.outcomeReason === "generation_failed";
  const selectedGeneratedProgram = selectedIsGenerationFailure
    ? selectedGeneratedSource
    : selectedGeneratedSource ?? selectedTrial?.source ?? null;
  const selectedShowsDiagnosticSource = Boolean(
    selectedTrial && selectedGeneratedSource && selectedGeneratedSource !== selectedTrial.source,
  );
  const selectedProgramDiff =
    !selectedIsGenerationFailure && selectedGeneratedProgram && selectedMixedSource
      ? buildSourceDiff(selectedMixedSource.source, selectedGeneratedProgram)
      : null;
  const selectedHasInlineProgramDiff = Boolean(
    selectedProgramDiff &&
      (selectedProgramDiff.summary.added > 0 || selectedProgramDiff.summary.removed > 0),
  );
  const selectedGeneratedProgramSummary = selectedHasInlineProgramDiff
    ? `${formatMixedSourceSummary(selectedMixedSource)} • +${selectedProgramDiff?.summary.added} / -${selectedProgramDiff?.summary.removed} inline diff`
    : undefined;
  const progressPercent = getProgressPercent(detail.track);
  const coveragePercent = getCoveragePercent(detail.track);
  const attentionCount = getAttentionCount(detail.track);
  const scoreChart = buildScoreChart(visibleTrials, detail.track.bestTrialId);
  const bestTrialId =
    detail.track.bestTrialId ??
    (detail.trials.length === 0
      ? null
      : detail.trials.reduce((best, trial) => (trial.score > best.score ? trial : best), detail.trials[0]).trialId);
  const selectedIsBestTrial = selectedTrial ? isBestTrial(detail.track, selectedTrial.trialId) : false;
  const isSectionExpanded = (sectionId: string) => expandedSectionIds.includes(sectionId);
  const toggleSection = (sectionId: string) => {
    setExpandedSectionIds((current) =>
      current.includes(sectionId)
        ? current.filter((existingSectionId) => existingSectionId !== sectionId)
        : [...current, sectionId],
    );
  };

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
                  <TrackProgressBar className="track-card-bar" track={track} />
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
                        {bestTrialId ? `${compactIdentifier(bestTrialId)} is the best trial so far.` : "Waiting for scored trials."}
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
                    <div className="progress-strip-shell">
                      <TrackProgressBar track={detail.track} />
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
                    <span>
                      {scoreChart.scaleMode === "zoomed" ? "Zoomed range" : "Range"} {formatNumber(scoreChart.yMin, 4)} to{" "}
                      {formatNumber(scoreChart.yMax, 4)}
                    </span>
                    {scoreChart.scaleMode === "zoomed" ? (
                      <span>
                        {scoreChart.clippedLowCount} lower outlier{scoreChart.clippedLowCount === 1 ? "" : "s"} pinned to the
                        baseline
                      </span>
                    ) : null}
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
                      {scoreChart.scaleMode === "zoomed" ? (
                        <path
                          className="score-axis-break"
                          d={`M ${SCORE_CHART_PADDING.left - 4} ${SCORE_CHART_HEIGHT - SCORE_CHART_PADDING.bottom - 20}
                            l 4 4 l 4 -4 l 4 4`}
                        />
                      ) : null}
                      {[scoreChart.yMax, (scoreChart.yMax + scoreChart.yMin) / 2, scoreChart.yMin].map((tick) => {
                        const y =
                          SCORE_CHART_PADDING.top +
                          ((scoreChart.yMax - tick) /
                            Math.max(
                              scoreChart.scaleMode === "zoomed" ? MIN_ZOOMED_SCORE_CHART_RANGE : MIN_SCORE_CHART_RANGE,
                              scoreChart.yMax - scoreChart.yMin,
                            )) *
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
                              {formatNumber(tick, scoreChart.tickDigits)}
                            </text>
                          </g>
                        );
                      })}
                      {scoreChart.linePath ? <path className="score-line" d={scoreChart.linePath} /> : null}
                      {scoreChart.points.map((point, index) => (
                        <g key={point.trialId}>
                          <circle
                            className={`score-point tone-${point.tone} ${point.score === null ? "pending" : "scored"} ${point.isBest ? "best-point" : ""}`}
                            cx={point.x}
                            cy={point.y}
                            r={point.score === null ? 3.5 : point.isBest ? 6 : 4.5}
                            tabIndex={0}
                            aria-label={summarizeScorePoint(point).join(" • ")}
                            onMouseEnter={() => setHoveredScorePoint(point)}
                            onMouseLeave={() => setHoveredScorePoint((current) => (current?.trialId === point.trialId ? null : current))}
                            onFocus={() => setHoveredScorePoint(point)}
                            onBlur={() => setHoveredScorePoint((current) => (current?.trialId === point.trialId ? null : current))}
                          >
                            <title>{`#${index + 1} ${point.trialId}`}</title>
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
                    {hoveredScorePoint ? (
                      <div
                        className="score-point-tooltip"
                        style={{
                          left: `${(hoveredScorePoint.x / SCORE_CHART_WIDTH) * 100}%`,
                          top: `${(hoveredScorePoint.y / SCORE_CHART_HEIGHT) * 100}%`,
                        }}
                        role="status"
                        aria-live="polite"
                      >
                        {summarizeScorePoint(hoveredScorePoint).map((line, index) => (
                          <div
                            key={`${hoveredScorePoint.trialId}:${line}`}
                            className={index === 0 ? "score-point-tooltip-title" : "score-point-tooltip-line"}
                          >
                            {line}
                          </div>
                        ))}
                      </div>
                    ) : null}
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
                      <th scope="col">Status</th>
                      <th scope="col">Trial</th>
                      <th scope="col">Task</th>
                      <th scope="col">Score</th>
                      <th scope="col">Accuracy</th>
                      <th scope="col">Best Epoch</th>
                      <th scope="col">Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleTrials.map((trial) => (
                      <tr
                        key={trial.trialId}
                        role="button"
                        tabIndex={0}
                        aria-label={`Open trial ${trial.trialId}`}
                        className={`trial-row tone-${getTrialTone(trial)} ${selectedTrial?.trialId === trial.trialId ? "active" : ""} ${isBestTrial(detail.track, trial.trialId) ? "best-trial" : ""}`}
                        onClick={() => openInspector(trial.trialId)}
                        onKeyDown={(event) => handleTrialKeyDown(event, trial.trialId)}
                      >
                        <td>
                          <div className="trial-status-row">
                            <span className={`status-badge status-${trial.status}`}>
                              <span className={`status-indicator ${trial.status}`} />
                              {formatStatusLabel(trial.status)}
                            </span>
                            <span className="trial-status-duration">{formatDuration(trial.durationSec)}</span>
                          </div>
                        </td>
                        <td>
                          <div className="trial-cell-title-row">
                            <div className="trial-cell-primary">{trial.trialId}</div>
                            {isBestTrial(detail.track, trial.trialId) ? <span className="flag-chip flag-best">best so far</span> : null}
                            {trial.outcomeReason ? <span className="flag-chip">{trial.outcomeReason}</span> : null}
                            {trial.errorType ? <span className="flag-chip flag-danger">{trial.errorType}</span> : null}
                            {trial.timedOut ? <span className="flag-chip flag-warning">timed out</span> : null}
                            {trial.hadUnscoredWorkAtTimeout ? <span className="flag-chip flag-warning">unevaluated work</span> : null}
                            {trial.hasError ? <span className="flag-chip flag-danger">error payload</span> : null}
                          </div>
                          <div className="trial-cell-secondary">{getTrialNarrative(trial)}</div>
                        </td>
                        <td>
                          {trial.taskDescription ? (
                            <div className="trial-task-snippet" title={trial.taskDescription}>
                              {summarizeTaskDescription(trial.taskDescription)}
                            </div>
                          ) : (
                            <span className="trial-cell-secondary">No task description</span>
                          )}
                        </td>
                        <td>{formatNumber(trial.score, 4)}</td>
                        <td>{formatNumber(trial.accuracy, 4)}</td>
                        <td>{formatBestEpoch(trial.bestEvalEpoch, trial.epochsCompleted)}</td>
                        <td>
                          <div className="trial-cell-primary">{trial.model ?? "unknown model"}</div>
                          <div className="trial-cell-secondary">{trial.backend ?? "unknown backend"}</div>
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
                      {selectedIsBestTrial ? <span className="flag-chip flag-best">best so far</span> : null}
                      {selectedTrialRank ? <span className="meta-chip">Rank #{selectedTrialRank} by score</span> : null}
                      {renderLauncherRunBadge(selectedTrial)}
                      {renderWandbRunBadge(selectedTrial)}
                    </div>
                  </div>

                  <div className="trial-summary-grid">
                    <div className="trial-summary-column">
                      <section className="trial-summary-panel">
                        <CollapsibleSection
                          collapsible={false}
                          id="trial-overview"
                          title="Overview"
                          titleClassName="trial-summary-section-label"
                          toggleClassName="trial-summary-section-toggle"
                        >
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
                                value: selectedTrial.lastPhase ? (
                                  <span className="flag-chip">{selectedTrial.lastPhase}</span>
                                ) : (
                                  "—"
                                ),
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
                        </CollapsibleSection>
                      </section>

                      <section className="trial-summary-panel">
                        <CollapsibleSection
                          collapsible={false}
                          id="trial-metrics"
                          title="Metrics"
                          titleClassName="trial-summary-section-label"
                          toggleClassName="trial-summary-section-toggle"
                        >
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
                        </CollapsibleSection>
                      </section>
                    </div>

                    <div className="trial-summary-column">
                      <section className="trial-summary-panel">
                        <CollapsibleSection
                          collapsible={false}
                          id="trial-run-timeline"
                          title="Run timeline"
                          titleClassName="trial-summary-section-label"
                          toggleClassName="trial-summary-section-toggle"
                        >
                          <div className="timeline-list">
                            {[
                              { label: "Queued", value: formatDate(selectedTrial.createdAt) },
                              { label: "Started", value: formatDate(selectedTrial.startedAt) },
                              { label: "Finished", value: formatDate(selectedTrial.finishedAt) },
                              ...(selectedCrashSummary
                                ? [
                                    {
                                      label: "Crash detail",
                                      title: selectedCrashDetails ?? undefined,
                                      value: selectedCrashSummary,
                                    },
                                  ]
                                : []),
                              ...(selectedGenerationAssertionSummary
                                ? [
                                    {
                                      label: "Generation assertions",
                                      value: selectedGenerationAssertionSummary,
                                    },
                                  ]
                                : []),
                            ].map((row: TimelineRow) => (
                              <div className="timeline-row" key={row.label}>
                                <span>{row.label}</span>
                                <strong title={row.title}>{row.value}</strong>
                              </div>
                            ))}
                          </div>
                        </CollapsibleSection>
                      </section>

                      <section className="trial-summary-panel">
                        <CollapsibleSection
                          collapsible={false}
                          id="trial-generation-provenance"
                          title="Generation provenance"
                          titleClassName="trial-summary-section-label"
                          toggleClassName="trial-summary-section-toggle"
                        >
                          {selectedGenerationPropertyGroups.length > 0 ? (
                            <div className="trial-summary-group-stack">
                              {selectedGenerationPropertyGroups.map((group) => (
                                <section className="trial-summary-subsection" key={group.label}>
                                  <div className="trial-summary-subsection-label">{group.label}</div>
                                  <div className="context-stack">
                                    {group.entries.map((entry) => (
                                      <div className="context-row" key={`${group.label}:${entry.label}`}>
                                        <span>{entry.label}</span>
                                        <strong className={entry.mono ? "trial-summary-mono" : undefined}>
                                          {entry.values.length === 1 ? (
                                            renderPropertyValue(entry.values[0], "trial-summary-link")
                                          ) : (
                                            <span className="property-chip-list">
                                              {entry.values.map((item) => {
                                                const itemClassName = `meta-chip ${entry.mono ? "meta-chip-mono" : ""}`.trim();
                                                const key = `${group.label}:${entry.label}:${item.text}`;
                                                return item.href ? (
                                                  <span key={key}>
                                                    {renderPropertyValue(item, itemClassName)}
                                                  </span>
                                                ) : (
                                                  <span key={key} className={itemClassName}>
                                                    {item.text}
                                                  </span>
                                                );
                                              })}
                                            </span>
                                          )}
                                        </strong>
                                      </div>
                                    ))}
                                  </div>
                                </section>
                              ))}
                            </div>
                          ) : (
                            <p className="section-copy">No provenance payload recorded.</p>
                          )}
                        </CollapsibleSection>
                      </section>
                    </div>
                  </div>
                </article>

                <div className="inspector-grid">
                  <article className="analysis-card wide-card">
                    <CollapsibleSection
                      id="trial-task-description"
                      expanded={isSectionExpanded("trial-task-description")}
                      onToggle={() => toggleSection("trial-task-description")}
                      title="Task description"
                      titleTag="h3"
                      toggleClassName="analysis-card-header"
                    >
                      <HighlightedCode
                        code={selectedTaskDescription ?? "No task description recorded."}
                        language={detectPromptLanguage(selectedTaskDescription ?? "")}
                        wrap
                      />
                    </CollapsibleSection>
                  </article>

                  <article className="analysis-card wide-card">
                    <CollapsibleSection
                      id="trial-generated-program"
                      expanded={isSectionExpanded("trial-generated-program")}
                      onToggle={() => toggleSection("trial-generated-program")}
                      title={selectedIsGenerationFailure ? "Generation attempt" : "Generated program"}
                      titleTag="h3"
                      summary={selectedGeneratedProgramSummary}
                      toggleClassName="analysis-card-header"
                    >
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
                        diffBefore={selectedHasInlineProgramDiff ? selectedMixedSource?.source : null}
                        language="python"
                        wrap
                      />
                    </CollapsibleSection>
                  </article>

                  {selectedTrial.hasError ? (
                    <article className="analysis-card wide-card">
                      <CollapsibleSection
                        id="trial-error-payload"
                        expanded={isSectionExpanded("trial-error-payload")}
                        onToggle={() => toggleSection("trial-error-payload")}
                        title="Error payload"
                        titleTag="h3"
                        toggleClassName="analysis-card-header"
                      >
                        <HighlightedCode code={formatJsonBlock(selectedTrial.errorJson)} language="json" wrap />
                      </CollapsibleSection>
                    </article>
                  ) : null}

                  <article className="analysis-card wide-card">
                    <CollapsibleSection
                      id="trial-system-prompt"
                      expanded={isSectionExpanded("trial-system-prompt")}
                      onToggle={() => toggleSection("trial-system-prompt")}
                      title="System prompt"
                      titleTag="h3"
                      toggleClassName="analysis-card-header"
                    >
                      <HighlightedCode
                        code={selectedSystemPrompt ?? "No system prompt recorded."}
                        language={detectPromptLanguage(selectedSystemPrompt ?? "")}
                        wrap
                      />
                    </CollapsibleSection>
                  </article>

                  <article className="analysis-card wide-card">
                    <CollapsibleSection
                      id="trial-user-prompt"
                      expanded={isSectionExpanded("trial-user-prompt")}
                      onToggle={() => toggleSection("trial-user-prompt")}
                      title="User prompt"
                      titleTag="h3"
                      toggleClassName="analysis-card-header"
                    >
                      <HighlightedCode
                        code={selectedUserPrompt ?? "No user prompt recorded."}
                        language={detectPromptLanguage(selectedUserPrompt ?? "")}
                        wrap
                      />
                    </CollapsibleSection>
                  </article>

                  <article className="analysis-card wide-card">
                    <CollapsibleSection
                      id="trial-reasoning-trace"
                      expanded={isSectionExpanded("trial-reasoning-trace")}
                      onToggle={() => toggleSection("trial-reasoning-trace")}
                      title="Reasoning trace"
                      titleTag="h3"
                      toggleClassName="analysis-card-header"
                    >
                      <MarkdownContent
                        content={selectedReasoningText ?? "No reasoning trace recorded."}
                      />
                    </CollapsibleSection>
                  </article>

                  <article className="analysis-card wide-card">
                    <CollapsibleSection
                      id="trial-raw-llm-response"
                      expanded={isSectionExpanded("trial-raw-llm-response")}
                      onToggle={() => toggleSection("trial-raw-llm-response")}
                      title="Response"
                      titleTag="h3"
                      toggleClassName="analysis-card-header"
                    >
                      <HighlightedCode
                        code={selectedResponseText ?? "No raw response recorded."}
                        language={detectPromptLanguage(selectedResponseText ?? "")}
                        wrap
                      />
                    </CollapsibleSection>
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
