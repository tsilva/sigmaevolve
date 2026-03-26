import { notFound } from "next/navigation";

import { getPool, hasDatabaseUrl } from "@/lib/db";
import { mapTrackListItem, mapTrialListItem } from "@/lib/mappers";
import type {
  PaginatedTrialsResponse,
  TrackDetailResponse,
  TrackListItem,
  TrialStatusFilter,
} from "@/lib/types";

const DEFAULT_TRIAL_LIMIT = 50;
const MAX_TRIAL_LIMIT = 100;
const KNOWN_TRIAL_STATUSES = new Set<TrialStatusFilter>([
  "queued",
  "dispatching",
  "active",
  "finished",
  "error",
]);

function isTrialStatusFilter(value: string): value is Exclude<TrialStatusFilter, "all"> {
  return KNOWN_TRIAL_STATUSES.has(value as TrialStatusFilter);
}

type TrialCursor = {
  createdAt: string;
  trialId: string;
};

function encodeCursor(value: TrialCursor): string {
  return Buffer.from(JSON.stringify(value), "utf8").toString("base64url");
}

function decodeCursor(value: string | null): TrialCursor | null {
  // Treat empty cursors as an unpaginated request.
  if (!value) {
    return null;
  }

  // Reject malformed cursors instead of throwing from the route handler.
  try {
    const parsed = JSON.parse(Buffer.from(value, "base64url").toString("utf8")) as Partial<TrialCursor>;
    if (typeof parsed.createdAt !== "string" || typeof parsed.trialId !== "string") {
      return null;
    }
    return parsed as TrialCursor;
  } catch {
    return null;
  }
}

export function parseStatusFilter(value: string | null): TrialStatusFilter {
  // Accept only the known status values that the API can query directly.
  if (value && isTrialStatusFilter(value)) {
    return value;
  }

  return "all";
}

export function parseLimit(value: string | null): number {
  // Fall back to the default limit when the request omits a value.
  if (!value) {
    return DEFAULT_TRIAL_LIMIT;
  }

  // Clamp invalid or oversized limits into the supported range.
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_TRIAL_LIMIT;
  }
  return Math.min(parsed, MAX_TRIAL_LIMIT);
}

export async function listTrackSummaries(): Promise<TrackListItem[]> {
  // Refuse dashboard queries when the database URL is not configured.
  if (!hasDatabaseUrl()) {
    throw new Error("DATABASE_URL is required for the dashboard.");
  }

  // Load tracks together with aggregate trial stats and current best score.
  const pool = getPool();
  const result = await pool.query(
    `
      select
        t.track_id as "trackId",
        t.dataset_id as "datasetId",
        t.created_at as "createdAt",
        coalesce(stats.total_trials, 0)::int as "totalTrials",
        coalesce(stats.queued_trials, 0)::int as "queuedTrials",
        coalesce(stats.dispatching_trials, 0)::int as "dispatchingTrials",
        coalesce(stats.active_trials, 0)::int as "activeTrials",
        coalesce(stats.finished_trials, 0)::int as "finishedTrials",
        coalesce(stats.error_trials, 0)::int as "errorTrials",
        coalesce(stats.succeeded_trials, 0)::int as "succeededTrials",
        best.best_score as "bestScore",
        best.best_trial_id as "bestTrialId",
        greatest(
          t.created_at,
          coalesce(stats.last_activity_at, t.created_at)
        ) as "lastActivityAt"
      from tracks t
      left join lateral (
        select
          count(*)::int as total_trials,
          count(*) filter (where status = 'queued')::int as queued_trials,
          count(*) filter (where status = 'dispatching')::int as dispatching_trials,
          count(*) filter (where status = 'active')::int as active_trials,
          count(*) filter (where status = 'finished')::int as finished_trials,
          count(*) filter (where status = 'error')::int as error_trials,
          count(*) filter (
            where status = 'finished'
              and metrics_json is not null
          )::int as succeeded_trials,
          max(coalesce(finished_at, started_at, created_at)) as last_activity_at
        from trials
        where track_id = t.track_id
      ) stats on true
      left join lateral (
        select
          trial_id as best_trial_id,
          nullif(metrics_json ->> 'accuracy', '')::double precision as best_score
        from trials
        where track_id = t.track_id
          and status = 'finished'
          and metrics_json is not null
        order by
          nullif(metrics_json ->> 'accuracy', '')::double precision desc nulls last,
          finished_at desc nulls last,
          created_at desc,
          trial_id desc
        limit 1
      ) best on true
      order by "lastActivityAt" desc, t.created_at desc
    `,
  );

  // Map raw rows into the shared dashboard response type.
  return result.rows.map(mapTrackListItem);
}

export async function getNewestTrackId(): Promise<string | null> {
  // Refuse dashboard queries when the database URL is not configured.
  if (!hasDatabaseUrl()) {
    throw new Error("DATABASE_URL is required for the dashboard.");
  }

  // Return the newest track id so the UI can redirect to the latest run.
  const pool = getPool();
  const result = await pool.query<{ track_id: string }>(
    `select track_id from tracks order by created_at desc limit 1`,
  );
  return result.rows[0]?.track_id ?? null;
}

export async function getTrackSummary(trackId: string): Promise<TrackListItem | null> {
  // Reuse the list query so summary formatting stays consistent.
  const tracks = await listTrackSummaries();
  return tracks.find((track) => track.trackId === trackId) ?? null;
}

export async function listTrials(
  trackId: string,
  options: {
    status?: TrialStatusFilter;
    cursor?: string | null;
    limit?: number;
  } = {},
): Promise<PaginatedTrialsResponse> {
  // Normalize request options before building the SQL query.
  const status = options.status ?? "all";
  const limit = Math.min(options.limit ?? DEFAULT_TRIAL_LIMIT, MAX_TRIAL_LIMIT);
  if (!hasDatabaseUrl()) {
    throw new Error("DATABASE_URL is required for the dashboard.");
  }

  // Build the WHERE clause incrementally from the optional filters.
  const pool = getPool();
  const values: Array<string | number> = [trackId];
  const whereClauses = [`track_id = $1`];

  if (status !== "all") {
    values.push(status);
    whereClauses.push(`status = $${values.length}`);
  }

  const cursor = decodeCursor(options.cursor ?? null);
  if (cursor) {
    values.push(cursor.createdAt);
    values.push(cursor.trialId);
    const createdAtIndex = values.length - 1;
    const trialIdIndex = values.length;

    // Page strictly before the last seen trial tuple.
    whereClauses.push(
      `(created_at, trial_id) < ($${createdAtIndex}::timestamptz, $${trialIdIndex})`,
    );
  }

  // Request one extra row so the API can detect whether another page exists.
  values.push(limit + 1);

  // Load the filtered trials together with the fields used by the detail view.
  const result = await pool.query(
    `
      select
        trial_id as "trialId",
        status,
        outcome_reason as "outcomeReason",
        provenance_json -> 'launcher' ->> 'run_id' as "modalRunId",
        provenance_json -> 'launcher' ->> 'run_url' as "modalRunUrl",
        coalesce(nullif(metrics_json ->> 'accuracy', '')::double precision, 0) as score,
        source,
        error_json as "errorJson",
        provenance_json as "provenanceJson",
        nullif(metrics_json ->> 'accuracy', '')::double precision as accuracy,
        nullif(metrics_json ->> 'time_to_best_eval_sec', '')::double precision as "timeToBestEvalSec",
        coalesce((metrics_json ->> 'timed_out')::boolean, false) as "timedOut",
        nullif(metrics_json ->> 'time_since_last_eval_sec', '')::double precision as "timeSinceLastEvalSec",
        coalesce((metrics_json ->> 'had_unscored_work_at_timeout')::boolean, false) as "hadUnscoredWorkAtTimeout",
        metrics_json ->> 'last_phase' as "lastPhase",
        provenance_json ->> 'backend' as backend,
        provenance_json ->> 'model' as model,
        dispatch_attempts as "dispatchAttempts",
        created_at as "createdAt",
        started_at as "startedAt",
        finished_at as "finishedAt",
        case
          when started_at is null then null
          else extract(epoch from (coalesce(finished_at, now()) - started_at))
        end as "durationSec",
        (
          status = 'error'
        ) as "hasError"
      from trials
      where ${whereClauses.join(" and ")}
      order by created_at desc, trial_id desc
      limit $${values.length}
    `,
    values,
  );

  // Slice to the requested page size and derive the next cursor from the last row.
  const rows = result.rows.map(mapTrialListItem);
  const page = rows.slice(0, limit);
  const nextCursor =
    rows.length > limit
      ? encodeCursor({
          createdAt: page[page.length - 1].createdAt,
          trialId: page[page.length - 1].trialId,
        })
      : null;

  return {
    trials: page,
    nextCursor,
  };
}

export async function getTrackDetail(trackId: string): Promise<TrackDetailResponse | null> {
  const [track, trials] = await Promise.all([
    getTrackSummary(trackId),
    listTrials(trackId, { status: "all", limit: DEFAULT_TRIAL_LIMIT }),
  ]);

  if (!track) {
    return null;
  }

  return {
    track,
    trials: trials.trials,
    nextCursor: trials.nextCursor,
  };
}

export async function getTrackDetailOrThrow(trackId: string): Promise<TrackDetailResponse> {
  const detail = await getTrackDetail(trackId);
  if (!detail) {
    notFound();
  }
  return detail;
}
