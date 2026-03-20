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

type TrialCursor = {
  createdAt: string;
  trialId: string;
};

function encodeCursor(value: TrialCursor): string {
  return Buffer.from(JSON.stringify(value), "utf8").toString("base64url");
}

function decodeCursor(value: string | null): TrialCursor | null {
  if (!value) {
    return null;
  }
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
  if (value === "queued" || value === "dispatching" || value === "active" || value === "finished") {
    return value;
  }
  return "all";
}

export function parseLimit(value: string | null): number {
  if (!value) {
    return DEFAULT_TRIAL_LIMIT;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_TRIAL_LIMIT;
  }
  return Math.min(parsed, MAX_TRIAL_LIMIT);
}

export async function listTrackSummaries(): Promise<TrackListItem[]> {
  if (!hasDatabaseUrl()) {
    throw new Error("DATABASE_URL is required for the dashboard.");
  }

  const pool = getPool();
  const result = await pool.query(
    `
      select
        t.track_id as "trackId",
        t.name,
        t.dataset_id as "datasetId",
        t.created_at as "createdAt",
        count(r.trial_id)::int as "totalTrials",
        count(*) filter (where r.status = 'queued')::int as "queuedTrials",
        count(*) filter (where r.status = 'dispatching')::int as "dispatchingTrials",
        count(*) filter (where r.status = 'active')::int as "activeTrials",
        count(*) filter (where r.status = 'finished')::int as "finishedTrials",
        count(*) filter (
          where r.status = 'finished'
            and r.metrics_json is not null
        )::int as "succeededTrials",
        max(r.score) filter (
          where r.status = 'finished'
            and r.metrics_json is not null
        ) as "bestScore",
        greatest(
          t.created_at,
          coalesce(max(coalesce(r.finished_at, r.started_at, r.created_at)), t.created_at)
        ) as "lastActivityAt"
      from tracks t
      left join trials r on r.track_id = t.track_id
      group by t.track_id, t.name, t.dataset_id, t.created_at
      order by "lastActivityAt" desc, t.created_at desc
    `,
  );

  return result.rows.map(mapTrackListItem);
}

export async function getNewestTrackId(): Promise<string | null> {
  if (!hasDatabaseUrl()) {
    throw new Error("DATABASE_URL is required for the dashboard.");
  }

  const pool = getPool();
  const result = await pool.query<{ track_id: string }>(
    `select track_id from tracks order by created_at desc limit 1`,
  );
  return result.rows[0]?.track_id ?? null;
}

export async function getTrackSummary(trackId: string): Promise<TrackListItem | null> {
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
  const status = options.status ?? "all";
  const limit = Math.min(options.limit ?? DEFAULT_TRIAL_LIMIT, MAX_TRIAL_LIMIT);
  if (!hasDatabaseUrl()) {
    throw new Error("DATABASE_URL is required for the dashboard.");
  }

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
    whereClauses.push(`(created_at, trial_id) < ($${values.length - 1}::timestamptz, $${values.length})`);
  }

  values.push(limit + 1);

  const result = await pool.query(
    `
      select
        trial_id as "trialId",
        status,
        outcome_reason as "outcomeReason",
        score,
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
          status = 'finished'
          and outcome_reason in ('crashed', 'eval_failed', 'stale')
        ) as "hasError"
      from trials
      where ${whereClauses.join(" and ")}
      order by created_at desc, trial_id desc
      limit $${values.length}
    `,
    values,
  );

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
