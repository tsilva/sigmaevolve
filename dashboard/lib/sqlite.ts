import { getSqliteDatabase } from "@/lib/db";
import { mapTrackListItem, mapTrialListItem } from "@/lib/mappers";
import type {
  PaginatedTrialsResponse,
  TrackListItem,
  TrialListItem,
  TrialStatusFilter,
} from "@/lib/types";

type TrialCursor = {
  createdAt: string;
  trialId: string;
};

function parseJsonRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== "string") {
    return {};
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return parsed as Record<string, unknown>;
  } catch {
    return {};
  }
}

function asNullableNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string" && value.length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function asBoolean(value: unknown): boolean {
  return value === true;
}

function asNullableString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function durationInSeconds(startedAt: string | null, finishedAt: string | null): number | null {
  if (!startedAt) {
    return null;
  }

  const startMs = Date.parse(startedAt);
  const endMs = Date.parse(finishedAt ?? new Date().toISOString());
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
    return null;
  }

  return Math.max(0, (endMs - startMs) / 1000);
}

function mapSqliteTrialListItem(row: {
  trialId: string;
  status: TrialListItem["status"];
  outcomeReason: string | null;
  score: number | null;
  provenanceJson: string | null;
  metricsJson: string | null;
  errorJson: string | null;
  dispatchAttempts: number | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
}): TrialListItem {
  const metrics = parseJsonRecord(row.metricsJson);
  const provenance = parseJsonRecord(row.provenanceJson);

  return mapTrialListItem({
    trialId: row.trialId,
    status: row.status,
    outcomeReason: row.outcomeReason,
    score: row.score,
    accuracy: asNullableNumber(metrics.accuracy),
    timeToBestEvalSec: asNullableNumber(metrics.time_to_best_eval_sec),
    timedOut: asBoolean(metrics.timed_out),
    timeSinceLastEvalSec: asNullableNumber(metrics.time_since_last_eval_sec),
    hadUnscoredWorkAtTimeout: asBoolean(metrics.had_unscored_work_at_timeout),
    lastPhase: asNullableString(metrics.last_phase),
    backend: asNullableString(provenance.backend),
    model: asNullableString(provenance.model),
    dispatchAttempts: row.dispatchAttempts,
    createdAt: row.createdAt,
    startedAt: row.startedAt,
    finishedAt: row.finishedAt,
    durationSec: durationInSeconds(row.startedAt, row.finishedAt),
    hasError: row.errorJson !== null,
  });
}

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
    return { createdAt: parsed.createdAt, trialId: parsed.trialId };
  } catch {
    return null;
  }
}

export async function listSqliteTrackSummaries(): Promise<TrackListItem[]> {
  const database = getSqliteDatabase();
  if (!database) {
    return [];
  }

  const rows = database
    .prepare(
      `
        select
          t.track_id as trackId,
          t.name as name,
          t.dataset_id as datasetId,
          t.created_at as createdAt,
          count(r.trial_id) as totalTrials,
          sum(case when r.status = 'queued' then 1 else 0 end) as queuedTrials,
          sum(case when r.status = 'dispatching' then 1 else 0 end) as dispatchingTrials,
          sum(case when r.status = 'active' then 1 else 0 end) as activeTrials,
          sum(case when r.status = 'finished' then 1 else 0 end) as finishedTrials,
          sum(
            case
              when r.status = 'finished' and r.outcome_reason = 'succeeded' then 1
              else 0
            end
          ) as succeededTrials,
          max(
            case
              when r.status = 'finished' and r.outcome_reason = 'succeeded' then r.score
              else null
            end
          ) as bestScore,
          max(
            t.created_at,
            coalesce(max(coalesce(r.finished_at, r.started_at, r.created_at)), t.created_at)
          ) as lastActivityAt
        from tracks t
        left join trials r on r.track_id = t.track_id
        group by t.track_id, t.name, t.dataset_id, t.created_at
        order by lastActivityAt desc, t.created_at desc
      `,
    )
    .all() as Array<Record<string, string | number | null>>;

  return rows.map((row) =>
    mapTrackListItem({
      trackId: String(row.trackId),
      name: row.name === null ? null : String(row.name),
      datasetId: String(row.datasetId),
      createdAt: String(row.createdAt),
      totalTrials: row.totalTrials,
      queuedTrials: row.queuedTrials,
      dispatchingTrials: row.dispatchingTrials,
      activeTrials: row.activeTrials,
      finishedTrials: row.finishedTrials,
      succeededTrials: row.succeededTrials,
      bestScore: row.bestScore,
      lastActivityAt: row.lastActivityAt === null ? null : String(row.lastActivityAt),
    }),
  );
}

export async function getSqliteNewestTrackId(): Promise<string | null> {
  const database = getSqliteDatabase();
  if (!database) {
    return null;
  }

  const row = database
    .prepare("select track_id as trackId from tracks order by created_at desc limit 1")
    .get() as { trackId?: string } | undefined;

  return row?.trackId ?? null;
}

export async function listSqliteTrials(
  trackId: string,
  options: {
    status?: TrialStatusFilter;
    cursor?: string | null;
    limit: number;
  },
): Promise<PaginatedTrialsResponse> {
  const database = getSqliteDatabase();
  if (!database) {
    return { trials: [], nextCursor: null };
  }

  const whereClauses = ["track_id = ?"];
  const values: Array<string | number> = [trackId];

  if (options.status && options.status !== "all") {
    whereClauses.push("status = ?");
    values.push(options.status);
  }

  const cursor = decodeCursor(options.cursor ?? null);
  if (cursor) {
    whereClauses.push("(created_at < ? or (created_at = ? and trial_id < ?))");
    values.push(cursor.createdAt, cursor.createdAt, cursor.trialId);
  }

  values.push(options.limit + 1);

  const rows = database
    .prepare(
      `
        select
          trial_id as trialId,
          status,
          outcome_reason as outcomeReason,
          score,
          provenance_json as provenanceJson,
          metrics_json as metricsJson,
          error_json as errorJson,
          dispatch_attempts as dispatchAttempts,
          created_at as createdAt,
          started_at as startedAt,
          finished_at as finishedAt
        from trials
        where ${whereClauses.join(" and ")}
        order by created_at desc, trial_id desc
        limit ?
      `,
    )
    .all(...values) as Array<{
    trialId: string;
    status: TrialListItem["status"];
    outcomeReason: string | null;
    score: number | null;
    provenanceJson: string | null;
    metricsJson: string | null;
    errorJson: string | null;
    dispatchAttempts: number | null;
    createdAt: string;
    startedAt: string | null;
    finishedAt: string | null;
  }>;

  const mapped = rows.map(mapSqliteTrialListItem);
  const page = mapped.slice(0, options.limit);
  const nextCursor =
    mapped.length > options.limit
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
