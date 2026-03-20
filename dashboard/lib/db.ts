import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

import { Client, Pool } from "pg";

import type { DashboardNotification } from "@/lib/types";

const DASHBOARD_CHANNEL = "sigmaevolve_dashboard";

declare global {
  var __sigmaevolveDashboardPool: Pool | undefined;
  var __sigmaevolveDashboardSqlite: DatabaseSync | undefined;
  var __sigmaevolveDashboardSqlitePath: string | undefined;
}

const SQLITE_CANDIDATE_PATHS = [
  process.env.SIGMAEVOLVE_SQLITE_PATH,
  path.join(process.cwd(), "sigmaevolve.sqlite"),
  path.join(process.cwd(), "../sigmaevolve.sqlite"),
  path.join(process.cwd(), "../../sigmaevolve.sqlite"),
].filter((value): value is string => Boolean(value));

export function hasDatabaseUrl(): boolean {
  return Boolean(process.env.DATABASE_URL);
}

function databaseUrl(): string {
  const value = process.env.DATABASE_URL;
  if (!value) {
    throw new Error("DATABASE_URL is required.");
  }
  return value;
}

export function findSqliteDatabasePath(): string | null {
  for (const candidate of SQLITE_CANDIDATE_PATHS) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

export function getSqliteDatabase(): DatabaseSync | null {
  const sqlitePath = findSqliteDatabasePath();
  if (!sqlitePath) {
    return null;
  }

  if (
    !globalThis.__sigmaevolveDashboardSqlite ||
    globalThis.__sigmaevolveDashboardSqlitePath !== sqlitePath
  ) {
    globalThis.__sigmaevolveDashboardSqlite?.close();
    globalThis.__sigmaevolveDashboardSqlite = new DatabaseSync(sqlitePath);
    globalThis.__sigmaevolveDashboardSqlitePath = sqlitePath;
  }

  return globalThis.__sigmaevolveDashboardSqlite;
}

export function getPool(): Pool {
  if (!globalThis.__sigmaevolveDashboardPool) {
    globalThis.__sigmaevolveDashboardPool = new Pool({
      connectionString: databaseUrl(),
      max: 5,
    });
  }
  return globalThis.__sigmaevolveDashboardPool;
}

function isDashboardNotification(value: unknown): value is DashboardNotification {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as Partial<DashboardNotification>;
  return (
    typeof candidate.trackId === "string" &&
    (candidate.reason === "trial_changed" || candidate.reason === "track_changed")
  );
}

export async function subscribeToDashboardNotifications(
  emit: (notification: DashboardNotification) => void,
): Promise<() => Promise<void>> {
  if (!hasDatabaseUrl()) {
    return async () => {};
  }

  const client = new Client({
    connectionString: databaseUrl(),
    application_name: "sigmaevolve-dashboard-sse",
  });

  try {
    await client.connect();
    await client.query(`LISTEN ${DASHBOARD_CHANNEL}`);
  } catch (error) {
    console.error("Dashboard SSE disabled because LISTEN setup failed.", error);
    await client.end().catch(() => {});
    return async () => {};
  }

  const onNotification = (message: { payload?: string | null }) => {
    if (!message.payload) {
      return;
    }
    try {
      const payload = JSON.parse(message.payload);
      if (isDashboardNotification(payload)) {
        emit(payload);
      }
    } catch {
      return;
    }
  };

  client.on("notification", onNotification);

  return async () => {
    client.off("notification", onNotification);
    try {
      await client.query(`UNLISTEN ${DASHBOARD_CHANNEL}`);
    } finally {
      await client.end();
    }
  };
}
