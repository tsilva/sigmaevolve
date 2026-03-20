import { Client, Pool } from "pg";

import type { DashboardNotification } from "@/lib/types";

const DASHBOARD_CHANNEL = "sigmaevolve_dashboard";

declare global {
  var __sigmaevolveDashboardPool: Pool | undefined;
}

export function hasDatabaseUrl(): boolean {
  return Boolean(process.env.DATABASE_URL || process.env.SIGMAEVOLVE_DATABASE_URL);
}

function databaseUrl(): string {
  const value = process.env.DATABASE_URL || process.env.SIGMAEVOLVE_DATABASE_URL;
  if (!value) {
    throw new Error("DATABASE_URL or SIGMAEVOLVE_DATABASE_URL is required.");
  }
  return value;
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
