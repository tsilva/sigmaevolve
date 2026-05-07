import { Client, Pool } from "pg";

import type { DashboardNotification } from "@/lib/types";

const DASHBOARD_CHANNEL = "sigmaevolve_dashboard";

declare global {
  var __sigmaevolveDashboardPool: Pool | undefined;
}

export function hasDatabaseUrl(): boolean {
  return Boolean(process.env.DATABASE_URL);
}

function databaseUrl(): string {
  // Validate the database URL once before any connection is attempted.
  const value = process.env.DATABASE_URL;
  if (!value) {
    throw new Error("DATABASE_URL is required.");
  }
  const parsed = new URL(value);
  const sslMode = parsed.searchParams.get("sslmode");
  if (sslMode === "prefer" || sslMode === "require" || sslMode === "verify-ca") {
    parsed.searchParams.set("sslmode", "verify-full");
  }
  return parsed.toString();
}

export function getPool(): Pool {
  // Reuse a singleton connection pool across hot reloads and requests.
  if (!globalThis.__sigmaevolveDashboardPool) {
    globalThis.__sigmaevolveDashboardPool = new Pool({
      connectionString: databaseUrl(),
      max: 5,
    });
  }
  return globalThis.__sigmaevolveDashboardPool;
}

function isDashboardNotification(value: unknown): value is DashboardNotification {
  // Validate the minimal payload shape emitted by the backend notifications.
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
  // Disable LISTEN/NOTIFY cleanly when the dashboard has no database connection.
  if (!hasDatabaseUrl()) {
    return async () => {};
  }

  // Open a dedicated client for the dashboard notification channel.
  const client = new Client({
    connectionString: databaseUrl(),
    application_name: "sigmaevolve-dashboard-sse",
  });

  try {
    await client.connect();
    await client.query(`LISTEN ${DASHBOARD_CHANNEL}`);
  } catch (error) {
    // Fall back silently when LISTEN/NOTIFY cannot be initialized.
    console.error("Dashboard SSE disabled because LISTEN setup failed.", error);
    await client.end().catch(() => {});
    return async () => {};
  }

  // Parse and forward only well-formed dashboard notifications.
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
    // Unsubscribe and close the client when the stream is torn down.
    client.off("notification", onNotification);
    try {
      await client.query(`UNLISTEN ${DASHBOARD_CHANNEL}`);
    } finally {
      await client.end();
    }
  };
}
