import { EventEmitter } from "node:events";
import { getPool } from "@/lib/db";

const capture = vi.hoisted(() => vi.fn());
vi.mock("@sentry/nextjs", () => ({ captureException: capture }));
vi.mock("pg", () => ({
  Pool: class extends EventEmitter {
    query = vi.fn();
  },
  Client: class extends EventEmitter {},
}));

describe("dashboard idle database errors", () => {
  beforeEach(() => {
    delete globalThis.__sigmaevolveDashboardPool;
    vi.stubEnv("DATABASE_URL", "postgresql://fixture:fixture@127.0.0.1:1/fixture");
    vi.stubEnv("SIGMAEVOLVE_DATABASE_TRANSPORT", undefined);
    capture.mockClear();
  });

  afterEach(() => {
    delete globalThis.__sigmaevolveDashboardPool;
    vi.unstubAllEnvs();
  });

  it("reports an idle connection termination without an uncaught EventEmitter error", () => {
    const pool = getPool();
    const error = new Error("terminating connection due to administrator command");
    expect(() => pool.emit("error", error)).not.toThrow();
    expect(capture).toHaveBeenCalledWith(error, expect.objectContaining({
      tags: { component: "dashboard-database-pool" },
    }));
    expect(getPool()).toBe(pool);
    expect(pool.listenerCount("error")).toBe(1);
  });

  it("preserves query rejection so request error handling can respond", async () => {
    const pool = getPool();
    const error = new Error("query failed");
    vi.mocked(pool.query).mockRejectedValueOnce(error as never);
    await expect(pool.query("SELECT 1")).rejects.toBe(error);
  });
});
