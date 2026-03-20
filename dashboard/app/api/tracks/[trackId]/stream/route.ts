import { createDashboardSseResponse } from "@/lib/sse";
import { subscribeToDashboardNotifications } from "@/lib/db";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request) {
  return createDashboardSseResponse({
    signal: request.signal,
    subscribe: subscribeToDashboardNotifications,
  });
}
