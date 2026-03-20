import { NextResponse } from "next/server";

import { listTrials, parseLimit, parseStatusFilter } from "@/lib/queries";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ trackId: string }> },
) {
  const { trackId } = await context.params;
  const url = new URL(request.url);
  const trials = await listTrials(trackId, {
    cursor: url.searchParams.get("cursor"),
    limit: parseLimit(url.searchParams.get("limit")),
    status: parseStatusFilter(url.searchParams.get("status")),
  });
  return NextResponse.json(trials);
}
