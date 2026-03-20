import { NextResponse } from "next/server";

import { getTrackDetail } from "@/lib/queries";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  _request: Request,
  context: { params: Promise<{ trackId: string }> },
) {
  const { trackId } = await context.params;
  const detail = await getTrackDetail(trackId);
  if (!detail) {
    return NextResponse.json({ error: "Track not found" }, { status: 404 });
  }
  return NextResponse.json(detail);
}
