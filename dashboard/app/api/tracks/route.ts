import { NextResponse } from "next/server";

import { listTrackSummaries } from "@/lib/queries";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const tracks = await listTrackSummaries();
  return NextResponse.json(tracks);
}
