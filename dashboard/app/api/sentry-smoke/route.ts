import * as Sentry from "@sentry/nextjs";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request) {
  const smokeToken = process.env.SENTRY_SMOKE_TOKEN;
  const requestToken = request.headers.get("x-sentry-smoke-token");

  if (!smokeToken || requestToken !== smokeToken) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  Sentry.captureMessage("sigmaevolve-sentry-smoke", {
    level: "info",
    tags: {
      smoke_test: "sentry",
    },
  });

  await Sentry.flush(2_000);

  return NextResponse.json({ ok: true });
}
