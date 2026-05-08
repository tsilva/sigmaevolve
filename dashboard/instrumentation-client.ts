import * as Sentry from "@sentry/nextjs";

import { getSentryTracesSampleRate } from "./sentry.shared.config";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  enabled: Boolean(process.env.NEXT_PUBLIC_SENTRY_DSN),
  tracesSampleRate: getSentryTracesSampleRate(),
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
