import * as Sentry from "@sentry/nextjs";

import { getSentryTracesSampleRate } from "./sentry.shared.config";

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  enabled: Boolean(process.env.SENTRY_DSN),
  tracesSampleRate: getSentryTracesSampleRate(),
});
