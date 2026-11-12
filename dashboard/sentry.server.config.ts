import * as Sentry from "@sentry/nextjs";

import { getSentryTracesSampleRate, sentryCompatibilityOptions } from "./sentry.shared.config";

Sentry.init({
  ...sentryCompatibilityOptions,
  dsn: process.env.SENTRY_DSN,
  enabled: Boolean(process.env.SENTRY_DSN),
  tracesSampleRate: getSentryTracesSampleRate(),
});
