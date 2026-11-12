export function getSentryTracesSampleRate(): number {
  const rawSampleRate = process.env.SENTRY_TRACES_SAMPLE_RATE ?? "0.1";
  const sampleRate = Number(rawSampleRate);

  if (!Number.isFinite(sampleRate) || sampleRate < 0 || sampleRate > 1) {
    return 0.1;
  }

  return sampleRate;
}
// Preserve v10 collection and transaction behavior across all dashboard runtimes.
export const sentryCompatibilityOptions = {
  traceLifecycle: "static" as const,
  dataCollection: {
    userInfo: false,
    cookies: false,
    httpHeaders: {
      request: { deny: ["forwarded", "-ip", "remote-", "via", "-user"] },
      response: { deny: ["forwarded", "-ip", "remote-", "via", "-user"] },
    },
    httpBodies: [],
    urlQueryParams: { deny: ["forwarded", "-ip", "remote-", "via", "-user"] },
    genAI: { inputs: false, outputs: false },
    databaseQueryData: false,
    queues: false,
    graphQL: { document: false, variables: false },
    frameContextLines: 7,
  },
};
