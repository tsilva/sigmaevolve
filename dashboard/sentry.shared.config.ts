export function getSentryTracesSampleRate(): number {
  const rawSampleRate = process.env.SENTRY_TRACES_SAMPLE_RATE ?? "0.1";
  const sampleRate = Number(rawSampleRate);

  if (!Number.isFinite(sampleRate) || sampleRate < 0 || sampleRate > 1) {
    return 0.1;
  }

  return sampleRate;
}
