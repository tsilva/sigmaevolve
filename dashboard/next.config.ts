import type { NextConfig } from "next";
import { withSentryConfig } from "@sentry/nextjs";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  distDir: process.env.NEXT_DEV_OUTPUT_DIR || ".next",
};

const isHostedBuild = process.env.VERCEL === "1" && Boolean(process.env.VERCEL_DEPLOYMENT_ID);

export default withSentryConfig(nextConfig, {
  webpack: { disableSentryConfig: !isHostedBuild },
  useRunAfterProductionCompileHook: isHostedBuild ? undefined : false,
  authToken: isHostedBuild ? process.env.SENTRY_AUTH_TOKEN : undefined,
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  sentryUrl: process.env.SENTRY_URL,
  silent: !process.env.CI,
  sourcemaps: {
    disable: !isHostedBuild || !process.env.SENTRY_AUTH_TOKEN,
  },
  widenClientFileUpload: true,
});
