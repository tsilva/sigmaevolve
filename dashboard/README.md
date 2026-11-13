<div align="center">
  <img src="./logo.png" alt="SigmaEvolve" width="260" />
</div>

# SigmaEvolve Dashboard

Live dashboard for browsing SigmaEvolve experiment tracks, trial state, and evaluation telemetry from a shared Postgres database.

## Requirements

- Node.js 20+
- A reachable Postgres instance exposed through `DATABASE_URL`

## Local Development

```bash
npm install
cp .env.example .env.local
npm run dev -- --port auto
```

Open the local URL printed by the server. The root route redirects to the newest track when data exists; otherwise it shows an empty-state screen until the Python harness writes tracks into the shared database.
Track and trial pages preload the dashboard's monospace font. The empty homepage loads only the shared sans-serif font.

## Environment

- `DATABASE_URL`: Postgres connection string used for track and trial queries.
- `NEXT_PUBLIC_SITE_URL`: Absolute URL used for canonical metadata, Open Graph URLs, `robots.txt`, and `sitemap.xml`. Defaults to `https://sigmaevolve.tsilva.eu`.
- `SENTRY_DSN`: Server-side Sentry DSN.
- `NEXT_PUBLIC_SENTRY_DSN`: Browser Sentry DSN. This can use the same DSN value as `SENTRY_DSN`.
- `SENTRY_ORG`: Sentry organization slug for release and source-map upload.
- `SENTRY_PROJECT`: Sentry project slug for release and source-map upload.
- `SENTRY_AUTH_TOKEN`: Sentry token used by hosted Vercel builds to upload source maps. Local builds skip uploads.
- `SENTRY_TRACES_SAMPLE_RATE`: Performance tracing sample rate from `0` to `1`; defaults to `0.1`.
- `SENTRY_SMOKE_TOKEN`: Private token for the protected Sentry smoke-test route.

Keep private values out of tracked files. Production delivery preserves separate environment credentials.

## Commands

```bash
npm run dev -- --port auto
npm run build
npm run start
npm run test
```
