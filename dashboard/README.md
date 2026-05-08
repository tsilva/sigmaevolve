<div align="center">
  <img src="./logo.png" alt="SigmaEvolve" width="420" />
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
npm run dev
```

The app starts on `http://localhost:3000`. The root route redirects to the newest track when data exists; otherwise it shows an empty-state screen until the Python harness writes tracks into the shared database.

## Environment

- `DATABASE_URL`: Postgres connection string used for track and trial queries.
- `NEXT_PUBLIC_SITE_URL`: Absolute production URL used for canonical metadata, Open Graph URLs, `robots.txt`, and `sitemap.xml`.
- `SENTRY_DSN`: Server-side Sentry DSN.
- `NEXT_PUBLIC_SENTRY_DSN`: Browser Sentry DSN. This can use the same DSN value as `SENTRY_DSN`.
- `SENTRY_ORG`: Sentry organization slug for release and source-map upload.
- `SENTRY_PROJECT`: Sentry project slug for release and source-map upload.
- `SENTRY_AUTH_TOKEN`: Sentry token used by Vercel builds to upload source maps.
- `SENTRY_TRACES_SAMPLE_RATE`: Performance tracing sample rate from `0` to `1`; defaults to `0.1`.
- `SENTRY_SMOKE_TOKEN`: Private token for the protected Sentry smoke-test route.

Keep real Sentry values in `.env.local` or Vercel environment variables, not in tracked files. To push local values from `.env.local` or `.env` into the linked Vercel project, run:

```bash
npm run sentry:env:push
```

The script uploads `production` and `preview` values by default. Add `-- --force` to overwrite existing Vercel values, or pass explicit environments such as `npm run sentry:env:push -- production preview development`.

## Commands

```bash
npm run dev
npm run build
npm run start
npm run sentry:env:push
npm run test
```
