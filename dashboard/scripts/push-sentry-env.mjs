import { spawn } from "node:child_process";

import nextEnv from "@next/env";

const { loadEnvConfig } = nextEnv;

loadEnvConfig(process.cwd(), true);

const args = process.argv.slice(2);
const force = args.includes("--force");
const environments = args.filter((arg) => arg !== "--force");
const targets = environments.length > 0 ? environments : ["production", "preview"];

const requiredValues = {
  SENTRY_DSN: process.env.SENTRY_DSN,
  NEXT_PUBLIC_SENTRY_DSN:
    process.env.NEXT_PUBLIC_SENTRY_DSN || process.env.SENTRY_DSN,
  SENTRY_ORG: process.env.SENTRY_ORG,
  SENTRY_PROJECT: process.env.SENTRY_PROJECT,
  SENTRY_AUTH_TOKEN: process.env.SENTRY_AUTH_TOKEN,
  SENTRY_TRACES_SAMPLE_RATE: process.env.SENTRY_TRACES_SAMPLE_RATE || "0.1",
  SENTRY_SMOKE_TOKEN: process.env.SENTRY_SMOKE_TOKEN,
};

if (process.env.SENTRY_URL) {
  requiredValues.SENTRY_URL = process.env.SENTRY_URL;
}

const missing = Object.entries(requiredValues)
  .filter(([, value]) => !value)
  .map(([key]) => key);

if (missing.length > 0) {
  console.error(`Missing Sentry env values: ${missing.join(", ")}`);
  console.error("Set them in dashboard/.env.local or dashboard/.env first.");
  process.exit(1);
}

async function addVercelEnv(name, value, environment) {
  const commandArgs = ["env", "add", name, environment, "--yes"];

  if (force) {
    commandArgs.push("--force");
  }

  if (
    (name === "SENTRY_AUTH_TOKEN" || name === "SENTRY_SMOKE_TOKEN") &&
    environment !== "development"
  ) {
    commandArgs.push("--sensitive");
  }

  await new Promise((resolve, reject) => {
    const child = spawn("vercel", commandArgs, {
      stdio: ["pipe", "inherit", "inherit"],
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`vercel ${commandArgs.join(" ")} exited with ${code}`));
    });

    child.stdin.end(`${value}\n`);
  });
}

for (const [name, value] of Object.entries(requiredValues)) {
  for (const environment of targets) {
    console.log(`Uploading ${name} to Vercel ${environment}...`);
    await addVercelEnv(name, value, environment);
  }
}

console.log(`Uploaded Sentry env values to: ${targets.join(", ")}`);
