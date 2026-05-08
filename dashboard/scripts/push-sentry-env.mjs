import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";

import nextEnv from "@next/env";

const { loadEnvConfig } = nextEnv;

loadEnvConfig(process.cwd(), true);

const args = process.argv.slice(2);
const force = args.includes("--force");
const environments = args.filter((arg) => arg !== "--force");
const targets = environments.length > 0 ? environments : ["production", "preview"];
const projectConfig = JSON.parse(await readFile(".vercel/project.json", "utf8"));
const envEndpoint = `/v10/projects/${projectConfig.projectId}/env?teamId=${projectConfig.orgId}`;

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

async function runVercelApi(path, options = {}) {
  const commandArgs = ["api", path];

  if (options.method) {
    commandArgs.push("--method", options.method);
  }

  if (options.input) {
    commandArgs.push("--input", "-");
  }

  if (options.silent) {
    commandArgs.push("--silent");
  }

  return new Promise((resolve, reject) => {
    const child = spawn("vercel", commandArgs, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout);
        return;
      }

      reject(
        new Error(
          `vercel ${commandArgs.join(" ")} exited with ${code}: ${stderr}`,
        ),
      );
    });

    child.stdin.end(options.input ?? "");
  });
}

async function getExistingEnvVars() {
  const output = await runVercelApi(envEndpoint);
  const payload = JSON.parse(output);

  return payload.envs ?? payload.env ?? [];
}

async function deleteExistingEnvVars(name, existingEnvVars) {
  const matchingEnvVars = existingEnvVars.filter(
    (envVar) =>
      envVar.key === name &&
      envVar.target?.some((target) => targets.includes(target)),
  );

  if (matchingEnvVars.length > 0 && !force) {
    throw new Error(`${name} already exists in Vercel. Re-run with --force.`);
  }

  for (const envVar of matchingEnvVars) {
    const deleteEndpoint = `/v9/projects/${projectConfig.projectId}/env/${envVar.id}?teamId=${projectConfig.orgId}`;
    await runVercelApi(deleteEndpoint, { method: "DELETE", silent: true });
  }
}

async function createVercelEnv(name, value) {
  const isSecret = name === "SENTRY_AUTH_TOKEN" || name === "SENTRY_SMOKE_TOKEN";

  await runVercelApi(envEndpoint, {
    method: "POST",
    input: JSON.stringify({
      key: name,
      value,
      type: isSecret ? "sensitive" : "encrypted",
      target: targets,
    }),
    silent: true,
  });
}

const existingEnvVars = await getExistingEnvVars();

for (const [name, value] of Object.entries(requiredValues)) {
  console.log(`Uploading ${name} to Vercel ${targets.join(", ")}...`);
  await deleteExistingEnvVars(name, existingEnvVars);
  await createVercelEnv(name, value);
}

console.log(`Uploaded Sentry env values to: ${targets.join(", ")}`);
