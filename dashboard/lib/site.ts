const DEFAULT_SITE_URL = "https://sigmaevolve.vercel.app";

export function getSiteUrl(): URL {
  // Prefer the configured public URL and fall back to the production default.
  const candidate = process.env.NEXT_PUBLIC_SITE_URL?.trim() || DEFAULT_SITE_URL;

  try {
    return new URL(candidate);
  } catch {
    // Recover from malformed environment values with the known-good default URL.
    return new URL(DEFAULT_SITE_URL);
  }
}
