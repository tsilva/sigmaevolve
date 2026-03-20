const DEFAULT_SITE_URL = "https://sigmaevolve.vercel.app";

export function getSiteUrl(): URL {
  const candidate = process.env.NEXT_PUBLIC_SITE_URL?.trim() || DEFAULT_SITE_URL;

  try {
    return new URL(candidate);
  } catch {
    return new URL(DEFAULT_SITE_URL);
  }
}
