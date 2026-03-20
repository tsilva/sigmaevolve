import type { MetadataRoute } from "next";

import { getSiteUrl } from "@/lib/site";
import { hasDatabaseUrl } from "@/lib/db";
import { listTrackSummaries } from "@/lib/queries";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const siteUrl = getSiteUrl();
  const entries: MetadataRoute.Sitemap = [
    {
      url: new URL("/", siteUrl).toString(),
      lastModified: new Date(),
      changeFrequency: "daily",
      priority: 1,
    },
  ];

  if (!hasDatabaseUrl()) {
    return entries;
  }

  try {
    const tracks = await listTrackSummaries();
    return entries.concat(
      tracks.map((track) => ({
        url: new URL(`/tracks/${track.trackId}`, siteUrl).toString(),
        lastModified: track.lastActivityAt,
        changeFrequency: "hourly",
        priority: 0.8,
      })),
    );
  } catch {
    return entries;
  }
}
