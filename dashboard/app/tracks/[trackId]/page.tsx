import { redirect } from "next/navigation";

import { DashboardShell } from "@/components/dashboard-shell";
import { EmptyState } from "@/components/empty-state";
import { getTrackDetailOrThrow, listTrackSummaries } from "@/lib/queries";
import { mono } from "../fonts";

export const dynamic = "force-dynamic";

export default async function TrackPage({
  params,
  searchParams,
}: {
  params: Promise<{ trackId: string }>;
  searchParams: Promise<{ trial?: string }>;
}) {
  const { trackId } = await params;
  const { trial } = await searchParams;

  if (trial) {
    redirect(`/tracks/${trackId}/trials/${encodeURIComponent(trial)}`);
  }

  const [tracks, detail] = await Promise.all([listTrackSummaries(), getTrackDetailOrThrow(trackId)]);

  if (tracks.length === 0) {
    return <EmptyState />;
  }

  return (
    <DashboardShell
      className={mono.variable}
      initialTracks={tracks}
      initialDetail={detail}
      selectedTrackId={trackId}
      initialSelectedTrialId={trial ?? null}
    />
  );
}
