import { DashboardShell } from "@/components/dashboard-shell";
import { EmptyState } from "@/components/empty-state";
import { getTrackDetailOrThrow, listTrackSummaries } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default async function TrackPage({
  params,
}: {
  params: Promise<{ trackId: string }>;
}) {
  const { trackId } = await params;
  const [tracks, detail] = await Promise.all([listTrackSummaries(), getTrackDetailOrThrow(trackId)]);

  if (tracks.length === 0) {
    return <EmptyState />;
  }

  return <DashboardShell initialTracks={tracks} initialDetail={detail} selectedTrackId={trackId} />;
}
