import { DashboardShell } from "@/components/dashboard-shell";
import { EmptyState } from "@/components/empty-state";
import { getTrackDetailOrThrow, listTrackSummaries } from "@/lib/queries";

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
  const [tracks, detail] = await Promise.all([listTrackSummaries(), getTrackDetailOrThrow(trackId)]);

  if (tracks.length === 0) {
    return <EmptyState />;
  }

  return (
    <DashboardShell
      initialTracks={tracks}
      initialDetail={detail}
      selectedTrackId={trackId}
      initialSelectedTrialId={trial ?? null}
    />
  );
}
