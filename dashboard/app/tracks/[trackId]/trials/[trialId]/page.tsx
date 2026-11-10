import { DashboardShell } from "@/components/dashboard-shell";
import { EmptyState } from "@/components/empty-state";
import { getTrackDetailOrThrow, listTrackSummaries } from "@/lib/queries";
import { mono } from "../../../fonts";

export const dynamic = "force-dynamic";

export default async function TrackTrialPage({
  params,
}: {
  params: Promise<{ trackId: string; trialId: string }>;
}) {
  const { trackId, trialId } = await params;
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
      initialSelectedTrialId={trialId}
    />
  );
}
