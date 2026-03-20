import { redirect } from "next/navigation";

import { EmptyState } from "@/components/empty-state";
import { getNewestTrackId } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const newestTrackId = await getNewestTrackId();

  if (!newestTrackId) {
    return <EmptyState />;
  }

  redirect(`/tracks/${newestTrackId}`);
}
