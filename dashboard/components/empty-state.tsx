export function EmptyState() {
  return (
    <main className="empty-state panel">
      <div className="eyebrow">SigmaEvolve</div>
      <h1>No tracks yet.</h1>
      <p>
        The dashboard is live, but the shared Postgres database does not have any experiment tracks yet.
        Create a track from the Python harness and this view will populate automatically.
      </p>
    </main>
  );
}
