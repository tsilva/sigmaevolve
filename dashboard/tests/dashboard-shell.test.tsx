// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { DashboardShell } from "@/components/dashboard-shell";
import type { TrackDetailResponse, TrackListItem, TrialListItem } from "@/lib/types";

const navigationState = vi.hoisted(() => ({
  pathname: "/tracks/track_1",
  replace: vi.fn(),
  search: "",
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    className,
    href,
  }: {
    children: React.ReactNode;
    className?: string;
    href: string;
  }) => (
    <a className={className} href={href}>
      {children}
    </a>
  ),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => navigationState.pathname,
  useRouter: () => ({
    replace: navigationState.replace,
  }),
  useSearchParams: () => new URLSearchParams(navigationState.search),
}));

vi.mock("@/hooks/use-track-live-updates", () => ({
  useTrackLiveUpdates: () => "stream",
}));

vi.mock("@/components/highlighted-code", () => ({
  HighlightedCode: ({
    code,
  }: {
    code: string;
  }) => <pre>{code}</pre>,
}));

function createTrial(overrides: Partial<TrialListItem>): TrialListItem {
  return {
    trialId: "trial_1",
    status: "finished",
    outcomeReason: "completed",
    score: 0.91,
    accuracy: 0.91,
    timeToBestEvalSec: 12,
    timedOut: false,
    timeSinceLastEvalSec: 4,
    hadUnscoredWorkAtTimeout: false,
    lastPhase: "eval",
    backend: "openrouter",
    model: "google/gemini",
    dispatchAttempts: 1,
    createdAt: "2026-03-20T15:00:00.000Z",
    startedAt: "2026-03-20T15:01:00.000Z",
    finishedAt: "2026-03-20T15:02:00.000Z",
    durationSec: 60,
    hasError: false,
    source: "print('hello')\n",
    errorJson: null,
    provenanceJson: { backend: "openrouter", request_messages: [] },
    ...overrides,
  };
}

const tracks: TrackListItem[] = [
  {
    trackId: "track_1",
    name: "mnist-baseline",
    datasetId: "mnist:v1",
    createdAt: "2026-03-20T14:00:00.000Z",
    totalTrials: 3,
    queuedTrials: 0,
    dispatchingTrials: 0,
    activeTrials: 0,
    finishedTrials: 3,
    succeededTrials: 3,
    bestScore: 0.9342,
    lastActivityAt: "2026-03-20T15:10:00.000Z",
  },
];

const baseTrials: TrialListItem[] = [
  createTrial({
    trialId: "trial_2",
    score: 0.9342,
    accuracy: 0.9342,
    createdAt: "2026-03-20T15:10:00.000Z",
  }),
  createTrial({
    trialId: "trial_1",
    score: 0.9123,
    accuracy: 0.9123,
    createdAt: "2026-03-20T15:00:00.000Z",
  }),
];

function createDetail(trials: TrialListItem[] = baseTrials): TrackDetailResponse {
  return {
    track: tracks[0],
    trials,
    nextCursor: null,
  };
}

function renderShell(options?: {
  detail?: TrackDetailResponse;
  initialSelectedTrialId?: string | null;
  search?: string;
}) {
  navigationState.search = options?.search ?? "";
  navigationState.replace.mockReset();

  return render(
    <DashboardShell
      initialDetail={options?.detail ?? createDetail()}
      initialTracks={tracks}
      initialSelectedTrialId={options?.initialSelectedTrialId ?? null}
      selectedTrackId="track_1"
    />,
  );
}

describe("DashboardShell", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    navigationState.pathname = "/tracks/track_1";
    navigationState.search = "";
    navigationState.replace.mockReset();
    globalThis.fetch = vi.fn();
  });

  it("auto-selects the newest visible trial when no trial param is provided", async () => {
    renderShell();

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1?trial=trial_2", { scroll: false });
    });

    expect(screen.getByRole("heading", { name: "trial_2" })).toBeTruthy();
  });

  it("respects a valid trial param on first render", () => {
    renderShell({
      initialSelectedTrialId: "trial_1",
      search: "trial=trial_1",
    });

    expect(screen.getByRole("heading", { name: "trial_1" })).toBeTruthy();
    expect(navigationState.replace).not.toHaveBeenCalled();
  });

  it("falls back to the newest visible trial when the trial param is invalid", async () => {
    renderShell({
      initialSelectedTrialId: "missing_trial",
      search: "trial=missing_trial",
    });

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1?trial=trial_2", { scroll: false });
    });

    expect(screen.getByRole("heading", { name: "trial_2" })).toBeTruthy();
  });

  it("updates the selected trial and URL when a user clicks another trial", async () => {
    renderShell({
      initialSelectedTrialId: "trial_2",
      search: "trial=trial_2",
    });

    fireEvent.click(screen.getByRole("button", { name: "Select trial trial_1" }));

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1?trial=trial_1", { scroll: false });
    });

    expect(screen.getByRole("heading", { name: "trial_1" })).toBeTruthy();
  });

  it("falls back to the first visible filtered trial when the current trial is filtered out", async () => {
    const queuedTrials = [
      createTrial({
        trialId: "trial_queued",
        status: "queued",
        outcomeReason: null,
        score: 0,
        accuracy: null,
      }),
    ];

    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ trials: queuedTrials, nextCursor: null }),
    } as Response);

    renderShell({
      initialSelectedTrialId: "trial_2",
      search: "trial=trial_2",
    });

    fireEvent.click(screen.getByRole("button", { name: "queued" }));

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1?trial=trial_queued", { scroll: false });
    });

    expect(screen.getByRole("heading", { name: "trial_queued" })).toBeTruthy();
  });

  it("clears the detail pane when a filter returns no trials", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ trials: [], nextCursor: null }),
    } as Response);

    renderShell({
      initialSelectedTrialId: "trial_2",
      search: "trial=trial_2",
    });

    fireEvent.click(screen.getByRole("button", { name: "queued" }));

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1", { scroll: false });
    });

    expect(screen.getByText("Detail pane cleared")).toBeTruthy();
  });
});
