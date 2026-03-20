// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { DashboardShell } from "@/components/dashboard-shell";
import type { TrackDetailResponse, TrackListItem, TrialListItem } from "@/lib/types";

const navigationState = vi.hoisted(() => ({
  pathname: "/tracks/track_1",
  replace: vi.fn(),
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
    totalTrials: 2,
    queuedTrials: 0,
    dispatchingTrials: 0,
    activeTrials: 0,
    finishedTrials: 2,
    succeededTrials: 2,
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
  pathname?: string;
}) {
  navigationState.pathname =
    options?.pathname ??
    (options?.initialSelectedTrialId
      ? `/tracks/track_1/trials/${options.initialSelectedTrialId}`
      : "/tracks/track_1");
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
    navigationState.replace.mockReset();
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/tracks")) {
        return {
          ok: true,
          json: async () => tracks,
        } as Response;
      }
      if (url.includes("/api/tracks/track_1/trials?")) {
        return {
          ok: true,
          json: async () => ({ trials: baseTrials, nextCursor: null }),
        } as Response;
      }
      throw new Error(`Unexpected fetch call: ${url}`);
    });
  });

  it("auto-selects the newest visible trial when no trial param is provided", async () => {
    renderShell();

    await waitFor(() => {
      expect(screen.getByText("How each run went")).toBeTruthy();
    });

    expect(navigationState.replace).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Open trial trial_2" })).toBeTruthy();
  });

  it("respects a valid trial param on first render", () => {
    renderShell({
      initialSelectedTrialId: "trial_1",
    });

    expect(screen.getByRole("heading", { name: "trial_1" })).toBeTruthy();
    expect(navigationState.replace).not.toHaveBeenCalled();
  });

  it("falls back to the newest visible trial when the trial param is invalid", async () => {
    renderShell({
      initialSelectedTrialId: "missing_trial",
      pathname: "/tracks/track_1/trials/missing_trial",
    });

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1/trials/trial_2", { scroll: false });
    });

    expect(screen.getByRole("heading", { name: "trial_2" })).toBeTruthy();
  });

  it("updates the selected trial and URL when a user clicks another trial", async () => {
    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "Open trial trial_1" }));

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1/trials/trial_1", { scroll: false });
    });

    expect(screen.getByRole("heading", { name: "trial_1" })).toBeTruthy();
    expect(screen.queryByText("How each run went")).toBeNull();
  });

  it("keeps the explorer visible when a filter changes the visible trial set", async () => {
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
    });

    fireEvent.click(screen.getByRole("button", { name: "queued" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Open trial trial_queued" })).toBeTruthy();
    });

    expect(navigationState.replace).not.toHaveBeenCalled();
    expect(screen.getByText("How each run went")).toBeTruthy();
  });

  it("keeps the inspector mounted for the selected trial", async () => {
    renderShell({
      initialSelectedTrialId: "trial_2",
    });

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "trial_2" })).toBeTruthy();
    });

    expect(screen.getByText("Why the selected run behaved that way")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Close detail panel" })).toBeNull();
  });

  it("returns from the inspector to the trial explorer", async () => {
    renderShell({
      initialSelectedTrialId: "trial_2",
    });

    fireEvent.click(screen.getByRole("button", { name: "Back to trial explorer" }));

    await waitFor(() => {
      expect(navigationState.replace).toHaveBeenCalledWith("/tracks/track_1", { scroll: false });
    });

    expect(screen.getByText("How each run went")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "trial_2" })).toBeNull();
  });

  it("collapses and re-expands the tracks sidebar", () => {
    const { container } = renderShell({
      initialSelectedTrialId: "trial_2",
    });

    fireEvent.click(screen.getByRole("button", { name: "Collapse tracks sidebar" }));

    expect(screen.queryByRole("heading", { name: "Research lanes" })).toBeNull();
    expect(screen.getByRole("button", { name: "Expand tracks sidebar" })).toBeTruthy();
    expect(container.querySelector("main")?.className).toContain("tracks-collapsed");

    fireEvent.click(screen.getByRole("button", { name: "Expand tracks sidebar" }));

    expect(screen.getByRole("heading", { name: "Research lanes" })).toBeTruthy();
  });

  it("clears the detail pane when a filter returns no trials", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ trials: [], nextCursor: null }),
    } as Response);

    renderShell();

    fireEvent.click(screen.getByRole("button", { name: "queued" }));

    await waitFor(() => {
      expect(screen.getByText("Nothing matches the current filter.")).toBeTruthy();
    });

    expect(navigationState.replace).not.toHaveBeenCalled();
  });

  it("renders a score history chart for the trials currently displayed in the table", () => {
    const { container } = renderShell();

    expect(screen.getByRole("img", { name: "Score history for the trials currently displayed in the table" })).toBeTruthy();
    expect(screen.getByText("Score History")).toBeTruthy();
    expect(container.querySelectorAll("circle.score-point").length).toBe(baseTrials.length);
  });

  it("updates the score history chart when the visible table rows change", async () => {
    const { container } = renderShell();

    expect(container.querySelectorAll("circle.score-point").length).toBe(2);

    fireEvent.change(screen.getByRole("searchbox"), {
      target: { value: "trial_1" },
    });

    await waitFor(() => {
      expect(container.querySelectorAll("circle.score-point").length).toBe(1);
    });

    expect(screen.getByText("1 scored / 1 displayed")).toBeTruthy();
  });
});
