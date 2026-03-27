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
    diffBefore,
  }: {
    code: string;
    diffBefore?: string | null;
  }) => <pre>{diffBefore ? `${diffBefore}\n${code}` : code}</pre>,
}));

function createTrial(overrides: Partial<TrialListItem>): TrialListItem {
  return {
    trialId: "trial_1",
    status: "finished",
    outcomeReason: "completed",
    modalRunId: null,
    modalRunUrl: null,
    score: 0.91,
    accuracy: 0.91,
    bestEvalEpoch: 3,
    epochsCompleted: 5,
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
    errorType: null,
    source: "print('hello')\n",
    taskDescription: null,
    responseText: null,
    reasoningText: null,
    generatedSource: null,
    generationAssertionsPassed: null,
    generationAssertionFailures: [],
    errorJson: null,
    provenanceJson: { backend: "openrouter", request_messages: [] },
    ...overrides,
  };
}

const tracks: TrackListItem[] = [
  {
    trackId: "track_1",
    datasetId: "mnist:v1",
    createdAt: "2026-03-20T14:00:00.000Z",
    totalTrials: 2,
    queuedTrials: 0,
    dispatchingTrials: 0,
    activeTrials: 0,
    finishedTrials: 2,
    errorTrials: 0,
    succeededTrials: 2,
    bestScore: 0.9342,
    bestTrialId: "trial_2",
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
  tracks?: TrackListItem[];
}) {
  navigationState.pathname =
    options?.pathname ??
    (options?.initialSelectedTrialId
      ? `/tracks/track_1/trials/${options.initialSelectedTrialId}`
      : "/tracks/track_1");
  navigationState.replace.mockReset();
  const initialTracks = options?.tracks ?? tracks;

  return render(
    <DashboardShell
      initialDetail={options?.detail ?? createDetail()}
      initialTracks={initialTracks}
      initialSelectedTrialId={options?.initialSelectedTrialId ?? null}
      selectedTrackId="track_1"
    />,
  );
}

function toggleSection(title: string) {
  fireEvent.click(screen.getByRole("button", { name: new RegExp(title, "i") }));
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

  it("adds hover and focus tooltip text to progress breakdown segments", () => {
    const customTrack: TrackListItem = {
      ...tracks[0],
      totalTrials: 6,
      queuedTrials: 1,
      dispatchingTrials: 1,
      activeTrials: 2,
      finishedTrials: 1,
      errorTrials: 1,
    };

    const { container } = render(
      <DashboardShell
        initialDetail={{
          track: customTrack,
          trials: baseTrials,
          nextCursor: null,
        }}
        initialTracks={[customTrack]}
        initialSelectedTrialId={null}
        selectedTrackId="track_1"
      />,
    );

    const activeSegment = container.querySelector(".progress-segment.active");
    expect(activeSegment?.getAttribute("title")).toBe("Running: 2");
    expect(activeSegment?.getAttribute("data-tooltip")).toBe("Running: 2");
    expect(activeSegment?.getAttribute("aria-label")).toBe("Running: 2");
    expect(activeSegment?.getAttribute("tabindex")).toBe("0");
  });

  it("renders the sidebar progress bar with the same segmented status breakdown", () => {
    const customTrack: TrackListItem = {
      ...tracks[0],
      totalTrials: 6,
      queuedTrials: 1,
      dispatchingTrials: 1,
      activeTrials: 2,
      finishedTrials: 1,
      errorTrials: 1,
    };

    const { container } = render(
      <DashboardShell
        initialDetail={{
          track: customTrack,
          trials: baseTrials,
          nextCursor: null,
        }}
        initialTracks={[customTrack]}
        initialSelectedTrialId={null}
        selectedTrackId="track_1"
      />,
    );

    const sidebarSegments = container.querySelectorAll(".track-card-bar .progress-segment");
    expect(sidebarSegments).toHaveLength(5);

    const dispatchingSegment = container.querySelector(".track-card-bar .progress-segment.dispatching");
    expect(dispatchingSegment?.getAttribute("title")).toBe("Dispatching: 1");
    expect(dispatchingSegment?.getAttribute("data-tooltip")).toBe("Dispatching: 1");
    expect(dispatchingSegment?.getAttribute("aria-label")).toBe("Dispatching: 1");
    expect(dispatchingSegment?.getAttribute("tabindex")).toBe("0");
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

  it("does not show a launcher badge in the table when the trial is not selected", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_modal",
          modalRunId: "fc-123",
          modalRunUrl: "https://modal.com/apps/test/runs/fc-123",
        }),
      ]),
    });

    expect(screen.queryByRole("link", { name: "Open launcher run for trial_modal" })).toBeNull();
  });

  it("renders note badges in the trial cell and removes the notes column", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_noted",
          outcomeReason: "duplicate",
          errorType: "generation_failed",
          timedOut: true,
          hadUnscoredWorkAtTimeout: true,
          hasError: true,
        }),
      ]),
    });

    expect(screen.queryByRole("columnheader", { name: "Notes" })).toBeNull();

    const row = screen.getByRole("button", { name: "Open trial trial_noted" });
    expect(row.textContent).toContain("noted");
    expect(row.textContent).toContain("duplicate");
    expect(row.textContent).toContain("generation_failed");
    expect(row.textContent).toContain("timed out");
    expect(row.textContent).toContain("unevaluated work");
    expect(row.textContent).toContain("error payload");
  });

  it("shows the task description in the trials table when recorded", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_tasked",
          taskDescription:
            "Tune the training loop to improve validation accuracy without destabilizing the early stopping policy.",
        }),
      ]),
    });

    expect(screen.getByRole("columnheader", { name: "Task" })).toBeTruthy();
    expect(
      screen.getByText(
        "Tune the training loop to improve validation accuracy without destabilizing the early stopping policy.",
      ),
    ).toBeTruthy();
  });

  it("shows launcher and wandb badges in the inspector when the selected trial has remote run URLs", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_modal",
          modalRunId: "fc-123",
          modalRunUrl: "https://modal.com/apps/test/runs/fc-123",
          provenanceJson: {
            backend: "openrouter",
            request_messages: [],
            wandb: {
              run_url: "https://wandb.ai/tsilva/sigmaevolve/runs/knro5y92",
            },
          },
        }),
      ]),
      initialSelectedTrialId: "trial_modal",
    });

    const launcherLink = screen.getByRole("link", { name: "Open launcher run for trial_modal" });
    expect(launcherLink.getAttribute("href")).toBe("https://modal.com/apps/test/runs/fc-123");
    expect(launcherLink.textContent).toBe("Launcher");

    const wandbLink = screen.getByRole("link", { name: "Open Weights & Biases run for trial_modal" });
    expect(wandbLink.getAttribute("href")).toBe("https://wandb.ai/tsilva/sigmaevolve/runs/knro5y92");
    expect(wandbLink.textContent).toBe("W&B");
  });

  it("embeds the prompt-source diff into the generated program when snippets are recorded", () => {
    const detail = createDetail([
      createTrial({
        trialId: "trial_diff",
        source: "print('new candidate')\n",
        generatedSource: "print('new candidate')\n",
        provenanceJson: {
          backend: "openrouter",
          request_messages: [
            {
              role: "user",
              content: [
                "Use this parent trial as the base candidate:",
                "```python",
                "print('old parent')",
                "```",
                "",
                "Avoid the failure modes seen in these recent negative trials:",
                "```python",
                "print('bad candidate')",
                "```",
              ].join("\n"),
            },
          ],
        },
      }),
    ]);

    renderShell({
      detail,
      initialSelectedTrialId: "trial_diff",
      pathname: "/tracks/track_1/trials/trial_diff",
    });

    expect(screen.getByText("Generated program")).toBeTruthy();
    expect(screen.getByText("2 prompt sources • +1 / -3 inline diff")).toBeTruthy();
    expect(screen.getByText(/print\('new candidate'\)/)).toBeTruthy();
    expect(screen.getByText(/print\('old parent'\)/)).toBeTruthy();
    expect(screen.getByText(/print\('bad candidate'\)/)).toBeTruthy();
  });

  it("diffs against the current program instead of reference programs when the prompt is structured", () => {
    const detail = createDetail([
      createTrial({
        trialId: "trial_structured_diff",
        source: ["def train():", "    return 'improved'", ""].join("\n"),
        generatedSource: null,
        provenanceJson: {
          backend: "openrouter",
          request_messages: [
            {
              role: "user",
              content: [
                "OBJECTIVE:",
                "- Propose a safe improvement to CURRENT PROGRAM.",
                "REFERENCE PROGRAMS:",
                "```python",
                "def train():",
                "    return 'reference'",
                "```",
                "CURRENT PROGRAM:",
                "Patch this program. SEARCH blocks must match text from CURRENT PROGRAM, not from REFERENCE PROGRAMS.",
                "```python",
                "def train():",
                "    return 'baseline'",
                "```",
                "REPLACEMENTS:",
              ].join("\n"),
            },
          ],
        },
      }),
    ]);

    renderShell({
      detail,
      initialSelectedTrialId: "trial_structured_diff",
      pathname: "/tracks/track_1/trials/trial_structured_diff",
    });

    expect(screen.getByText("2 prompt sources • diffing CURRENT PROGRAM • +1 / -1 inline diff")).toBeTruthy();
    expect(screen.queryByText("return 'reference'")).toBeNull();
    expect(screen.getByText(/return 'baseline'/)).toBeTruthy();
    expect(screen.getByText(/return 'improved'/)).toBeTruthy();
  });

  it("renders generation trace fields and diagnostic-source fallback for failed generation attempts", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_generation_failed",
          outcomeReason: "generation_failed",
          hasError: true,
          source: "# diagnostic stub\n",
          responseText: null,
          generatedSource: "print('attempted candidate')\n",
          generationAssertionsPassed: false,
          generationAssertionFailures: ["candidate modified immutable text outside evolve blocks"],
          provenanceJson: {
            backend: "openrouter",
            request_messages: [],
            generation: {
              system_prompt: "system prompt text",
              user_prompt: "user prompt text",
            },
          },
        }),
      ]),
      initialSelectedTrialId: "trial_generation_failed",
    });

    toggleSection("System prompt");
    toggleSection("User prompt");
    toggleSection("Response");
    toggleSection("Reasoning trace");

    expect(screen.getByText("system prompt text")).toBeTruthy();
    expect(screen.getByText("user prompt text")).toBeTruthy();
    expect(screen.getByText("No task description recorded.")).toBeTruthy();
    expect(screen.getByText("No raw response recorded.")).toBeTruthy();
    expect(screen.getByText("No reasoning trace recorded.")).toBeTruthy();
    expect(screen.getByText("Generation attempt")).toBeTruthy();
    expect(screen.queryByText("Generated program")).toBeNull();
    expect(screen.queryByText("Mixed vs generated diff")).toBeNull();
    expect(screen.getByText("print('attempted candidate')")).toBeTruthy();
    expect(screen.getByText("candidate modified immutable text outside evolve blocks")).toBeTruthy();
    expect(screen.getByText(/stored row source is diagnostic-only/i)).toBeTruthy();
  });

  it("hides empty crash detail and generation assertion rows when nothing was recorded", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_without_optional_timeline_fields",
          errorJson: null,
          generationAssertionsPassed: null,
          generationAssertionFailures: [],
        }),
      ]),
      initialSelectedTrialId: "trial_without_optional_timeline_fields",
    });

    expect(screen.queryByText("Crash detail")).toBeNull();
    expect(screen.queryByText("Generation assertions")).toBeNull();
    expect(screen.queryByText("No crash detail recorded.")).toBeNull();
    expect(screen.queryByText("Not recorded")).toBeNull();
  });

  it("shows the persisted raw LLM response for successful trials", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_success_raw_response",
          responseText:
            "<<<<<<< SEARCH\nprint('hello')\n=======\nprint('hello world')\n>>>>>>> REPLACE",
        }),
      ]),
      initialSelectedTrialId: "trial_success_raw_response",
    });

    toggleSection("Response");

    expect(screen.getByText("Response")).toBeTruthy();
    expect(screen.getByText(/<<<<<<< SEARCH/)).toBeTruthy();
    expect(screen.queryByText("No raw response recorded.")).toBeNull();
  });

  it("shows the extracted task description when recorded", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_task_description",
          taskDescription: "Broaden the hidden layers to add capacity without changing the training loop.",
        }),
      ]),
      initialSelectedTrialId: "trial_task_description",
    });

    expect(screen.getByText("Task description")).toBeTruthy();
    expect(
      screen.getByText("Broaden the hidden layers to add capacity without changing the training loop."),
    ).toBeTruthy();
    expect(screen.queryByText("No task description recorded.")).toBeNull();
  });

  it("shows the persisted reasoning trace when recorded", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_reasoning_trace",
          reasoningText: "The model compared two candidate patches before choosing one.",
        }),
      ]),
      initialSelectedTrialId: "trial_reasoning_trace",
    });

    toggleSection("Reasoning trace");

    expect(screen.getByText("Reasoning trace")).toBeTruthy();
    expect(screen.getByText("The model compared two candidate patches before choosing one.")).toBeTruthy();
    expect(screen.queryByText("No reasoning trace recorded.")).toBeNull();
  });

  it("shows a safe note when the provider returns an encrypted reasoning payload", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_encrypted_reasoning_trace",
          reasoningText:
            "Reasoning trace unavailable. Provider returned encrypted reasoning blocks (google-gemini-v1).",
        }),
      ]),
      initialSelectedTrialId: "trial_encrypted_reasoning_trace",
    });

    toggleSection("Reasoning trace");

    expect(
      screen.getByText(
        "Reasoning trace unavailable. Provider returned encrypted reasoning blocks (google-gemini-v1).",
      ),
    ).toBeTruthy();
    expect(
      screen.queryByText("AY89a19Jsq7xtfbhrjynErTNjrdDbmfbe3gDcIH4rvFJEp195oIBbTyfgiQ1/5l2oko="),
    ).toBeNull();
  });

  it("puts error payload first in the inspector card stack when recorded", () => {
    const { container } = renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_error",
          hasError: true,
          errorJson: {
            kind: "generation_failed",
            message: "candidate parse failed",
          },
        }),
      ]),
      initialSelectedTrialId: "trial_error",
    });

    const inspectorGrid = container.querySelector(".inspector-grid");
    const cardHeadings = Array.from(inspectorGrid?.querySelectorAll("h3") ?? []).map((heading) => heading.textContent);

    expect(cardHeadings[0]).toBe("Error payload");
    expect(cardHeadings[1]).toBe("Task description");
    expect(cardHeadings[2]).toBe("Generated program");
  });

  it("keeps the error payload expanded whenever it is present", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_error_expanded",
          hasError: true,
          errorJson: {
            kind: "generation_failed",
            message: "candidate parse failed",
          },
        }),
      ]),
      initialSelectedTrialId: "trial_error_expanded",
    });

    expect(screen.queryByRole("button", { name: /^error payload$/i })).toBeNull();
    expect(screen.getByText(/"message": "candidate parse failed"/)).toBeTruthy();
  });

  it("shows reasoning trace before response in the inspector card stack", () => {
    const { container } = renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_reasoning_before_response",
          reasoningText: "Reasoning content",
          responseText: "Response content",
        }),
      ]),
      initialSelectedTrialId: "trial_reasoning_before_response",
    });

    const cardHeadings = Array.from(container.querySelectorAll(".inspector-grid h3")).map(
      (heading) => heading.textContent,
    );
    const reasoningIndex = cardHeadings.indexOf("Reasoning trace");
    const responseIndex = cardHeadings.indexOf("Response");

    expect(reasoningIndex).toBeGreaterThan(-1);
    expect(responseIndex).toBeGreaterThan(-1);
    expect(reasoningIndex).toBeLessThan(responseIndex);
  });

  it("merges the assertion status and empty failure state into one row", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_generation_passed",
          generationAssertionsPassed: true,
          generationAssertionFailures: [],
        }),
      ]),
      initialSelectedTrialId: "trial_generation_passed",
    });

    expect(screen.getByText("Generation assertions")).toBeTruthy();
    expect(screen.getByText("Passed")).toBeTruthy();
    expect(screen.queryByText("Assertion failures")).toBeNull();
    expect(screen.queryByText("No assertion failures recorded.")).toBeNull();
  });

  it("renders provenance as property rows instead of a raw JSON block", () => {
    const { container } = renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_provenance",
          provenanceJson: {
            backend: "openrouter",
            model: "x-ai/grok-4.1-fast",
            generation_config: {
              model: "x-ai/grok-4.1-fast",
              temperature: 0.2,
              max_tokens: 1200,
              retry_count: 1,
            },
            context_trial_ids: ["trial_parent_a", "trial_parent_b"],
          },
        }),
      ]),
      initialSelectedTrialId: "trial_provenance",
    });

    expect(screen.getByText("Generation provenance")).toBeTruthy();
    const subsectionLabels = Array.from(container.querySelectorAll(".trial-summary-subsection-label")).map(
      (element) => element.textContent,
    );
    expect(subsectionLabels).toContain("Model");
    expect(subsectionLabels).not.toContain("Launcher");
    expect(screen.getByText("Config Temperature")).toBeTruthy();
    expect(screen.getByText("0.2")).toBeTruthy();
    expect(screen.getByText("Current Program Trial")).toBeTruthy();
    const currentProgramTrialLink = screen.getByRole("link", { name: "trial_parent_a" });
    expect(currentProgramTrialLink.getAttribute("href")).toBe("/tracks/track_1/trials/trial_parent_a");
    expect(screen.getByText("Reference Program Trials")).toBeTruthy();
    const referenceProgramTrialLink = screen.getByRole("link", { name: "trial_parent_b" });
    expect(referenceProgramTrialLink.getAttribute("href")).toBe("/tracks/track_1/trials/trial_parent_b");
    expect(screen.queryByText("Context Trials")).toBeNull();
    expect(screen.queryByText('"generation_config"')).toBeNull();
  });

  it("shows launcher badges instead of launcher provenance URL rows", () => {
    const { container } = renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_launcher",
          modalRunId: "fc-123",
          modalRunUrl: "https://modal.com/apps/test/runs/fc-123",
          provenanceJson: {
            backend: "openrouter",
            model: "x-ai/grok-4.1-fast",
            launcher: {
              run_id: "fc-123",
              run_url: "https://modal.com/apps/test/runs/fc-123",
            },
          },
        }),
      ]),
      initialSelectedTrialId: "trial_launcher",
    });

    const subsectionLabels = Array.from(container.querySelectorAll(".trial-summary-subsection-label")).map(
      (element) => element.textContent,
    );
    expect(subsectionLabels).not.toContain("Launcher");
    expect(screen.queryByText("Launcher Run Id")).toBeNull();
    expect(screen.queryByText("fc-123")).toBeNull();
    expect(screen.queryByText("Launcher Run Url")).toBeNull();
    const launcherLink = screen.getByRole("link", { name: "Open launcher run for trial_launcher" });
    expect(launcherLink.getAttribute("href")).toBe("https://modal.com/apps/test/runs/fc-123");
    expect(launcherLink.textContent).toBe("Launcher");
  });

  it("hides wandb identifiers from the trial provenance panel and promotes the run url to a badge", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_wandb",
          provenanceJson: {
            backend: "openrouter",
            model: "x-ai/grok-4.1-fast",
            wandb_project: "sigmaevolve",
            wandb_entity: "tsilva",
            wandb_run_id: "0oosq56x",
            wandb_run_name: "track_2bfe1dc3071342489daddd272b6dd807:trial_ae821512cb4041d0a0609b5ddca63ca7",
            wandb: {
              run_url: "https://wandb.ai/tsilva/sigmaevolve/runs/knro5y92",
            },
            extra_context: "still visible",
          },
        }),
      ]),
      initialSelectedTrialId: "trial_wandb",
    });

    expect(screen.queryByText("Wandb Project")).toBeNull();
    expect(screen.queryByText("sigmaevolve")).toBeNull();
    expect(screen.queryByText("Wandb Entity")).toBeNull();
    expect(screen.queryByText("tsilva")).toBeNull();
    expect(screen.queryByText("Wandb Run Id")).toBeNull();
    expect(screen.queryByText("0oosq56x")).toBeNull();
    expect(screen.queryByText("Wandb Run Name")).toBeNull();
    expect(screen.queryByText("Wandb Run Url")).toBeNull();
    expect(
      screen.queryByText("track_2bfe1dc3071342489daddd272b6dd807:trial_ae821512cb4041d0a0609b5ddca63ca7"),
    ).toBeNull();
    expect(screen.getByText("Extra Context")).toBeTruthy();
    expect(screen.getByText("still visible")).toBeTruthy();
    const wandbLink = screen.getByRole("link", { name: "Open Weights & Biases run for trial_wandb" });
    expect(wandbLink.getAttribute("href")).toBe("https://wandb.ai/tsilva/sigmaevolve/runs/knro5y92");
    expect(wandbLink.textContent).toBe("W&B");
  });

  it("keeps the summary sections open and defaults only task description and generated program to expanded", () => {
    renderShell({
      detail: createDetail([
        createTrial({
          trialId: "trial_fixed_sections",
          provenanceJson: null,
        }),
      ]),
      initialSelectedTrialId: "trial_fixed_sections",
    });

    expect(screen.queryByRole("button", { name: /^overview$/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /^metrics$/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /^run timeline$/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /^generation provenance$/i })).toBeNull();
    expect(screen.getByText("Trial ID")).toBeTruthy();
    expect(screen.getByText("Dispatch Attempts")).toBeTruthy();
    expect(screen.getByText("Queued")).toBeTruthy();
    expect(screen.getByText("No provenance payload recorded.")).toBeTruthy();
    expect(screen.getByText("No task description recorded.")).toBeTruthy();
    expect(screen.getByText("print('hello')")).toBeTruthy();
    expect(screen.queryByText("No raw response recorded.")).toBeNull();

    toggleSection("Response");

    expect(screen.getByText("No raw response recorded.")).toBeTruthy();
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

  it("uses the trials-table compact id style for research lane ids", () => {
    renderShell({
      tracks: [
        {
          ...tracks[0],
          trackId: "track_1234567890abcdef",
        },
      ],
    });

    expect(screen.getByText("123def")).toBeTruthy();
    expect(screen.getByTitle("track_1234567890abcdef")).toBeTruthy();
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

  it("highlights the best-so-far trial in the explorer and score chart", () => {
    const { container } = renderShell();

    const bestRow = screen.getByRole("button", { name: "Open trial trial_2" });
    expect(bestRow.className).toContain("best-trial");
    expect(container.querySelector('circle.score-point.best-point[aria-label^="trial_2"]')).toBeTruthy();
    expect(screen.getByText(/trial_2 is the best trial so far\./i)).toBeTruthy();
    expect(screen.getByText("best so far")).toBeTruthy();
  });

  it("shows status as the first table column and stacks duration below the badge", () => {
    const { container } = renderShell();

    const headers = Array.from(container.querySelectorAll(".trial-table thead th")).map((cell) => cell.textContent);
    expect(headers).toEqual(["Status", "Trial", "Task", "Score", "val_acc", "Best Epoch", "Model"]);

    const firstStatusCell = container.querySelector(".trial-table tbody td");
    expect(firstStatusCell?.querySelector(".status-badge")?.textContent).toContain("finished");
    expect(firstStatusCell?.querySelector(".trial-status-duration")?.textContent).toBe("1m 0s");
  });

  it("shows best epoch in best/total format in the trials table", () => {
    renderShell();

    expect(screen.getAllByText("3/5")).toHaveLength(2);
  });

  it("highlights the best-so-far trial in the inspector", () => {
    renderShell({
      initialSelectedTrialId: "trial_2",
    });

    expect(screen.getByRole("heading", { name: "trial_2" })).toBeTruthy();
    expect(screen.getByText("best so far")).toBeTruthy();
  });

  it("shows trial details when a score point is hovered", () => {
    const { container } = renderShell();

    const points = Array.from(container.querySelectorAll("circle.score-point"));
    expect(points.length).toBeGreaterThan(0);

    fireEvent.mouseEnter(points[0]);

    const tooltip = container.querySelector(".score-point-tooltip");
    expect(tooltip?.textContent).toContain("trial_1");
    expect(tooltip?.textContent).toContain("Score: 0.9123");
    expect(tooltip?.textContent).toContain("Model: google/gemini");
  });

  it("zooms the score chart when low outliers flatten a tight high-score cluster", () => {
    const clusteredTrials = [
      createTrial({
        trialId: "trial_low",
        status: "error",
        outcomeReason: "failed",
        score: 0,
        accuracy: null,
        createdAt: "2026-03-20T15:00:00.000Z",
      }),
      createTrial({
        trialId: "trial_a",
        score: 0.9962,
        accuracy: 0.9962,
        createdAt: "2026-03-20T15:01:00.000Z",
      }),
      createTrial({
        trialId: "trial_b",
        score: 0.9968,
        accuracy: 0.9968,
        createdAt: "2026-03-20T15:02:00.000Z",
      }),
      createTrial({
        trialId: "trial_c",
        score: 0.9974,
        accuracy: 0.9974,
        createdAt: "2026-03-20T15:03:00.000Z",
      }),
    ];

    const { container } = renderShell({
      detail: createDetail(clusteredTrials),
    });

    expect(screen.getByText("1 lower outlier pinned to the baseline")).toBeTruthy();
    expect(container.querySelector(".score-axis-break")).toBeTruthy();

    const getPointY = (trialId: string) =>
      Number(
        container
          .querySelector(`circle.score-point[aria-label^="${trialId}"]`)
          ?.getAttribute("cy"),
      );

    const lowPointY = getPointY("trial_low");
    const highClusterYs = ["trial_a", "trial_b", "trial_c"].map(getPointY);

    expect(Math.max(...highClusterYs) - Math.min(...highClusterYs)).toBeGreaterThan(20);
    expect(lowPointY).toBeGreaterThan(Math.max(...highClusterYs));
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
