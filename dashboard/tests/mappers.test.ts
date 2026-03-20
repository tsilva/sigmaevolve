import { mapTrackListItem, mapTrialListItem } from "@/lib/mappers";

describe("dashboard row mappers", () => {
  it("maps aggregated track rows into track list items", () => {
    const mapped = mapTrackListItem({
      trackId: "track_1",
      name: "mnist-baseline",
      datasetId: "mnist:v1",
      createdAt: "2026-03-20T15:00:00.000Z",
      totalTrials: "5",
      queuedTrials: "1",
      dispatchingTrials: "1",
      activeTrials: "0",
      finishedTrials: "3",
      succeededTrials: "2",
      bestScore: "0.9321",
      lastActivityAt: "2026-03-20T15:10:00.000Z",
    });

    expect(mapped).toEqual({
      trackId: "track_1",
      name: "mnist-baseline",
      datasetId: "mnist:v1",
      createdAt: "2026-03-20T15:00:00.000Z",
      totalTrials: 5,
      queuedTrials: 1,
      dispatchingTrials: 1,
      activeTrials: 0,
      finishedTrials: 3,
      succeededTrials: 2,
      bestScore: 0.9321,
      lastActivityAt: "2026-03-20T15:10:00.000Z",
    });
  });

  it("maps trial rows with missing metrics and errors", () => {
    const mapped = mapTrialListItem({
      trialId: "trial_1",
      status: "active",
      outcomeReason: null,
      score: "0",
      accuracy: null,
      timeToBestEvalSec: null,
      timedOut: false,
      timeSinceLastEvalSec: null,
      hadUnscoredWorkAtTimeout: false,
      lastPhase: null,
      backend: "openrouter",
      model: "google/gemini",
      dispatchAttempts: "2",
      createdAt: "2026-03-20T15:00:00.000Z",
      startedAt: null,
      finishedAt: null,
      durationSec: null,
      hasError: true,
      source: "print('hello')\n",
      errorJson: { stderr: "boom" },
    });

    expect(mapped).toEqual({
      trialId: "trial_1",
      status: "active",
      outcomeReason: null,
      score: 0,
      accuracy: null,
      timeToBestEvalSec: null,
      timedOut: false,
      timeSinceLastEvalSec: null,
      hadUnscoredWorkAtTimeout: false,
      lastPhase: null,
      backend: "openrouter",
      model: "google/gemini",
      dispatchAttempts: 2,
      createdAt: "2026-03-20T15:00:00.000Z",
      startedAt: null,
      finishedAt: null,
      durationSec: null,
      hasError: true,
      source: "print('hello')\n",
      errorJson: { stderr: "boom" },
    });
  });
});
