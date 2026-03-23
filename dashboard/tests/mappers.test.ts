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
      errorTrials: "1",
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
      errorTrials: 1,
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
      modalRunId: null,
      modalRunUrl: null,
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
      errorType: "generation_invalid_candidate",
      source: "print('hello')\n",
      errorJson: { stderr: "boom" },
      provenanceJson: {
        model: "google/gemini",
        request_messages: [],
        generation: {
          response_text: "raw response",
          generated_source: "print('candidate')\n",
          assertions_passed: false,
          assertion_failures: ["candidate modified immutable text outside evolve blocks"],
        },
      },
    });

    expect(mapped).toEqual({
      trialId: "trial_1",
      status: "active",
      outcomeReason: null,
      modalRunId: null,
      modalRunUrl: null,
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
      errorType: "generation_invalid_candidate",
      source: "print('hello')\n",
      responseText: "raw response",
      generatedSource: "print('candidate')\n",
      generationAssertionsPassed: false,
      generationAssertionFailures: ["candidate modified immutable text outside evolve blocks"],
      errorJson: { stderr: "boom" },
      provenanceJson: {
        model: "google/gemini",
        request_messages: [],
        generation: {
          response_text: "raw response",
          generated_source: "print('candidate')\n",
          assertions_passed: false,
          assertion_failures: ["candidate modified immutable text outside evolve blocks"],
        },
      },
    });
  });

  it("does not flag successful diagnostics as an execution error", () => {
    const mapped = mapTrialListItem({
      trialId: "trial_success",
      status: "finished",
      outcomeReason: "succeeded",
      modalRunId: "fc-123",
      modalRunUrl: "https://modal.com/apps/test/runs/fc-123",
      score: "0.927",
      accuracy: "0.927",
      timeToBestEvalSec: "1.97",
      timedOut: false,
      timeSinceLastEvalSec: "4.19",
      hadUnscoredWorkAtTimeout: false,
      lastPhase: "finished",
      backend: "baseline",
      model: "linear-classifier",
      dispatchAttempts: "1",
      createdAt: "2026-03-20T15:00:00.000Z",
      startedAt: "2026-03-20T15:01:00.000Z",
      finishedAt: "2026-03-20T15:02:00.000Z",
      durationSec: "60",
      hasError: false,
      errorType: null,
      source: "print('ok')\n",
      errorJson: { stderr: "", eval_artifacts: ["/tmp/eval_0001.npz"] },
      provenanceJson: { model: "baseline", request_messages: [] },
    });

    expect(mapped.hasError).toBe(false);
    expect(mapped.modalRunUrl).toBe("https://modal.com/apps/test/runs/fc-123");
    expect(mapped.errorJson).toEqual({ stderr: "", eval_artifacts: ["/tmp/eval_0001.npz"] });
    expect(mapped.responseText).toBeNull();
    expect(mapped.generatedSource).toBeNull();
    expect(mapped.generationAssertionsPassed).toBeNull();
    expect(mapped.generationAssertionFailures).toEqual([]);
  });

  it("ignores a truthy hasError row value when the payload has no error signal", () => {
    const mapped = mapTrialListItem({
      trialId: "trial_success_string_flag",
      status: "finished",
      outcomeReason: "succeeded",
      modalRunId: null,
      modalRunUrl: null,
      score: "0.927",
      accuracy: "0.927",
      timeToBestEvalSec: "1.97",
      timedOut: false,
      timeSinceLastEvalSec: "4.19",
      hadUnscoredWorkAtTimeout: false,
      lastPhase: "finished",
      backend: "baseline",
      model: "linear-classifier",
      dispatchAttempts: "1",
      createdAt: "2026-03-20T15:00:00.000Z",
      startedAt: "2026-03-20T15:01:00.000Z",
      finishedAt: "2026-03-20T15:02:00.000Z",
      durationSec: "60",
      hasError: "f" as unknown as boolean,
      errorType: null,
      source: "print('ok')\n",
      errorJson: { stderr: "", eval_artifacts: ["/tmp/eval_0001.npz"] },
      provenanceJson: { model: "baseline", request_messages: [] },
    });

    expect(mapped.hasError).toBe(false);
  });
});
