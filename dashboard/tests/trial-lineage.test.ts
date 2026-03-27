import { buildTrialLineageGraph } from "@/lib/trial-lineage";
import type { TrialListItem } from "@/lib/types";

function createTrial(overrides: Partial<TrialListItem>): TrialListItem {
  return {
    trialId: "trial_1",
    status: "finished",
    outcomeReason: "succeeded",
    modalRunId: null,
    modalRunUrl: null,
    score: 0.9,
    accuracy: 0.9,
    bestEvalEpoch: null,
    epochsCompleted: null,
    timeToBestEvalSec: null,
    timedOut: false,
    timeSinceLastEvalSec: null,
    hadUnscoredWorkAtTimeout: false,
    lastPhase: "finished",
    backend: "openrouter",
    model: "test/model",
    dispatchAttempts: 1,
    createdAt: "2026-03-20T15:00:00.000Z",
    startedAt: null,
    finishedAt: null,
    durationSec: null,
    hasError: false,
    errorType: null,
    source: "print('ok')\n",
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

describe("buildTrialLineageGraph", () => {
  it("builds a fork tree from current-program parents and tracks inspiration links", () => {
    const graph = buildTrialLineageGraph([
      createTrial({
        trialId: "trial_root",
        backend: "baseline",
        model: "baseline",
        createdAt: "2026-03-20T15:00:00.000Z",
        provenanceJson: { backend: "baseline", parent_trial_ids: [] },
      }),
      createTrial({
        trialId: "trial_child_a",
        createdAt: "2026-03-20T15:01:00.000Z",
        provenanceJson: {
          backend: "openrouter",
          request_messages: [],
          context_trial_ids: ["trial_root"],
        },
      }),
      createTrial({
        trialId: "trial_child_b",
        createdAt: "2026-03-20T15:02:00.000Z",
        provenanceJson: {
          backend: "openrouter",
          request_messages: [],
          context_trial_ids: ["trial_root", "trial_child_a"],
        },
      }),
      createTrial({
        trialId: "trial_grandchild",
        createdAt: "2026-03-20T15:03:00.000Z",
        provenanceJson: {
          backend: "openrouter",
          request_messages: [],
          context_trial_ids: ["trial_child_a"],
        },
      }),
    ]);

    expect(graph.nodeCount).toBe(4);
    expect(graph.forkEdgeCount).toBe(3);
    expect(graph.inspirationEdgeCount).toBe(1);
    expect(graph.roots).toHaveLength(1);

    const root = graph.roots[0];
    expect(root.trial.trialId).toBe("trial_root");
    expect(root.directForkCount).toBe(2);
    expect(root.descendantCount).toBe(3);
    expect(root.children.map((child) => child.trial.trialId)).toEqual(["trial_child_a", "trial_child_b"]);

    const childA = root.children[0];
    expect(childA.inspirationUseTrialIds).toEqual(["trial_child_b"]);
    expect(childA.directForkCount).toBe(1);
    expect(childA.children[0].trial.trialId).toBe("trial_grandchild");
  });

  it("keeps nodes with missing parents as disconnected roots", () => {
    const graph = buildTrialLineageGraph([
      createTrial({
        trialId: "trial_orphan",
        createdAt: "2026-03-20T15:04:00.000Z",
        provenanceJson: {
          backend: "openrouter",
          request_messages: [],
          context_trial_ids: ["trial_missing"],
        },
      }),
    ]);

    expect(graph.roots).toHaveLength(1);
    expect(graph.roots[0].trial.trialId).toBe("trial_orphan");
    expect(graph.roots[0].hasMissingParent).toBe(true);
    expect(graph.forkEdgeCount).toBe(0);
  });
});
