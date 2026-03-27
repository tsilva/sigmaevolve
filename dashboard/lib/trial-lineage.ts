import type { TrialListItem } from "@/lib/types";

export type TrialLineageNode = {
  children: TrialLineageNode[];
  descendantCount: number;
  directForkCount: number;
  hasMissingParent: boolean;
  inspirationTrialIds: string[];
  inspirationUseTrialIds: string[];
  parentTrialId: string | null;
  trial: TrialListItem;
};

export type TrialLineageGraph = {
  forkEdgeCount: number;
  inspirationEdgeCount: number;
  nodeCount: number;
  roots: TrialLineageNode[];
};

type TrialLineageDraft = {
  children: TrialLineageDraft[];
  descendantCount: number;
  directForkCount: number;
  hasMissingParent: boolean;
  inspirationTrialIds: string[];
  inspirationUseTrialIds: string[];
  parentTrialId: string | null;
  trial: TrialListItem;
};

function asTrialIdList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((entry): entry is string => typeof entry === "string" && entry.trim().length > 0);
}

function getParentTrialId(trial: TrialListItem): string | null {
  const contextTrialIds = asTrialIdList(trial.provenanceJson?.context_trial_ids);
  if (contextTrialIds.length > 0) {
    return contextTrialIds[0];
  }

  const parentTrialIds = asTrialIdList(trial.provenanceJson?.parent_trial_ids);
  return parentTrialIds[0] ?? null;
}

function getInspirationTrialIds(trial: TrialListItem): string[] {
  const contextTrialIds = asTrialIdList(trial.provenanceJson?.context_trial_ids);
  return contextTrialIds.length > 1 ? contextTrialIds.slice(1) : [];
}

function compareTrials(left: TrialListItem, right: TrialListItem): number {
  const createdAtDelta = new Date(left.createdAt).getTime() - new Date(right.createdAt).getTime();
  if (createdAtDelta !== 0) {
    return createdAtDelta;
  }

  return left.trialId.localeCompare(right.trialId);
}

function compareNodes(left: TrialLineageDraft, right: TrialLineageDraft): number {
  return compareTrials(left.trial, right.trial);
}

function finalizeNode(node: TrialLineageDraft): TrialLineageNode {
  const children = node.children
    .sort(compareNodes)
    .map((child) => finalizeNode(child));
  const descendantCount = children.reduce((count, child) => count + 1 + child.descendantCount, 0);

  return {
    children,
    descendantCount,
    directForkCount: children.length,
    hasMissingParent: node.hasMissingParent,
    inspirationTrialIds: [...node.inspirationTrialIds],
    inspirationUseTrialIds: [...node.inspirationUseTrialIds].sort(),
    parentTrialId: node.parentTrialId,
    trial: node.trial,
  };
}

export function buildTrialLineageGraph(trials: TrialListItem[]): TrialLineageGraph {
  const sortedTrials = [...trials].sort(compareTrials);
  const drafts = new Map<string, TrialLineageDraft>();

  for (const trial of sortedTrials) {
    drafts.set(trial.trialId, {
      children: [],
      descendantCount: 0,
      directForkCount: 0,
      hasMissingParent: false,
      inspirationTrialIds: getInspirationTrialIds(trial),
      inspirationUseTrialIds: [],
      parentTrialId: getParentTrialId(trial),
      trial,
    });
  }

  let forkEdgeCount = 0;
  let inspirationEdgeCount = 0;

  for (const node of drafts.values()) {
    if (node.parentTrialId && node.parentTrialId !== node.trial.trialId) {
      const parent = drafts.get(node.parentTrialId);
      if (parent) {
        parent.children.push(node);
        forkEdgeCount += 1;
      } else {
        node.hasMissingParent = true;
      }
    }

    for (const inspirationTrialId of node.inspirationTrialIds) {
      if (inspirationTrialId === node.trial.trialId) {
        continue;
      }

      const inspirationSource = drafts.get(inspirationTrialId);
      if (!inspirationSource) {
        continue;
      }

      inspirationSource.inspirationUseTrialIds.push(node.trial.trialId);
      inspirationEdgeCount += 1;
    }
  }

  const roots = Array.from(drafts.values())
    .filter((node) => {
      if (!node.parentTrialId || node.parentTrialId === node.trial.trialId) {
        return true;
      }

      return !drafts.has(node.parentTrialId);
    })
    .sort(compareNodes)
    .map((node) => finalizeNode(node));

  return {
    forkEdgeCount,
    inspirationEdgeCount,
    nodeCount: sortedTrials.length,
    roots,
  };
}
