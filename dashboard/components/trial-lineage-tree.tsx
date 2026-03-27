"use client";

import type { TrialLineageGraph, TrialLineageNode } from "@/lib/trial-lineage";

type TrialLineageTreeProps = {
  graph: TrialLineageGraph;
  isLoading?: boolean;
  onOpenTrial: (trialId: string) => void;
  selectedTrialId: string | null;
};

function formatNumber(value: number | null, digits = 4): string {
  if (value === null) {
    return "—";
  }

  return value.toFixed(digits);
}

function formatCompactEntityId(value: string, prefix?: string): string {
  const trimmed = prefix && value.startsWith(prefix) ? value.slice(prefix.length) : value;
  if (trimmed.length <= 6) {
    return trimmed;
  }

  return `${trimmed.slice(0, 3)}${trimmed.slice(-3)}`;
}

function formatShortTrialId(value: string): string {
  return formatCompactEntityId(value, "trial_");
}

function summarizeTaskDescription(value: string | null, maxLength = 72): string {
  if (!value) {
    return "No task description";
  }

  const collapsed = value.replace(/\s+/g, " ").trim();
  if (collapsed.length === 0) {
    return "No task description";
  }

  if (collapsed.length <= maxLength) {
    return collapsed;
  }

  return `${collapsed.slice(0, maxLength - 1).trimEnd()}…`;
}

function TreeNode({
  node,
  onOpenTrial,
  selectedTrialId,
}: {
  node: TrialLineageNode;
  onOpenTrial: (trialId: string) => void;
  selectedTrialId: string | null;
}) {
  const isSelected = node.trial.trialId === selectedTrialId;
  const taskSummary = summarizeTaskDescription(node.trial.taskDescription);

  return (
    <li className="trial-tree-item">
      <button
        type="button"
        className={`trial-tree-node status-${node.trial.status} ${isSelected ? "active" : ""}`.trim()}
        onClick={() => onOpenTrial(node.trial.trialId)}
        aria-label={`Open lineage node ${node.trial.trialId}`}
      >
        <span className="trial-tree-node-title" title={node.trial.trialId}>
          {formatShortTrialId(node.trial.trialId)}
        </span>
        <span className="trial-tree-node-score">{formatNumber(node.trial.accuracy ?? node.trial.score)}</span>
        <span className="trial-tree-node-status">{node.trial.status}</span>
        <span className="trial-tree-node-task" title={node.trial.taskDescription ?? "No task description"}>
          {taskSummary}
        </span>
      </button>

      {node.children.length > 0 ? (
        <ol className="trial-tree-list trial-tree-children">
          {node.children.map((child) => (
            <TreeNode
              key={child.trial.trialId}
              node={child}
              onOpenTrial={onOpenTrial}
              selectedTrialId={selectedTrialId}
            />
          ))}
        </ol>
      ) : null}
    </li>
  );
}

export function TrialLineageTree({
  graph,
  isLoading = false,
  onOpenTrial,
  selectedTrialId,
}: TrialLineageTreeProps) {
  if (graph.roots.length === 0) {
    return (
      <section className="workspace-card tree-panel">
        <section className="empty-panel">
          <div className="eyebrow">No trials</div>
          <h3>This track does not have any lineage to render yet.</h3>
        </section>
      </section>
    );
  }

  return (
    <section className="workspace-card tree-panel">
      {isLoading ? <div className="tree-loading-banner">Loading the remaining lineage nodes…</div> : null}
      <div className="trial-tree-shell">
        <ol className="trial-tree-list">
          {graph.roots.map((root) => (
            <TreeNode
              key={root.trial.trialId}
              node={root}
              onOpenTrial={onOpenTrial}
              selectedTrialId={selectedTrialId}
            />
          ))}
        </ol>
      </div>
    </section>
  );
}
