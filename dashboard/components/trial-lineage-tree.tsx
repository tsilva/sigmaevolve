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

function findNode(nodes: TrialLineageNode[], trialId: string | null): TrialLineageNode | null {
  if (!trialId) {
    return null;
  }

  for (const node of nodes) {
    if (node.trial.trialId === trialId) {
      return node;
    }

    const child = findNode(node.children, trialId);
    if (child) {
      return child;
    }
  }

  return null;
}

function countNodes(nodes: TrialLineageNode[]): number {
  return nodes.reduce((total, node) => total + 1 + countNodes(node.children), 0);
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
  const selectedNode = findNode(graph.roots, selectedTrialId);
  const selectedTrial = selectedNode?.trial ?? null;
  const nodeCount = countNodes(graph.roots);

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
      <div className="lineage-header">
        <div>
          <h2 className="lineage-title">Lineage Map</h2>
          <p className="section-copy">Explore how each trial descends from its parent, then open a node to inspect source, prompts, and outcomes.</p>
        </div>
        <div className="hero-meta">
          <span className="meta-chip">{nodeCount} trials</span>
          <span className="meta-chip live-chip">Best path</span>
          <span className="meta-chip danger-chip">Failure branches</span>
        </div>
      </div>
      {isLoading ? <div className="tree-loading-banner">Loading the remaining lineage nodes…</div> : null}
      <div className="lineage-workspace">
        <div className="trial-tree-shell">
          <div className="lineage-toolbar" aria-label="Lineage filters">
            <span className="filter-chip active">All {nodeCount}</span>
            <span className="filter-chip">Finished</span>
            <span className="filter-chip">Error</span>
            <span className="filter-chip">Duplicate</span>
            <span className="filter-chip">Fit view</span>
          </div>
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
        <aside className="lineage-inspector">
          <div className="inspector-label">Selected node</div>
          {selectedTrial ? (
            <>
              <h3 title={selectedTrial.trialId}>node {formatShortTrialId(selectedTrial.trialId)}</h3>
              <strong className="lineage-score">{formatNumber(selectedTrial.accuracy ?? selectedTrial.score)}</strong>
              <div className="context-stack">
                <div className="context-row">
                  <span>Status</span>
                  <strong>status {selectedTrial.status}</strong>
                </div>
                <div className="context-row">
                  <span>Model</span>
                  <strong>{selectedTrial.model ?? "unknown model"}</strong>
                </div>
                <div className="context-row">
                  <span>Backend</span>
                  <strong>{selectedTrial.backend ?? "unknown backend"}</strong>
                </div>
              </div>
              <p className="section-copy" title={selectedTrial.taskDescription ?? "No task description"}>
                Task summary: {summarizeTaskDescription(selectedTrial.taskDescription, 180)}
              </p>
              <button type="button" className="load-more lineage-open" onClick={() => onOpenTrial(selectedTrial.trialId)}>
                Open in Inspector
              </button>
            </>
          ) : (
            <p className="section-copy">Select a trial node to inspect score, model, and provenance.</p>
          )}
        </aside>
      </div>
    </section>
  );
}
