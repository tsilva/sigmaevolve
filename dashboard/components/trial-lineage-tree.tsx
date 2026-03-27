"use client";

import type { TrialLineageGraph, TrialLineageNode } from "@/lib/trial-lineage";
import type { TrackListItem } from "@/lib/types";

type TrialLineageTreeProps = {
  bestTrialId: string | null;
  graph: TrialLineageGraph;
  isLoading?: boolean;
  onOpenTrial: (trialId: string) => void;
  selectedTrialId: string | null;
  track: TrackListItem;
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

function formatStatusLabel(value: string): string {
  return value === "active" ? "running" : value;
}

function getNodeTone(node: TrialLineageNode): "success" | "warning" | "danger" | "neutral" {
  const trial = node.trial;
  if (trial.status === "error") {
    return "danger";
  }
  if (trial.status !== "finished") {
    return "neutral";
  }
  if (trial.outcomeReason === "duplicate") {
    return "warning";
  }
  if (trial.hasError || trial.timedOut || trial.hadUnscoredWorkAtTimeout) {
    return "warning";
  }
  return "success";
}

function renderTrialIdChips(trialIds: string[], onOpenTrial: (trialId: string) => void) {
  return (
    <span className="property-chip-list">
      {trialIds.map((trialId) => (
        <button
          key={trialId}
          type="button"
          className="meta-chip meta-chip-mono trial-tree-link-chip"
          onClick={() => onOpenTrial(trialId)}
        >
          {trialId}
        </button>
      ))}
    </span>
  );
}

function TreeNode({
  bestTrialId,
  node,
  onOpenTrial,
  selectedTrialId,
}: {
  bestTrialId: string | null;
  node: TrialLineageNode;
  onOpenTrial: (trialId: string) => void;
  selectedTrialId: string | null;
}) {
  const tone = getNodeTone(node);
  const isBest = node.trial.trialId === bestTrialId;
  const isSelected = node.trial.trialId === selectedTrialId;

  return (
    <li className="trial-tree-item">
      <article
        className={`trial-tree-node tone-${tone} ${isBest ? "best-trial" : ""} ${isSelected ? "active" : ""}`.trim()}
      >
        <div className="trial-tree-node-header">
          <button
            type="button"
            className="trial-tree-node-button"
            onClick={() => onOpenTrial(node.trial.trialId)}
            aria-label={`Open lineage node ${node.trial.trialId}`}
          >
            <span className="trial-tree-node-title-row">
              <span className="trial-tree-node-title" title={node.trial.trialId}>
                {formatShortTrialId(node.trial.trialId)}
              </span>
              {isBest ? <span className="flag-chip flag-best">best so far</span> : null}
            </span>
            <span className="trial-tree-node-copy">
              {node.trial.model ?? "unknown model"} · score {formatNumber(node.trial.accuracy ?? node.trial.score)}
            </span>
          </button>

          <div className="trial-tree-node-chips">
            <span className={`status-badge status-${node.trial.status}`}>
              <span className={`status-indicator ${node.trial.status}`} />
              {formatStatusLabel(node.trial.status)}
            </span>
            {node.trial.outcomeReason ? <span className="flag-chip">{node.trial.outcomeReason}</span> : null}
          </div>
        </div>

        <div className="trial-tree-node-stats">
          <span className="meta-chip">{node.directForkCount} direct fork{node.directForkCount === 1 ? "" : "s"}</span>
          <span className="meta-chip">{node.descendantCount} descendant{node.descendantCount === 1 ? "" : "s"}</span>
          <span className="meta-chip">
            {node.inspirationUseTrialIds.length} inspiration use{node.inspirationUseTrialIds.length === 1 ? "" : "s"}
          </span>
        </div>

        <div className="trial-tree-node-relations">
          {node.hasMissingParent && node.parentTrialId ? (
            <div className="trial-tree-relation-row">
              <span className="trial-tree-relation-label">Forked from</span>
              <span className="meta-chip meta-chip-mono">{node.parentTrialId}</span>
            </div>
          ) : null}

          {node.inspirationTrialIds.length > 0 ? (
            <div className="trial-tree-relation-row">
              <span className="trial-tree-relation-label">Inspired by</span>
              {renderTrialIdChips(node.inspirationTrialIds, onOpenTrial)}
            </div>
          ) : null}

          {node.inspirationUseTrialIds.length > 0 ? (
            <div className="trial-tree-relation-row">
              <span className="trial-tree-relation-label">Inspires</span>
              {renderTrialIdChips(node.inspirationUseTrialIds, onOpenTrial)}
            </div>
          ) : null}
        </div>
      </article>

      {node.children.length > 0 ? (
        <ol className="trial-tree-list trial-tree-children">
          {node.children.map((child) => (
            <TreeNode
              key={child.trial.trialId}
              bestTrialId={bestTrialId}
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
  bestTrialId,
  graph,
  isLoading = false,
  onOpenTrial,
  selectedTrialId,
  track,
}: TrialLineageTreeProps) {
  return (
    <section className="workspace-card tree-panel">
      <div className="section-heading">
        <div className="eyebrow">Lineage Tree</div>
        <h2 className="section-title">How candidates fork and cross-pollinate</h2>
        <p className="section-copy">
          Each branch follows the current-program parent. Inspiration links call out the side references that shaped a
          candidate without becoming its direct parent.
        </p>
      </div>

      <div className="hero-metrics tree-metrics">
        <article className="metric-tile">
          <span className="metric-label">Loaded Trials</span>
          <strong className="metric-value">{graph.nodeCount}</strong>
          <span className="metric-note">
            {graph.nodeCount === track.totalTrials
              ? "Whole track loaded."
              : `${track.totalTrials - graph.nodeCount} older trial${track.totalTrials - graph.nodeCount === 1 ? "" : "s"} still loading.`}
          </span>
        </article>
        <article className="metric-tile">
          <span className="metric-label">Roots</span>
          <strong className="metric-value">{graph.roots.length}</strong>
          <span className="metric-note">Independent starting points or nodes with missing parents.</span>
        </article>
        <article className="metric-tile">
          <span className="metric-label">Fork Edges</span>
          <strong className="metric-value">{graph.forkEdgeCount}</strong>
          <span className="metric-note">Direct parent-to-child branches in the tree.</span>
        </article>
        <article className="metric-tile">
          <span className="metric-label">Inspiration Links</span>
          <strong className="metric-value">{graph.inspirationEdgeCount}</strong>
          <span className="metric-note">Cross-links recorded as inspiration context.</span>
        </article>
      </div>

      {isLoading ? <div className="tree-loading-banner">Loading the remaining lineage nodes…</div> : null}

      {graph.roots.length === 0 ? (
        <section className="empty-panel">
          <div className="eyebrow">No trials</div>
          <h3>This track does not have any lineage to render yet.</h3>
          <p className="section-copy">Queue more trials and the tree will fill in automatically.</p>
        </section>
      ) : (
        <div className="trial-tree-shell">
          <ol className="trial-tree-list">
            {graph.roots.map((root) => (
              <TreeNode
                key={root.trial.trialId}
                bestTrialId={bestTrialId}
                node={root}
                onOpenTrial={onOpenTrial}
                selectedTrialId={selectedTrialId}
              />
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
