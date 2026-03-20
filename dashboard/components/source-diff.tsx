type DiffRow =
  | {
      kind: "context" | "add" | "remove";
      beforeLineNumber: number | null;
      afterLineNumber: number | null;
      content: string;
    }
  | {
      kind: "skip";
      hiddenCount: number;
    };

type SourceDiffProps = {
  before: string;
  after: string;
};

function splitLines(value: string): string[] {
  const normalized = value.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!normalized) {
    return [];
  }

  const trimmed = normalized.endsWith("\n") ? normalized.slice(0, -1) : normalized;
  if (!trimmed) {
    return [];
  }

  return trimmed.split("\n");
}

function buildDiffRows(before: string, after: string): DiffRow[] {
  const beforeLines = splitLines(before);
  const afterLines = splitLines(after);
  const matrix = Array.from({ length: beforeLines.length + 1 }, () => Array(afterLines.length + 1).fill(0));

  for (let beforeIndex = beforeLines.length - 1; beforeIndex >= 0; beforeIndex -= 1) {
    for (let afterIndex = afterLines.length - 1; afterIndex >= 0; afterIndex -= 1) {
      if (beforeLines[beforeIndex] === afterLines[afterIndex]) {
        matrix[beforeIndex][afterIndex] = matrix[beforeIndex + 1][afterIndex + 1] + 1;
      } else {
        matrix[beforeIndex][afterIndex] = Math.max(matrix[beforeIndex + 1][afterIndex], matrix[beforeIndex][afterIndex + 1]);
      }
    }
  }

  const rows: DiffRow[] = [];
  let beforeIndex = 0;
  let afterIndex = 0;
  let beforeLineNumber = 1;
  let afterLineNumber = 1;

  while (beforeIndex < beforeLines.length && afterIndex < afterLines.length) {
    if (beforeLines[beforeIndex] === afterLines[afterIndex]) {
      rows.push({
        kind: "context",
        beforeLineNumber,
        afterLineNumber,
        content: beforeLines[beforeIndex],
      });
      beforeIndex += 1;
      afterIndex += 1;
      beforeLineNumber += 1;
      afterLineNumber += 1;
      continue;
    }

    if (matrix[beforeIndex + 1][afterIndex] >= matrix[beforeIndex][afterIndex + 1]) {
      rows.push({
        kind: "remove",
        beforeLineNumber,
        afterLineNumber: null,
        content: beforeLines[beforeIndex],
      });
      beforeIndex += 1;
      beforeLineNumber += 1;
      continue;
    }

    rows.push({
      kind: "add",
      beforeLineNumber: null,
      afterLineNumber,
      content: afterLines[afterIndex],
    });
    afterIndex += 1;
    afterLineNumber += 1;
  }

  while (beforeIndex < beforeLines.length) {
    rows.push({
      kind: "remove",
      beforeLineNumber,
      afterLineNumber: null,
      content: beforeLines[beforeIndex],
    });
    beforeIndex += 1;
    beforeLineNumber += 1;
  }

  while (afterIndex < afterLines.length) {
    rows.push({
      kind: "add",
      beforeLineNumber: null,
      afterLineNumber,
      content: afterLines[afterIndex],
    });
    afterIndex += 1;
    afterLineNumber += 1;
  }

  const changedIndexes = rows.flatMap((row, index) => (row.kind === "context" ? [] : [index]));
  if (changedIndexes.length === 0) {
    return rows;
  }

  const visibleIndexes = new Set<number>();
  for (const index of changedIndexes) {
    for (let windowIndex = Math.max(0, index - 3); windowIndex <= Math.min(rows.length - 1, index + 3); windowIndex += 1) {
      visibleIndexes.add(windowIndex);
    }
  }

  const collapsed: DiffRow[] = [];
  let hiddenCount = 0;
  for (let index = 0; index < rows.length; index += 1) {
    if (!visibleIndexes.has(index)) {
      hiddenCount += 1;
      continue;
    }

    if (hiddenCount > 0) {
      collapsed.push({
        kind: "skip",
        hiddenCount,
      });
      hiddenCount = 0;
    }

    collapsed.push(rows[index]);
  }

  if (hiddenCount > 0) {
    collapsed.push({
      kind: "skip",
      hiddenCount,
    });
  }

  return collapsed;
}

function countRows(rows: DiffRow[]): { added: number; removed: number } {
  let added = 0;
  let removed = 0;

  for (const row of rows) {
    if (row.kind === "add") {
      added += 1;
    }
    if (row.kind === "remove") {
      removed += 1;
    }
  }

  return { added, removed };
}

function formatLineNumber(value: number | null): string {
  return value === null ? "" : String(value);
}

export function SourceDiff({ before, after }: SourceDiffProps) {
  const rows = buildDiffRows(before, after);
  const { added, removed } = countRows(rows);

  if (added === 0 && removed === 0) {
    return <p className="section-copy">No line-level differences detected between the mixed input sources and the generated script.</p>;
  }

  return (
    <div className="source-diff-shell">
      <div className="source-diff-summary">
        <span className="meta-chip">+{added} additions</span>
        <span className="meta-chip">-{removed} removals</span>
      </div>
      <pre className="code-block source-diff-block">
        <code>
          {rows.map((row, index) => {
            if (row.kind === "skip") {
              return (
                <span key={`skip-${index}`} className="source-diff-row source-diff-skip">
                  <span className="source-diff-gutter" aria-hidden="true">
                    …
                  </span>
                  <span className="source-diff-line-number" aria-hidden="true" />
                  <span className="source-diff-line-number" aria-hidden="true" />
                  <span className="source-diff-content">{`${row.hiddenCount} unchanged lines hidden`}</span>
                </span>
              );
            }

            const marker = row.kind === "add" ? "+" : row.kind === "remove" ? "-" : " ";
            return (
              <span key={`${row.kind}-${row.beforeLineNumber ?? "x"}-${row.afterLineNumber ?? "x"}-${index}`} className={`source-diff-row source-diff-${row.kind}`}>
                <span className="source-diff-gutter" aria-hidden="true">
                  {marker}
                </span>
                <span className="source-diff-line-number" aria-hidden="true">
                  {formatLineNumber(row.beforeLineNumber)}
                </span>
                <span className="source-diff-line-number" aria-hidden="true">
                  {formatLineNumber(row.afterLineNumber)}
                </span>
                <span className="source-diff-content">{row.content || " "}</span>
              </span>
            );
          })}
        </code>
      </pre>
    </div>
  );
}
