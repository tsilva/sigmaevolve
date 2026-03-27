export type SourceDiffRow = {
  kind: "context" | "add" | "remove";
  beforeLineIndex: number | null;
  beforeLineNumber: number | null;
  afterLineIndex: number | null;
  afterLineNumber: number | null;
  content: string;
};

export type SourceDiffSummary = {
  added: number;
  removed: number;
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

export function buildSourceDiffRows(before: string, after: string): SourceDiffRow[] {
  const beforeLines = splitLines(before);
  const afterLines = splitLines(after);
  const matrix = Array.from({ length: beforeLines.length + 1 }, () =>
    Array(afterLines.length + 1).fill(0),
  );

  for (let beforeIndex = beforeLines.length - 1; beforeIndex >= 0; beforeIndex -= 1) {
    for (let afterIndex = afterLines.length - 1; afterIndex >= 0; afterIndex -= 1) {
      if (beforeLines[beforeIndex] === afterLines[afterIndex]) {
        matrix[beforeIndex][afterIndex] = matrix[beforeIndex + 1][afterIndex + 1] + 1;
      } else {
        matrix[beforeIndex][afterIndex] = Math.max(
          matrix[beforeIndex + 1][afterIndex],
          matrix[beforeIndex][afterIndex + 1],
        );
      }
    }
  }

  const rows: SourceDiffRow[] = [];
  let beforeIndex = 0;
  let afterIndex = 0;
  let beforeLineNumber = 1;
  let afterLineNumber = 1;

  while (beforeIndex < beforeLines.length && afterIndex < afterLines.length) {
    if (beforeLines[beforeIndex] === afterLines[afterIndex]) {
      rows.push({
        kind: "context",
        beforeLineIndex: beforeIndex,
        beforeLineNumber,
        afterLineIndex: afterIndex,
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
        beforeLineIndex: beforeIndex,
        beforeLineNumber,
        afterLineIndex: null,
        afterLineNumber: null,
        content: beforeLines[beforeIndex],
      });
      beforeIndex += 1;
      beforeLineNumber += 1;
      continue;
    }

    rows.push({
      kind: "add",
      beforeLineIndex: null,
      beforeLineNumber: null,
      afterLineIndex: afterIndex,
      afterLineNumber,
      content: afterLines[afterIndex],
    });
    afterIndex += 1;
    afterLineNumber += 1;
  }

  while (beforeIndex < beforeLines.length) {
    rows.push({
      kind: "remove",
      beforeLineIndex: beforeIndex,
      beforeLineNumber,
      afterLineIndex: null,
      afterLineNumber: null,
      content: beforeLines[beforeIndex],
    });
    beforeIndex += 1;
    beforeLineNumber += 1;
  }

  while (afterIndex < afterLines.length) {
    rows.push({
      kind: "add",
      beforeLineIndex: null,
      beforeLineNumber: null,
      afterLineIndex: afterIndex,
      afterLineNumber,
      content: afterLines[afterIndex],
    });
    afterIndex += 1;
    afterLineNumber += 1;
  }

  return rows;
}

export function summarizeSourceDiff(rows: SourceDiffRow[]): SourceDiffSummary {
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

export function buildSourceDiff(before: string, after: string): {
  rows: SourceDiffRow[];
  summary: SourceDiffSummary;
} {
  const rows = buildSourceDiffRows(before, after);
  return {
    rows,
    summary: summarizeSourceDiff(rows),
  };
}
