"use client";

import { Highlight, themes, type Language } from "prism-react-renderer";

import { buildSourceDiff } from "@/lib/source-diff";

type HighlightedCodeProps = {
  code: string;
  diffBefore?: string | null;
  language: Language;
  wrap?: boolean;
};

export function HighlightedCode({
  code,
  diffBefore,
  language,
  wrap = false,
}: HighlightedCodeProps) {
  const diff = diffBefore ? buildSourceDiff(diffBefore, code) : null;

  return (
    <Highlight code={code} language={language} theme={themes.vsDark}>
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className={[
            "code-block",
            wrap ? "code-block-wrap" : "",
            diff ? "code-block-diff" : "",
            className,
          ]
            .filter(Boolean)
            .join(" ")}
          style={style}
        >
          <code>
            {diff
              ? diff.rows.map((row, index) => {
                  const line = row.afterLineIndex === null ? null : (tokens[row.afterLineIndex] ?? []);
                  const lineProps = line ? getLineProps({ line }) : null;
                  const rowClassName = [
                    lineProps?.className,
                    "code-block-line",
                    "code-block-diff-row",
                    row.kind === "add"
                      ? "code-block-line-add"
                      : row.kind === "remove"
                        ? "code-block-line-remove"
                        : "code-block-line-context",
                  ]
                    .filter(Boolean)
                    .join(" ");
                  const isEmptyLine =
                    line === null
                      ? row.content.length === 0
                      : line.every((token) => token.content.trim().length === 0);
                  const marker = row.kind === "add" ? "+" : row.kind === "remove" ? "-" : " ";
                  const lineNumber = row.afterLineNumber ?? row.beforeLineNumber ?? "";

                  return (
                    <span
                      key={`${row.kind}-${row.beforeLineIndex ?? "x"}-${row.afterLineIndex ?? "x"}-${index}`}
                      {...(lineProps ?? {})}
                      className={rowClassName}
                    >
                      <span className="code-block-diff-gutter" aria-hidden="true">
                        {marker}
                      </span>
                      <span className="code-block-diff-line-number" aria-hidden="true">
                        {lineNumber}
                      </span>
                      <span className="code-block-diff-content">
                        {isEmptyLine ? (
                          <span aria-hidden="true">{"\u00A0"}</span>
                        ) : line ? (
                          line.map((token, tokenIndex) => {
                            const tokenProps = getTokenProps({ token });
                            return <span key={tokenIndex} {...tokenProps} />;
                          })
                        ) : (
                          row.content
                        )}
                      </span>
                    </span>
                  );
                })
              : tokens.map((line, index) => {
                  const lineProps = getLineProps({ line });
                  const className = [lineProps.className, "code-block-line"].filter(Boolean).join(" ");
                  const isEmptyLine = line.every((token) => token.content.trim().length === 0);
                  return (
                    <span key={index} {...lineProps} className={className}>
                      {isEmptyLine ? (
                        <span aria-hidden="true">{"\u00A0"}</span>
                      ) : (
                        line.map((token, tokenIndex) => {
                          const tokenProps = getTokenProps({ token });
                          return <span key={tokenIndex} {...tokenProps} />;
                        })
                      )}
                    </span>
                  );
                })}
          </code>
        </pre>
      )}
    </Highlight>
  );
}
