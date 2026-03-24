"use client";

import { Highlight, themes, type Language } from "prism-react-renderer";

type HighlightedCodeProps = {
  code: string;
  language: Language;
  wrap?: boolean;
};

export function HighlightedCode({
  code,
  language,
  wrap = false,
}: HighlightedCodeProps) {
  return (
    <Highlight code={code} language={language} theme={themes.vsDark}>
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className={`code-block ${wrap ? "code-block-wrap" : ""} ${className}`.trim()}
          style={style}
        >
          <code>
            {tokens.map((line, index) => {
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
