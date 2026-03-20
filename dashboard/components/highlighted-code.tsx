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
              return (
                <span key={index} {...lineProps}>
                  {line.map((token, tokenIndex) => {
                    const tokenProps = getTokenProps({ token });
                    return <span key={tokenIndex} {...tokenProps} />;
                  })}
                  {"\n"}
                </span>
              );
            })}
          </code>
        </pre>
      )}
    </Highlight>
  );
}
