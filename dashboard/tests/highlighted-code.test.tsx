// @vitest-environment jsdom

import { render } from "@testing-library/react";

import { HighlightedCode } from "@/components/highlighted-code";

describe("HighlightedCode", () => {
  it("renders each logical line as a block and preserves blank lines without injected newline nodes", () => {
    const { container } = render(<HighlightedCode code={"first\n\nsecond"} language="markdown" wrap />);

    const lines = Array.from(container.querySelectorAll(".code-block-line"));

    expect(lines).toHaveLength(3);
    expect(lines.map((line) => line.textContent)).toEqual(["first", "\u00A0", "second"]);
  });
});
