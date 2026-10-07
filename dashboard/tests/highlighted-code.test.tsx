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

  it("renders inline diff rows inside the code block when a comparison source is provided", () => {
    const { container } = render(
      <HighlightedCode
        code={"alpha\nbeta updated\ngamma\n"}
        diffBefore={"alpha\nbeta\ngamma\n"}
        language="python"
        wrap
      />,
    );

    const rows = Array.from(container.querySelectorAll(".code-block-diff-row"));

    expect(rows.map((row) => row.textContent)).toEqual([
      " 1alpha",
      "-2beta",
      "+2beta updated",
      " 3gamma",
    ]);
    expect(container.querySelectorAll(".code-block-line-remove")).toHaveLength(1);
    expect(container.querySelectorAll(".code-block-line-add")).toHaveLength(1);
  });
});
