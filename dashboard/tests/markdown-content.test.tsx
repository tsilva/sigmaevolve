// @vitest-environment jsdom

import { render, screen } from "@testing-library/react";

import { MarkdownContent } from "@/components/markdown-content";

describe("MarkdownContent", () => {
  it("renders headings, emphasis, lists, links, and code blocks from markdown", () => {
    render(
      <MarkdownContent
        content={[
          "### Plan",
          "",
          "Use **safer** batching.",
          "",
          "- inspect metrics",
          "- update optimizer",
          "",
          "See [details](https://example.com/run).",
          "",
          "```python",
          "print('hello')",
          "```",
        ].join("\n")}
      />,
    );

    expect(screen.getByRole("heading", { name: "Plan", level: 3 })).toBeTruthy();
    expect(screen.getByText("safer")).toBeTruthy();
    expect(screen.getByText("inspect metrics")).toBeTruthy();
    const link = screen.getByRole("link", { name: "details" });
    expect(link.getAttribute("href")).toBe("https://example.com/run");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(screen.getByText("print('hello')")).toBeTruthy();
  });

  it("keeps internal links in-app", () => {
    render(<MarkdownContent content="See [trial](/tracks/track_1/trials/trial_1)." />);

    const link = screen.getByRole("link", { name: "trial" });
    expect(link.getAttribute("href")).toBe("/tracks/track_1/trials/trial_1");
    expect(link.getAttribute("target")).toBeNull();
  });
});
